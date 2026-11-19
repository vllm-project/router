//! Worker discovery: query /get_server_info and the optional KV event source
//! control route, spawn one subscriber per (worker, rank). One indexer per
//! cache-identity core.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use parking_lot::Mutex;
use serde::Deserialize;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::config::StaticKvEventSources;
use crate::kv_index::storage::KvBlockIndexer;
use crate::kv_index::subscriber::SubscriberHandle;
use crate::kv_index::CacheCapabilities;
use crate::kv_index::EventSubscriberHandle;
use crate::kv_index::{IngestionSignal, SourceId};
use crate::protocols::worker_spec::ServerInfo;

mod lifecycle;
mod probe;
mod reconcile;
mod registration;

pub(crate) use lifecycle::CONTROL_TIMEOUT;
use lifecycle::{LayoutWorkerProvider, WorkerLifecycle};
use reconcile::{static_hybrid_sources_match, StaticHybridMatch};

use probe::{
    fetch_kv_event_sources, fetch_native_full_layout, grpc_kv_event_source_entries, grpc_model_id,
    layout_model, server_model_id,
};
pub(crate) use probe::{query_pd_server_info, query_worker_metadata};

/// The worker-stable core of a cache identity. One indexer per key.
///
/// Per-block namespaces and owner incarnations are tracked inside the index.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub model: Arc<str>,
    pub block_size: u32,
}

/// One per-DP-rank entry from the KV event source control probe (gRPC
/// `GetKvEventSources` or HTTP `/kv_event_sources`). Mirrors KVEventsConfig.
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub struct KvEventSourceEntry {
    #[serde(default)]
    pub endpoint: String, // PUB, may be a bind addr (tcp://*:port)
    #[serde(default)]
    pub replay_endpoint: Option<String>, // ROUTER
    #[serde(default)]
    pub topic: String,
    #[serde(default)]
    pub hwm: Option<i32>,
    #[serde(default)]
    pub buffer_steps: Option<u64>,
    #[serde(default)]
    pub publisher: Option<String>, // "zmq" | "null"
    #[serde(default)]
    pub encoding: Option<String>, // "msgpack"
    #[serde(default)]
    pub schema_version: Option<u32>,
    #[serde(default)]
    pub enable_kv_cache_events: Option<bool>,
}

/// /kv_event_sources body: a JSON object keyed by DP rank (as string).
pub type KvEventSourcesResponse = HashMap<String, KvEventSourceEntry>;

/// What on_worker_added learned and spawned.
#[derive(Debug, Clone, Default)]
pub struct WorkerKvInfo {
    pub instance_id: String,
    pub block_size: Option<u32>,
    pub ranks: Vec<u32>,
    /// Cache identity shared by every worker feeding the same placement index.
    /// Absent when discovery cannot create an index for this worker.
    pub cache_key: Option<CacheKey>,
}

pub(crate) struct WorkerDiscovery {
    pub(crate) server_info: Result<ServerInfo, String>,
    pub(crate) sources: Result<KvEventSourcesResponse, SourcesUnavailable>,
    /// gRPC has a separate GetModelInfo RPC; preserve its failure instead of
    /// silently turning the worker URL into a cache identity.
    model_id: Result<Option<String>, String>,
    pub(crate) hybrid: Option<CacheCapabilities>,
}

/// What the optional KV event source control probe found. A 404/405 means the
/// engine cannot report sources, not that it is broken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourcesProbe {
    /// The worker does not implement the extension; re-probing cannot help.
    Unsupported,
    /// Transport or server failure; the next discovery pass retries.
    Failed,
}

impl SourcesProbe {
    /// Whether another probe could plausibly succeed.
    pub(crate) fn retryable(self) -> bool {
        matches!(self, Self::Failed)
    }
}

/// A failed control probe plus whether retrying could ever help.
#[derive(Debug, Clone)]
pub(crate) struct SourcesUnavailable {
    pub(crate) probe: SourcesProbe,
    pub(crate) detail: String,
}

/// Per-worker subscriber state retained for removal-time cleanup.
struct WorkerSubs {
    active: Arc<AtomicBool>,
    key: CacheKey,
    source_id: SourceId,
    worker_origin: Option<String>,
    replay_endpoints: Vec<String>,
    live_sources: Vec<LiveSource>,
    ranks: Vec<u32>,
    indexer: Arc<KvBlockIndexer>,
    handles: Vec<SubscriberHandle>,
    hybrid: Option<Arc<LayoutWorkerProvider>>,
    hybrid_handles: Vec<Arc<EventSubscriberHandle>>,
}

struct LiveSource {
    publisher: String,
    topic: String,
    rank: u32,
    lease: crate::kv_index::source::binding::SourceLease,
}

/// Orchestrates discovery + subscriber lifecycle for the router's workers.
/// One indexer per `CacheKey`, shared by every `(worker, rank)` of that
/// identity.
pub struct KvIndexSupervisor {
    client: reqwest::Client,
    api_key: Option<String>,
    full_attention_http: bool,
    http_pd: bool,
    mooncake_prefill: bool,
    /// Explicit event sources override discovery for configured regular workers.
    static_sources: HashMap<String, StaticKvEventSources>,
    /// Static KV event sources per prefill URL; these workers skip HTTP
    /// discovery entirely.
    static_pd_sources: HashMap<String, crate::config::types::PdKvEventSources>,
    /// Shared across every subscriber — one I/O thread pool, not one per task.
    ctx: Arc<zmq::Context>,
    /// One indexer per cache-identity core.
    indexers: DashMap<CacheKey, Arc<KvBlockIndexer>>,
    /// Live-worker ref-count per identity; drop the indexer on last departure.
    identity_refs: Mutex<HashMap<CacheKey, usize>>,
    signal_tx: mpsc::Sender<IngestionSignal>,
    /// Per-worker subscriber state.
    workers: Mutex<HashMap<String, WorkerSubs>>,
    lifecycles: Mutex<HashMap<String, Arc<WorkerLifecycle>>>,
    refresh_lock: tokio::sync::Mutex<()>,
    /// First-seen block_size, for cross-worker mismatch warning.
    first_block_size: Mutex<Option<u32>>,
}

impl std::fmt::Debug for KvIndexSupervisor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("KvIndexSupervisor")
            .field("indexers", &self.indexers.len())
            .field("workers", &self.workers.lock().len())
            .finish()
    }
}

impl KvIndexSupervisor {
    /// Construct the supervisor. The returned receiver streams ingestion signals
    /// to the trust arbiter; drop it to ignore signals.
    pub fn new(client: reqwest::Client) -> (Self, mpsc::Receiver<IngestionSignal>) {
        let (signal_tx, signal_rx) = mpsc::channel::<IngestionSignal>(256);
        let supervisor = KvIndexSupervisor {
            client,
            api_key: None,
            full_attention_http: false,
            http_pd: false,
            mooncake_prefill: false,
            static_sources: HashMap::new(),
            static_pd_sources: HashMap::new(),
            ctx: Arc::new(zmq::Context::new()),
            indexers: DashMap::new(),
            identity_refs: Mutex::new(HashMap::new()),
            signal_tx,
            workers: Mutex::new(HashMap::new()),
            lifecycles: Mutex::new(HashMap::new()),
            refresh_lock: tokio::sync::Mutex::new(()),
            first_block_size: Mutex::new(None),
        };
        (supervisor, signal_rx)
    }

    pub(crate) fn new_http_full_attention(
        client: reqwest::Client,
    ) -> (Self, mpsc::Receiver<IngestionSignal>) {
        let (mut supervisor, signals) = Self::new(client);
        supervisor.full_attention_http = true;
        (supervisor, signals)
    }

    /// HTTP P/D accepts the native Python control endpoints and full-attention KV
    /// events.
    pub(crate) fn new_pd(
        client: reqwest::Client,
        static_pd_sources: HashMap<String, crate::config::types::PdKvEventSources>,
    ) -> (Self, mpsc::Receiver<IngestionSignal>) {
        let (mut supervisor, signals) = Self::new(client);
        supervisor.http_pd = true;
        supervisor.static_sources = static_pd_sources
            .iter()
            .filter(|(_, config)| config.hybrid.is_some())
            .map(|(worker, config)| (worker.clone(), pd_static_source(worker, config)))
            .collect();
        supervisor.static_pd_sources = static_pd_sources;
        (supervisor, signals)
    }

    /// Resolve source-owned feedback to registered URLs, never remote metadata addresses.
    pub fn worker_urls_for_source(&self, source: &SourceId) -> Vec<String> {
        self.workers
            .lock()
            .iter()
            .filter(|(_, w)| &w.source_id == source)
            .map(|(url, _)| url.clone())
            .collect()
    }

    pub(crate) fn set_api_key(&mut self, api_key: Option<String>) {
        self.api_key = api_key;
    }

    pub(crate) fn set_mooncake_prefill(&mut self, enabled: bool) {
        self.mooncake_prefill = enabled;
    }

    pub fn with_static_sources(
        mut self,
        static_sources: HashMap<String, StaticKvEventSources>,
    ) -> Self {
        self.static_sources = static_sources;
        self
    }

    /// Resolve a worker's declared sources, folding per-rank declarations
    /// (`base@0`, `base@1`) into the base entry. Regular startup discovers the
    /// base URL and fans it out to its DP ranks afterwards, so a declaration
    /// keyed by rank has to be merged here or it is never applied.
    fn resolved_static_source(&self, worker_url: &str) -> Option<StaticKvEventSources> {
        let exact = self.static_sources.get(worker_url).cloned();
        if worker_url.contains('@') {
            return exact;
        }
        let prefix = format!("{worker_url}@");
        let mut per_rank: Vec<(u32, &StaticKvEventSources)> = Vec::new();
        for (key, source) in &self.static_sources {
            if let Some(rank) = key
                .strip_prefix(prefix.as_str())
                .and_then(|suffix| suffix.parse::<u32>().ok())
            {
                per_rank.push((rank, source));
            }
        }
        if per_rank.is_empty() {
            return exact;
        }
        per_rank.sort_by_key(|(rank, _)| *rank);
        let mut merged = exact
            .or_else(|| per_rank.first().map(|(_, source)| (*source).clone()))
            .expect("per-rank declarations are non-empty");
        // Every rank publishes its own endpoints: collect them instead of
        // keeping only the first non-empty list, or the second rank's publisher
        // and replay channel are silently dropped.
        let mut endpoints = merged.endpoints.clone();
        let mut replay_endpoints = merged.replay_endpoints.clone().unwrap_or_default();
        for (rank, source) in per_rank {
            for endpoint in &source.endpoints {
                if !endpoints.contains(endpoint) {
                    endpoints.push(endpoint.clone());
                }
            }
            if let Some(list) = &source.replay_endpoints {
                for endpoint in list {
                    if !replay_endpoints.contains(endpoint) {
                        replay_endpoints.push(endpoint.clone());
                    }
                }
            }
            if merged.topic.is_empty() {
                merged.topic = source.topic.clone();
            }
            if merged.model.is_empty() {
                merged.model = source.model.clone();
            }
            match (&mut merged.hybrid, &source.hybrid) {
                (Some(base), Some(overlay)) => merge_hybrid_layout(base, overlay),
                (None, Some(overlay)) => merged.hybrid = Some(overlay.clone()),
                _ => {}
            }
            tracing::debug!(
                worker_url,
                rank,
                "merged a rank-pinned hybrid declaration into its base worker"
            );
        }
        merged.endpoints = endpoints;
        merged.replay_endpoints = (!replay_endpoints.is_empty()).then_some(replay_endpoints);
        Some(merged)
    }

    /// Query both endpoints for `worker_url`, spawn one subscriber per rank into
    /// this worker's identity's indexer. A missing source route is a soft skip.
    pub async fn on_worker_added(&self, worker_url: &str) -> Result<WorkerKvInfo, String> {
        // Discovery awaits without the owner lock. A failed/stale re-add must
        // leave the prior registration and its observed placement intact.
        let (observed, lifecycle) = {
            let workers = self.workers.lock();
            let owner = workers.get(worker_url).map(|worker| worker.active.clone());
            let lifecycle = self.lifecycle_for_worker(worker_url);
            (owner, lifecycle)
        };
        let discovery = self.discover(worker_url).await;
        self.install_observed_worker(
            worker_url,
            discovery,
            observed,
            Some(lifecycle),
            HashSet::new(),
        )
    }

    /// Prepare metadata once; P/D refresh fingerprints and installs this same snapshot.
    pub(crate) async fn discover(&self, worker_url: &str) -> WorkerDiscovery {
        if self.http_pd {
            return match self.static_pd_sources.get(worker_url) {
                Some(config) => {
                    let source = pd_static_source(worker_url, config);
                    if source.hybrid.is_none() {
                        static_pd_discovery(worker_url, config)
                    } else {
                        match static_hybrid_sources_match(
                            &self.client,
                            worker_url,
                            self.api_key.as_deref(),
                            &source,
                            None,
                        )
                        .await
                        {
                            StaticHybridMatch::Differs => failed_hybrid_discovery(
                                "configured hybrid event sources do not match worker".into(),
                            ),
                            StaticHybridMatch::Unavailable(SourcesProbe::Failed) => failed_hybrid_discovery(
                                "cannot confirm configured hybrid event sources while the control probe is failing".into(),
                            ),
                            StaticHybridMatch::Unavailable(SourcesProbe::Unsupported) => {
                                log_static_layout_trusted(worker_url, SourcesProbe::Unsupported);
                                static_pd_discovery(worker_url, config)
                            }
                            StaticHybridMatch::Matches => static_pd_discovery(worker_url, config),
                        }
                    }
                }
                None => {
                    discover_worker(&self.client, worker_url, true, self.api_key.as_deref()).await
                }
            };
        }
        let grpc = crate::backend::is_grpc_url(worker_url);
        let static_source = self.resolved_static_source(worker_url);
        let full_attention_http = !grpc && self.full_attention_http;
        match &static_source {
            // A layout entry without endpoints: the declaration carries the
            // identity and groups, the engine reports the publishers.
            Some(config) if layout_only(config) => {
                let discovery =
                    discover_worker(&self.client, worker_url, false, self.api_key.as_deref()).await;
                match discovery.sources {
                    Ok(sources) if !sources.is_empty() => {
                        match layout_topic_mismatch(config, &sources) {
                            Some(reason) => failed_hybrid_discovery(reason),
                            None => static_worker_discovery_with_sources(
                                worker_url,
                                config,
                                Ok(sources),
                            ),
                        }
                    }
                    Ok(_) => failed_hybrid_discovery(format!(
                        "engine reports no KV event sources for the declared hybrid layout \
                         ({worker_url}); declare --kv-event-source"
                    )),
                    Err(unavailable) => {
                        // No endpoints to fall back to, so this worker gets no
                        // KV subscribers; the reason is already logged.
                        log_static_layout_trusted(worker_url, unavailable.probe);
                        static_worker_discovery_with_sources(worker_url, config, Err(unavailable))
                    }
                }
            }
            Some(config) => {
                info!(
                    worker_url,
                    model = %config.model,
                    block_size = config.block_size,
                    ranks = config.endpoints.len(),
                    "using configured static KV event sources"
                );
                if config.hybrid.is_some() {
                    match static_hybrid_sources_match(
                        &self.client,
                        worker_url,
                        self.api_key.as_deref(),
                        config,
                        None,
                    )
                    .await
                    {
                        StaticHybridMatch::Differs => {
                            return failed_hybrid_discovery(
                                "configured hybrid event sources do not match worker".into(),
                            );
                        }
                        StaticHybridMatch::Unavailable(SourcesProbe::Failed) => {
                            return failed_hybrid_discovery(
                                "cannot confirm configured hybrid event sources while the control probe is failing".into(),
                            );
                        }
                        StaticHybridMatch::Unavailable(SourcesProbe::Unsupported) => {
                            log_static_layout_trusted(worker_url, SourcesProbe::Unsupported)
                        }
                        StaticHybridMatch::Matches => {}
                    }
                }
                static_worker_discovery(worker_url, config)
            }
            None if full_attention_http => {
                discover_full_attention_http_worker(
                    &self.client,
                    worker_url,
                    self.api_key.as_deref(),
                )
                .await
            }
            None => discover_worker(&self.client, worker_url, false, self.api_key.as_deref()).await,
        }
    }

    /// Expose the discovered worker's live index through the policy's read-only port.
    pub fn match_provider_for_worker(
        &self,
        worker_url: &str,
    ) -> Option<Arc<dyn crate::policies::TieredMatchProvider>> {
        self.workers
            .lock()
            .get(worker_url)
            .map(|worker| match &worker.hybrid {
                Some(provider) => provider.clone() as Arc<dyn crate::policies::TieredMatchProvider>,
                None => worker.indexer.clone() as Arc<dyn crate::policies::TieredMatchProvider>,
            })
    }

    fn check_block_size(&self, worker_url: &str, block_size: Option<u32>) {
        let Some(bs) = block_size else { return };
        let mut first = self.first_block_size.lock();
        match *first {
            None => *first = Some(bs),
            Some(prev) if prev != bs => {
                warn!(
                    "kv_index {}: block_size {} differs from first-seen {} — DEGRADED risk",
                    worker_url, bs, prev
                );
            }
            _ => {}
        }
    }
}

/// Overlay one declaration's ranks and adapters onto another. A rank-pinned
/// declaration replaces the base entry of the same rank; other ranks are
/// untouched.
fn merge_hybrid_layout(
    base: &mut crate::kv_index::CacheCapabilities,
    overlay: &crate::kv_index::CacheCapabilities,
) {
    for rank in &overlay.ranks {
        match base
            .ranks
            .iter_mut()
            .find(|existing| existing.rank == rank.rank)
        {
            Some(existing) => *existing = rank.clone(),
            None => base.ranks.push(rank.clone()),
        }
    }
    base.ranks.sort_by_key(|rank| rank.rank);
    for adapter in &overlay.adapters {
        if !base
            .adapters
            .iter()
            .any(|existing| existing.name == adapter.name)
        {
            base.adapters.push(adapter.clone());
        }
    }
}

fn failed_hybrid_discovery(error: String) -> WorkerDiscovery {
    WorkerDiscovery {
        server_info: Err(error.clone()),
        sources: Err(SourcesUnavailable {
            probe: SourcesProbe::Failed,
            detail: error.clone(),
        }),
        model_id: Err(error),
        hybrid: None,
    }
}

/// A declared static layout outranks a missing control endpoint: it is the
/// operator's contract, not a fallback. Say which reason we trusted.
fn log_static_layout_trusted(worker_url: &str, probe: SourcesProbe) {
    match probe {
        // Stock vLLM serves no HTTP route, so this is the expected answer.
        SourcesProbe::Unsupported => debug!(
            worker_url,
            "engine does not expose /kv_event_sources (added in vLLM 0.31.0); \
             trusting the configured hybrid layout"
        ),
        SourcesProbe::Failed => warn!(
            worker_url,
            "engine did not answer the KV event source probe; \
             trusting the configured hybrid layout"
        ),
    }
}

impl Drop for KvIndexSupervisor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

async fn discover_worker(
    client: &reqwest::Client,
    worker_url: &str,
    http_pd: bool,
    api_key: Option<&str>,
) -> WorkerDiscovery {
    if crate::backend::is_grpc_url(worker_url) {
        let timeout = Duration::from_secs(10);
        let server = crate::backend::control::get_grpc_server_info(worker_url, timeout).await;
        let model =
            grpc_model_id(crate::backend::control::get_grpc_model_info(worker_url, timeout).await);
        let sources = crate::backend::control::get_grpc_kv_event_sources(worker_url, timeout)
            .await
            .map_err(SourcesUnavailable::from);
        let server_info = server.map(|server| {
            let model_id = model.as_ref().ok().and_then(|model| model.clone());
            ServerInfo {
                model_id,
                instance_id: (!server.instance_id.is_empty()).then_some(server.instance_id),
                kv_block_size: (server.kv_block_size != 0).then_some(server.kv_block_size),
                effective_attention_block_size: server
                    .effective_attention_block_size
                    .and_then(|size| u32::try_from(size).ok()),
                ..Default::default()
            }
        });
        let sources =
            sources.map(|response| grpc_kv_event_source_entries(worker_url, response.sources));
        return WorkerDiscovery {
            server_info,
            sources,
            model_id: model,
            hybrid: None,
        };
    }
    if !http_pd {
        return WorkerDiscovery {
            server_info: Err(
                "HTTP worker model and block-size metadata must be configured statically"
                    .to_string(),
            ),
            sources: fetch_kv_event_sources(client, worker_url, api_key).await,
            model_id: Ok(Some(worker_url.to_string())),
            hybrid: None,
        };
    }
    discover_full_attention_http_worker(client, worker_url, api_key).await
}

/// The data-parallel rank a worker URL pins, as in `http://host:8000@3`.
fn dp_rank_suffix(worker_url: &str) -> Option<u32> {
    let (base, rank) = worker_url.rsplit_once('@')?;
    if base.is_empty() {
        return None;
    }
    rank.parse().ok()
}

fn static_worker_discovery(worker_url: &str, config: &StaticKvEventSources) -> WorkerDiscovery {
    let model = config.model.trim().to_owned();
    let rank_suffix = dp_rank_suffix(worker_url);
    let server_info = ServerInfo {
        // Model/block size are declarations; the registry URL is not a boot ID.
        model_id: Some(model.clone()),
        instance_id: Some(worker_url.to_owned()),
        kv_block_size: Some(config.block_size),
        ..Default::default()
    };
    let sources = config
        .endpoints
        .iter()
        .enumerate()
        .map(|(index, endpoint)| {
            // With a rank-pinned URL the operator declares exactly that rank's
            // publisher, so the single configured endpoint belongs to it.
            let rank = rank_suffix.unwrap_or(index as u32);
            let endpoint = if rank_suffix.is_some() {
                config.endpoints.first().unwrap_or(endpoint)
            } else {
                endpoint
            };
            (
                rank.to_string(),
                KvEventSourceEntry {
                    endpoint: endpoint.clone(),
                    replay_endpoint: config
                        .replay_endpoints
                        .as_ref()
                        .and_then(|replay| {
                            if rank_suffix.is_some() {
                                replay.first()
                            } else {
                                replay.get(index)
                            }
                        })
                        .cloned(),
                    topic: config
                        .hybrid
                        .as_ref()
                        .and_then(|layout| layout.ranks.iter().find(|item| item.rank == rank))
                        .and_then(|item| item.publisher_topic.clone())
                        .unwrap_or_else(|| config.topic.clone()),
                    hwm: None,
                    // Protocol constants are adapter assumptions, not observations.
                    publisher: Some("zmq".to_owned()),
                    encoding: Some("msgpack".to_owned()),
                    schema_version: Some(1),
                    enable_kv_cache_events: Some(true),
                    ..Default::default()
                },
            )
        })
        .collect();
    WorkerDiscovery {
        server_info: Ok(server_info),
        sources: Ok(sources),
        model_id: Ok(Some(model)),
        hybrid: config.hybrid.clone(),
    }
}

/// A declared hybrid layout whose per-rank publishers the engine reports.
fn layout_only(config: &StaticKvEventSources) -> bool {
    config.hybrid.is_some() && config.endpoints.is_empty()
}

/// Same identity as [`static_worker_discovery`], with endpoints from the engine.
fn static_worker_discovery_with_sources(
    worker_url: &str,
    config: &StaticKvEventSources,
    sources: Result<KvEventSourcesResponse, SourcesUnavailable>,
) -> WorkerDiscovery {
    let mut discovery = static_worker_discovery(worker_url, config);
    discovery.sources = sources;
    discovery
}

/// The layout names the publisher topic each rank must use. A discovered
/// publisher that advertises another topic belongs to a different boot, so
/// subscribing to it would index the wrong stream.
fn layout_topic_mismatch(
    config: &StaticKvEventSources,
    sources: &KvEventSourcesResponse,
) -> Option<String> {
    let layout = config.hybrid.as_ref()?;
    for (rank, entry) in sources {
        let Ok(rank) = rank.parse::<u32>() else {
            continue;
        };
        let declared = layout
            .ranks
            .iter()
            .find(|item| item.rank == rank)
            .and_then(|item| item.publisher_topic.as_deref())
            .filter(|topic| !topic.is_empty());
        if let Some(declared) = declared {
            if entry.topic != declared {
                return Some(format!(
                    "discovered KV event topic for rank {rank} is '{}', layout declares \
                     '{declared}'",
                    entry.topic
                ));
            }
        }
    }
    None
}

async fn discover_full_attention_http_worker(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> WorkerDiscovery {
    let (server_info, sources, layout) = tokio::join!(
        query_worker_metadata(client, worker_url, api_key),
        fetch_kv_event_sources(client, worker_url, api_key),
        tokio::time::timeout(
            CONTROL_TIMEOUT,
            fetch_native_full_layout(client, worker_url, api_key),
        ),
    );
    let model_id = server_info
        .as_ref()
        .map(|info| Some(server_model_id(info).unwrap_or_else(|| worker_url.to_string())))
        .map_err(Clone::clone);
    let hybrid = match (&server_info, &sources, &model_id, layout) {
        (Ok(info), Ok(sources), Ok(Some(model)), Ok(Ok(Some(layout)))) => {
            let engine_model = layout_model(info, model, &layout);
            let mut active = sources.iter().filter(|(_, source)| events_enabled(source));
            let mut topics = HashSet::new();
            let matches = active.clone().count() == layout.ranks.len()
                && active.all(|(key, source)| {
                    key.parse::<u32>().ok().is_some_and(|rank| {
                        key == &rank.to_string()
                            && topics.insert(source.topic.as_str())
                            && layout.rank_capability(rank).is_some_and(|capability| {
                                capability.hash_block_size == discovered_block_size(info)
                            })
                            && layout
                                .verify_layout(
                                    rank,
                                    Some(engine_model),
                                    Some(&source.topic),
                                    source.replay_endpoint.as_deref(),
                                )
                                .is_ok()
                    })
                });
            if matches {
                info!(
                    worker_url,
                    "using discovered full-attention KV hash contract"
                );
            }
            matches.then_some(layout)
        }
        _ => None,
    };
    WorkerDiscovery {
        server_info,
        sources,
        model_id,
        hybrid,
    }
}

/// Build a discovery result purely from static configuration, without
/// querying the worker's HTTP control plane.
fn static_pd_discovery(
    worker_url: &str,
    config: &crate::config::types::PdKvEventSources,
) -> WorkerDiscovery {
    static_worker_discovery(worker_url, &pd_static_source(worker_url, config))
}

fn pd_static_source(
    worker_url: &str,
    config: &crate::config::types::PdKvEventSources,
) -> StaticKvEventSources {
    StaticKvEventSources {
        model: config
            .model
            .as_deref()
            .filter(|model| !model.trim().is_empty())
            .unwrap_or(worker_url)
            .to_owned(),
        block_size: config.block_size,
        endpoints: config.endpoints.clone(),
        replay_endpoints: config.replay_endpoints.clone(),
        topic: String::new(),
        hybrid: config.hybrid.clone(),
    }
}

/// Native vLLM HTTP may omit wire fields. Validate any advertised fields
/// against the ZMQ MessagePack v1 format used for live events and replay.
fn events_enabled(entry: &KvEventSourceEntry) -> bool {
    if entry.publisher.as_deref() != Some("zmq") {
        return false;
    }
    // Omitted `encoding`/`schema_version` mean msgpack schema 1; any other
    // value is rejected.
    if !matches!(entry.encoding.as_deref(), None | Some("msgpack")) {
        return false;
    }
    if !matches!(entry.schema_version, None | Some(1)) {
        return false;
    }
    if matches!(entry.enable_kv_cache_events, Some(false)) {
        return false;
    }
    !entry.endpoint.is_empty()
}

/// Names the metadata sources that were tried, so a warning matches the
/// transport the worker actually speaks.
fn metadata_source_label(worker_url: &str) -> &'static str {
    if crate::backend::is_grpc_url(worker_url) {
        "GetServerInfo"
    } else {
        "/get_server_info, /server_info or /metrics"
    }
}

/// Same for event sources: a gRPC worker only exposes the control RPC.
fn event_source_label(worker_url: &str) -> &'static str {
    if crate::backend::is_grpc_url(worker_url) {
        "GetKvEventSources"
    } else {
        "/kv_event_sources"
    }
}

/// Rewrite a ZMQ bind host before validating the TCP connect endpoint.
/// `tcp://*:5557` → `tcp://10.0.0.5:5557`.
fn resolve_endpoint(endpoint: &str, worker_host: &str) -> Result<String, String> {
    if let Some(rest) = endpoint.strip_prefix("tcp://") {
        if let Some((host, port)) = rest.rsplit_once(':') {
            if matches!(host, "*" | "0.0.0.0" | "[::]") {
                return crate::kv_index::source::endpoints::canonical_endpoint(&format!(
                    "tcp://{worker_host}:{port}"
                ));
            }
        }
        return crate::kv_index::source::endpoints::canonical_endpoint(endpoint);
    }
    // IPC/inproc and other transports retain their existing ZMQ contract;
    // the HTTP-only worker mapping helper must not gate gRPC discovery.
    Ok(endpoint.to_string())
}

/// Extract the host from a worker URL (`http://10.0.0.5:8000` → `10.0.0.5`).
fn worker_host(worker_url: &str) -> Option<String> {
    let with_scheme = if worker_url.contains("://") {
        worker_url.to_string()
    } else {
        format!("http://{}", worker_url)
    };
    url::Url::parse(&with_scheme)
        .ok()?
        .host_str()
        .map(|h| h.to_string())
}

fn discovered_block_size(server_info: &ServerInfo) -> Option<u32> {
    server_info
        .kv_block_size
        .filter(|size| *size > 0)
        .or_else(|| {
            server_info
                .effective_attention_block_size
                .filter(|size| *size > 0)
        })
}

mod metrics_meta;

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests;

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod rank_merge_tests {
    use super::*;

    #[test]
    fn rank_pinned_declarations_fold_into_their_base_worker() {
        let base = "http://worker:8000";
        let rank = |rank: u32, model: &str| {
            serde_json::json!({
                "rank": rank, "model": model, "weight_version": "default",
                "weight_identity": null, "publisher_topic": format!("topic-{rank}"),
                "instance_topic": true, "hash_algorithm": "sha256_cbor",
                "root_hash": "9bd96a485ad84efdafb72ee48a1d7a69bcead0f8f0433173941b276b9581eef0",
                "full_hashes": true, "hash_block_size": 16, "hash_lookahead_tokens": 0,
                "speculative_tokens": 0, "prefix_caching": true, "groups": [],
            })
        };
        let mut sources = serde_json::Map::new();
        sources.insert(
            base.to_string(),
            serde_json::json!({
                "endpoints": ["tcp://base:5557"], "topic": "topic-0", "block_size": 16, "model": "m0",
                "hybrid": {"schema_version": 1, "ranks": [rank(0, "m0")]},
            }),
        );
        sources.insert(
            format!("{base}@1"),
            serde_json::json!({
                "endpoints": ["tcp://rank1:5557"], "topic": "topic-1", "block_size": 16, "model": "m1",
                "hybrid": {"schema_version": 1, "ranks": [rank(1, "m1")]},
            }),
        );
        let (mut supervisor, _signals) = KvIndexSupervisor::new(reqwest::Client::new());
        supervisor.static_sources =
            serde_json::from_value(serde_json::Value::Object(sources)).expect("sources");
        let resolved = supervisor
            .resolved_static_source(base)
            .expect("base resolves with its rank declarations");
        let hybrid = resolved.hybrid.expect("hybrid layout");
        assert_eq!(hybrid.ranks.len(), 2);
        assert_eq!(hybrid.ranks[0].rank, 0);
        assert_eq!(hybrid.ranks[0].model, "m0");
        assert_eq!(hybrid.ranks[1].rank, 1);
        assert_eq!(hybrid.ranks[1].model, "m1");
        // Both ranks keep their publisher: keeping only the first non-empty
        // list silently drops the second rank's stream.
        assert_eq!(
            resolved.endpoints,
            vec![
                "tcp://base:5557".to_string(),
                "tcp://rank1:5557".to_string()
            ]
        );
    }
}
