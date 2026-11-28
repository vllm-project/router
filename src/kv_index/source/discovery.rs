//! Worker discovery: query /get_server_info and the optional KV event source
//! control route, then spawn one subscriber per (worker, rank).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use parking_lot::Mutex;
use serde::Deserialize;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::config::{KvConnector, StaticKvEventSources};
use crate::kv_index::storage::KvBlockIndexer;
use crate::kv_index::subscriber::EventSource;
use crate::kv_index::EventSubscriberHandle;
use crate::kv_index::{CacheCapabilities, GroupCapability, RankCapability};
use crate::kv_index::{IngestionSignal, SourceId};

mod lifecycle;
mod observer;
mod probe;
mod reconcile;
mod registration;

pub(crate) use lifecycle::CONTROL_TIMEOUT;
use lifecycle::{LayoutWorkerProvider, WorkerLifecycle};
use metrics_meta::EngineCacheConfigs;
use observer::WorkerObservation;
use reconcile::{static_hybrid_sources_match, StaticHybridMatch};

#[cfg(all(test, feature = "cpu", feature = "ready"))]
pub(crate) use probe::worker_serves_base_model;
use probe::{
    base_model_names, fetch_kv_event_sources, fetch_native_sources, grpc_model_id, layout_model,
    query_model_list, query_worker_metadata_and_models, served_model, server_instance_id,
    server_model_id,
};
pub use probe::{get_grpc_kv_event_sources, GrpcKvEventSourcesResponse};
pub(crate) use probe::{inferred_lora_name, served_model_labels};

/// The worker-stable core of a cache identity. One indexer per key.
///
/// Per-block namespaces and owner incarnations are tracked inside the index.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub model: Arc<str>,
    pub block_size: u32,
}

/// Worker metadata used when constructing a KV cache identity.
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct ServerInfo {
    pub model_id: Option<String>,

    pub model_path: Option<String>,

    pub priority: Option<u32>,

    pub cost: Option<f32>,

    pub worker_type: Option<String>,

    // Optional fields allow discovery on workers with older metadata schemas.
    #[serde(default)]
    pub instance_id: Option<String>,
    #[serde(default, alias = "block_size")]
    pub kv_block_size: Option<u32>,
    #[serde(default)]
    pub effective_attention_block_size: Option<u32>,
    /// Whether the worker's Mooncake connector trims compressed-model Prefill.
    #[serde(default)]
    pub mooncake_truncates_compressed: Option<bool>,
    /// Engine revision used to interpret versioned cache lookup rules.
    #[serde(skip)]
    pub(crate) vllm_version: Option<(u32, u32)>,
    /// `--served-model-name` aliases, so the router can tell a base-model alias
    /// from a LoRA adapter selected through the request's `model`.
    #[serde(default)]
    pub served_model_aliases: Vec<String>,
}

/// One DP rank's event publisher, reported by the worker control plane.
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
    pub publisher: Option<String>,
    #[serde(default)]
    pub encoding: Option<String>,
    #[serde(default)]
    pub schema_version: Option<u32>,
    #[serde(default)]
    pub enable_kv_cache_events: Option<bool>,
}

/// /kv_event_sources body: a JSON object keyed by DP rank (as string).
pub type KvEventSourcesResponse = HashMap<String, KvEventSourceEntry>;

fn event_sources_for_worker(
    worker_url: &str,
    source: &SourceId,
    sources: &KvEventSourcesResponse,
    include_replay: bool,
) -> Result<Vec<EventSource>, String> {
    let host = worker_host(worker_url).unwrap_or_else(|| worker_url.to_owned());
    let mut ranks = HashSet::new();
    sources
        .iter()
        .filter(|(_, entry)| events_enabled(entry))
        .map(|(rank, entry)| {
            let dp_rank = rank.parse().map_err(|_| "invalid KV source rank")?;
            if !ranks.insert(dp_rank) {
                return Err("duplicate normalized KV source rank".into());
            }
            let pub_endpoint = resolve_endpoint(&entry.endpoint, &host)
                .map_err(|error| format!("invalid KV publisher endpoint: {error}"))?;
            let replay_endpoint = entry
                .replay_endpoint
                .as_deref()
                .filter(|_| include_replay)
                .map(|address| resolve_endpoint(address, &host))
                .transpose()
                .map_err(|error| format!("invalid KV replay endpoint: {error}"))?;
            Ok(EventSource {
                source: source.clone(),
                dp_rank,
                pub_endpoint,
                replay_endpoint,
                topic: entry.topic.clone(),
                hwm: entry.hwm,
            })
        })
        .collect()
}

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
    /// Aliases already read during discovery, if a model-list probe was needed.
    base_models: Option<HashSet<String>>,
    pub(crate) hybrid: Option<CacheCapabilities>,
    pub(super) assumed_plain_full: bool,
    /// Per-engine reports from discovery or the optional admission scrape.
    /// None means no usable snapshot was collected, for example when probing
    /// was inapplicable, skipped for budget, or failed.
    engine_cache_configs: Option<EngineCacheConfigs>,
}

/// What the optional KV event source control probe found. A 404/405 means the
/// engine cannot report sources, not that it is broken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourcesProbe {
    /// The worker does not implement the extension; re-probing cannot help.
    Unsupported,
    /// Transport or server failure; the next discovery pass retries.
    Failed,
    /// The declared source disagrees with the worker or is malformed.
    StaticMismatch,
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
    provider: Arc<LayoutWorkerProvider>,
    handles: Vec<Arc<EventSubscriberHandle>>,
}

struct LiveSource {
    publisher: String,
    topic: String,
    rank: u32,
    lease: crate::kv_index::source::binding::SourceLease,
}

#[derive(Default)]
struct InstalledWorkers {
    indexers: DashMap<CacheKey, Arc<KvBlockIndexer>>,
    identity_refs: Mutex<HashMap<CacheKey, usize>>,
    workers: Mutex<HashMap<String, WorkerSubs>>,
}

/// Discovers workers and keeps one shared index per cache identity.
pub struct KvIndexSupervisor {
    client: reqwest::Client,
    api_key: Option<String>,
    full_attention_http: bool,
    http_pd: bool,
    pd_connector: Option<KvConnector>,
    strict_recovery: bool,
    /// Explicit event sources override discovery for configured regular workers.
    static_sources: HashMap<String, StaticKvEventSources>,
    /// Static KV event sources per prefill URL. A verified static layout can
    /// still use the optional HTTP metrics probe for hash admission.
    static_pd_sources: HashMap<String, crate::config::types::PdKvEventSources>,
    /// Shared across every subscriber — one I/O thread pool, not one per task.
    ctx: Arc<zmq::Context>,
    installed: Arc<InstalledWorkers>,
    signal_tx: mpsc::Sender<IngestionSignal>,
    observers: Option<Mutex<HashMap<String, WorkerObservation>>>,
    lifecycles: Mutex<HashMap<String, Arc<WorkerLifecycle>>>,
    refresh_lock: tokio::sync::Mutex<()>,
    /// First-seen block_size, for cross-worker mismatch warning.
    first_block_size: Mutex<Option<u32>>,
}

impl std::fmt::Debug for KvIndexSupervisor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("KvIndexSupervisor")
            .field("indexers", &self.installed.indexers.len())
            .field("workers", &self.installed.workers.lock().len())
            .finish()
    }
}

impl KvIndexSupervisor {
    /// Observe event continuity without decoding cache identities.
    pub(crate) fn new_observer(client: reqwest::Client) -> (Self, mpsc::Receiver<IngestionSignal>) {
        let (mut supervisor, signals) = Self::indexed(client);
        supervisor.observers = Some(Mutex::new(HashMap::new()));
        (supervisor, signals)
    }

    fn indexed(client: reqwest::Client) -> (Self, mpsc::Receiver<IngestionSignal>) {
        let (signal_tx, signal_rx) = mpsc::channel::<IngestionSignal>(256);
        let supervisor = KvIndexSupervisor {
            client,
            api_key: None,
            full_attention_http: false,
            http_pd: false,
            pd_connector: None,
            strict_recovery: false,
            static_sources: HashMap::new(),
            static_pd_sources: HashMap::new(),
            ctx: Arc::new(zmq::Context::new()),
            installed: Arc::new(InstalledWorkers::default()),
            signal_tx,
            observers: None,
            lifecycles: Mutex::new(HashMap::new()),
            refresh_lock: tokio::sync::Mutex::new(()),
            first_block_size: Mutex::new(None),
        };
        (supervisor, signal_rx)
    }

    pub(crate) fn new_http_full_attention(
        client: reqwest::Client,
    ) -> (Self, mpsc::Receiver<IngestionSignal>) {
        let (mut supervisor, signals) = Self::indexed(client);
        supervisor.full_attention_http = true;
        (supervisor, signals)
    }

    /// HTTP P/D accepts the native Python control endpoints and full-attention KV
    /// events.
    pub(crate) fn new_pd(
        client: reqwest::Client,
        static_pd_sources: HashMap<String, crate::config::types::PdKvEventSources>,
    ) -> (Self, mpsc::Receiver<IngestionSignal>) {
        let (mut supervisor, signals) = Self::indexed(client);
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
        if let Some(observers) = &self.observers {
            return observers
                .lock()
                .iter()
                .filter(|(_, worker)| &worker.source == source)
                .map(|(url, _)| url.clone())
                .collect();
        }
        self.installed
            .workers
            .lock()
            .iter()
            .filter(|(_, w)| &w.source_id == source)
            .map(|(url, _)| url.clone())
            .collect()
    }

    pub(crate) fn set_api_key(&mut self, api_key: Option<String>) {
        self.api_key = api_key;
    }

    pub(crate) fn set_pd_connector(&mut self, connector: KvConnector) {
        self.pd_connector = Some(connector);
    }

    pub(crate) fn set_strict_recovery(&mut self, enabled: bool) {
        self.strict_recovery = enabled;
    }

    pub fn with_static_sources(
        mut self,
        static_sources: HashMap<String, StaticKvEventSources>,
    ) -> Self {
        if !static_sources.is_empty() {
            self.observers = None;
        }
        self.static_sources = static_sources;
        self
    }

    /// Resolve a worker's declared sources, folding per-rank declarations
    /// (`base@0`, `base@1`) into the base entry. Regular startup discovers the
    /// base URL and fans it out to its DP ranks afterwards, so a declaration
    /// keyed by rank has to be merged here or it is never applied.
    fn resolved_static_source(&self, worker_url: &str) -> Option<WorkerDiscovery> {
        let exact = self.static_sources.get(worker_url);
        if split_worker_rank(worker_url).1.is_some() {
            return exact.map(|source| static_worker_discovery(worker_url, source));
        }
        let prefix = format!("{worker_url}@");
        let mut per_rank = Vec::new();
        for (key, source) in &self.static_sources {
            if let Some(rank) = key
                .strip_prefix(prefix.as_str())
                .and_then(|suffix| suffix.parse::<u32>().ok())
            {
                per_rank.push((rank, key, source));
            }
        }
        per_rank.sort_by_key(|(rank, _, _)| *rank);
        if let Some(duplicate) = per_rank.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            return Some(failed_hybrid_discovery(
                SourcesProbe::StaticMismatch,
                format!(
                    "duplicate static KV event sources for DP rank {} of {worker_url}",
                    duplicate[0].0
                ),
            ));
        }
        let mut per_rank = per_rank.into_iter();
        let mut merged = if let Some(source) = exact {
            static_worker_discovery(worker_url, source)
        } else {
            let (_, key, source) = per_rank.next()?;
            let mut first = static_worker_discovery(key, source);
            if let Ok(info) = &mut first.server_info {
                info.instance_id = Some(worker_url.to_owned());
            }
            first
        };
        for (rank, key, source) in per_rank {
            let overlay = static_worker_discovery(key, source);
            // Overlay publishers by DP rank, keeping each replay address
            // attached to its live publisher, even across sparse rank IDs.
            let (Ok(sources), Ok(new_sources)) = (&mut merged.sources, overlay.sources) else {
                return Some(failed_hybrid_discovery(
                    SourcesProbe::StaticMismatch,
                    format!("invalid static KV event sources for {worker_url}"),
                ));
            };
            sources.extend(new_sources);
            if merged
                .model_id
                .as_ref()
                .is_ok_and(|model| model.as_deref().is_none_or(str::is_empty))
            {
                merged.model_id = overlay.model_id;
                if let Ok(info) = &mut merged.server_info {
                    info.model_id = merged.model_id.as_ref().ok().and_then(Clone::clone);
                }
            }
            match (&mut merged.hybrid, overlay.hybrid) {
                (Some(base), Some(overlay)) => merge_hybrid_layout(base, &overlay),
                (None, Some(overlay)) => merged.hybrid = Some(overlay),
                _ => {}
            }
            tracing::debug!(
                worker_url,
                rank,
                "merged a rank-pinned hybrid declaration into its base worker"
            );
        }
        Some(merged)
    }

    /// Query both endpoints for `worker_url`, spawn one subscriber per rank into
    /// this worker's identity's indexer. A missing source route is a soft skip.
    pub async fn on_worker_added(&self, worker_url: &str) -> Result<WorkerKvInfo, String> {
        self.on_worker_added_with_models(worker_url)
            .await
            .map(|(info, _)| info)
    }

    pub(crate) async fn on_worker_added_with_models(
        &self,
        worker_url: &str,
    ) -> Result<(WorkerKvInfo, Option<HashSet<String>>), String> {
        if let Some(observers) = &self.observers {
            return self
                .observe_worker(worker_url, observers)
                .await
                .map(|info| (info, None));
        }
        // Check the observed owner again after discovery before changing its bindings.
        let (observed, lifecycle) = {
            let workers = self.installed.workers.lock();
            let owner = workers.get(worker_url).map(|worker| worker.active.clone());
            let lifecycle = self.lifecycle_for_worker(worker_url);
            (owner, lifecycle)
        };
        let discovery = self.discover(worker_url).await;
        let models = discovery.base_models.clone();
        let hashes = observed
            .as_ref()
            .and_then(|_| discovery.engine_cache_configs.clone());
        let installation = self.install_observed_worker(
            worker_url,
            discovery,
            observed,
            Some(lifecycle.clone()),
            None,
        );
        if !installation
            .as_ref()
            .is_ok_and(|info| info.cache_key.is_some())
        {
            if let Some(hashes) = hashes {
                self.fence_failed_registration(worker_url, &lifecycle, &hashes);
            }
        }
        Ok((installation?, models))
    }

    /// Discover worker metadata and available hashes for a new registration.
    pub(crate) async fn discover(&self, worker_url: &str) -> WorkerDiscovery {
        let mut discovery = self.discover_for_role(worker_url, None).await;
        self.probe_admission_hashes(worker_url, &mut discovery, None)
            .await;
        discovery
    }

    async fn discover_for_role(
        &self,
        worker_url: &str,
        prefill_connector: Option<KvConnector>,
    ) -> WorkerDiscovery {
        let static_source = if self.http_pd {
            self.static_pd_sources
                .get(worker_url)
                .filter(|config| config.hybrid.is_some() || !config.endpoints.is_empty())
                .map(|config| {
                    static_worker_discovery(worker_url, &pd_static_source(worker_url, config))
                })
        } else {
            self.resolved_static_source(worker_url)
        };
        let mut discovery = match static_source {
            Some(discovery) if layout_only(&discovery) => {
                self.discover_declared_layout(worker_url, discovery).await
            }
            Some(mut discovery) => {
                if discovery.sources.is_err() {
                    return discovery;
                }
                if let Err((probe, reason)) =
                    self.verify_static_source(worker_url, &discovery).await
                {
                    discovery.sources = Err(SourcesUnavailable {
                        probe,
                        detail: reason.into(),
                    });
                    return discovery;
                }
                info!(worker_url, "using configured static KV event sources");
                discovery
            }
            None => {
                let mut discovery = discover_worker(
                    &self.client,
                    worker_url,
                    self.http_pd || self.full_attention_http,
                    self.api_key.as_deref(),
                )
                .await;
                if discovery.assumed_plain_full
                    && self
                        .lifecycles
                        .lock()
                        .get(worker_url)
                        .is_some_and(|lifecycle| {
                            lifecycle
                                .state
                                .lock()
                                .descriptor
                                .as_ref()
                                .is_some_and(probe::WorkerDescriptor::has_nonplain_layout)
                        })
                {
                    // Do not assume a plain layout for a previously observed hybrid worker.
                    discovery.hybrid = None;
                    discovery.assumed_plain_full = false;
                }
                discovery
            }
        };
        let Some(layout) = discovery.hybrid.as_ref() else {
            return discovery;
        };
        let connector = prefill_connector;
        let bounded_nixl = connector == Some(KvConnector::Nixl)
            && layout.ranks.iter().any(|rank| {
                rank.groups
                    .iter()
                    .any(|group| group.prefix_replay_tokens > 0)
            });
        let has_tail_drop = layout.ranks.iter().any(|rank| {
            rank.tail_drop_groups
                .as_ref()
                .is_some_and(|groups| !groups.is_empty())
        });
        let (base, _) = split_worker_rank(worker_url);
        if bounded_nixl || has_tail_drop {
            if let Some(info) = discovery
                .server_info
                .as_mut()
                .ok()
                .filter(|info| info.vllm_version.is_none())
            {
                info.vllm_version = tokio::time::timeout(
                    lifecycle::OPTIONAL_METADATA_BUDGET,
                    probe::query_vllm_version(&self.client, base, self.api_key.as_deref()),
                )
                .await
                .ok()
                .flatten();
            }
        }
        if !matches!(
            connector,
            Some(KvConnector::Nixl | KvConnector::Mooncake | KvConnector::MooncakeV2)
        ) || !layout.ranks.iter().any(|rank| {
            rank.groups
                .iter()
                .any(|group| group.kind == "mla_attention" && group.tokens_per_state.is_some())
        }) {
            return discovery;
        }
        if let Some(info) = discovery
            .server_info
            .as_mut()
            .ok()
            .filter(|info| info.mooncake_truncates_compressed.is_none())
        {
            // A cold /server_info can spend time collecting worker diagnostics.
            let metadata_timeout =
                if matches!(connector, Some(KvConnector::Nixl | KvConnector::MooncakeV2)) {
                    CONTROL_TIMEOUT / 2
                } else {
                    lifecycle::OPTIONAL_METADATA_BUDGET
                };
            if let Ok(Ok(runtime)) = tokio::time::timeout(
                metadata_timeout,
                probe::query_server_info_json(&self.client, base, self.api_key.as_deref()),
            )
            .await
            {
                info.mooncake_truncates_compressed = runtime.mooncake_truncates_compressed;
            }
        }
        discovery
    }

    async fn discover_declared_layout(
        &self,
        worker_url: &str,
        mut discovery: WorkerDiscovery,
    ) -> WorkerDiscovery {
        let (base, _) = split_worker_rank(worker_url);
        let api_key = self.api_key.as_deref();
        let grpc = crate::backend::is_grpc_url(base);
        let (sources, (model, base_models)) =
            tokio::join!(fetch_kv_event_sources(&self.client, base, api_key), async {
                if let Some(model) = discovery
                    .model_id
                    .as_ref()
                    .ok()
                    .and_then(Option::as_ref)
                    .filter(|model| !model.trim().is_empty())
                {
                    (Some(model.clone()), None)
                } else if grpc {
                    let model = grpc_model_id(
                        crate::backend::control::get_grpc_model_info(base, CONTROL_TIMEOUT).await,
                    )
                    .ok()
                    .flatten();
                    let base_models = model.iter().cloned().collect();
                    (model, Some(base_models))
                } else {
                    let models = query_model_list(&self.client, base, api_key)
                        .await
                        .unwrap_or_default();
                    (
                        served_model(&models).map(|(model, _)| model),
                        Some(base_model_names(&models)),
                    )
                }
            },);
        discovery.base_models = base_models;
        let Ok(sources) = sources else {
            discovery.sources = sources;
            return discovery;
        };
        if sources.is_empty() {
            return failed_hybrid_discovery(
                SourcesProbe::Failed,
                "worker reports no KV event sources".into(),
            );
        }
        if let Some(reason) = layout_topic_mismatch(discovery.hybrid.as_ref(), &sources) {
            discovery.sources = Err(SourcesUnavailable {
                probe: SourcesProbe::StaticMismatch,
                detail: reason,
            });
            return discovery;
        }
        let Some(model) = model else {
            return failed_hybrid_discovery(
                SourcesProbe::Failed,
                "worker reports no served base model for KV layout".into(),
            );
        };
        if let Ok(info) = &mut discovery.server_info {
            info.model_id = Some(model.clone());
        }
        discovery.model_id = Ok(Some(model.clone()));
        if let Some(layout) = &mut discovery.hybrid {
            if layout
                .ranks
                .iter()
                .any(|rank| rank.hash_block_size.is_none())
            {
                let engines = match tokio::time::timeout(
                    CONTROL_TIMEOUT,
                    probe::query_engine_cache_configs(&self.client, base, api_key),
                )
                .await
                .unwrap_or_else(|_| Err("metrics probe timed out".into()))
                {
                    Ok(engines) => engines,
                    Err(reason) => return failed_hybrid_discovery(SourcesProbe::Failed, reason),
                };
                for rank in &mut layout.ranks {
                    if !sources
                        .get(&rank.rank.to_string())
                        .is_some_and(events_enabled)
                    {
                        continue;
                    }
                    let observed = engines
                        .get(&rank.rank)
                        .and_then(|engine| engine.hash_block_size);
                    match (rank.hash_block_size, observed) {
                        (Some(declared), Some(observed)) if declared != observed => {
                            return failed_hybrid_discovery(
                                SourcesProbe::StaticMismatch,
                                format!("DP rank {}: configured hash_block_size {declared} differs from worker metrics {observed}", rank.rank),
                            );
                        }
                        (None, Some(size)) => rank.hash_block_size = Some(size),
                        (None, None) => {
                            return failed_hybrid_discovery(
                                SourcesProbe::Failed,
                                format!("DP rank {}: worker metrics do not report hash_block_size. Set it in the KV layout.", rank.rank),
                            );
                        }
                        _ => {}
                    }
                }
                if let Ok(info) = &mut discovery.server_info {
                    info.kv_block_size = layout.ranks.iter().find_map(|rank| rank.hash_block_size);
                }
                discovery.engine_cache_configs = Some(engines);
            }
            for rank in &mut layout.ranks {
                if rank.model.trim().is_empty() {
                    rank.model.clone_from(&model);
                }
                if rank.publisher_topic.is_none() {
                    rank.publisher_topic = sources
                        .get(&rank.rank.to_string())
                        .filter(|entry| events_enabled(entry))
                        .map(|entry| entry.topic.clone());
                }
            }
        }
        discovery.sources = Ok(sources);
        discovery
    }

    async fn verify_static_source(
        &self,
        worker_url: &str,
        discovery: &WorkerDiscovery,
    ) -> Result<(), (SourcesProbe, &'static str)> {
        if discovery.hybrid.is_none() {
            return Ok(());
        }
        match static_hybrid_sources_match(
            &self.client,
            worker_url,
            self.api_key.as_deref(),
            discovery,
            None,
        )
        .await
        {
            StaticHybridMatch::Differs
            | StaticHybridMatch::Unavailable(SourcesProbe::StaticMismatch) => Err((
                SourcesProbe::StaticMismatch,
                "configured hybrid event sources do not match worker",
            )),
            StaticHybridMatch::Unavailable(SourcesProbe::Failed) => Err((
                SourcesProbe::Failed,
                "cannot confirm configured hybrid event sources while the control probe is failing",
            )),
            StaticHybridMatch::Unavailable(SourcesProbe::Unsupported) => {
                debug!(worker_url, "engine does not expose KV event sources; trusting the configured hybrid layout");
                Ok(())
            }
            StaticHybridMatch::Matches => Ok(()),
        }
    }

    /// Expose the discovered worker's live index through the policy's read-only port.
    pub fn match_provider_for_worker(
        &self,
        worker_url: &str,
    ) -> Option<Arc<dyn crate::policies::TieredMatchProvider>> {
        self.installed
            .workers
            .lock()
            .get(worker_url)
            .map(|worker| worker.provider.clone() as Arc<dyn crate::policies::TieredMatchProvider>)
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

fn failed_hybrid_discovery(probe: SourcesProbe, error: String) -> WorkerDiscovery {
    WorkerDiscovery {
        server_info: Err(error.clone()),
        sources: Err(SourcesUnavailable {
            probe,
            detail: error.clone(),
        }),
        model_id: Err(error),
        base_models: None,
        hybrid: None,
        assumed_plain_full: false,
        engine_cache_configs: None,
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
        let (server, model, (sources, layout)) = tokio::join!(
            crate::backend::control::get_grpc_server_info(worker_url, timeout),
            crate::backend::control::get_grpc_model_info(worker_url, timeout),
            fetch_native_sources(client, worker_url, api_key, timeout),
        );
        let model_id = grpc_model_id(model);
        let server_info = server.map(|server| ServerInfo {
            model_id: model_id.as_ref().ok().and_then(Clone::clone),
            instance_id: (!server.instance_id.is_empty()).then_some(server.instance_id),
            kv_block_size: (server.kv_block_size != 0).then_some(server.kv_block_size),
            effective_attention_block_size: server
                .effective_attention_block_size
                .and_then(|size| u32::try_from(size).ok()),
            vllm_version: probe::parse_vllm_version(&server.engine_version),
            ..Default::default()
        });
        let base_models = model_id
            .as_ref()
            .ok()
            .into_iter()
            .flat_map(|id| id.iter().cloned())
            .collect();
        let mut discovery =
            native_worker_discovery(worker_url, server_info, sources, model_id, layout);
        discovery.base_models = Some(base_models);
        return discovery;
    }
    if !http_pd {
        return WorkerDiscovery {
            server_info: Err(
                "HTTP worker model and block-size metadata must be configured statically"
                    .to_string(),
            ),
            sources: fetch_kv_event_sources(client, worker_url, api_key).await,
            model_id: Ok(Some(worker_url.to_string())),
            base_models: None,
            hybrid: None,
            assumed_plain_full: false,
            engine_cache_configs: None,
        };
    }
    discover_full_attention_http_worker(client, worker_url, api_key).await
}

/// Split a pinned data-parallel rank, as in `http://host:8000@3`.
pub(crate) fn split_worker_rank(worker_url: &str) -> (&str, Option<u32>) {
    let Some((base, rank)) = worker_url.rsplit_once('@') else {
        return (worker_url, None);
    };
    match rank.parse() {
        Ok(rank) if !base.is_empty() => (base, Some(rank)),
        _ => (worker_url, None),
    }
}

fn static_worker_discovery(worker_url: &str, config: &StaticKvEventSources) -> WorkerDiscovery {
    let model = config.model.trim().to_owned();
    let (_, rank_suffix) = split_worker_rank(worker_url);
    let server_info = ServerInfo {
        // Model/block size are declarations; the registry URL is not a boot ID.
        model_id: Some(model.clone()),
        instance_id: Some(worker_url.to_owned()),
        kv_block_size: Some(config.block_size),
        ..Default::default()
    };
    let requires_rank_alignment = rank_suffix.is_some() || config.hybrid.is_some();
    if (rank_suffix.is_some() && config.endpoints.len() > 1)
        || config.replay_endpoints.as_ref().is_some_and(|replay| {
            replay.len() > config.endpoints.len()
                || (requires_rank_alignment && replay.len() != config.endpoints.len())
        })
    {
        return WorkerDiscovery {
            server_info: Ok(server_info),
            sources: Err(SourcesUnavailable {
                probe: SourcesProbe::StaticMismatch,
                detail: format!(
                    "invalid static publisher and replay endpoint counts for '{worker_url}'"
                ),
            }),
            model_id: Ok(Some(model)),
            base_models: None,
            hybrid: config.hybrid.clone(),
            assumed_plain_full: false,
            engine_cache_configs: None,
        };
    }
    let sources = config
        .endpoints
        .iter()
        .enumerate()
        .map(|(index, endpoint)| {
            // With a rank-pinned URL the operator declares exactly that rank's
            // publisher, so the single configured endpoint belongs to it.
            let rank = rank_suffix.unwrap_or(index as u32);
            (
                rank.to_string(),
                KvEventSourceEntry {
                    endpoint: endpoint.clone(),
                    replay_endpoint: config
                        .replay_endpoints
                        .as_ref()
                        .and_then(|replay| replay.get(index))
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
                },
            )
        })
        .collect();
    WorkerDiscovery {
        server_info: Ok(server_info),
        sources: Ok(sources),
        model_id: Ok(Some(model)),
        base_models: None,
        hybrid: config.hybrid.clone(),
        engine_cache_configs: None,
        assumed_plain_full: config
            .hybrid
            .as_ref()
            .is_some_and(CacheCapabilities::is_plain_full_attention),
    }
}

/// A declared hybrid layout whose per-rank publishers the engine reports.
fn layout_only(discovery: &WorkerDiscovery) -> bool {
    discovery.hybrid.is_some() && discovery.sources.as_ref().is_ok_and(HashMap::is_empty)
}

/// The layout names the publisher topic each rank must use. A discovered
/// publisher that advertises another topic belongs to a different boot, so
/// subscribing to it would index the wrong stream.
fn layout_topic_mismatch(
    layout: Option<&CacheCapabilities>,
    sources: &KvEventSourcesResponse,
) -> Option<String> {
    let layout = layout?;
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
    let (server, (sources, layout)) = tokio::join!(
        query_worker_metadata_and_models(client, worker_url, api_key),
        fetch_native_sources(client, worker_url, api_key, CONTROL_TIMEOUT),
    );
    let server_info = server.server_info;
    let model_id = server_info
        .as_ref()
        .map(|info| Some(server_model_id(info).unwrap_or_else(|| worker_url.to_string())))
        .map_err(Clone::clone);
    let mut discovery = native_worker_discovery(worker_url, server_info, sources, model_id, layout);
    discovery.base_models = server.base_models;
    discovery.engine_cache_configs = server.engine_cache_configs;
    discovery
}

fn native_worker_discovery(
    worker_url: &str,
    mut server_info: Result<ServerInfo, String>,
    sources: Result<KvEventSourcesResponse, SourcesUnavailable>,
    model_id: Result<Option<String>, String>,
    layout: Result<Option<CacheCapabilities>, String>,
) -> WorkerDiscovery {
    let (hybrid, assumed_plain_full) = match (&server_info, &sources, &model_id, layout) {
        (Ok(info), Ok(sources), Ok(Some(model)), Ok(Some(layout))) => {
            match validate_native_publishers(&layout, sources, info, model) {
                Ok(()) => {
                    info!(worker_url, "using the worker's discovered KV cache layout");
                    (Some(layout), false)
                }
                Err(reason) => {
                    debug!(worker_url, %reason, "worker-reported KV layout rejected");
                    (None, false)
                }
            }
        }
        (Ok(info), Ok(sources), Ok(Some(model)), Ok(None)) => {
            let layout = synthesize_plain_full_layout(info, sources, model);
            if layout.is_some() {
                debug!(
                    worker_url,
                    "checking the default full-attention KV hash contract"
                );
            }
            let assumed = layout.is_some();
            (layout, assumed)
        }
        (_, _, _, Err(error)) => {
            debug!(worker_url, reason = %error, "worker-reported KV layout rejected");
            (None, false)
        }
        (_, _, _, Ok(Some(_))) => {
            let reason = if let Err(error) = &server_info {
                error.as_str()
            } else if let Err(error) = &sources {
                error.detail.as_str()
            } else if let Err(error) = &model_id {
                error.as_str()
            } else {
                "worker did not report a served model"
            };
            debug!(worker_url, %reason, "worker-reported KV layout rejected");
            (None, false)
        }
        _ => (None, false),
    };
    if let (Ok(info), Some(layout)) = (&mut server_info, &hybrid) {
        if !layout.is_plain_full_attention() {
            // Policy matching uses hash units. The attention block can be
            // larger than the hash unit when partial pages are reusable.
            info.kv_block_size = layout.ranks.first().and_then(|rank| rank.hash_block_size);
            info.effective_attention_block_size = None;
        }
    }
    WorkerDiscovery {
        server_info,
        sources,
        model_id,
        base_models: None,
        hybrid,
        assumed_plain_full,
        engine_cache_configs: None,
    }
}

fn validate_native_publishers(
    layout: &CacheCapabilities,
    sources: &KvEventSourcesResponse,
    info: &ServerInfo,
    model: &str,
) -> Result<(), String> {
    let active = sources.iter().filter(|(_, source)| events_enabled(source));
    if active.clone().count() != layout.ranks.len() {
        return Err(
            "the reported layouts and enabled event sources have different rank counts".into(),
        );
    }
    let engine_model = layout_model(info, model, layout);
    let plain = layout.is_plain_full_attention();
    let mut topics = HashSet::new();
    for (key, source) in active {
        let rejected = |reason| format!("DP rank {key}: {reason}");
        let rank = key
            .parse::<u32>()
            .ok()
            .filter(|rank| key == &rank.to_string())
            .ok_or_else(|| rejected("invalid event source rank"))?;
        if !topics.insert(source.topic.as_str()) {
            return Err(rejected("event topic is shared by multiple ranks"));
        }
        let verified = layout
            .verify_layout(
                rank,
                Some(engine_model),
                Some(&source.topic),
                source.replay_endpoint.as_deref(),
            )
            .map_err(rejected)?;
        if plain && Some(verified.hash_block_size) != discovered_block_size(info) {
            return Err(rejected(
                "layout hash block size differs from worker metadata",
            ));
        }
    }
    Ok(())
}

/// Omitting a layout declares ordinary single-group attention. Root and block
/// size still require matching store events before the source can earn credit.
fn synthesize_plain_full_layout(
    info: &ServerInfo,
    sources: &KvEventSourcesResponse,
    model: &str,
) -> Option<CacheCapabilities> {
    // The HTTP discovery error path may substitute the URL as a model name.
    server_model_id(info)?;
    let size = discovered_block_size(info)?;
    let mut ranks = sources
        .iter()
        .filter(|(_, source)| events_enabled(source))
        .map(|(key, source)| {
            let rank = key.parse::<u32>().ok()?;
            if key != &rank.to_string()
                || source.topic.is_empty()
                || source.replay_endpoint.as_deref().is_none_or(str::is_empty)
            {
                return None;
            }
            Some(plain_full_rank(
                rank,
                model.to_owned(),
                source.topic.clone(),
                size,
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    if ranks.is_empty() {
        return None;
    }
    ranks.sort_unstable_by_key(|rank| rank.rank);
    Some(CacheCapabilities {
        schema_version: 1,
        adapters: Vec::new(),
        ranks,
    })
}

pub(in crate::kv_index) fn plain_full_rank(
    rank: u32,
    model: String,
    topic: String,
    size: u32,
) -> RankCapability {
    RankCapability {
        rank,
        declared_groups: Some(1),
        model,
        weight_version: "default".into(),
        publisher_topic: Some(topic),
        instance_topic: true,
        hash_algorithm: "sha256_cbor".into(),
        root_hash: None,
        hash_seed: Some("vllm-none-hash".into()),
        full_hashes: true,
        hash_block_size: Some(size),
        partial_hash_hits: false,
        hash_lookahead_tokens: 0,
        speculative_tokens: 0,
        tail_drop_groups: None,
        dense_lookup_groups: None,
        prefix_caching: true,
        groups: vec![GroupCapability {
            group_idx: 0,
            kind: "full_attention".into(),
            spec_type: "FullAttentionSpec".into(),
            block_size: size,
            sliding_window: None,
            prefix_cacheable: true,
            prefix_replay_tokens: 0,
            mamba_cache_mode: None,
            tokens_per_state: None,
            cache_role: None,
        }],
    }
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
            .unwrap_or(if config.hybrid.is_some() && config.endpoints.is_empty() {
                ""
            } else {
                worker_url
            })
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
    let (worker_url, _) = split_worker_rank(worker_url);
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
    effective_block_size(
        server_info.effective_attention_block_size,
        server_info.kv_block_size,
    )
}

fn effective_block_size(effective: Option<u32>, configured: Option<u32>) -> Option<u32> {
    effective
        .filter(|&size| size != 0)
        .or_else(|| configured.filter(|&size| size != 0))
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
                "publisher_topic": format!("topic-{rank}"),
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
            format!("{base}@3"),
            serde_json::json!({
                "endpoints": ["tcp://rank3:5557"], "replay_endpoints": ["tcp://rank3:5558"],
                "topic": "topic-3", "block_size": 16, "model": "m3",
                "hybrid": {"schema_version": 1, "ranks": [rank(3, "m3")]},
            }),
        );
        let (mut supervisor, _signals) = KvIndexSupervisor::indexed(reqwest::Client::new());
        supervisor.static_sources =
            serde_json::from_value(serde_json::Value::Object(sources)).expect("sources");
        let resolved = supervisor
            .resolved_static_source(base)
            .expect("base resolves with its rank declarations");
        let hybrid = resolved.hybrid.expect("hybrid layout");
        assert_eq!(hybrid.ranks.len(), 2);
        assert_eq!(hybrid.ranks[0].rank, 0);
        assert_eq!(hybrid.ranks[0].model, "m0");
        assert_eq!(hybrid.ranks[1].rank, 3);
        assert_eq!(hybrid.ranks[1].model, "m3");
        let sources = resolved.sources.unwrap();
        assert_eq!(sources.len(), 2);
        assert_eq!(sources["0"].endpoint, "tcp://base:5557");
        assert!(sources["0"].replay_endpoint.is_none());
        assert_eq!(sources["3"].endpoint, "tcp://rank3:5557");
        assert_eq!(
            sources["3"].replay_endpoint.as_deref(),
            Some("tcp://rank3:5558")
        );

        let pinned = format!("{base}@3");
        let duplicate = format!("{base}@03");
        let config = supervisor.static_sources[&pinned].clone();
        supervisor.static_sources.insert(duplicate.clone(), config);
        let error = supervisor
            .resolved_static_source(base)
            .unwrap()
            .sources
            .unwrap_err();
        assert!(error.detail.contains("duplicate static KV event sources"));
        supervisor.static_sources.remove(&duplicate);

        supervisor
            .static_sources
            .get_mut(&pinned)
            .unwrap()
            .replay_endpoints = Some(Vec::new());
        assert!(supervisor
            .resolved_static_source(&pinned)
            .unwrap()
            .sources
            .is_err());
        assert!(supervisor
            .resolved_static_source(base)
            .unwrap()
            .sources
            .is_err());
        supervisor
            .static_sources
            .get_mut(&pinned)
            .unwrap()
            .replay_endpoints = Some(vec!["tcp://rank3:5558".into()]);

        let declaration = supervisor.static_sources.get_mut(base).unwrap();
        declaration.endpoints.push("tcp://rank1:5557".into());
        declaration.replay_endpoints = Some(vec!["tcp://rank1:5558".into()]);
        assert!(supervisor
            .resolved_static_source(base)
            .unwrap()
            .sources
            .is_err());
    }
}
