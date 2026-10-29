//! Worker discovery: query /get_server_info + /kv_event_sources, spawn one
//! subscriber per (worker, rank). One indexer per cache-identity core.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use parking_lot::Mutex;
use serde::Deserialize;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::config::StaticKvEventSources;
use crate::kv_index::indexer::KvBlockIndexer;
use crate::kv_index::subscriber::{EventSource, SubscriberHandle};
use crate::kv_index::{HashMode, IngestionSignal, SourceId};
use crate::protocols::worker_spec::ServerInfo;

/// The worker-stable core of a cache identity. One indexer per key.
///
/// ponytail: full per-block identity isolation needs `lora_name`/`extra_keys`
/// on every remove, but vLLM's `BlockRemoved` carries neither. Keying on the
/// worker-stable triple isolates the common multi-model axis (different models
/// never collide in one tree) and defers lora-level isolation to an upstream
/// change that adds identity to remove events.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub model: Arc<str>,
    pub hash_mode: HashMode,
    pub block_size: u32,
}

/// One per-DP-rank entry from /kv_event_sources. Mirrors vLLM's KVEventsConfig.
#[derive(Debug, Clone, Default, Deserialize)]
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
    pub(crate) sources: Result<KvEventSourcesResponse, String>,
    /// gRPC has a separate GetModelInfo RPC; preserve its failure instead of
    /// silently turning the worker URL into a cache identity.
    model_id: Result<Option<String>, String>,
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
}

struct LiveSource {
    publisher: String,
    topic: String,
    rank: u32,
    lease: super::binding::SourceLease,
}

/// Orchestrates discovery + subscriber lifecycle for the router's workers.
/// One indexer per `CacheKey`, shared by every `(worker, rank)` of that
/// identity.
pub struct KvIndexSupervisor {
    client: reqwest::Client,
    api_key: Option<String>,
    hash_mode: HashMode,
    full_attention_http: bool,
    http_pd: bool,
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
    /// First-seen block_size, for cross-worker mismatch warning.
    first_block_size: Mutex<Option<u32>>,
}

impl std::fmt::Debug for KvIndexSupervisor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("KvIndexSupervisor")
            .field("hash_mode", &self.hash_mode)
            .field("indexers", &self.indexers.len())
            .field("workers", &self.workers.lock().len())
            .finish()
    }
}

impl KvIndexSupervisor {
    /// Construct the supervisor. The returned receiver streams ingestion signals
    /// to the trust arbiter; drop it to ignore signals.
    pub fn new(
        client: reqwest::Client,
        hash_mode: HashMode,
    ) -> (Self, mpsc::Receiver<IngestionSignal>) {
        let (signal_tx, signal_rx) = mpsc::channel::<IngestionSignal>(256);
        let supervisor = KvIndexSupervisor {
            client,
            api_key: None,
            hash_mode,
            full_attention_http: false,
            http_pd: false,
            static_sources: HashMap::new(),
            static_pd_sources: HashMap::new(),
            ctx: Arc::new(zmq::Context::new()),
            indexers: DashMap::new(),
            identity_refs: Mutex::new(HashMap::new()),
            signal_tx,
            workers: Mutex::new(HashMap::new()),
            first_block_size: Mutex::new(None),
        };
        (supervisor, signal_rx)
    }

    pub(crate) fn new_http_full_attention(
        client: reqwest::Client,
        hash_mode: HashMode,
    ) -> (Self, mpsc::Receiver<IngestionSignal>) {
        let (mut supervisor, signals) = Self::new(client, hash_mode);
        supervisor.full_attention_http = true;
        (supervisor, signals)
    }

    /// HTTP P/D accepts the native Python control endpoints and full-attention KV
    /// events.
    pub(crate) fn new_pd(
        client: reqwest::Client,
        hash_mode: HashMode,
        static_pd_sources: HashMap<String, crate::config::types::PdKvEventSources>,
    ) -> (Self, mpsc::Receiver<IngestionSignal>) {
        let (mut supervisor, signals) = Self::new(client, hash_mode);
        supervisor.http_pd = true;
        supervisor.static_pd_sources = static_pd_sources;
        (supervisor, signals)
    }

    pub fn hash_mode(&self) -> HashMode {
        self.hash_mode
    }

    pub(crate) fn set_api_key(&mut self, api_key: Option<String>) {
        self.api_key = api_key;
    }

    pub fn with_static_sources(
        mut self,
        static_sources: HashMap<String, StaticKvEventSources>,
    ) -> Self {
        self.static_sources = static_sources;
        self
    }

    /// Query both endpoints for `worker_url`, spawn one subscriber per rank into
    /// this worker's identity's indexer. A 404 on /kv_event_sources (engine
    /// without the endpoint or events off) is a soft skip.
    pub async fn on_worker_added(&self, worker_url: &str) -> Result<WorkerKvInfo, String> {
        // Discovery awaits without the owner lock. A failed/stale re-add must
        // leave the prior registration and its observed placement intact.
        let observed = self
            .workers
            .lock()
            .get(worker_url)
            .map(|worker| worker.active.clone());
        let discovery = self.discover(worker_url).await;
        self.install_observed_worker(worker_url, discovery, observed)
    }

    /// Prepare metadata once; P/D refresh fingerprints and installs this same snapshot.
    pub(crate) async fn discover(&self, worker_url: &str) -> WorkerDiscovery {
        if self.http_pd {
            return match self.static_pd_sources.get(worker_url) {
                Some(config) => static_pd_discovery(worker_url, config),
                None => {
                    discover_worker(&self.client, worker_url, true, self.api_key.as_deref()).await
                }
            };
        }
        let grpc = crate::backend::is_grpc_url(worker_url);
        let static_source = self.static_sources.get(worker_url);
        let full_attention_http = !grpc && self.full_attention_http;
        match static_source {
            Some(config) => {
                info!(
                    worker_url,
                    model = %config.model,
                    block_size = config.block_size,
                    ranks = config.endpoints.len(),
                    "using configured static KV event sources"
                );
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

    pub(crate) fn install_worker(
        &self,
        worker_url: &str,
        discovery: WorkerDiscovery,
    ) -> Result<WorkerKvInfo, String> {
        let observed = self
            .workers
            .lock()
            .get(worker_url)
            .map(|worker| worker.active.clone());
        self.install_observed_worker(worker_url, discovery, observed)
    }

    fn install_observed_worker(
        &self,
        worker_url: &str,
        discovery: WorkerDiscovery,
        observed: Option<Arc<AtomicBool>>,
    ) -> Result<WorkerKvInfo, String> {
        let grpc = crate::backend::is_grpc_url(worker_url);
        let full_attention_http =
            !grpc && self.full_attention_http && !self.static_sources.contains_key(worker_url);
        let full_attention = full_attention_http || (self.http_pd && !grpc);
        let server_info = discovery.server_info.unwrap_or_else(|e| {
            warn!(
                "kv_index {}: {} — falling back to URL as instance/model id",
                worker_url, e
            );
            ServerInfo {
                model_id: Some(worker_url.to_string()),
                ..Default::default()
            }
        });

        let instance_id = server_info
            .instance_id
            .clone()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| worker_url.to_string());
        let block_size = discovered_block_size(&server_info);
        let model_id = match discovery.model_id {
            Ok(Some(model_id)) if !model_id.trim().is_empty() => Some(model_id),
            Ok(_) => {
                warn!(
                    "kv_index {}: GetModelInfo returned an empty model_id — cannot create cache identity",
                    worker_url
                );
                None
            }
            Err(error) => {
                warn!(
                    "kv_index {}: GetModelInfo failed: {} — cannot create cache identity",
                    worker_url, error
                );
                None
            }
        };

        let sources = match discovery.sources {
            Ok(s) => s,
            Err(e) => {
                warn!(
                    "kv_index {}: /kv_event_sources: {} — no subscribers",
                    worker_url, e
                );
                return Ok(WorkerKvInfo {
                    instance_id,
                    block_size,
                    ranks: vec![],
                    cache_key: None,
                });
            }
        };

        let Some(block_size) = block_size else {
            warn!(
                "kv_index {}: no block_size from /get_server_info — cannot index",
                worker_url
            );
            return Ok(WorkerKvInfo {
                instance_id,
                block_size: None,
                ranks: vec![],
                cache_key: None,
            });
        };
        let Some(model_id) = model_id else {
            return Ok(WorkerKvInfo {
                instance_id,
                block_size: Some(block_size),
                ranks: vec![],
                cache_key: None,
            });
        };
        let key = CacheKey {
            model: Arc::from(model_id.as_str()),
            hash_mode: self.hash_mode,
            block_size,
        };
        let worker_host = worker_host(worker_url).unwrap_or_else(|| worker_url.to_string());
        let source_id: SourceId = Arc::from(instance_id.as_str());
        let mut event_sources = Vec::with_capacity(sources.len());
        let mut normalized_ranks = HashSet::new();

        for (rank_str, entry) in &sources {
            if !events_enabled(entry) {
                continue;
            }
            let rank: u32 = rank_str.parse().map_err(|_| "invalid KV source rank")?;
            if !normalized_ranks.insert(rank) {
                return Err("duplicate normalized KV source rank".into());
            }
            let pub_endpoint = resolve_endpoint(&entry.endpoint, &worker_host)
                .map_err(|error| format!("invalid KV publisher endpoint: {error}"))?;
            let replay_endpoint = entry
                .replay_endpoint
                .as_deref()
                .map(|e| resolve_endpoint(e, &worker_host))
                .transpose()
                .map_err(|error| format!("invalid KV replay endpoint: {error}"))?;
            let ev = EventSource {
                source: source_id.clone(),
                dp_rank: rank,
                pub_endpoint,
                replay_endpoint,
                topic: entry.topic.clone(),
                hwm: entry.hwm,
            };
            event_sources.push(ev);
        }

        // Full-attention HTTP bindings require plain origins. Generic discovery retains
        // its backend-specific URI scope, including HTTP base paths.
        let worker_origin = if full_attention_http {
            Some(super::endpoints::canonical_worker(worker_url)?.0)
        } else {
            None
        };
        // Admission and publication share the existing source-owner lock. Two
        // concurrent discoveries must not both accept an ambiguous binding.
        let mut workers = self.workers.lock();
        let unchanged = match (observed.as_ref(), workers.get(worker_url)) {
            (Some(observed), Some(current)) => Arc::ptr_eq(observed, &current.active),
            (None, None) => true,
            _ => false,
        };
        if !unchanged {
            return Err("KV worker was registered while discovery was pending".into());
        }
        if worker_origin.as_ref().is_some_and(|origin| {
            workers.iter().any(|(url, worker)| {
                url != worker_url && worker.worker_origin.as_ref() == Some(origin)
            })
        }) {
            return Err("KV worker origins must be independent".into());
        }
        let mut replay_endpoints = Vec::new();
        for source in &event_sources {
            if let Some(endpoint) = &source.replay_endpoint {
                // This replay protocol sends only start_seq, not a source key.
                if replay_endpoints.contains(endpoint)
                    || workers.iter().any(|(url, worker)| {
                        url != worker_url && worker.replay_endpoints.contains(endpoint)
                    })
                {
                    return Err("KV replay endpoints must be independent".into());
                }
                replay_endpoints.push(endpoint.clone());
            }
            for (url, worker) in workers.iter().filter(|(url, _)| url.as_str() != worker_url) {
                if worker.live_sources.iter().any(|live| {
                    live.publisher == source.pub_endpoint
                        && live.topic == source.topic
                        && live.rank == source.dp_rank
                }) {
                    return Err("live KV streams must be independent".into());
                }
                if worker.key == key
                    && worker.source_id == source_id
                    && worker.ranks.contains(&source.dp_rank)
                {
                    return Err(format!("KV index targets must be independent: {url}"));
                }
            }
        }
        // No admission can fail below this point. Share rank guards by exact
        // publisher/topic and promote them monotonically before publishing B.
        let active = Arc::new(AtomicBool::new(true));
        let mut live_sources: Vec<LiveSource> = Vec::with_capacity(event_sources.len());
        for source in &event_sources {
            let strict_rank = workers
                .values()
                .flat_map(|worker| &worker.live_sources)
                .chain(live_sources.iter())
                .find(|live| live.publisher == source.pub_endpoint && live.topic == source.topic)
                .map(|live| live.lease.strict_rank.clone())
                .unwrap_or_else(|| Arc::new(Mutex::new(false)));
            let shared_rank = workers
                .iter()
                .filter(|(url, _)| url.as_str() != worker_url)
                .flat_map(|(_, worker)| &worker.live_sources)
                .any(|live| {
                    live.publisher == source.pub_endpoint
                        && live.topic == source.topic
                        && live.rank != source.dp_rank
                })
                || event_sources.iter().any(|other| {
                    other.pub_endpoint == source.pub_endpoint
                        && other.topic == source.topic
                        && other.dp_rank != source.dp_rank
                });
            if shared_rank {
                let mut strict = strict_rank.lock();
                if !*strict {
                    *strict = true;
                    for worker in workers.values() {
                        for live in &worker.live_sources {
                            if live.publisher == source.pub_endpoint && live.topic == source.topic {
                                worker
                                    .indexer
                                    .invalidate_source(&worker.source_id, live.rank);
                            }
                        }
                    }
                }
            }
            live_sources.push(LiveSource {
                publisher: source.pub_endpoint.clone(),
                topic: source.topic.clone(),
                rank: source.dp_rank,
                lease: super::binding::SourceLease {
                    strict_rank,
                    active: active.clone(),
                },
            });
        }
        if let Some(previous) = workers.remove(worker_url) {
            self.retire_worker(previous);
        }
        let indexer = self.acquire_identity(&key);
        let mut ranks = Vec::with_capacity(event_sources.len());
        let mut handles = Vec::with_capacity(event_sources.len());
        for (ev, live) in event_sources.into_iter().zip(&live_sources) {
            let rank = ev.dp_rank;
            let handle = crate::kv_index::subscriber::spawn_bound(
                ev,
                indexer.clone(),
                Some(self.signal_tx.clone()),
                self.ctx.clone(),
                full_attention.then_some(key.block_size),
                live.lease.clone(),
            );
            info!(
                "kv_index {}: spawned subscriber rank{} (instance {})",
                worker_url, rank, instance_id
            );
            handles.push(handle);
            ranks.push(rank);
        }

        workers.insert(
            worker_url.to_string(),
            WorkerSubs {
                active,
                key: key.clone(),
                source_id,
                worker_origin,
                replay_endpoints,
                live_sources,
                ranks: ranks.clone(),
                indexer,
                handles,
            },
        );
        self.check_block_size(worker_url, Some(block_size));
        Ok(WorkerKvInfo {
            instance_id,
            block_size: Some(block_size),
            ranks,
            cache_key: Some(key),
        })
    }

    /// Shut down this worker's subscribers, clear its residency, release its
    /// identity ref.
    pub fn on_worker_removed(&self, worker_url: &str) {
        let mut workers = self.workers.lock();
        if let Some(subs) = workers.remove(worker_url) {
            self.retire_worker(subs);
            info!("kv_index {}: removed subscribers", worker_url);
        }
    }

    fn retire_worker(&self, subs: WorkerSubs) {
        subs.active.store(false, Ordering::Release);
        for handle in &subs.handles {
            handle.shutdown();
        }
        for live in &subs.live_sources {
            let _guard = live.lease.strict_rank.lock();
            subs.indexer.invalidate_source(&subs.source_id, live.rank);
        }
        self.release_identity(&subs.key);
    }

    /// Shut down every subscriber and drop all indexers.
    pub fn shutdown(&self) {
        let mut workers = self.workers.lock();
        for (_url, subs) in workers.drain() {
            self.retire_worker(subs);
        }
        drop(workers);
        info!("kv_index supervisor shut down");
    }

    /// Expose the discovered worker's live index through the policy's read-only port.
    pub fn match_provider_for_worker(
        &self,
        worker_url: &str,
    ) -> Option<Arc<dyn crate::policies::TieredMatchProvider>> {
        self.workers
            .lock()
            .get(worker_url)
            .map(|worker| worker.indexer.clone() as Arc<dyn crate::policies::TieredMatchProvider>)
    }

    /// Ref-count an identity; on first acquisition create its indexer.
    fn acquire_identity(&self, key: &CacheKey) -> Arc<KvBlockIndexer> {
        let mut refs = self.identity_refs.lock();
        let count = refs.entry(key.clone()).or_insert(0);
        if *count == 0 {
            self.indexers
                .insert(key.clone(), Arc::new(KvBlockIndexer::new()));
        }
        *count += 1;
        drop(refs);
        self.indexers.get(key).expect("just inserted").clone()
    }

    /// Decrement an identity's ref-count; on last release drop its indexer.
    fn release_identity(&self, key: &CacheKey) {
        let mut refs = self.identity_refs.lock();
        if let Some(c) = refs.get_mut(key) {
            *c = c.saturating_sub(1);
            if *c == 0 {
                refs.remove(key);
                drop(refs);
                self.indexers.remove(key);
                return;
            }
        }
        drop(refs);
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
        let sources = crate::backend::control::get_grpc_kv_event_sources(worker_url, timeout).await;
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
        };
    }
    if !http_pd {
        return WorkerDiscovery {
            server_info: Err(
                "HTTP worker model and block-size metadata must be configured statically"
                    .to_string(),
            ),
            sources: query_kv_event_sources(client, worker_url).await,
            model_id: Ok(Some(worker_url.to_string())),
        };
    }
    let server_info = query_pd_server_info(client, worker_url, api_key).await;
    let model_id = server_info
        .as_ref()
        .map(|info| Some(server_model_id(info).unwrap_or_else(|| worker_url.to_string())))
        .map_err(|error| error.clone());
    WorkerDiscovery {
        server_info,
        sources: if http_pd {
            query_pd_kv_event_sources(client, worker_url, api_key).await
        } else {
            query_kv_event_sources(client, worker_url).await
        },
        model_id,
    }
}

fn static_worker_discovery(worker_url: &str, config: &StaticKvEventSources) -> WorkerDiscovery {
    let model = config.model.trim().to_owned();
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
        .map(|(rank, endpoint)| {
            (
                rank.to_string(),
                KvEventSourceEntry {
                    endpoint: endpoint.clone(),
                    replay_endpoint: config
                        .replay_endpoints
                        .as_ref()
                        .and_then(|replay| replay.get(rank).cloned()),
                    topic: config.topic.clone(),
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
    }
}
// Only the full-attention HTTP adapter uses this optional control-metadata contract.
// Generic discovery retains main's behavior and does not assume stock vLLM
// exposes /get_server_info.
async fn discover_full_attention_http_worker(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> WorkerDiscovery {
    let server_info =
        query_worker_json::<ServerInfo>(client, worker_url, "get_server_info", true, api_key).await;
    let model_id = server_info
        .as_ref()
        .map(|info| Some(server_model_id(info).unwrap_or_else(|| worker_url.to_string())))
        .map_err(Clone::clone);
    WorkerDiscovery {
        server_info,
        sources: query_worker_json(client, worker_url, "kv_event_sources", true, api_key).await,
        model_id,
    }
}

/// Build a discovery result purely from static configuration, without
/// querying the worker's HTTP control plane.
fn static_pd_discovery(
    worker_url: &str,
    config: &crate::config::types::PdKvEventSources,
) -> WorkerDiscovery {
    static_worker_discovery(
        worker_url,
        &StaticKvEventSources {
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
        },
    )
}

fn grpc_kv_event_source_entries(
    worker_url: &str,
    sources: Vec<crate::backend::pb::KvEventSource>,
) -> KvEventSourcesResponse {
    sources
        .into_iter()
        .filter_map(|source| {
            let Some(rank) = source.data_parallel_rank else {
                warn!(
                    "kv_index {}: gRPC KV event source has no data_parallel_rank — skipping source",
                    worker_url
                );
                return None;
            };
            Some((
                rank.to_string(),
                KvEventSourceEntry {
                    endpoint: source.endpoint,
                    replay_endpoint: (!source.replay_endpoint.is_empty())
                        .then_some(source.replay_endpoint),
                    topic: source.topic,
                    hwm: i32::try_from(source.hwm).ok(),
                    buffer_steps: Some(u64::from(source.buffer_steps)),
                    publisher: Some(source.transport),
                    encoding: Some(source.encoding),
                    schema_version: Some(source.schema_version),
                    // The Control RPC has no per-source enable flag; it only
                    // returns sources when KV events are enabled.
                    enable_kv_cache_events: Some(true),
                },
            ))
        })
        .collect()
}

fn server_model_id(server_info: &ServerInfo) -> Option<String> {
    server_info
        .model_id
        .clone()
        .filter(|id| !id.trim().is_empty())
        .or_else(|| {
            server_info
                .model_path
                .as_ref()
                .and_then(|path| path.split('/').next_back().map(str::to_string))
                .filter(|id| !id.trim().is_empty())
        })
}

fn grpc_model_id(
    model: Result<crate::backend::pb::ModelInfo, String>,
) -> Result<Option<String>, String> {
    model.map(|model| {
        let model_id = model.model_id;
        (!model_id.trim().is_empty()).then_some(model_id)
    })
}

async fn query_kv_event_sources(
    client: &reqwest::Client,
    worker_url: &str,
) -> Result<KvEventSourcesResponse, String> {
    query_worker_json(client, worker_url, "kv_event_sources", false, None).await
}

async fn query_worker_json<T: serde::de::DeserializeOwned>(
    client: &reqwest::Client,
    worker_url: &str,
    path: &str,
    authenticated: bool,
    api_key: Option<&str>,
) -> Result<T, String> {
    let url = format!("{}/{path}", worker_url.trim_end_matches('/'));
    let request = if authenticated {
        worker_get(client, &url, api_key)
    } else {
        client.get(&url)
    };
    let response = request
        .send()
        .await
        .map_err(|e| format!("{path} {url}: {e}"))?;
    if !response.status().is_success() {
        return Err(format!("{path} {url}: status {}", response.status()));
    }
    response
        .json()
        .await
        .map_err(|e| format!("{path} parse: {e}"))
}

/// A rank emits events only for the one wire format the subscriber decodes.
fn events_enabled(entry: &KvEventSourceEntry) -> bool {
    if entry.publisher.as_deref() != Some("zmq")
        || entry.encoding.as_deref() != Some("msgpack")
        || entry.schema_version != Some(1)
    {
        return false;
    }
    if matches!(entry.enable_kv_cache_events, Some(false)) {
        return false;
    }
    !entry.endpoint.is_empty()
}

pub(crate) async fn query_pd_server_info(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Result<ServerInfo, String> {
    if let Ok(info) =
        query_worker_json::<ServerInfo>(client, worker_url, "get_server_info", true, api_key).await
    {
        if discovered_block_size(&info).is_some() {
            return Ok(info);
        }
    }
    // Native Python vLLM exposes engine configuration through this endpoint.
    let body: serde_json::Value = query_worker_json(
        client,
        worker_url,
        "server_info?config_format=json",
        true,
        api_key,
    )
    .await?;
    let config = body
        .get("vllm_config")
        .ok_or("server_info: missing vllm_config")?;
    let model = config
        .pointer("/model_config/model")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let block_size = config
        .pointer("/cache_config/block_size")
        .and_then(serde_json::Value::as_u64)
        .and_then(|size| u32::try_from(size).ok())
        .filter(|size| *size > 0);
    Ok(ServerInfo {
        model_id: model.clone(),
        model_path: model,
        kv_block_size: block_size,
        ..Default::default()
    })
}

pub(crate) async fn query_pd_kv_event_sources(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Result<KvEventSourcesResponse, String> {
    let mut sources: KvEventSourcesResponse =
        query_worker_json(client, worker_url, "kv_event_sources", true, api_key).await?;
    // Native Python KVEventsConfig omits these fixed ZMQ wire declarations.
    for source in sources.values_mut() {
        if source.publisher.as_deref() == Some("zmq")
            && source.encoding.is_none()
            && source.schema_version.is_none()
        {
            source.encoding = Some("msgpack".into());
            source.schema_version = Some(1);
        }
    }
    Ok(sources)
}

pub(crate) fn worker_get(
    client: &reqwest::Client,
    url: &str,
    api_key: Option<&str>,
) -> reqwest::RequestBuilder {
    let request = client.get(url);
    match api_key {
        Some(key) => request.bearer_auth(key),
        None => request,
    }
}

/// Rewrite a ZMQ bind host before validating the TCP connect endpoint.
/// `tcp://*:5557` → `tcp://10.0.0.5:5557`.
fn resolve_endpoint(endpoint: &str, worker_host: &str) -> Result<String, String> {
    if let Some(rest) = endpoint.strip_prefix("tcp://") {
        if let Some((host, port)) = rest.rsplit_once(':') {
            if matches!(host, "*" | "0.0.0.0" | "[::]") {
                return super::endpoints::canonical_endpoint(&format!(
                    "tcp://{worker_host}:{port}"
                ));
            }
        }
        return super::endpoints::canonical_endpoint(endpoint);
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

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::kv_index::{MatchQuery, StorageTier};

    #[tokio::test]
    async fn configured_event_sources_skip_unavailable_control_discovery() {
        let publisher_context = zmq::Context::new();
        let publisher = publisher_context.socket(zmq::PUB).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
        for worker_url in ["grpc://127.0.0.1:1", "http://127.0.0.1:1/proxy"] {
            let source = StaticKvEventSources {
                endpoints: vec![endpoint.clone(), endpoint.clone()],
                replay_endpoints: None,
                block_size: 32,
                model: "shared-model".into(),
                topic: "kv/generic".into(),
            };
            let sources = HashMap::from([(worker_url.to_string(), source)]);
            // An explicit static entry must remain generic when
            // installed by the HTTP Router's full-attention constructor.
            let (supervisor, _signals) = KvIndexSupervisor::new_http_full_attention(
                reqwest::Client::new(),
                HashMode::Sha256,
            );
            let supervisor = supervisor.with_static_sources(sources);

            let mut info = supervisor.on_worker_added(worker_url).await.unwrap();
            assert_eq!(info.block_size, Some(32));
            info.ranks.sort_unstable();
            assert_eq!(info.ranks, vec![0, 1]);
            assert_eq!(info.cache_key.unwrap().model.as_ref(), "shared-model");
            assert!(supervisor.match_provider_for_worker(worker_url).is_some());
            assert!(supervisor.workers.lock()[worker_url]
                .worker_origin
                .is_none());
            supervisor.on_worker_removed(worker_url);
        }
    }

    #[test]
    fn worker_auth_uses_only_the_configured_key() {
        let client = reqwest::Client::new();
        for (key, expected) in [
            (None, None),
            (Some("backend-key"), Some("Bearer backend-key")),
        ] {
            let request = worker_get(&client, "https://worker/v1/models", key)
                .build()
                .unwrap();
            assert_eq!(
                request
                    .headers()
                    .get("authorization")
                    .map(|value| value.to_str().unwrap()),
                expected
            );
        }
    }

    #[test]
    fn resolve_endpoint_rewrites_bind_star() {
        assert_eq!(
            resolve_endpoint("tcp://*:5557", "10.0.0.5").unwrap(),
            "tcp://10.0.0.5:5557"
        );
        assert_eq!(
            resolve_endpoint("tcp://0.0.0.0:5558", "10.0.0.5").unwrap(),
            "tcp://10.0.0.5:5558"
        );
    }

    #[test]
    fn resolve_endpoint_keeps_resolved_host() {
        assert_eq!(
            resolve_endpoint("tcp://10.0.0.5:5557", "ignored").unwrap(),
            "tcp://10.0.0.5:5557"
        );
    }

    #[test]
    fn resolve_endpoint_passthrough_non_tcp() {
        assert_eq!(
            resolve_endpoint("ipc:///tmp/x", "h").unwrap(),
            "ipc:///tmp/x"
        );
        assert_eq!(resolve_endpoint("inproc://x", "h").unwrap(), "inproc://x");
    }

    #[test]
    fn resolved_tcp_sources_reuse_connect_validation_without_http_worker_admission() {
        let host = worker_host("grpc://[::1]:50051").unwrap();
        assert_eq!(
            resolve_endpoint("tcp://[::]:5557", &host).unwrap(),
            "tcp://[::1]:5557"
        );
        assert_eq!(
            resolve_endpoint("tcp://[::ffff:127.0.0.1]:5557", &host).unwrap(),
            "tcp://127.0.0.1:5557"
        );
        assert!(resolve_endpoint("tcp://*:0", &host).is_err());
    }

    #[test]
    fn worker_host_extracts_from_url() {
        assert_eq!(
            worker_host("http://10.0.0.5:8000").as_deref(),
            Some("10.0.0.5")
        );
        assert_eq!(worker_host("10.0.0.5:8000").as_deref(), Some("10.0.0.5"));
        assert_eq!(
            worker_host("http://host.example:80").as_deref(),
            Some("host.example")
        );
    }

    #[test]
    fn events_enabled_gates_on_publisher_and_endpoint() {
        let mut e = KvEventSourceEntry::default();
        assert!(!events_enabled(&e)); // empty endpoint
        e.endpoint = "tcp://*:5557".into();
        e.publisher = Some("zmq".into());
        assert!(!events_enabled(&e)); // Legacy discovery requires explicit wire metadata.
        e.encoding = Some("msgpack".into());
        assert!(!events_enabled(&e)); // Reject a partial explicit declaration.
        e.schema_version = Some(1);
        assert!(events_enabled(&e));
        e.publisher = Some("null".into());
        assert!(!events_enabled(&e));
        e.publisher = Some("other".into());
        assert!(!events_enabled(&e));
        e.publisher = Some("zmq".into());
        e.enable_kv_cache_events = Some(false);
        assert!(!events_enabled(&e));
        e.enable_kv_cache_events = Some(true);
        assert!(events_enabled(&e));
        e.encoding = Some("json".into());
        assert!(!events_enabled(&e));
        e.encoding = Some("msgpack".into());
        e.schema_version = Some(2);
        assert!(!events_enabled(&e));
    }

    #[test]
    fn grpc_model_id_preserves_errors_and_rejects_empty_ids() {
        let error = "GetModelInfo unavailable".to_string();
        assert_eq!(grpc_model_id(Err(error.clone())), Err(error));
        assert_eq!(
            grpc_model_id(Ok(crate::backend::pb::ModelInfo::default())),
            Ok(None)
        );
        assert_eq!(
            grpc_model_id(Ok(crate::backend::pb::ModelInfo {
                model_id: "   ".into(),
                ..Default::default()
            })),
            Ok(None)
        );
        assert_eq!(
            grpc_model_id(Ok(crate::backend::pb::ModelInfo {
                model_id: "model-a".into(),
                ..Default::default()
            })),
            Ok(Some("model-a".into()))
        );
    }

    #[test]
    fn server_info_parses_kv_fields_and_aliases() {
        let s: ServerInfo = serde_json::from_str(
            r#"{"model_id":"m","kv_block_size":16,"instance_id":"x","data_parallel_size":2}"#,
        )
        .unwrap();
        assert_eq!(s.model_id.as_deref(), Some("m"));
        assert_eq!(s.kv_block_size, Some(16));
        assert_eq!(s.instance_id.as_deref(), Some("x"));
        // `data_parallel_size` is tolerated but unused; rank comes from /kv_event_sources.
        // block_size alias.
        let s: ServerInfo = serde_json::from_str(r#"{"block_size":32}"#).unwrap();
        assert_eq!(s.kv_block_size, Some(32));
        // empty body → defaults, no error.
        let s: ServerInfo = serde_json::from_str("{}").unwrap();
        assert!(s.kv_block_size.is_none());
    }

    #[test]
    fn zero_block_size_is_treated_as_missing() {
        let server_info = ServerInfo {
            kv_block_size: Some(0),
            effective_attention_block_size: Some(16),
            ..Default::default()
        };
        assert_eq!(discovered_block_size(&server_info), Some(16));

        let server_info = ServerInfo {
            kv_block_size: Some(0),
            effective_attention_block_size: Some(0),
            ..Default::default()
        };
        assert_eq!(discovered_block_size(&server_info), None);
    }

    #[test]
    fn kv_event_sources_decodes_rank_map() {
        let body = r#"{
            "0": {"endpoint":"tcp://*:5557","replay_endpoint":"tcp://*:5558","topic":"","hwm":100000,"publisher":"zmq","encoding":"msgpack","schema_version":1,"enable_kv_cache_events":true},
            "1": {"endpoint":"tcp://*:5559","replay_endpoint":null,"topic":"","publisher":"zmq","encoding":"msgpack","schema_version":1}
        }"#;
        let m: KvEventSourcesResponse = serde_json::from_str(body).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m["0"].hwm, Some(100_000));
        assert_eq!(m["0"].encoding.as_deref(), Some("msgpack"));
        assert_eq!(m["0"].schema_version, Some(1));
        assert!(m["1"].replay_endpoint.is_none());
    }

    #[test]
    fn grpc_sources_without_rank_are_skipped_instead_of_positionally_assigned() {
        let source = |rank| crate::backend::pb::KvEventSource {
            transport: "zmq".to_string(),
            endpoint: "tcp://worker:5557".to_string(),
            topic: String::new(),
            replay_endpoint: String::new(),
            data_parallel_rank: rank,
            encoding: "msgpack".to_string(),
            schema_version: 1,
            buffer_steps: 1,
            hwm: 100,
            max_queue_size: 0,
        };

        let entries =
            grpc_kv_event_source_entries("grpc://worker:8000", vec![source(None), source(Some(3))]);

        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key("3"));
        assert_eq!(entries["3"].publisher.as_deref(), Some("zmq"));
        assert_eq!(entries["3"].enable_kv_cache_events, Some(true));
    }

    #[test]
    fn static_regular_sources_preserve_mapping_alias_and_topic() {
        for (worker_count, block_size, topic) in [
            (2, 16, Some("kv/alias-v2")),
            (1, 8, None),
            (1, 32, Some("")),
        ] {
            let workers: Vec<_> = (0..worker_count)
                .map(|worker| format!("http://worker-{worker}:8000"))
                .collect();
            let mut map = serde_json::Map::new();
            for (worker, url) in workers.iter().enumerate() {
                let mut source = serde_json::json!({
                    "endpoints": [format!("tcp://worker-{worker}:5557")],
                    "replay_endpoints": [format!("tcp://worker-{worker}:5558")],
                    "block_size": block_size,
                    "model": "served-alias-v2"
                });
                if let Some(topic) = topic {
                    source["topic"] = serde_json::json!(topic);
                }
                map.insert(url.clone(), source);
            }
            let sources: HashMap<String, StaticKvEventSources> =
                serde_json::from_value(serde_json::Value::Object(map)).unwrap();
            assert_eq!(sources.len(), worker_count);
            for worker in &workers {
                let discovery = static_worker_discovery(worker, &sources[worker]);
                assert_eq!(discovery.model_id, Ok(Some("served-alias-v2".into())));
                let info = discovery.server_info.unwrap();
                assert_eq!(info.instance_id.as_deref(), Some(worker.as_str()));
                assert_eq!(discovered_block_size(&info), Some(block_size));
                let entries = discovery.sources.unwrap();
                assert_eq!(entries.len(), 1);
                let entry = &entries["0"];
                assert_eq!(entry.endpoint, sources[worker].endpoints[0]);
                assert_eq!(
                    entry.replay_endpoint,
                    sources[worker]
                        .replay_endpoints
                        .as_ref()
                        .map(|endpoints| endpoints[0].clone())
                );
                assert_eq!(entry.topic, topic.unwrap_or_default());
                assert_eq!(entry.hwm, None);
                assert!(events_enabled(entry));
            }
        }
    }

    async fn source_boundary_worker(
        publisher: &str,
        replay: &str,
        instance: &str,
    ) -> (String, tokio::task::JoinHandle<()>) {
        use serde_json::json;

        let info =
            json!({"model_id": "boundary-model", "instance_id": instance, "kv_block_size": 8});
        let sources = json!({"0": {
            "endpoint": publisher, "replay_endpoint": replay, "topic": "kv/boundary",
            "publisher": "zmq", "encoding": "msgpack", "schema_version": 1,
            "enable_kv_cache_events": true
        }});
        source_boundary_worker_with_metadata(info, sources).await
    }

    async fn source_boundary_worker_with_metadata(
        info: serde_json::Value,
        sources: serde_json::Value,
    ) -> (String, tokio::task::JoinHandle<()>) {
        use axum::{routing::get, Json, Router};

        let app = Router::new()
            .route(
                "/get_server_info",
                get(move || {
                    let info = info.clone();
                    async move { Json(info) }
                }),
            )
            .route(
                "/kv_event_sources",
                get(move || {
                    let sources = sources.clone();
                    async move { Json(sources) }
                }),
            );
        let app = app.clone().nest("/proxy", app);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (worker, task)
    }

    async fn boundary_subscription(publisher: &zmq::Socket, topic: &str, skip_unsubscribe: bool) {
        let subscribed = [b"\x01".as_slice(), topic.as_bytes()].concat();
        let unsubscribed = [b"\x00".as_slice(), topic.as_bytes()].concat();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                match publisher.recv_bytes(zmq::DONTWAIT) {
                    Ok(frame) if skip_unsubscribe && frame == unsubscribed => continue,
                    Ok(frame) => {
                        assert_eq!(frame, subscribed);
                        break;
                    }
                    Err(zmq::Error::EAGAIN) => tokio::task::yield_now().await,
                    Err(error) => panic!("boundary subscription: {error}"),
                }
            }
        })
        .await
        .unwrap();
    }

    fn stored_block(hash: i64, tokens: &[u32], block_size: u32, medium: &str) -> serde_json::Value {
        serde_json::json!({
            "type": "BlockStored", "block_hashes": [hash], "parent_block_hash": null,
            "token_ids": tokens, "block_size": block_size, "medium": medium,
            "lora_id": null, "lora_name": null
        })
    }

    fn store_payload(
        hash: i64,
        tokens: &[u32],
        block_size: u32,
        medium: &str,
        rank: Option<u32>,
    ) -> Vec<u8> {
        rmp_serde::to_vec(&(1.0, [stored_block(hash, tokens, block_size, medium)], rank)).unwrap()
    }

    fn block_query(tokens: &[u32], block_size: u32, tiers: &[StorageTier]) -> MatchQuery {
        MatchQuery {
            group_idx: 0,
            local_hashes: crate::kv_index::local_block_hash_strings(tokens, block_size, None),
            tiers_of_interest: tiers.to_vec(),
        }
    }

    #[tokio::test]
    async fn concurrent_source_admission_preserves_topic_rank_and_cache_identity_boundaries() {
        use serde_json::json;

        for (topic_b, rank_b, model_b, same_publisher, accepted) in [
            ("kv/other", 0u32, "boundary-model", true, true),
            ("kv/boundary", 1, "boundary-model", true, true),
            ("kv/boundary", 0, "other-model", false, true),
            ("kv/boundary", 0, "other-model", true, false),
            ("kv/boundary", 0, "boundary-model", true, false),
            ("kv/boundary", 0, "boundary-model", false, false),
        ] {
            let context = zmq::Context::new();
            let publisher_a = context.socket(zmq::XPUB).unwrap();
            let publisher_b = context.socket(zmq::XPUB).unwrap();
            for publisher in [&publisher_a, &publisher_b] {
                publisher.bind("tcp://127.0.0.1:*").unwrap();
            }
            let endpoint_a = publisher_a.get_last_endpoint().unwrap().unwrap();
            let endpoint_b = publisher_b.get_last_endpoint().unwrap().unwrap();
            let sources = |endpoint: &str, topic: &str, rank: u32| {
                json!({rank.to_string(): {
                    "endpoint": endpoint, "topic": topic, "publisher": "zmq",
                    "encoding": "msgpack", "schema_version": 1, "enable_kv_cache_events": true
                }})
            };
            let info = |model, instance| json!({"model_id": model, "instance_id": instance, "kv_block_size": 8});
            let (worker_a, task_a) = source_boundary_worker_with_metadata(
                info("boundary-model", "shared-target"),
                sources(&endpoint_a, "kv/boundary", 0),
            )
            .await;
            let (worker_b, task_b) = source_boundary_worker_with_metadata(
                info(
                    model_b,
                    if topic_b == "kv/other" {
                        "other-topic-target"
                    } else {
                        "shared-target"
                    },
                ),
                sources(
                    if same_publisher {
                        &endpoint_a
                    } else {
                        &endpoint_b
                    },
                    topic_b,
                    rank_b,
                ),
            )
            .await;
            let (supervisor, _) = KvIndexSupervisor::new_http_full_attention(
                reqwest::Client::new(),
                HashMode::Sha256,
            );
            let (a, b) = tokio::join!(
                supervisor.on_worker_added(&worker_a),
                supervisor.on_worker_added(&worker_b)
            );
            if accepted {
                assert!(a.is_ok() && b.is_ok(), "{a:?} {b:?}");
                assert_eq!(supervisor.workers.lock().len(), 2);
                if same_publisher && rank_b == 1 {
                    let workers = supervisor.workers.lock();
                    let guard_a = &workers[&worker_a].live_sources[0].lease.strict_rank;
                    let guard_b = &workers[&worker_b].live_sources[0].lease.strict_rank;
                    assert!(Arc::ptr_eq(guard_a, guard_b));
                    assert!(*guard_a.lock());
                }
            } else {
                assert_ne!(a.is_ok(), b.is_ok(), "{a:?} {b:?}");
                assert_eq!(supervisor.workers.lock().len(), 1);
            }
            supervisor.shutdown();
            task_a.abort();
            task_b.abort();
        }
    }

    #[tokio::test]
    async fn source_group_rejects_duplicate_normalized_ranks_without_mutation() {
        use serde_json::json;

        let context = zmq::Context::new();
        let publisher = context.socket(zmq::XPUB).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let entry = json!({"endpoint": publisher.get_last_endpoint().unwrap().unwrap(),
            "publisher": "zmq", "encoding": "msgpack", "schema_version": 1,
            "enable_kv_cache_events": true});
        let (worker, task) = source_boundary_worker_with_metadata(
            json!({"model_id": "boundary-model", "instance_id": "duplicate-rank", "kv_block_size": 8}),
            json!({"0": entry, "00": entry}),
        ).await;
        let (supervisor, _) =
            KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
        assert!(supervisor
            .on_worker_added(&worker)
            .await
            .unwrap_err()
            .contains("duplicate normalized"));
        assert!(supervisor.workers.lock().is_empty());
        assert!(supervisor.indexers.is_empty());
        assert!(supervisor.identity_refs.lock().is_empty());
        assert!(matches!(
            publisher.recv_bytes(zmq::DONTWAIT),
            Err(zmq::Error::EAGAIN)
        ));
        task.abort();
    }

    #[tokio::test]
    async fn failed_readd_preserves_registration_then_accepted_readd_retires_it() {
        use axum::{routing::get, Json, Router};
        use serde_json::json;

        let context = zmq::Context::new();
        let publisher_a = context.socket(zmq::XPUB).unwrap();
        let publisher_b = context.socket(zmq::XPUB).unwrap();
        for publisher in [&publisher_a, &publisher_b] {
            publisher.set_xpub_verbose(true).unwrap();
            publisher.bind("tcp://127.0.0.1:*").unwrap();
        }
        let source = |endpoint: &str| {
            json!({"0": {"endpoint": endpoint, "publisher": "zmq",
            "topic": "kv/boundary", "encoding": "msgpack", "schema_version": 1,
            "enable_kv_cache_events": true}})
        };
        let endpoint_a = publisher_a.get_last_endpoint().unwrap().unwrap();
        let endpoint_b = publisher_b.get_last_endpoint().unwrap().unwrap();
        let sources_a = Arc::new(Mutex::new(source(&endpoint_a)));
        let mutable_sources = sources_a.clone();
        let app = Router::new()
            .route(
                "/get_server_info",
                get(|| async {
                    Json(json!({
                        "model_id": "boundary-model", "instance_id": "readd-a", "kv_block_size": 8
                    }))
                }),
            )
            .route(
                "/kv_event_sources",
                get(move || {
                    let value = mutable_sources.lock().clone();
                    async move { Json(value) }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker_a = format!("http://{}", listener.local_addr().unwrap());
        let task_a = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (worker_b, task_b) = source_boundary_worker_with_metadata(
            json!({"model_id": "boundary-model", "instance_id": "readd-b", "kv_block_size": 8}),
            source(&endpoint_b),
        )
        .await;
        let (supervisor, mut signals) =
            KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
        supervisor.on_worker_added(&worker_a).await.unwrap();
        supervisor.on_worker_added(&worker_b).await.unwrap();
        boundary_subscription(&publisher_a, "kv/boundary", true).await;
        let payload = store_payload(1, &[1; 8], 8, "GPU", Some(0));
        let query = block_query(&[1; 8], 8, &[StorageTier::Device]);
        publisher_a
            .send_multipart(
                [b"kv/boundary".as_slice(), &0i64.to_be_bytes(), &payload],
                0,
            )
            .unwrap();
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(5), signals.recv())
                .await
                .unwrap(),
            Some(IngestionSignal::Advance { last_seq: 0, .. })
        ));
        let provider = supervisor.match_provider_for_worker(&worker_a).unwrap();
        assert_eq!(provider.find_tiered_matches(&query).len(), 1);
        let original = supervisor.workers.lock()[&worker_a].active.clone();
        *sources_a.lock() = source(&endpoint_b);
        assert!(supervisor
            .on_worker_added(&worker_a)
            .await
            .unwrap_err()
            .contains("live KV streams"));
        assert!(original.load(Ordering::Acquire));
        assert!(Arc::ptr_eq(
            &original,
            &supervisor.workers.lock()[&worker_a].active
        ));
        assert!(supervisor.match_provider_for_worker(&worker_a).is_some());
        assert_eq!(provider.find_tiered_matches(&query).len(), 1);
        let valid_entry = source(&endpoint_a)["0"].clone();
        let mut bad_publisher = valid_entry.clone();
        bad_publisher["endpoint"] = json!("tcp://127.0.0.1:0");
        let mut bad_replay = valid_entry.clone();
        bad_replay["replay_endpoint"] = json!("tcp://127.0.0.1:0");
        for (case, invalid_sources) in [
            ("bad rank", json!({"not-a-rank": valid_entry})),
            ("bad publisher", json!({"0": bad_publisher})),
            ("bad replay", json!({"0": bad_replay})),
            (
                "mixed valid and malformed",
                json!({"0": valid_entry, "1": bad_publisher}),
            ),
        ] {
            *sources_a.lock() = invalid_sources;
            assert!(
                supervisor.on_worker_added(&worker_a).await.is_err(),
                "{case} must reject the complete re-add"
            );
            assert!(original.load(Ordering::Acquire), "{case}");
            assert!(
                Arc::ptr_eq(&original, &supervisor.workers.lock()[&worker_a].active),
                "{case}"
            );
            assert!(
                supervisor.match_provider_for_worker(&worker_a).is_some(),
                "{case}"
            );
            assert_eq!(provider.find_tiered_matches(&query).len(), 1, "{case}");
            assert_eq!(
                supervisor.identity_refs.lock().values().sum::<usize>(),
                2,
                "{case}"
            );
        }
        *sources_a.lock() = source(&endpoint_a);
        supervisor.on_worker_added(&worker_a).await.unwrap();
        assert!(!original.load(Ordering::Acquire));
        assert!(!Arc::ptr_eq(
            &original,
            &supervisor.workers.lock()[&worker_a].active
        ));
        assert!(provider.find_tiered_matches(&query).is_empty());
        boundary_subscription(&publisher_a, "kv/boundary", true).await;
        publisher_a
            .send_multipart(
                [b"kv/boundary".as_slice(), &0i64.to_be_bytes(), &payload],
                0,
            )
            .unwrap();
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(5), signals.recv())
                .await
                .unwrap(),
            Some(IngestionSignal::Advance { last_seq: 0, .. })
        ));
        assert_eq!(provider.find_tiered_matches(&query).len(), 1);
        assert_eq!(supervisor.identity_refs.lock().values().sum::<usize>(), 2);
        supervisor.shutdown();
        task_a.abort();
        task_b.abort();
    }

    #[tokio::test]
    async fn shared_publisher_rank_promotion_revokes_unranked_credit_before_registration() {
        use serde_json::json;

        let context = zmq::Context::new();
        let publisher = context.socket(zmq::XPUB).unwrap();
        publisher.set_xpub_verbose(true).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
        let source = |rank: u32| {
            json!({rank.to_string(): {
                "endpoint": endpoint, "publisher": "zmq", "topic": "kv/boundary",
                "encoding": "msgpack", "schema_version": 1, "enable_kv_cache_events": true
            }})
        };
        let info = |instance| {
            json!({
                "model_id": "boundary-model", "instance_id": instance, "kv_block_size": 8
            })
        };
        let (worker_a, task_a) =
            source_boundary_worker_with_metadata(info("promotion-a"), source(0)).await;
        let (worker_b, task_b) =
            source_boundary_worker_with_metadata(info("promotion-b"), source(1)).await;
        let (supervisor, mut signals) =
            KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
        let send = |sequence: i64, token: u32, rank: Option<u32>| {
            let payload = store_payload(sequence + 1, &[token; 8], 8, "GPU", rank);
            publisher
                .send_multipart(
                    [b"kv/boundary".as_slice(), &sequence.to_be_bytes(), &payload],
                    0,
                )
                .unwrap();
        };
        let query = |token| block_query(&[token; 8], 8, &[StorageTier::Device]);
        supervisor.on_worker_added(&worker_a).await.unwrap();
        boundary_subscription(&publisher, "kv/boundary", true).await;
        send(0, 1, None);
        assert!(
            matches!(tokio::time::timeout(Duration::from_secs(5), signals.recv()).await.unwrap(),
            Some(IngestionSignal::Advance { source, last_seq: 0 }) if source.as_ref() == "promotion-a")
        );
        let provider = supervisor.match_provider_for_worker(&worker_a).unwrap();
        let positive = provider.find_tiered_matches(&query(1));
        assert_eq!(positive.len(), 1);
        assert_eq!(positive[0].target.instance_id.as_ref(), "promotion-a");
        supervisor.on_worker_added(&worker_b).await.unwrap();
        assert!(provider.find_tiered_matches(&query(1)).is_empty());
        boundary_subscription(&publisher, "kv/boundary", true).await;

        send(1, 2, None);
        let mut rejected_sources = HashSet::new();
        for _ in 0..2 {
            let signal = tokio::time::timeout(Duration::from_secs(5), signals.recv())
                .await
                .unwrap();
            match signal {
                Some(IngestionSignal::Gap {
                    source, to_seq: 1, ..
                }) => {
                    rejected_sources.insert(source);
                }
                other => panic!("missing shared rank must reject: {other:?}"),
            }
        }
        assert_eq!(rejected_sources.len(), 2);
        assert!(provider.find_tiered_matches(&query(2)).is_empty());
        for (sequence, token, rank) in [(1, 3, 0), (2, 4, 1), (3, 5, 0)] {
            send(sequence, token, Some(rank));
            let mut advanced_sources = HashSet::new();
            tokio::time::timeout(Duration::from_secs(5), async {
                while advanced_sources.len() != 2 {
                    if let Some(IngestionSignal::Advance { source, last_seq }) =
                        signals.recv().await
                    {
                        assert_eq!(last_seq, sequence);
                        advanced_sources.insert(source);
                    }
                }
            })
            .await
            .unwrap();
            let hits = provider.find_tiered_matches(&query(token));
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].target.dp_rank, rank);
            assert_eq!(
                hits[0].target.instance_id.as_ref(),
                if rank == 0 {
                    "promotion-a"
                } else {
                    "promotion-b"
                }
            );
            assert_eq!(hits[0].segments, [(0, 1)]);
            // Foreign traffic must preserve already-qualified owned observations.
            assert_eq!(provider.find_tiered_matches(&query(3)).len(), 1);
            if sequence >= 2 {
                assert_eq!(provider.find_tiered_matches(&query(4)).len(), 1);
            }
        }
        assert!(provider.find_tiered_matches(&query(1)).is_empty());
        assert!(provider.find_tiered_matches(&query(2)).is_empty());
        supervisor.shutdown();
        task_a.abort();
        task_b.abort();
    }

    #[tokio::test]
    async fn live_source_and_target_conflicts_reject_before_subscription_without_replay() {
        use serde_json::json;

        for static_a in [false, true] {
            for same_publisher in [true, false] {
                let context = zmq::Context::new();
                let publisher_a = context.socket(zmq::XPUB).unwrap();
                let publisher_b = context.socket(zmq::XPUB).unwrap();
                for publisher in [&publisher_a, &publisher_b] {
                    publisher.set_xpub_verbose(true).unwrap();
                    publisher.bind("tcp://127.0.0.1:*").unwrap();
                }
                let endpoint_a = publisher_a.get_last_endpoint().unwrap().unwrap();
                let endpoint_b = publisher_b.get_last_endpoint().unwrap().unwrap();
                let source = |endpoint: &str| {
                    json!({"0": {
                        "endpoint": endpoint, "topic": "kv/boundary", "publisher": "zmq",
                        "encoding": "msgpack", "schema_version": 1, "enable_kv_cache_events": true
                    }})
                };
                let info = |instance: &str| {
                    json!({
                        "model_id": "boundary-model", "instance_id": instance, "kv_block_size": 8
                    })
                };
                let (worker_a, task_a) =
                    source_boundary_worker_with_metadata(info("boundary-a"), source(&endpoint_a))
                        .await;
                let instance_a = if static_a {
                    worker_a.as_str()
                } else {
                    "boundary-a"
                };
                let (worker_b, task_b) = source_boundary_worker_with_metadata(
                    info(if same_publisher {
                        "boundary-b"
                    } else {
                        instance_a
                    }),
                    source(if same_publisher {
                        &endpoint_a
                    } else {
                        &endpoint_b
                    }),
                )
                .await;
                let (mut supervisor, mut signals) =
                    KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
                supervisor.full_attention_http = true;
                if static_a {
                    supervisor.static_sources = serde_json::from_value(json!({&worker_a: {
                        "endpoints": [endpoint_a], "topic": "kv/boundary",
                        "block_size": 8, "model": "boundary-model"
                    }}))
                    .unwrap();
                }
                supervisor.on_worker_added(&worker_a).await.unwrap();
                boundary_subscription(&publisher_a, "kv/boundary", true).await;
                let payload = store_payload(1, &[1; 8], 8, "GPU", Some(0));
                publisher_a
                    .send_multipart(
                        [b"kv/boundary".as_slice(), &0i64.to_be_bytes(), &payload],
                        0,
                    )
                    .unwrap();
                assert!(matches!(
                    tokio::time::timeout(Duration::from_secs(5), signals.recv())
                        .await
                        .unwrap(),
                    Some(IngestionSignal::Advance { last_seq: 0, .. })
                ));
                let query = block_query(&[1; 8], 8, &[StorageTier::Device]);
                let provider = supervisor.match_provider_for_worker(&worker_a).unwrap();
                assert_eq!(provider.find_tiered_matches(&query).len(), 1);
                let admission = supervisor.on_worker_added(&worker_b).await;
                let error =
                    admission.expect_err("conflicting worker must be rejected before publication");
                assert!(
                    error.contains(if same_publisher {
                        "live KV streams must be independent"
                    } else {
                        "KV index targets must be independent"
                    }),
                    "{error}"
                );
                assert!(supervisor.match_provider_for_worker(&worker_b).is_none());
                assert_eq!(supervisor.workers.lock().len(), 1);
                assert!(matches!(
                    publisher_a.recv_bytes(zmq::DONTWAIT),
                    Err(zmq::Error::EAGAIN)
                ));
                assert!(matches!(
                    publisher_b.recv_bytes(zmq::DONTWAIT),
                    Err(zmq::Error::EAGAIN)
                ));
                let retained = provider.find_tiered_matches(&query);
                assert_eq!(retained.len(), 1);
                assert_eq!(retained[0].target.instance_id.as_ref(), instance_a);
                assert_eq!(retained[0].segments, [(0, 1)]);
                supervisor.shutdown();
                task_a.abort();
                task_b.abort();
            }
        }
    }

    #[tokio::test]
    async fn discovered_sources_reject_shared_replay_before_wrong_owner_is_published() {
        let context = zmq::Context::new();
        let a = context.socket(zmq::XPUB).unwrap();
        let b = context.socket(zmq::XPUB).unwrap();
        let replay = context.socket(zmq::ROUTER).unwrap();
        for socket in [&a, &b, &replay] {
            socket.bind("tcp://127.0.0.1:*").unwrap();
        }
        let replay_endpoint = replay.get_last_endpoint().unwrap().unwrap();
        let (worker_a, http_a) = source_boundary_worker(
            &a.get_last_endpoint().unwrap().unwrap(),
            &replay_endpoint,
            "boundary-a",
        )
        .await;
        let (worker_b, http_b) = source_boundary_worker(
            &b.get_last_endpoint().unwrap().unwrap(),
            &replay_endpoint,
            "boundary-b",
        )
        .await;
        let (supervisor, mut signals) =
            KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
        supervisor.on_worker_added(&worker_a).await.unwrap();
        boundary_subscription(&a, "kv/boundary", true).await;
        a.send_multipart(
            [
                b"kv/boundary".as_slice(),
                &0i64.to_be_bytes(),
                &store_payload(1, &[1; 8], 8, "GPU", Some(0)),
            ],
            0,
        )
        .unwrap();
        assert!(
            matches!(tokio::time::timeout(Duration::from_secs(5), signals.recv()).await.unwrap(),
            Some(IngestionSignal::Advance { source, last_seq: 0 }) if source.as_ref() == "boundary-a")
        );
        let query = |token| block_query(&[token; 8], 8, &[StorageTier::Device]);
        assert_eq!(
            supervisor
                .match_provider_for_worker(&worker_a)
                .unwrap()
                .find_tiered_matches(&query(1))
                .len(),
            1
        );
        let error = supervisor.on_worker_added(&worker_b).await.unwrap_err();
        assert!(
            error.contains("replay endpoints must be independent"),
            "{error}"
        );
        assert!(supervisor.match_provider_for_worker(&worker_b).is_none());
        assert!(matches!(
            b.recv_bytes(zmq::DONTWAIT),
            Err(zmq::Error::EAGAIN)
        ));
        let retained = supervisor
            .match_provider_for_worker(&worker_a)
            .unwrap()
            .find_tiered_matches(&query(1));
        assert_eq!(retained.len(), 1);
        assert_eq!(retained[0].target.instance_id.as_ref(), "boundary-a");
        assert_eq!(retained[0].segments, [(0, 1)]);
        supervisor.shutdown();
        http_a.abort();
        http_b.abort();
    }

    #[tokio::test]
    async fn discovered_source_bindings_are_atomic_and_released_with_the_worker() {
        let context = zmq::Context::new();
        let publisher_a = context.socket(zmq::XPUB).unwrap();
        let publisher_b = context.socket(zmq::XPUB).unwrap();
        let replay = context.socket(zmq::ROUTER).unwrap();
        for socket in [&publisher_a, &publisher_b, &replay] {
            socket.bind("tcp://127.0.0.1:*").unwrap();
        }
        let replay_endpoint = replay.get_last_endpoint().unwrap().unwrap();
        let (worker_a, http_a) = source_boundary_worker(
            &publisher_a.get_last_endpoint().unwrap().unwrap(),
            &replay_endpoint,
            "binding-a",
        )
        .await;
        let (worker_b, http_b) = source_boundary_worker(
            &publisher_b.get_last_endpoint().unwrap().unwrap(),
            &replay_endpoint,
            "binding-b",
        )
        .await;
        let worker_a = format!("{worker_a}/");
        let (supervisor, _signals) =
            KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
        let (a, b) = tokio::join!(
            supervisor.on_worker_added(&worker_a),
            supervisor.on_worker_added(&worker_b)
        );
        assert_ne!(a.is_ok(), b.is_ok());
        let (accepted, rejected) = if a.is_ok() {
            (&worker_a, &worker_b)
        } else {
            (&worker_b, &worker_a)
        };
        assert_eq!(supervisor.workers.lock().len(), 1);
        assert_eq!(
            supervisor
                .identity_refs
                .lock()
                .values()
                .copied()
                .sum::<usize>(),
            1
        );
        assert!(supervisor.match_provider_for_worker(accepted).is_some());
        assert!(supervisor.match_provider_for_worker(rejected).is_none());
        supervisor.on_worker_removed(accepted);
        assert!(supervisor.workers.lock().is_empty());
        assert!(supervisor.identity_refs.lock().is_empty());
        supervisor.on_worker_added(rejected).await.unwrap();
        let alias = if rejected.ends_with('/') {
            rejected.trim_end_matches('/').to_owned()
        } else {
            format!("{rejected}/")
        };
        let error = supervisor.on_worker_added(&alias).await.unwrap_err();
        assert!(error.contains("worker origins must be independent"));
        assert!(supervisor.workers.lock().contains_key(rejected));
        assert!(!supervisor.workers.lock().contains_key(&alias));
        assert_eq!(
            supervisor
                .identity_refs
                .lock()
                .values()
                .copied()
                .sum::<usize>(),
            1
        );
        supervisor.on_worker_removed(rejected);
        let (first, second) = tokio::join!(
            supervisor.on_worker_added(rejected),
            supervisor.on_worker_added(rejected)
        );
        assert_ne!(first.is_ok(), second.is_ok());
        assert_eq!(supervisor.workers.lock().len(), 1);
        assert_eq!(
            supervisor
                .identity_refs
                .lock()
                .values()
                .copied()
                .sum::<usize>(),
            1
        );
        supervisor.shutdown();
        assert!(supervisor.workers.lock().is_empty());
        assert!(supervisor.identity_refs.lock().is_empty());
        http_a.abort();
        http_b.abort();
    }

    #[tokio::test]
    async fn dropping_supervisor_retires_inflight_replay_without_clearing_other_sources() {
        for explicit_shutdown in [false, true] {
            let context = Arc::new(zmq::Context::new());
            let publisher = context.socket(zmq::XPUB).unwrap();
            let replay = context.socket(zmq::ROUTER).unwrap();
            for socket in [&publisher, &replay] {
                socket.bind("tcp://127.0.0.1:*").unwrap();
            }
            let (worker, server) = source_boundary_worker(
                &publisher.get_last_endpoint().unwrap().unwrap(),
                &replay.get_last_endpoint().unwrap().unwrap(),
                "retiring-source",
            )
            .await;
            let (supervisor, mut signals) = KvIndexSupervisor::new_http_full_attention(
                reqwest::Client::new(),
                HashMode::Sha256,
            );
            supervisor.on_worker_added(&worker).await.unwrap();
            boundary_subscription(&publisher, "kv/boundary", true).await;
            let (indexer, active) = {
                let workers = supervisor.workers.lock();
                let entry = &workers[&worker];
                (entry.indexer.clone(), entry.active.clone())
            };
            let provider = supervisor.match_provider_for_worker(&worker).unwrap();
            let (neighbor_tx, mut neighbor_signals) = mpsc::channel(8);
            let neighbor = crate::kv_index::subscriber::spawn(
                EventSource {
                    source: Arc::from("independent-source"),
                    dp_rank: 4,
                    pub_endpoint: publisher.get_last_endpoint().unwrap().unwrap(),
                    replay_endpoint: None,
                    topic: "kv/independent".into(),
                    hwm: None,
                },
                indexer,
                Some(neighbor_tx),
                context.clone(),
            );
            boundary_subscription(&publisher, "kv/independent", false).await;
            let payload =
                |hash, token, rank| store_payload(hash, &[token; 8], 8, "GPU", Some(rank));
            let query = |token| block_query(&[token; 8], 8, &[StorageTier::Device]);
            for (topic, body) in [
                ("kv/boundary", payload(1, 1, 0)),
                ("kv/independent", payload(2, 2, 4)),
            ] {
                publisher
                    .send_multipart([topic.as_bytes(), &0i64.to_be_bytes(), &body], 0)
                    .unwrap();
            }
            for receiver in [&mut signals, &mut neighbor_signals] {
                assert!(matches!(
                    tokio::time::timeout(Duration::from_secs(5), receiver.recv())
                        .await
                        .unwrap(),
                    Some(IngestionSignal::Advance { last_seq: 0, .. })
                ));
            }
            assert_eq!(provider.find_tiered_matches(&query(1)).len(), 1);
            assert_eq!(provider.find_tiered_matches(&query(2)).len(), 1);
            publisher
                .send_multipart(
                    [
                        b"kv/boundary".as_slice(),
                        &2i64.to_be_bytes(),
                        &payload(3, 3, 0),
                    ],
                    0,
                )
                .unwrap();
            let request = tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    match replay.recv_multipart(zmq::DONTWAIT) {
                        Ok(request) => break request,
                        Err(zmq::Error::EAGAIN) => tokio::task::yield_now().await,
                        Err(error) => panic!("retiring replay request: {error}"),
                    }
                }
            })
            .await
            .unwrap();
            assert_eq!(request.last().unwrap(), &1i64.to_be_bytes());
            if explicit_shutdown {
                supervisor.shutdown();
                supervisor.shutdown();
                assert!(supervisor.workers.lock().is_empty());
                assert!(supervisor.identity_refs.lock().is_empty());
            }
            drop(supervisor);
            assert!(!active.load(Ordering::Acquire));
            for (seq, body) in [
                (1i64, payload(4, 4, 0)),
                (2, payload(3, 3, 0)),
                (-1, vec![]),
            ] {
                replay
                    .send_multipart(
                        [
                            request[0].clone(),
                            vec![],
                            if seq == -1 {
                                vec![]
                            } else {
                                b"kv/boundary".to_vec()
                            },
                            seq.to_be_bytes().to_vec(),
                            body,
                        ],
                        0,
                    )
                    .unwrap();
            }
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    match publisher.recv_bytes(zmq::DONTWAIT) {
                        Ok(frame) => {
                            assert_eq!(frame, b"\x00kv/boundary");
                            break;
                        }
                        Err(zmq::Error::EAGAIN) => tokio::task::yield_now().await,
                        Err(error) => panic!("retired subscription: {error}"),
                    }
                }
            })
            .await
            .unwrap();
            for token in [1, 3, 4] {
                assert!(provider.find_tiered_matches(&query(token)).is_empty());
            }
            let retained = provider.find_tiered_matches(&query(2));
            assert_eq!(retained.len(), 1);
            assert_eq!(
                retained[0].target.instance_id.as_ref(),
                "independent-source"
            );
            assert_eq!(retained[0].target.dp_rank, 4);
            assert_eq!(retained[0].segments, [(0, 1)]);
            neighbor.shutdown();
            server.abort();
        }
    }

    #[tokio::test]
    async fn generic_static_sources_preserve_worker_base_paths_in_full_attention_router() {
        let context = zmq::Context::new();
        let publisher = context.socket(zmq::XPUB).unwrap();
        let replay = context.socket(zmq::ROUTER).unwrap();
        for socket in [&publisher, &replay] {
            socket.bind("tcp://127.0.0.1:*").unwrap();
        }
        let (worker, http_task) = source_boundary_worker(
            &publisher.get_last_endpoint().unwrap().unwrap(),
            &replay.get_last_endpoint().unwrap().unwrap(),
            "prefixed-source",
        )
        .await;
        let prefixed = format!("{worker}/proxy");
        let (generic, _) =
            KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
        let generic = generic.with_static_sources(HashMap::from([(
            prefixed.clone(),
            StaticKvEventSources {
                endpoints: vec![publisher.get_last_endpoint().unwrap().unwrap()],
                replay_endpoints: Some(vec![replay.get_last_endpoint().unwrap().unwrap()]),
                block_size: 8,
                model: "prefixed-model".into(),
                topic: "kv/boundary".into(),
            },
        )]));
        let result = generic.on_worker_added(&prefixed).await;
        let registered = generic.match_provider_for_worker(&prefixed).is_some();
        let original_absent = generic.match_provider_for_worker(&worker).is_none();
        assert!(generic.workers.lock()[&prefixed].worker_origin.is_none());
        generic.shutdown();
        let (full_attention, _) =
            KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
        let full_attention_result = full_attention.on_worker_added(&prefixed).await;
        let unqualified = full_attention
            .match_provider_for_worker(&prefixed)
            .is_none();
        full_attention.shutdown();
        http_task.abort();
        let info = result.unwrap();
        assert_eq!(info.instance_id, prefixed);
        assert_eq!(info.ranks, [0]);
        assert!(registered);
        assert!(original_absent);
        assert!(full_attention_result
            .unwrap_err()
            .contains("plain HTTP(S) origins"));
        assert!(unqualified);
    }

    #[tokio::test]
    async fn full_attention_source_block_size_mismatch_revokes_only_affected_source() {
        for (expected, actual) in [(16, 8), (8, 16)] {
            check_full_attention_source_block_size(expected, actual).await;
        }
    }

    async fn check_full_attention_source_block_size(expected: u32, actual: u32) {
        use serde_json::json;

        let context = zmq::Context::new();
        let publisher = context.socket(zmq::XPUB).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
        let (supervisor, mut signals) =
            KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
        let rank = 3;
        let sources = json!({(rank.to_string()): {
            "endpoint": endpoint, "topic": "kv/dense", "publisher": "zmq",
            "encoding": "msgpack", "schema_version": 1, "enable_kv_cache_events": true
        }});
        let (worker, http_task) = source_boundary_worker_with_metadata(
            json!({"model_id": "dense-model", "instance_id": "dense-boot", "kv_block_size": expected}), sources
        ).await;
        let info = supervisor.on_worker_added(&worker).await.unwrap();
        assert_eq!(info.cache_key.as_ref().unwrap().block_size, expected);
        assert_eq!(info.ranks, [rank]);
        boundary_subscription(&publisher, "kv/dense", false).await;
        let indexer = supervisor
            .workers
            .lock()
            .get(&worker)
            .unwrap()
            .indexer
            .clone();
        let old_tokens: Vec<u32> = (0..expected).collect();
        let new_tokens: Vec<u32> = (100..100 + actual).collect();
        let query = |tokens: &[u32]| {
            block_query(
                tokens,
                expected,
                &[StorageTier::Device, StorageTier::HostPinned],
            )
        };
        let mut neighbors = Vec::new();
        for (source, neighbor_rank, topic) in [
            ("other-source", rank, "kv/other-source"),
            (info.instance_id.as_str(), rank + 1, "kv/other-rank"),
        ] {
            let (tx, mut rx) = mpsc::channel(8);
            neighbors.push(crate::kv_index::subscriber::spawn(
                EventSource {
                    source: Arc::from(source),
                    dp_rank: neighbor_rank,
                    pub_endpoint: publisher.get_last_endpoint().unwrap().unwrap(),
                    replay_endpoint: None,
                    topic: topic.into(),
                    hwm: None,
                },
                indexer.clone(),
                Some(tx),
                Arc::new(zmq::Context::new()),
            ));
            boundary_subscription(&publisher, topic, false).await;
            let payload = store_payload(70, &old_tokens, expected, "GPU", Some(neighbor_rank));
            assert!(crate::kv_index::decode_batch(&payload).is_ok());
            publisher
                .send_multipart([topic.as_bytes(), &0i64.to_be_bytes(), &payload], 0)
                .unwrap();
            assert!(matches!(
                tokio::time::timeout(Duration::from_secs(5), rx.recv())
                    .await
                    .unwrap(),
                Some(IngestionSignal::Advance { last_seq: 0, .. })
            ));
        }
        let source_hits = |tokens: &[u32]| {
            indexer
                .find_matches(&query(tokens))
                .into_iter()
                .filter(|hit| {
                    hit.target.instance_id.as_ref() == info.instance_id
                        && hit.target.dp_rank == rank
                })
                .collect::<Vec<_>>()
        };
        for (seq, block_size, tokens) in [(0i64, expected, &old_tokens), (1, actual, &new_tokens)] {
            let payload = store_payload(seq + 7, tokens, block_size, "CPU", Some(rank));
            publisher
                .send_multipart([b"kv/dense".as_slice(), &seq.to_be_bytes(), &payload], 0)
                .unwrap();
            if seq == 0 {
                assert!(matches!(
                    tokio::time::timeout(Duration::from_secs(5), signals.recv())
                        .await
                        .unwrap(),
                    Some(IngestionSignal::Advance { last_seq: 0, .. })
                ));
                let hits = source_hits(&old_tokens);
                assert_eq!(hits.len(), 1);
                assert_eq!(hits[0].segments, [(0, 1)]);
                assert_eq!(hits[0].target.dp_rank, rank);
            } else {
                tokio::time::timeout(Duration::from_secs(5), async {
                    loop {
                        if source_hits(&old_tokens).is_empty() {
                            assert!(indexer.find_matches(&query(&new_tokens)).is_empty());
                            let remaining = indexer.find_matches(&query(&old_tokens));
                            assert_eq!(remaining.len(), 2, "mismatch invalidated another source/rank");
                            assert!(remaining.iter().all(|hit| hit.segments == [(0, 1)]));
                            break;
                        }
                        if let Ok(signal) = signals.try_recv() {
                            assert!(!matches!(signal, IngestionSignal::Advance { last_seq: 1, .. }),
                                "accepted mismatched Dense batch: source={expected}, event={actual}, old={:?}, mismatched={:?}",
                                indexer.find_matches(&query(&old_tokens)),
                                indexer.find_matches(&query(&new_tokens)));
                        }
                        tokio::task::yield_now().await;
                    }
                }).await.expect("mismatch must revoke the previously indexed source");
            }
        }
        supervisor.shutdown();
        drop(neighbors);
        http_task.abort();
    }

    #[tokio::test]
    async fn http_and_static_sources_ingest_nonempty_topics_through_live_zmq() {
        use serde_json::json;

        for (use_static, block_size) in [(false, 8u32), (true, 32)] {
            let context = zmq::Context::new();
            let publisher = context.socket(zmq::XPUB).unwrap();
            publisher.bind("tcp://127.0.0.1:*").unwrap();
            let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
            let rank = if use_static { 0u32 } else { 3 };
            let (mut supervisor, mut signals) = KvIndexSupervisor::new_http_full_attention(
                reqwest::Client::new(),
                HashMode::Sha256,
            );
            let (worker, http_task) = if use_static {
                let worker = "http://127.0.0.1:1".to_string();
                let value = json!({
                    &worker: {
                        "endpoints": [endpoint], "block_size": block_size,
                        "model": "served-alias", "topic": "kv/served-alias"
                    }
                });
                supervisor.static_sources = serde_json::from_value(value).unwrap();
                (worker, None)
            } else {
                let sources = json!({(rank.to_string()): {
                    "endpoint": endpoint, "topic": "kv/served-alias", "hwm": 128,
                    "publisher": "zmq", "encoding": "msgpack", "schema_version": 1,
                    "enable_kv_cache_events": true
                }});
                let (worker, task) = source_boundary_worker_with_metadata(
                    json!({
                        "model_id": "served-alias", "instance_id": "observed-boot", "kv_block_size": block_size
                    }),
                    sources,
                )
                .await;
                (worker, Some(task))
            };
            let info = supervisor.on_worker_added(&worker).await.unwrap();
            assert_eq!(info.ranks, [rank]);
            assert_eq!(
                info.cache_key.as_ref().unwrap().model.as_ref(),
                "served-alias"
            );
            boundary_subscription(&publisher, "kv/served-alias", false).await;
            let tokens: Vec<u32> = (0..block_size).collect();
            let payload = store_payload(
                7,
                &tokens,
                block_size,
                if use_static { "GPU" } else { "CPU" },
                Some(rank),
            );
            assert!(crate::kv_index::decode_batch(&payload).is_ok());
            publisher
                .send_multipart(
                    [b"kv/served-alias".as_slice(), &0i64.to_be_bytes(), &payload],
                    0,
                )
                .unwrap();
            assert!(matches!(
                tokio::time::timeout(Duration::from_secs(5), signals.recv())
                    .await
                    .unwrap(),
                Some(IngestionSignal::Advance { last_seq: 0, .. })
            ));
            let tier = if use_static {
                StorageTier::Device
            } else {
                StorageTier::HostPinned
            };
            let indexer = supervisor
                .workers
                .lock()
                .get(&worker)
                .unwrap()
                .indexer
                .clone();
            let hits = indexer.find_matches(&block_query(&tokens, block_size, &[tier]));
            assert_eq!(hits.len(), 1, "use_static={use_static}");
            assert_eq!(hits[0].target.instance_id.as_ref(), info.instance_id);
            assert_eq!(hits[0].target.dp_rank, rank);
            assert_eq!(hits[0].tier, tier);
            assert_eq!(hits[0].segments, [(0, 1)]);
            supervisor.shutdown();
            if let Some(task) = http_task {
                task.abort();
            }
        }
    }

    #[test]
    fn static_pd_discovery_builds_one_source_per_rank() {
        let config = crate::config::types::PdKvEventSources {
            endpoints: vec!["tcp://10.0.0.5:5557".into(), "tcp://10.0.0.5:5559".into()],
            replay_endpoints: Some(vec!["tcp://10.0.0.5:5558".into()]),
            block_size: 32,
            model: Some("m".into()),
        };
        let discovery = static_pd_discovery("http://10.0.0.5:8000", &config);

        let server_info = discovery.server_info.unwrap();
        assert_eq!(server_info.model_id.as_deref(), Some("m"));
        assert_eq!(discovered_block_size(&server_info), Some(32));
        assert_eq!(discovery.model_id, Ok(Some("m".into())));

        let sources = discovery.sources.unwrap();
        assert_eq!(sources.len(), 2);
        for rank in ["0", "1"] {
            assert!(
                events_enabled(&sources[rank]),
                "rank {rank} should be enabled"
            );
        }
        assert_eq!(sources["0"].endpoint, "tcp://10.0.0.5:5557");
        assert_eq!(
            sources["0"].replay_endpoint.as_deref(),
            Some("tcp://10.0.0.5:5558")
        );
        // Replay list shorter than endpoints: rank 1 has no replay endpoint.
        assert_eq!(sources["1"].replay_endpoint, None);
        assert_eq!(sources["1"].endpoint, "tcp://10.0.0.5:5559");
    }

    #[test]
    fn static_pd_discovery_falls_back_to_worker_url_as_model() {
        let config = crate::config::types::PdKvEventSources {
            endpoints: vec!["tcp://127.0.0.1:5557".into()],
            replay_endpoints: None,
            block_size: 16,
            model: None,
        };
        let discovery = static_pd_discovery("http://127.0.0.1:8000", &config);
        assert_eq!(discovery.model_id, Ok(Some("http://127.0.0.1:8000".into())));
        // A blank model also falls back — an empty identity would prevent
        // cache-key creation downstream.
        let config = crate::config::types::PdKvEventSources {
            model: Some("  ".into()),
            ..config
        };
        let discovery = static_pd_discovery("http://127.0.0.1:8000", &config);
        assert_eq!(discovery.model_id, Ok(Some("http://127.0.0.1:8000".into())));
    }

    /// A worker with static sources must become indexable without any HTTP
    /// control plane — the discovery client would fail every request.
    #[tokio::test]
    async fn static_sources_skip_http_discovery() {
        let publisher_context = zmq::Context::new();
        let publisher = publisher_context.socket(zmq::PUB).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = publisher.get_last_endpoint().unwrap().unwrap();

        let worker_url = "http://127.0.0.1:1"; // nothing listens here
        let static_sources = std::iter::once((
            worker_url.to_string(),
            crate::config::types::PdKvEventSources {
                endpoints: vec![endpoint],
                replay_endpoints: None,
                block_size: 16,
                model: None,
            },
        ))
        .collect();
        let (supervisor, _signals) =
            KvIndexSupervisor::new_pd(reqwest::Client::new(), HashMode::Sha256, static_sources);

        let info = supervisor.on_worker_added(worker_url).await.unwrap();
        assert_eq!(info.block_size, Some(16));
        assert_eq!(info.ranks, vec![0]);
        let key = info.cache_key.expect("static sources must create a key");
        assert_eq!(key.model.as_ref(), worker_url);
        assert!(supervisor.match_provider_for_worker(worker_url).is_some());
        supervisor.on_worker_removed(worker_url);
    }

    #[tokio::test]
    async fn pd_static_full_attention_rejects_entire_bad_batch_with_source_block_size() {
        use serde_json::json;

        let context = zmq::Context::new();
        let publisher = context.socket(zmq::XPUB).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
        let worker = "http://127.0.0.1:1";
        let (supervisor, mut signals) = KvIndexSupervisor::new_pd(
            reqwest::Client::new(),
            HashMode::Sha256,
            HashMap::from([(
                worker.to_owned(),
                crate::config::types::PdKvEventSources {
                    endpoints: vec![endpoint],
                    replay_endpoints: None,
                    block_size: 8,
                    model: Some("pd-model".into()),
                },
            )]),
        );
        let info = supervisor.on_worker_added(worker).await.unwrap();
        assert_eq!(info.cache_key.unwrap().block_size, 8);
        boundary_subscription(&publisher, "", false).await;
        let stored = |hash, start, size| {
            stored_block(
                hash,
                &(start..start + size).collect::<Vec<_>>(),
                size,
                "GPU",
            )
        };
        let batch = rmp_serde::to_vec(&(
            1.0,
            [
                json!({"type": "BlockRemoved", "block_hashes": [9], "medium": "GPU"}),
                json!({"type": "AllBlocksCleared"}),
                stored(1, 20, 4),
                stored(2, 100, 8),
            ],
            0u32,
        ))
        .unwrap();
        publisher
            .send_multipart([b"".as_slice(), &0i64.to_be_bytes(), &batch], 0)
            .unwrap();
        // A subsequent valid frame is a deterministic processing barrier;
        // a rejected earlier batch must not become visible after resume.
        let marker = rmp_serde::to_vec(&(2.0, [stored(3, 200, 8)], 0u32)).unwrap();
        publisher
            .send_multipart([b"".as_slice(), &1i64.to_be_bytes(), &marker], 0)
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if matches!(
                    signals.recv().await,
                    Some(IngestionSignal::Advance { last_seq: 1, .. })
                ) {
                    break;
                }
            }
        })
        .await
        .unwrap();
        let index = supervisor.workers.lock()[worker].indexer.clone();
        let query = |start| {
            block_query(
                &(start..start + 8).collect::<Vec<_>>(),
                8,
                &[StorageTier::Device],
            )
        };
        assert!(
            index.find_matches(&query(100)).is_empty(),
            "bad block-size batch leaked its valid Store"
        );
        assert_eq!(index.find_matches(&query(200)).len(), 1);
        supervisor.shutdown();
    }
}
