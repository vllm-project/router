//! Worker discovery: query /get_server_info + /kv_event_sources, spawn one
//! subscriber per (worker, rank). One indexer per cache-identity core.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use dashmap::DashMap;
use parking_lot::Mutex;
use serde::Deserialize;
use tokio::sync::mpsc;
use tracing::{info, warn};

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

/// Operator-supplied immutable source facts for independently addressed workers.
/// This is a deployment contract, not remote identity attestation.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct StaticKvEventSources {
    endpoints: Vec<String>,
    replay_endpoints: Option<Vec<String>>,
    #[serde(default)]
    topic: String,
    #[serde(default)]
    hwm: Option<i32>,
    block_size: u32,
    model: String,
}

#[derive(Debug)]
struct StaticSingleSource {
    endpoint: String,
    replay_endpoint: Option<String>,
    topic: String,
    hwm: Option<i32>,
    block_size: u32,
    model: String,
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

struct WorkerDiscovery {
    server_info: Result<ServerInfo, String>,
    sources: Result<KvEventSourcesResponse, String>,
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
    strict_rank: Arc<Mutex<bool>>,
}

/// Orchestrates discovery + subscriber lifecycle for the router's workers.
/// One indexer per `CacheKey`, shared by every `(worker, rank)` of that
/// identity.
pub struct KvIndexSupervisor {
    client: reqwest::Client,
    hash_mode: HashMode,
    dense_http: bool,
    raw_static_sources: Option<String>,
    static_sources: OnceLock<Result<HashMap<String, StaticSingleSource>, String>>,
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
            hash_mode,
            dense_http: false,
            raw_static_sources: None,
            static_sources: OnceLock::new(),
            ctx: Arc::new(zmq::Context::new()),
            indexers: DashMap::new(),
            identity_refs: Mutex::new(HashMap::new()),
            signal_tx,
            workers: Mutex::new(HashMap::new()),
            first_block_size: Mutex::new(None),
        };
        (supervisor, signal_rx)
    }

    pub(crate) fn new_http_dense(
        client: reqwest::Client,
        hash_mode: HashMode,
        worker_urls: &[String],
        raw_static_sources: Option<String>,
    ) -> Result<(Self, mpsc::Receiver<IngestionSignal>), String> {
        let (mut supervisor, signals) = Self::new(client, hash_mode);
        supervisor.dense_http = true;
        supervisor.raw_static_sources = raw_static_sources;
        if worker_urls
            .iter()
            .any(|url| !crate::backend::is_grpc_url(url))
        {
            supervisor.http_static_sources()?;
        }
        Ok((supervisor, signals))
    }

    fn http_static_sources(&self) -> Result<&HashMap<String, StaticSingleSource>, String> {
        self.static_sources
            .get_or_init(|| {
                self.raw_static_sources
                    .as_deref()
                    .map(parse_static_sources)
                    .unwrap_or_else(|| Ok(HashMap::new()))
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    /// Validate the immutable HTTP override before health checks or DP expansion.
    pub(crate) fn validate_http_worker(
        &self,
        worker_url: &str,
        dp_size: usize,
    ) -> Result<(), String> {
        if self.dense_http && !crate::backend::is_grpc_url(worker_url) {
            if self.raw_static_sources.is_some() && dp_size != 1 {
                return Err(
                    "static HTTP KV sources require independently addressed DP1 workers".into(),
                );
            }
            self.http_static_sources()?;
        }
        Ok(())
    }

    pub fn hash_mode(&self) -> HashMode {
        self.hash_mode
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
        let static_source = if self.dense_http && !crate::backend::is_grpc_url(worker_url) {
            self.http_static_sources()?.get(worker_url)
        } else {
            None
        };
        let discovery = match static_source {
            Some(source) => static_discovery(worker_url, source),
            None => discover_worker(&self.client, worker_url).await,
        };
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

        // Dense HTTP bindings require plain origins. Generic discovery retains
        // its backend-specific URI scope, including HTTP base paths.
        let worker_origin = if self.dense_http && !crate::backend::is_grpc_url(worker_url) {
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
        let mut live_sources: Vec<LiveSource> = Vec::with_capacity(event_sources.len());
        for source in &event_sources {
            let strict_rank = workers
                .values()
                .flat_map(|worker| &worker.live_sources)
                .chain(live_sources.iter())
                .find(|live| live.publisher == source.pub_endpoint && live.topic == source.topic)
                .map(|live| live.strict_rank.clone())
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
                strict_rank,
            });
        }
        if let Some(previous) = workers.remove(worker_url) {
            self.retire_worker(previous);
        }
        let indexer = self.acquire_identity(&key);
        let active = Arc::new(AtomicBool::new(true));
        let mut ranks = Vec::with_capacity(event_sources.len());
        let mut handles = Vec::with_capacity(event_sources.len());
        for (ev, live) in event_sources.into_iter().zip(&live_sources) {
            let rank = ev.dp_rank;
            let handle = crate::kv_index::subscriber::spawn_bound(
                ev,
                indexer.clone(),
                Some(self.signal_tx.clone()),
                self.ctx.clone(),
                (self.dense_http && !crate::backend::is_grpc_url(worker_url))
                    .then_some(key.block_size),
                live.strict_rank.clone(),
                active.clone(),
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
            let _guard = live.strict_rank.lock();
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

fn parse_static_sources(value: &str) -> Result<HashMap<String, StaticSingleSource>, String> {
    let sources: HashMap<String, StaticKvEventSources> =
        serde_json::from_str(value).map_err(|error| format!("static KV sources: {error}"))?;
    let mut live_streams = HashSet::new();
    let mut replay_endpoints = HashSet::new();
    let mut worker_origins = HashSet::new();
    let mut normalized = HashMap::with_capacity(sources.len());
    for (worker, source) in sources {
        if source.endpoints.len() != 1
            || source
                .replay_endpoints
                .as_ref()
                .is_some_and(|endpoints| endpoints.len() != 1)
            || source.block_size == 0
            || source.model.trim().is_empty()
            || source.hwm.is_some_and(|hwm| hwm < 0)
        {
            return Err(format!("invalid static DP1 KV source for {worker}"));
        }
        if !worker_origins.insert(super::endpoints::canonical_worker(&worker)?.0) {
            return Err("static KV worker origins must be independent".into());
        }
        let endpoint =
            crate::kv_index::parse_endpoint_mapping(&format!("{worker}={}", source.endpoints[0]))?
                .1;
        if !live_streams.insert((endpoint.clone(), source.topic.clone())) {
            return Err("static live KV streams must be independent".into());
        }
        let replay_endpoint = source
            .replay_endpoints
            .map(|replay| {
                crate::kv_index::parse_endpoint_mapping(&format!("{worker}={}", replay[0]))
                    .map(|(_, endpoint)| endpoint)
            })
            .transpose()?;
        if let Some(replay) = &replay_endpoint {
            if !replay_endpoints.insert(replay.clone()) {
                return Err("static KV replay endpoints must be independent".into());
            }
        }
        normalized.insert(
            worker,
            StaticSingleSource {
                endpoint,
                replay_endpoint,
                topic: source.topic,
                hwm: source.hwm,
                block_size: source.block_size,
                model: source.model,
            },
        );
    }
    Ok(normalized)
}

/// Normalize operator configuration into the live discovery shape.
/// These are operator-provided wire facts, not a remote boot attestation.
fn static_discovery(worker_url: &str, config: &StaticSingleSource) -> WorkerDiscovery {
    let server_info = ServerInfo {
        model_id: Some(config.model.clone()),
        instance_id: Some(worker_url.to_string()),
        kv_block_size: Some(config.block_size),
        ..Default::default()
    };
    WorkerDiscovery {
        server_info: Ok(server_info),
        sources: Ok(HashMap::from([(
            "0".to_owned(),
            KvEventSourceEntry {
                endpoint: config.endpoint.clone(),
                replay_endpoint: config.replay_endpoint.clone(),
                topic: config.topic.clone(),
                hwm: config.hwm,
                publisher: Some("zmq".to_owned()),
                encoding: Some("msgpack".to_owned()),
                schema_version: Some(1),
                enable_kv_cache_events: Some(true),
                ..Default::default()
            },
        )])),
        model_id: Ok(Some(config.model.clone())),
    }
}

async fn discover_worker(client: &reqwest::Client, worker_url: &str) -> WorkerDiscovery {
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
    let server_info = query_server_info(client, worker_url).await;
    let model_id = server_info
        .as_ref()
        .map(|info| Some(server_model_id(info).unwrap_or_else(|| worker_url.to_string())))
        .map_err(|error| error.clone());
    WorkerDiscovery {
        server_info,
        sources: query_kv_event_sources(client, worker_url).await,
        model_id,
    }
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

async fn query_server_info(
    client: &reqwest::Client,
    worker_url: &str,
) -> Result<ServerInfo, String> {
    let url = format!("{}/get_server_info", worker_url.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("get_server_info {url}: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("get_server_info {url}: status {}", resp.status()));
    }
    resp.json::<ServerInfo>()
        .await
        .map_err(|e| format!("get_server_info parse: {e}"))
}

async fn query_kv_event_sources(
    client: &reqwest::Client,
    worker_url: &str,
) -> Result<KvEventSourcesResponse, String> {
    let url = format!("{}/kv_event_sources", worker_url.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("kv_event_sources {url}: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(format!("kv_event_sources {url}: status {status}"));
    }
    resp.json::<KvEventSourcesResponse>()
        .await
        .map_err(|e| format!("kv_event_sources parse: {e}"))
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(!events_enabled(&e)); // encoding and schema are required
        e.encoding = Some("msgpack".into());
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
    fn static_regular_sources_preserve_mapping_alias_topic_and_hwm() {
        for (worker_count, block_size, topic, hwm) in [
            (2, 16, Some("kv/alias-v2"), Some(128)),
            (1, 8, None, None),
            (1, 32, Some(""), Some(0)),
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
                if let Some(hwm) = hwm {
                    source["hwm"] = serde_json::json!(hwm);
                }
                map.insert(url.clone(), source);
            }
            let sources = parse_static_sources(&serde_json::to_string(&map).unwrap()).unwrap();
            assert_eq!(sources.len(), worker_count);
            for worker in &workers {
                let discovery = static_discovery(worker, &sources[worker]);
                assert_eq!(discovery.model_id, Ok(Some("served-alias-v2".into())));
                let info = discovery.server_info.unwrap();
                assert_eq!(info.instance_id.as_deref(), Some(worker.as_str()));
                assert_eq!(discovered_block_size(&info), Some(block_size));
                let entries = discovery.sources.unwrap();
                assert_eq!(entries.len(), 1);
                let entry = &entries["0"];
                assert_eq!(entry.endpoint, sources[worker].endpoint);
                assert_eq!(entry.replay_endpoint, sources[worker].replay_endpoint);
                assert_eq!(entry.topic, topic.unwrap_or_default());
                assert_eq!(entry.hwm, hwm);
                assert!(events_enabled(entry));
            }
        }
    }

    #[test]
    fn static_regular_mapping_allows_worker_without_static_source() {
        let workers: [String; 2] = ["http://worker:8000".into(), "http://worker:8001".into()];
        let sources = parse_static_sources(
            r#"{"http://worker:8000":{"endpoints":["tcp://worker:5557"],"model":"served-alias","block_size":16}}"#,
        ).unwrap();
        assert_eq!(sources.len(), 1);
        assert!(sources.contains_key(&workers[0]));
        assert!(!sources.contains_key(&workers[1]));
    }

    #[test]
    fn static_sources_allow_distinct_topics_on_one_publisher() {
        let sources = parse_static_sources(r#"{
            "http://worker:8000":{"endpoints":["tcp://worker:5557"],"model":"m","block_size":8,"topic":"kv/a"},
            "http://worker:8001":{"endpoints":["tcp://WORKER:5557"],"model":"m","block_size":8,"topic":"kv/b"}
        }"#).unwrap();
        assert_eq!(sources.len(), 2);
        assert_eq!(
            sources["http://worker:8000"].endpoint,
            sources["http://worker:8001"].endpoint
        );
    }

    #[test]
    fn static_sources_reject_shared_replay_endpoints() {
        use serde_json::json;

        let workers: [String; 2] = ["http://worker:8000".into(), "http://worker:8001".into()];
        let value = json!({
            &workers[0]: {"endpoints": ["tcp://worker:5557"],
                "replay_endpoints": ["tcp://WORKER:5558"], "block_size": 16, "model": "served-alias"},
            &workers[1]: {"endpoints": ["tcp://worker:5559"],
                "replay_endpoints": ["tcp://worker:5558"], "block_size": 16, "model": "served-alias"}
        });
        assert!(parse_static_sources(&value.to_string())
            .unwrap_err()
            .contains("replay endpoints must be independent"));
    }

    #[test]
    fn static_sources_reject_worker_origin_aliases_and_keep_original_keys() {
        use serde_json::json;

        let source = |publisher: u16, replay: Option<&str>| {
            let mut value = json!({
                "endpoints": [format!("tcp://worker:{publisher}")],
                "block_size": 16, "model": "served-alias"
            });
            if let Some(replay) = replay {
                value["replay_endpoints"] = json!([replay]);
            }
            value
        };
        for (first, second) in [
            ("http://worker:8000", "http://worker:8000/"),
            ("http://WORKER:80", "http://worker/"),
            ("https://WORKER:443", "https://worker"),
        ] {
            let value = json!({first: source(5557, None), second: source(5559, None)});
            assert!(parse_static_sources(&value.to_string())
                .unwrap_err()
                .contains("worker origins must be independent"));
        }
        for (first, second) in [
            ("http://WORKER:80/", "https://worker:443"),
            ("http://worker:8000/", "http://worker:8001"),
            ("http://127.0.0.1:8000", "http://[::ffff:127.0.0.1]:8000"),
        ] {
            let value =
                json!({first: source(5557, None), second: source(5559, Some("tcp://worker:5560"))});
            let parsed = parse_static_sources(&value.to_string()).unwrap();
            assert_eq!(parsed.len(), 2);
            assert!(parsed.contains_key(first));
            assert!(parsed.contains_key(second));
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

    async fn boundary_subscription(publisher: &zmq::Socket) {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                match publisher.recv_bytes(zmq::DONTWAIT) {
                    Ok(frame) if frame == b"\x00kv/boundary" => continue,
                    Ok(frame) => {
                        assert_eq!(frame, b"\x01kv/boundary");
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
            let (supervisor, _) = KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
            let (a, b) = tokio::join!(
                supervisor.on_worker_added(&worker_a),
                supervisor.on_worker_added(&worker_b)
            );
            if accepted {
                assert!(a.is_ok() && b.is_ok(), "{a:?} {b:?}");
                assert_eq!(supervisor.workers.lock().len(), 2);
                if same_publisher && rank_b == 1 {
                    let workers = supervisor.workers.lock();
                    let guard_a = &workers[&worker_a].live_sources[0].strict_rank;
                    let guard_b = &workers[&worker_b].live_sources[0].strict_rank;
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
        let (supervisor, _) = KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
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
        use crate::kv_index::{MatchQuery, StorageTier};
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
            KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
        supervisor.on_worker_added(&worker_a).await.unwrap();
        supervisor.on_worker_added(&worker_b).await.unwrap();
        boundary_subscription(&publisher_a).await;
        let payload = rmp_serde::to_vec(&(
            1.0,
            [json!({
                "type": "BlockStored", "block_hashes": [1], "parent_block_hash": null,
                "token_ids": vec![1; 8], "block_size": 8, "medium": "GPU",
                "lora_id": null, "lora_name": null
            })],
            0u32,
        ))
        .unwrap();
        let query = MatchQuery {
            group_idx: 0,
            local_hashes: crate::kv_index::local_block_hash_strings(&[1; 8], 8, None),
            tiers_of_interest: vec![StorageTier::Device],
        };
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
        boundary_subscription(&publisher_a).await;
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
        use crate::kv_index::{MatchQuery, StorageTier};
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
        let (supervisor, mut signals) = KvIndexSupervisor::new_http_dense(
            reqwest::Client::new(),
            HashMode::Sha256,
            &[worker_a.clone(), worker_b.clone()],
            None,
        )
        .unwrap();
        let send = |sequence: i64, token: u32, rank: Option<u32>| {
            let payload = rmp_serde::to_vec(&(1.0, [json!({
                "type": "BlockStored", "block_hashes": [sequence + 1], "parent_block_hash": null,
                "token_ids": vec![token; 8], "block_size": 8, "medium": "GPU",
                "lora_id": null, "lora_name": null
            })], rank)).unwrap();
            publisher
                .send_multipart(
                    [b"kv/boundary".as_slice(), &sequence.to_be_bytes(), &payload],
                    0,
                )
                .unwrap();
        };
        let query = |token| MatchQuery {
            group_idx: 0,
            local_hashes: crate::kv_index::local_block_hash_strings(&[token; 8], 8, None),
            tiers_of_interest: vec![StorageTier::Device],
        };
        supervisor.on_worker_added(&worker_a).await.unwrap();
        boundary_subscription(&publisher).await;
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
        boundary_subscription(&publisher).await;

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
        use crate::kv_index::{MatchQuery, StorageTier};
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
                supervisor.dense_http = true;
                if static_a {
                    supervisor.static_sources = OnceLock::from(parse_static_sources(
                        &json!({&worker_a: {
                            "endpoints": [endpoint_a], "topic": "kv/boundary",
                            "block_size": 8, "model": "boundary-model"
                        }})
                        .to_string(),
                    ));
                }
                supervisor.on_worker_added(&worker_a).await.unwrap();
                boundary_subscription(&publisher_a).await;
                let payload = rmp_serde::to_vec(&(
                    1.0,
                    [json!({
                        "type": "BlockStored", "block_hashes": [1], "parent_block_hash": null,
                        "token_ids": vec![1; 8], "block_size": 8, "medium": "GPU",
                        "lora_id": null, "lora_name": null
                    })],
                    0u32,
                ))
                .unwrap();
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
                let query = MatchQuery {
                    group_idx: 0,
                    local_hashes: crate::kv_index::local_block_hash_strings(&[1; 8], 8, None),
                    tiers_of_interest: vec![StorageTier::Device],
                };
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
        use crate::kv_index::{MatchQuery, StorageTier};
        use serde_json::json;

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
            KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
        supervisor.on_worker_added(&worker_a).await.unwrap();
        boundary_subscription(&a).await;
        let payload = |hash: u32, token: u32| {
            rmp_serde::to_vec(&(
                1.0,
                [json!({
                    "type": "BlockStored", "block_hashes": [hash], "parent_block_hash": null,
                    "token_ids": vec![token; 8], "block_size": 8, "medium": "GPU",
                    "lora_id": null, "lora_name": null
                })],
                0u32,
            ))
            .unwrap()
        };
        a.send_multipart(
            [
                b"kv/boundary".as_slice(),
                &0i64.to_be_bytes(),
                &payload(1, 1),
            ],
            0,
        )
        .unwrap();
        assert!(
            matches!(tokio::time::timeout(Duration::from_secs(5), signals.recv()).await.unwrap(),
            Some(IngestionSignal::Advance { source, last_seq: 0 }) if source.as_ref() == "boundary-a")
        );
        let query = |token| MatchQuery {
            group_idx: 0,
            local_hashes: crate::kv_index::local_block_hash_strings(&[token; 8], 8, None),
            tiers_of_interest: vec![StorageTier::Device],
        };
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
        let (supervisor, _signals) = KvIndexSupervisor::new_http_dense(
            reqwest::Client::new(),
            HashMode::Sha256,
            &[worker_a.clone(), worker_b.clone()],
            None,
        )
        .unwrap();
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
    async fn generic_http_discovery_preserves_worker_base_paths() {
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
        let (generic, _) = KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
        let result = generic.on_worker_added(&prefixed).await;
        let registered = generic.match_provider_for_worker(&prefixed).is_some();
        let original_absent = generic.match_provider_for_worker(&worker).is_none();
        generic.shutdown();
        let (dense, _) = KvIndexSupervisor::new_http_dense(
            reqwest::Client::new(),
            HashMode::Sha256,
            std::slice::from_ref(&prefixed),
            None,
        )
        .unwrap();
        let dense_result = dense.on_worker_added(&prefixed).await;
        let unqualified = dense.match_provider_for_worker(&prefixed).is_none();
        dense.shutdown();
        http_task.abort();
        let info = result.unwrap();
        assert_eq!(info.instance_id, "prefixed-source");
        assert_eq!(info.ranks, [0]);
        assert!(registered);
        assert!(original_absent);
        assert!(dense_result.unwrap_err().contains("plain HTTP(S) origins"));
        assert!(unqualified);
    }

    #[test]
    fn empty_and_grpc_pools_defer_known_invalid_http_static_configuration() {
        for workers in [vec![], vec!["grpc://worker:50051".to_owned()]] {
            let (supervisor, _) = KvIndexSupervisor::new_http_dense(
                reqwest::Client::new(),
                HashMode::Sha256,
                &workers,
                Some("not valid JSON".into()),
            )
            .unwrap();
            assert!(supervisor.dense_http);
            assert!(supervisor.static_sources.get().is_none());
            assert!(supervisor
                .validate_http_worker("grpc://worker:50051", 1)
                .is_ok());
            assert!(supervisor
                .validate_http_worker("http://worker:8000", 1)
                .unwrap_err()
                .contains("static KV sources"));
            supervisor.shutdown();
        }
        let (generic, _) = KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
        assert!(!generic.dense_http);
        generic.shutdown();
    }

    #[tokio::test]
    async fn dense_source_block_size_mismatch_revokes_only_affected_source() {
        for (expected, actual) in [(16, 8), (8, 16)] {
            for use_static in [false, true] {
                check_dense_source_block_size(expected, actual, use_static).await;
            }
        }
    }

    async fn check_dense_source_block_size(expected: u32, actual: u32, use_static: bool) {
        use crate::kv_index::{MatchQuery, StorageTier};
        use axum::{routing::get, Json, Router};
        use serde_json::json;

        let context = zmq::Context::new();
        let publisher = context.socket(zmq::XPUB).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
        let (mut supervisor, mut signals) =
            KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
        // Static Regular and the discovered Dense consumer share this admission.
        supervisor.dense_http = true;
        let rank = if use_static { 0 } else { 3 };
        let (worker, http_task) = if use_static {
            let worker = "http://127.0.0.1:1".to_string();
            supervisor.static_sources = OnceLock::from(parse_static_sources(
                &json!({&worker: {
                    "endpoints": [endpoint], "block_size": expected,
                    "model": "dense-model", "topic": "kv/dense"
                }})
                .to_string(),
            ));
            (worker, None)
        } else {
            let sources = json!({(rank.to_string()): {
                "endpoint": endpoint, "topic": "kv/dense", "publisher": "zmq",
                "encoding": "msgpack", "schema_version": 1, "enable_kv_cache_events": true
            }});
            let app = Router::new()
                .route("/get_server_info", get(move || async move { Json(json!({
                    "model_id": "dense-model", "instance_id": "dense-boot", "kv_block_size": expected
                })) }))
                .route("/kv_event_sources", get(move || {
                    let sources = sources.clone();
                    async move { Json(sources) }
                }));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let worker = format!("http://{}", listener.local_addr().unwrap());
            let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            (worker, Some(task))
        };
        let info = supervisor.on_worker_added(&worker).await.unwrap();
        assert_eq!(info.cache_key.as_ref().unwrap().block_size, expected);
        assert_eq!(info.ranks, [rank]);
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                match publisher.recv_bytes(zmq::DONTWAIT) {
                    Ok(frame) => {
                        assert_eq!(frame, b"\x01kv/dense");
                        break;
                    }
                    Err(zmq::Error::EAGAIN) => tokio::task::yield_now().await,
                    Err(error) => panic!("XPUB subscription: {error}"),
                }
            }
        })
        .await
        .unwrap();
        let indexer = supervisor
            .workers
            .lock()
            .get(&worker)
            .unwrap()
            .indexer
            .clone();
        let old_tokens: Vec<u32> = (0..expected).collect();
        let new_tokens: Vec<u32> = (100..100 + actual).collect();
        let query = |tokens: &[u32]| MatchQuery {
            group_idx: 0,
            local_hashes: crate::kv_index::local_block_hash_strings(tokens, expected, None),
            tiers_of_interest: vec![StorageTier::Device, StorageTier::HostPinned],
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
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    match publisher.recv_bytes(zmq::DONTWAIT) {
                        Ok(frame) => {
                            assert_eq!(&frame[1..], topic.as_bytes());
                            break;
                        }
                        Err(zmq::Error::EAGAIN) => tokio::task::yield_now().await,
                        Err(error) => panic!("XPUB neighbor subscription: {error}"),
                    }
                }
            })
            .await
            .unwrap();
            let payload = rmp_serde::to_vec(&(
                1.0,
                [json!({
                    "type": "BlockStored", "block_hashes": [70], "parent_block_hash": null,
                    "token_ids": old_tokens, "block_size": expected, "medium": "GPU",
                    "lora_id": null, "lora_name": null
                })],
                neighbor_rank,
            ))
            .unwrap();
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
            let payload = rmp_serde::to_vec(&(
                1.0,
                [json!({
                    "type": "BlockStored", "block_hashes": [seq + 7], "parent_block_hash": null,
                    "token_ids": tokens, "block_size": block_size, "lora_id": null,
                    "lora_name": null, "medium": if use_static { "GPU" } else { "CPU" }
                })],
                rank,
            ))
            .unwrap();
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
                                "accepted mismatched Dense batch: source={expected}, event={actual}, static={use_static}, old={:?}, mismatched={:?}",
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
        if let Some(task) = http_task {
            task.abort();
        }
    }

    #[tokio::test]
    async fn http_and_static_sources_ingest_nonempty_topics_through_live_zmq() {
        use crate::kv_index::{MatchQuery, StorageTier};
        use axum::{routing::get, Json, Router};
        use serde_json::json;

        for (use_static, block_size) in [
            (false, 8u32),
            (false, 16),
            (false, 32),
            (true, 8),
            (true, 16),
            (true, 32),
        ] {
            let context = zmq::Context::new();
            let publisher = context.socket(zmq::XPUB).unwrap();
            publisher.bind("tcp://127.0.0.1:*").unwrap();
            let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
            let rank = if use_static { 0u32 } else { 3 };
            let (mut supervisor, mut signals) =
                KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
            let (worker, http_task) = if use_static {
                let worker = "http://127.0.0.1:1".to_string();
                let value = json!({
                    &worker: {
                        "endpoints": [endpoint], "block_size": block_size,
                        "model": "served-alias", "topic": "kv/served-alias", "hwm": 128
                    }
                })
                .to_string();
                supervisor.dense_http = true;
                supervisor.static_sources = OnceLock::from(parse_static_sources(&value));
                (worker, None)
            } else {
                let sources = json!({(rank.to_string()): {
                    "endpoint": endpoint, "topic": "kv/served-alias", "hwm": 128,
                    "publisher": "zmq", "encoding": "msgpack", "schema_version": 1,
                    "enable_kv_cache_events": true
                }});
                let app = Router::new()
                    .route("/get_server_info", get(move || async move { Json(json!({
                        "model_id": "served-alias", "instance_id": "observed-boot", "kv_block_size": block_size
                    })) }))
                    .route("/kv_event_sources", get(move || {
                        let sources = sources.clone();
                        async move { Json(sources) }
                    }));
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                let worker = format!("http://{}", listener.local_addr().unwrap());
                let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
                (worker, Some(task))
            };
            let info = supervisor.on_worker_added(&worker).await.unwrap();
            assert_eq!(info.ranks, [rank]);
            assert_eq!(
                info.cache_key.as_ref().unwrap().model.as_ref(),
                "served-alias"
            );
            let subscription = tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    match publisher.recv_bytes(zmq::DONTWAIT) {
                        Ok(frame) => break frame,
                        Err(zmq::Error::EAGAIN) => {
                            tokio::time::sleep(Duration::from_millis(10)).await
                        }
                        Err(error) => panic!("XPUB subscription: {error}"),
                    }
                }
            })
            .await
            .unwrap();
            assert_eq!(&subscription, b"\x01kv/served-alias");
            let tokens: Vec<u32> = (0..block_size).collect();
            let event = json!({
                "type": "BlockStored", "block_hashes": [7], "parent_block_hash": null,
                "token_ids": tokens, "block_size": block_size, "lora_id": null, "lora_name": null,
                "medium": if use_static { "GPU" } else { "CPU" }
            });
            let payload = rmp_serde::to_vec(&(1.0, [event], rank)).unwrap();
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
            let hits = indexer.find_matches(&MatchQuery {
                group_idx: 0,
                local_hashes: crate::kv_index::local_block_hash_strings(&tokens, block_size, None),
                tiers_of_interest: vec![tier],
            });
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
    fn static_regular_sources_reject_invalid_dp1_settings() {
        let worker = "http://worker:8000".to_string();
        let valid = serde_json::json!({
            "endpoints": ["tcp://worker:5557"],
            "block_size": 16,
            "model": "served-alias"
        });
        for (field, invalid) in [
            ("endpoints", serde_json::json!([])),
            (
                "endpoints",
                serde_json::json!(["tcp://worker:5557", "tcp://worker:5559"]),
            ),
            ("endpoints", serde_json::json!(["tcp://worker:0"])),
            ("replay_endpoints", serde_json::json!([])),
            ("replay_endpoints", serde_json::json!(["tcp://worker:0"])),
            ("block_size", serde_json::json!(0)),
            ("model", serde_json::json!(" ")),
            ("hwm", serde_json::json!(-1)),
            ("hwm", serde_json::json!(2147483648_u64)),
            ("unknown_setting", serde_json::json!(true)),
        ] {
            let mut source = valid.clone();
            source[field] = invalid;
            let value = serde_json::json!({ &worker: source }).to_string();
            assert!(parse_static_sources(&value).is_err(), "{value}");
        }
        let mut missing_model = valid.clone();
        missing_model.as_object_mut().unwrap().remove("model");
        let value = serde_json::json!({ &worker: missing_model }).to_string();
        assert!(parse_static_sources(&value).is_err());

        let value = serde_json::json!({ "http://unknown:8000": &valid }).to_string();
        assert!(parse_static_sources(&value)
            .unwrap()
            .contains_key("http://unknown:8000"));
        let other = "http://worker:8001".to_string();
        let value = serde_json::json!({ &worker: &valid, &other: &valid }).to_string();
        assert!(parse_static_sources(&value)
            .unwrap_err()
            .contains("live KV streams must be independent"));
    }
}
