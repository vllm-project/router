//! Worker discovery: query /get_server_info + /kv_event_sources, spawn one
//! subscriber per (worker, rank). One indexer per cache-identity core.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use parking_lot::Mutex;
use serde::Deserialize;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::config::StaticKvEventSources;
use crate::kv_index::indexer::KvBlockIndexer;
use crate::kv_index::subscriber::{spawn as spawn_subscriber, EventSource, SubscriberHandle};
use crate::kv_index::{ClearScope, HashMode, IngestionSignal, ResidencyOwner, SourceId};
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
    key: CacheKey,
    source_id: SourceId,
    ranks: Vec<u32>,
    indexer: Arc<KvBlockIndexer>,
    handles: Vec<SubscriberHandle>,
}

/// Orchestrates discovery + subscriber lifecycle for the router's workers.
/// One indexer per `CacheKey`, shared by every `(worker, rank)` of that
/// identity.
pub struct KvIndexSupervisor {
    client: reqwest::Client,
    api_key: Option<String>,
    hash_mode: HashMode,
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

    /// HTTP P/D accepts the native Python control endpoints and dense KV
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
        self.on_worker_removed(worker_url);
        let discovery = self.discover(worker_url).await;
        self.install_worker(worker_url, discovery)
    }

    /// Prepare metadata once; P/D refresh fingerprints and installs this same snapshot.
    pub(crate) async fn discover(&self, worker_url: &str) -> WorkerDiscovery {
        if self.http_pd {
            match self.static_pd_sources.get(worker_url) {
                Some(config) => static_pd_discovery(worker_url, config),
                None => {
                    discover_worker(&self.client, worker_url, true, self.api_key.as_deref()).await
                }
            }
        } else {
            match self.static_sources.get(worker_url) {
                Some(config) => static_worker_discovery(worker_url, config),
                None => {
                    discover_worker(&self.client, worker_url, false, self.api_key.as_deref()).await
                }
            }
        }
    }

    pub(crate) fn install_worker(
        &self,
        worker_url: &str,
        discovery: WorkerDiscovery,
    ) -> Result<WorkerKvInfo, String> {
        self.on_worker_removed(worker_url);
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
        self.check_block_size(worker_url, block_size);
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
        let indexer = self.acquire_identity(&key);

        let worker_host = worker_host(worker_url).unwrap_or_else(|| worker_url.to_string());
        let source_id: SourceId = Arc::from(instance_id.as_str());
        let mut ranks = Vec::with_capacity(sources.len());
        let mut handles = Vec::with_capacity(sources.len());

        for (rank_str, entry) in &sources {
            if !events_enabled(entry) {
                continue;
            }
            let rank: u32 = match rank_str.parse() {
                Ok(r) => r,
                Err(_) => {
                    warn!(
                        "kv_index {}: bad rank key '{}', skipping",
                        worker_url, rank_str
                    );
                    continue;
                }
            };
            let pub_endpoint = resolve_endpoint(&entry.endpoint, &worker_host);
            let replay_endpoint = entry
                .replay_endpoint
                .as_deref()
                .map(|e| resolve_endpoint(e, &worker_host));
            let ev = EventSource {
                source: source_id.clone(),
                dp_rank: rank,
                pub_endpoint,
                replay_endpoint,
                topic: entry.topic.clone(),
                hwm: entry.hwm,
            };
            let spawn = if self.http_pd {
                crate::kv_index::subscriber::spawn_pd
            } else {
                spawn_subscriber
            };
            let handle = spawn(
                ev,
                indexer.clone(),
                Some(self.signal_tx.clone()),
                self.ctx.clone(),
            );
            info!(
                "kv_index {}: spawned subscriber rank{} (instance {})",
                worker_url, rank, instance_id
            );
            handles.push(handle);
            ranks.push(rank);
        }

        self.workers.lock().insert(
            worker_url.to_string(),
            WorkerSubs {
                key: key.clone(),
                source_id,
                ranks: ranks.clone(),
                indexer,
                handles,
            },
        );
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
        let entry = self.workers.lock().remove(worker_url);
        if let Some(subs) = entry {
            for h in subs.handles {
                h.shutdown();
            }
            for rank in &subs.ranks {
                let owner = ResidencyOwner::Worker {
                    source: subs.source_id.clone(),
                    dp_rank: *rank,
                    incarnation: 0,
                };
                subs.indexer.clear(&owner, ClearScope::Worker);
            }
            self.release_identity(&subs.key);
            info!("kv_index {}: removed subscribers", worker_url);
        }
    }

    /// Shut down every subscriber and drop all indexers.
    pub fn shutdown(&self) {
        let mut workers = self.workers.lock();
        let keys: Vec<CacheKey> = workers.values().map(|s| s.key.clone()).collect();
        for (_url, subs) in workers.drain() {
            for h in subs.handles {
                h.shutdown();
            }
        }
        drop(workers);
        for key in keys {
            self.release_identity(&key);
        }
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

/// Rewrite a ZMQ bind address's host (`*` / `0.0.0.0`) to `worker_host`.
/// `tcp://*:5557` → `tcp://10.0.0.5:5557`.
fn resolve_endpoint(endpoint: &str, worker_host: &str) -> String {
    if let Some(rest) = endpoint.strip_prefix("tcp://") {
        if let Some((host, port)) = rest.split_once(':') {
            if host == "*" || host == "0.0.0.0" {
                return format!("tcp://{}:{}", worker_host, port);
            }
        }
    }
    endpoint.to_string()
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
            resolve_endpoint("tcp://*:5557", "10.0.0.5"),
            "tcp://10.0.0.5:5557"
        );
        assert_eq!(
            resolve_endpoint("tcp://0.0.0.0:5558", "10.0.0.5"),
            "tcp://10.0.0.5:5558"
        );
    }

    #[test]
    fn resolve_endpoint_keeps_resolved_host() {
        assert_eq!(
            resolve_endpoint("tcp://10.0.0.5:5557", "ignored"),
            "tcp://10.0.0.5:5557"
        );
    }

    #[test]
    fn resolve_endpoint_passthrough_non_tcp() {
        assert_eq!(resolve_endpoint("ipc:///tmp/x", "h"), "ipc:///tmp/x");
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
}
