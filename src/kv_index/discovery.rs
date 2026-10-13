//! Worker discovery: query /get_server_info + /kv_event_sources, spawn one
//! subscriber per (worker, rank). One indexer per cache-identity core.

use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::Mutex;
use serde::Deserialize;
use tokio::sync::mpsc;
use tracing::{info, warn};

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
    hash_mode: HashMode,
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
            ctx: Arc::new(zmq::Context::new()),
            indexers: DashMap::new(),
            identity_refs: Mutex::new(HashMap::new()),
            signal_tx,
            workers: Mutex::new(HashMap::new()),
            first_block_size: Mutex::new(None),
        };
        (supervisor, signal_rx)
    }

    pub fn hash_mode(&self) -> HashMode {
        self.hash_mode
    }

    /// Query both endpoints for `worker_url`, spawn one subscriber per rank into
    /// this worker's identity's indexer. A 404 on /kv_event_sources (engine
    /// without the endpoint or events off) is a soft skip.
    pub async fn on_worker_added(&self, worker_url: &str) -> Result<WorkerKvInfo, String> {
        // Re-add would overwrite prior handles — leak the identity ref and race
        // stale subscribers into the indexer. Tear the old ones down first.
        if self.workers.lock().contains_key(worker_url) {
            warn!(
                "kv_index {}: already tracked — respawning subscribers",
                worker_url
            );
            self.on_worker_removed(worker_url);
        }

        let server_info = query_server_info(&self.client, worker_url)
            .await
            .unwrap_or_else(|e| {
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
        let model_id = server_info.model_id.clone().or_else(|| {
            server_info
                .model_path
                .as_ref()
                .and_then(|p| p.split('/').next_back().map(str::to_string))
        });
        let model_id = match model_id {
            Some(m) => m,
            None => {
                warn!("kv_index {}: no model_id/model_path — using URL; cross-worker KV sharing disabled", worker_url);
                worker_url.to_string()
            }
        };
        let block_size = server_info
            .kv_block_size
            .or(server_info.effective_attention_block_size);
        self.check_block_size(worker_url, block_size);

        let sources = match query_kv_event_sources(&self.client, worker_url).await {
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
            let handle = spawn_subscriber(
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
                key,
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

/// A rank emits events iff publisher != "null" and events are enabled.
fn events_enabled(entry: &KvEventSourceEntry) -> bool {
    if matches!(entry.publisher.as_deref(), Some("null")) {
        return false;
    }
    if matches!(entry.enable_kv_cache_events, Some(false)) {
        return false;
    }
    !entry.endpoint.is_empty()
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(events_enabled(&e));
        e.publisher = Some("null".into());
        assert!(!events_enabled(&e));
        e.publisher = Some("zmq".into());
        e.enable_kv_cache_events = Some(false);
        assert!(!events_enabled(&e));
        e.enable_kv_cache_events = Some(true);
        assert!(events_enabled(&e));
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
    fn kv_event_sources_decodes_rank_map() {
        let body = r#"{
            "0": {"endpoint":"tcp://*:5557","replay_endpoint":"tcp://*:5558","topic":"","hwm":100000,"publisher":"zmq","enable_kv_cache_events":true},
            "1": {"endpoint":"tcp://*:5559","replay_endpoint":null,"topic":"","publisher":"zmq"}
        }"#;
        let m: KvEventSourcesResponse = serde_json::from_str(body).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m["0"].hwm, Some(100_000));
        assert!(m["1"].replay_endpoint.is_none());
    }
}
