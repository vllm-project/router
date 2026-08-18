/*
    Cache-Aware Load Balancing Router

    This router combines two strategies to optimize both cache utilization and request distribution:

    1. Cache-Aware Routing (Approximate Tree)
    2. Load Balancing (Shortest Queue with Balance Thresholds)

    The router dynamically switches between these strategies based on load conditions:
    - Uses load balancing when the system is imbalanced
    - Uses cache-aware routing when the system is balanced

    A system is considered imbalanced if both conditions are met:
    1. (max - min) > abs_threshold
    2. max > rel_threshold * min

    Strategy Details:

    1. Cache-Aware Routing (Approximate Tree)
    -------------------------------------------
    This strategy maintains an approximate radix tree for each worker based on request history,
    eliminating the need for direct cache state queries. The tree stores raw text characters
    instead of token IDs to avoid tokenization overhead.

    Process:
    a. For each request, find the worker with the highest prefix match
    b. If match rate > cache_threshold:
    Route to the worker with highest match (likely has relevant data cached)
    c. If match rate ≤ cache_threshold:
    Route to the worker with smallest tree size (most available cache capacity)
    d. Background maintenance:
    Periodically evict least recently used leaf nodes to prevent memory overflow

    2. Load Balancing (Shortest Queue)
    -------------------------------------------
    This strategy tracks pending request counts per worker and routes new requests
    to the least busy worker when the system is detected to be imbalanced.

    Configuration Parameters:
    ------------------------
    1. cache_threshold: (float, 0.0 to 1.0)
    Minimum prefix match ratio to use highest-match routing.
    Below this threshold, routes to worker with most available cache space.

    2. balance_abs_threshold: (integer)
    Absolute difference threshold for load imbalance detection.
    System is potentially imbalanced if (max_load - min_load) > abs_threshold

    3. balance_rel_threshold: (float)
    Relative ratio threshold for load imbalance detection.
    System is potentially imbalanced if max_load > min_load * rel_threshold
    Used in conjunction with abs_threshold to determine final imbalance state.

    4. eviction_interval_secs: (integer)
    Interval between LRU eviction cycles for the approximate trees.

    5. max_tree_size: (integer)
    Maximum nodes per tree. When exceeded, LRU leaf nodes are evicted
    during the next eviction cycle.
*/

use super::hash_key::{extract_hash_key_from_body, extract_hash_key_from_headers};
use super::{
    get_healthy_worker_indices, CacheAwareConfig, LoadBalancingPolicy, RequestHeaders,
    RoutingSelection,
};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::policies::normalize_model_key;
use crate::tree::Tree;
use dashmap::DashMap;
use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{debug, info};

/// How long an untouched session entry survives before eviction.
const SESSION_ENTRY_TTL_SECS: u64 = 3600;

/// Value of a session entry: the worker URL and the last-access timestamp.
pub type SessionEntry = (String, u64);
/// Per-model session map: session_id -> SessionEntry.
pub type SessionMap = DashMap<String, SessionEntry>;

/// Cache-aware routing policy
///
/// Routes requests based on cache affinity when load is balanced,
/// switches to shortest-queue routing when load is imbalanced.
/// Maintains separate trees per model for multi-model support.
///
/// Session keys are matched by exact string equality in a dedicated
/// per-model map; the prefix tree only holds text-like keys (full chat
/// history for chat requests, plain prompts otherwise).
#[derive(Debug)]
pub struct CacheAwarePolicy {
    config: CacheAwareConfig,
    trees: Arc<DashMap<String, Arc<Tree>>>, // model_id -> Arc<Tree>
    // model_id -> session_id -> SessionEntry
    session_keys: Arc<DashMap<String, Arc<SessionMap>>>,
    eviction_handle: Option<thread::JoinHandle<()>>,
}

/// Current unix time in milliseconds, for session-entry aging.
fn session_now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Drop session entries that were not touched for the TTL, then enforce the
/// per-model capacity by evicting the least recently accessed entries.
fn evict_session_entries(map: &SessionMap, max_entries: usize) {
    let cutoff = session_now_ms().saturating_sub(SESSION_ENTRY_TTL_SECS * 1000);
    map.retain(|_, (_, last_access_ms)| *last_access_ms >= cutoff);

    if map.len() > max_entries {
        let mut entries: Vec<(String, u64)> =
            map.iter().map(|e| (e.key().clone(), e.value().1)).collect();
        entries.sort_by_key(|(_, last_access_ms)| *last_access_ms);
        let to_remove = entries.len() - max_entries;
        for (key, _) in entries.into_iter().take(to_remove) {
            map.remove(&key);
        }
    }
}

impl CacheAwarePolicy {
    pub fn new() -> Self {
        Self::with_config(CacheAwareConfig::default())
    }

    pub fn with_config(config: CacheAwareConfig) -> Self {
        let trees = Arc::new(DashMap::<String, Arc<Tree>>::new());
        let session_keys = Arc::new(DashMap::<String, Arc<SessionMap>>::new());

        // Start background eviction thread if configured
        let eviction_handle = if config.eviction_interval_secs > 0 {
            let trees_clone = Arc::clone(&trees);
            let session_keys_clone = Arc::clone(&session_keys);
            let max_tree_size = config.max_tree_size;
            let interval = config.eviction_interval_secs;

            Some(thread::spawn(move || loop {
                thread::sleep(Duration::from_secs(interval));

                // Evict for all model trees
                for entry in trees_clone.iter() {
                    let model_id = entry.key();
                    let tree = entry.value();
                    tree.evict_tenant_by_size(max_tree_size);
                    debug!(
                        "Cache eviction completed for model {}, max_size: {}",
                        model_id, max_tree_size
                    );
                }

                // Sweep stale/overflowing session entries
                for entry in session_keys_clone.iter() {
                    let model_id = entry.key().clone();
                    let map = entry.value();
                    evict_session_entries(map, max_tree_size);
                    debug!("Session key eviction completed for model {}", model_id);
                }
            }))
        } else {
            None
        };

        Self {
            config,
            trees,
            session_keys,
            eviction_handle,
        }
    }

    /// Add a single worker to the tree (incremental update)
    pub fn add_worker(&self, worker: &dyn Worker) {
        let tree_key = normalize_model_key(worker.model_id());
        let tree = self
            .trees
            .entry(tree_key.to_string())
            .or_insert_with(|| Arc::new(Tree::new()));
        tree.insert("", worker.url());
        self.session_keys
            .entry(tree_key.to_string())
            .or_insert_with(|| Arc::new(DashMap::new()));
    }

    /// Add a worker by URL and model (for backward compatibility)
    pub fn add_worker_by_url(&self, url: &str, model_id: &str) {
        let tree = self
            .trees
            .entry(model_id.to_string())
            .or_insert_with(|| Arc::new(Tree::new()));
        tree.insert("", url);
        self.session_keys
            .entry(model_id.to_string())
            .or_insert_with(|| Arc::new(DashMap::new()));
    }

    /// Remove a worker from the tree
    pub fn remove_worker(&self, worker: &dyn Worker) {
        let tree_key = normalize_model_key(worker.model_id());
        if let Some(tree) = self.trees.get(tree_key) {
            tree.remove_tenant(worker.url());
        }
        if let Some(map) = self.session_keys.get(tree_key) {
            map.retain(|_, (url, _)| url != worker.url());
        }
    }

    /// Remove a worker by URL (removes from all model trees for backward compatibility)
    pub fn remove_worker_by_url(&self, url: &str) {
        // Remove from all trees since we don't know which model it belongs to
        for tree_ref in self.trees.iter() {
            tree_ref.value().remove_tenant(url);
        }
        for map_ref in self.session_keys.iter() {
            map_ref.value().retain(|_, (entry_url, _)| entry_url != url);
        }
    }

    /// Run cache eviction to prevent unbounded growth
    pub fn evict_cache(&self, max_size: usize) {
        for tree_ref in self.trees.iter() {
            let model_id = tree_ref.key();
            let tree = tree_ref.value();
            tree.evict_tenant_by_size(max_size);
            debug!(
                "Cache eviction for model {}, max_size: {}",
                model_id, max_size
            );
        }
        for map_ref in self.session_keys.iter() {
            evict_session_entries(map_ref.value(), max_size);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn select_worker_min_load(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        fallback_text: Option<&str>,
        healthy_indices: &[usize],
        model_id: &str,
        max_load: usize,
        min_load: usize,
    ) -> Option<RoutingSelection> {
        // Log load balancing trigger (only compute worker loads if debug enabled)
        if tracing::enabled!(tracing::Level::DEBUG) {
            let worker_loads: Vec<(&str, usize)> =
                workers.iter().map(|w| (w.url(), w.load())).collect();
            debug!(
                "Load balancing triggered | max: {} | min: {} | workers: {:?}",
                max_load, min_load, worker_loads
            );
        }

        RouterMetrics::record_load_balancing_event();
        RouterMetrics::record_cache_aware_decision("load_balance");
        RouterMetrics::set_load_range(max_load, min_load);

        // Use shortest queue when imbalanced
        let min_load_idx = healthy_indices
            .iter()
            .min_by_key(|&&idx| workers[idx].load())
            .copied()?;

        // Even in imbalanced mode, maintain cache state: the session map
        // learns the session key (exact match) and the tree learns the full
        // chat history, so a later balanced request sharing the same prefix
        // can discover the worker that already cached it.
        let worker_url = workers[min_load_idx].url();
        if fallback_text.is_some() {
            if let Some(request_text) = request_text {
                if !request_text.trim().is_empty() {
                    self.session_keys
                        .entry(model_id.to_string())
                        .or_insert_with(|| Arc::new(DashMap::new()))
                        .insert(
                            request_text.to_string(),
                            (worker_url.to_string(), session_now_ms()),
                        );
                }
            }
            if let Some(tree) = self.trees.get(model_id).map(|entry| entry.value().clone()) {
                if let Some(text) = fallback_text {
                    if !text.is_empty() {
                        tree.insert(text, worker_url);
                    }
                }
            }
        } else if let Some(tree) = self.trees.get(model_id).map(|entry| entry.value().clone()) {
            if let Some(text) = request_text {
                if !text.is_empty() {
                    tree.insert(text, worker_url);
                }
            }
        }

        // Increment processed counter
        workers[min_load_idx].increment_processed();
        RouterMetrics::record_processed_request(workers[min_load_idx].url());
        RouterMetrics::record_policy_decision(self.name(), workers[min_load_idx].url());

        Some(RoutingSelection {
            index: min_load_idx,
            decision: Some("load_balance"),
        })
    }

    fn select_worker_cache_aware_with_decision(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        fallback_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<RoutingSelection> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Determine the model for this set of workers (router pre-filters by model)
        // All workers should be from the same model.
        let model_id = normalize_model_key(workers[healthy_indices[0]].model_id());

        // Get current load statistics - compute min/max in single pass without allocation.
        let (min_load, max_load) = workers.iter().fold((usize::MAX, 0usize), |(min, max), w| {
            let load = w.load();
            (min.min(load), max.max(load))
        });
        let min_load = if min_load == usize::MAX { 0 } else { min_load };

        // Check if load is imbalanced.
        let is_imbalanced = max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
            && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold);

        debug!(
            "Load status for model: max_load={}, min_load={}, is_imbalanced={}",
            max_load, min_load, is_imbalanced
        );

        if is_imbalanced {
            return self.select_worker_min_load(
                workers,
                request_text,
                fallback_text,
                &healthy_indices,
                model_id,
                max_load,
                min_load,
            );
        }

        // Session identity is derived the same way hash policies derive their
        // key (extract_hash_key priority: HTTP headers, then body fields),
        // with the explicit routing text winning when present: for chat
        // requests the routing text already IS the session id extracted from
        // session_params, and text-like prompts keep their prefix semantics.
        let header_key = headers.and_then(extract_hash_key_from_headers);
        let body_key = extract_hash_key_from_body(request_text);
        let request_text_non_empty = request_text.filter(|text| !text.trim().is_empty());
        let primary_text = request_text_non_empty
            .or(header_key.as_deref())
            .or(body_key.as_deref())
            .unwrap_or("");
        // A fallback key marks the chat path. Session-derived keys (HTTP
        // headers or body fields) use exact-match session semantics even
        // without a fallback key, so a request carrying only an x-session-id
        // header still gets sticky session affinity. Plain text keys keep
        // prefix-tree semantics even when unrelated headers are present.
        // Session keys are matched by exact string equality against the
        // session map (no prefix semantics, so session-1 can never collide
        // with session-12). When there is no extractable history (e.g. an
        // image-only chat), a session miss falls back to min-load instead of
        // probing an empty history key.
        let fallback_provided = fallback_text.is_some();
        let fallback_text = fallback_text.filter(|text| !text.trim().is_empty());
        let session_keyed = !primary_text.is_empty()
            && (fallback_provided
                || (request_text_non_empty.is_none()
                    && (header_key.is_some() || body_key.is_some())));
        let primary_text = if primary_text.is_empty() {
            fallback_text.unwrap_or("")
        } else {
            primary_text
        };

        let session_map = if session_keyed {
            Some(
                self.session_keys
                    .entry(model_id.to_string())
                    .or_insert_with(|| Arc::new(DashMap::new()))
                    .clone(),
            )
        } else {
            None
        };

        // Get the tree reference without locking the entire HashMap.
        // DashMap only locks the specific shard containing this key.
        let tree = self.trees.get(model_id).map(|entry| entry.value().clone());

        let keys: Vec<_> = self.trees.iter().map(|entry| entry.key().clone()).collect();
        debug!("Available tree keys: {:?}", keys);

        let Some(tree) = tree else {
            debug!(
                "Warning: No tree found for model '{}', using random worker selection",
                model_id
            );
            RouterMetrics::record_cache_aware_decision("no_tree_random");
            let mut rng = rand::rng();
            let random_idx = rng.random_range(0..healthy_indices.len());
            let selected_idx = healthy_indices[random_idx];

            workers[selected_idx].increment_processed();
            RouterMetrics::record_processed_request(workers[selected_idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[selected_idx].url());

            return Some(RoutingSelection {
                index: selected_idx,
                decision: Some("no_tree_random"),
            });
        };
        debug!("Using cache-aware routing for model '{}'", model_id);

        let probe = |text: &str, cache_threshold: f32| {
            let result = tree.prefix_match_with_counts(text);
            let match_rate = if result.input_char_count == 0 {
                0.0
            } else {
                result.matched_char_count as f32 / result.input_char_count as f32
            };
            let selected_idx = if match_rate > cache_threshold {
                let tenant_url: &str = &result.tenant;
                workers
                    .iter()
                    .position(|w| w.url() == tenant_url)
                    .filter(|&idx| workers[idx].is_healthy())
            } else {
                healthy_indices
                    .iter()
                    .min_by_key(|&&idx| workers[idx].load())
                    .copied()
            };

            (selected_idx, match_rate, result.tenant.to_string())
        };

        // Probe the full-history fallback at the configured threshold; when no
        // usable history text exists, select min-load directly so the session
        // key still gets exact-match semantics instead of a text probe.
        let probe_fallback_or_min_load =
            |fallback_text: Option<&str>| -> (Option<usize>, &'static str, Option<String>) {
                match fallback_text {
                    Some(fallback_text) => {
                        let (fallback_idx, fallback_match_rate, fallback_tenant) =
                            probe(fallback_text, self.config.cache_threshold);
                        if fallback_match_rate > self.config.cache_threshold {
                            if let Some(idx) = fallback_idx {
                                (Some(idx), "full_history_match", None)
                            } else {
                                (
                                    healthy_indices.first().copied(),
                                    "stale_tenant_fallback",
                                    Some(fallback_tenant),
                                )
                            }
                        } else {
                            (fallback_idx, "full_history_low_match", None)
                        }
                    }
                    None => {
                        let min_load_idx = healthy_indices
                            .iter()
                            .min_by_key(|&&idx| workers[idx].load())
                            .copied();
                        (min_load_idx, "empty_history_min_load", None)
                    }
                }
            };

        let (selected_idx, decision, stale_tenant) = if session_keyed {
            // Exact-match session lookup: the session key never enters the
            // prefix tree, so session ids cannot prefix-collide.
            let session_map = session_map
                .as_ref()
                .expect("session map must exist when session_keyed");
            let cached = session_map
                .get(primary_text)
                .map(|entry| entry.value().clone());
            match cached {
                Some((worker_url, _last_access)) => {
                    match workers
                        .iter()
                        .position(|w| w.url() == worker_url)
                        .filter(|&idx| workers[idx].is_healthy())
                    {
                        Some(idx) => (Some(idx), "session_id_match", None),
                        None => {
                            // Stale worker for this session: drop the entry and
                            // fall back to the history probe.
                            session_map.remove(primary_text);
                            debug!("Removed stale session entry for worker {}", worker_url);
                            probe_fallback_or_min_load(fallback_text)
                        }
                    }
                }
                None => probe_fallback_or_min_load(fallback_text),
            }
        } else {
            let (idx, match_rate, tenant) = probe(primary_text, self.config.cache_threshold);
            let decision = if match_rate > self.config.cache_threshold {
                "cache_affinity"
            } else {
                "low_match_min_load"
            };
            let stale_tenant = if idx.is_none() && match_rate > self.config.cache_threshold {
                Some(tenant)
            } else {
                None
            };

            (idx, decision, stale_tenant)
        };

        if let Some(tenant) = stale_tenant {
            tree.remove_tenant(&tenant);
            debug!("Removed stale worker {} from cache tree", tenant);
        }

        if let Some(idx) = selected_idx {
            RouterMetrics::record_cache_aware_decision(decision);

            let worker_url = workers[idx].url();
            if session_keyed {
                // The session map is authoritative for the session key; the
                // prefix tree only learns text-like history keys. Teaching the
                // history on a session hit keeps the grown conversation prefix
                // available for clients that later drop the session id.
                session_map
                    .as_ref()
                    .expect("session map must exist when session_keyed")
                    .insert(
                        primary_text.to_string(),
                        (worker_url.to_string(), session_now_ms()),
                    );
                if let Some(history_text) = fallback_text {
                    tree.insert(history_text, worker_url);
                }
            } else {
                tree.insert(primary_text, worker_url);
            }

            workers[idx].increment_processed();
            RouterMetrics::record_processed_request(workers[idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[idx].url());

            return Some(RoutingSelection {
                index: idx,
                decision: Some(decision),
            });
        }

        if decision == "stale_tenant_fallback" {
            RouterMetrics::record_cache_aware_decision("stale_tenant_fallback");
        } else {
            RouterMetrics::record_cache_aware_decision("first_healthy_fallback");
        }
        if let Some(idx) = healthy_indices.first().copied() {
            workers[idx].increment_processed();
            RouterMetrics::record_processed_request(workers[idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[idx].url());

            Some(RoutingSelection {
                index: idx,
                decision: Some(if decision == "stale_tenant_fallback" {
                    "stale_tenant_fallback"
                } else {
                    "first_healthy_fallback"
                }),
            })
        } else {
            None
        }
    }

    fn select_worker_cache_aware(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        fallback_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        self.select_worker_cache_aware_with_decision(workers, request_text, fallback_text, headers)
            .map(|selection| selection.index)
    }
}

impl LoadBalancingPolicy for CacheAwarePolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        self.select_worker_cache_aware(workers, request_text, None, headers)
    }

    fn select_worker_with_fallback_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        fallback_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        self.select_worker_cache_aware(workers, request_text, fallback_text, headers)
    }

    fn select_worker_with_fallback_headers_with_decision(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        fallback_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<RoutingSelection> {
        self.select_worker_cache_aware_with_decision(workers, request_text, fallback_text, headers)
    }

    fn name(&self) -> &'static str {
        "cache_aware"
    }

    fn needs_request_text(&self) -> bool {
        true // Cache-aware policy needs request text for cache affinity
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        // Could track success rates per worker for more intelligent routing
        if !success {
            // Optionally reduce affinity for failed requests
            tracing::debug!(
                "Request to {} completed with success={}",
                worker_url,
                success
            );
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn select_worker_pair_with_headers(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<(usize, usize)> {
        // DEPRECATED: This method is no longer used when separate policies are configured.
        // The PD router now uses separate policies for prefill and decode selection.
        // This implementation remains for backward compatibility when a single policy is used.

        // In PD mode with single policy:
        // - Prefill: Use cache-aware routing for better cache utilization
        // - Decode: Use least-load routing for better load distribution

        // Select prefill worker using cache-aware logic
        let prefill_idx =
            self.select_worker_with_headers(prefill_workers, request_text, headers)?;

        // Select decode worker using least-load logic
        let healthy_decode = get_healthy_worker_indices(decode_workers);
        if healthy_decode.is_empty() {
            return None;
        }

        let decode_idx = healthy_decode
            .iter()
            .min_by_key(|&&idx| decode_workers[idx].load())
            .copied()?;

        Some((prefill_idx, decode_idx))
    }

    fn requires_initialization(&self) -> bool {
        true // Cache-aware policy requires init_workers() to set up trees
    }

    fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        // Group workers by model
        info!(
            "Initializing workers for cache-aware policy: {}",
            workers
                .iter()
                .map(|w| w.url())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let mut model_workers: HashMap<String, Vec<&Arc<dyn Worker>>> = HashMap::new();
        for worker in workers {
            let tree_key = normalize_model_key(worker.model_id());
            model_workers
                .entry(tree_key.to_string())
                .or_default()
                .push(worker);
        }

        // Initialize tree for each model
        for (tree_key, model_workers) in model_workers {
            info!(
                "Creating tree for model key: '{}' with {} workers",
                tree_key,
                model_workers.len()
            );
            let tree = self
                .trees
                .entry(tree_key)
                .or_insert_with(|| Arc::new(Tree::new()))
                .clone();
            for worker in model_workers {
                tree.insert("", worker.url());
            }
        }
    }
}

impl Default for CacheAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CacheAwarePolicy {
    fn drop(&mut self) {
        // Note: We can't properly stop the eviction thread since it's in an infinite loop
        // In a production system, we'd use a channel or atomic flag to signal shutdown
        if let Some(handle) = self.eviction_handle.take() {
            // The thread will continue running until the program exits
            // This is acceptable for now since the router typically runs for the lifetime of the program
            drop(handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    #[test]
    fn test_cache_aware_with_balanced_load() {
        // Create policy without eviction thread for testing
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        // Initialize the policy with workers
        policy.init_workers(&workers);

        // First request should be distributed
        let idx1 = policy.select_worker(&workers, Some("hello world")).unwrap();

        // Same request should go to same worker (cache hit)
        let idx2 = policy.select_worker(&workers, Some("hello world")).unwrap();
        assert_eq!(idx1, idx2);

        // Similar request should also go to same worker
        let idx3 = policy.select_worker(&workers, Some("hello")).unwrap();
        assert_eq!(idx1, idx3);
    }

    #[test]
    fn test_cache_aware_falls_back_to_session_header_when_text_empty() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];
        policy.init_workers(&workers);

        let mut headers = RequestHeaders::new();
        headers.insert("x-session-id".to_string(), "agent-session-1".to_string());

        let idx1 = policy
            .select_worker_with_headers(&workers, Some(""), Some(&headers))
            .unwrap();
        let idx2 = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();

        assert_eq!(idx1, idx2);
    }

    #[test]
    fn test_cache_aware_header_session_key_uses_exact_match_without_fallback() {
        // A session-derived header key (x-session-id) on a request without a
        // fallback key must use exact-match session semantics: repeated
        // requests stick to one worker and prefix-related header values do
        // not collide.
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];
        policy.init_workers(&workers);

        let mut headers_a = RequestHeaders::new();
        headers_a.insert("x-session-id".to_string(), "agent-session-1".to_string());

        let idx1 = policy
            .select_worker_with_headers(&workers, Some(""), Some(&headers_a))
            .unwrap();

        // Same header again: exact hit on the session map.
        let sel2 = policy
            .select_worker_with_fallback_headers_with_decision(
                &workers,
                Some(""),
                None,
                Some(&headers_a),
            )
            .unwrap();
        assert_eq!(sel2.index, idx1);
        assert_eq!(sel2.decision, Some("session_id_match"));

        // A prefix-related header value is a different session under exact
        // equality: it must not ride on agent-session-1's worker.
        workers[idx1].increment_load();
        let mut headers_b = RequestHeaders::new();
        headers_b.insert("x-session-id".to_string(), "agent-session-12".to_string());
        let sel3 = policy
            .select_worker_with_fallback_headers_with_decision(
                &workers,
                Some(""),
                None,
                Some(&headers_b),
            )
            .unwrap();
        assert_ne!(sel3.index, idx1);
    }

    #[test]
    fn test_cache_aware_text_key_keeps_prefix_semantics_with_headers_present() {
        // Non-chat text keys keep prefix-tree semantics even when a session
        // header is present: the explicit routing text wins, so prompt
        // prefix affinity is not replaced by exact matching.
        let config = CacheAwareConfig {
            cache_threshold: 0.3,
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];
        policy.init_workers(&workers);

        let mut headers = RequestHeaders::new();
        headers.insert("x-session-id".to_string(), "some-session".to_string());

        let shared_prefix = "system:You are a coding agent\ntool:read_file";
        let idx1 = policy
            .select_worker_with_headers(&workers, Some(shared_prefix), Some(&headers))
            .unwrap();

        // Same prefix, different suffix, different header: prefix affinity
        // still routes to the same worker (prefix-tree semantics, not exact
        // matching on the full text).
        let mut headers_b = RequestHeaders::new();
        headers_b.insert("x-session-id".to_string(), "another-session".to_string());
        let idx2 = policy
            .select_worker_with_headers(
                &workers,
                Some(&format!("{shared_prefix}\nuser:different suffix")),
                Some(&headers_b),
            )
            .unwrap();
        assert_eq!(idx1, idx2);
    }

    #[test]
    fn test_cache_aware_stable_text_precedes_session_header() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];
        policy.init_workers(&workers);

        let mut headers_a = RequestHeaders::new();
        headers_a.insert("x-session-id".to_string(), "session-a".to_string());
        let mut headers_b = RequestHeaders::new();
        headers_b.insert("x-session-id".to_string(), "session-b".to_string());

        let stable_agent_prefix = "system:You are a coding agent\ntool:read_file";
        let idx1 = policy
            .select_worker_with_headers(&workers, Some(stable_agent_prefix), Some(&headers_a))
            .unwrap();
        let idx2 = policy
            .select_worker_with_headers(&workers, Some(stable_agent_prefix), Some(&headers_b))
            .unwrap();

        assert_eq!(idx1, idx2);
    }

    #[test]
    fn test_cache_aware_session_id_falls_back_to_full_history() {
        let config = CacheAwareConfig {
            cache_threshold: 0.3,
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];
        policy.init_workers(&workers);

        let shared_history = "system:bench\nuser:shared long prefix";
        let idx1 = policy
            .select_worker_with_fallback_headers(
                &workers,
                Some("session-a"),
                Some(shared_history),
                None,
            )
            .unwrap();

        let idx2 = policy
            .select_worker_with_fallback_headers(
                &workers,
                Some("session-b"),
                Some(shared_history),
                None,
            )
            .unwrap();
        assert_eq!(idx1, idx2);

        let idx3 = policy
            .select_worker_with_fallback_headers(
                &workers,
                Some("session-a"),
                Some("system:bench\nuser:different prefix"),
                None,
            )
            .unwrap();
        assert_eq!(idx1, idx3);
    }

    #[test]
    fn test_cache_aware_fallback_uses_exact_session_match() {
        let config = CacheAwareConfig {
            cache_threshold: 0.3,
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];
        policy.init_workers(&workers);

        let idx1 = policy
            .select_worker_with_fallback_headers(
                &workers,
                Some("session-1"),
                Some("system:first unrelated prefix"),
                None,
            )
            .unwrap();
        assert_eq!(idx1, 0);

        workers[0].increment_load();

        // session-12 is a different key under exact equality even though it
        // shares a long string prefix with session-1.
        let idx2 = policy
            .select_worker_with_fallback_headers(
                &workers,
                Some("session-12"),
                Some("system:second unrelated prefix"),
                None,
            )
            .unwrap();

        assert_eq!(idx2, 1);
    }

    #[test]
    fn test_cache_aware_session_hit_records_full_history() {
        // On a session hit the tree must also learn the current full
        // history: the client may later drop the session id, and the tree
        // must know the grown conversation prefix the worker cached, not
        // the shorter one from an earlier turn.
        let config = CacheAwareConfig {
            cache_threshold: 0.5,
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];
        policy.init_workers(&workers);

        let short_history = "system:bench\nuser:turn one";
        let long_history = format!(
            "{}\nassistant:{}\nuser:turn two",
            short_history,
            "x".repeat(150)
        );

        // Turn 1: session-a learns the short history.
        let idx1 = policy
            .select_worker_with_fallback_headers(
                &workers,
                Some("session-a"),
                Some(short_history),
                None,
            )
            .unwrap();

        // Turn 2: exact session hit, with a much longer history.
        let idx2 = policy
            .select_worker_with_fallback_headers(
                &workers,
                Some("session-a"),
                Some(&long_history),
                None,
            )
            .unwrap();
        assert_eq!(idx1, idx2);

        // A request without a session id must still match the grown
        // history (0.5 threshold): the stale short history alone would
        // only match ~13% of the long key and fall to min-load.
        workers[0].increment_load();
        let idx3 = policy
            .select_worker_with_fallback_headers(&workers, Some(""), Some(&long_history), None)
            .unwrap();
        assert_eq!(idx3, idx1, "grown history must be learned on session hit");
    }

    #[test]
    fn test_cache_aware_empty_history_uses_exact_session_match() {
        // A session key with no extractable history (e.g. image-only chat)
        // is matched by exact equality against the session map: session-12
        // never collides with session-1 no matter how long their shared
        // string prefix is.
        let config = CacheAwareConfig {
            cache_threshold: 0.3,
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];
        policy.init_workers(&workers);

        let sel1 = policy
            .select_worker_with_fallback_headers_with_decision(
                &workers,
                Some("session-1"),
                Some(""),
                None,
            )
            .unwrap();
        assert_eq!(sel1.index, 0);
        assert_eq!(sel1.decision, Some("empty_history_min_load"));

        workers[0].increment_load();

        let sel2 = policy
            .select_worker_with_fallback_headers_with_decision(
                &workers,
                Some("session-12"),
                Some(""),
                None,
            )
            .unwrap();
        // Not the session-1 affinity: exact equality says this is a
        // different session, so fall back to min-load.
        assert_eq!(sel2.index, 1);
        assert_eq!(sel2.decision, Some("empty_history_min_load"));
    }

    #[test]
    fn test_cache_aware_session_id_match_is_exact() {
        let config = CacheAwareConfig {
            cache_threshold: 0.3,
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];
        policy.init_workers(&workers);

        let sel1 = policy
            .select_worker_with_fallback_headers_with_decision(
                &workers,
                Some("session-1"),
                Some("system:first prefix"),
                None,
            )
            .unwrap();
        workers[0].increment_load();

        // Exact key: session affinity wins over the min-load worker even
        // though worker 0 is busier (loads are still balanced).
        let sel2 = policy
            .select_worker_with_fallback_headers_with_decision(
                &workers,
                Some("session-1"),
                Some("system:first prefix with more text"),
                None,
            )
            .unwrap();
        assert_eq!(sel2.index, sel1.index);
        assert_eq!(sel2.decision, Some("session_id_match"));
    }

    #[test]
    fn test_cache_aware_removed_worker_clears_session_entries() {
        // remove_worker must drop the worker's session entries eagerly so a
        // session can never exact-hit a worker that was removed from the
        // policy.
        let config = CacheAwareConfig {
            cache_threshold: 0.5,
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];
        policy.init_workers(&workers);

        let history = "system:bench\nuser:hello";
        let idx1 = policy
            .select_worker_with_fallback_headers(&workers, Some("session-a"), Some(history), None)
            .unwrap();
        assert_eq!(idx1, 0);

        policy.remove_worker(workers[idx1].as_ref());
        workers[0].increment_load();

        // The session entry for the removed worker must be gone, so
        // session-a cannot exact-hit it; with the history tenant also
        // removed, the request lands on min-load (worker 2).
        let idx2 = policy
            .select_worker_with_fallback_headers(&workers, Some("session-a"), Some(history), None)
            .unwrap();
        assert_ne!(idx2, idx1);
    }

    #[test]
    fn test_evict_session_entries_ttl_and_capacity() {
        let map: SessionMap = DashMap::new();
        let now = session_now_ms();
        map.insert("fresh".to_string(), ("w1".to_string(), now));
        map.insert(
            "stale".to_string(),
            ("w1".to_string(), now - (SESSION_ENTRY_TTL_SECS * 1000) - 1),
        );
        map.insert("oldest-fresh".to_string(), ("w1".to_string(), now - 1000));
        map.insert("newest".to_string(), ("w2".to_string(), now));

        evict_session_entries(&map, 2);

        // TTL removes the stale entry; capacity then keeps the two most
        // recently accessed entries.
        assert!(map.contains_key("fresh"));
        assert!(map.contains_key("newest"));
        assert!(!map.contains_key("stale"));
        assert!(!map.contains_key("oldest-fresh"));
    }

    #[test]
    fn test_cache_aware_imbalanced_path_teaches_full_history() {
        let config = CacheAwareConfig {
            cache_threshold: 0.3,
            balance_abs_threshold: 1,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];
        policy.init_workers(&workers);

        // Force an imbalance so the selection takes the min-load path.
        workers[0].increment_load();
        workers[0].increment_load();

        let shared_history = "system:bench\nuser:shared long prefix";
        let idx1 = policy
            .select_worker_with_fallback_headers(
                &workers,
                Some("session-a"),
                Some(shared_history),
                None,
            )
            .unwrap();
        assert_eq!(idx1, 1, "imbalanced selection must pick min-load worker");

        // Rebalance. The full history must have been learned during the
        // imbalanced request, so a new session with the same prefix is
        // attracted to the same worker.
        workers[0].decrement_load();
        workers[0].decrement_load();

        let idx2 = policy
            .select_worker_with_fallback_headers(
                &workers,
                Some("session-b"),
                Some(shared_history),
                None,
            )
            .unwrap();
        assert_eq!(idx2, idx1, "learned history must attract the new session");
    }

    #[test]
    fn test_cache_aware_with_imbalanced_load() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 0, // Disable eviction thread
            max_tree_size: 10000,
        });

        let worker1 = BasicWorker::new("http://w1:8000".to_string(), WorkerType::Regular);
        let worker2 = BasicWorker::new("http://w2:8000".to_string(), WorkerType::Regular);

        // Create significant load imbalance
        for _ in 0..20 {
            worker1.increment_load();
        }
        // worker2 has load 0

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker1), Arc::new(worker2)];
        policy.init_workers(&workers);

        // Should select worker2 (lower load) despite cache affinity
        for _ in 0..5 {
            let idx = policy.select_worker(&workers, Some("test")).unwrap();
            assert_eq!(idx, 1); // Should always pick worker2
        }
    }

    #[test]
    fn test_cache_aware_worker_removal() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        policy.init_workers(&workers);

        // Route some requests
        policy.select_worker(&workers, Some("test1"));
        policy.select_worker(&workers, Some("test2"));

        // Remove a worker
        policy.remove_worker_by_url("http://w1:8000");
        workers[0].set_healthy(false);

        // All requests should now go to worker2
        let idx = policy.select_worker(&workers, Some("test1")).unwrap();
        assert_eq!(idx, 1);
    }
}
