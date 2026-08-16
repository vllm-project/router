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

use super::hash_key::extract_hash_key_from_headers;
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
use std::time::Duration;
use tracing::{debug, info};

const FALLBACK_SESSION_CACHE_THRESHOLD: f32 = 0.999;

/// Cache-aware routing policy
///
/// Routes requests based on cache affinity when load is balanced,
/// switches to shortest-queue routing when load is imbalanced.
/// Maintains separate trees per model for multi-model support.
#[derive(Debug)]
pub struct CacheAwarePolicy {
    config: CacheAwareConfig,
    trees: Arc<DashMap<String, Arc<Tree>>>, // model_id -> Arc<Tree>
    eviction_handle: Option<thread::JoinHandle<()>>,
}

impl CacheAwarePolicy {
    pub fn new() -> Self {
        Self::with_config(CacheAwareConfig::default())
    }

    pub fn with_config(config: CacheAwareConfig) -> Self {
        let trees = Arc::new(DashMap::<String, Arc<Tree>>::new());

        // Start background eviction thread if configured
        let eviction_handle = if config.eviction_interval_secs > 0 {
            let trees_clone = Arc::clone(&trees);
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
            }))
        } else {
            None
        };

        Self {
            config,
            trees,
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
    }

    /// Add a worker by URL and model (for backward compatibility)
    pub fn add_worker_by_url(&self, url: &str, model_id: &str) {
        let tree = self
            .trees
            .entry(model_id.to_string())
            .or_insert_with(|| Arc::new(Tree::new()));
        tree.insert("", url);
    }

    /// Remove a worker from the tree
    pub fn remove_worker(&self, worker: &dyn Worker) {
        let tree_key = normalize_model_key(worker.model_id());
        if let Some(tree) = self.trees.get(tree_key) {
            tree.remove_tenant(worker.url());
        }
    }

    /// Remove a worker by URL (removes from all model trees for backward compatibility)
    pub fn remove_worker_by_url(&self, url: &str) {
        // Remove from all trees since we don't know which model it belongs to
        for tree_ref in self.trees.iter() {
            tree_ref.value().remove_tenant(url);
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

        // Even in imbalanced mode, update the tree to maintain cache state.
        // Teach it the session key and the full chat history: the worker has
        // just processed and cached the whole prompt, so a later balanced
        // request sharing the same prefix must be able to discover it.
        let tree = self.trees.get(model_id).map(|entry| entry.value().clone());

        if let Some(tree) = tree {
            let worker_url = workers[min_load_idx].url();
            for text in [request_text, fallback_text].into_iter().flatten() {
                if !text.is_empty() {
                    tree.insert(text, worker_url);
                }
            }
        } else {
            debug!(
                "Warning: No tree found for model '{}', skipping cache update",
                model_id
            );
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

        // Use explicit routing text first; fall back to session headers for
        // clients that carry affinity only in HTTP metadata.
        let header_key = headers.and_then(extract_hash_key_from_headers);
        let primary_text = request_text
            .filter(|text| !text.trim().is_empty())
            .or(header_key.as_deref())
            .unwrap_or("");
        // A chat request may carry a session key with no extractable history
        // (e.g. an image-only user message). Keep the two-stage session
        // semantics in that case: probe the session key at the strict 0.999
        // threshold and fall back to min-load instead of probing the empty
        // history at cache_threshold, which would let prefix-related session
        // ids (session-1\x1f vs session-12\x1f) pass as cache hits.
        let fallback_provided = fallback_text.is_some();
        let fallback_text = fallback_text.filter(|text| !text.trim().is_empty());
        let use_fallback_probe = fallback_provided && !primary_text.is_empty();
        let primary_text = if primary_text.is_empty() {
            fallback_text.unwrap_or("")
        } else {
            primary_text
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
        // usable history text exists, select min-load directly so the strict
        // session semantics (0.999) still apply to the session key itself.
        let probe_fallback_or_min_load =
            |fallback_text: Option<&str>| -> (Option<usize>, &'static str, Option<String>, bool) {
                match fallback_text {
                    Some(fallback_text) => {
                        let (fallback_idx, fallback_match_rate, fallback_tenant) =
                            probe(fallback_text, self.config.cache_threshold);
                        if fallback_match_rate > self.config.cache_threshold {
                            if let Some(idx) = fallback_idx {
                                (Some(idx), "full_history_match", None, true)
                            } else {
                                (
                                    healthy_indices.first().copied(),
                                    "stale_tenant_fallback",
                                    Some(fallback_tenant),
                                    true,
                                )
                            }
                        } else {
                            (fallback_idx, "full_history_low_match", None, true)
                        }
                    }
                    None => {
                        let min_load_idx = healthy_indices
                            .iter()
                            .min_by_key(|&&idx| workers[idx].load())
                            .copied();
                        (min_load_idx, "empty_history_min_load", None, false)
                    }
                }
            };

        let (selected_idx, decision, stale_tenant, used_fallback) = if use_fallback_probe {
            let (primary_idx, primary_match_rate, primary_tenant) =
                probe(primary_text, FALLBACK_SESSION_CACHE_THRESHOLD);

            let session_hit = primary_match_rate > FALLBACK_SESSION_CACHE_THRESHOLD;
            if session_hit {
                if let Some(idx) = primary_idx {
                    (Some(idx), "session_id_match", None, false)
                } else {
                    // Stale worker for the session key; drop it and probe
                    // the history.
                    tree.remove_tenant(&primary_tenant);
                    debug!("Removed stale worker {} from cache tree", primary_tenant);
                    RouterMetrics::record_cache_aware_decision("session_id_fallback");
                    probe_fallback_or_min_load(fallback_text)
                }
            } else {
                RouterMetrics::record_cache_aware_decision("session_id_fallback");
                probe_fallback_or_min_load(fallback_text)
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

            (idx, decision, stale_tenant, false)
        };

        let selected_text = if used_fallback {
            fallback_text.unwrap_or(primary_text)
        } else {
            primary_text
        };

        if let Some(tenant) = stale_tenant {
            tree.remove_tenant(&tenant);
            debug!("Removed stale worker {} from cache tree", tenant);
        }

        if let Some(idx) = selected_idx {
            RouterMetrics::record_cache_aware_decision(decision);

            let worker_url = workers[idx].url();
            tree.insert(selected_text, worker_url);

            if used_fallback && !primary_text.is_empty() && primary_text != selected_text {
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
    fn test_cache_aware_fallback_uses_strict_session_threshold() {
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
                Some("session-1\x1f"),
                Some("system:first unrelated prefix"),
                None,
            )
            .unwrap();
        assert_eq!(idx1, 0);

        workers[0].increment_load();

        let idx2 = policy
            .select_worker_with_fallback_headers(
                &workers,
                Some("session-12\x1f"),
                Some("system:second unrelated prefix"),
                None,
            )
            .unwrap();

        assert_eq!(idx2, 1);
    }

    #[test]
    fn test_cache_aware_empty_history_keeps_strict_session_threshold() {
        // A session key with no extractable history (e.g. image-only chat)
        // must still be probed at the strict 0.999 threshold. session-1\x1f
        // and session-12\x1f share 80% of their chars, which would pass the
        // configured 0.3 threshold and falsely count as a session hit.
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
                Some("session-1\x1f"),
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
                Some("session-12\x1f"),
                Some(""),
                None,
            )
            .unwrap();
        // Not a session hit (0.8 < 0.999): must fall back to min-load
        // instead of following the session-1 affinity.
        assert_eq!(sel2.index, 1);
        assert_eq!(sel2.decision, Some("empty_history_min_load"));
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
                Some("session-a\x1f"),
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
                Some("session-b\x1f"),
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
