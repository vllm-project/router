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

use super::{get_healthy_worker_indices, CacheAwareConfig, LoadBalancingPolicy, RequestHeaders};
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

    /// Is this specific worker hot enough that a request should be steered away
    /// from it, even at the cost of a cache miss?
    ///
    /// Same thresholds as before, but the comparison is `this worker` against the
    /// least-loaded worker, not `the busiest worker` against the least-loaded one.
    /// A worker that is merely busier than its peers keeps serving its cached
    /// prefixes; only one that is genuinely running away gives them up.
    fn is_worker_overloaded(&self, worker_load: usize, min_load: usize) -> bool {
        worker_load.saturating_sub(min_load) > self.config.balance_abs_threshold
            && (worker_load as f32) > (min_load as f32 * self.config.balance_rel_threshold)
    }
}

impl LoadBalancingPolicy for CacheAwarePolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Determine the model for this set of workers (router pre-filters by model)
        // All workers should be from the same model
        let model_id = normalize_model_key(workers[healthy_indices[0]].model_id());

        // Get current load statistics - compute min/max in single pass without allocation
        let (min_load, max_load) = workers.iter().fold((usize::MAX, 0usize), |(min, max), w| {
            let load = w.load();
            (min.min(load), max.max(load))
        });
        let min_load = if min_load == usize::MAX { 0 } else { min_load };

        debug!(
            "Load status for model: max_load={}, min_load={}",
            max_load, min_load
        );

        // NOTE: the load check is applied per request, against the worker this
        // request actually wants, rather than fleet-wide. The previous behaviour
        // asked "is any pair of workers imbalanced?" and, if so, discarded cache
        // affinity for *every* request -- including requests whose preferred
        // worker was idle. Under prefill/decode disaggregation that is
        // catastrophic: prefill worker load includes queued requests, so at high
        // concurrency the fleet spread clears the threshold almost permanently,
        // routing degenerates to shortest-queue, and the prefix cache stops being
        // used at all. Measured on SWE-bench Pro (4 prefill / 6 decode, 1024 in
        // parallel), that took the prefill prefix cache hit rate from 88.9% to
        // 68.5% -- a 2.8x increase in prompt tokens actually computed -- and
        // halved end-to-end throughput.
        //
        // Shedding load only matters for the worker that is actually hot, so that
        // is the only case where affinity is given up. See is_worker_overloaded.

        // Use cache-aware routing
        let text = request_text.unwrap_or("");

        // Get the tree reference without locking the entire HashMap
        // DashMap only locks the specific shard containing this key
        let tree = self.trees.get(model_id).map(|entry| entry.value().clone());

        let keys: Vec<_> = self.trees.iter().map(|entry| entry.key().clone()).collect();
        debug!("Available tree keys: {:?}", keys);

        let Some(tree) = tree else {
            // No tree for this model, log warning and use random selection
            debug!(
                "Warning: No tree found for model '{}', using random worker selection",
                model_id
            );
            // Return a random healthy worker
            let mut rng = rand::rng();
            let random_idx = rng.random_range(0..healthy_indices.len());
            let selected_idx = healthy_indices[random_idx];

            workers[selected_idx].increment_processed();
            RouterMetrics::record_processed_request(workers[selected_idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[selected_idx].url());

            return Some(selected_idx);
        };
        debug!("Using cache-aware routing for model '{}'", model_id);
        // Now we work with the tree without holding the HashMap lock
        // Use prefix_match_with_counts to avoid redundant chars().count() calls
        let result = tree.prefix_match_with_counts(text);
        let match_rate = if result.input_char_count == 0 {
            0.0
        } else {
            result.matched_char_count as f32 / result.input_char_count as f32
        };

        debug!(
            "Cache match for model '{}': matched_chars={}, input_chars={}, match_rate={:.2}",
            model_id, result.matched_char_count, result.input_char_count, match_rate
        );
        // Select worker without String allocation
        let selected_idx = if match_rate > self.config.cache_threshold {
            // Cache hit path: find worker by URL (compare &str directly, no allocation)
            let tenant_url: &str = &result.tenant;
            let cached_idx = workers
                .iter()
                .position(|w| w.url() == tenant_url)
                .filter(|&idx| workers[idx].is_healthy());

            match cached_idx {
                // The worker holding this prefix is running away from the rest of
                // the fleet, so pay the cache miss and shed the load. This is the
                // hot-spot case that load balancing exists for.
                Some(idx) if self.is_worker_overloaded(workers[idx].load(), min_load) => {
                    RouterMetrics::record_load_balancing_event();
                    RouterMetrics::set_load_range(max_load, min_load);
                    debug!(
                        "Cached worker {} overloaded (load={}, min={}), steering away",
                        workers[idx].url(),
                        workers[idx].load(),
                        min_load
                    );
                    healthy_indices
                        .iter()
                        .min_by_key(|&&idx| workers[idx].load())
                        .copied()
                }
                other => other,
            }
        } else {
            // Low cache match: use worker with minimum load
            healthy_indices
                .iter()
                .min_by_key(|&&idx| workers[idx].load())
                .copied()
        };

        if let Some(idx) = selected_idx {
            // Update the tree with this request (use worker URL directly, no allocation)
            tree.insert(text, workers[idx].url());

            // Increment processed counter
            workers[idx].increment_processed();
            RouterMetrics::record_processed_request(workers[idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[idx].url());

            return Some(idx);
        }

        // Selected worker no longer exists or unhealthy, remove stale tenant from tree
        if match_rate > self.config.cache_threshold {
            let tenant_url: &str = &result.tenant;
            tree.remove_tenant(tenant_url);
            debug!("Removed stale worker {} from cache tree", tenant_url);
        }

        // Fallback to first healthy worker
        if let Some(idx) = healthy_indices.first().copied() {
            workers[idx].increment_processed();
            RouterMetrics::record_processed_request(workers[idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[idx].url());

            Some(idx)
        } else {
            None
        }
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

    /// Helper: pin a worker's load counter to an exact value.
    fn set_load(worker: &Arc<dyn Worker>, n: usize) {
        worker.reset_load();
        for _ in 0..n {
            worker.increment_load();
        }
    }

    fn three_workers() -> Vec<Arc<dyn Worker>> {
        vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w3:8000".to_string(),
                WorkerType::Regular,
            )),
        ]
    }

    fn balance_policy() -> CacheAwarePolicy {
        CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 0,
            max_tree_size: 10000,
        })
    }

    /// Regression test: a hot worker elsewhere in the fleet must not cost cache
    /// affinity for requests that want an idle worker.
    ///
    /// The previous implementation gated on `max_load - min_load` across the whole
    /// fleet, so one runaway worker disabled prefix routing for every request. Under
    /// P/D disaggregation, where prefill load counters include queued requests, that
    /// gate is essentially always open at high concurrency and the prefix cache stops
    /// being used at all.
    #[test]
    fn test_affinity_survives_unrelated_hot_worker() {
        let policy = balance_policy();
        let workers = three_workers();
        policy.init_workers(&workers);

        // Make w3 strictly the least loaded so the first request tenants to it.
        set_load(&workers[0], 2);
        set_load(&workers[1], 1);
        set_load(&workers[2], 0);
        let primed = policy
            .select_worker(&workers, Some("conversation-A"))
            .unwrap();
        assert_eq!(primed, 2, "setup: first request should tenant to w3");

        // Now w1 runs away. Fleet is wildly imbalanced (100 vs 0), which the old
        // fleet-wide gate would have treated as "ignore the cache entirely".
        // w3 -- the worker actually holding this prefix -- stays cold.
        set_load(&workers[0], 100);
        set_load(&workers[1], 0);
        set_load(&workers[2], 1);

        for _ in 0..5 {
            let idx = policy
                .select_worker(&workers, Some("conversation-A"))
                .unwrap();
            assert_eq!(
                idx, 2,
                "affinity must be preserved: the cached worker w3 is not the hot one"
            );
        }
    }

    /// The other half of the contract: when the worker holding the prefix is itself
    /// the runaway, the cache miss is worth paying and the request is steered away.
    /// This is the hot-spot case load balancing exists for.
    #[test]
    fn test_affinity_yields_when_cached_worker_is_the_hot_one() {
        let policy = balance_policy();
        let workers = three_workers();
        policy.init_workers(&workers);

        // Tenant "conversation-B" to w1.
        set_load(&workers[0], 0);
        set_load(&workers[1], 1);
        set_load(&workers[2], 2);
        let primed = policy
            .select_worker(&workers, Some("conversation-B"))
            .unwrap();
        assert_eq!(primed, 0, "setup: first request should tenant to w1");

        // w1 is now the runaway and holds the prefix.
        set_load(&workers[0], 100);
        set_load(&workers[1], 0);
        set_load(&workers[2], 0);

        for _ in 0..5 {
            let idx = policy
                .select_worker(&workers, Some("conversation-B"))
                .unwrap();
            assert_ne!(idx, 0, "must steer away from the overloaded cached worker");
        }
    }

    /// A worker that is merely busier than its peers keeps serving its cached
    /// prefixes -- the override is for runaways, not for ordinary variation.
    #[test]
    fn test_affinity_survives_mild_load_difference() {
        let policy = balance_policy();
        let workers = three_workers();
        policy.init_workers(&workers);

        set_load(&workers[0], 2);
        set_load(&workers[1], 1);
        set_load(&workers[2], 0);
        let primed = policy
            .select_worker(&workers, Some("conversation-C"))
            .unwrap();
        assert_eq!(primed, 2);

        // w3 is busier than the others but well inside balance_abs_threshold (5).
        set_load(&workers[0], 0);
        set_load(&workers[1], 0);
        set_load(&workers[2], 4);

        let idx = policy
            .select_worker(&workers, Some("conversation-C"))
            .unwrap();
        assert_eq!(idx, 2, "a 4-request lead is not a hot spot");
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
