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

use super::{
    get_healthy_worker_indices, CacheAwareConfig, LoadBalancingPolicy, RequestHeaders,
    SelectionContext,
};
use crate::core::{Worker, WorkerLoadGuard};
use crate::metrics::RouterMetrics;
use crate::policies::normalize_model_key;
use crate::tree::{PrefixMatchResult, Tree};
use dashmap::DashMap;
use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tracing::{debug, info};

/// Backend candidate for a side-effect-free cache-affinity query.
#[derive(Debug, Clone, Copy)]
pub struct CacheAwareCandidate<'a> {
    pub target_id: &'a str,
    pub kv_pressure: f64,
}

/// Cache-affinity result committed separately by the caller.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheAwarePlacement {
    pub target_id: String,
    pub match_rate: f32,
}

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

    /// Select by the product of text-prefix affinity and current KV headroom.
    pub fn evaluate_initial_placement(
        &self,
        model_id: &str,
        request_text: &str,
        candidates: &[CacheAwareCandidate<'_>],
    ) -> Option<CacheAwarePlacement> {
        let tree = self.trees.get(normalize_model_key(model_id));
        let input_chars = request_text.chars().count();
        candidates
            .iter()
            .map(|candidate| {
                let matched_chars = tree.as_ref().map_or(0, |tree| {
                    tree.prefix_match_tenant_char_count(request_text, candidate.target_id)
                });
                let affinity = if input_chars == 0 {
                    0.0
                } else {
                    matched_chars as f64 / input_chars as f64
                };
                let headroom = 1.0 - candidate.kv_pressure.clamp(0.0, 1.0);
                (candidate, affinity, headroom, (1.0 + affinity) * headroom)
            })
            .max_by(|left, right| {
                left.3
                    .total_cmp(&right.3)
                    .then_with(|| left.2.total_cmp(&right.2))
                    .then_with(|| left.1.total_cmp(&right.1))
                    .then_with(|| right.0.target_id.cmp(left.0.target_id))
            })
            .map(|(candidate, affinity, _, _)| CacheAwarePlacement {
                target_id: candidate.target_id.to_string(),
                match_rate: affinity as f32,
            })
    }

    /// Update the affinity tree after the selected target is committed.
    pub fn commit_placement(&self, model_id: &str, request_text: &str, target_id: &str) {
        if request_text.is_empty() {
            return;
        }
        self.trees
            .entry(normalize_model_key(model_id).to_string())
            .or_insert_with(|| Arc::new(Tree::new()))
            .insert(request_text, target_id);
    }

    /// Remove one unavailable target from a model's affinity tree.
    pub fn remove_placement_target(&self, model_id: &str, target_id: &str) {
        if let Some(tree) = self.trees.get(normalize_model_key(model_id)) {
            tree.remove_tenant(target_id);
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

    fn select_worker_min_load(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        healthy_indices: &[usize],
        model_id: &str,
        max_load: usize,
        min_load: usize,
    ) -> Option<usize> {
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
        RouterMetrics::set_load_range(max_load, min_load);

        let min_load_idx = Self::least_loaded(workers, healthy_indices)?;

        // Even in imbalanced mode, update the tree to maintain cache state
        if let Some(text) = request_text {
            // Get the tree reference without locking the entire HashMap
            // DashMap only locks the specific shard containing this key
            let tree = self.trees.get(model_id).map(|entry| entry.value().clone());

            if let Some(tree) = tree {
                let worker_url = workers[min_load_idx].url();
                tree.insert(text, worker_url);
            } else {
                debug!(
                    "Warning: No tree found for model '{}', skipping cache update",
                    model_id
                );
            }
        }

        Self::record_selection(workers, min_load_idx);

        Some(min_load_idx)
    }

    fn load_range(&self, workers: &[Arc<dyn Worker>], healthy: &[usize]) -> (usize, usize) {
        // Use the same eligible workers for load statistics and selection.
        let (min, max) = healthy
            .iter()
            .fold((usize::MAX, 0usize), |(min, max), &idx| {
                let load = workers[idx].load();
                (min.min(load), max.max(load))
            });
        (if min == usize::MAX { 0 } else { min }, max)
    }

    fn is_imbalanced(&self, min: usize, max: usize) -> bool {
        max.saturating_sub(min) > self.config.balance_abs_threshold
            && (max as f32) > (min as f32 * self.config.balance_rel_threshold)
    }

    fn least_loaded(workers: &[Arc<dyn Worker>], healthy: &[usize]) -> Option<usize> {
        healthy
            .iter()
            .min_by_key(|&&idx| workers[idx].load())
            .copied()
    }

    /// Choose using an approximate affinity observation, but current eligibility
    /// and load. Called within the pool's admission critical section, when used.
    /// Returns whether the observed cache target has become unavailable.
    fn choose_from_affinity(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy: &[usize],
        affinity: Option<&PrefixMatchResult>,
    ) -> Option<(usize, bool)> {
        let Some(result) = affinity else {
            let idx = *healthy.get(rand::rng().random_range(0..healthy.len()))?;
            return Some((idx, false));
        };
        let match_rate = if result.input_char_count == 0 {
            0.0
        } else {
            result.matched_char_count as f32 / result.input_char_count as f32
        };
        if match_rate > self.config.cache_threshold {
            if let Some(idx) = workers
                .iter()
                .position(|worker| worker.url() == result.tenant.as_ref())
                .filter(|&idx| workers[idx].is_healthy())
            {
                return Some((idx, false));
            }
            return healthy.first().copied().map(|idx| (idx, true));
        }
        Self::least_loaded(workers, healthy).map(|idx| (idx, false))
    }

    fn record_selection(workers: &[Arc<dyn Worker>], idx: usize) {
        workers[idx].increment_processed();
        RouterMetrics::record_processed_request(workers[idx].url());
        RouterMetrics::record_policy_decision("cache_aware", workers[idx].url());
    }

    /// Prepare cache affinity outside the admission lock, then recheck load and
    /// eligibility before committing. Legacy selection keeps its original
    /// single-pass flow and shares the policy decisions below.
    fn select_with_commit<T>(
        &self,
        context: &SelectionContext,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        commit: impl FnOnce(usize) -> T,
    ) -> Option<(usize, T)> {
        let selection = context.lock.lock();
        let healthy = get_healthy_worker_indices(workers);
        let model_id = normalize_model_key(workers[*healthy.first()?].model_id());
        let (min, max) = self.load_range(workers, &healthy);
        if self.is_imbalanced(min, max) {
            let idx = Self::least_loaded(workers, &healthy)?;
            let reservation = commit(idx);
            drop(selection);
            // Do not scan the prompt when load balancing already decides placement.
            if let Some(text) = request_text {
                let tree = self.trees.get(model_id).map(|entry| entry.value().clone());
                if let Some(tree) = tree {
                    tree.insert(text, workers[idx].url());
                }
            }
            RouterMetrics::record_load_balancing_event();
            RouterMetrics::set_load_range(max, min);
            Self::record_selection(workers, idx);
            return Some((idx, reservation));
        }
        drop(selection);

        let text = request_text.unwrap_or("");
        let tree = self.trees.get(model_id).map(|entry| entry.value().clone());
        // The tree is an approximate history of assignments, not an authoritative
        // KV directory. Concurrent requests can observe the same older affinity.
        let affinity = tree
            .as_ref()
            .map(|tree| tree.prefix_match_with_counts(text));

        let selection = context.lock.lock();
        let healthy = get_healthy_worker_indices(workers);
        healthy.first()?;
        let (min, max) = self.load_range(workers, &healthy);
        let imbalanced = self.is_imbalanced(min, max);
        let (idx, stale) = if imbalanced {
            (Self::least_loaded(workers, &healthy)?, false)
        } else {
            self.choose_from_affinity(workers, &healthy, affinity.as_ref())?
        };
        let reservation = commit(idx);
        drop(selection);

        if let Some(tree) = tree {
            if stale {
                if let Some(result) = affinity {
                    tree.remove_tenant(result.tenant.as_ref());
                }
            } else if !imbalanced || request_text.is_some() {
                // Publish placement before returning, but do not serialize other
                // admissions behind a potentially long radix-tree insertion.
                tree.insert(text, workers[idx].url());
            }
        }
        if imbalanced {
            RouterMetrics::record_load_balancing_event();
            RouterMetrics::set_load_range(max, min);
        }
        Self::record_selection(workers, idx);
        Some((idx, reservation))
    }
}

impl LoadBalancingPolicy for CacheAwarePolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        let model_id = normalize_model_key(workers[*healthy.first()?].model_id());
        let (min, max) = self.load_range(workers, &healthy);
        if self.is_imbalanced(min, max) {
            return self.select_worker_min_load(
                workers,
                request_text,
                &healthy,
                model_id,
                max,
                min,
            );
        }

        // Preserve legacy selection without adding a reservation or a second
        // load check. Only the atomic admission hook rechecks after preparation.
        let text = request_text.unwrap_or("");
        let tree = self.trees.get(model_id).map(|entry| entry.value().clone());
        let affinity = tree
            .as_ref()
            .map(|tree| tree.prefix_match_with_counts(text));
        let (idx, stale) = self.choose_from_affinity(workers, &healthy, affinity.as_ref())?;
        if let Some(tree) = tree {
            if stale {
                if let Some(result) = affinity {
                    tree.remove_tenant(result.tenant.as_ref());
                }
            } else {
                tree.insert(text, workers[idx].url());
            }
        }
        Self::record_selection(workers, idx);
        Some(idx)
    }

    fn select_and_reserve<'a>(
        &self,
        context: &SelectionContext,
        workers: &'a [Arc<dyn Worker>],
        request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<(usize, WorkerLoadGuard<'a>)> {
        self.select_with_commit(context, workers, request_text, |idx| {
            WorkerLoadGuard::new(workers[idx].as_ref())
        })
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
    fn unhealthy_worker_load_does_not_disable_cache_affinity() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 0,
            balance_rel_threshold: 1.0,
            eviction_interval_secs: 0,
            max_tree_size: 100,
        });
        let workers: Vec<Arc<dyn Worker>> = (0..3)
            .map(|index| {
                Arc::new(BasicWorker::new(
                    format!("http://worker-{index}:8000"),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect();
        workers[0].set_healthy(false);
        workers[2].set_healthy(false);
        policy.init_workers(&workers);
        for _ in 0..5 {
            workers[0].increment_load();
            workers[1].increment_load();
        }

        // Establish affinity while worker 1 is the only healthy worker.
        assert_eq!(policy.select_worker(&workers, Some("sticky")), Some(1));
        workers[0].set_healthy(true);

        // Both eligible workers have equal load, so the unhealthy zero-load
        // worker must not switch the policy into load-balancing mode.
        assert_eq!(policy.select_worker(&workers, Some("sticky")), Some(1));
    }

    #[test]
    fn unhealthy_worker_load_does_not_disable_reserved_cache_affinity() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 0,
            balance_rel_threshold: 1.0,
            eviction_interval_secs: 0,
            max_tree_size: 100,
        });
        let workers: Vec<Arc<dyn Worker>> = (0..3)
            .map(|index| {
                Arc::new(BasicWorker::new(
                    format!("http://worker-{index}:8000"),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect();
        workers[0].set_healthy(false);
        workers[2].set_healthy(false);
        policy.init_workers(&workers);
        for _ in 0..5 {
            workers[0].increment_load();
            workers[1].increment_load();
        }

        // Establish affinity while worker 1 is the only healthy worker.
        assert_eq!(policy.select_worker(&workers, Some("sticky")), Some(1));
        workers[0].set_healthy(true);

        // Both eligible workers have equal load, so the unhealthy zero-load
        // worker must not switch the policy into load-balancing mode.
        let context = SelectionContext::default();
        let (idx, guard) = policy
            .select_and_reserve(&context, &workers, Some("sticky"), None)
            .unwrap();
        assert_eq!(idx, 1);
        assert_eq!(workers[1].load(), 6);
        drop(guard);
        assert_eq!(workers[1].load(), 5);
    }

    #[test]
    fn concurrent_cold_reservations_observe_each_other() {
        use std::sync::Barrier;
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let workers: Vec<Arc<dyn Worker>> = (0..8)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://d-{i}"),
                    WorkerType::Decode,
                )) as Arc<dyn Worker>
            })
            .collect();
        policy.init_workers(&workers);
        let context = SelectionContext::default();
        let start = Barrier::new(32);
        let selected = Barrier::new(32);
        std::thread::scope(|scope| {
            let tasks: Vec<_> = (0..32)
                .map(|i| {
                    let (policy, workers, context, start, selected) =
                        (&policy, &workers, &context, &start, &selected);
                    scope.spawn(move || {
                        let text = char::from_u32(0x400 + i).unwrap().to_string().repeat(8192);
                        start.wait();
                        let (_, guard) = policy
                            .select_and_reserve(context, workers, Some(&text), None)
                            .unwrap();
                        selected.wait();
                        guard
                    })
                })
                .collect();
            let guards: Vec<_> = tasks.into_iter().map(|task| task.join().unwrap()).collect();
            assert!(workers.iter().all(|w| w.load() == 4));
            drop(guards);
        });
        assert!(workers.iter().all(|w| w.load() == 0));
    }

    #[test]
    fn prepared_affinity_rechecks_worker_eligibility_and_cold_load() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let workers: Vec<Arc<dyn Worker>> = (0..2)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://d-{i}"),
                    WorkerType::Decode,
                )) as Arc<dyn Worker>
            })
            .collect();
        let tree = Tree::new();
        tree.insert("warm conversation", workers[0].url());
        let warm = tree.prefix_match_with_counts("warm conversation");
        let cold = tree.prefix_match_with_counts("unrelated text");
        // Both observations precede the changed load and health state.
        let pending = WorkerLoadGuard::new(workers[0].as_ref());
        assert_eq!(
            policy.choose_from_affinity(&workers, &[0, 1], Some(&cold)),
            Some((1, false))
        );
        workers[0].set_healthy(false);
        let healthy = get_healthy_worker_indices(&workers);
        assert_eq!(
            policy.choose_from_affinity(&workers, &healthy, Some(&warm)),
            Some((1, true))
        );
        drop(pending);
    }

    #[test]
    fn reservation_commits_under_pool_lock_and_preserves_sentinel_load() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(
            "http://d".into(),
            WorkerType::Decode,
        ))];
        policy.init_workers(&workers);
        let context = SelectionContext::default();
        let sentinel = WorkerLoadGuard::new(workers[0].as_ref());
        let (_, guard) = policy
            .select_with_commit(&context, &workers, Some("text"), |idx| {
                assert!(context.lock.try_lock().is_none());
                WorkerLoadGuard::new(workers[idx].as_ref())
            })
            .unwrap();
        assert!(context.lock.try_lock().is_some());
        assert_eq!(workers[0].load(), 2);
        drop(guard);
        assert_eq!(workers[0].load(), 1);
        drop(sentinel);
    }

    #[test]
    fn shared_policy_does_not_share_pool_admission_locks() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(
            "http://d".into(),
            WorkerType::Decode,
        ))];
        policy.init_workers(&workers);
        let other_pool = SelectionContext::default();
        let context = SelectionContext::default();
        let _held = other_pool.lock.lock();
        let (_, guard) = policy
            .select_and_reserve(&context, &workers, Some("text"), None)
            .unwrap();
        assert_eq!(workers[0].load(), 1);
        drop(guard);
        assert_eq!(workers[0].load(), 0);
    }

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
    fn literal_token_ids_prefix_keeps_cache_affinity() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
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
        let key = "token_ids: explain this field";
        let model_id = normalize_model_key(workers[0].model_id());
        policy
            .trees
            .get(model_id)
            .expect("initialized model cache")
            .insert(key, workers[1].url());

        assert_eq!(policy.select_worker(&workers, Some(key)), Some(1));
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

    #[test]
    fn initial_placement_combines_affinity_and_headroom_without_mutation() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let candidates = [
            CacheAwareCandidate {
                target_id: "rank-0",
                kv_pressure: 0.2,
            },
            CacheAwareCandidate {
                target_id: "rank-1",
                kv_pressure: 0.1,
            },
        ];
        assert_eq!(
            policy
                .evaluate_initial_placement("model", "shared suffix", &candidates)
                .unwrap()
                .target_id,
            "rank-1"
        );
        policy.commit_placement("model", "shared first", "rank-0");
        let equal_pressure = [
            CacheAwareCandidate {
                target_id: "rank-0",
                kv_pressure: 0.1,
            },
            CacheAwareCandidate {
                target_id: "rank-1",
                kv_pressure: 0.1,
            },
        ];
        assert_eq!(
            policy
                .evaluate_initial_placement("model", "shared second", &equal_pressure)
                .unwrap()
                .target_id,
            "rank-0"
        );
    }
}
