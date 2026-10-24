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
    Route to the worker with the lowest load
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
    Below this threshold, routes to the worker with the lowest load.

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
    Maximum characters per worker in each model tree. When exceeded, LRU
    leaf entries are evicted during the next eviction cycle. This is not
    a hard process-memory limit.
*/

use super::{get_healthy_worker_indices, CacheAwareConfig, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::policies::normalize_model_key;
use crate::protocols::spec::cache_routing;
use crate::tree::Tree;
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
                let affinity =
                    cache_routing::match_rate(request_text, matched_chars, input_chars) as f64;
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

        // Use shortest queue when imbalanced
        let min_load_idx = healthy_indices
            .iter()
            .min_by_key(|&&idx| workers[idx].load())
            .copied()?;

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

        // Increment processed counter
        workers[min_load_idx].increment_processed();
        RouterMetrics::record_processed_request(workers[min_load_idx].url());
        RouterMetrics::record_policy_decision(self.name(), workers[min_load_idx].url());

        Some(min_load_idx)
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
        let text = request_text.unwrap_or("");
        let key_kind = cache_routing::kind(text);
        RouterMetrics::record_cache_routing_key(key_kind);
        debug!(
            key_kind,
            prefix_information = matches!(key_kind, "tokens" | "text" | "legacy_text"),
            "Cache-aware routing input"
        );

        // Determine the model for this set of workers (router pre-filters by model)
        // All workers should be from the same model
        let model_id = normalize_model_key(workers[healthy_indices[0]].model_id());

        // Base the load decision on workers that are eligible for selection.
        // Unhealthy workers must not affect whether healthy workers use cache
        // affinity or load balancing.
        let (min_load, max_load) = healthy_indices
            .iter()
            .map(|&idx| workers[idx].load())
            .fold((usize::MAX, 0usize), |(min, max), load| {
                (min.min(load), max.max(load))
            });
        let min_load = if min_load == usize::MAX { 0 } else { min_load };

        // Check if load is imbalanced
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
                &healthy_indices,
                model_id,
                max_load,
                min_load,
            );
        }

        // Use cache-aware routing when balanced
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
        let match_rate =
            cache_routing::match_rate(text, result.matched_char_count, result.input_char_count);

        debug!(
            "Cache match for model '{}': matched_chars={}, input_chars={}, match_rate={:.2}",
            model_id, result.matched_char_count, result.input_char_count, match_rate
        );
        // Select worker without String allocation
        let selected_idx = if match_rate > self.config.cache_threshold {
            // Cache hit path: find worker by URL (compare &str directly, no allocation)
            let tenant_url: &str = &result.tenant;
            workers
                .iter()
                .position(|w| w.url() == tenant_url)
                .filter(|&idx| workers[idx].is_healthy())
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
    use crate::protocols::spec::{CompletionRequest, GenerationRequest};

    #[test]
    fn content_keys_do_not_turn_unrelated_tokens_or_sessions_into_hits() {
        let key = |value| {
            serde_json::from_value::<CompletionRequest>(value)
                .unwrap()
                .extract_cache_routing_key()
        };
        let pairs = [
            (
                serde_json::json!({"prompt": [1,2,3,4]}),
                serde_json::json!({"prompt": [2,3,4,5]}),
            ),
            (
                serde_json::json!({"prompt": [1]}),
                serde_json::json!({"prompt": [2]}),
            ),
            (
                serde_json::json!({"prompt": [1,2,3,4]}),
                serde_json::json!({"prompt": [1,2,9,8]}),
            ),
            (
                serde_json::json!({"prompt": "a", "session_params": {"session_id": "sess-0001"}}),
                serde_json::json!({"prompt": "a", "session_params": {"session_id": "sess-0002"}}),
            ),
        ];
        for (seed, query) in pairs {
            let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
                eviction_interval_secs: 0,
                ..Default::default()
            });
            let workers: Vec<Arc<dyn Worker>> = (0..2)
                .map(|i| {
                    Arc::new(BasicWorker::new(
                        format!("http://w{i}:8000"),
                        WorkerType::Regular,
                    )) as Arc<dyn Worker>
                })
                .collect();
            policy.init_workers(&workers);
            let seed = key(seed);
            policy.commit_placement(workers[0].model_id(), &seed, workers[0].url());
            workers[0].increment_load();
            // The absolute imbalance guard does not fire at (1,0).
            // Equal-length unrelated tokens and similar session IDs must miss.
            assert_eq!(policy.select_worker(&workers, Some(&key(query))), Some(1));
            // The original complete key still has affinity to worker 0.
            assert_eq!(policy.select_worker(&workers, Some(&seed)), Some(0));
        }
    }

    #[test]
    fn token_keys_keep_real_prefix_affinity() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let workers: Vec<Arc<dyn Worker>> = (0..2)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://w{i}:8000"),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect();
        policy.init_workers(&workers);
        let seed = cache_routing::from_value(&serde_json::json!({"prompt": [1,2,3,4]}));
        let query = cache_routing::from_value(&serde_json::json!({"prompt": [1,2,3,5]}));
        policy.commit_placement(workers[0].model_id(), &seed, workers[0].url());
        workers[0].increment_load();
        assert_eq!(policy.select_worker(&workers, Some(&query)), Some(0));
    }

    #[test]
    fn long_sessions_balance_shared_opening_and_reuse_their_full_history() {
        for common_tokens in [400, 1800, 2048, 4096, 8192] {
            // Match the CLI defaults, with periodic eviction disabled for determinism.
            let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
                cache_threshold: 0.3,
                balance_abs_threshold: 64,
                balance_rel_threshold: 1.5,
                max_tree_size: 1 << 26,
                eviction_interval_secs: 0,
            });
            let workers: Vec<Arc<dyn Worker>> = (0..4)
                .map(|i| {
                    Arc::new(BasicWorker::new(
                        format!("http://w{i}:8000"),
                        WorkerType::Regular,
                    )) as Arc<dyn Worker>
                })
                .collect();
            policy.init_workers(&workers);
            let mut sessions = Vec::new();
            let mut counts = [0; 4];
            for session in 0..30 {
                let mut ids = vec![session + 100; 52_000];
                ids[..common_tokens].fill(42);
                let idx = policy
                    .select_worker(&workers, Some(&cache_routing::tokens(&ids)))
                    .unwrap();
                workers[idx].increment_load();
                counts[idx] += 1;
                sessions.push((ids, idx));
            }
            assert_eq!(counts, [8, 8, 7, 7], "shared tokens: {common_tokens}");

            // A new turn must distinguish histories beyond the old 2048-token window.
            for (mut ids, owner) in sessions {
                ids.extend([300; 1024]);
                assert_eq!(
                    policy.select_worker(&workers, Some(&cache_routing::tokens(&ids))),
                    Some(owner),
                    "shared tokens: {common_tokens}"
                );
            }
        }
    }

    #[test]
    fn long_chat_text_does_not_treat_a_shared_opening_as_a_full_match() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.3,
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let workers: Vec<Arc<dyn Worker>> = (0..2)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://w{i}:8000"),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect();
        policy.init_workers(&workers);
        let chat_key = |suffix: &str| {
            cache_routing::for_route(
                &serde_json::json!({"messages": [{"role": "user", "content":
                    format!("{}{}", "界".repeat(18_000), suffix.repeat(82_000))}]}),
                "/v1/chat/completions",
            )
        };
        let first = chat_key("甲");
        assert_eq!(policy.select_worker(&workers, Some(&first)), Some(0));
        workers[0].increment_load();
        assert_eq!(
            policy.select_worker(&workers, Some(&chat_key("乙"))),
            Some(1)
        );
        assert_eq!(policy.select_worker(&workers, Some(&first)), Some(0));
    }

    #[test]
    fn full_token_keys_remain_subject_to_the_character_budget() {
        let tree = Tree::new();
        let key = cache_routing::tokens(&vec![42; 52_000]);
        tree.insert(&key, "worker");
        assert_eq!(tree.get_tenant_char_count()["worker"], key.len());
        tree.evict_tenant_by_size(100_000);
        assert_eq!(tree.get_tenant_char_count()["worker"], 0);
        assert_eq!(tree.prefix_match_tenant_char_count(&key, "worker"), 0);
    }

    /// Isolated CPU/RSS probe, not a timing assertion or an engine benchmark.
    #[test]
    #[ignore = "long-prompt cost probe; run alone with --ignored --nocapture"]
    fn full_prefix_cost_probe() {
        use std::sync::Barrier;
        use std::time::Instant;

        fn rss_kib() -> Option<usize> {
            std::fs::read_to_string("/proc/self/status")
                .ok()?
                .lines()
                .find_map(|line| line.strip_prefix("VmRSS:"))?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        }

        let token_count: usize = std::env::var("CACHE_PROBE_TOKENS")
            .unwrap_or_else(|_| "52000".into())
            .parse()
            .unwrap();
        let concurrency: usize = std::env::var("CACHE_PROBE_CONCURRENCY")
            .unwrap_or_else(|_| "8".into())
            .parse()
            .unwrap();
        assert!(token_count > 8192 && concurrency > 0);
        let requests: Vec<CompletionRequest> = (0..32)
            .map(|session| {
                let mut ids = vec![session + 100; token_count];
                ids[..8192].fill(42);
                serde_json::from_value(serde_json::json!({"prompt": ids})).unwrap()
            })
            .collect();
        let workers: Vec<Arc<dyn Worker>> = (0..4)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://w{i}:8000"),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect();
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.3,
            balance_abs_threshold: 64,
            balance_rel_threshold: 1.5,
            max_tree_size: 1 << 26,
            eviction_interval_secs: 0,
        });
        policy.init_workers(&workers);
        let rss_before = rss_kib();
        let warm_start = Instant::now();
        for request in &requests {
            let key = request.extract_cache_routing_key();
            let idx = policy.select_worker(&workers, Some(&key)).unwrap();
            workers[idx].increment_load();
        }
        let warm_ms = warm_start.elapsed().as_secs_f64() * 1000.0;
        let barrier = Barrier::new(concurrency);
        let mut samples = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..concurrency)
                .map(|thread_id| {
                    let (requests, workers, policy, barrier) =
                        (&requests, &workers, &policy, &barrier);
                    scope.spawn(move || {
                        barrier.wait();
                        (0..64)
                            .map(|i| {
                                let request = &requests[(i + thread_id) % requests.len()];
                                let start = Instant::now();
                                let key = request.extract_cache_routing_key();
                                let key_us = start.elapsed().as_micros();
                                assert!(policy.select_worker(workers, Some(&key)).is_some());
                                (key_us, start.elapsed().as_micros())
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        let mut key_samples: Vec<_> = samples.iter().map(|sample| sample.0).collect();
        key_samples.sort_unstable();
        samples.sort_unstable_by_key(|sample| sample.1);
        let p50 = (samples.len() - 1) / 2;
        let p99 = (samples.len() - 1) * 99 / 100;
        let accounted_chars: usize = policy
            .trees
            .iter()
            .flat_map(|tree| tree.get_tenant_char_count().into_values())
            .sum();
        eprintln!(
            "tokens={token_count} concurrency={concurrency} samples={} key_bytes={} \
             warm_ms={warm_ms:.2} key_p50_us={} key_p99_us={} \
             key_select_insert_p50_us={} key_select_insert_p99_us={} \
             accounted_chars={accounted_chars} rss_before_kib={rss_before:?} rss_after_kib={:?}",
            samples.len(),
            requests[0].extract_cache_routing_key().len(),
            key_samples[p50],
            key_samples[p99],
            samples[p50].1,
            samples[p99].1,
            rss_kib(),
        );
    }

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
