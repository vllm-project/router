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
    This strategy maintains an approximate radix tree from request history.
    Router paths use backend-effective token keys, shared with KV-aware input
    preparation, including every prompt in a batch. Legacy text-only callers
    can still supply text keys. This history is not an engine residency index.

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
    Maximum characters per worker in each model tree, shared across LoRA and
    cache-salt identities. When exceeded, LRU
    leaf entries are evicted during the next eviction cycle. This is not
    a hard process-memory limit.
*/

use super::{
    get_healthy_worker_indices, pick_tied, CacheAwareConfig, KvRequestTokens, LoadBalancingPolicy,
    RequestHeaders, SelectionContext, WorkerTokenInputs,
};
use crate::core::{Worker, WorkerLoadGuard};
use crate::metrics::RouterMetrics;
use crate::policies::normalize_model_key;
use crate::protocols::spec::cache_routing;
use crate::tree::{PrefixMatchResult, Tree};
use dashmap::DashMap;
use rand::{Rng, SeedableRng};
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

/// Prefix matching prepared before PD admission; selection samples fresh load.
pub struct PreparedCacheSelection {
    tree: Arc<Tree>,
    candidates: HashMap<String, (Vec<Arc<str>>, f64)>,
}

/// One eligibility and load observation, reusable for ordered candidate subsets.
pub(crate) struct PreparedCacheSelectionView<'a> {
    workers: &'a [Arc<dyn Worker>],
    prepared: Option<&'a PreparedCacheSelection>,
    indices: Vec<usize>,
    loads: Vec<Option<usize>>,
    // Sample once per view. Re-evaluating the same view has no RNG side effects.
    tie_seed: u64,
}

impl PreparedCacheSelectionView<'_> {
    pub(crate) fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub(crate) fn load(&self, index: usize) -> Option<usize> {
        self.loads.get(index).copied().flatten()
    }
}

/// An evaluated choice has no effects until the caller commits the actual one.
pub(crate) struct PreparedCacheDecision<'a> {
    target: Option<PreparedCacheTarget<'a>>,
    min_load: usize,
    max_load: usize,
    imbalanced: bool,
    outcome: Option<&'static str>,
    cache_update: Option<PreparedCacheUpdate<'a>>,
}

impl PreparedCacheDecision<'_> {
    pub(crate) fn worker_index(&self) -> Option<usize> {
        self.target.as_ref().map(|target| target.index)
    }
}

struct PreparedCacheTarget<'a> {
    index: usize,
    worker: &'a dyn Worker,
}

struct PreparedCacheUpdate<'a> {
    tree: &'a Tree,
    keys: &'a [Arc<str>],
    match_rate: f64,
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

    /// Compare backend-effective prompts, scoped by the request's model.
    pub fn select_worker_with_inputs(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: KvRequestTokens<'_>,
        model_id: Option<&str>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
        stage: &'static str,
    ) -> Option<usize> {
        let prepared = self.prepare_selection(workers, tokens, model_id, lora_name, cache_salt);
        self.select_prepared(workers, prepared.as_ref(), stage)
    }

    pub(crate) fn prepare_selection(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: KvRequestTokens<'_>,
        model_id: Option<&str>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
    ) -> Option<PreparedCacheSelection> {
        let healthy = get_healthy_worker_indices(workers);
        let first = workers.get(*healthy.first()?)?;
        // PD workers may have no model label. Request model/adapter names must
        // still isolate their cache history even when every label is unknown.
        let model = normalize_model_key(model_id.unwrap_or_else(|| first.model_id())).to_owned();
        // Keep cache identities in the keys so all scopes share the model's
        // per-worker eviction budget instead of allocating a tree per salt.
        let scope_prefix = if lora_name.is_some() || cache_salt.is_some() {
            use sha2::{Digest, Sha256};
            let identity = serde_json::to_vec(&(lora_name, cache_salt)).ok()?;
            format!("\0ca:s:{:x}:", Sha256::digest(identity))
        } else {
            String::new()
        };
        let tree = self
            .trees
            .entry(model)
            .or_insert_with(|| Arc::new(Tree::new()))
            .clone();
        // Token arrays and DP ranks commonly share prompts. Encode each
        // distinct prompt once rather than allocating eight bytes per token
        // for every candidate.
        let mut encoded: HashMap<&[u32], Arc<str>> = HashMap::new();
        let mut candidates = HashMap::new();
        for index in healthy {
            let worker = &workers[index];
            let prompts = tokens.prompts_for(worker.as_ref());
            if prompts.is_empty() || prompts.iter().any(|ids| ids.is_empty()) {
                continue;
            }
            let keys: Vec<_> = prompts
                .iter()
                .map(|ids| {
                    encoded
                        .entry(ids)
                        .or_insert_with(|| {
                            Arc::from(cache_routing::token_ids_with_prefix(ids, &scope_prefix))
                        })
                        .clone()
                })
                .collect();
            let mut matched = 0usize;
            let mut input = 0usize;
            for key in &keys {
                let hit = tree.prefix_match_tenant_char_count(key, worker.url());
                // The ASCII identity prefix isolates caches but earns no
                // token credit, including partial matches between digests.
                let (hit_tokens, input_tokens) = cache_routing::token_prefix_counts(
                    &key[scope_prefix.len()..],
                    hit.saturating_sub(scope_prefix.len()),
                );
                matched += hit_tokens;
                input += input_tokens;
            }
            let rate = if input == 0 {
                0.0
            } else {
                matched as f64 / input as f64
            };
            candidates.insert(worker.url().to_owned(), (keys, rate));
        }
        Some(PreparedCacheSelection { tree, candidates })
    }

    pub(crate) fn select_prepared(
        &self,
        workers: &[Arc<dyn Worker>],
        prepared: Option<&PreparedCacheSelection>,
        stage: &'static str,
    ) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        // Legacy callers read loads again when selecting after the range check.
        let decision = self.evaluate_prepared_with_load(
            workers,
            prepared,
            &healthy,
            |index| workers[index].load(),
            pick_tied,
        );
        self.commit_prepared(decision, stage)
    }

    pub(crate) fn selection_view<'a>(
        &self,
        workers: &'a [Arc<dyn Worker>],
        prepared: Option<&'a PreparedCacheSelection>,
    ) -> PreparedCacheSelectionView<'a> {
        let indices = get_healthy_worker_indices(workers);
        let mut loads = vec![None; workers.len()];
        for &index in &indices {
            loads[index] = Some(workers[index].load());
        }
        PreparedCacheSelectionView {
            workers,
            prepared,
            indices,
            loads,
            tie_seed: rand::rng().random(),
        }
    }

    /// Recompute the baseline within this ordered subset, using only the view.
    /// Indices outside the captured eligible set cannot become candidates.
    pub(crate) fn evaluate_prepared<'a>(
        &self,
        view: &PreparedCacheSelectionView<'a>,
        indices: &[usize],
    ) -> PreparedCacheDecision<'a> {
        let candidates: Vec<_> = indices
            .iter()
            .copied()
            .filter(|&index| view.load(index).is_some())
            .collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(view.tie_seed);
        self.evaluate_prepared_with_load(
            view.workers,
            view.prepared,
            &candidates,
            |index| view.load(index).expect("captured eligible candidate"),
            |tied| (!tied.is_empty()).then(|| tied[rng.random_range(0..tied.len())]),
        )
    }

    fn evaluate_prepared_with_load<'a>(
        &self,
        workers: &'a [Arc<dyn Worker>],
        prepared: Option<&'a PreparedCacheSelection>,
        candidates: &[usize],
        mut load: impl FnMut(usize) -> usize,
        pick: impl FnMut(&[usize]) -> Option<usize>,
    ) -> PreparedCacheDecision<'a> {
        let (min_load, max_load) = candidates
            .iter()
            .fold((usize::MAX, 0), |(min, max), &index| {
                let value = load(index);
                (min.min(value), max.max(value))
            });
        // Preserve load_range's sentinel handling, including usize::MAX loads.
        let min_load = if min_load == usize::MAX { 0 } else { min_load };
        let imbalanced = self.is_imbalanced(min_load, max_load);
        let mut decision = PreparedCacheDecision {
            target: None,
            min_load,
            max_load,
            imbalanced,
            outcome: None,
            cache_update: None,
        };
        let matches: Option<Vec<_>> = candidates
            .iter()
            .map(|&index| prepared?.candidates.get(workers[index].url()))
            .collect();
        let Some((prepared, matches)) = prepared.zip(matches) else {
            decision.outcome = Some("missing_tokens");
            decision.target = Self::pick_least_loaded_with(
                candidates.iter().map(|&index| (index, load(index))),
                pick,
            )
            .map(|index| PreparedCacheTarget {
                index,
                worker: workers[index].as_ref(),
            });
            return decision;
        };
        let best_rate = matches
            .iter()
            .map(|(_, rate)| *rate)
            .fold(0.0_f64, f64::max);
        let use_prefix = !imbalanced && best_rate > self.config.cache_threshold as f64;
        let Some(position) = Self::pick_least_loaded_with(
            (0..candidates.len())
                .filter(|&i| !use_prefix || matches[i].1 == best_rate)
                .map(|i| (i, load(candidates[i]))),
            pick,
        ) else {
            return decision;
        };
        decision.target = Some(PreparedCacheTarget {
            index: candidates[position],
            worker: workers[candidates[position]].as_ref(),
        });
        decision.outcome = Some(if imbalanced {
            "load_imbalance"
        } else if use_prefix {
            "prefix_hit"
        } else {
            "prefix_miss"
        });
        decision.cache_update = Some(PreparedCacheUpdate {
            tree: prepared.tree.as_ref(),
            keys: &matches[position].0,
            match_rate: matches[position].1,
        });
        decision
    }

    /// Commit only the final choice; discarded evaluations publish nothing.
    pub(crate) fn commit_prepared(
        &self,
        decision: PreparedCacheDecision<'_>,
        stage: &'static str,
    ) -> Option<usize> {
        if decision.imbalanced {
            RouterMetrics::record_load_balancing_event();
            RouterMetrics::set_load_range(decision.max_load, decision.min_load);
        }
        if let (Some(target), Some(update)) = (&decision.target, &decision.cache_update) {
            for key in update.keys {
                update.tree.insert(key, target.worker.url());
            }
        }
        if let Some(outcome) = decision.outcome {
            RouterMetrics::record_prefix_routing_decision("cache_aware", stage, outcome);
        }
        let index = decision.worker_index()?;
        let target = decision.target?;
        if let Some(update) = decision.cache_update {
            debug!(
                stage,
                selected = target.worker.url(),
                match_rate = update.match_rate,
                batch_prompts = update.keys.len(),
                "Cache prefix decision"
            );
        }
        Self::record_worker_selection(target.worker);
        Some(index)
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
            self.trees
                .entry(model_id.to_owned())
                .or_insert_with(|| Arc::new(Tree::new()))
                .insert(text, workers[min_load_idx].url());
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
        Self::pick_least_loaded(healthy.iter().map(|&idx| (idx, workers[idx].load())))
    }

    /// Picks uniformly among the `(id, load)` candidates with the lowest load
    /// and returns the `id` of the chosen candidate.
    ///
    /// The `id` is opaque here: `least_loaded()` passes worker indices, while
    /// `select_prepared()` passes indices into `healthy`.
    ///
    /// Each load is read once so that concurrent updates cannot change
    /// which candidates count as tied.
    fn pick_least_loaded(candidates: impl Iterator<Item = (usize, usize)>) -> Option<usize> {
        Self::pick_least_loaded_with(candidates, pick_tied)
    }

    fn pick_least_loaded_with(
        candidates: impl Iterator<Item = (usize, usize)>,
        mut pick: impl FnMut(&[usize]) -> Option<usize>,
    ) -> Option<usize> {
        let mut min_load = usize::MAX;
        let mut tied = Vec::new();
        for (id, load) in candidates {
            if load < min_load {
                min_load = load;
                tied.clear();
            }
            if load == min_load {
                tied.push(id);
            }
        }
        pick(&tied)
    }

    /// Choose using an approximate affinity observation, but current eligibility
    /// and load. Called within the pool's admission critical section, when used.
    /// Returns whether the observed cache target has become unavailable.
    fn choose_from_affinity(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy: &[usize],
        affinity: Option<&PrefixMatchResult>,
        text: &str,
    ) -> Option<(usize, bool)> {
        let Some(result) = affinity else {
            let idx = *healthy.get(rand::rng().random_range(0..healthy.len()))?;
            return Some((idx, false));
        };
        let match_rate =
            cache_routing::match_rate(text, result.matched_char_count, result.input_char_count);
        if match_rate > self.config.cache_threshold {
            if let Some(idx) = workers
                .iter()
                .position(|worker| worker.url() == result.tenant.as_ref())
                .filter(|&idx| workers[idx].is_healthy())
            {
                return Some((idx, false));
            }
            return Self::least_loaded(workers, healthy).map(|idx| (idx, true));
        }
        Self::least_loaded(workers, healthy).map(|idx| (idx, false))
    }

    fn record_selection(workers: &[Arc<dyn Worker>], idx: usize) {
        Self::record_worker_selection(workers[idx].as_ref());
    }

    fn record_worker_selection(worker: &dyn Worker) {
        worker.increment_processed();
        RouterMetrics::record_processed_request(worker.url());
        RouterMetrics::record_policy_decision("cache_aware", worker.url());
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
            self.choose_from_affinity(workers, &healthy, affinity.as_ref(), text)?
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
        let (idx, stale) = self.choose_from_affinity(workers, &healthy, affinity.as_ref(), text)?;
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

    fn tracks_request_load(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "cache_aware"
    }

    /// Re-running needs the prefix selection prepared at admission; without it
    /// the request body would have to be carried across the Prefill.
    fn can_redecide_at_dispatch(&self, decode_cache_ready: bool) -> bool {
        decode_cache_ready
    }

    fn prepare_cache_selection(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: &WorkerTokenInputs,
        model_id: Option<&str>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
    ) -> Option<PreparedCacheSelection> {
        self.prepare_selection(
            workers,
            KvRequestTokens::PerWorker(tokens),
            model_id,
            lora_name,
            cache_salt,
        )
    }

    fn needs_cache_routing_key(&self) -> bool {
        true
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

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use crate::protocols::spec::{CompletionRequest, GenerationRequest};

    fn cache_fixture(
        worker_count: usize,
        config: CacheAwareConfig,
    ) -> (CacheAwarePolicy, Vec<Arc<dyn Worker>>) {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..config
        });
        let workers: Vec<Arc<dyn Worker>> = (0..worker_count)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://w{i}:8000"),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect();
        policy.init_workers(&workers);
        (policy, workers)
    }

    #[test]
    fn prepared_evaluations_only_commit_actual_choice() {
        let (policy, workers) = cache_fixture(3, CacheAwareConfig::default());
        let ids = [1, 2, 3, 4];
        let prepared = policy
            .prepare_selection(
                &workers,
                KvRequestTokens::Shared(Some(&ids)),
                None,
                None,
                None,
            )
            .unwrap();
        let view = policy.selection_view(&workers, Some(&prepared));
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _metrics = metrics::set_default_local_recorder(&recorder);
        let before = handle.render();

        let full_target = policy
            .evaluate_prepared(&view, view.indices())
            .worker_index();
        let subset_target = policy.evaluate_prepared(&view, &[2, 1]).worker_index();
        assert!(matches!(full_target, Some(0..=2)));
        assert!(matches!(subset_target, Some(1 | 2)));
        for _ in 0..3 {
            assert_eq!(
                policy
                    .evaluate_prepared(&view, view.indices())
                    .worker_index(),
                full_target
            );
            assert_eq!(
                policy.evaluate_prepared(&view, &[2, 1]).worker_index(),
                subset_target
            );
        }
        let actual = policy.evaluate_prepared(&view, &[1]);
        assert_eq!(handle.render(), before);
        for worker in &workers {
            assert_eq!(worker.processed_requests(), 0);
            let key = &prepared.candidates[worker.url()].0[0];
            assert_eq!(
                prepared
                    .tree
                    .prefix_match_tenant_char_count(key, worker.url()),
                0
            );
        }

        assert_eq!(policy.commit_prepared(actual, "prefill"), Some(1));
        for (index, worker) in workers.iter().enumerate() {
            assert_eq!(worker.processed_requests(), usize::from(index == 1));
            let key = &prepared.candidates[worker.url()].0[0];
            assert_eq!(
                prepared
                    .tree
                    .prefix_match_tenant_char_count(key, worker.url()),
                if index == 1 { key.chars().count() } else { 0 }
            );
        }
        let rendered = handle.render();
        for name in [
            "vllm_router_processed_requests_total",
            "vllm_router_policy_decisions_total",
            "vllm_router_prefix_routing_decisions_total",
        ] {
            let samples: Vec<_> = rendered
                .lines()
                .filter(|line| line.starts_with(name))
                .collect();
            assert_eq!(samples.len(), 1, "{rendered}");
            assert_eq!(
                samples[0]
                    .split_whitespace()
                    .last()
                    .unwrap()
                    .parse::<f64>()
                    .unwrap(),
                1.0
            );
        }
    }

    #[test]
    fn prepared_ties_are_repeatable_per_view_without_first_worker_bias() {
        let (policy, workers) = cache_fixture(3, CacheAwareConfig::default());
        workers[2].increment_load();
        let ids = [1, 2, 3, 4];
        let prepared = policy.prepare_selection(
            &workers,
            KvRequestTokens::Shared(Some(&ids)),
            None,
            None,
            None,
        );
        for input in [None, prepared.as_ref()] {
            let mut selected = std::collections::HashSet::new();
            for seed in 0..64 {
                let mut view = policy.selection_view(&workers, input);
                view.tie_seed = seed;
                let target = policy
                    .evaluate_prepared(&view, view.indices())
                    .worker_index()
                    .unwrap();
                assert_eq!(
                    policy
                        .evaluate_prepared(&view, view.indices())
                        .worker_index(),
                    Some(target)
                );
                assert!(target < 2, "a higher-load worker must not win a tie");
                selected.insert(target);
            }
            assert_eq!(
                selected.len(),
                2,
                "both equally ranked workers remain selectable"
            );
        }
        assert!(workers
            .iter()
            .all(|worker| worker.processed_requests() == 0));
    }

    #[test]
    fn prepared_commit_keeps_the_originating_worker_and_tree() {
        let (policy, workers) = cache_fixture(2, CacheAwareConfig::default());
        let ids = [1, 2, 3, 4];
        let first = policy
            .prepare_selection(
                &workers,
                KvRequestTokens::Shared(Some(&ids)),
                Some("model-a"),
                None,
                None,
            )
            .unwrap();
        let other_workers = std::slice::from_ref(&workers[0]);
        let second = policy
            .prepare_selection(
                other_workers,
                KvRequestTokens::Shared(Some(&ids)),
                Some("model-b"),
                None,
                None,
            )
            .unwrap();
        let first_decision = {
            let view = policy.selection_view(&workers, Some(&first));
            policy.evaluate_prepared(&view, &[1])
        };
        let second_decision = {
            let view = policy.selection_view(other_workers, Some(&second));
            policy.evaluate_prepared(&view, view.indices())
        };
        // The other view has no index 1 and uses a different model's tree.
        assert_eq!(first_decision.worker_index(), Some(1));
        assert_eq!(policy.commit_prepared(second_decision, "decode"), Some(0));
        assert_eq!(policy.commit_prepared(first_decision, "prefill"), Some(1));
        for (prepared, chosen) in [(&first, 1), (&second, 0)] {
            let key = &prepared.candidates[workers[chosen].url()].0[0];
            for (index, worker) in workers.iter().enumerate() {
                assert_eq!(
                    prepared
                        .tree
                        .prefix_match_tenant_char_count(key, worker.url()),
                    if index == chosen {
                        key.chars().count()
                    } else {
                        0
                    }
                );
            }
        }
        assert!(workers
            .iter()
            .all(|worker| worker.processed_requests() == 1));
    }

    #[test]
    fn prepared_subsets_recompute_prefix_imbalance_and_ties() {
        let (policy, workers) = cache_fixture(
            4,
            CacheAwareConfig {
                balance_abs_threshold: 3,
                ..Default::default()
            },
        );
        let ids = [1, 2, 3, 4];
        for (index, length) in [(0, 4), (1, 3), (2, 3)] {
            policy.select_worker_with_inputs(
                std::slice::from_ref(&workers[index]),
                KvRequestTokens::Shared(Some(&ids[..length])),
                None,
                None,
                None,
                "prefill",
            );
        }
        for (worker, load) in workers.iter().zip([5, 2, 2, 20]) {
            for _ in 0..load {
                worker.increment_load();
            }
        }
        let prepared = policy.prepare_selection(
            &workers,
            KvRequestTokens::Shared(Some(&ids)),
            None,
            None,
            None,
        );
        let view = policy.selection_view(&workers, prepared.as_ref());
        let full = policy.evaluate_prepared(&view, view.indices());
        assert!(matches!(full.worker_index(), Some(1 | 2)));
        assert_eq!(full.outcome, Some("load_imbalance"));
        // Removing the overloaded worker must recompute imbalance, enabling affinity.
        let balanced = policy.evaluate_prepared(&view, &[0, 1, 2]);
        assert_eq!(
            (balanced.worker_index(), balanced.outcome),
            (Some(0), Some("prefix_hit"))
        );
        // Removing the global best prefix must allow the best prefix in the subset.
        let tied = policy.evaluate_prepared(&view, &[2, 1]);
        assert!(matches!(tied.worker_index(), Some(1 | 2)));
        assert_eq!(tied.outcome, Some("prefix_hit"));
    }

    #[test]
    fn prepared_view_freezes_loads_and_eligibility() {
        let (policy, workers) = cache_fixture(2, CacheAwareConfig::default());
        workers[1].increment_load();
        let view = policy.selection_view(&workers, None);
        workers[0].increment_load();
        workers[0].increment_load();
        workers[0].set_healthy(false);
        assert_eq!(view.load(0), Some(0));
        assert_eq!(
            policy
                .evaluate_prepared(&view, view.indices())
                .worker_index(),
            Some(0)
        );
        // A new observation is required before committing after eligibility changes.
        let fresh = policy.selection_view(&workers, None);
        assert_eq!(fresh.indices(), &[1]);
        assert_eq!(
            policy.evaluate_prepared(&fresh, &[0, 1]).worker_index(),
            Some(1)
        );
        assert_eq!(policy.select_prepared(&workers, None, "prefill"), Some(1));
    }

    #[test]
    fn prepared_live_load_selection_preserves_two_stage_reads() {
        let (policy, workers) = cache_fixture(2, CacheAwareConfig::default());
        workers[1].increment_load();
        let ids = [1, 2, 3, 4];
        let mut prepared = policy
            .prepare_selection(
                &workers,
                KvRequestTokens::Shared(Some(&ids)),
                None,
                None,
                None,
            )
            .unwrap();
        for input in [None, Some(&prepared)] {
            let view = policy.selection_view(&workers, input);
            assert_eq!(
                policy
                    .evaluate_prepared(&view, view.indices())
                    .worker_index(),
                Some(0)
            );
            // A dispatch changes loads between the legacy range and minimum scans.
            let mut reads = [(0, 0), (1, 1), (0, 9), (1, 0)].into_iter();
            let decision = policy.evaluate_prepared_with_load(
                &workers,
                input,
                view.indices(),
                |index| {
                    let (expected_index, value) = reads.next().expect("unexpected load read");
                    assert_eq!(index, expected_index);
                    value
                },
                pick_tied,
            );
            assert!(reads.next().is_none());
            assert_eq!((decision.min_load, decision.max_load), (0, 1));
            assert_eq!(policy.commit_prepared(decision, "regular"), Some(1));
        }

        prepared.candidates.get_mut(workers[1].url()).unwrap().1 = 1.0;
        // With affinity, the second scan reads only the best-prefix candidate.
        let mut reads = [(0, 0), (1, 1), (1, 9)].into_iter();
        let decision = policy.evaluate_prepared_with_load(
            &workers,
            Some(&prepared),
            &[0, 1],
            |index| {
                let (expected_index, value) = reads.next().expect("unexpected load read");
                assert_eq!(index, expected_index);
                value
            },
            pick_tied,
        );
        assert!(reads.next().is_none());
        assert_eq!(
            (decision.worker_index(), decision.outcome),
            (Some(1), Some("prefix_hit"))
        );
    }

    #[test]
    fn prepared_baseline_keeps_strict_thresholds_and_missing_input_fallback() {
        let (policy, workers) = cache_fixture(
            2,
            CacheAwareConfig {
                balance_abs_threshold: 1,
                balance_rel_threshold: 2.0,
                ..Default::default()
            },
        );
        let ids = [1, 2, 3, 4];
        policy.select_worker_with_inputs(
            std::slice::from_ref(&workers[1]),
            KvRequestTokens::Shared(Some(&ids[..2])),
            None,
            None,
            None,
            "prefill",
        );
        workers[1].increment_load();
        let mut prepared = policy
            .prepare_selection(
                &workers,
                KvRequestTokens::Shared(Some(&ids)),
                None,
                None,
                None,
            )
            .unwrap();
        let view = policy.selection_view(&workers, Some(&prepared));
        let equal_threshold = policy.evaluate_prepared(&view, view.indices());
        assert_eq!(
            (equal_threshold.worker_index(), equal_threshold.outcome),
            (Some(0), Some("prefix_miss"))
        );

        // Use full affinity, while 4 == 2 * relative threshold keeps the pool balanced.
        prepared.candidates.get_mut(workers[1].url()).unwrap().1 = 1.0;
        for _ in 0..2 {
            workers[0].increment_load();
        }
        for _ in 0..3 {
            workers[1].increment_load();
        }
        let view = policy.selection_view(&workers, Some(&prepared));
        let equal_relative = policy.evaluate_prepared(&view, view.indices());
        assert_eq!(
            (equal_relative.worker_index(), equal_relative.outcome),
            (Some(1), Some("prefix_hit"))
        );
        workers[1].increment_load();
        let view = policy.selection_view(&workers, Some(&prepared));
        let imbalanced = policy.evaluate_prepared(&view, view.indices());
        assert_eq!(
            (imbalanced.worker_index(), imbalanced.outcome),
            (Some(0), Some("load_imbalance"))
        );

        // A partial preparation falls back for the whole subset and earns no tree update.
        prepared.candidates.remove(workers[0].url());
        let view = policy.selection_view(&workers, Some(&prepared));
        let missing = policy.evaluate_prepared(&view, view.indices());
        assert_eq!(
            (missing.worker_index(), missing.outcome),
            (Some(0), Some("missing_tokens"))
        );
        assert!(missing.cache_update.is_none());
        let complete_subset = policy.evaluate_prepared(&view, &[1]);
        assert_eq!(
            (complete_subset.worker_index(), complete_subset.outcome),
            (Some(1), Some("prefix_hit"))
        );
    }

    #[test]
    fn prepared_empty_selection_preserves_missing_token_metric() {
        let (policy, workers) = cache_fixture(1, CacheAwareConfig::default());
        let ids = [1];
        let prepared = policy
            .prepare_selection(
                &workers,
                KvRequestTokens::Shared(Some(&ids)),
                None,
                None,
                None,
            )
            .unwrap();
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _metrics = metrics::set_default_local_recorder(&recorder);
        assert_eq!(
            policy.select_prepared(&[], Some(&prepared), "prefill"),
            None
        );
        assert!(handle.render().is_empty());
        assert_eq!(policy.select_prepared(&[], None, "prefill"), None);
        let rendered = handle.render();
        assert!(
            rendered.lines().any(|line| {
                line.starts_with("vllm_router_prefix_routing_decisions_total")
                    && line.contains("outcome=\"missing_tokens\"")
                    && line.ends_with(" 1")
            }),
            "{rendered}"
        );
        assert!(!rendered.contains("vllm_router_processed_requests_total"));
    }

    #[test]
    fn cache_identity_scopes_share_budget_without_sharing_prefix_credit() {
        let config = CacheAwareConfig {
            cache_threshold: 0.4,
            max_tree_size: 256,
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config.clone());
        let workers: Vec<Arc<dyn Worker>> = (0..2)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://w{i}:8000"),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect();
        workers[1].increment_load();
        let ids = [1, 2, 3, 4];
        for (lora, salt) in [
            (None, None),
            (Some("adapter"), None),
            (None, Some("salt")),
            (Some("adapter"), Some("salt")),
        ] {
            let fresh = policy
                .prepare_selection(
                    &workers,
                    KvRequestTokens::Shared(Some(&ids)),
                    Some("model"),
                    lora,
                    salt,
                )
                .unwrap();
            assert_eq!(fresh.candidates[workers[1].url()].1, 0.0);
            policy.select_worker_with_inputs(
                std::slice::from_ref(&workers[1]),
                KvRequestTokens::Shared(Some(&ids)),
                Some("model"),
                lora,
                salt,
                "regular",
            );
            let partial = policy
                .prepare_selection(
                    &workers,
                    KvRequestTokens::Shared(Some(&[1, 2, 9, 10])),
                    Some("model"),
                    lora,
                    salt,
                )
                .unwrap();
            // Namespace characters must not inflate either token count.
            assert_eq!(partial.candidates[workers[1].url()].1, 0.5);
            assert_eq!(
                policy.select_prepared(&workers, Some(&partial), "prefill"),
                Some(1)
            );
        }
        // Each salted prompt fits the budget individually. Their combined
        // history must still be reclaimed, for every worker, on each cycle.
        for cycle in 0..2 {
            for session in 0..32 {
                let salt = format!("session-{cycle}-{session}");
                for worker in &workers {
                    policy.select_worker_with_inputs(
                        std::slice::from_ref(worker),
                        KvRequestTokens::Shared(Some(&ids)),
                        Some("model"),
                        None,
                        Some(&salt),
                        "prefill",
                    );
                }
            }
            assert_eq!(policy.trees.len(), 1);
            let tree = policy.trees.get("model").unwrap();
            for worker in &workers {
                assert!(tree.get_tenant_char_count()[worker.url()] > config.max_tree_size);
            }
            drop(tree);
            policy.evict_cache(config.max_tree_size);
            let tree = policy.trees.get("model").unwrap();
            for worker in &workers {
                assert!(tree.get_tenant_char_count()[worker.url()] <= config.max_tree_size);
            }
        }
    }

    #[test]
    fn discovered_pools_use_all_effective_batch_prompts_without_initialization() {
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
        let seed: Arc<[Vec<u32>]> = Arc::from(vec![vec![1], vec![7; 64]]);
        let seed_inputs = workers
            .iter()
            .map(|worker| (worker.base_url().to_owned(), seed.clone()))
            .collect();
        // Discovery can first expose only rank/worker 1. No init_workers call.
        assert_eq!(
            policy.select_worker_with_inputs(
                std::slice::from_ref(&workers[1]),
                KvRequestTokens::PerWorker(&seed_inputs),
                Some("model-a"),
                None,
                None,
                "prefill"
            ),
            Some(0)
        );
        workers[1].increment_load();
        let query: Arc<[Vec<u32>]> = Arc::from(vec![vec![999], vec![7; 64]]);
        let query_inputs = workers
            .iter()
            .map(|worker| (worker.base_url().to_owned(), query.clone()))
            .collect();
        // Prompt 0 misses. The long second prompt must still retain affinity.
        assert_eq!(
            policy.select_worker_with_inputs(
                &workers,
                KvRequestTokens::PerWorker(&query_inputs),
                Some("model-a"),
                None,
                None,
                "prefill"
            ),
            Some(1)
        );
        // Unlabeled PD workers must not share hints across request models.
        assert_eq!(
            policy.select_worker_with_inputs(
                &workers,
                KvRequestTokens::PerWorker(&query_inputs),
                Some("model-b"),
                None,
                None,
                "prefill",
            ),
            Some(0)
        );
        // A salt identifies a distinct engine cache and historical hint scope.
        assert_eq!(
            policy.select_worker_with_inputs(
                &workers,
                KvRequestTokens::PerWorker(&query_inputs),
                Some("model-a"),
                None,
                Some("another-cache"),
                "prefill"
            ),
            Some(0)
        );
    }

    #[test]
    fn incomplete_effective_inputs_do_not_claim_a_prefix_hit() {
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
        let ids = [1, 2, 3, 4];
        policy
            .select_worker_with_inputs(
                std::slice::from_ref(&workers[1]),
                KvRequestTokens::Shared(Some(&ids)),
                None,
                None,
                None,
                "regular",
            )
            .unwrap();
        workers[1].increment_load();
        let mut inputs = crate::policies::WorkerTokenInputs::new();
        inputs.insert(
            workers[1].base_url().to_owned(),
            Arc::from(vec![ids.to_vec()]),
        );
        assert_eq!(
            policy.select_worker_with_inputs(
                &workers,
                KvRequestTokens::PerWorker(&inputs),
                None,
                None,
                None,
                "regular"
            ),
            Some(0)
        );
        // Restoring the missing worker restores a real prefix comparison.
        inputs.insert(
            workers[0].base_url().to_owned(),
            Arc::from(vec![ids.to_vec()]),
        );
        assert_eq!(
            policy.select_worker_with_inputs(
                &workers,
                KvRequestTokens::PerWorker(&inputs),
                None,
                None,
                None,
                "regular"
            ),
            Some(1)
        );
    }

    #[test]
    fn content_keys_distinguish_misses_and_reusable_prefixes() {
        let key = |value| {
            serde_json::from_value::<CompletionRequest>(value)
                .unwrap()
                .extract_cache_routing_key()
        };
        let pairs = [
            (
                serde_json::json!({"prompt": [1,2,3,4]}),
                serde_json::json!({"prompt": [2,3,4,5]}),
                1,
            ),
            (
                serde_json::json!({"prompt": [1]}),
                serde_json::json!({"prompt": [2]}),
                1,
            ),
            (
                serde_json::json!({"prompt": [1,2,3,4]}),
                serde_json::json!({"prompt": [1,2,9,8]}),
                1,
            ),
            (
                serde_json::json!({"prompt": "a", "session_params": {"session_id": "sess-0001"}}),
                serde_json::json!({"prompt": "a", "session_params": {"session_id": "sess-0002"}}),
                1,
            ),
            (
                serde_json::json!({"prompt": [1,2,3,4]}),
                serde_json::json!({"prompt": [1,2,3,5]}),
                0,
            ),
        ];
        for (seed, query, expected) in pairs {
            let (policy, workers) = cache_fixture(2, CacheAwareConfig::default());
            let seed = key(seed);
            policy.commit_placement(workers[0].model_id(), &seed, workers[0].url());
            workers[0].increment_load();
            // The absolute imbalance guard does not fire at (1,0).
            // Only a prefix strictly above cache_threshold keeps worker 0.
            assert_eq!(
                policy.select_worker(&workers, Some(&key(query))),
                Some(expected)
            );
            // The original complete key still has affinity to worker 0.
            assert_eq!(policy.select_worker(&workers, Some(&seed)), Some(0));
        }
    }

    #[test]
    fn long_sessions_balance_shared_opening_and_reuse_their_full_history() {
        // Cover both a partial and a complete match in the old 2048-token window.
        for common_tokens in [1800, 8192] {
            // Use the default tree budget; the fixture disables periodic eviction.
            let (policy, workers) = cache_fixture(
                4,
                CacheAwareConfig {
                    cache_threshold: 0.3,
                    balance_abs_threshold: 64,
                    balance_rel_threshold: 1.5,
                    ..Default::default()
                },
            );
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
            // Load ties pick a random worker, so only the balance is fixed.
            let mut sorted = counts;
            sorted.sort_unstable_by(|a, b| b.cmp(a));
            assert_eq!(sorted, [8, 8, 7, 7], "shared tokens: {common_tokens}");

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
        let (policy, workers) = cache_fixture(
            2,
            CacheAwareConfig {
                cache_threshold: 0.3,
                ..Default::default()
            },
        );
        let chat_key = |suffix: &str| {
            cache_routing::for_route(
                &serde_json::json!({"messages": [{"role": "user", "content":
                    format!("{}{}", "界".repeat(18_000), suffix.repeat(82_000))}]}),
                "/v1/chat/completions",
            )
        };
        let first = chat_key("甲");
        let owner = policy.select_worker(&workers, Some(&first)).unwrap();
        workers[owner].increment_load();
        assert_eq!(
            policy.select_worker(&workers, Some(&chat_key("乙"))),
            Some(1 - owner)
        );
        assert_eq!(policy.select_worker(&workers, Some(&first)), Some(owner));
    }

    fn select_tokens(policy: &CacheAwarePolicy, workers: &[Arc<dyn Worker>], ids: &[u32]) -> usize {
        policy
            .select_worker_with_inputs(
                workers,
                KvRequestTokens::Shared(Some(ids)),
                Some("model"),
                None,
                None,
                "regular",
            )
            .unwrap()
    }

    /// Selects 64 times and records which workers were chosen.
    /// A two-way tie misses a given candidate with probability 2^-64; the two
    /// events are disjoint, so this check fails with probability 2^-63.
    fn reached(worker_count: usize, mut select: impl FnMut(u32) -> usize) -> Vec<bool> {
        let mut seen = vec![false; worker_count];
        for round in 0..64 {
            seen[select(round)] = true;
        }
        seen
    }

    #[test]
    fn least_load_ties_spread_without_letting_a_busier_worker_win() {
        let imbalanced = CacheAwareConfig {
            balance_abs_threshold: 0,
            balance_rel_threshold: 1.0,
            ..Default::default()
        };
        for (path, config) in [
            ("prefix_miss", CacheAwareConfig::default()),
            ("missing_tokens", CacheAwareConfig::default()),
            ("load_imbalance", imbalanced),
        ] {
            let (policy, workers) = cache_fixture(3, config);
            workers[2].increment_load();
            if path == "load_imbalance" {
                // Only the imbalance guard may move this prompt off worker 2.
                select_tokens(&policy, std::slice::from_ref(&workers[2]), &[7; 8]);
            }
            let seen = reached(3, |round| match path {
                "prefix_miss" => select_tokens(&policy, &workers, &[round + 100; 8]),
                "missing_tokens" => policy.select_prepared(&workers, None, "regular").unwrap(),
                _ => select_tokens(&policy, &workers, &[7; 8]),
            });
            assert_eq!(seen, [true, true, false], "{path}");
        }
    }

    #[test]
    fn equal_prefix_matches_spread_ties_and_keep_the_match() {
        let (policy, workers) = cache_fixture(3, CacheAwareConfig::default());
        let ids = [7; 8];
        // Worker 2 is idle but has no history, so it must never outrank a match.
        for worker in &workers[..2] {
            select_tokens(&policy, std::slice::from_ref(worker), &ids);
            worker.increment_load();
        }
        let seen = reached(3, |_| select_tokens(&policy, &workers, &ids));
        assert_eq!(seen, [true, true, false]);
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
            policy.choose_from_affinity(&workers, &[0, 1], Some(&cold), "unrelated text"),
            Some((1, false))
        );
        workers[0].set_healthy(false);
        let healthy = get_healthy_worker_indices(&workers);
        assert_eq!(
            policy.choose_from_affinity(&workers, &healthy, Some(&warm), "warm conversation"),
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
