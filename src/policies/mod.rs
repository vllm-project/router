//! Load balancing policies for VLLM router
//!
//! This module provides a unified abstraction for routing policies that work
//! across both regular and prefill-decode (PD) routing modes.

use crate::core::{Worker, WorkerLoadGuard};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

mod cache_aware;
mod consistent_hash;
mod factory;
pub(crate) mod hash_key;
mod kv_aware;
mod power_of_two;
mod random;
mod registry;
mod rendezvous_hash;
mod round_robin;

pub(crate) use cache_aware::PreparedCacheSelection;
pub use cache_aware::{CacheAwareCandidate, CacheAwarePlacement, CacheAwarePolicy};
pub use consistent_hash::ConsistentHashPolicy;
pub use consistent_hash::VIRTUAL_NODES_PER_WORKER;
pub use factory::PolicyFactory;
#[cfg(all(test, feature = "cpu", feature = "ready"))]
pub(crate) use kv_aware::kv_aware_policy;
pub use kv_aware::{
    cost_components, kv_aware_ops, AffinityKey, AffinityRecord, CandidateView, CostComponents,
    CostDecision, CostParams, InMemorySlotManager, KvAffinityTable, KvAwareOps, KvAwarePolicy,
    KvRequestSlot, KvRequestTokens, KvSelectionReceipt, LoadProvider, LoadSample,
    MultiplicativeScorer, Picker, PrefillWorkEstimate, Scorer, SelectionError, SlotManager,
    StagePick, TemperaturePicker, TieredMatchProvider, TransferCredits, WorkerTokenInputs,
};
pub use power_of_two::PowerOfTwoPolicy;
pub use random::RandomPolicy;
pub use registry::PolicyRegistry;
pub use rendezvous_hash::RendezvousHashPolicy;
pub use round_robin::RoundRobinPolicy;

/// HTTP headers passed to policies for routing decisions
/// Key is lowercase header name, value is header value
pub type RequestHeaders = HashMap<String, String>;

/// Coordinates selection and reservation for one worker pool.
///
/// Reuse the same context for admissions to the same pool. It belongs to the
/// caller, not the policy: one policy may be shared by separate P and D pools.
#[derive(Debug, Default)]
pub struct SelectionContext {
    lock: parking_lot::Mutex<()>,
}

/// Core trait for load balancing policies
///
/// This trait provides a unified interface for implementing routing algorithms
/// that can work with both regular single-worker selection and PD dual-worker selection.
pub trait LoadBalancingPolicy: Send + Sync + Debug {
    /// Select a single worker from the available workers
    ///
    /// This is used for regular routing mode where requests go to a single worker.
    /// Now uses Arc<dyn Worker> for better performance and to avoid unnecessary cloning.
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
    ) -> Option<usize> {
        // Default implementation delegates to select_worker_with_headers without headers
        self.select_worker_with_headers(workers, request_text, None)
    }

    /// Select a single worker with optional HTTP headers for routing decisions
    ///
    /// Policies like consistent_hash can use headers (e.g., X-Session-ID) for routing.
    /// Default implementation ignores headers and falls back to basic selection.
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize>;

    /// Select a worker and reserve its local load before another admission
    /// through this context can make a load-based decision.
    ///
    /// The caller owns the returned guard's lifetime. Policies with expensive
    /// preparation can override this hook, keeping only the final decision and
    /// reservation under the context lock. The default preserves existing policy
    /// behavior, including policies which maintain their own selection state.
    fn select_and_reserve<'a>(
        &self,
        context: &SelectionContext,
        workers: &'a [Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<(usize, WorkerLoadGuard<'a>)> {
        let _selection = context.lock.lock();
        let idx = self.select_worker_with_headers(workers, request_text, headers)?;
        Some((idx, WorkerLoadGuard::new(workers[idx].as_ref())))
    }

    /// Select a pair of workers (prefill and decode) for PD routing
    ///
    /// Returns indices of (prefill_worker, decode_worker) from their respective arrays.
    /// Default implementation uses select_worker for each array independently.
    fn select_worker_pair(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
    ) -> Option<(usize, usize)> {
        self.select_worker_pair_with_headers(prefill_workers, decode_workers, request_text, None)
    }

    /// Select a pair of workers with optional HTTP headers for routing decisions
    fn select_worker_pair_with_headers(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<(usize, usize)> {
        // Default implementation: independently select from each pool
        let prefill_idx =
            self.select_worker_with_headers(prefill_workers, request_text, headers)?;
        let decode_idx = self.select_worker_with_headers(decode_workers, request_text, headers)?;
        Some((prefill_idx, decode_idx))
    }

    /// Update policy state after request completion
    ///
    /// This is called when a request completes (successfully or not) to allow
    /// policies to update their internal state.
    fn on_request_complete(&self, _worker_url: &str, _success: bool) {
        // Default: no-op for stateless policies
    }

    /// Get policy name for metrics and debugging
    fn name(&self) -> &'static str;

    /// Check if this policy needs request text for routing decisions
    fn needs_request_text(&self) -> bool {
        false // Default: most policies don't need request text
    }

    /// Whether request text must use the content-preserving cache routing key.
    fn needs_cache_routing_key(&self) -> bool {
        false
    }

    /// Whether the router must track requests in flight for this policy.
    fn tracks_request_load(&self) -> bool {
        false
    }

    /// Check if this policy needs HTTP headers for routing decisions
    fn needs_headers(&self) -> bool {
        false // Default: most policies don't need headers
    }

    /// Update worker load information
    ///
    /// This is called periodically with current load information for load-aware policies.
    fn update_loads(&self, _loads: &std::collections::HashMap<String, isize>) {
        // Default: no-op for policies that don't use load information
    }

    /// Reset any internal state
    ///
    /// This is useful for policies that maintain state (e.g., round-robin counters).
    fn reset(&self) {
        // Default: no-op for stateless policies
    }

    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;

    /// Returns true if this policy requires init_workers() to be called before use.
    /// Override to return true for stateful policies like cache-aware routing.
    fn requires_initialization(&self) -> bool {
        false // Default: most policies don't need initialization
    }

    /// Initialize the policy with a set of workers.
    /// Override for stateful policies that need to set up internal data structures.
    /// Default is no-op for stateless policies like round-robin.
    fn init_workers(&self, _workers: &[Arc<dyn Worker>]) {
        // Default: no-op for policies that don't need initialization
    }
}

/// Configuration for cache-aware policy
#[derive(Debug, Clone)]
pub struct CacheAwareConfig {
    pub cache_threshold: f32,
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    pub eviction_interval_secs: u64,
    /// Per-worker character budget in each model tree, enforced by periodic eviction.
    pub max_tree_size: usize,
}

impl Default for CacheAwareConfig {
    fn default() -> Self {
        Self {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 30,
            max_tree_size: 1 << 25,
        }
    }
}

/// Helper function to filter healthy workers and return their indices
pub(crate) fn get_healthy_worker_indices(workers: &[Arc<dyn Worker>]) -> Vec<usize> {
    workers
        .iter()
        .enumerate()
        .filter(|(_, w)| w.is_healthy() && w.circuit_breaker().can_execute())
        .map(|(idx, _)| idx)
        .collect()
}

/// Helper function to normalize model_id to a key for policy lookups.
///
/// Use "default" for unknown/empty model_ids for backward compatibility
#[inline]
pub(crate) fn normalize_model_key(model_id: &str) -> &str {
    if model_id.is_empty() || model_id == "unknown" {
        "default"
    } else {
        model_id
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    #[test]
    fn test_get_healthy_worker_indices() {
        let workers: Vec<Arc<dyn Worker>> = vec![
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
        ];

        // All healthy initially
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 1, 2]);

        // Mark one unhealthy
        workers[1].set_healthy(false);
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 2]);
    }
}
