//! Load balancing policies for VLLM router
//!
//! This module provides a unified abstraction for routing policies that work
//! across both regular and prefill-decode (PD) routing modes.

use crate::core::Worker;
use crate::protocols::spec::RoutingPrompt;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
mod smetric;

pub use cache_aware::{CacheAwareCandidate, CacheAwarePlacement, CacheAwarePolicy};
pub use consistent_hash::ConsistentHashPolicy;
pub use consistent_hash::VIRTUAL_NODES_PER_WORKER;
pub use factory::PolicyFactory;
pub use kv_aware::{
    cost_components, kv_aware_ops, CandidateView, CostComponents, CostDecision, CostParams,
    InMemorySlotManager, KvAwareOps, KvAwarePolicy, KvRequestSlot, KvSelectionReceipt,
    LoadProvider, LoadSample, MultiplicativeScorer, Picker, Scorer, SelectionError, SlotManager,
    TemperaturePicker, TieredMatchProvider, TransferCredits,
};
pub use power_of_two::PowerOfTwoPolicy;
pub use random::RandomPolicy;
pub use registry::PolicyRegistry;
pub use rendezvous_hash::RendezvousHashPolicy;
pub use round_robin::RoundRobinPolicy;

/// HTTP headers passed to policies for routing decisions
/// Key is lowercase header name, value is header value
pub type RequestHeaders = HashMap<String, String>;

#[derive(Clone, Copy, Debug)]
pub struct KvRequestContext<'a> {
    pub token_ids: &'a [u32],
    pub lora_name: Option<&'a str>,
    pub cache_salt: Option<&'a str>,
}

/// Optional request metadata for affinity-aware selection.
#[derive(Clone, Default)]
pub struct RoutingContext<'a> {
    pub prompt: Option<&'a RoutingPrompt>,
    pub kv: Option<KvRequestContext<'a>>,
    pub allow_affinity: bool,
    pub colocated: bool,
    pub route: &'a str,
}

/// Number of consecutive pure-decode tokens aggregated into one TPOT sample.
pub const DECODE_SAMPLE_TOKENS: u32 = 8;

/// Shared per-worker state; trackers reuse its Arc rather than allocate callbacks.
pub trait RequestObserver: Send + Sync {
    fn charge(&self, work: u64);
    /// Release charged work; elapsed is present only for a first-token observation.
    fn finish(&self, work: u64, elapsed: Option<Duration>);
    /// Release `work` from pending while recording `sample_work / elapsed` when learning prefill rate.
    fn finish_with_sample(&self, work: u64, sample_work: u64, elapsed: Option<Duration>) {
        let _ = sample_work;
        self.finish(work, elapsed);
    }
    /// Return the count of *other* co-resident requests on this worker iff no
    /// prefill work is currently pending (`pending == 0`), or `None` when
    /// contaminated by concurrent prefill or when online decode learning is off.
    fn pure_decode_peers(&self) -> Option<usize> {
        None
    }
    /// Record one pure-decode inter-token latency sample `(peers, tpot_secs)`.
    fn record_decode_step(&self, _peers: usize, _tpot_secs: f64) {}
}

#[derive(Clone, Copy, Debug)]
enum TrackerPhase {
    Prefill {
        started: Instant,
    },
    Decode {
        /// `(window_start, peers, tokens_accumulated)` during an uncontaminated pure-decode run.
        window_anchor: Option<(Instant, usize, u32)>,
    },
    Done,
}

/// Own one prefill work reservation (released once on first token or Drop) and
/// optionally sample pure-decode inter-token intervals across streaming output.
pub struct RequestTracker {
    observer: Option<Arc<dyn RequestObserver>>,
    work: u64,
    sample_work: u64,
    sample_rate: bool,
    estimated_prefill: Option<Duration>,
    phase: TrackerPhase,
}

impl RequestTracker {
    pub fn new(observer: Arc<dyn RequestObserver>, work: u64) -> Self {
        Self::new_with_sample(observer, work, work, None)
    }

    pub(crate) fn new_with_sample(
        observer: Arc<dyn RequestObserver>,
        work: u64,
        sample_work: u64,
        estimated_prefill: Option<Duration>,
    ) -> Self {
        observer.charge(work);
        Self {
            observer: Some(observer),
            work,
            sample_work,
            sample_rate: true,
            estimated_prefill,
            phase: TrackerPhase::Prefill {
                started: Instant::now(),
            },
        }
    }

    pub(crate) fn without_rate_sample(mut self) -> Self {
        self.sample_rate = false;
        self
    }

    pub(crate) fn estimated_prefill(&self) -> Option<Duration> {
        self.estimated_prefill
    }

    pub(crate) fn release_pending_without_sample(&mut self) {
        if matches!(self.phase, TrackerPhase::Prefill { .. }) {
            if let Some(observer) = self.observer.take() {
                observer.finish_with_sample(self.work, self.sample_work, None);
            }
            self.phase = TrackerPhase::Done;
        }
    }

    pub fn on_first_token(&mut self) {
        self.on_output_tokens(1);
    }

    /// Signal that output has started (releasing prefill work and recording TTFT)
    /// without contributing an unverified token count to decode TPOT / beta learning.
    pub fn on_first_output_unverified(&mut self) {
        let Some(observer) = self.observer.as_ref() else {
            return;
        };
        if let TrackerPhase::Prefill { started } = self.phase {
            let now = Instant::now();
            let elapsed = self
                .sample_rate
                .then(|| now.saturating_duration_since(started));
            observer.finish_with_sample(self.work, self.sample_work, elapsed);
            self.phase = TrackerPhase::Decode {
                window_anchor: None,
            };
        }
    }

    pub fn on_output_tokens(&mut self, count: u32) {
        if count == 0 {
            return;
        }
        let Some(observer) = self.observer.as_ref() else {
            return;
        };
        let now = Instant::now();
        match &mut self.phase {
            TrackerPhase::Prefill { started } => {
                let elapsed = self
                    .sample_rate
                    .then(|| now.saturating_duration_since(*started));
                observer.finish_with_sample(self.work, self.sample_work, elapsed);
                let window_anchor = observer.pure_decode_peers().map(|peers| (now, peers, 0));
                self.phase = TrackerPhase::Decode { window_anchor };
            }
            TrackerPhase::Decode { window_anchor } => match observer.pure_decode_peers() {
                None => {
                    // Concurrent prefill started on this worker; discard partial decode window.
                    *window_anchor = None;
                }
                Some(peers) => match *window_anchor {
                    None => {
                        *window_anchor = Some((now, peers, 0));
                    }
                    Some((_, win_peers, _)) if win_peers != peers => {
                        *window_anchor = Some((now, peers, 0));
                    }
                    Some((win_start, _, win_tokens)) => {
                        let total_tokens = win_tokens.saturating_add(count);
                        if total_tokens >= DECODE_SAMPLE_TOKENS {
                            let dt = now.saturating_duration_since(win_start).as_secs_f64();
                            if dt > 0.0 {
                                observer.record_decode_step(peers, dt / f64::from(total_tokens));
                            }
                            *window_anchor = Some((now, peers, 0));
                        } else {
                            *window_anchor = Some((win_start, peers, total_tokens));
                        }
                    }
                },
            },
            TrackerPhase::Done => {}
        }
    }
}

impl Drop for RequestTracker {
    fn drop(&mut self) {
        self.release_pending_without_sample();
    }
}

/// Core trait for load balancing policies
///
/// This trait provides a unified interface for implementing routing algorithms
/// that can work with both regular single-worker selection and PD dual-worker selection.
pub trait LoadBalancingPolicy: Send + Sync + Debug {
    fn on_worker_removed(&self, _worker: &Arc<dyn Worker>) {}

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

    /// Select with optional affinity metadata and request-scoped tracking.
    fn select_request(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
        _context: &RoutingContext<'_>,
    ) -> Option<(usize, Option<RequestTracker>)> {
        self.select_worker_with_headers(workers, request_text, headers)
            .map(|idx| (idx, None))
    }

    /// Request typed affinity metadata only when the policy consumes it.
    fn needs_affinity_prompt(&self) -> bool {
        false
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
    pub max_tree_size: usize,
}

impl Default for CacheAwareConfig {
    fn default() -> Self {
        Self {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 30,
            max_tree_size: 10000,
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

#[cfg(test)]
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
