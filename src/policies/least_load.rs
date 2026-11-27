//! Least-load placement. This policy does not remember a session.
//!
//! The signal in this PR is in-flight requests (`Worker::load`). When the
//! session affinity module is on, that module passes its own active-session
//! counts into [`least_by_load`] instead of calling this policy.

use super::{
    get_healthy_worker_indices, ConsistentHashPolicy, LoadBalancingPolicy, RequestHeaders,
};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use std::sync::Arc;

/// Place a request on the healthy worker with the smallest in-flight load.
#[derive(Debug, Default)]
pub struct LeastLoadPolicy;

impl LeastLoadPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl LoadBalancingPolicy for LeastLoadPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }
        let loads: Vec<(usize, usize, String)> = healthy
            .iter()
            .map(|&idx| (idx, workers[idx].load(), workers[idx].url().to_string()))
            .collect();
        let idx = least_by_load(&loads, request_text.unwrap_or(""))?;
        let worker = workers[idx].url();
        RouterMetrics::record_processed_request(worker);
        RouterMetrics::record_policy_decision(self.name(), worker);
        Some(idx)
    }

    fn name(&self) -> &'static str {
        "least_load"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn tracks_request_load(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Fewest load, ties by `hash(tie_break:url)`.
///
/// `loads` is `(worker index, load, worker url)`.
pub fn least_by_load(loads: &[(usize, usize, String)], tie_break: &str) -> Option<usize> {
    let min = loads.iter().map(|(_, load, _)| *load).min()?;
    let mut best: Option<(usize, u64)> = None;
    for (idx, load, url) in loads {
        if *load != min {
            continue;
        }
        let weight = ConsistentHashPolicy::fbi_hash(&format!("{tie_break}:{url}"));
        match best {
            Some((_, best_weight)) if best_weight >= weight => {}
            _ => best = Some((*idx, weight)),
        }
    }
    best.map(|(idx, _)| idx)
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, Worker, WorkerType};

    fn workers(loads: &[usize]) -> Vec<Arc<dyn Worker>> {
        loads
            .iter()
            .enumerate()
            .map(|(i, load)| {
                let worker = Arc::new(BasicWorker::new(
                    format!("http://w{i}:8000"),
                    WorkerType::Regular,
                ));
                for _ in 0..*load {
                    worker.increment_load();
                }
                worker as Arc<dyn Worker>
            })
            .collect()
    }

    #[test]
    fn picks_the_worker_with_fewest_in_flight_requests() {
        let policy = LeastLoadPolicy::new();
        let workers = workers(&[3, 1, 2]);
        assert_eq!(policy.select_worker(&workers, Some("s")).unwrap(), 1);
    }

    #[test]
    fn tie_is_stable_for_the_same_text() {
        let policy = LeastLoadPolicy::new();
        let workers = workers(&[0, 0, 0]);
        let first = policy.select_worker(&workers, Some("session-a")).unwrap();
        let second = policy.select_worker(&workers, Some("session-a")).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn skips_unhealthy_workers() {
        let policy = LeastLoadPolicy::new();
        let workers = workers(&[0, 5]);
        workers[0].set_healthy(false);
        assert_eq!(policy.select_worker(&workers, None).unwrap(), 1);
    }

    #[test]
    fn empty_pool_returns_none() {
        let policy = LeastLoadPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![];
        assert!(policy.select_worker(&workers, None).is_none());
    }
}
