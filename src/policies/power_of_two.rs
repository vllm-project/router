//! Power-of-two choices load balancing policy

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use rand::Rng;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::info;

/// Power-of-two choices policy
///
/// Randomly selects two workers and routes to the one with lower load.
/// This provides good load distribution with minimal coordination overhead.
#[derive(Debug)]
pub struct PowerOfTwoPolicy {
    /// Cached load information from external monitoring
    cached_loads: RwLock<HashMap<String, isize>>,
}

impl PowerOfTwoPolicy {
    pub fn new() -> Self {
        Self {
            cached_loads: RwLock::new(HashMap::new()),
        }
    }

    fn get_worker_load(&self, worker: &dyn Worker) -> isize {
        let local = worker.load() as isize;

        // Combine cached load (external monitoring, e.g. vLLM /load endpoint)
        // with local counter (in-flight requests tracked by router).
        // This avoids thundering herd when cached loads go stale between
        // 5-second polling intervals.
        if let Ok(loads) = self.cached_loads.read() {
            if let Some(&cached) = loads.get(worker.url()) {
                return cached + local;
            }
        }

        local
    }
}

impl LoadBalancingPolicy for PowerOfTwoPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        if healthy_indices.len() == 1 {
            return Some(healthy_indices[0]);
        }

        // Select two random workers
        let mut rng = rand::rng();
        let idx1 = rng.random_range(0..healthy_indices.len());
        let mut idx2 = rng.random_range(0..healthy_indices.len());

        // Ensure we pick two different workers
        while idx2 == idx1 {
            idx2 = rng.random_range(0..healthy_indices.len());
        }

        let worker_idx1 = healthy_indices[idx1];
        let worker_idx2 = healthy_indices[idx2];

        // Compare loads and select the less loaded one
        let load1 = self.get_worker_load(workers[worker_idx1].as_ref());
        let load2 = self.get_worker_load(workers[worker_idx2].as_ref());

        // Log selection for debugging
        let selected_idx = if load1 <= load2 {
            worker_idx1
        } else {
            worker_idx2
        };

        info!(
            "Power-of-two selection: {}={} vs {}={} -> selected {}",
            workers[worker_idx1].url(),
            load1,
            workers[worker_idx2].url(),
            load2,
            workers[selected_idx].url()
        );

        // Increment processed counter
        workers[selected_idx].increment_processed();
        RouterMetrics::record_processed_request(workers[selected_idx].url());
        RouterMetrics::record_policy_decision(self.name(), workers[selected_idx].url());

        Some(selected_idx)
    }

    fn name(&self) -> &'static str {
        "power_of_two"
    }

    fn update_loads(&self, loads: &HashMap<String, isize>) {
        if let Ok(mut cached) = self.cached_loads.write() {
            *cached = loads.clone();
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for PowerOfTwoPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    #[test]
    fn test_power_of_two_selection() {
        let policy = PowerOfTwoPolicy::new();
        let worker1 = BasicWorker::new("http://w1:8000".to_string(), WorkerType::Regular);
        let worker2 = BasicWorker::new("http://w2:8000".to_string(), WorkerType::Regular);
        let worker3 = BasicWorker::new("http://w3:8000".to_string(), WorkerType::Regular);

        // Set different loads
        for _ in 0..10 {
            worker1.increment_load();
        }
        for _ in 0..5 {
            worker2.increment_load();
        }
        // worker3 has load 0

        let workers: Vec<Arc<dyn Worker>> =
            vec![Arc::new(worker1), Arc::new(worker2), Arc::new(worker3)];

        // Run multiple selections
        let mut selected_counts = [0; 3];
        for _ in 0..100 {
            if let Some(idx) = policy.select_worker(&workers, None) {
                selected_counts[idx] += 1;
            }
        }

        // Worker with lowest load (worker3) should be selected most often
        assert!(selected_counts[2] > selected_counts[1]);
        assert!(selected_counts[1] > selected_counts[0]);
    }

    #[test]
    fn test_power_of_two_with_cached_loads() {
        let policy = PowerOfTwoPolicy::new();
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

        // Update cached loads
        let mut loads = HashMap::new();
        loads.insert("http://w1:8000".to_string(), 100);
        loads.insert("http://w2:8000".to_string(), 10);
        policy.update_loads(&loads);

        // Should prefer worker2 with lower cached load
        let mut w2_selected = 0;
        for _ in 0..50 {
            if let Some(idx) = policy.select_worker(&workers, None) {
                if idx == 1 {
                    w2_selected += 1;
                }
            }
        }

        // With only 2 workers, power-of-two always picks the same pair,
        // so the less-loaded one wins every time
        assert_eq!(w2_selected, 50);
    }

    /// Verify that cached_load and local_counter are combined:
    /// When two workers have equal cached_load, local_counter should break the tie.
    /// This prevents thundering herd when cached_load goes stale between polling intervals.
    #[test]
    fn test_power_of_two_combines_cached_and_local_load() {
        let policy = PowerOfTwoPolicy::new();
        let worker1 = BasicWorker::new("http://w1:8000".to_string(), WorkerType::Regular);
        let worker2 = BasicWorker::new("http://w2:8000".to_string(), WorkerType::Regular);

        // Set equal cached_loads, but worker1 has higher local counter
        let mut loads = HashMap::new();
        loads.insert("http://w1:8000".to_string(), 50);
        loads.insert("http://w2:8000".to_string(), 50);
        policy.update_loads(&loads);

        // Increment worker1's local counter
        for _ in 0..5 {
            worker1.increment_load();
        }
        // worker2 local counter remains 0

        // worker1 total load = cached(50) + local(5) = 55
        // worker2 total load = cached(50) + local(0) = 50
        // So worker2 should be selected more often
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker1), Arc::new(worker2)];

        let mut w2_selected = 0;
        for _ in 0..50 {
            if let Some(idx) = policy.select_worker(&workers, None) {
                if idx == 1 {
                    w2_selected += 1;
                }
            }
        }

        // With only 2 workers, the less-loaded one is always selected
        assert_eq!(w2_selected, 50);
    }

    /// Verify that without cached_load, only local_counter is used
    #[test]
    fn test_power_of_two_without_cached_loads_uses_local() {
        let policy = PowerOfTwoPolicy::new();
        let worker1 = BasicWorker::new("http://w1:8000".to_string(), WorkerType::Regular);
        let worker2 = BasicWorker::new("http://w2:8000".to_string(), WorkerType::Regular);

        // No cached_load set, rely only on local counter
        for _ in 0..10 {
            worker1.increment_load();
        }

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker1), Arc::new(worker2)];

        let mut w2_selected = 0;
        for _ in 0..50 {
            if let Some(idx) = policy.select_worker(&workers, None) {
                if idx == 1 {
                    w2_selected += 1;
                }
            }
        }

        // With only 2 workers, the less-loaded one is always selected
        assert_eq!(w2_selected, 50);
    }

    #[test]
    fn test_power_of_two_single_worker() {
        let policy = PowerOfTwoPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(
            "http://w1:8000".to_string(),
            WorkerType::Regular,
        ))];

        // With single worker, should always select it
        assert_eq!(policy.select_worker(&workers, None), Some(0));
    }
}
