//! Least-connection sticky session routing policy
//!
//! New sessions are assigned to the least-loaded healthy worker.
//! Returning sessions (by X-Session-Id) are routed to the same worker.

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::policies::hash_key;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::info;

/// Least-connection sticky session policy
///
/// New sessions are assigned to the least-loaded healthy worker.
/// Returning sessions are routed to the previously assigned worker.
/// If the assigned worker becomes unhealthy, the session is re-assigned.
#[derive(Debug)]
pub struct LeastConnStickyPolicy {
    /// Maps session keys to their assigned worker URLs.
    /// TODO: this map grows without bound; add TTL-based eviction (see vllm-project/router#235
    /// for an implementation with session expiry and explicit release).
    session_map: RwLock<HashMap<String, String>>,
    /// Cached load information from external monitoring
    cached_loads: RwLock<HashMap<String, isize>>,
    /// Tracks session assignments per worker for load-aware placement of new sessions.
    /// The HTTP layer's in-flight count (increment_load) is updated after policy selection,
    /// so it reads as zero for all workers during early startup, causing all new sessions
    /// to pile onto the first backend. This counter is updated immediately at selection time.
    session_counts: RwLock<HashMap<String, usize>>,
}

impl LeastConnStickyPolicy {
    pub fn new() -> Self {
        Self {
            session_map: RwLock::new(HashMap::new()),
            cached_loads: RwLock::new(HashMap::new()),
            session_counts: RwLock::new(HashMap::new()),
        }
    }

    fn get_worker_load(&self, worker: &dyn Worker) -> isize {
        if let Ok(loads) = self.cached_loads.read() {
            if let Some(&load) = loads.get(worker.url()) {
                return load;
            }
        }
        let base = worker.load() as isize;
        let extra = self.session_counts
            .read()
            .ok()
            .and_then(|c| c.get(worker.url()).copied())
            .unwrap_or(0) as isize;
        base + extra
    }

    fn find_least_loaded(&self, workers: &[Arc<dyn Worker>], healthy_indices: &[usize]) -> usize {
        let mut best_idx = healthy_indices[0];
        let mut best_load = self.get_worker_load(workers[best_idx].as_ref());
        for &idx in &healthy_indices[1..] {
            let load = self.get_worker_load(workers[idx].as_ref());
            if load < best_load {
                best_load = load;
                best_idx = idx;
            }
        }
        best_idx
    }
}

impl LoadBalancingPolicy for LeastConnStickyPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return None;
        }

        // TODO: make the session header (currently X-Session-Id) configurable;
        // consistent_hash has the same limitation via shared hash_key module
        let session_key = hash_key::extract_hash_key(request_text, headers);

        // Check if this session is already mapped to a healthy worker
        if let Ok(map) = self.session_map.read() {
            if let Some(worker_url) = map.get(&session_key) {
                if let Some(idx) = healthy_indices
                    .iter()
                    .find(|&&i| workers[i].url() == worker_url)
                    .copied()
                {
                    info!(
                        "Least-conn-sticky: existing session '{}' -> worker '{}' (index={})",
                        session_key,
                        workers[idx].url(),
                        idx
                    );
                    workers[idx].increment_processed();
                    RouterMetrics::record_processed_request(workers[idx].url());
                    RouterMetrics::record_policy_decision(self.name(), workers[idx].url());
                    return Some(idx);
                }
            }
        }

        // New session or mapped worker is unhealthy: pick least loaded
        let selected_idx = self.find_least_loaded(workers, &healthy_indices);

        let worker_url = workers[selected_idx].url().to_string();
        if let Ok(mut map) = self.session_map.write() {
            map.insert(session_key.clone(), worker_url.clone());
        }
        if let Ok(mut counts) = self.session_counts.write() {
            *counts.entry(worker_url).or_insert(0) += 1;
        }

        info!(
            "Least-conn-sticky: new session '{}' -> worker '{}' (index={}, load={})",
            session_key,
            workers[selected_idx].url(),
            selected_idx,
            self.get_worker_load(workers[selected_idx].as_ref())
        );

        workers[selected_idx].increment_processed();
        RouterMetrics::record_processed_request(workers[selected_idx].url());
        RouterMetrics::record_policy_decision(self.name(), workers[selected_idx].url());

        Some(selected_idx)
    }

    fn name(&self) -> &'static str {
        "least_conn_sticky"
    }

    fn needs_headers(&self) -> bool {
        true
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

impl Default for LeastConnStickyPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    #[test]
    fn test_new_sessions_go_to_least_loaded() {
        let policy = LeastConnStickyPolicy::new();
        let worker1 = BasicWorker::new("http://w1:8000".to_string(), WorkerType::Regular);
        let worker2 = BasicWorker::new("http://w2:8000".to_string(), WorkerType::Regular);
        let worker3 = BasicWorker::new("http://w3:8000".to_string(), WorkerType::Regular);

        for _ in 0..10 {
            worker1.increment_load();
        }
        for _ in 0..5 {
            worker2.increment_load();
        }

        let workers: Vec<Arc<dyn Worker>> =
            vec![Arc::new(worker1), Arc::new(worker2), Arc::new(worker3)];

        let mut headers = HashMap::new();
        headers.insert("x-session-id".to_string(), "sess-new".to_string());

        let idx = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();
        assert_eq!(idx, 2); // worker3 has load 0
    }

    #[test]
    fn test_returning_session_is_sticky() {
        let policy = LeastConnStickyPolicy::new();
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

        let mut headers = HashMap::new();
        headers.insert("x-session-id".to_string(), "sess-sticky".to_string());

        let first_idx = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();

        // Increase load on the selected worker so it's no longer least-loaded
        for _ in 0..100 {
            workers[first_idx].increment_load();
        }

        // Same session should still go to the same worker
        let second_idx = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();
        assert_eq!(first_idx, second_idx);
    }

    #[test]
    fn test_unhealthy_worker_reassignment() {
        let policy = LeastConnStickyPolicy::new();
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

        let mut headers = HashMap::new();
        headers.insert("x-session-id".to_string(), "sess-failover".to_string());

        let first_idx = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();

        // Mark the assigned worker as unhealthy
        workers[first_idx].set_healthy(false);

        let second_idx = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();
        assert_ne!(first_idx, second_idx);
    }

    #[test]
    fn test_single_worker() {
        let policy = LeastConnStickyPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(
            "http://w1:8000".to_string(),
            WorkerType::Regular,
        ))];

        assert_eq!(
            policy.select_worker_with_headers(&workers, None, None),
            Some(0)
        );
    }
}
