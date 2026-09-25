//! Sticky round-robin load balancing policy.
//!
//! New sessions are assigned in round-robin order and remain pinned to that
//! worker while it is healthy. Keyless requests use round-robin without
//! creating an affinity entry.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use dashmap::mapref::entry::Entry;
use dashmap::DashMap;

use super::{get_healthy_worker_indices, hash_key, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::metrics::RouterMetrics;

/// Strict per-session affinity with round-robin placement for new sessions.
#[derive(Debug)]
pub struct StickyRoundRobinPolicy {
    /// Session key to assigned worker URL.
    assignments: DashMap<String, String>,
    /// Cursor used only when assigning a new session or remapping one.
    session_counter: AtomicUsize,
    /// Independent cursor for keyless traffic so it cannot shift session placement.
    keyless_counter: AtomicUsize,
}

impl StickyRoundRobinPolicy {
    pub fn new() -> Self {
        Self {
            assignments: DashMap::new(),
            session_counter: AtomicUsize::new(0),
            keyless_counter: AtomicUsize::new(0),
        }
    }

    fn round_robin_pick(counter: &AtomicUsize, healthy_indices: &[usize]) -> usize {
        let count = counter.fetch_add(1, Ordering::Relaxed);
        healthy_indices[count % healthy_indices.len()]
    }

    fn healthy_index_for(&self, workers: &[Arc<dyn Worker>], url: &str) -> Option<usize> {
        workers.iter().position(|worker| {
            worker.url() == url && worker.is_healthy() && worker.circuit_breaker().can_execute()
        })
    }

    fn record_selection(&self, workers: &[Arc<dyn Worker>], index: usize) {
        let worker = &workers[index];
        worker.increment_processed();
        RouterMetrics::record_processed_request(worker.url());
        RouterMetrics::record_policy_decision(self.name(), worker.url());
    }

    fn select_for_key(
        &self,
        workers: &[Arc<dyn Worker>],
        key: &str,
        sticky: bool,
    ) -> Option<usize> {
        if !sticky {
            let healthy_indices = get_healthy_worker_indices(workers);
            if healthy_indices.is_empty() {
                return None;
            }
            let index = Self::round_robin_pick(&self.keyless_counter, &healthy_indices);
            self.record_selection(workers, index);
            return Some(index);
        }

        if let Some(url) = self
            .assignments
            .get(key)
            .map(|assignment| assignment.value().clone())
        {
            if let Some(index) = self.healthy_index_for(workers, &url) {
                self.record_selection(workers, index);
                return Some(index);
            }
        }

        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return None;
        }

        // The entry guard ensures concurrent first requests for one session
        // agree on a single worker.
        let index = match self.assignments.entry(key.to_string()) {
            Entry::Occupied(mut entry) => {
                if let Some(index) = self.healthy_index_for(workers, entry.get()) {
                    index
                } else {
                    let index = Self::round_robin_pick(&self.session_counter, &healthy_indices);
                    *entry.get_mut() = workers[index].url().to_string();
                    index
                }
            }
            Entry::Vacant(entry) => {
                let index = Self::round_robin_pick(&self.session_counter, &healthy_indices);
                entry.insert(workers[index].url().to_string());
                index
            }
        };

        self.record_selection(workers, index);
        Some(index)
    }
}

impl LoadBalancingPolicy for StickyRoundRobinPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        if workers.is_empty() {
            return None;
        }

        let key = hash_key::extract_hash_key(request_text, headers);
        let sticky = !key.starts_with("request:") && !key.starts_with("request_hash:");
        self.select_for_key(workers, &key, sticky)
    }

    fn name(&self) -> &'static str {
        "sticky_round_robin"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn needs_headers(&self) -> bool {
        true
    }

    fn reset(&self) {
        self.assignments.clear();
        self.session_counter.store(0, Ordering::Relaxed);
        self.keyless_counter.store(0, Ordering::Relaxed);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for StickyRoundRobinPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use std::collections::HashMap;

    fn workers(count: usize) -> Vec<Arc<dyn Worker>> {
        (0..count)
            .map(|index| {
                Arc::new(BasicWorker::new(
                    format!("http://w{index}:8000"),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect()
    }

    fn session_headers(session_id: &str) -> RequestHeaders {
        HashMap::from([("x-session-id".to_string(), session_id.to_string())])
    }

    #[test]
    fn new_sessions_round_robin_across_workers() {
        let policy = StickyRoundRobinPolicy::new();
        let workers = workers(3);

        let pick = |session_id: &str| {
            policy.select_worker_with_headers(&workers, None, Some(&session_headers(session_id)))
        };

        assert_eq!(pick("s0"), Some(0));
        assert_eq!(pick("s1"), Some(1));
        assert_eq!(pick("s2"), Some(2));
        assert_eq!(pick("s3"), Some(0));
    }

    #[test]
    fn existing_session_stays_on_its_worker() {
        let policy = StickyRoundRobinPolicy::new();
        let workers = workers(3);

        let first =
            policy.select_worker_with_headers(&workers, None, Some(&session_headers("session-a")));
        policy.select_worker_with_headers(&workers, None, Some(&session_headers("session-b")));
        policy.select_worker_with_headers(&workers, None, Some(&session_headers("session-c")));

        for _ in 0..5 {
            assert_eq!(
                policy.select_worker_with_headers(
                    &workers,
                    None,
                    Some(&session_headers("session-a")),
                ),
                first
            );
        }
    }

    #[test]
    fn unhealthy_assignment_is_remapped() {
        let policy = StickyRoundRobinPolicy::new();
        let workers = workers(3);
        let headers = session_headers("session-a");

        let first = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();
        workers[first].set_healthy(false);

        let remapped = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();

        assert_ne!(remapped, first);
        assert!(workers[remapped].is_healthy());
        assert_eq!(
            policy
                .assignments
                .get("header:x-session-id:session-a")
                .map(|assignment| assignment.value().clone()),
            Some(workers[remapped].url().to_string())
        );
    }

    #[test]
    fn keyless_requests_round_robin_without_affinity_entries() {
        let policy = StickyRoundRobinPolicy::new();
        let workers = workers(3);

        assert_eq!(
            policy.select_worker_with_headers(&workers, Some("null"), None),
            Some(0)
        );
        assert_eq!(
            policy.select_worker_with_headers(&workers, Some("null"), None),
            Some(1)
        );
        assert!(policy.assignments.is_empty());

        // Keyless traffic must not consume the cursor used to place sessions.
        assert_eq!(
            policy.select_worker_with_headers(&workers, None, Some(&session_headers("session-a")),),
            Some(0)
        );
    }

    #[test]
    fn no_healthy_workers_returns_none() {
        let policy = StickyRoundRobinPolicy::new();
        let workers = workers(2);
        for worker in &workers {
            worker.set_healthy(false);
        }
        let headers = session_headers("session-a");

        assert_eq!(
            policy.select_worker_with_headers(&workers, None, Some(&headers)),
            None
        );
        assert_eq!(
            policy.select_worker_with_headers(&workers, Some("null"), None),
            None
        );
    }
}
