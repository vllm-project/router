//! Sticky least-loaded session routing policy
//!
//! Adapted from SumanthRH/router commit b50d926f3cf8868e12e8395fa02ce1aa2b1ae5ae
//! (Apache-2.0), with typed-body routing and per-entry expiration fixes.
//!
//! This policy combines session stickiness (so all requests for a given
//! trajectory/session reuse the same vLLM replica and maximize KV cache reuse)
//! with load balancing across replicas (so *new* sessions are assigned to the
//! replica with the fewest active sessions).
//!
//! Behavior:
//! - The router bookkeeps the set of active sessions per replica.
//! - For an existing session: requests with the same session id are routed to
//!   the replica that session was originally assigned to.
//! - For a new session: the session is assigned to the healthy replica with the
//!   least number of active sessions. Ties are broken deterministically using
//!   consistent hashing (rendezvous / highest-random-weight) so that, given the
//!   same set of least-loaded replicas, a session is always assigned to the same
//!   replica.
//! - Sessions expire after a configurable TTL (default 2 hours, overridable via
//!   the `VLLM_ROUTER_SLL_SESSION_EXPIRATION_IN_S` environment variable). This
//!   protects against leaked sessions when a `finish_session` call is never
//!   received (e.g. a cancelled trajectory).
//! - Sessions can be released explicitly via [`finish_session`].
//!
//! Requests without any session identifier are still load-balanced to the
//! least-loaded replica, but are *not* recorded as active sessions.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tracing::{debug, info, warn};

use super::get_healthy_worker_indices;
use super::hash_key;
use super::ConsistentHashPolicy;
use super::LoadBalancingPolicy;
use super::RequestHeaders;
use crate::core::Worker;
use crate::metrics::RouterMetrics;

/// Default idle-session expiration (2 hours).
pub const DEFAULT_SESSION_EXPIRATION_SECS: u64 = 7200;

/// Environment variable to override the session expiration (in seconds).
pub const SESSION_EXPIRATION_ENV: &str = "VLLM_ROUTER_SLL_SESSION_EXPIRATION_IN_S";

/// Minimum interval between full expiration sweeps to bound per-request cost.
const SWEEP_INTERVAL_SECS: u64 = 60;

/// Bookkeeping for a single active session.
#[derive(Debug)]
struct SessionEntry {
    worker_url: String,
    last_access: Instant,
}

/// Internal mutable state guarded by a single mutex so that "find least-loaded
/// replica + record session" is atomic across concurrent new-session requests.
#[derive(Debug)]
struct SllState {
    /// session id -> assigned replica
    sessions: HashMap<String, SessionEntry>,
    /// replica url -> number of active sessions
    active_counts: HashMap<String, usize>,
    /// Timestamp of the last expiration sweep.
    last_sweep: Instant,
}

/// Sticky least-loaded routing policy.
#[derive(Debug)]
pub struct StickyLeastLoadedPolicy {
    state: Mutex<SllState>,
    session_expiration: Duration,
}

impl StickyLeastLoadedPolicy {
    /// Create a new policy, reading the session expiration from the environment
    /// (falling back to [`DEFAULT_SESSION_EXPIRATION_SECS`]).
    pub fn new() -> Self {
        Self::with_expiration_secs(Self::expiration_secs_from_env())
    }

    /// Create a new policy with an explicit session expiration (in seconds).
    pub fn with_expiration_secs(secs: u64) -> Self {
        Self {
            state: Mutex::new(SllState {
                sessions: HashMap::new(),
                active_counts: HashMap::new(),
                last_sweep: Instant::now(),
            }),
            session_expiration: Duration::from_secs(secs),
        }
    }

    fn expiration_secs_from_env() -> u64 {
        match std::env::var(SESSION_EXPIRATION_ENV) {
            Ok(val) => match val.trim().parse::<u64>() {
                Ok(secs) => secs,
                Err(_) => {
                    warn!(
                        "Invalid {} value '{}', using default {}s",
                        SESSION_EXPIRATION_ENV, val, DEFAULT_SESSION_EXPIRATION_SECS
                    );
                    DEFAULT_SESSION_EXPIRATION_SECS
                }
            },
            Err(_) => DEFAULT_SESSION_EXPIRATION_SECS,
        }
    }

    /// Decrement the active-session count for a replica, removing the entry when
    /// it reaches zero.
    fn decrement_count(counts: &mut HashMap<String, usize>, worker_url: &str) {
        if let Some(count) = counts.get_mut(worker_url) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                counts.remove(worker_url);
            }
        }
    }

    /// Remove sessions that have not been accessed within the expiration window.
    /// Runs at most once per [`SWEEP_INTERVAL_SECS`] to bound cost.
    fn sweep_expired(state: &mut SllState, expiration: Duration, now: Instant) {
        if now.duration_since(state.last_sweep) < Duration::from_secs(SWEEP_INTERVAL_SECS) {
            return;
        }

        let expired: Vec<String> = state
            .sessions
            .iter()
            .filter(|(_, entry)| now.duration_since(entry.last_access) >= expiration)
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired {
            if let Some(entry) = state.sessions.remove(&key) {
                Self::decrement_count(&mut state.active_counts, &entry.worker_url);
                debug!("SLL: expired session '{}' on '{}'", key, entry.worker_url);
            }
        }

        state.last_sweep = now;
    }

    /// Pick the least-loaded replica among `candidates`, breaking ties with
    /// consistent (rendezvous) hashing of `tie_break_key`.
    ///
    /// `candidates` is a slice of `(worker_index, worker_url)` for healthy
    /// replicas. Returns the chosen `worker_index`.
    fn select_least_loaded(
        counts: &HashMap<String, usize>,
        candidates: &[(usize, String)],
        tie_break_key: &str,
    ) -> usize {
        let min_count = candidates
            .iter()
            .map(|(_, url)| counts.get(url).copied().unwrap_or(0))
            .min()
            .unwrap_or(0);

        // Among replicas at the minimum load, deterministically pick one using
        // rendezvous hashing (highest hash wins).
        let mut best: Option<(usize, u64)> = None;
        for (idx, url) in candidates {
            let count = counts.get(url).copied().unwrap_or(0);
            if count != min_count {
                continue;
            }
            let weight = ConsistentHashPolicy::fbi_hash(&format!("{}:{}", tie_break_key, url));
            match best {
                Some((_, best_weight)) if best_weight >= weight => {}
                _ => best = Some((*idx, weight)),
            }
        }

        // `candidates` is non-empty (callers ensure healthy workers exist), so
        // `best` is always populated.
        best.map(|(idx, _)| idx).unwrap_or(candidates[0].0)
    }

    fn record_selection(&self, worker: &Arc<dyn Worker>) {
        worker.increment_processed();
        RouterMetrics::record_processed_request(worker.url());
        RouterMetrics::record_policy_decision(self.name(), worker.url());
    }

    /// Number of currently active (tracked) sessions. Exposed for tests/metrics.
    pub fn active_session_count(&self) -> usize {
        self.state.lock().unwrap().sessions.len()
    }

    /// Number of active sessions assigned to a specific replica. For tests.
    pub fn active_count_for(&self, worker_url: &str) -> usize {
        self.state
            .lock()
            .unwrap()
            .active_counts
            .get(worker_url)
            .copied()
            .unwrap_or(0)
    }
}

impl LoadBalancingPolicy for StickyLeastLoadedPolicy {
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

        let candidates: Vec<(usize, String)> = healthy_indices
            .iter()
            .map(|&idx| (idx, workers[idx].url().to_string()))
            .collect();

        let session_id = hash_key::extract_session_id(request_text, headers);
        let mut state = self.state.lock().unwrap();
        let now = Instant::now();
        Self::sweep_expired(&mut state, self.session_expiration, now);

        // Requests without a session id: load-balance but don't record state.
        let session_id = match session_id {
            Some(id) => id,
            None => {
                let tie_break = request_text.unwrap_or("");
                let idx = Self::select_least_loaded(&state.active_counts, &candidates, tie_break);
                drop(state);
                self.record_selection(&workers[idx]);
                debug!("SLL: stateless request routed to '{}'", workers[idx].url());
                return Some(idx);
            }
        };

        // Existing session: route to the same replica if it's still healthy.
        if let Some((worker_url, last_access)) = state
            .sessions
            .get(&session_id)
            .map(|e| (e.worker_url.clone(), e.last_access))
        {
            if let Some((idx, _)) = candidates.iter().find(|(_, url)| {
                *url == worker_url && now.duration_since(last_access) < self.session_expiration
            }) {
                let idx = *idx;
                if let Some(entry) = state.sessions.get_mut(&session_id) {
                    entry.last_access = now;
                }
                drop(state);
                self.record_selection(&workers[idx]);
                debug!(
                    "SLL: session '{}' -> existing replica '{}'",
                    session_id, worker_url
                );
                return Some(idx);
            }

            // Expired session or unavailable replica: release the old assignment.
            state.sessions.remove(&session_id);
            Self::decrement_count(&mut state.active_counts, &worker_url);
            debug!(
                "SLL: session '{}' replica '{}' unavailable, reassigning",
                session_id, worker_url
            );
        }

        // New session: assign to least-loaded replica (tie-broken by hashing).
        let idx = Self::select_least_loaded(&state.active_counts, &candidates, &session_id);
        let worker_url = workers[idx].url().to_string();
        state.sessions.insert(
            session_id.clone(),
            SessionEntry {
                worker_url: worker_url.clone(),
                last_access: now,
            },
        );
        *state.active_counts.entry(worker_url.clone()).or_insert(0) += 1;
        let new_count = state.active_counts[&worker_url];
        drop(state);

        self.record_selection(&workers[idx]);
        info!(
            "SLL: new session '{}' -> replica '{}' (active sessions on replica: {})",
            session_id, worker_url, new_count
        );
        Some(idx)
    }

    fn finish_session(&self, session_id: &str) {
        let mut state = self.state.lock().unwrap();
        if let Some(entry) = state.sessions.remove(session_id) {
            Self::decrement_count(&mut state.active_counts, &entry.worker_url);
            info!(
                "SLL: finished session '{}' on replica '{}'",
                session_id, entry.worker_url
            );
        } else {
            debug!("SLL: finish_session for unknown session '{}'", session_id);
        }
    }

    fn name(&self) -> &'static str {
        "sticky_least_loaded"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn needs_request_body(&self) -> bool {
        true
    }

    fn needs_headers(&self) -> bool {
        true
    }

    fn reset(&self) {
        let mut state = self.state.lock().unwrap();
        state.sessions.clear();
        state.active_counts.clear();
        state.last_sweep = Instant::now();
        info!("SLL: policy reset - all sessions cleared");
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for StickyLeastLoadedPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use std::collections::HashMap as StdHashMap;

    fn make_workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.to_string(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect()
    }

    fn header_with_session(session_id: &str) -> RequestHeaders {
        let mut headers: RequestHeaders = StdHashMap::new();
        headers.insert("x-session-id".to_string(), session_id.to_string());
        headers
    }

    #[test]
    fn test_same_session_sticky() {
        let policy = StickyLeastLoadedPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let headers = header_with_session("sess-1");

        let idx1 = policy.select_worker_with_headers(&workers, None, Some(&headers));
        let idx2 = policy.select_worker_with_headers(&workers, None, Some(&headers));
        let idx3 = policy.select_worker_with_headers(&workers, None, Some(&headers));

        assert!(idx1.is_some());
        assert_eq!(idx1, idx2);
        assert_eq!(idx2, idx3);

        // Only one active session should be tracked across repeated requests.
        assert_eq!(policy.active_session_count(), 1);
    }

    #[test]
    fn test_new_sessions_balance_across_replicas() {
        let policy = StickyLeastLoadedPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        // Assign three distinct sessions; each should land on a distinct replica
        // because new sessions go to the least-loaded replica.
        for i in 0..3 {
            let headers = header_with_session(&format!("sess-{}", i));
            policy
                .select_worker_with_headers(&workers, None, Some(&headers))
                .unwrap();
        }

        assert_eq!(policy.active_session_count(), 3);
        for w in &workers {
            assert_eq!(
                policy.active_count_for(w.url()),
                1,
                "expected each replica to hold exactly one session"
            );
        }
    }

    #[test]
    fn test_finish_session_releases_capacity() {
        let policy = StickyLeastLoadedPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

        let headers = header_with_session("sess-x");
        let idx = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();
        let assigned_url = workers[idx].url().to_string();
        assert_eq!(policy.active_count_for(&assigned_url), 1);

        policy.finish_session("sess-x");
        assert_eq!(policy.active_session_count(), 0);
        assert_eq!(policy.active_count_for(&assigned_url), 0);
    }

    #[test]
    fn test_finish_unknown_session_is_noop() {
        let policy = StickyLeastLoadedPolicy::new();
        // Should not panic and should leave state empty.
        policy.finish_session("does-not-exist");
        assert_eq!(policy.active_session_count(), 0);
    }

    #[test]
    fn test_session_id_from_body() {
        let policy = StickyLeastLoadedPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let body = r#"{"session_id": "body-sess", "prompt": "hi"}"#;

        let idx1 = policy.select_worker_with_headers(&workers, Some(body), None);
        let idx2 = policy.select_worker_with_headers(&workers, Some(body), None);
        assert_eq!(idx1, idx2);
        assert_eq!(policy.active_session_count(), 1);

        policy.finish_session("body-sess");
        assert_eq!(policy.active_session_count(), 0);
    }

    #[test]
    fn test_stateless_requests_not_tracked() {
        let policy = StickyLeastLoadedPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let body = r#"{"prompt": "no session here"}"#;

        let idx = policy.select_worker_with_headers(&workers, Some(body), None);
        assert!(idx.is_some());
        // No session identifier -> nothing tracked.
        assert_eq!(policy.active_session_count(), 0);
    }

    #[test]
    fn test_reassign_when_replica_unhealthy() {
        let policy = StickyLeastLoadedPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let headers = header_with_session("sess-move");

        let idx = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();
        let original_url = workers[idx].url().to_string();

        // Mark the assigned replica unhealthy; the session must be reassigned to
        // the remaining healthy replica.
        workers[idx].set_healthy(false);
        let new_idx = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();
        assert_ne!(workers[new_idx].url(), original_url);
        assert!(workers[new_idx].is_healthy());

        // Old replica should no longer hold the session.
        assert_eq!(policy.active_count_for(&original_url), 0);
        assert_eq!(policy.active_count_for(workers[new_idx].url()), 1);
    }

    #[test]
    fn test_expired_session_is_swept() {
        // Zero expiration => any session is immediately eligible for expiry, but
        // the sweep only runs every SWEEP_INTERVAL_SECS. Force the sweep by
        // back-dating last_sweep.
        let policy = StickyLeastLoadedPolicy::with_expiration_secs(0);
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let headers = header_with_session("sess-ttl");

        policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();
        assert_eq!(policy.active_session_count(), 1);

        {
            let mut state = policy.state.lock().unwrap();
            state.last_sweep = Instant::now() - Duration::from_secs(SWEEP_INTERVAL_SECS + 1);
        }

        // A subsequent request triggers the sweep, removing the expired session
        // and creating a fresh one.
        let other = header_with_session("sess-other");
        policy
            .select_worker_with_headers(&workers, None, Some(&other))
            .unwrap();
        // The expired session was swept; only the freshly created one remains.
        assert_eq!(policy.active_session_count(), 1);
        assert_eq!(
            policy.active_count_for("http://w1:8000") + policy.active_count_for("http://w2:8000"),
            1
        );
    }

    #[test]
    fn test_concurrent_new_sessions_reserve_balanced_capacity() {
        let policy = Arc::new(StickyLeastLoadedPolicy::new());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let barrier = std::sync::Barrier::new(60);
        std::thread::scope(|scope| {
            for i in 0..60 {
                let policy = &policy;
                let workers = &workers;
                let barrier = &barrier;
                scope.spawn(move || {
                    let headers = header_with_session(&format!("session-{i}"));
                    barrier.wait();
                    let first = policy.select_worker_with_headers(workers, None, Some(&headers));
                    assert_eq!(
                        first,
                        policy.select_worker_with_headers(workers, None, Some(&headers))
                    );
                });
            }
        });
        assert_eq!(policy.active_session_count(), 60);
        for worker in &workers {
            assert_eq!(policy.active_count_for(worker.url()), 60 / workers.len());
        }
    }

    #[test]
    fn test_expired_session_rebinds_before_next_sweep() {
        let policy = StickyLeastLoadedPolicy::with_expiration_secs(1);
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        for session in ["expired", "active"] {
            policy.select_worker_with_headers(
                &workers[..1],
                None,
                Some(&header_with_session(session)),
            );
        }
        policy
            .state
            .lock()
            .unwrap()
            .sessions
            .get_mut("expired")
            .unwrap()
            .last_access = Instant::now() - Duration::from_secs(2);

        assert_eq!(
            policy.select_worker_with_headers(
                &workers,
                None,
                Some(&header_with_session("expired"))
            ),
            Some(1),
            "expired affinity must not be refreshed before the global sweep"
        );
        assert_eq!(policy.active_count_for(workers[0].url()), 1);
        assert_eq!(policy.active_count_for(workers[1].url()), 1);
    }

    #[test]
    fn test_repeated_finish_preserves_other_sessions() {
        let policy = StickyLeastLoadedPolicy::new();
        let workers = make_workers(&["http://w1:8000"]);
        for session in ["finished", "active"] {
            policy.select_worker_with_headers(&workers, None, Some(&header_with_session(session)));
        }
        policy.finish_session("finished");
        policy.finish_session("finished");
        assert_eq!(policy.active_session_count(), 1);
        assert_eq!(policy.active_count_for(workers[0].url()), 1);
    }

    #[test]
    fn test_removed_worker_rebinds_session() {
        let policy = StickyLeastLoadedPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let headers = header_with_session("session");
        let original = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();
        let remaining = vec![workers[1 - original].clone()];
        assert_eq!(
            policy.select_worker_with_headers(&remaining, None, Some(&headers)),
            Some(0)
        );
        assert_eq!(policy.active_count_for(workers[original].url()), 0);
        assert_eq!(policy.active_count_for(remaining[0].url()), 1);
    }
}
