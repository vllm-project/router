//! Sticky-table load balancing policy
//!
//! Decouples session *placement* from session *stickiness*: the first
//! request of a session is placed on the least-loaded healthy worker and
//! the mapping is recorded; every later request looks the worker up in
//! the table. Placement therefore reacts to live load (which no pure
//! hash can), stickiness is exact rather than statistical, and a session
//! can be re-placed — worker death or overload migration — by rewriting
//! one table entry, at the cost of that session rebuilding its KV prefix
//! once on the new worker.
//!
//! Requests without any session identity (no session header, no
//! session/user body field) are routed least-loaded without touching the
//! table.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tracing::{debug, info};

use super::{get_healthy_worker_indices, hash_key, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::metrics::RouterMetrics;

/// Sessions end silently; drop a mapping after this long without traffic.
const SESSION_TTL: Duration = Duration::from_secs(1800);
/// How often to scan the table for expired sessions.
const EVICTION_INTERVAL: Duration = Duration::from_secs(60);
/// Table size ceiling; beyond it new sessions are routed but not recorded.
const MAX_SESSIONS: usize = 262_144;
/// Migrate a session away from its home worker only when the home is both
/// this many in-flight requests above the least-loaded worker...
const MIGRATE_ABS_THRESHOLD: isize = 8;
/// ...and this many times its load. Both must hold, so small absolute
/// wobbles at low load and proportional wobbles at high load don't flap.
const MIGRATE_REL_THRESHOLD: f64 = 2.0;
/// ...and the request body is smaller than this. Body size is a proxy for
/// context length, i.e. for what a migration costs: moving a session to a
/// worker that has no prefix for it turns an incremental prefill into a
/// full one. Trace replay showed unconditional migration doubling total
/// prefill work and amplifying the very overload it tried to escape, so
/// only sessions that are provably cheap to rebuild may move.
const MIGRATE_MAX_BODY_BYTES: usize = 32 * 1024;

struct SessionEntry {
    worker_url: String,
    last_seen: Instant,
}

struct Table {
    sessions: HashMap<String, SessionEntry>,
    /// Sessions currently mapped to each worker URL. Placement tie-break:
    /// under a uniform load snapshot (cold start, idle system) raw
    /// least-loaded clusters every new session onto one momentarily-cool
    /// worker; fewest-resident-sessions breaks those ties evenly.
    resident: HashMap<String, usize>,
    last_eviction: Instant,
}

impl Table {
    fn insert(&mut self, key: String, worker_url: String, now: Instant) {
        *self.resident.entry(worker_url.clone()).or_insert(0) += 1;
        if let Some(old) = self.sessions.insert(
            key,
            SessionEntry {
                worker_url,
                last_seen: now,
            },
        ) {
            self.uncount(&old.worker_url);
        }
    }

    fn uncount(&mut self, worker_url: &str) {
        if let Some(n) = self.resident.get_mut(worker_url) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                self.resident.remove(worker_url);
            }
        }
    }
}

/// Sticky-table policy: least-loaded placement, exact table stickiness.
pub struct StickyTablePolicy {
    table: Mutex<Table>,
    /// Load reports from external monitoring; falls back to the worker's
    /// local in-flight counter when absent.
    cached_loads: Mutex<HashMap<String, isize>>,
}

impl StickyTablePolicy {
    pub fn new() -> Self {
        Self {
            table: Mutex::new(Table {
                sessions: HashMap::new(),
                resident: HashMap::new(),
                last_eviction: Instant::now(),
            }),
            cached_loads: Mutex::new(HashMap::new()),
        }
    }

    fn worker_load(&self, worker: &dyn Worker) -> isize {
        if let Ok(loads) = self.cached_loads.lock() {
            if let Some(&load) = loads.get(worker.url()) {
                return load;
            }
        }
        worker.load() as isize
    }

    /// Session identity, or None for anonymous requests. Reuses the shared
    /// header/body extractors but deliberately not the whole-body fallback:
    /// a per-request hash would fill the table with garbage entries.
    fn session_key(request_text: Option<&str>, headers: Option<&RequestHeaders>) -> Option<String> {
        if let Some(hdrs) = headers {
            if let Some(key) = hash_key::extract_hash_key_from_headers(hdrs) {
                return Some(key);
            }
        }
        hash_key::extract_hash_key_from_body(request_text)
    }

    /// One policy instance can serve both the prefill and the decode pool
    /// (registry falls back to the shared default policy), so table keys
    /// carry a fingerprint of the pool's worker set — otherwise the two
    /// pools would endlessly overwrite each other's mappings.
    fn pool_fingerprint(workers: &[Arc<dyn Worker>]) -> u64 {
        let mut hasher = DefaultHasher::new();
        for w in workers {
            w.url().hash(&mut hasher);
        }
        hasher.finish()
    }

    fn least_loaded(&self, workers: &[Arc<dyn Worker>], candidates: &[usize]) -> usize {
        *candidates
            .iter()
            .min_by_key(|&&idx| self.worker_load(workers[idx].as_ref()))
            .expect("candidates is non-empty")
    }

    /// Placement target: least in-flight load, ties broken by fewest
    /// resident sessions (see `Table::resident`).
    fn placement(&self, workers: &[Arc<dyn Worker>], candidates: &[usize], table: &Table) -> usize {
        *candidates
            .iter()
            .min_by_key(|&&idx| {
                let w = workers[idx].as_ref();
                (
                    self.worker_load(w),
                    table.resident.get(w.url()).copied().unwrap_or(0),
                )
            })
            .expect("candidates is non-empty")
    }

    fn evict_expired(&self, table: &mut Table, now: Instant) {
        if now.duration_since(table.last_eviction) < EVICTION_INTERVAL {
            return;
        }
        let before = table.sessions.len();
        let mut resident = std::mem::take(&mut table.resident);
        table.sessions.retain(|_, entry| {
            let live = now.duration_since(entry.last_seen) < SESSION_TTL;
            if !live {
                if let Some(n) = resident.get_mut(&entry.worker_url) {
                    *n = n.saturating_sub(1);
                }
            }
            live
        });
        resident.retain(|_, n| *n > 0);
        table.resident = resident;
        table.last_eviction = now;
        let evicted = before - table.sessions.len();
        if evicted > 0 {
            debug!(
                "Sticky table evicted {} expired sessions, {} remain",
                evicted,
                table.sessions.len()
            );
        }
    }

    fn record_selection(&self, workers: &[Arc<dyn Worker>], idx: usize) -> Option<usize> {
        workers[idx].increment_processed();
        RouterMetrics::record_processed_request(workers[idx].url());
        RouterMetrics::record_policy_decision(self.name(), workers[idx].url());
        Some(idx)
    }
}

impl std::fmt::Debug for StickyTablePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sessions = self.table.lock().map(|t| t.sessions.len()).unwrap_or(0);
        f.debug_struct("StickyTablePolicy")
            .field("sessions", &sessions)
            .finish()
    }
}

impl LoadBalancingPolicy for StickyTablePolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }

        let Some(session) = Self::session_key(request_text, headers) else {
            let idx = self.least_loaded(workers, &healthy);
            return self.record_selection(workers, idx);
        };
        let key = format!("{:x}:{}", Self::pool_fingerprint(workers), session);

        let now = Instant::now();
        let mut table = self.table.lock().ok()?;
        self.evict_expired(&mut table, now);

        let home_idx = table.sessions.get(&key).and_then(|entry| {
            healthy
                .iter()
                .copied()
                .find(|&idx| workers[idx].url() == entry.worker_url)
        });

        let idx = match home_idx {
            Some(home) => {
                let home_load = self.worker_load(workers[home].as_ref());
                let coolest = self.least_loaded(workers, &healthy);
                let min_load = self.worker_load(workers[coolest].as_ref());
                let overloaded = home_load - min_load >= MIGRATE_ABS_THRESHOLD
                    && home_load as f64 >= MIGRATE_REL_THRESHOLD * min_load.max(1) as f64;
                let cheap_to_move =
                    request_text.is_some_and(|t| t.len() <= MIGRATE_MAX_BODY_BYTES);
                if overloaded && cheap_to_move {
                    info!(
                        "Sticky table migrating session off {} (load {}) to {} (load {})",
                        workers[home].url(),
                        home_load,
                        workers[coolest].url(),
                        min_load
                    );
                    coolest
                } else {
                    home
                }
            }
            None => {
                let placed = self.placement(workers, &healthy, &table);
                info!(
                    "Sticky table placing new session on {} (load {}, {} sessions tracked)",
                    workers[placed].url(),
                    self.worker_load(workers[placed].as_ref()),
                    table.sessions.len()
                );
                placed
            }
        };

        let entry_exists = table.sessions.contains_key(&key);
        if entry_exists || table.sessions.len() < MAX_SESSIONS {
            table.insert(key, workers[idx].url().to_string(), now);
        } else {
            debug!("Sticky table full ({} sessions); routing without recording", MAX_SESSIONS);
        }
        drop(table);

        self.record_selection(workers, idx)
    }

    fn name(&self) -> &'static str {
        "sticky_table"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn needs_headers(&self) -> bool {
        true
    }

    fn update_loads(&self, loads: &HashMap<String, isize>) {
        if let Ok(mut cached) = self.cached_loads.lock() {
            *cached = loads.clone();
        }
    }

    fn reset(&self) {
        if let Ok(mut table) = self.table.lock() {
            table.sessions.clear();
            table.resident.clear();
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for StickyTablePolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    fn make_workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|u| {
                Arc::new(BasicWorker::new(u.to_string(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect()
    }

    fn session_headers(id: &str) -> RequestHeaders {
        let mut h = RequestHeaders::new();
        h.insert("x-session-id".to_string(), id.to_string());
        h
    }

    #[test]
    fn new_session_placed_least_loaded() {
        let policy = StickyTablePolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        workers[0].increment_load();
        workers[0].increment_load();
        workers[1].increment_load();

        let headers = session_headers("sess-a");
        let idx = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();
        assert_eq!(idx, 2);
    }

    #[test]
    fn session_sticks_despite_load_changes() {
        let policy = StickyTablePolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let headers = session_headers("sess-b");

        let first = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();
        // Tilt load toward the chosen worker, but below the migration bar.
        for _ in 0..MIGRATE_ABS_THRESHOLD - 1 {
            workers[first].increment_load();
        }
        for _ in 0..20 {
            let again = policy
                .select_worker_with_headers(&workers, None, Some(&headers))
                .unwrap();
            assert_eq!(again, first);
        }
    }

    #[test]
    fn session_replaced_when_home_unhealthy() {
        let policy = StickyTablePolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let headers = session_headers("sess-c");

        let first = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();
        workers[first].set_healthy(false);
        let second = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();
        assert_ne!(second, first);

        // The re-placement is itself sticky once the old home recovers.
        workers[first].set_healthy(true);
        let third = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();
        assert_eq!(third, second);
    }

    #[test]
    fn cheap_session_migrates_off_overloaded_home() {
        let policy = StickyTablePolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let headers = session_headers("sess-d");
        let small_body = r#"{"messages":[{"role":"user","content":"hi"}]}"#;

        let first = policy
            .select_worker_with_headers(&workers, Some(small_body), Some(&headers))
            .unwrap();
        for _ in 0..MIGRATE_ABS_THRESHOLD {
            workers[first].increment_load();
        }
        let second = policy
            .select_worker_with_headers(&workers, Some(small_body), Some(&headers))
            .unwrap();
        assert_ne!(second, first);

        // Migration rewrites the table: the session now sticks to the new home.
        let third = policy
            .select_worker_with_headers(&workers, Some(small_body), Some(&headers))
            .unwrap();
        assert_eq!(third, second);
    }

    #[test]
    fn expensive_session_stays_home_despite_overload() {
        let policy = StickyTablePolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let headers = session_headers("sess-d2");
        let big_body = format!(
            r#"{{"messages":[{{"role":"user","content":"{}"}}]}}"#,
            "x".repeat(MIGRATE_MAX_BODY_BYTES)
        );

        let first = policy
            .select_worker_with_headers(&workers, Some(&big_body), Some(&headers))
            .unwrap();
        for _ in 0..2 * MIGRATE_ABS_THRESHOLD as usize {
            workers[first].increment_load();
        }
        // Re-prefilling a huge context costs more than queueing saves.
        let second = policy
            .select_worker_with_headers(&workers, Some(&big_body), Some(&headers))
            .unwrap();
        assert_eq!(second, first);
    }

    #[test]
    fn anonymous_requests_route_least_loaded_without_recording() {
        let policy = StickyTablePolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        workers[0].increment_load();

        let body = r#"{"messages":[{"role":"assistant","content":"hi"}],"model":"m"}"#;
        let idx = policy
            .select_worker_with_headers(&workers, Some(body), None)
            .unwrap();
        assert_eq!(idx, 1);
        assert_eq!(policy.table.lock().unwrap().sessions.len(), 0);
    }

    #[test]
    fn body_user_field_is_a_session_key() {
        let policy = StickyTablePolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

        let body = r#"{"model":"m","user":"alice","messages":[{"role":"user","content":"hi"}]}"#;
        let first = policy
            .select_worker_with_headers(&workers, Some(body), None)
            .unwrap();
        workers[first].increment_load();
        let second = policy
            .select_worker_with_headers(&workers, Some(body), None)
            .unwrap();
        assert_eq!(second, first);
    }

    #[test]
    fn shared_instance_keeps_pools_independent() {
        // One instance serving two pools (the registry's default-policy
        // fallback) must not let the pools clobber each other's mappings.
        let policy = StickyTablePolicy::new();
        let prefill = make_workers(&["http://p1:8000", "http://p2:8000"]);
        let decode = make_workers(&["http://d1:8000", "http://d2:8000"]);
        let headers = session_headers("sess-e");

        let p_first = policy
            .select_worker_with_headers(&prefill, None, Some(&headers))
            .unwrap();
        let d_first = policy
            .select_worker_with_headers(&decode, None, Some(&headers))
            .unwrap();
        for _ in 0..10 {
            let p = policy
                .select_worker_with_headers(&prefill, None, Some(&headers))
                .unwrap();
            let d = policy
                .select_worker_with_headers(&decode, None, Some(&headers))
                .unwrap();
            assert_eq!(p, p_first);
            assert_eq!(d, d_first);
        }
        assert_eq!(policy.table.lock().unwrap().sessions.len(), 2);
    }
}
