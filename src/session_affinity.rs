//! Hard session pin in front of a stateless policy.
//!
//! The module stores the worker for a session key. It calls the policy only
//! when that pin is missing, expired, or the worker is gone. `kv_aware`,
//! `sticky_least_loaded`, and the key-owner policies do not use it.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::core::Worker;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffinityOutcome {
    NoSession,
    StickyHit,
    Bind,
    Rebind,
    RetryBypass,
}

#[derive(Debug, Clone, Copy)]
pub struct SessionCounts<'a> {
    counts: &'a HashMap<String, usize>,
}

impl<'a> SessionCounts<'a> {
    pub fn get(self, url: &str) -> usize {
        self.counts.get(url).copied().unwrap_or(0)
    }
}

#[derive(Debug)]
struct Pin {
    url: String,
    last_access: Instant,
}

#[derive(Debug, Default)]
struct State {
    pins: HashMap<String, Pin>,
    counts: HashMap<String, usize>,
    /// One lock per key so two first requests for that key cannot both bind.
    binders: HashMap<String, Arc<Mutex<()>>>,
}

/// Pin key that keeps the identity source, so `agent_hint` and a body
/// `session_id` with the same text do not share a worker.
pub fn pin_key(source: &str, program_id: &str) -> String {
    format!("{source}|{program_id}")
}

#[derive(Debug)]
pub struct SessionAffinity {
    enabled: bool,
    ttl: Duration,
    state: Mutex<State>,
}

impl SessionAffinity {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ttl: Duration::from_secs(7200),
            state: Mutex::new(State::default()),
        }
    }

    pub fn hard(ttl_secs: u64) -> Self {
        Self {
            enabled: true,
            ttl: Duration::from_secs(ttl_secs),
            state: Mutex::new(State::default()),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Policies that already pin, or that own the key, stay outside this module.
    pub fn applies(policy_name: &str) -> bool {
        !matches!(
            policy_name,
            "kv_aware" | "consistent_hash" | "rendezvous_hash" | "sticky_least_loaded"
        )
    }

    pub fn active_sessions(&self, url: &str) -> usize {
        self.state
            .lock()
            .unwrap()
            .counts
            .get(url)
            .copied()
            .unwrap_or(0)
    }

    /// Release `session_id` and every `session_id:*` pin. `task-10` is kept.
    pub fn finish_session(&self, session_id: &str) -> usize {
        if !self.enabled {
            return 0;
        }
        let mut state = self.state.lock().unwrap();
        let keys: Vec<String> = state
            .pins
            .keys()
            .filter(|key| key_matches(key, session_id))
            .cloned()
            .collect();
        for key in &keys {
            if let Some(pin) = state.pins.remove(key) {
                decrement(&mut state.counts, &pin.url);
            }
            if state
                .binders
                .get(key)
                .is_some_and(|binder| Arc::strong_count(binder) == 1)
            {
                state.binders.remove(key);
            }
        }
        keys.len()
    }

    pub fn route<F>(
        &self,
        key: Option<&str>,
        workers: &[Arc<dyn Worker>],
        excluded: &[String],
        first_pick: F,
    ) -> Option<(usize, AffinityOutcome)>
    where
        F: FnOnce(SessionCounts<'_>) -> Option<usize>,
    {
        if !self.enabled {
            let idx = first_pick(SessionCounts {
                counts: &HashMap::new(),
            })?;
            return Some((idx, AffinityOutcome::NoSession));
        }
        let Some(key) = key.filter(|key| !key.is_empty()) else {
            let counts = {
                let mut state = self.state.lock().unwrap();
                evict_expired(&mut state, self.ttl, Instant::now());
                state.counts.clone()
            };
            let idx = first_pick(SessionCounts { counts: &counts })?;
            return Some((idx, AffinityOutcome::NoSession));
        };

        let binder = {
            let mut state = self.state.lock().unwrap();
            evict_expired(&mut state, self.ttl, Instant::now());
            Arc::clone(
                state
                    .binders
                    .entry(key.to_string())
                    .or_insert_with(|| Arc::new(Mutex::new(()))),
            )
        };
        let _bind = binder.lock().unwrap();

        let now = Instant::now();
        {
            let mut state = self.state.lock().unwrap();
            evict_expired(&mut state, self.ttl, now);
            if let Some(url) = live_url(&state, key, workers) {
                if excluded.iter().any(|item| item == &url) {
                    let counts = state.counts.clone();
                    drop(state);
                    let idx = first_pick(SessionCounts { counts: &counts })?;
                    return Some((idx, AffinityOutcome::RetryBypass));
                }
                if let Some(idx) = workers.iter().position(|worker| worker.url() == url) {
                    if let Some(pin) = state.pins.get_mut(key) {
                        pin.last_access = now;
                    }
                    return Some((idx, AffinityOutcome::StickyHit));
                }
            }
        }

        let counts = self.state.lock().unwrap().counts.clone();
        let had_pin = self.state.lock().unwrap().pins.contains_key(key);
        let idx = first_pick(SessionCounts { counts: &counts })?;
        let url = workers.get(idx)?.url().to_string();
        self.store(key, &url, Instant::now());
        Some((
            idx,
            if had_pin {
                AffinityOutcome::Rebind
            } else {
                AffinityOutcome::Bind
            },
        ))
    }

    fn store(&self, key: &str, url: &str, now: Instant) {
        let mut state = self.state.lock().unwrap();
        if let Some(previous) = state.pins.insert(
            key.to_string(),
            Pin {
                url: url.to_string(),
                last_access: now,
            },
        ) {
            if previous.url != url {
                decrement(&mut state.counts, &previous.url);
                *state.counts.entry(url.to_string()).or_insert(0) += 1;
            }
        } else {
            *state.counts.entry(url.to_string()).or_insert(0) += 1;
        }
    }
}

fn live_url(state: &State, key: &str, workers: &[Arc<dyn Worker>]) -> Option<String> {
    let pin = state.pins.get(key)?;
    let live = workers
        .iter()
        .any(|worker| worker.url() == pin.url && worker.is_healthy() && worker.is_available());
    live.then(|| pin.url.clone())
}

fn evict_expired(state: &mut State, ttl: Duration, now: Instant) {
    let expired: Vec<String> = state
        .pins
        .iter()
        .filter(|(_, pin)| now.saturating_duration_since(pin.last_access) >= ttl)
        .map(|(key, _)| key.clone())
        .collect();
    for key in expired {
        if let Some(pin) = state.pins.remove(&key) {
            decrement(&mut state.counts, &pin.url);
        }
        if state
            .binders
            .get(&key)
            .is_some_and(|binder| Arc::strong_count(binder) == 1)
        {
            state.binders.remove(&key);
        }
    }
}

fn identity_of(pin: &str) -> &str {
    pin.split_once('|').map(|(_, id)| id).unwrap_or(pin)
}

fn key_matches(pin: &str, session_id: &str) -> bool {
    let id = identity_of(pin);
    id == session_id
        || id
            .strip_prefix(session_id)
            .is_some_and(|rest| rest.starts_with(':'))
}

fn decrement(counts: &mut HashMap<String, usize>, url: &str) {
    if let Some(count) = counts.get_mut(url) {
        *count = count.saturating_sub(1);
        if *count == 0 {
            counts.remove(url);
        }
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn pool(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|url| {
                Arc::new(BasicWorker::new((*url).to_string(), WorkerType::Regular))
                    as Arc<dyn Worker>
            })
            .collect()
    }

    #[test]
    fn hard_pin_calls_the_policy_once_per_key() {
        let affinity = SessionAffinity::hard(7200);
        let workers = pool(&["http://w0:8000", "http://w1:8000"]);
        let calls = AtomicUsize::new(0);
        let pick = |_: SessionCounts<'_>| {
            calls.fetch_add(1, Ordering::SeqCst);
            Some(1)
        };
        let (first, outcome) = affinity.route(Some("task-1"), &workers, &[], pick).unwrap();
        assert_eq!((first, outcome), (1, AffinityOutcome::Bind));
        let (second, outcome) = affinity
            .route(Some("task-1"), &workers, &[], |_| {
                panic!("policy called again")
            })
            .unwrap();
        assert_eq!((second, outcome), (1, AffinityOutcome::StickyHit));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(affinity.active_sessions("http://w1:8000"), 1);
    }

    #[test]
    fn round_robin_first_pick_sticks_for_the_same_session() {
        let affinity = SessionAffinity::hard(7200);
        let workers = pool(&["http://w0:8000", "http://w1:8000"]);
        let cursor = AtomicUsize::new(0);
        let pick = || {
            affinity.route(Some("chat"), &workers, &[], |_| {
                Some(cursor.fetch_add(1, Ordering::SeqCst) % 2)
            })
        };
        assert_eq!(pick().unwrap().0, 0);
        assert_eq!(pick().unwrap().0, 0);
        assert_eq!(cursor.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn least_load_first_pick_sees_active_session_counts() {
        let affinity = SessionAffinity::hard(7200);
        let workers = pool(&["http://w0:8000", "http://w1:8000"]);
        affinity
            .route(Some("a"), &workers, &[], |_| Some(0))
            .unwrap();
        let (idx, _) = affinity
            .route(Some("b"), &workers, &[], |counts| {
                let loads = vec![
                    (
                        0,
                        counts.get("http://w0:8000"),
                        "http://w0:8000".to_string(),
                    ),
                    (
                        1,
                        counts.get("http://w1:8000"),
                        "http://w1:8000".to_string(),
                    ),
                ];
                crate::policies::least_by_load(&loads, "b")
            })
            .unwrap();
        assert_eq!(idx, 1);
    }

    #[test]
    fn retry_exclusion_does_not_move_the_pin() {
        let affinity = SessionAffinity::hard(7200);
        let workers = pool(&["http://w0:8000", "http://w1:8000"]);
        affinity
            .route(Some("task-1"), &workers, &[], |_| Some(0))
            .unwrap();
        let (idx, outcome) = affinity
            .route(
                Some("task-1"),
                &workers,
                &["http://w0:8000".to_string()],
                |_| Some(1),
            )
            .unwrap();
        assert_eq!((idx, outcome), (1, AffinityOutcome::RetryBypass));
        let (again, outcome) = affinity
            .route(Some("task-1"), &workers, &[], |_| panic!("moved"))
            .unwrap();
        assert_eq!((again, outcome), (0, AffinityOutcome::StickyHit));
    }

    #[test]
    fn finish_session_releases_the_key_and_its_agents() {
        let affinity = SessionAffinity::hard(7200);
        let workers = pool(&["http://w0:8000"]);
        for key in ["task-1", "task-1:lead", "task-10"] {
            affinity
                .route(Some(key), &workers, &[], |_| Some(0))
                .unwrap();
        }
        assert_eq!(affinity.finish_session("task-1"), 2);
        assert_eq!(affinity.active_sessions("http://w0:8000"), 1);
        let (_idx, outcome) = affinity
            .route(Some("task-10"), &workers, &[], |_| {
                panic!("task-10 was released")
            })
            .unwrap();
        assert_eq!(outcome, AffinityOutcome::StickyHit);
    }

    #[test]
    fn disabled_module_does_not_store() {
        let affinity = SessionAffinity::off();
        let workers = pool(&["http://w0:8000", "http://w1:8000"]);
        let (idx, outcome) = affinity
            .route(Some("task-1"), &workers, &[], |_| Some(0))
            .unwrap();
        assert_eq!((idx, outcome), (0, AffinityOutcome::NoSession));
        assert_eq!(affinity.finish_session("task-1"), 0);
    }

    #[test]
    fn kv_aware_and_key_owners_do_not_apply() {
        assert!(!SessionAffinity::applies("kv_aware"));
        assert!(!SessionAffinity::applies("consistent_hash"));
        assert!(!SessionAffinity::applies("rendezvous_hash"));
        assert!(!SessionAffinity::applies("sticky_least_loaded"));
        assert!(SessionAffinity::applies("round_robin"));
        assert!(SessionAffinity::applies("random"));
        assert!(SessionAffinity::applies("least_load"));
        assert!(SessionAffinity::applies("power_of_two"));
        assert!(SessionAffinity::applies("cache_aware"));
    }

    #[test]
    fn expired_pins_do_not_inflate_least_load_counts() {
        let affinity = SessionAffinity::hard(0);
        let workers = pool(&["http://w0:8000", "http://w1:8000"]);
        affinity
            .route(Some("a"), &workers, &[], |_| Some(0))
            .unwrap();
        let (idx, _) = affinity
            .route(Some("b"), &workers, &[], |counts| {
                assert_eq!(counts.get("http://w0:8000"), 0);
                Some(1)
            })
            .unwrap();
        assert_eq!(idx, 1);
        assert_eq!(affinity.active_sessions("http://w0:8000"), 0);
    }

    #[test]
    fn distinct_identity_sources_do_not_share_a_pin() {
        let affinity = SessionAffinity::hard(7200);
        let workers = pool(&["http://w0:8000", "http://w1:8000"]);
        let body = pin_key("body_session", "task-1");
        let hint = pin_key("agent_hint", "task-1");
        affinity
            .route(Some(&body), &workers, &[], |_| Some(0))
            .unwrap();
        let (idx, outcome) = affinity
            .route(Some(&hint), &workers, &[], |_| Some(1))
            .unwrap();
        assert_eq!((idx, outcome), (1, AffinityOutcome::Bind));
        assert_eq!(affinity.finish_session("task-1"), 2);
        assert_eq!(affinity.active_sessions("http://w0:8000"), 0);
    }
}
