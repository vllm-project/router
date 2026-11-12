//! Advisory Completion observations. A generation belongs to exactly
//! one registered target and CacheKey within this policy; it is not a Worker UUID.

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use super::{KvAwarePolicy, TargetBinding};
use crate::core::{ConnectionMode, Worker, WorkerType};

const MAX_ENTRIES: usize = 128;
const PREFIX_TOKENS: usize = 256;
const TTL: Duration = Duration::from_secs(60);

/// Captured before the first preparation RPC. Do not reuse its evidence for
/// another Worker object at the same URL, even on a retry of the same request.
pub struct CompletionHistoryRequest {
    registrations: HashSet<u64>,
    pub(super) prefill: bool,
    pub(super) origin: std::sync::Weak<CompletionHistory>,
}

impl CompletionHistoryRequest {
    pub(super) fn belongs_to(&self, history: Option<&Arc<CompletionHistory>>) -> bool {
        history.is_some_and(|history| self.origin.ptr_eq(&Arc::downgrade(history)))
    }

    pub(super) fn generation(&self, worker: &dyn Worker, binding: &TargetBinding) -> Option<u64> {
        (self.registrations.contains(&binding.generation) && binding.is_history_worker(worker))
            .then_some(binding.generation)
    }
}

impl TargetBinding {
    fn is_history_worker(&self, worker: &dyn Worker) -> bool {
        self.history_worker
            .as_ref()
            .is_some_and(|registered| std::ptr::addr_eq(registered.as_ptr(), worker))
    }
}

pub struct HistoryObservation {
    generation: u64,
    prefix: Arc<[u32]>,
    pub(super) origin: std::sync::Weak<CompletionHistory>,
}

/// A prefix lookup prepared before P/D admission. It owns no load or tokens;
/// admission rechecks its expiry and registration before using its overlap.
#[derive(Debug, Clone)]
pub struct PrefillHistoryHint {
    generation: u64,
    origin: std::sync::Weak<CompletionHistory>,
    hit: Option<HistoryHit>,
}

impl PrefillHistoryHint {
    /// Mint the selected P's observation from the same prepared input used for
    /// this lookup. Call only after admission, outside its load lock.
    pub(crate) fn observation(&self, ids: &[u32]) -> HistoryObservation {
        HistoryObservation::new(self.generation, ids, self.origin.clone())
    }
}

impl HistoryObservation {
    pub(super) fn new(
        generation: u64,
        tokens: &[u32],
        origin: std::sync::Weak<CompletionHistory>,
    ) -> Self {
        Self {
            generation,
            prefix: Arc::from(&tokens[..tokens.len().min(PREFIX_TOKENS)]),
            origin,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct HistoryHit {
    length: usize,
    expires: Instant,
}

impl HistoryHit {
    pub(super) fn len_at(&self, now: Instant) -> usize {
        if self.expires > now {
            self.length
        } else {
            0
        }
    }
}

#[derive(Default)]
pub(super) struct CompletionHistory {
    records: Mutex<VecDeque<(HistoryObservation, Instant)>>,
}

impl CompletionHistory {
    fn insert(&self, observation: HistoryObservation, now: Instant) {
        let mut records = self.records.lock();
        records.retain(|(_, expires)| *expires > now);
        if records.len() == MAX_ENTRIES {
            records.pop_front();
        }
        records.push_back((observation, now + TTL));
    }

    pub(super) fn match_prefix(
        &self,
        generation: u64,
        ids: &[u32],
        now: Instant,
    ) -> Option<HistoryHit> {
        // Take references only. Prefix comparisons run outside both the store
        // lock and the policy's select-and-reserve lock.
        let prefixes: Vec<_> = self
            .records
            .lock()
            .iter()
            .filter(|(r, expires)| r.generation == generation && *expires > now)
            .map(|(r, expires)| (Arc::clone(&r.prefix), *expires))
            .collect();
        prefixes
            .into_iter()
            .filter_map(|(prefix, expires)| {
                let length = prefix.iter().zip(ids).take_while(|(a, b)| a == b).count();
                (length != 0).then_some(HistoryHit { length, expires })
            })
            .max_by_key(|hit| (hit.length, hit.expires))
    }
}

impl KvAwarePolicy {
    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    pub(crate) fn observed_completion_prefix(&self, worker: &dyn Worker, ids: &[u32]) -> usize {
        let now = Instant::now();
        let Some(store) = self.completion_history.get() else {
            return 0;
        };
        let Some(generation) = self
            .targets
            .get(worker.url())
            .filter(|b| b.is_history_worker(worker))
            .map(|b| b.generation)
        else {
            return 0;
        };
        store
            .match_prefix(generation, ids, now)
            .map_or(0, |hit| hit.len_at(now))
    }

    pub(super) fn capture_history_registrations(
        &self,
        workers: &[Arc<dyn Worker>],
        requested_model: &str,
        prefill: bool,
    ) -> CompletionHistoryRequest {
        let registrations = workers
            .iter()
            .filter_map(|worker| {
                let b = self.targets.get(worker.url())?;
                ((if prefill {
                    matches!(worker.worker_type(), WorkerType::Prefill { .. })
                        && worker.dp_size().is_none_or(|size| size == 1)
                } else {
                    worker.worker_type() == WorkerType::Regular
                }) && worker.connection_mode() == ConnectionMode::Http
                    && worker.dp_rank().is_none()
                    && b.target.dp_rank == 0
                    && b.is_history_worker(worker.as_ref())
                    && requested_model != worker.base_url()
                    && requested_model != worker.url()
                    && b.cache_key.as_ref()?.model.as_ref() == requested_model)
                    .then_some(b.generation)
            })
            .collect();
        CompletionHistoryRequest {
            registrations,
            prefill,
            origin: Arc::downgrade(self.completion_history.get_or_init(Default::default)),
        }
    }

    pub(super) fn prepare_prefill_history_hint(
        &self,
        capture: &CompletionHistoryRequest,
        worker: &dyn Worker,
        ids: &[u32],
    ) -> Option<PrefillHistoryHint> {
        if self.params.router_temperature != 0.0 || !capture.prefill || ids.is_empty() {
            return None;
        }
        let store = self.completion_history.get()?;
        if !capture.belongs_to(Some(store)) {
            return None;
        }
        let generation = {
            let binding = self.targets.get(worker.url())?;
            capture.generation(worker, &binding)?
        };
        Some(PrefillHistoryHint {
            generation,
            origin: capture.origin.clone(),
            hit: store.match_prefix(generation, ids, Instant::now()),
        })
    }

    pub(super) fn prefill_hint_overlap(
        &self,
        worker: &dyn Worker,
        hint: &PrefillHistoryHint,
    ) -> usize {
        let Some(store) = self.completion_history.get() else {
            return 0;
        };
        if self.params.router_temperature != 0.0 || !hint.origin.ptr_eq(&Arc::downgrade(store)) {
            return 0;
        }
        let Some(binding) = self.targets.get(worker.url()) else {
            return 0;
        };
        if binding.generation != hint.generation || !binding.is_history_worker(worker) {
            return 0;
        }
        hint.hit
            .as_ref()
            .map_or(0, |hit| hit.len_at(Instant::now()))
    }

    pub(super) fn commit_history_observation(
        &self,
        worker: &dyn Worker,
        observation: HistoryObservation,
    ) {
        // Keep the map read guard until publication. A replacement must either
        // come after this insertion or make its old generation fail this check.
        let Some(binding) = self.targets.get(worker.url()) else {
            return;
        };
        let Some(store) = self
            .completion_history
            .get()
            .filter(|store| observation.origin.ptr_eq(&Arc::downgrade(store)))
        else {
            return;
        };
        if binding.generation == observation.generation && binding.is_history_worker(worker) {
            store.insert(observation, Instant::now());
        }
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::policies::KvAwareOps;

    #[test]
    fn completion_history_bounds_expiry_and_generation_are_independent_of_credit() {
        let h = CompletionHistory::default();
        let now = Instant::now();
        let long = vec![9; PREFIX_TOKENS + 100];
        h.insert(HistoryObservation::new(1, &long, Default::default()), now);
        let hit = h.match_prefix(1, &long, now).unwrap();
        assert_eq!(hit.len_at(now), PREFIX_TOKENS);
        assert_eq!(hit.len_at(now + TTL), 0);
        assert!(h.match_prefix(1, &long, now + TTL).is_none());
        assert!(h.match_prefix(2, &long, now).is_none());
        for n in 0..MAX_ENTRIES {
            h.insert(
                HistoryObservation::new(2, &[1, n as u32], Default::default()),
                now,
            );
        }
        assert!(h.match_prefix(1, &long, now).is_none());
        assert_eq!(h.records.lock().len(), MAX_ENTRIES);
        assert_eq!(h.match_prefix(2, &[1, 4, 3], now).unwrap().len_at(now), 2);
        h.insert(
            HistoryObservation::new(3, &[2], Default::default()),
            now + TTL,
        );
        assert_eq!(h.records.lock().len(), 1);
    }

    #[test]
    fn completion_history_requires_named_identity_and_one_reported_rank() {
        use crate::core::BasicWorker;
        use crate::kv_index::{CacheKey, HashMode, KvBlockIndexer, WorkerKvInfo};
        let kv = KvAwarePolicy::default();
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorker::new("http://w".into(), WorkerType::Regular));
        for (model, ranks, expected) in [
            ("model", vec![0], true),
            ("http://w", vec![0], false),
            ("model", vec![0, 1], false),
            ("model", vec![], false),
        ] {
            kv.register_http_worker(
                &worker,
                Some(&WorkerKvInfo {
                    ranks,
                    cache_key: Some(CacheKey {
                        model: Arc::from(model),
                        hash_mode: HashMode::Sha256,
                        block_size: 16,
                    }),
                    ..Default::default()
                }),
                Some(Arc::new(KvBlockIndexer::new())),
                true,
            );
            let captured = kv
                .capture_completion_history(std::slice::from_ref(&worker), model)
                .unwrap();
            assert_eq!(
                captured
                    .generation(worker.as_ref(), &kv.binding_for_worker(worker.as_ref()))
                    .is_some(),
                expected,
                "{model}"
            );
        }
    }
}
