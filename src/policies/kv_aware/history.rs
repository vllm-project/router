//! Advisory Regular Completion observations. A generation belongs to exactly
//! one registered target and CacheKey within this policy; it is not a Worker UUID.

use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
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
}

impl CompletionHistoryRequest {
    pub(super) fn generation(&self, worker: &dyn Worker, binding: &TargetBinding) -> Option<u64> {
        (self.registrations.contains(&binding.registration_id) && binding.is_history_worker(worker))
            .then_some(binding.registration_id)
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
}

impl HistoryObservation {
    pub(super) fn new(generation: u64, tokens: &[u32]) -> Self {
        Self {
            generation,
            prefix: Arc::from(&tokens[..tokens.len().min(PREFIX_TOKENS)]),
        }
    }
}

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
    records: Mutex<HashMap<(u64, u64), Vec<(HistoryObservation, Instant)>>>,
}

impl CompletionHistory {
    fn insert(&self, observation: HistoryObservation, block_size: u32, now: Instant) {
        let Some(key) = Self::key(observation.generation, &observation.prefix, block_size) else {
            return;
        };
        let mut records = self.records.lock();
        if let Some(record) = records.get_mut(&key).and_then(|bucket| {
            bucket
                .iter_mut()
                .find(|(old, _)| old.prefix == observation.prefix)
        }) {
            *record = (observation, now + TTL);
            return;
        }
        records.retain(|_, bucket| {
            bucket.retain(|(_, expires)| *expires > now);
            !bucket.is_empty()
        });
        if records.values().map(Vec::len).sum::<usize>() == MAX_ENTRIES {
            // Evict from the largest registration first, then its oldest
            // successful observation. One hot target cannot consume all peers.
            let mut counts = HashMap::<u64, usize>::new();
            for ((id, _), bucket) in records.iter() {
                *counts.entry(*id).or_default() += bucket.len();
            }
            let victim = records
                .iter()
                .flat_map(|(key, bucket)| {
                    bucket
                        .iter()
                        .enumerate()
                        .map(|(i, (_, expires))| (*key, i, counts[&key.0], *expires))
                })
                .max_by_key(|(_, _, count, expires)| (*count, std::cmp::Reverse(*expires)))
                .map(|(key, i, _, _)| (key, i));
            if let Some((victim, i)) = victim {
                let bucket = records.get_mut(&victim).unwrap();
                bucket.swap_remove(i);
                if bucket.is_empty() {
                    records.remove(&victim);
                }
            }
        }
        records
            .entry(key)
            .or_default()
            .push((observation, now + TTL));
    }

    fn key(registration: u64, ids: &[u32], block_size: u32) -> Option<(u64, u64)> {
        let block = block_size as usize;
        if block == 0 || block > PREFIX_TOKENS || ids.len() < block {
            return None;
        }
        let mut hasher = DefaultHasher::new();
        ids[..block].hash(&mut hasher);
        Some((registration, hasher.finish()))
    }

    pub(super) fn remove_registration(&self, registration: u64) {
        self.records.lock().retain(|(id, _), _| *id != registration);
    }

    pub(super) fn match_prefix(
        &self,
        generation: u64,
        ids: &[u32],
        block_size: u32,
        now: Instant,
    ) -> Option<HistoryHit> {
        // Take references only. Prefix comparisons run outside both the store
        // lock and the policy's select-and-reserve lock.
        let key = Self::key(generation, ids, block_size)?;
        let prefixes: Vec<_> = self
            .records
            .lock()
            .get(&key)
            .into_iter()
            .flatten()
            .filter(|(_, expires)| *expires > now)
            .map(|(r, expires)| (Arc::clone(&r.prefix), *expires))
            .collect();
        prefixes
            .into_iter()
            .filter_map(|(prefix, expires)| {
                let overlap = prefix.iter().zip(ids).take_while(|(a, b)| a == b).count();
                let block = block_size as usize;
                let length = overlap / block * block;
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
        let Some((generation, block_size)) = self
            .targets
            .get(worker.url())
            .filter(|b| b.is_history_worker(worker))
            .map(|b| (b.registration_id, b.block_size))
        else {
            return 0;
        };
        store
            .match_prefix(generation, ids, block_size, now)
            .map_or(0, |hit| hit.len_at(now))
    }

    pub(super) fn capture_history_registrations(
        &self,
        workers: &[Arc<dyn Worker>],
        requested_model: &str,
    ) -> CompletionHistoryRequest {
        let registrations = workers
            .iter()
            .filter_map(|worker| {
                let b = self.targets.get(worker.url())?;
                (worker.worker_type() == WorkerType::Regular
                    && worker.connection_mode() == ConnectionMode::Http
                    && worker.dp_rank().is_none()
                    && b.target.dp_rank == 0
                    && b.is_history_worker(worker.as_ref())
                    && requested_model != worker.base_url()
                    && requested_model != worker.url()
                    && b.cache_key.as_ref()?.model.as_ref() == requested_model)
                    .then_some(b.registration_id)
            })
            .collect();
        self.completion_history.get_or_init(Default::default);
        CompletionHistoryRequest { registrations }
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
        let Some(store) = self.completion_history.get() else {
            return;
        };
        if binding.registration_id == observation.generation && binding.is_history_worker(worker) {
            store.insert(observation, binding.block_size, Instant::now());
        }
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::policies::KvAwareOps;

    #[test]
    fn completion_history_complete_blocks_deduplicate_without_query_refresh() {
        let now = Instant::now();
        for block in [1, 2, 8, 16, 32, 256] {
            let h = CompletionHistory::default();
            let ids = vec![7; block as usize + 1];
            h.insert(HistoryObservation::new(1, &ids), block, now);
            assert!(h
                .match_prefix(1, &ids[..block as usize - 1], block, now)
                .is_none());
            assert_eq!(
                h.match_prefix(1, &ids[..block as usize], block, now)
                    .unwrap()
                    .len_at(now),
                block as usize
            );
            assert_eq!(
                h.match_prefix(1, &ids, block, now).unwrap().len_at(now),
                if block == 1 { 2 } else { block as usize }
            );
            h.insert(HistoryObservation::new(1, &ids), block, now + TTL / 2);
            assert_eq!(h.records.lock().values().map(Vec::len).sum::<usize>(), 1);
            assert!(h.match_prefix(1, &ids, block, now + TTL).is_some());
            assert!(h
                .match_prefix(1, &ids, block, now + TTL + TTL / 2)
                .is_none());
            assert!(h.match_prefix(1, &ids, 0, now).is_none());
            assert!(h.match_prefix(1, &ids, 257, now).is_none());
            h.remove_registration(1);
            assert!(h.records.lock().is_empty());
        }
        // A first-block fingerprint only locates candidates; suffixes remain
        // exact and two observations can never be stitched into a longer hit.
        let h = CompletionHistory::default();
        h.insert(HistoryObservation::new(1, &[1, 2, 3, 4]), 2, now);
        h.insert(HistoryObservation::new(1, &[1, 2, 5, 6]), 2, now);
        assert_eq!(
            h.match_prefix(1, &[1, 2, 3, 6], 2, now)
                .unwrap()
                .len_at(now),
            2
        );
        assert_eq!(h.records.lock().values().map(Vec::len).sum::<usize>(), 2);
        // Model a fingerprint collision explicitly: candidate placement alone
        // must never establish an exact match for a different first block.
        let collision = CompletionHistory::default();
        collision.records.lock().insert(
            CompletionHistory::key(1, &[1, 2], 2).unwrap(),
            vec![(HistoryObservation::new(1, &[9, 8, 7, 6]), now + TTL)],
        );
        assert!(collision.match_prefix(1, &[1, 2, 3, 4], 2, now).is_none());
    }

    #[test]
    fn completion_history_bounds_expiry_and_generation_are_independent_of_credit() {
        let h = CompletionHistory::default();
        let now = Instant::now();
        let long = vec![9; PREFIX_TOKENS + 100];
        h.insert(HistoryObservation::new(1, &long), 1, now);
        let hit = h.match_prefix(1, &long, 1, now).unwrap();
        assert_eq!(hit.len_at(now), PREFIX_TOKENS);
        assert_eq!(hit.len_at(now + TTL), 0);
        assert!(h.match_prefix(1, &long, 1, now + TTL).is_none());
        assert!(h.match_prefix(2, &long, 1, now).is_none());
        for n in 0..MAX_ENTRIES {
            h.insert(HistoryObservation::new(2, &[1, n as u32]), 1, now);
        }
        assert!(h.match_prefix(1, &long, 1, now).is_some());
        assert_eq!(
            h.records.lock().values().map(Vec::len).sum::<usize>(),
            MAX_ENTRIES
        );
        assert_eq!(
            h.match_prefix(2, &[1, 4, 3], 1, now).unwrap().len_at(now),
            2
        );
        h.insert(HistoryObservation::new(3, &[2]), 1, now + TTL);
        assert_eq!(h.records.lock().values().map(Vec::len).sum::<usize>(), 1);
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
            kv.register_history_worker(
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
