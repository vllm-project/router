//! Bounded advisory prefixes, independent of request and backend protocols.
//! A registration ID scopes exact token identity; it is not a Worker UUID.
//! This completion-routing token-prefix history is unrelated to Responses `HistoryBackend` storage.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

const MAX_ENTRIES: usize = 128;
const PREFIX_TOKENS: usize = 256;
const TTL: Duration = Duration::from_secs(60);

pub struct HistoryObservation {
    pub(crate) registration_id: u64,
    prefix: Arc<[u32]>,
}

impl HistoryObservation {
    pub(crate) fn new(registration_id: u64, tokens: &[u32]) -> Self {
        Self {
            registration_id,
            prefix: Arc::from(&tokens[..tokens.len().min(PREFIX_TOKENS)]),
        }
    }
}

pub(crate) struct HistoryHit {
    length: usize,
    expires: Instant,
}

impl HistoryHit {
    pub(crate) fn len_at(&self, now: Instant) -> usize {
        if self.expires > now {
            self.length
        } else {
            0
        }
    }
}

type HistoryBucket = Vec<(HistoryObservation, Instant)>;

#[derive(Default)]
pub(crate) struct CompletionHistory {
    records: Mutex<HashMap<(u64, u64), HistoryBucket>>,
}

impl CompletionHistory {
    pub(crate) fn insert(&self, observation: HistoryObservation, block_size: u32, now: Instant) {
        let Some(key) = Self::key(observation.registration_id, &observation.prefix, block_size)
        else {
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

    pub(crate) fn remove_registration(&self, registration: u64) {
        self.records.lock().retain(|(id, _), _| *id != registration);
    }

    pub(crate) fn match_prefix(
        &self,
        registration_id: u64,
        ids: &[u32],
        block_size: u32,
        now: Instant,
    ) -> Option<HistoryHit> {
        // Take references only. Prefix comparisons run outside both the store
        // lock and the policy's select-and-reserve lock.
        let key = Self::key(registration_id, ids, block_size)?;
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

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

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
    fn completion_history_bounds_expiry_and_registration_are_independent_of_credit() {
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
}
