//! Shared hash-page storage and source transactions, for full and mixed attention.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::{mapref::entry::Entry, DashMap};
use parking_lot::RwLock;

use crate::kv_hash::SequenceHash;
mod event_addresses;
mod page_table;
mod records;
pub(crate) mod source_transaction;

use self::event_addresses::{prefix_key, EventAddresses, FullBlock};
use self::page_table::{ActivePages, PageTable};
pub(crate) use self::page_table::{PageEvidence, PageTiers};
pub(crate) use self::records::TierBits;
use self::source_transaction::SourceWrite;
pub(crate) use self::source_transaction::{IndexBatch, IndexSnapshot};
use self::source_transaction::{StreamControl, StreamPhase, StreamState};
use crate::kv_index::types::{
    ClearScope, Locality, ResidencyOwner, RoutableTarget, SourceId, StorageTier,
};

pub use crate::kv_index::types::{MatchQuery, TierMatch};

type Fast = foldhash::fast::RandomState;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PageWriteError {
    Capacity,
    ConflictingHash,
}

fn same_clear_owner(owner: &ResidencyOwner, other: &ResidencyOwner) -> bool {
    match (owner, other) {
        (
            ResidencyOwner::Worker {
                source: s1,
                dp_rank: r1,
                ..
            },
            ResidencyOwner::Worker {
                source: s2,
                dp_rank: r2,
                ..
            },
        ) => s1 == s2 && r1 == r2,
        (
            ResidencyOwner::CacheOwner { pool_id: p1, .. },
            ResidencyOwner::CacheOwner { pool_id: p2, .. },
        ) => p1 == p2,
        _ => false,
    }
}
/// Serializes one worker/rank across its cache groups.
struct SourceLock {
    target: RoutableTarget,
    access: RwLock<SourceData>,
    generation: AtomicU64,
    revision: AtomicU64,
}

#[derive(Default)]
struct SourceData {
    records: SourceRecords,
    streams: HashMap<ResidencyOwner, StreamState>,
}

#[derive(Default)]
struct SourceRecords {
    hash_keys: ActivePages,
    addresses: EventAddresses,
}

/// Identity and trust generation of one source at snapshot time.
pub(crate) struct SourceVersion {
    source: Arc<SourceLock>,
    generation: u64,
}

impl SourceLock {
    fn new(target: RoutableTarget) -> Self {
        Self {
            target,
            access: RwLock::new(SourceData::default()),
            generation: AtomicU64::new(0),
            revision: AtomicU64::new(0),
        }
    }
}

/// One shared fact index per cache identity, sharded by cache group.
/// Implements `TieredMatchProvider` directly for full attention.
pub struct KvBlockIndexer {
    hash_pages: PageTable,
    suspended_sources: DashMap<(SourceId, u32), ()>,
    /// Gates may be removed only by `retire_source` under the global barrier.
    // Membership changes only on source attachment/retirement. A compact map
    // avoids scanning CPU-count-sized sets of empty shards on every query.
    source_locks: RwLock<HashMap<RoutableTarget, Arc<SourceLock>, Fast>>,
    /// Ordinary access shares the barrier; pool clear and retirement exclude it.
    domain: RwLock<()>,
}

impl KvBlockIndexer {
    pub fn new() -> Self {
        KvBlockIndexer {
            hash_pages: PageTable::default(),
            suspended_sources: DashMap::new(),
            source_locks: RwLock::new(HashMap::default()),
            domain: RwLock::new(()),
        }
    }

    fn source_lock(&self, target: &RoutableTarget) -> Arc<SourceLock> {
        if let Some(source) = self.source_locks.read().get(target) {
            return source.clone();
        }
        self.source_locks
            .write()
            .entry(target.clone())
            .or_insert_with(|| Arc::new(SourceLock::new(target.clone())))
            .clone()
    }

    /// Check a snapshot after dropping its locks. Ordinary store/remove is a
    /// valid snapshot transition; suspension, resume, or retirement invalidates
    /// the result. This does not reserve engine blocks.
    pub(crate) fn source_version_is_current(&self, version: &SourceVersion) -> bool {
        // Resume and retirement increment the generation before revealing
        // the source. Read it last to fence snapshots across that transition.
        let target = &version.source.target;
        !self
            .suspended_sources
            .contains_key(&(target.instance_id.clone(), target.dp_rank))
            && version.source.generation.load(Ordering::Acquire) == version.generation
    }

    pub fn store(
        &self,
        group_idx: u32,
        owner: ResidencyOwner,
        tier: StorageTier,
        parent_seq: Option<Arc<str>>,
        blocks: &[FullBlock],
    ) {
        let target = owner.target();
        self.with_source_batch(&target, |batch| {
            let _ =
                batch.store_full_pages(group_idx, &owner, tier, parent_seq.as_ref(), blocks, false);
        });
    }

    pub fn remove(
        &self,
        group_idx: u32,
        owner: &ResidencyOwner,
        tier: StorageTier,
        seq_hashes: &[Arc<str>],
    ) {
        let target = owner.target();
        self.with_source_batch(&target, |batch| {
            batch.remove_full_pages(group_idx, owner, tier, seq_hashes, false)
        });
    }

    /// Clear one owner's residency across every group. A worker reset /
    /// `AllBlocksCleared` spans all kv-cache-spec groups the worker holds.
    pub fn clear(&self, owner: &ResidencyOwner, scope: ClearScope) {
        if matches!(owner, ResidencyOwner::Worker { .. }) && scope == ClearScope::Worker {
            self.with_source_batch(&owner.target(), |batch| batch.clear(owner, scope));
        } else {
            self.with_batch(|batch| batch.clear(owner, scope));
        }
    }

    /// Hide a source's observations while contiguous replay repairs its state.
    /// Correctness does not depend on delivery of advisory ingestion signals.
    pub fn suspend_source(&self, source: &SourceId, dp_rank: u32) {
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        let _domain = self.domain.read();
        let source_lock = self.source_lock(&target);
        let _guard = source_lock.access.write();
        self.suspend_source_unlocked(source, dp_rank);
    }

    fn suspend_source_unlocked(&self, source: &SourceId, dp_rank: u32) {
        self.suspended_sources.insert((source.clone(), dp_rank), ());
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        self.source_lock(&target)
            .generation
            .fetch_add(1, Ordering::AcqRel);
    }

    /// Stop crediting a lossy source and discard all its old observations.
    pub fn invalidate_source(&self, source: &SourceId, dp_rank: u32) {
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        let _domain = self.domain.read();
        let source_lock = self.source_lock(&target);
        let mut data = source_lock.access.write();
        let _revision = SourceWrite::new(&source_lock.revision);
        self.invalidate_source_unlocked(source, dp_rank, &mut data.records);
    }

    /// Permanently retire a fenced source. The global barrier drains all
    /// transactions before removing its gate; in-flight owned snapshots retain
    /// the old gate and observe its incremented generation.
    pub(crate) fn retire_source(&self, source: &SourceId, dp_rank: u32) {
        let _domain = self.domain.write();
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        let gate = self.source_lock(&target);
        let mut data = gate.access.write();
        let _revision = SourceWrite::new(&gate.revision);
        self.invalidate_source_unlocked(source, dp_rank, &mut data.records);
        drop(data);
        self.source_locks.write().remove(&target);
        self.suspended_sources.remove(&(source.clone(), dp_rank));
    }

    fn invalidate_source_unlocked(
        &self,
        source: &SourceId,
        dp_rank: u32,
        records: &mut SourceRecords,
    ) {
        self.suspend_source_unlocked(source, dp_rank);
        let owners: Vec<_> = records.hash_keys.keys().cloned().collect();
        for owner in owners {
            self.hash_pages.clear_owner(&mut records.hash_keys, &owner);
        }
        records.addresses = EventAddresses::default();
    }

    /// Enable fresh observations after complete replay, or after discarding
    /// incomplete replay and applying a new valid live batch to a clean source.
    /// Already-active sources must not invalidate concurrent query snapshots.
    pub fn resume_source(&self, source: &SourceId, dp_rank: u32) {
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        let _domain = self.domain.read();
        let source_lock = self.source_lock(&target);
        let _guard = source_lock.access.write();
        self.resume_source_unlocked(source, dp_rank);
    }

    fn resume_source_unlocked(&self, source: &SourceId, dp_rank: u32) {
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        // Never hold a suspended-sources map entry while acquiring a
        // source-lock map entry: final query checks access both maps.
        let gate = self.source_lock(&target);
        if let Entry::Occupied(entry) = self.suspended_sources.entry((source.clone(), dp_rank)) {
            // Fence the visibility transition before removing its hidden state.
            gate.generation.fetch_add(1, Ordering::AcqRel);
            entry.remove();
        }
    }

    pub fn find_matches(&self, query: &MatchQuery) -> Vec<TierMatch> {
        use crate::kv_index::types::push_block_segment;

        let mut parent = [0; 32];
        let keys: Vec<_> = query
            .local_hashes
            .iter()
            .map(|block| {
                parent = prefix_key(&parent, block);
                parent
            })
            .collect();
        let _domain = self.domain.read();
        let versions: Vec<_> = self
            .source_locks
            .read()
            .values()
            .map(|source| {
                let revision = source.revision.load(Ordering::Acquire);
                (source.clone(), revision)
            })
            .collect();
        let group = query.group_idx;
        let single_source = match versions.as_slice() {
            [(source, _)] => source.access.try_read().is_some_and(|data| {
                data.records
                    .hash_keys
                    .values()
                    .all(|pages| pages.device_only())
            }),
            _ => false,
        };
        let (prefix_len, head, tail) = if single_source {
            Default::default()
        } else {
            self.hash_pages.collect_split(group, None, &keys)
        };
        let mut by_target: HashMap<_, FullEvidence, Fast> = HashMap::default();
        for (owner, tiers) in head {
            let evidence = by_target.entry(owner.target()).or_default();
            evidence.prefix.0 |= tiers.0;
            if owner.locality() == Locality::Local {
                evidence.local_prefix.0 |= tiers.0;
            }
        }
        for (owner, tiers) in tail {
            by_target
                .entry(owner.target())
                .or_default()
                .tail
                .push((owner, tiers));
        }
        let mut matches = Vec::new();
        for (source, revision) in versions {
            let target = &source.target;
            let version = SourceVersion {
                generation: source.generation.load(Ordering::Acquire),
                source: source.clone(),
            };
            let mut evidence = by_target.remove(target).unwrap_or_default();
            let mut prefix_len = prefix_len;
            if single_source
                || !revision.is_multiple_of(2)
                || source.revision.load(Ordering::Acquire) != revision
            {
                // Read one committed snapshot for a sole source or to repair stale postings.
                let data = source.access.read();
                prefix_len = 0;
                evidence.prefix = TierBits::default();
                evidence.local_prefix = TierBits::default();
                evidence.tail = data
                    .records
                    .hash_keys
                    .iter()
                    .map(|(owner, pages)| {
                        (
                            owner.as_ref().clone(),
                            self.hash_pages.for_owner(Some(pages), group, None, &keys),
                        )
                    })
                    .collect();
            }
            let mut tiers = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            let mut local = TIERS.map(|tier| evidence.local_prefix.contains(tier));
            for (owner, claims) in evidence.tail {
                for (tier_idx, tier) in TIERS.into_iter().enumerate() {
                    if !query.tiers_of_interest.contains(&tier) {
                        continue;
                    }
                    let bits = &mut tiers[tier_idx];
                    for (position, claim) in claims.iter().enumerate() {
                        if claim.contains(tier) {
                            if bits.len() < claims.len() {
                                bits.resize(claims.len(), false);
                            }
                            bits[position] = true;
                            local[tier_idx] |= owner.locality() == Locality::Local;
                        }
                    }
                }
            }
            if !self.source_version_is_current(&version) {
                continue;
            }
            for (idx, (tier, hits)) in TIERS.into_iter().zip(tiers).enumerate() {
                let mut segments = Vec::new();
                if evidence.prefix.contains(tier) && query.tiers_of_interest.contains(&tier) {
                    segments.push((0, u32::try_from(prefix_len).unwrap_or(u32::MAX)));
                }
                for (position, hit) in hits.into_iter().enumerate() {
                    if hit {
                        let Ok(start) = u32::try_from(prefix_len + position) else {
                            break;
                        };
                        let Some(end) = start.checked_add(1) else {
                            break;
                        };
                        push_block_segment(&mut segments, start, end);
                    }
                }
                if !segments.is_empty() {
                    matches.push(TierMatch {
                        target: target.clone(),
                        tier,
                        segments,
                        locality: if local[idx] {
                            Locality::Local
                        } else {
                            Locality::Remote
                        },
                    });
                }
            }
        }
        matches
    }

    /// Run a whole publisher batch under one domain write lock.
    pub(crate) fn with_batch<R>(&self, f: impl FnOnce(&mut IndexBatch<'_>) -> R) -> R {
        let _guard = self.domain.write();
        f(&mut IndexBatch {
            index: self,
            scope: None,
            source_data: None,
        })
    }

    /// A source batch is atomic for this target without blocking other targets.
    pub(crate) fn with_source_batch<R>(
        &self,
        target: &RoutableTarget,
        f: impl FnOnce(&mut IndexBatch<'_>) -> R,
    ) -> R {
        let _domain = self.domain.read();
        let source = self.source_lock(target);
        let mut data = source.access.write();
        let _revision = SourceWrite::new(&source.revision);
        f(&mut IndexBatch {
            index: self,
            scope: Some(target.clone()),
            source_data: Some(&mut data.records),
        })
    }

    /// Apply a stream batch and its cache mutations under the same source lock.
    pub(crate) fn with_stream_batch<R>(
        &self,
        target: &RoutableTarget,
        owner: &ResidencyOwner,
        control: &Arc<StreamControl>,
        f: impl FnOnce(&mut StreamState, &mut IndexBatch<'_>) -> R,
    ) -> R {
        self.stream_batch(target, owner, control, true, f).unwrap()
    }

    /// Cleanup must not recreate a gate or owner that discovery already retired.
    pub(crate) fn with_existing_stream_batch<R>(
        &self,
        target: &RoutableTarget,
        owner: &ResidencyOwner,
        control: &Arc<StreamControl>,
        f: impl FnOnce(&mut StreamState, &mut IndexBatch<'_>) -> R,
    ) -> Option<R> {
        self.stream_batch(target, owner, control, false, f)
    }

    fn stream_batch<R>(
        &self,
        target: &RoutableTarget,
        owner: &ResidencyOwner,
        control: &Arc<StreamControl>,
        create: bool,
        f: impl FnOnce(&mut StreamState, &mut IndexBatch<'_>) -> R,
    ) -> Option<R> {
        assert_eq!(owner.target(), *target, "stream crossed targets");
        let _domain = self.domain.read();
        let source = if create {
            self.source_lock(target)
        } else {
            self.source_locks.read().get(target)?.clone()
        };
        let mut data = source.access.write();
        let _revision = SourceWrite::new(&source.revision);
        let SourceData { records, streams } = &mut *data;
        let state = if create {
            streams
                .entry(owner.clone())
                .or_insert_with(|| StreamState::new(control.clone()))
        } else {
            streams.get_mut(owner)?
        };
        assert!(
            state.has_control(control),
            "stream owner reused with another control"
        );
        let mut tx = IndexBatch {
            index: self,
            scope: Some(target.clone()),
            source_data: Some(records),
        };
        let result = f(state, &mut tx);
        if state.lease_id == 0 && state.next_seq.is_none() && !state.visible() {
            streams.remove(owner);
        }
        Some(result)
    }

    pub(crate) fn stream_lease_is_active(&self, owner: &ResidencyOwner, lease: u64) -> bool {
        let _domain = self.domain.read();
        let Some(source) = self.source_locks.read().get(&owner.target()).cloned() else {
            return false;
        };
        let data = source.access.read();
        data.streams
            .get(owner)
            .is_some_and(|state| state.lease_id == lease && state.phase != StreamPhase::Retired)
    }

    /// Read all groups from one attached source without creating a source gate.
    pub(crate) fn with_source_snapshot<R>(
        &self,
        target: &RoutableTarget,
        f: impl FnOnce(&IndexSnapshot<'_>) -> R,
    ) -> Option<R> {
        let _domain = self.domain.read();
        let source = self.source_locks.read().get(target).cloned()?;
        let data = source.access.read();
        Some(f(&IndexSnapshot {
            index: self,
            source: &source,
            source_data: &data,
        }))
    }

    pub(crate) fn has_multiple_sources(&self) -> bool {
        self.source_locks.read().len() > 1
    }

    pub(crate) fn collect_hash_groups(
        &self,
        groups: impl Iterator<Item = u32>,
        namespace: Option<&str>,
        keys: &[SequenceHash],
    ) -> HashMap<u32, PageEvidence> {
        let _domain = self.domain.read();
        let versions = Arc::new(
            self.source_locks
                .read()
                .iter()
                .map(|(target, source)| {
                    let gate = source.clone();
                    let version = gate.revision.load(Ordering::Acquire);
                    (target.clone(), (gate, version))
                })
                .collect(),
        );
        groups
            .map(|group| {
                let (prefix_len, head, holders) =
                    self.hash_pages.collect_split(group, namespace, keys);
                (
                    group,
                    PageEvidence {
                        versions: Arc::clone(&versions),
                        prefix_len,
                        prefix: head.into_iter().collect(),
                        holders,
                    },
                )
            })
            .collect()
    }

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn collect_hash_pages(&self, group: u32, keys: &[SequenceHash]) -> PageEvidence {
        self.collect_hash_groups(std::iter::once(group), None, keys)
            .remove(&group)
            .unwrap()
    }
}

#[derive(Default)]
struct FullEvidence {
    prefix: TierBits,
    local_prefix: TierBits,
    tail: Vec<(ResidencyOwner, Vec<TierBits>)>,
}

const TIERS: [StorageTier; 4] = [
    StorageTier::Device,
    StorageTier::HostPinned,
    StorageTier::Disk,
    StorageTier::External,
];

impl Default for KvBlockIndexer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests;
