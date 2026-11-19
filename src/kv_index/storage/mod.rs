//! Shared hash-page storage and source transactions, for full and mixed attention.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::{mapref::entry::Entry, DashMap};
use parking_lot::RwLock;

use crate::kv_hash::SequenceHash;
mod page_table;
mod records;
pub(crate) mod source_transaction;

use self::page_table::{ActivePages, PageTable};
pub(crate) use self::page_table::{PageEvidence, PageTiers};
use self::source_transaction::SourceWrite;
pub(crate) use self::source_transaction::{IndexBatch, IndexSnapshot};
use self::source_transaction::{StreamControl, StreamPhase, StreamState};
use crate::kv_index::types::{ResidencyOwner, RoutableTarget, SourceId, StorageTier, TierBits};

type Fast = foldhash::fast::RandomState;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PageWriteError {
    Capacity,
    ConflictingHash,
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
    active_pages: ActivePages,
    streams: HashMap<ResidencyOwner, StreamState>,
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
pub struct KvBlockIndexer {
    hash_pages: PageTable,
    suspended_sources: DashMap<(SourceId, u32), ()>,
    /// Gates may be removed only by `retire_source` under the global barrier.
    // Membership changes only on source attachment/retirement. A compact map
    // avoids scanning CPU-count-sized sets of empty shards on every query.
    source_locks: RwLock<HashMap<RoutableTarget, Arc<SourceLock>, Fast>>,
    /// Ordinary access shares the barrier; retirement excludes it.
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

    /// Hide a source while contiguous replay repairs its observations.
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
        self.invalidate_source_unlocked(source, dp_rank, &mut data.active_pages);
        drop(data);
        self.source_locks.write().remove(&target);
        self.suspended_sources.remove(&(source.clone(), dp_rank));
    }

    fn invalidate_source_unlocked(
        &self,
        source: &SourceId,
        dp_rank: u32,
        active_pages: &mut ActivePages,
    ) {
        self.suspend_source_unlocked(source, dp_rank);
        let owners: Vec<_> = active_pages.keys().cloned().collect();
        for owner in owners {
            self.hash_pages.clear_owner(active_pages, &owner);
        }
    }

    /// Enable fresh observations after successful replay.
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
        let SourceData {
            active_pages,
            streams,
        } = &mut *data;
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
            target,
            active_pages,
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

impl Default for KvBlockIndexer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests;
