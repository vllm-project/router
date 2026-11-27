//! Source-scoped cache updates and consistent snapshots.

use super::*;
use smallvec::SmallVec;
use std::borrow::Cow;

/// A batch holds the write lock for one source. Its closure must not re-enter
/// any locking `KvBlockIndexer` method.
pub(crate) struct IndexBatch<'a> {
    pub(super) index: &'a KvBlockIndexer,
    pub(super) target: &'a RoutableTarget,
    pub(super) active_pages: &'a mut ActivePages,
    pub(super) source: &'a SourceLock,
    pub(super) write: SourceWrite<'a>,
}

/// A consistent view under one source read lock.
pub(crate) struct IndexSnapshot<'a> {
    pub(super) index: &'a KvBlockIndexer,
    pub(super) source: &'a Arc<SourceLock>,
    pub(super) source_data: &'a SourceData,
}

impl IndexBatch<'_> {
    fn check_target(&self, target: &RoutableTarget) {
        assert_eq!(self.target, target, "source transaction crossed targets");
    }

    pub(crate) fn suspend_source(&self, source: &SourceId, rank: u32) {
        self.check_target(&RoutableTarget {
            instance_id: source.clone(),
            dp_rank: rank,
        });
        self.index.suspend_source_unlocked(source, rank);
    }

    pub(crate) fn resume_source(&self, source: &SourceId, rank: u32) {
        self.check_target(&RoutableTarget {
            instance_id: source.clone(),
            dp_rank: rank,
        });
        self.index.resume_source_unlocked(source, rank);
    }

    /// Remove only this precise owner, including its incarnation.
    pub(crate) fn clear_owner_exact(&mut self, owner: &ResidencyOwner) {
        self.check_target(&owner.target());
        self.write.mark_removal();
        *self.source.planned.write() = None;
        self.index.hash_pages.clear_owner(self.active_pages, owner);
    }
}

impl IndexSnapshot<'_> {
    fn check_owner(&self, owner: &ResidencyOwner) {
        assert_eq!(
            owner.target_parts(),
            (&self.source.target.instance_id, self.source.target.dp_rank),
            "source snapshot crossed targets"
        );
    }

    pub(crate) fn stream(&self, owner: &ResidencyOwner) -> Option<&StreamState> {
        self.check_owner(owner);
        self.source_data.streams.get(owner)
    }

    pub(in crate::kv_index) fn contains_any(
        &self,
        owner: &ResidencyOwner,
        group: u32,
        namespace: Option<&str>,
        keys: &[SequenceHash],
    ) -> bool {
        self.source_data
            .active_pages
            .get(owner)
            .is_some_and(|pages| pages.contains_any(group, namespace, keys))
    }

    pub(crate) fn source_version(&self) -> SourceVersion {
        SourceVersion {
            generation: self.source.generation.load(Ordering::Acquire),
            source: self.source.clone(),
        }
    }
    /// Optimistic postings are accepted only for an unchanged committed source.
    /// All stale groups are reread together under the caller's source read gate.
    pub(crate) fn hash_page_runs<'a>(
        &self,
        owner: &ResidencyOwner,
        namespace: Option<&str>,
        keys: &[SequenceHash],
        by_group: &'a GroupPages,
        groups: impl Iterator<Item = PageQuery> + Clone,
    ) -> SmallVec<[PageTiers<'a>; 1]> {
        self.check_owner(owner);
        let stable = groups.clone().all(|group| {
            by_group
                .get(&group)
                .is_some_and(|e| self.source.committed_at(&e.scan))
        });
        groups
            .map(|group| {
                if stable {
                    by_group[&group].runs_for_owner(owner)
                } else {
                    PageTiers {
                        prefix_len: 0,
                        prefix: TierBits::default(),
                        tail: Cow::Owned(self.index.hash_pages.for_owner(
                            self.source_data.active_pages.get(owner),
                            group.group,
                            namespace,
                            keys,
                        )),
                    }
                }
            })
            .collect()
    }

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(crate) fn hash_page_tiers<'a>(
        &self,
        owner: &ResidencyOwner,
        namespace: Option<&str>,
        keys: &[SequenceHash],
        by_group: &'a GroupPages,
        groups: impl Iterator<Item = u32> + Clone,
    ) -> Vec<Cow<'a, [TierBits]>> {
        self.hash_page_runs(
            owner,
            namespace,
            keys,
            by_group,
            groups.map(PageQuery::from),
        )
        .into_iter()
        .map(PageTiers::expanded)
        .collect()
    }
}

/// Mark the entire source batch, including failures and clears, as one mutation.
pub(super) struct SourceWrite<'a> {
    revision: &'a AtomicU64,
    index: &'a KvBlockIndexer,
    removal: Option<parking_lot::RwLockReadGuard<'a, ()>>,
}
impl<'a> SourceWrite<'a> {
    pub(super) fn new(revision: &'a AtomicU64, index: &'a KvBlockIndexer) -> Self {
        let previous = revision.fetch_or(1, Ordering::AcqRel);
        assert!(
            previous.is_multiple_of(2),
            "source writes must be serialized"
        );
        Self {
            revision,
            index,
            removal: None,
        }
    }

    pub(super) fn mark_removal(&mut self) {
        if self.removal.is_none() {
            self.removal = Some(self.index.removals.read());
        }
    }
}
impl Drop for SourceWrite<'_> {
    fn drop(&mut self) {
        // Pages must already be updated when this commit becomes part of a scan's cutoff.
        let previous = self.index.committed.fetch_add(2, Ordering::AcqRel);
        assert!(previous < u64::MAX - 1, "index commit sequence exhausted");
        self.revision.store(previous + 2, Ordering::Release);
    }
}

impl IndexBatch<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn store_hash_pages(
        &mut self,
        owner: &ResidencyOwner,
        group: u32,
        tier: StorageTier,
        namespace: Option<&str>,
        hashes: &[SequenceHash],
        incremental: bool,
        max_entries: usize,
    ) -> Result<(), PageWriteError> {
        self.check_target(&owner.target());
        self.index.hash_pages.store(
            self.active_pages,
            owner,
            group,
            tier,
            namespace,
            hashes,
            incremental,
            max_entries,
        )
    }
    pub(crate) fn remove_hash_pages(
        &mut self,
        owner: &ResidencyOwner,
        group: u32,
        tier: StorageTier,
        hashes: &[SequenceHash],
        incremental: bool,
    ) -> Result<(), PageWriteError> {
        self.check_target(&owner.target());
        self.write.mark_removal();
        self.index
            .hash_pages
            .remove(self.active_pages, owner, group, tier, hashes, incremental)
    }
}

#[derive(Default)]
pub(crate) struct StreamControl {
    visible_generation: AtomicU64,
}

impl StreamControl {
    pub(crate) fn generation(&self) -> u64 {
        self.visible_generation.load(Ordering::Acquire)
    }
}

pub(crate) struct StreamState {
    control: Arc<StreamControl>,
    pub(crate) reset_generation: u64,
    pub(crate) next_seq: Option<u64>,
    pub(crate) lease_id: u64,
    pub(crate) phase: StreamPhase,
    pub(crate) root_verified: bool,
    pub(crate) rejection_reported: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamPhase {
    Retired,
    Replaying,
    Live,
}

fn next_generation() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let generation = NEXT.fetch_add(1, Ordering::Relaxed);
    assert_ne!(generation, u64::MAX, "KV stream generation exhausted");
    generation
}

impl StreamState {
    pub(crate) fn new(control: Arc<StreamControl>) -> Self {
        control.visible_generation.store(0, Ordering::Release);
        Self {
            control,
            reset_generation: 0,
            next_seq: None,
            lease_id: 0,
            phase: StreamPhase::Retired,
            root_verified: false,
            rejection_reported: false,
        }
    }

    pub(crate) fn has_control(&self, control: &Arc<StreamControl>) -> bool {
        Arc::ptr_eq(&self.control, control)
    }

    pub(crate) fn visible(&self) -> bool {
        self.reset_generation != 0 && self.control.generation() == self.reset_generation
    }

    pub(crate) fn publish(&self) {
        self.control
            .visible_generation
            .store(self.reset_generation, Ordering::Release);
    }

    pub(crate) fn hide(&self) {
        self.control.visible_generation.store(0, Ordering::Release);
    }

    pub(crate) fn bump_generation(&mut self) {
        let visible = self.visible();
        self.reset_generation = next_generation();
        if visible {
            self.publish();
        }
    }

    pub(crate) fn reset(&mut self, next_seq: Option<u64>, visible: bool) {
        self.bump_generation();
        self.next_seq = next_seq;
        self.root_verified = false;
        self.phase = if visible {
            StreamPhase::Live
        } else if next_seq.is_some() {
            StreamPhase::Replaying
        } else {
            StreamPhase::Retired
        };
        if visible {
            self.publish();
        } else {
            self.control.visible_generation.store(0, Ordering::Release);
        }
    }

    pub(crate) fn begin_lease(&mut self) -> u64 {
        self.lease_id = next_generation();
        self.rejection_reported = false;
        self.lease_id
    }
}
