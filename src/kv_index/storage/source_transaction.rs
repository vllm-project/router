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

    /// Remove only this precise owner, including its incarnation.
    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(crate) fn clear_owner_exact(&mut self, owner: &ResidencyOwner) {
        self.check_target(&owner.target());
        *self.source.planned.write() = None;
        self.index.hash_pages.clear_owner(self.active_pages, owner);
    }

    pub(in crate::kv_index) fn clear_pages(&mut self, matches: impl Fn(&ResidencyOwner) -> bool) {
        *self.source.planned.write() = None;
        let owners: Vec<_> = self
            .active_pages
            .keys()
            .filter(|owner| matches(owner))
            .cloned()
            .collect();
        for owner in owners {
            self.index.hash_pages.clear_owner(self.active_pages, &owner);
        }
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

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
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
    committed: &'a AtomicU64,
}
impl<'a> SourceWrite<'a> {
    pub(super) fn new(revision: &'a AtomicU64, committed: &'a AtomicU64) -> Self {
        let previous = revision.fetch_or(1, Ordering::AcqRel);
        assert!(
            previous.is_multiple_of(2),
            "source writes must be serialized"
        );
        Self {
            revision,
            committed,
        }
    }
}
impl Drop for SourceWrite<'_> {
    fn drop(&mut self) {
        // Pages must already be updated when this commit becomes part of a scan's cutoff.
        let previous = self.committed.fetch_add(2, Ordering::AcqRel);
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
    ) {
        self.check_target(&owner.target());
        self.index
            .hash_pages
            .remove(self.active_pages, owner, group, tier, hashes, incremental);
    }
}
