#![cfg(all(test, feature = "cpu", feature = "ready"))]

use super::*;

impl KvBlockIndexer {
    /// A source batch is atomic for this target without blocking other targets.
    pub(super) fn with_source_batch<R>(
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
            target,
            active_pages: &mut data.active_pages,
        })
    }

    pub(crate) fn suspend_source(&self, source: &SourceId, dp_rank: u32) {
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        let _domain = self.domain.read();
        let source_lock = self.source_lock(&target);
        let _guard = source_lock.access.write();
        self.suspend_source_unlocked(source, dp_rank);
    }

    pub(crate) fn invalidate_source(&self, source: &SourceId, dp_rank: u32) {
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        let _domain = self.domain.read();
        let source_lock = self.source_lock(&target);
        let mut data = source_lock.access.write();
        let _revision = SourceWrite::new(&source_lock.revision);
        self.invalidate_source_unlocked(source, dp_rank, &mut data.active_pages);
    }

    pub(crate) fn resume_source(&self, source: &SourceId, dp_rank: u32) {
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        let _domain = self.domain.read();
        let source_lock = self.source_lock(&target);
        let _guard = source_lock.access.write();
        self.resume_source_unlocked(source, dp_rank);
    }
}

fn worker(name: &str, dp_rank: u32) -> ResidencyOwner {
    ResidencyOwner::Worker {
        source: SourceId::from(name),
        dp_rank,
        incarnation: 0,
    }
}

fn store(
    index: &KvBlockIndexer,
    owner: &ResidencyOwner,
    group: u32,
    key: SequenceHash,
    tier: StorageTier,
) {
    index.with_source_batch(&owner.target(), |batch| {
        batch
            .store_hash_pages(owner, group, tier, None, &[key], false, usize::MAX)
            .unwrap();
    });
}

fn visible_tiers(
    index: &KvBlockIndexer,
    owner: &ResidencyOwner,
    group: u32,
    key: SequenceHash,
) -> TierBits {
    let evidence = index.collect_hash_pages(group, &[key]);
    let (pages, version) = evidence.unchanged(&owner.target(), owner).unwrap();
    if index.source_version_is_current(&version) {
        pages.expanded().first().copied().unwrap_or_default()
    } else {
        TierBits::default()
    }
}

#[test]
fn retired_streams_are_removed_without_reusing_snapshot_generations() {
    let index = KvBlockIndexer::new();
    let owner = worker("a", 0);
    let target = owner.target();
    let control = Arc::new(StreamControl::default());
    let mut previous = 0;
    for _ in 0..8 {
        let generation = index.with_stream_batch(&target, &owner, &control, |state, _| {
            state.reset(Some(0), true);
            state.reset_generation
        });
        assert_ne!(generation, previous);
        previous = generation;
        index.with_stream_batch(&target, &owner, &control, |state, _| {
            state.reset(None, false)
        });
        assert!(index.source_lock(&target).access.read().streams.is_empty());
    }
}

#[test]
fn read_snapshot_does_not_recreate_unknown_or_retired_source() {
    let index = KvBlockIndexer::new();
    let owner = worker("retired", 0);
    let target = owner.target();
    assert!(index.with_source_snapshot(&target, |_| ()).is_none());
    store(&index, &owner, 0, [1; 32], StorageTier::Device);
    index.retire_source(&target.instance_id, target.dp_rank);
    assert!(index.with_source_snapshot(&target, |_| ()).is_none());
    assert!(index.source_locks.read().is_empty());
}

#[test]
fn source_suspension_and_invalidation_cover_tiers_groups_and_only_that_rank() {
    let index = KvBlockIndexer::new();
    let rank0 = worker("a", 0);
    let rank1 = worker("a", 1);
    let key = [1; 32];
    for group in [0, 1] {
        store(&index, &rank0, group, key, StorageTier::Device);
        store(&index, &rank0, group, key, StorageTier::HostPinned);
        store(&index, &rank1, group, key, StorageTier::Device);
    }
    let source = SourceId::from("a");
    let assert_visibility = |rank0_visible| {
        for group in [0, 1] {
            let tiers = visible_tiers(&index, &rank0, group, key);
            assert_eq!(tiers.contains(StorageTier::Device), rank0_visible);
            assert_eq!(tiers.contains(StorageTier::HostPinned), rank0_visible);
            assert!(visible_tiers(&index, &rank1, group, key).contains(StorageTier::Device));
        }
    };

    assert_visibility(true);
    index.suspend_source(&source, 0);
    assert_visibility(false);
    index.resume_source(&source, 0);
    assert_visibility(true);
    index.invalidate_source(&source, 0);
    index.resume_source(&source, 0);
    assert_visibility(false);
}

#[test]
fn source_batch_publishes_all_groups_atomically() {
    use std::sync::mpsc;
    use std::time::Duration;

    let index = Arc::new(KvBlockIndexer::new());
    let owner = worker("a", 0);
    let key = [1; 32];
    let (entered, ready) = mpsc::channel();
    let (release, resume) = mpsc::channel();
    let writer_index = index.clone();
    let writer_owner = owner.clone();
    let writer = std::thread::spawn(move || {
        writer_index.with_source_batch(&writer_owner.target(), |batch| {
            for group in [0, 1] {
                batch
                    .store_hash_pages(
                        &writer_owner,
                        group,
                        StorageTier::Device,
                        None,
                        &[key],
                        false,
                        8,
                    )
                    .unwrap();
                if group == 0 {
                    entered.send(()).unwrap();
                    resume.recv_timeout(Duration::from_secs(5)).unwrap();
                }
            }
        });
    });
    ready.recv_timeout(Duration::from_secs(5)).unwrap();
    let (started, scheduled) = mpsc::channel();
    let (done, result) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        let groups = index.collect_hash_groups(0..2, None, &[key]);
        assert!(groups[&0].unchanged(&owner.target(), &owner).is_none());
        started.send(()).unwrap();
        let pages = index
            .with_source_snapshot(&owner.target(), |view| {
                view.hash_page_tiers(&owner, None, &[key], &groups, 0..2)
            })
            .unwrap();
        done.send(pages.iter().all(|group| {
            group
                .first()
                .is_some_and(|page| page.contains(StorageTier::Device))
        }))
        .unwrap();
    });
    scheduled.recv_timeout(Duration::from_secs(5)).unwrap();
    let premature = result.recv_timeout(Duration::from_millis(100));
    release.send(()).unwrap();
    writer.join().unwrap();
    reader.join().unwrap();
    assert!(matches!(premature, Err(mpsc::RecvTimeoutError::Timeout)));
    assert!(result.recv_timeout(Duration::from_secs(5)).unwrap());
}

#[test]
fn retiring_many_sources_reclaims_gates_without_erasing_other_owners() {
    let index = KvBlockIndexer::new();
    let keep = worker("keep", 0);
    let key = [1; 32];
    store(&index, &keep, 0, key, StorageTier::Device);
    for id in 0..1000 {
        let name = format!("retired-{id}");
        store(&index, &worker(&name, 0), 0, key, StorageTier::Device);
        index.retire_source(&SourceId::from(name), 0);
    }
    assert_eq!(index.source_locks.read().len(), 1);
    assert!(index.suspended_sources.is_empty());
    assert!(visible_tiers(&index, &keep, 0, key).contains(StorageTier::Device));
}
