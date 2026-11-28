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
        let write = SourceWrite::new(&source.revision, self);
        *source.planned.write() = None;
        let mut batch = IndexBatch {
            index: self,
            target,
            active_pages: &mut data.active_pages,
            source: &source,
            write,
        };
        f(&mut batch)
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
        let mut write = SourceWrite::new(&source_lock.revision, self);
        self.invalidate_source_unlocked(source, dp_rank, &mut data.active_pages, &mut write);
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
    ResidencyOwner {
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
            .store_hash_pages(
                owner,
                group,
                tier,
                None,
                &[key],
                tier == StorageTier::Device,
                usize::MAX,
            )
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
    let version = evidence
        .validate_source(index, &owner.target(), &[key])
        .unwrap();
    if version.is_current() {
        evidence.runs_for_owner(owner).get(0).unwrap_or_default()
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
fn source_snapshot_allows_another_rank_to_retire() {
    let index = KvBlockIndexer::new();
    let first = worker("worker", 0);
    let second = worker("worker", 1);
    store(&index, &first, 0, [1; 32], StorageTier::Device);
    store(&index, &second, 0, [1; 32], StorageTier::Device);

    std::thread::scope(|scope| {
        let (retired, completed) = std::sync::mpsc::channel();
        index
            .with_source_snapshot(&first.target(), |_| {
                let index = &index;
                scope.spawn(move || {
                    let target = second.target();
                    index.retire_source(&target.instance_id, target.dp_rank);
                    let _ = retired.send(());
                });
                completed.recv_timeout(std::time::Duration::from_secs(5))
            })
            .unwrap()
            .expect("an independent rank can retire during this snapshot");
    });
    assert!(visible_tiers(&index, &first, 0, [1; 32]).contains(StorageTier::Device));
}

#[test]
fn read_snapshot_does_not_recreate_unknown_or_retired_source() {
    let index = KvBlockIndexer::new();
    let owner = worker("retired", 0);
    let target = owner.target();
    assert!(index.with_source_snapshot(&target, |_| ()).is_none());
    store(&index, &owner, 0, [1; 32], StorageTier::Device);
    let before = index
        .with_source_snapshot(&target, |view| view.source_version())
        .unwrap();
    index.retire_source(&target.instance_id, target.dp_rank);
    assert!(index.with_source_snapshot(&target, |_| ()).is_none());
    assert!(index.source_locks.read().is_empty());

    store(&index, &owner, 0, [2; 32], StorageTier::Device);
    assert!(!before.is_current());
    assert_eq!(
        visible_tiers(&index, &owner, 0, [1; 32]),
        TierBits::default()
    );
    assert!(visible_tiers(&index, &owner, 0, [2; 32]).contains(StorageTier::Device));
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
                        true,
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
        let groups = index.collect_hash_groups(0u32..2, None, &[key]);
        assert!(groups[&0.into()]
            .validate_source(&index, &owner.target(), &[key])
            .is_none());
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
    assert!(visible_tiers(&index, &keep, 0, key).contains(StorageTier::Device));
}

#[test]
fn page_positions_are_independent_of_store_batch_boundaries() {
    let owner = worker("a", 0);
    let stored = [[1; 32], [2; 32], [3; 32], [4; 32], [5; 32], [6; 32]];
    // Collect the same hash twice and place a miss between two hits.
    let requested = [stored[0], stored[1], stored[1], [0; 32], stored[4]];
    let hit = TierBits(TierBits::bit(StorageTier::Device));
    for shape in [&[4, 2][..], &[1; 6][..], &[2; 3][..]] {
        let index = KvBlockIndexer::new();
        let mut pages = stored.as_slice();
        for &count in shape {
            let (chunk, rest) = pages.split_at(count);
            index.with_source_batch(&owner.target(), |tx| {
                tx.store_hash_pages(&owner, 0, StorageTier::Device, None, chunk, true, 8)
                    .unwrap()
            });
            pages = rest;
        }

        let evidence = index.collect_hash_pages(0, &requested);
        assert_eq!(
            evidence.for_owner(&owner).as_ref(),
            &[hit, hit, hit, TierBits::default(), hit],
        );
    }
}

#[test]
fn page_evidence_can_cross_only_one_known_disjoint_batch() {
    let index = KvBlockIndexer::new();
    let owner = worker("worker", 0);
    let target = owner.target();
    let control = Arc::new(StreamControl::default());
    let requested = [[1; 32]];
    let unrelated = [2; 32];
    let mut collision = requested[0];
    collision[31] ^= 1;
    let plan = |key: &SequenceHash| {
        PageChanges::from_batches(std::iter::once(Some(std::slice::from_ref(key))), 1)
    };

    for group in 0..2 {
        store(&index, &owner, group, requested[0], StorageTier::Device);
    }
    let collect = || index.collect_hash_groups(0u32..2, None, &requested);
    let can_reuse = |evidence: &GroupPages| {
        PageEvidence::unchanged_groups(
            &index,
            &target,
            &owner,
            &requested,
            (0..2).map(|group| evidence.get(&group.into())),
        )
        .is_some()
    };
    let exact = |evidence: &GroupPages| {
        index
            .with_source_snapshot(&target, |snapshot| {
                snapshot
                    .hash_page_tiers(&owner, None, &requested, evidence, 0..2)
                    .iter()
                    .map(|pages| {
                        pages
                            .first()
                            .is_some_and(|page| page.contains(StorageTier::Device))
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap()
    };

    let before = collect();
    index.with_page_mutation_batch(&target, &owner, &control, plan(&unrelated), |_, tx| {
        tx.store_hash_pages(&owner, 1, StorageTier::Device, None, &[unrelated], true, 8)
            .unwrap();
        assert!(can_reuse(&before));
    });
    assert!(can_reuse(&before));

    let before_eviction = collect();
    index.with_page_mutation_batch(&target, &owner, &control, plan(&requested[0]), |_, tx| {
        tx.remove_hash_pages(&owner, 1, StorageTier::Device, &requested, true)
            .unwrap();
        assert!(!can_reuse(&before_eviction));
    });
    assert!(!can_reuse(&before_eviction));
    index.with_page_mutation_batch(&target, &owner, &control, plan(&unrelated), |_, tx| {
        tx.remove_hash_pages(&owner, 1, StorageTier::Device, &[unrelated], true)
            .unwrap();
    });
    assert!(!can_reuse(&before_eviction));
    assert_eq!(exact(&before_eviction), [true, false]);

    let before_collision = collect();
    index.with_page_mutation_batch(&target, &owner, &control, plan(&collision), |_, tx| {
        tx.store_hash_pages(&owner, 1, StorageTier::Device, None, &[collision], true, 8)
            .unwrap();
    });
    assert!(!can_reuse(&before_collision));
    assert_eq!(exact(&before_collision), [true, false]);

    let before_clear = collect();
    index.with_page_mutation_batch(&target, &owner, &control, plan(&unrelated), |_, tx| {
        // An otherwise disjoint batch can invalidate its entire owner on an admission error.
        tx.clear_owner_exact(&owner);
        assert!(!can_reuse(&before_clear));
    });
    assert!(!can_reuse(&before_clear));
    assert_eq!(exact(&before_clear), [false, false]);
}

#[test]
fn shared_miss_tracks_only_requested_rank_publication() {
    let index = KvBlockIndexer::new();
    let first = worker("worker", 0);
    let other = worker("worker", 1);
    let requested = [1; 32];
    let retained = [2; 32];
    let target = first.target();
    store(&index, &first, 0, retained, StorageTier::Device);
    store(&index, &other, 0, retained, StorageTier::Device);
    let miss = index.collect_hash_pages(0, &[requested]);
    assert!(miss.scan.confirms_miss(&index));
    let verified_miss = |evidence: &PageEvidence, owner: &ResidencyOwner| {
        PageEvidence::unchanged_groups(
            &index,
            &owner.target(),
            owner,
            &[requested],
            std::iter::once(Some(evidence)),
        )
        .map(|m| matches!(m, PageMatch::Miss))
    };
    assert_eq!(verified_miss(&miss, &first), Some(true));

    // An unrelated rank's commit does not force this rank's proof to be refreshed.
    store(&index, &other, 0, requested, StorageTier::HostPinned);
    assert!(!miss.scan.confirms_miss(&index));
    assert_eq!(verified_miss(&miss, &first), Some(true));
    assert_eq!(verified_miss(&miss, &other), None);
    let sparse = index.collect_hash_pages(0, &[requested]);
    assert_eq!(verified_miss(&sparse, &first), Some(true));
    assert_eq!(sparse.scan.confirmed.get(), Some(&true));
    assert_eq!(verified_miss(&sparse, &other), Some(false));

    // A later group's pages cannot restore a prefix without the first group.
    store(&index, &first, 1, requested, StorageTier::Device);
    let groups = index.collect_hash_groups(0u32..2, None, &[requested]);
    let hybrid_match = || {
        PageEvidence::unchanged_groups(
            &index,
            &target,
            &first,
            &[requested],
            (0..2).map(|group| groups.get(&group.into())),
        )
    };
    assert!(matches!(hybrid_match(), Some(PageMatch::Miss)));

    index.with_source_batch(&target, |tx| {
        // The confirmed miss remains valid at the published cutoff until commit.
        assert!(matches!(hybrid_match(), Some(PageMatch::Miss)));
        tx.store_hash_pages(&first, 0, StorageTier::Device, None, &[requested], true, 8)
            .unwrap();
    });
    assert!(hybrid_match().is_none());

    let old = index.collect_hash_pages(0, &[requested]);
    index.retire_source(&target.instance_id, target.dp_rank);
    let replacement = index.source_lock(&target);
    assert!(!replacement.committed_at(&old.scan));
    assert!(old.validate_source(&index, &target, &[requested]).is_none());
}

#[test]
fn shared_miss_can_cross_an_uncommitted_store() {
    let index = KvBlockIndexer::new();
    let holder = worker("worker", 0);
    let key = [1; 32];
    let observed = index.collect_hash_pages(0, &[key]);
    index.with_source_batch(&holder.target(), |tx| {
        tx.store_hash_pages(&holder, 0, StorageTier::Device, None, &[key], true, 8)
            .unwrap();
        // The page was absent at the published cutoff. Its store cannot hide
        // a prior hit, and the old miss expires when this batch commits.
        assert!(observed.scan.confirms_miss(&index));
    });
    assert!(!observed.scan.confirms_miss(&index));
    assert!(visible_tiers(&index, &holder, 0, key).contains(StorageTier::Device));
}

#[test]
fn shared_miss_does_not_accept_an_uncommitted_removal() {
    let index = KvBlockIndexer::new();
    let holder = worker("worker", 0);
    let key = [1; 32];
    let group = PageQuery {
        group: 0,
        stop_at_gap: true,
    };
    store(&index, &holder, 0, key, StorageTier::Device);
    let observed = index.with_source_batch(&holder.target(), |tx| {
        tx.remove_hash_pages(&holder, 0, StorageTier::Device, &[key], true)
            .unwrap();
        let observed = index
            .collect_prefix(group, &[key])
            .expect("an unpublished removal must leave evidence for exact fallback");
        assert!(!observed.confirms_empty(&index));
        assert!(PageEvidence::unchanged_groups(
            &index,
            &holder.target(),
            &holder,
            &[key],
            std::iter::once(Some(&observed)),
        )
        .is_none());
        tx.store_hash_pages(&holder, 0, StorageTier::Device, None, &[key], true, 8)
            .unwrap();
        observed
    });
    assert!(!observed.scan.confirms_miss(&index));
    assert!(visible_tiers(&index, &holder, 0, key).contains(StorageTier::Device));
    assert!(index.collect_prefix(group, &[key]).is_some());
    assert!(index.collect_prefix(group, &[[2; 32]]).is_none());
}

#[test]
fn shared_first_page_miss_ignores_changes_only_beyond_the_prefix() {
    let index = KvBlockIndexer::new();
    let owner = worker("worker", 0);
    let target = owner.target();
    let control = Arc::new(StreamControl::default());
    let keys = [[1; 32], [2; 32]];
    let prefix = PageQuery {
        group: 0,
        stop_at_gap: true,
    };
    let complete = PageQuery::from(0);
    store(&index, &owner, 0, [3; 32], StorageTier::Device);
    let evidence = index.collect_hash_groups([prefix, complete].into_iter(), None, &keys);
    let matched = |query| {
        PageEvidence::unchanged_groups(
            &index,
            &target,
            &owner,
            &keys,
            std::iter::once(evidence.get(&query)),
        )
    };

    // Even within a planned transaction, only the prefix scan can ignore a
    // residency change to the second page while the first one is absent.
    let plan = PageChanges::from_batches(std::iter::once(Some(&keys[1..])), 1);
    index.with_page_mutation_batch(&target, &owner, &control, plan, |_, tx| {
        tx.remove_hash_pages(&owner, 0, StorageTier::Device, &keys[1..], true)
            .unwrap();
        tx.store_hash_pages(&owner, 0, StorageTier::Device, None, &keys[1..], true, 8)
            .unwrap();
        assert!(matches!(matched(prefix), Some(PageMatch::Miss)));
        assert!(matched(complete).is_none());
    });

    store(&index, &owner, 0, keys[0], StorageTier::Device);
    assert!(matched(prefix).is_none());
    assert!(visible_tiers(&index, &owner, 0, keys[0]).contains(StorageTier::Device));
}

#[test]
fn oversized_or_unbounded_mutations_require_an_exact_snapshot() {
    let hashes = [[1; 32], [2; 32]];
    assert!(
        PageChanges::from_batches([Some(&hashes[..1]), Some(&hashes[1..])].into_iter(), 1)
            .is_none()
    );
    assert!(PageChanges::from_batches([Some(&hashes[..]), None].into_iter(), 2).is_none());
    assert!(PageChanges::from_batches([Some(&hashes[..])].into_iter(), 2).is_some());
}
