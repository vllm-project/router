#![cfg(all(test, feature = "cpu", feature = "ready"))]

use sha2::{Digest, Sha256};

use super::*;

fn content(label: &str) -> [u8; 32] {
    Sha256::digest(label.as_bytes()).into()
}

fn worker(name: &str) -> ResidencyOwner {
    ResidencyOwner::Worker {
        source: SourceId::from(name),
        dp_rank: 0,
        incarnation: 0,
    }
}

fn worker_rank(name: &str, rank: u32) -> ResidencyOwner {
    ResidencyOwner::Worker {
        source: SourceId::from(name),
        dp_rank: rank,
        incarnation: 0,
    }
}

fn pool(pool_id: &str, pushed_by: &str) -> ResidencyOwner {
    ResidencyOwner::CacheOwner {
        pool_id: Arc::from(pool_id),
        source: SourceId::from(pushed_by),
        dp_rank: 0,
    }
}

fn query(hashes: &[&str], tiers: &[StorageTier]) -> MatchQuery {
    MatchQuery {
        group_idx: 0,
        local_hashes: hashes.iter().map(|h| content(h)).collect(),
        tiers_of_interest: tiers.to_vec(),
    }
}

fn query_group(group: u32, hashes: &[&str], tiers: &[StorageTier]) -> MatchQuery {
    MatchQuery {
        group_idx: group,
        local_hashes: hashes.iter().map(|h| content(h)).collect(),
        tiers_of_interest: tiers.to_vec(),
    }
}

fn h(s: &str) -> Arc<str> {
    Arc::<str>::from(s)
}

fn segments_of(res: &[TierMatch], target_id: &str, tier: StorageTier) -> Vec<(u32, u32)> {
    res.iter()
        .find(|m| &*m.target.instance_id == target_id && m.tier == tier)
        .map(|m| m.segments.clone())
        .unwrap_or_default()
}

fn depth_of(res: &[TierMatch], target_id: &str, tier: StorageTier) -> u32 {
    segments_of(res, target_id, tier)
        .into_iter()
        .find(|(start, _)| *start == 0)
        .map_or(0, |(_, end)| end)
}

#[test]
fn retired_streams_are_removed_without_reusing_snapshot_generations() {
    let index = KvBlockIndexer::new();
    let owner = worker("a");
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
    let owner = worker("retired");
    let target = owner.target();
    let read = || {
        let _ = index.with_source_snapshot(&target, |_| ());
        assert!(!index.source_locks.read().contains_key(&target));
    };
    read();
    index.store(0, owner, StorageTier::Device, None, &[(h("page"), [1; 32])]);
    index.retire_source(&target.instance_id, target.dp_rank);
    read();
}

#[test]
fn source_suspension_and_invalidation_cover_tiers_groups_and_only_that_rank() {
    let idx = KvBlockIndexer::new();
    let tiers = [
        StorageTier::Device,
        StorageTier::HostPinned,
        StorageTier::External,
    ];
    for group in [0, 1] {
        for (owner, tier, seq) in [
            (worker("a"), StorageTier::Device, "a-gpu"),
            (worker("a"), StorageTier::HostPinned, "a-cpu"),
            (pool("shared", "a"), StorageTier::External, "a-pool"),
            (worker_rank("a", 1), StorageTier::Device, "a-rank1"),
            (pool("shared", "b"), StorageTier::External, "b-pool"),
        ] {
            idx.store(group, owner, tier, None, &[(h(seq), content("x"))]);
        }
        assert_eq!(
            idx.find_matches(&query_group(group, &["x"], &tiers)).len(),
            5
        );
    }
    let source = SourceId::from("a");
    for _ in 0..3 {
        idx.resume_source(&source, 0);
    }
    idx.suspend_source(&source, 0);
    for group in [0, 1] {
        let hits = idx.find_matches(&query_group(group, &["x"], &tiers));
        assert_eq!(hits.len(), 2);
        assert!(hits
            .iter()
            .all(|hit| hit.target.instance_id != source || hit.target.dp_rank != 0));
    }
    // Complete replay may resume credit without forgetting still-valid claims.
    idx.resume_source(&source, 0);
    assert_eq!(idx.find_matches(&query(&["x"], &tiers)).len(), 5);
    idx.resume_source(&source, 0);
    idx.invalidate_source(&source, 0);
    idx.resume_source(&source, 0);
    for group in [0, 1] {
        assert_eq!(
            idx.find_matches(&query_group(group, &["x"], &tiers)).len(),
            2
        );
    }
    // Only a fresh store can restore a cleared observation.
    idx.store(
        0,
        worker("a"),
        StorageTier::Device,
        None,
        &[(h("fresh"), content("y"))],
    );
    assert_eq!(idx.find_matches(&query(&["y"], &tiers)).len(), 1);
    assert_eq!(idx.find_matches(&query(&["x"], &tiers)).len(), 2);
}

#[test]
fn source_batches_publish_atomically_to_dense_queries_and_group_snapshots() {
    use std::sync::mpsc;
    use std::time::Duration;

    for dense in [false, true] {
        let index = Arc::new(KvBlockIndexer::new());
        let owner = worker("a");
        if dense {
            index.store(
                0,
                owner.clone(),
                StorageTier::Device,
                None,
                &[(h("s1"), content("x")), (h("s2"), content("y"))],
            );
        }
        let (entered, ready) = mpsc::channel();
        let (release, resume) = mpsc::channel();
        let writer_index = index.clone();
        let writer_owner = owner.clone();
        let writer = std::thread::spawn(move || {
            writer_index.with_source_batch(&writer_owner.target(), |batch| {
                if dense {
                    batch.remove_full_pages(
                        0,
                        &writer_owner,
                        StorageTier::Device,
                        &[h("s1")],
                        false,
                    );
                } else {
                    batch
                        .store_hash_pages(
                            &writer_owner,
                            0,
                            StorageTier::Device,
                            None,
                            &[[1; 32]],
                            false,
                            8,
                        )
                        .unwrap();
                }
                entered.send(()).unwrap();
                resume.recv_timeout(Duration::from_secs(5)).unwrap();
                if dense {
                    batch.remove_full_pages(
                        0,
                        &writer_owner,
                        StorageTier::Device,
                        &[h("s2")],
                        false,
                    );
                } else {
                    batch
                        .store_hash_pages(
                            &writer_owner,
                            1,
                            StorageTier::Device,
                            None,
                            &[[1; 32]],
                            false,
                            8,
                        )
                        .unwrap();
                }
            });
        });
        ready.recv_timeout(Duration::from_secs(5)).unwrap();
        let (started, scheduled) = mpsc::channel();
        let (done, result) = mpsc::channel();
        let reader_index = index.clone();
        let reader = std::thread::spawn(move || {
            started.send(()).unwrap();
            let complete = if dense {
                reader_index
                    .find_matches(&query(&["x", "y"], &[StorageTier::Device]))
                    .is_empty()
            } else {
                reader_index
                    .with_source_snapshot(&owner.target(), |view| {
                        [0, 1].into_iter().all(|group| {
                            view.hash_page_tiers(
                                &owner,
                                None,
                                &[[1; 32]],
                                &HashMap::new(),
                                std::iter::once(group),
                            )[0][0]
                                .contains(StorageTier::Device)
                        })
                    })
                    .unwrap()
            };
            done.send(complete).unwrap();
        });
        scheduled.recv_timeout(Duration::from_secs(5)).unwrap();
        let premature = result.recv_timeout(Duration::from_millis(100));
        release.send(()).unwrap();
        writer.join().unwrap();
        reader.join().unwrap();
        assert!(
            matches!(premature, Err(mpsc::RecvTimeoutError::Timeout)),
            "a query observed an incomplete source transaction (dense={dense})"
        );
        assert!(result.recv_timeout(Duration::from_secs(5)).unwrap());
    }
}

#[test]
fn retiring_many_sources_reclaims_gates_without_erasing_other_owners() {
    let index = KvBlockIndexer::new();
    index.store(
        0,
        worker("keep"),
        StorageTier::Device,
        None,
        &[(h("keep-hash"), content("x"))],
    );
    for id in 0..1000 {
        let name = format!("retired-{id}");
        let owner = worker(&name);
        index.store(
            0,
            owner,
            StorageTier::Device,
            None,
            &[(h("source-hash"), content("x"))],
        );
        index.retire_source(&SourceId::from(name), 0);
    }
    assert_eq!(index.source_locks.read().len(), 1);
    assert!(index.suspended_sources.is_empty());
    let matches = index.find_matches(&query(&["x"], &[StorageTier::Device]));
    assert_eq!(depth_of(&matches, "keep", StorageTier::Device), 1);
}

#[test]
fn interior_remove_breaks_prefix_descendants_survive() {
    // [a, b, c] as separate parent-chained nodes. Removing b (interior)
    // stops the prefix walk at a; c's claim survives but isn't credited
    // past the gap. Re-storing b re-links and credits the full prefix.
    let idx = KvBlockIndexer::new();
    idx.store(
        0,
        worker("w0"),
        StorageTier::Device,
        None,
        &[(h("sa"), content("a"))],
    );
    idx.store(
        0,
        worker("w0"),
        StorageTier::Device,
        Some(h("sa")),
        &[(h("sb"), content("b"))],
    );
    idx.store(
        0,
        worker("w0"),
        StorageTier::Device,
        Some(h("sb")),
        &[(h("sc"), content("c"))],
    );

    idx.remove(0, &worker("w0"), StorageTier::Device, &[h("sb")]);
    let res = idx.find_matches(&query(&["a", "b", "c"], &[StorageTier::Device]));
    // b removed mid-prefix: c survives as a separate segment. The cost
    // model credits only [0, d_gpu); the index reports both.
    assert_eq!(
        segments_of(&res, "w0", StorageTier::Device),
        vec![(0, 1), (2, 3)]
    );

    idx.store(
        0,
        worker("w0"),
        StorageTier::Device,
        Some(h("sa")),
        &[(h("sb"), content("b"))],
    );
    let res = idx.find_matches(&query(&["a", "b", "c"], &[StorageTier::Device]));
    assert_eq!(segments_of(&res, "w0", StorageTier::Device), vec![(0, 3)]);
}

#[test]
fn late_child_waits_for_reclaimed_parent_and_reconnects_without_new_store() {
    let idx = KvBlockIndexer::new();
    let owner = worker("w0");
    idx.store(
        0,
        owner.clone(),
        StorageTier::Device,
        None,
        &[(h("sa"), content("a"))],
    );
    idx.remove(0, &owner, StorageTier::Device, &[h("sa")]);
    idx.store(
        0,
        owner.clone(),
        StorageTier::Device,
        Some(h("sa")),
        &[(h("sb"), content("b"))],
    );
    let hits = idx.find_matches(&query(&["a", "b"], &[StorageTier::Device]));
    assert!(
        hits.is_empty(),
        "a missing parent cannot prove a reusable prefix"
    );
    idx.store(
        0,
        owner,
        StorageTier::Device,
        None,
        &[(h("sa"), content("a"))],
    );
    let hits = idx.find_matches(&query(&["a", "b"], &[StorageTier::Device]));
    assert_eq!(segments_of(&hits, "w0", StorageTier::Device), vec![(0, 2)]);
}

#[test]
fn clear_cache_owner_by_pool_id() {
    // Two pools pushed by the same worker share one (target, tier) entry,
    // so find_matches can't tell them apart — verify their records instead.
    let idx = KvBlockIndexer::new();
    idx.store(
        0,
        pool("poolA", "w0"),
        StorageTier::External,
        None,
        &[(h("h0"), content("h0"))],
    );
    idx.store(
        0,
        pool("poolB", "w0"),
        StorageTier::External,
        None,
        &[(h("h0"), content("h0"))],
    );
    idx.store(
        0,
        worker("w0"),
        StorageTier::Device,
        None,
        &[(h("h0"), content("h0"))],
    );

    idx.clear(&pool("poolA", "w0"), ClearScope::CacheOwner);

    let pages = idx.collect_hash_pages(0, &[prefix_key(&[0; 32], &content("h0"))]);
    assert!(pages.for_owner(&pool("poolA", "w0")).is_empty());
    assert!(pages.for_owner(&pool("poolB", "w0"))[0].contains(StorageTier::External));
    assert!(pages.for_owner(&worker("w0"))[0].contains(StorageTier::Device));
}

#[test]
fn cache_owner_hit_attributed_to_pushing_worker() {
    // A pool block pushed by w0: the hit's target is w0 (it can read what
    // it pushed), at the External tier, Remote locality.
    let idx = KvBlockIndexer::new();
    idx.store(
        0,
        pool("mooncake", "w0"),
        StorageTier::External,
        None,
        &[(h("h0"), content("h0"))],
    );

    let res = idx.find_matches(&query(&["h0"], &[StorageTier::External]));
    assert_eq!(res.len(), 1);
    let m = &res[0];
    assert_eq!(&*m.target.instance_id, "w0");
    assert_eq!(m.tier, StorageTier::External);
    assert_eq!(m.locality, Locality::Remote);
}

#[test]
fn device_plus_host_continuation_reports_both_tiers() {
    // Device holds [0,3) and host holds [3,10) as a parent-chained tail:
    // host lacks block 0 yet the engine reuses the 7-block tail, so the
    // index reports each tier's held intervals.
    let idx = KvBlockIndexer::new();
    idx.store(
        0,
        worker("w0"),
        StorageTier::Device,
        None,
        &[
            (h("s0"), content("h0")),
            (h("s1"), content("h1")),
            (h("s2"), content("h2")),
        ],
    );
    idx.store(
        0,
        worker("w0"),
        StorageTier::HostPinned,
        Some(h("s2")),
        &[
            (h("s3"), content("h3")),
            (h("s4"), content("h4")),
            (h("s5"), content("h5")),
            (h("s6"), content("h6")),
            (h("s7"), content("h7")),
            (h("s8"), content("h8")),
            (h("s9"), content("h9")),
        ],
    );

    let res = idx.find_matches(&query(
        &["h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9"],
        &[StorageTier::Device, StorageTier::HostPinned],
    ));
    assert_eq!(segments_of(&res, "w0", StorageTier::Device), vec![(0, 3)]);
    assert_eq!(
        segments_of(&res, "w0", StorageTier::HostPinned),
        vec![(3, 10)]
    );
}

#[test]
fn same_depth_multiple_owners_no_duplicate_segment() {
    // Two pools push the same block to the same worker at the same tier:
    // one (target, tier) at one depth must extend the segment by one, not
    // once per owner — otherwise the run-length loop double-counts the
    // depth into two identical segments.
    let idx = KvBlockIndexer::new();
    idx.store(
        0,
        pool("poolA", "w0"),
        StorageTier::External,
        None,
        &[(h("s0"), content("h0"))],
    );
    idx.store(
        0,
        pool("poolB", "w0"),
        StorageTier::External,
        None,
        &[(h("s0"), content("h0"))],
    );

    let res = idx.find_matches(&query(&["h0"], &[StorageTier::External]));
    assert_eq!(segments_of(&res, "w0", StorageTier::External), vec![(0, 1)]);
}

#[test]
fn removed_pending_block_cannot_reappear_when_parent_is_restored() {
    let index = KvBlockIndexer::new();
    let owner = worker("pending-remove");
    index.store(
        0,
        owner.clone(),
        StorageTier::Device,
        Some(h("a")),
        &[(h("b"), content("B"))],
    );
    index.remove(0, &owner, StorageTier::Device, &[h("b")]);
    index.store(
        0,
        owner,
        StorageTier::Device,
        None,
        &[(h("a"), content("A"))],
    );
    let hits = index.find_matches(&query(&["A", "B"], &[StorageTier::Device]));
    assert_eq!(
        segments_of(&hits, "pending-remove", StorageTier::Device),
        vec![(0, 1)]
    );
}

#[test]
fn cyclic_pending_chains_are_rejected_without_credit() {
    let index = KvBlockIndexer::new();
    let owner = worker("cycle");
    index.store(
        0,
        owner.clone(),
        StorageTier::Device,
        Some(h("a")),
        &[(h("b"), content("B"))],
    );
    index.store(
        0,
        owner,
        StorageTier::Device,
        Some(h("b")),
        &[(h("a"), content("A"))],
    );
    assert!(index
        .find_matches(&query(&["A", "B"], &[StorageTier::Device]))
        .is_empty());
}
