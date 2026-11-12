#![cfg(all(feature = "cpu", feature = "ready"))]

use std::sync::Arc;

use vllm_router_rs::kv_index::{
    ClearScope, KvBlockIndexer, MatchQuery, ResidencyOwner, StorageTier, TierMatch,
};

fn prefix_depth(hit: &TierMatch) -> u32 {
    match hit.segments.first() {
        Some((0, end)) => *end,
        _ => 0,
    }
}

#[test]
fn a_partial_compressed_edge_match_must_not_skip_to_its_child() {
    let index = KvBlockIndexer::new();
    let owner = ResidencyOwner::Worker {
        source: Arc::from("worker"),
        dp_rank: 0,
        incarnation: 0,
    };
    // Keep [x, y] compressed: no other store may split this edge before the query.
    index.store(
        0,
        owner.clone(),
        StorageTier::Device,
        None,
        &[
            (Arc::from("s1"), Arc::from("x")),
            (Arc::from("s2"), Arc::from("y")),
        ],
    );
    index.store(
        0,
        owner,
        StorageTier::Device,
        Some(Arc::from("s2")),
        &[(Arc::from("s3"), Arc::from("z"))],
    );
    let hits = index.find_matches(&MatchQuery {
        group_idx: 0,
        local_hashes: vec![Arc::from("x"), Arc::from("z")],
        tiers_of_interest: vec![StorageTier::Device],
    });
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].segments, vec![(0, 1)], "y was not matched");
}

fn worker(source: &str, dp_rank: u32) -> ResidencyOwner {
    ResidencyOwner::Worker {
        source: Arc::from(source),
        dp_rank,
        incarnation: 0,
    }
}

fn store(
    index: &KvBlockIndexer,
    group_idx: u32,
    owner: &ResidencyOwner,
    tier: StorageTier,
    parent: Option<&str>,
    blocks: &[(&str, &str)],
) {
    index.store(
        group_idx,
        owner.clone(),
        tier,
        parent.map(Arc::from),
        &blocks
            .iter()
            .map(|(seq, local)| (Arc::from(*seq), Arc::from(*local)))
            .collect::<Vec<_>>(),
    );
}

fn query(group_idx: u32, hashes: &[&str], tiers: &[StorageTier]) -> MatchQuery {
    MatchQuery {
        group_idx,
        local_hashes: hashes.iter().map(|hash| Arc::from(*hash)).collect(),
        tiers_of_interest: tiers.to_vec(),
    }
}

fn compressed_prefix_with_child() -> KvBlockIndexer {
    let index = KvBlockIndexer::new();
    let owner = worker("worker", 0);
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        None,
        &[("s1", "x"), ("s2", "y")],
    );
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        Some("s2"),
        &[("s3", "z")],
    );
    index
}

fn branch_at_first_compressed_block() -> KvBlockIndexer {
    let index = KvBlockIndexer::new();
    let owner = worker("worker", 0);
    // No other insertion may split this edge before the engine parent is used.
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        None,
        &[("sx", "x"), ("sy", "y"), ("sz", "z")],
    );
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        Some("sx"),
        &[("sw", "w")],
    );
    index
}

#[test]
fn interior_parent_branch_preserves_prefix_boundaries() {
    let index = branch_at_first_compressed_block();
    for (hashes, expected_depth) in [
        (vec!["x", "w"], 2),
        (vec!["x", "y", "z"], 3),
        (vec!["x", "y", "z", "w"], 3),
    ] {
        let hits = index.find_matches(&query(0, &hashes, &[StorageTier::Device]));
        assert_eq!(hits.len(), 1, "query {hashes:?}");
        assert_eq!(prefix_depth(&hits[0]), expected_depth, "query {hashes:?}");
    }
}

#[test]
fn middle_and_tail_engine_parents_keep_their_exact_prefix() {
    let owner = worker("worker", 0);
    for (parent, branch, branch_depth) in [
        ("sy", vec!["x", "y", "w"], 3),
        ("sz", vec!["x", "y", "z", "w"], 4),
    ] {
        let index = KvBlockIndexer::new();
        store(
            &index,
            0,
            &owner,
            StorageTier::Device,
            None,
            &[("sx", "x"), ("sy", "y"), ("sz", "z")],
        );
        store(
            &index,
            0,
            &owner,
            StorageTier::Device,
            Some(parent),
            &[("sw", "w")],
        );
        for (hashes, expected_depth) in [
            (branch, branch_depth),
            (vec!["x", "y", "z"], 3),
            (vec!["x"], 1),
            (vec!["x", "y"], 2),
        ] {
            let hits = index.find_matches(&query(0, &hashes, &[StorageTier::Device]));
            assert_eq!(hits.len(), 1, "parent {parent}, query {hashes:?}");
            assert_eq!(
                prefix_depth(&hits[0]),
                expected_depth,
                "parent {parent}, query {hashes:?}"
            );
        }
        if parent == "sy" {
            let hashes = ["x", "y", "z", "w"];
            let hits = index.find_matches(&query(0, &hashes, &[StorageTier::Device]));
            assert_eq!(hits.len(), 1, "parent {parent}, query {hashes:?}");
            assert_eq!(
                prefix_depth(&hits[0]),
                3,
                "parent {parent}, query {hashes:?}"
            );
        }
        assert!(index
            .find_matches(&query(0, &["w"], &[StorageTier::Device]))
            .is_empty());
    }
}

#[test]
fn existing_interior_branch_is_idempotent_and_extends_contiguously() {
    let index = branch_at_first_compressed_block();
    let owner = worker("worker", 0);
    for _ in 0..2 {
        store(
            &index,
            0,
            &owner,
            StorageTier::Device,
            Some("sx"),
            &[("sw", "w")],
        );
        store(
            &index,
            0,
            &owner,
            StorageTier::Device,
            Some("sx"),
            &[("sw", "w"), ("sv", "v")],
        );
        store(
            &index,
            0,
            &owner,
            StorageTier::Device,
            Some("sv"),
            &[("su", "u")],
        );
    }
    for (hashes, expected_depth) in [
        (vec!["x", "w"], 2),
        (vec!["x", "w", "v", "u"], 4),
        (vec!["x", "w", "u"], 2),
        (vec!["x", "y", "z"], 3),
        (vec!["x", "y", "z", "w"], 3),
    ] {
        let hits = index.find_matches(&query(0, &hashes, &[StorageTier::Device]));
        assert_eq!(hits.len(), 1, "query {hashes:?}");
        assert_eq!(prefix_depth(&hits[0]), expected_depth, "query {hashes:?}");
    }
}

#[test]
fn repeated_local_hashes_have_distinct_interior_engine_parents() {
    let index = KvBlockIndexer::new();
    let owner = worker("worker", 0);
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        None,
        &[("sx1", "x"), ("sy", "y"), ("sx2", "x"), ("sz", "z")],
    );
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        Some("sx1"),
        &[("sw", "w")],
    );
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        Some("sx2"),
        &[("sv", "v")],
    );
    for (hashes, expected_depth) in [
        (vec!["x", "w"], 2),
        (vec!["x", "y", "x", "v"], 4),
        (vec!["x", "y", "x", "z"], 4),
        (vec!["x", "v"], 1),
        (vec!["x", "y", "x", "w"], 3),
        (vec!["x", "y", "x", "z", "v"], 4),
    ] {
        let hits = index.find_matches(&query(0, &hashes, &[StorageTier::Device]));
        assert_eq!(hits.len(), 1, "query {hashes:?}");
        assert_eq!(prefix_depth(&hits[0]), expected_depth, "query {hashes:?}");
    }
}

#[test]
fn interior_parent_splits_preserve_children_and_relocate_suffix_anchors() {
    let index = KvBlockIndexer::new();
    let owner = worker("worker", 0);
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        None,
        &[("sx", "x"), ("sy", "y"), ("sz", "z")],
    );
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        Some("sz"),
        &[("sa", "a"), ("sb", "b")],
    );
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        Some("sx"),
        &[("sw", "w")],
    );
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        Some("sy"),
        &[("sq", "q")],
    );
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        Some("sb"),
        &[("sc", "c")],
    );
    for (hashes, expected_depth) in [
        (vec!["x", "w"], 2),
        (vec!["x", "y", "q"], 3),
        (vec!["x", "y", "z", "a", "b", "c"], 6),
        (vec!["x", "y", "z", "w"], 3),
        (vec!["x", "y", "z", "a", "c"], 4),
    ] {
        let hits = index.find_matches(&query(0, &hashes, &[StorageTier::Device]));
        assert_eq!(hits.len(), 1, "query {hashes:?}");
        assert_eq!(prefix_depth(&hits[0]), expected_depth, "query {hashes:?}");
    }
}

#[test]
fn interior_parent_split_preserves_isolated_owner_rank_tier_and_group_claims() {
    let index = KvBlockIndexer::new();
    let original = worker("worker", 0);
    let other_worker = worker("other", 0);
    let other_rank = worker("worker", 1);
    let pool = ResidencyOwner::CacheOwner {
        pool_id: Arc::from("pool"),
        source: Arc::from("worker"),
        dp_rank: 0,
    };
    let claims = [
        (&original, StorageTier::Device),
        (&other_worker, StorageTier::Device),
        (&other_rank, StorageTier::Device),
        (&original, StorageTier::HostPinned),
        (&pool, StorageTier::External),
    ];
    for (owner, tier) in claims {
        // All claims share the same engine identities, not separate aliases.
        store(
            &index,
            0,
            owner,
            tier,
            None,
            &[("sx", "x"), ("sy", "y"), ("sz", "z")],
        );
    }
    store(
        &index,
        0,
        &original,
        StorageTier::Device,
        Some("sx"),
        &[("sw", "w")],
    );
    store(
        &index,
        1,
        &other_worker,
        StorageTier::Device,
        None,
        &[("sx", "x"), ("sw", "w")],
    );
    let tiers = [
        StorageTier::Device,
        StorageTier::HostPinned,
        StorageTier::External,
    ];
    for (hashes, original_depth, other_depth) in [
        (vec!["x", "w"], 2, 1),
        (vec!["x", "y", "z"], 3, 3),
        (vec!["x", "y", "z", "w"], 3, 3),
    ] {
        let hits = index.find_matches(&query(0, &hashes, &tiers));
        assert_eq!(hits.len(), claims.len(), "query {hashes:?}");
        for (owner, tier) in claims {
            let hit = hits
                .iter()
                .find(|hit| hit.target == owner.target() && hit.tier == tier)
                .expect("every original claim survives the split");
            let expected_depth = if owner == &original && tier == StorageTier::Device {
                original_depth
            } else {
                other_depth
            };
            assert_eq!(
                prefix_depth(hit),
                expected_depth,
                "{owner:?}/{tier:?}: {hashes:?}"
            );
        }
    }
    let host_hits = index.find_matches(&query(0, &["x", "w"], &[StorageTier::HostPinned]));
    assert_eq!(host_hits.len(), 1);
    assert_eq!(prefix_depth(&host_hits[0]), 1);
    let group_hits = index.find_matches(&query(1, &["x", "w"], &[StorageTier::Device]));
    assert_eq!(group_hits.len(), 1);
    assert_eq!(group_hits[0].target, other_worker.target());
    assert_eq!(prefix_depth(&group_hits[0]), 2);
    assert!(index
        .find_matches(&query(2, &["x", "w"], &tiers))
        .is_empty());
}

#[test]
fn remove_and_clear_after_an_interior_split_preserve_other_claims() {
    let index = branch_at_first_compressed_block();
    let original = worker("worker", 0);
    let other_rank = worker("worker", 1);
    let pool = ResidencyOwner::CacheOwner {
        pool_id: Arc::from("pool"),
        source: Arc::from("worker"),
        dp_rank: 0,
    };
    for (owner, tier) in [
        (&other_rank, StorageTier::Device),
        (&original, StorageTier::HostPinned),
        (&pool, StorageTier::External),
    ] {
        store(
            &index,
            0,
            owner,
            tier,
            None,
            &[("sx", "x"), ("sy", "y"), ("sz", "z")],
        );
        store(&index, 0, owner, tier, Some("sx"), &[("sw", "w")]);
    }
    store(
        &index,
        1,
        &original,
        StorageTier::Device,
        None,
        &[("sx", "x"), ("sw", "w")],
    );
    index.remove(0, &original, StorageTier::Device, &[Arc::from("sy")]);
    for (hashes, original_segments, rank_segments) in [
        (vec!["x", "y", "z"], vec![(0, 1), (2, 3)], vec![(0, 3)]),
        (vec!["x", "w"], vec![(0, 2)], vec![(0, 2)]),
    ] {
        let hits = index.find_matches(&query(0, &hashes, &[StorageTier::Device]));
        assert_eq!(hits.len(), 2, "query {hashes:?}");
        for (owner, segments) in [(&original, original_segments), (&other_rank, rank_segments)] {
            let hit = hits
                .iter()
                .find(|hit| hit.target == owner.target())
                .expect("both ranks hold x");
            assert_eq!(hit.segments, segments, "{owner:?}: {hashes:?}");
        }
    }
    index.remove(0, &original, StorageTier::Device, &[Arc::from("sw")]);
    let branch_hits = index.find_matches(&query(0, &["x", "w"], &[StorageTier::Device]));
    assert_eq!(branch_hits.len(), 2);
    let original_branch = branch_hits
        .iter()
        .find(|hit| hit.target == original.target())
        .unwrap();
    assert_eq!(original_branch.segments, vec![(0, 1)]);
    let other_rank_branch = branch_hits
        .iter()
        .find(|hit| hit.target == other_rank.target())
        .unwrap();
    assert_eq!(other_rank_branch.segments, vec![(0, 2)]);
    for hashes in [vec!["x", "w"], vec!["x", "y", "z"]] {
        let host_hits = index.find_matches(&query(0, &hashes, &[StorageTier::HostPinned]));
        assert_eq!(host_hits.len(), 1);
        assert_eq!(prefix_depth(&host_hits[0]) as usize, hashes.len());
    }
    index.clear(&original, ClearScope::Worker);
    for hashes in [vec!["x", "w"], vec!["x", "y", "z"]] {
        let hits = index.find_matches(&query(
            0,
            &hashes,
            &[
                StorageTier::Device,
                StorageTier::HostPinned,
                StorageTier::External,
            ],
        ));
        assert_eq!(hits.len(), 2, "query {hashes:?}");
        for (owner, tier) in [
            (&other_rank, StorageTier::Device),
            (&pool, StorageTier::External),
        ] {
            let hit = hits
                .iter()
                .find(|hit| hit.target == owner.target() && hit.tier == tier)
                .expect("rank and cache-owner claims survive worker clear");
            assert_eq!(
                prefix_depth(hit) as usize,
                hashes.len(),
                "{owner:?}/{tier:?}: {hashes:?}"
            );
        }
    }
    assert!(index
        .find_matches(&query(1, &["x", "w"], &[StorageTier::Device]))
        .is_empty());
    index.clear(&pool, ClearScope::CacheOwner);
    assert!(index
        .find_matches(&query(0, &["x", "w"], &[StorageTier::External]))
        .is_empty());
    let remaining = index.find_matches(&query(0, &["x", "w"], &[StorageTier::Device]));
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].target, other_rank.target());
    assert_eq!(prefix_depth(&remaining[0]), 2);
}

#[test]
fn compressed_prefix_query_boundaries() {
    let index = compressed_prefix_with_child();
    for (hashes, expected_depth) in [
        (vec!["x", "y", "z"], 3),
        (vec!["x"], 1),
        (vec!["x", "y"], 2),
        (vec!["x", "y", "z", "missing"], 3),
    ] {
        let hits = index.find_matches(&query(0, &hashes, &[StorageTier::Device]));
        assert_eq!(hits.len(), 1, "query {hashes:?}");
        assert_eq!(prefix_depth(&hits[0]), expected_depth, "query {hashes:?}");
    }
    for hashes in [vec![], vec!["missing"], vec!["y", "z"]] {
        assert!(
            index
                .find_matches(&query(0, &hashes, &[StorageTier::Device]))
                .is_empty(),
            "query {hashes:?}"
        );
    }
}

#[test]
fn a_deeper_compressed_edge_mismatch_keeps_only_the_contiguous_prefix() {
    let index = KvBlockIndexer::new();
    let owner = worker("worker", 0);
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        None,
        &[("sa", "a"), ("sb", "b")],
    );
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        Some("sb"),
        &[("sc", "c"), ("sd", "d")],
    );
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        Some("sd"),
        &[("se", "e")],
    );
    for (hashes, expected_depth) in [
        (vec!["a", "b", "c", "e"], 3),
        (vec!["a", "b", "e"], 2),
        (vec!["a", "b", "c"], 3),
        (vec!["a", "b", "c", "d", "e"], 5),
    ] {
        let hits = index.find_matches(&query(0, &hashes, &[StorageTier::Device]));
        assert_eq!(hits.len(), 1, "query {hashes:?}");
        assert_eq!(prefix_depth(&hits[0]), expected_depth, "query {hashes:?}");
    }
}

#[test]
fn repeated_local_hashes_do_not_skip_distinct_sequence_positions() {
    let index = KvBlockIndexer::new();
    let owner = worker("worker", 0);
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        None,
        &[("sa", "a"), ("sx1", "x"), ("sb", "b")],
    );
    store(
        &index,
        0,
        &owner,
        StorageTier::Device,
        Some("sb"),
        &[("sx2", "x"), ("sc", "c")],
    );
    for (hashes, expected_depth) in [
        (vec!["a", "x", "x", "c"], 2),
        (vec!["a", "x", "b", "x", "c"], 5),
        (vec!["a", "x", "b", "x"], 4),
    ] {
        let hits = index.find_matches(&query(0, &hashes, &[StorageTier::Device]));
        assert_eq!(hits.len(), 1, "query {hashes:?}");
        assert_eq!(prefix_depth(&hits[0]), expected_depth, "query {hashes:?}");
    }
}

#[test]
fn a_real_alternate_branch_remains_matchable_by_its_owner() {
    let index = compressed_prefix_with_child();
    let alternate = worker("alternate", 0);
    // Shared x has the same engine identity; the diverging z has its own identity.
    store(
        &index,
        0,
        &alternate,
        StorageTier::Device,
        None,
        &[("s1", "x"), ("sxz", "z")],
    );
    for (hashes, expected_worker, expected_alternate) in
        [(vec!["x", "z"], 1, 2), (vec!["x", "y", "z"], 3, 1)]
    {
        let hits = index.find_matches(&query(0, &hashes, &[StorageTier::Device]));
        assert_eq!(hits.len(), 2, "query {hashes:?}");
        for (source, expected_depth) in [
            ("worker", expected_worker),
            ("alternate", expected_alternate),
        ] {
            let hit = hits
                .iter()
                .find(|hit| hit.target.instance_id.as_ref() == source)
                .expect("each owner holds x");
            assert_eq!(prefix_depth(hit), expected_depth, "{source}: {hashes:?}");
        }
    }
}

#[test]
fn target_rank_tier_and_group_claims_do_not_fill_each_others_gaps() {
    let index = compressed_prefix_with_child();
    let original = worker("worker", 0);
    let other_worker = worker("other", 0);
    let other_rank = worker("worker", 1);
    for (owner, tier) in [
        (&other_worker, StorageTier::Device),
        (&other_rank, StorageTier::Device),
        (&original, StorageTier::HostPinned),
    ] {
        store(&index, 0, owner, tier, None, &[("s1", "x")]);
        // This owner/tier has x and z, but no claim on the intervening y.
        store(&index, 0, owner, tier, Some("s2"), &[("s3", "z")]);
    }
    store(
        &index,
        1,
        &other_worker,
        StorageTier::Device,
        None,
        &[("s1", "x"), ("s2", "y"), ("s3", "z")],
    );
    let hits = index.find_matches(&query(
        0,
        &["x", "y", "z"],
        &[StorageTier::Device, StorageTier::HostPinned],
    ));
    assert_eq!(hits.len(), 4);
    for (owner, tier, expected_segments) in [
        (&original, StorageTier::Device, vec![(0, 3)]),
        (&other_worker, StorageTier::Device, vec![(0, 1), (2, 3)]),
        (&other_rank, StorageTier::Device, vec![(0, 1), (2, 3)]),
        (&original, StorageTier::HostPinned, vec![(0, 1), (2, 3)]),
    ] {
        let hit = hits
            .iter()
            .find(|hit| hit.target == owner.target() && hit.tier == tier)
            .expect("each target/tier holds x");
        assert_eq!(hit.segments, expected_segments);
    }
    let host_hits = index.find_matches(&query(0, &["x", "y", "z"], &[StorageTier::HostPinned]));
    assert_eq!(host_hits.len(), 1);
    assert_eq!(host_hits[0].tier, StorageTier::HostPinned);
    assert_eq!(host_hits[0].segments, vec![(0, 1), (2, 3)]);
    let group_hits = index.find_matches(&query(1, &["x", "y", "z"], &[StorageTier::Device]));
    assert_eq!(group_hits.len(), 1);
    assert_eq!(group_hits[0].target, other_worker.target());
    assert_eq!(group_hits[0].segments, vec![(0, 3)]);
    assert!(index
        .find_matches(&query(2, &["x", "y", "z"], &[StorageTier::Device]))
        .is_empty());
}

#[test]
fn equivalent_long_short_and_mixed_edges_have_the_same_prefix_semantics() {
    let owner = worker("worker", 0);
    let path = [
        ("sa", "a"),
        ("sb", "b"),
        ("sc", "c"),
        ("sd", "d"),
        ("se", "e"),
        ("sf", "f"),
    ];
    let indexes: Vec<_> = [vec![4, 2], vec![1, 1, 1, 1, 1, 1], vec![2, 2, 2]]
        .into_iter()
        .map(|batch_lengths| {
            let index = KvBlockIndexer::new();
            let mut offset: usize = 0;
            for batch_length in batch_lengths {
                let parent = offset.checked_sub(1).map(|pos| path[pos].0);
                store(
                    &index,
                    0,
                    &owner,
                    StorageTier::Device,
                    parent,
                    &path[offset..offset + batch_length],
                );
                offset += batch_length;
            }
            store(
                &index,
                0,
                &owner,
                StorageTier::Device,
                None,
                &[("sa", "a"), ("sb", "b"), ("sq", "q")],
            );
            index
        })
        .collect();
    for (hashes, expected_depth) in [
        (vec![], 0),
        (vec!["missing", "a"], 0),
        (vec!["a"], 1),
        (vec!["a", "b"], 2),
        (vec!["a", "q"], 1),
        (vec!["a", "b", "q"], 3),
        (vec!["a", "b", "c", "e", "f"], 3),
        (vec!["a", "b", "c", "d", "e"], 5),
        (vec!["a", "b", "c", "d", "f"], 4),
        (vec!["a", "b", "c", "d", "e", "f"], 6),
        (vec!["a", "b", "c", "d", "e", "f", "missing"], 6),
    ] {
        for (shape, index) in indexes.iter().enumerate() {
            let hits = index.find_matches(&query(0, &hashes, &[StorageTier::Device]));
            if expected_depth == 0 {
                assert!(hits.is_empty(), "shape {shape}, query {hashes:?}");
            } else {
                assert_eq!(hits.len(), 1, "shape {shape}, query {hashes:?}");
                assert_eq!(
                    prefix_depth(&hits[0]),
                    expected_depth,
                    "shape {shape}, query {hashes:?}"
                );
            }
        }
    }
}
