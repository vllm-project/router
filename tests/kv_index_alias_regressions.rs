use std::sync::Arc;

use vllm_router_rs::kv_index::{
    ClearScope, KvBlockIndexer, MatchQuery, ResidencyOwner, StorageTier,
};

#[test]
fn second_owner_sequence_alias_is_removable() {
    let index = KvBlockIndexer::new();
    let owner_a = ResidencyOwner::Worker {
        source: Arc::from("worker-A"),
        dp_rank: 3,
        incarnation: 0,
    };
    let owner_b = ResidencyOwner::Worker {
        source: Arc::from("worker-B"),
        dp_rank: 3,
        incarnation: 0,
    };

    // Distinct integer engine hashes share a token-local path, not a hash collision.
    index.store(
        0,
        owner_a,
        StorageTier::Device,
        None,
        &[(Arc::from("i:900"), Arc::from("local"))],
    );
    index.store(
        0,
        owner_b.clone(),
        StorageTier::Device,
        None,
        &[(Arc::from("i:42"), Arc::from("local"))],
    );
    let query = MatchQuery {
        group_idx: 0,
        local_hashes: vec![Arc::from("local")],
        tiers_of_interest: vec![StorageTier::Device],
    };
    assert_eq!(index.find_matches(&query).len(), 2);

    index.remove(0, &owner_b, StorageTier::Device, &[Arc::from("i:42")]);
    let hits = index.find_matches(&query);
    assert_eq!(
        hits.len(),
        1,
        "Remove must resolve the incoming sequence alias"
    );
    assert_eq!(hits[0].target.instance_id.as_ref(), "worker-A");
}

fn worker(source: &str, dp_rank: u32) -> ResidencyOwner {
    ResidencyOwner::Worker {
        source: Arc::from(source),
        dp_rank,
        incarnation: 0,
    }
}

fn pool(pool_id: &str, source: &str) -> ResidencyOwner {
    ResidencyOwner::CacheOwner {
        pool_id: Arc::from(pool_id),
        source: Arc::from(source),
        dp_rank: 3,
    }
}

fn blocks(pairs: &[(&str, &str)]) -> Vec<(Arc<str>, Arc<str>)> {
    pairs
        .iter()
        .map(|(seq, local)| (Arc::from(*seq), Arc::from(*local)))
        .collect()
}

fn depth(
    index: &KvBlockIndexer,
    group_idx: u32,
    owner: &ResidencyOwner,
    tier: StorageTier,
    path: &[&str],
) -> u32 {
    index
        .find_matches(&MatchQuery {
            group_idx,
            local_hashes: path.iter().map(|local| Arc::from(*local)).collect(),
            tiers_of_interest: vec![tier],
        })
        .iter()
        .find(|hit| hit.target == owner.target())
        .map_or(0, |hit| hit.matched_depth)
}

#[test]
fn distinct_aliases_can_be_removed_in_either_owner_order() {
    for first_is_a in [false, true] {
        let index = KvBlockIndexer::new();
        let a = worker("a", 3);
        let b = worker("b", 3);
        index.store(
            0,
            a.clone(),
            StorageTier::Device,
            None,
            &blocks(&[("i:900", "local")]),
        );
        index.store(
            0,
            b.clone(),
            StorageTier::Device,
            None,
            &blocks(&[("i:42", "local")]),
        );
        // A globally known alias is still a no-op for an owner that never stored it.
        index.remove(
            0,
            &a,
            StorageTier::Device,
            &[Arc::from("i:42"), Arc::from("i:404")],
        );
        assert_eq!(depth(&index, 0, &a, StorageTier::Device, &["local"]), 1);
        assert_eq!(depth(&index, 0, &b, StorageTier::Device, &["local"]), 1);
        let (first, first_seq, second, second_seq) = if first_is_a {
            (&a, "i:900", &b, "i:42")
        } else {
            (&b, "i:42", &a, "i:900")
        };
        index.remove(0, first, StorageTier::Device, &[Arc::from(first_seq)]);
        assert_eq!(depth(&index, 0, first, StorageTier::Device, &["local"]), 0);
        assert_eq!(depth(&index, 0, second, StorageTier::Device, &["local"]), 1);
        index.remove(0, second, StorageTier::Device, &[Arc::from(second_seq)]);
        assert_eq!(depth(&index, 0, second, StorageTier::Device, &["local"]), 0);
    }
}

#[test]
fn shared_engine_sequence_keeps_other_owner_claim() {
    let index = KvBlockIndexer::new();
    let a = worker("a", 3);
    let b = worker("b", 3);
    let path = blocks(&[("i:900", "local")]);
    for owner in [&a, &b] {
        index.store(0, owner.clone(), StorageTier::Device, None, &path);
    }
    index.remove(0, &a, StorageTier::Device, &[Arc::from("i:900")]);
    assert_eq!(depth(&index, 0, &a, StorageTier::Device, &["local"]), 0);
    assert_eq!(depth(&index, 0, &b, StorageTier::Device, &["local"]), 1);
    index.remove(0, &b, StorageTier::Device, &[Arc::from("i:900")]);
    assert_eq!(depth(&index, 0, &b, StorageTier::Device, &["local"]), 0);
}

#[test]
fn interior_alias_remove_and_restore_preserve_other_owner_and_suffix() {
    let index = KvBlockIndexer::new();
    let a = worker("a", 3);
    let b = worker("b", 3);
    index.store(
        0,
        a.clone(),
        StorageTier::Device,
        None,
        &blocks(&[("i:100", "x"), ("i:101", "y"), ("i:102", "z")]),
    );
    index.store(
        0,
        b.clone(),
        StorageTier::Device,
        None,
        &blocks(&[("i:200", "x"), ("i:201", "y"), ("i:202", "z")]),
    );
    index.remove(0, &b, StorageTier::Device, &[Arc::from("i:201")]);
    assert_eq!(
        depth(&index, 0, &a, StorageTier::Device, &["x", "y", "z"]),
        3
    );
    assert_eq!(
        depth(&index, 0, &b, StorageTier::Device, &["x", "y", "z"]),
        1
    );
    index.store(
        0,
        b.clone(),
        StorageTier::Device,
        Some(Arc::from("i:200")),
        &blocks(&[("i:201", "y")]),
    );
    assert_eq!(
        depth(&index, 0, &b, StorageTier::Device, &["x", "y", "z"]),
        3
    );
    index.remove(0, &b, StorageTier::Device, &[Arc::from("i:202")]);
    assert_eq!(
        depth(&index, 0, &b, StorageTier::Device, &["x", "y", "z"]),
        2
    );
}

#[test]
fn same_owner_claim_survives_until_last_alias_is_removed() {
    let index = KvBlockIndexer::new();
    let owner = worker("a", 3);
    for seq in ["i:900", "i:42", "b:0900"] {
        index.store(
            0,
            owner.clone(),
            StorageTier::Device,
            None,
            &blocks(&[(seq, "local")]),
        );
    }
    for (seq, expected) in [("i:42", 1), ("i:900", 1), ("b:0900", 0)] {
        index.remove(0, &owner, StorageTier::Device, &[Arc::from(seq)]);
        assert_eq!(
            depth(&index, 0, &owner, StorageTier::Device, &["local"]),
            expected
        );
    }
    for _ in 0..2 {
        index.store(
            0,
            owner.clone(),
            StorageTier::Device,
            None,
            &blocks(&[("i:42", "local")]),
        );
    }
    assert_eq!(depth(&index, 0, &owner, StorageTier::Device, &["local"]), 1);
    index.remove(
        0,
        &owner,
        StorageTier::Device,
        &[Arc::from("i:42"), Arc::from("i:42")],
    );
    assert_eq!(depth(&index, 0, &owner, StorageTier::Device, &["local"]), 0);
}

#[test]
fn historical_alias_parents_follow_splits_and_interior_continuations() {
    let index = KvBlockIndexer::new();
    let a = worker("a", 3);
    let b = worker("b", 3);
    index.store(
        0,
        a.clone(),
        StorageTier::Device,
        None,
        &blocks(&[("i:100", "x"), ("i:101", "y"), ("i:102", "z")]),
    );
    index.store(
        0,
        b.clone(),
        StorageTier::Device,
        None,
        &blocks(&[("i:200", "x"), ("i:201", "y"), ("i:202", "z")]),
    );
    // The parent is inside a compressed edge, not its last block.
    index.store(
        0,
        b.clone(),
        StorageTier::Device,
        Some(Arc::from("i:201")),
        &blocks(&[("i:203", "branch")]),
    );
    assert_eq!(
        depth(&index, 0, &b, StorageTier::Device, &["x", "y", "branch"]),
        3
    );
    assert_eq!(
        depth(&index, 0, &a, StorageTier::Device, &["x", "y", "z"]),
        3
    );
    index.remove(0, &b, StorageTier::Device, &[Arc::from("i:202")]);
    // Removed aliases still locate topology after a split.
    index.store(
        0,
        a.clone(),
        StorageTier::Device,
        Some(Arc::from("i:202")),
        &blocks(&[("i:103", "tail")]),
    );
    assert_eq!(
        depth(&index, 0, &a, StorageTier::Device, &["x", "y", "z", "tail"]),
        4
    );
    index.remove(0, &b, StorageTier::Device, &[Arc::from("i:201")]);
    assert_eq!(
        depth(&index, 0, &b, StorageTier::Device, &["x", "y", "branch"]),
        1
    );
    index.store(
        0,
        b.clone(),
        StorageTier::Device,
        Some(Arc::from("i:200")),
        &blocks(&[("i:201", "y")]),
    );
    assert_eq!(
        depth(&index, 0, &b, StorageTier::Device, &["x", "y", "branch"]),
        3
    );
}

#[test]
fn alias_remove_and_clear_preserve_rank_group_tier_and_owner_domains() {
    let index = KvBlockIndexer::new();
    let a = worker("a", 3);
    let other_rank = worker("a", 4);
    let cache = pool("pool", "a");
    let other_cache = pool("other-pool", "b");
    for group in [0, 1] {
        for (owner, tier, seq) in [
            (&a, StorageTier::Device, "i:100"),
            (&a, StorageTier::HostPinned, "i:101"),
            (&other_rank, StorageTier::Device, "i:102"),
            (&cache, StorageTier::External, "b:0100"),
            (&other_cache, StorageTier::External, "b:0101"),
        ] {
            index.store(group, owner.clone(), tier, None, &blocks(&[(seq, "local")]));
        }
    }
    index.remove(0, &a, StorageTier::Device, &[Arc::from("i:100")]);
    assert_eq!(depth(&index, 0, &a, StorageTier::Device, &["local"]), 0);
    assert_eq!(depth(&index, 1, &a, StorageTier::Device, &["local"]), 1);
    assert_eq!(depth(&index, 0, &a, StorageTier::HostPinned, &["local"]), 1);
    index.clear(&a, ClearScope::Worker);
    for group in [0, 1] {
        assert_eq!(depth(&index, group, &a, StorageTier::Device, &["local"]), 0);
        assert_eq!(
            depth(&index, group, &a, StorageTier::HostPinned, &["local"]),
            0
        );
        assert_eq!(
            depth(&index, group, &other_rank, StorageTier::Device, &["local"]),
            1
        );
        assert_eq!(
            depth(&index, group, &cache, StorageTier::External, &["local"]),
            1
        );
    }
    index.clear(&cache, ClearScope::CacheOwner);
    for group in [0, 1] {
        assert_eq!(
            depth(&index, group, &cache, StorageTier::External, &["local"]),
            0
        );
        assert_eq!(
            depth(
                &index,
                group,
                &other_cache,
                StorageTier::External,
                &["local"]
            ),
            1
        );
        assert_eq!(
            depth(&index, group, &other_rank, StorageTier::Device, &["local"]),
            1
        );
    }
}

#[test]
fn worker_parent_alias_can_extend_cache_owner_path() {
    let index = KvBlockIndexer::new();
    let owner = worker("a", 3);
    let cache = pool("pool", "a");
    index.store(
        0,
        owner,
        StorageTier::Device,
        None,
        &blocks(&[("i:100", "x"), ("i:101", "y")]),
    );
    index.store(
        0,
        cache.clone(),
        StorageTier::External,
        None,
        &blocks(&[("b:0100", "x")]),
    );
    index.store(
        0,
        cache.clone(),
        StorageTier::External,
        Some(Arc::from("i:100")),
        &blocks(&[("b:0101", "branch")]),
    );
    assert_eq!(
        depth(&index, 0, &cache, StorageTier::External, &["x", "branch"]),
        2
    );
}

#[test]
fn unknown_initial_parent_keeps_root_fallback() {
    let index = KvBlockIndexer::new();
    let owner = worker("a", 3);
    index.store(
        0,
        owner.clone(),
        StorageTier::Device,
        Some(Arc::from("i:999")),
        &blocks(&[("i:100", "x")]),
    );
    assert_eq!(depth(&index, 0, &owner, StorageTier::Device, &["x"]), 1);
}
