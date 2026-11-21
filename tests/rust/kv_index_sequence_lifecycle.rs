#![cfg(all(feature = "cpu", feature = "ready"))]

use std::sync::Arc;

use vllm_router_rs::kv_hash::BlockKeyGenerator;
use vllm_router_rs::kv_index::{
    ClearScope, KvBlockIndexer, MatchQuery, ResidencyOwner, StorageTier,
};

fn worker(rank: u32) -> ResidencyOwner {
    ResidencyOwner::Worker {
        source: Arc::from("publisher"),
        dp_rank: rank,
        incarnation: 0,
    }
}

#[test]
fn eviction_preserves_other_ranks_and_later_pages() {
    let index = KvBlockIndexer::new();
    let hashes =
        BlockKeyGenerator::from_seed(2, "vllm-none-hash").generate_block_keys(&[1, 2, 3, 4, 5, 6]);
    let query = MatchQuery {
        group_idx: 0,
        hashes: hashes.clone(),
        tiers_of_interest: vec![StorageTier::Device],
    };
    for rank in [0, 1] {
        index.store(0, worker(rank), StorageTier::Device, &hashes);
    }

    index.remove(0, &worker(0), StorageTier::Device, &hashes[1..2]);
    let mut hits = index.find_matches(&query);
    hits.sort_by_key(|hit| hit.target.dp_rank);
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].segments, [(0, 1), (2, 3)]);
    assert_eq!(hits[1].segments, [(0, 3)]);

    index.store(0, worker(0), StorageTier::Device, &hashes[1..2]);
    index.clear(&worker(1), ClearScope::Worker);
    let hits = index.find_matches(&query);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].target.dp_rank, 0);
    assert_eq!(hits[0].segments, [(0, 3)]);
}

#[test]
fn clearing_a_worker_does_not_remove_a_shared_pool_claim() {
    let index = KvBlockIndexer::new();
    let pool = ResidencyOwner::CacheOwner {
        pool_id: Arc::from("pool"),
        source: Arc::from("publisher"),
        dp_rank: 0,
    };
    let hashes = vec![[7; 32]];
    let query = MatchQuery {
        group_idx: 0,
        hashes: hashes.clone(),
        tiers_of_interest: vec![StorageTier::Device, StorageTier::External],
    };
    index.store(0, worker(0), StorageTier::Device, &hashes);
    index.store(0, pool.clone(), StorageTier::External, &hashes);
    index.clear(&worker(0), ClearScope::Worker);
    let hits = index.find_matches(&query);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].tier, StorageTier::External);
    index.clear(&pool, ClearScope::CacheOwner);
    assert!(index.find_matches(&query).is_empty());
}
