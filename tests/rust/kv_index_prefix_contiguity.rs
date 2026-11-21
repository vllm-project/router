#![cfg(all(feature = "cpu", feature = "ready"))]

use std::sync::Arc;

use vllm_router_rs::kv_hash::{BlockKeyGenerator, SequenceHash};
use vllm_router_rs::kv_index::{KvBlockIndexer, MatchQuery, ResidencyOwner, StorageTier};

fn keys(tokens: &[u32]) -> Vec<SequenceHash> {
    BlockKeyGenerator::from_seed(2, "vllm-none-hash").generate_block_keys(tokens)
}

fn worker(name: &str) -> ResidencyOwner {
    ResidencyOwner::Worker {
        source: Arc::from(name),
        dp_rank: 0,
        incarnation: 0,
    }
}

fn query(hashes: Vec<SequenceHash>) -> MatchQuery {
    MatchQuery {
        group_idx: 0,
        hashes,
        tiers_of_interest: vec![StorageTier::Device],
    }
}

#[test]
fn matching_token_blocks_do_not_fill_gaps_in_other_branches() {
    let index = KvBlockIndexer::new();
    let left = keys(&[11, 12, 21, 22, 41, 42]);
    let right = keys(&[11, 12, 31, 32, 41, 42]);
    assert_eq!(left[0], right[0]);
    assert_ne!(left[2], right[2]);
    index.store(0, worker("left"), StorageTier::Device, &left);
    index.store(0, worker("right"), StorageTier::Device, &right);

    let mut hits = index.find_matches(&query(right));
    hits.sort_by(|a, b| a.target.instance_id.cmp(&b.target.instance_id));
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].segments, [(0, 1)]);
    assert_eq!(hits[1].segments, [(0, 3)]);
}

#[test]
fn repeated_blocks_keep_distinct_positions_and_batchings_agree() {
    let input = [1, 2, 1, 2, 1, 2, 3];
    let hashes = keys(&input);
    assert_eq!(hashes.len(), 3);
    assert_ne!(hashes[0], hashes[1]);
    for batch_size in 1..=hashes.len() {
        let index = KvBlockIndexer::new();
        for batch in hashes.chunks(batch_size) {
            index.store(0, worker("same"), StorageTier::Device, batch);
        }
        let hits = index.find_matches(&query(keys(&input)));
        assert_eq!(hits[0].segments, [(0, 3)]);
    }
}
