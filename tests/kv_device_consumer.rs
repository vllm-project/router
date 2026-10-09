//! Synthetic event consumer using the shared index and read contract.
use std::sync::Arc;
use vllm_router_rs::{
    kv_events::{decode_batch, validate_device_dp1_batch},
    kv_hash::BlockKeyGenerator,
    kv_index::{
        local_hashes,
        wire::{ExternalBlockHash, KVEvent, KVEventBatch},
        ClearScope, KvBlockIndexer, MatchQuery, ResidencyOwner, StorageTier, TieredMatchProvider,
    },
};

fn opaque(hash: &ExternalBlockHash) -> Arc<str> {
    let ExternalBlockHash::Bytes(bytes) = hash else {
        panic!("admitted bytes required")
    };
    Arc::from(format!(
        "b:{}",
        bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ))
}

fn local_keys(tokens: &[u32]) -> Vec<Arc<str>> {
    local_hashes(tokens, 2, None)
        .iter()
        .map(|hash| {
            Arc::from(
                hash.iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>(),
            )
        })
        .collect()
}

fn apply(index: &KvBlockIndexer, owner: &ResidencyOwner, batch: &KVEventBatch) {
    // Runtime ingestion must also fence publication; this consumer is single-threaded.
    validate_device_dp1_batch(batch, 2).unwrap();
    for event in &batch.events {
        match event {
            KVEvent::BlockStored(event) => {
                let locals = local_keys(&event.token_ids);
                let blocks: Vec<_> = event.block_hashes.iter().map(opaque).zip(locals).collect();
                index.store(
                    0,
                    owner.clone(),
                    StorageTier::Device,
                    event.parent_block_hash.as_ref().map(opaque),
                    &blocks,
                );
            }
            KVEvent::BlockRemoved(event) => index.remove(
                0,
                owner,
                StorageTier::Device,
                &event.block_hashes.iter().map(opaque).collect::<Vec<_>>(),
            ),
            KVEvent::AllBlocksCleared(_) => index.clear(owner, ClearScope::Worker),
        }
    }
}

#[test]
fn decode_admit_store_query_remove_and_clear_use_the_common_contract() {
    let tokens = [1_u32, 2, 3, 4];
    let sequences = BlockKeyGenerator::new(2, 0).generate_block_keys(&tokens);
    let payload = rmpv::Value::Array(vec![
        1.0.into(),
        rmpv::Value::Array(vec![rmpv::Value::Map(vec![
            ("type".into(), "BlockStored".into()),
            (
                "block_hashes".into(),
                rmpv::Value::Array(
                    sequences
                        .iter()
                        .map(|hash| rmpv::Value::Binary(hash.to_vec()))
                        .collect(),
                ),
            ),
            ("parent_block_hash".into(), rmpv::Value::Nil),
            (
                "token_ids".into(),
                rmpv::Value::Array(tokens.iter().map(|token| (*token).into()).collect()),
            ),
            ("block_size".into(), 2.into()),
            ("lora_id".into(), rmpv::Value::Nil),
            ("lora_name".into(), rmpv::Value::Nil),
            ("medium".into(), "GPU".into()),
        ])]),
        0.into(),
    ]);
    let mut bytes = Vec::new();
    rmpv::encode::write_value(&mut bytes, &payload).unwrap();
    let batch = decode_batch(&bytes).unwrap();
    let owner = ResidencyOwner::Worker {
        source: Arc::from("publisher-A"),
        dp_rank: 0,
        incarnation: 9,
    };
    let index = KvBlockIndexer::new();
    let query = MatchQuery {
        group_idx: 0,
        local_hashes: local_keys(&tokens),
        tiers_of_interest: vec![StorageTier::Device],
    };
    let provider: &dyn TieredMatchProvider = &index;
    assert!(provider.find_tiered_matches(&query).is_empty());
    apply(&index, &owner, &batch);
    let hits = provider.find_tiered_matches(&query);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].matched_depth, 2); // Observation blocks, never reusable tokens.
    assert_eq!(hits[0].target.instance_id.as_ref(), "publisher-A");
    // A routing integration must bind this source to its actual target.
    let local = local_keys(&tokens);
    assert_ne!(
        local[0],
        opaque(&ExternalBlockHash::Bytes(sequences[0].to_vec()))
    );
    apply(&index, &owner, &batch); // Idempotent announcement.
    assert_eq!(provider.find_tiered_matches(&query)[0].matched_depth, 2);
    let other_owner = ResidencyOwner::Worker {
        source: Arc::from("publisher-B"),
        dp_rank: 0,
        incarnation: 9,
    };
    apply(&index, &other_owner, &batch);
    let hits = provider.find_tiered_matches(&query);
    assert_eq!(hits.len(), 2);
    for worker in ["publisher-A", "publisher-B"] {
        assert_eq!(
            hits.iter()
                .find(|hit| hit.target.instance_id.as_ref() == worker)
                .unwrap()
                .matched_depth,
            2
        );
    }

    let removed = rmpv::Value::Array(vec![
        2.0.into(),
        rmpv::Value::Array(vec![rmpv::Value::Map(vec![
            ("type".into(), "BlockRemoved".into()),
            ("medium".into(), "GPU".into()),
            (
                "block_hashes".into(),
                rmpv::Value::Array(vec![rmpv::Value::Binary(sequences[1].to_vec())]),
            ),
        ])]),
        0.into(),
    ]);
    bytes.clear();
    rmpv::encode::write_value(&mut bytes, &removed).unwrap();
    let removed = decode_batch(&bytes).unwrap();
    assert!(matches!(&removed.events[0], KVEvent::BlockRemoved(_)));
    apply(&index, &owner, &removed);
    let hits = provider.find_tiered_matches(&query);
    assert_eq!(hits.len(), 2);
    for (worker, depth) in [("publisher-A", 1), ("publisher-B", 2)] {
        assert_eq!(
            hits.iter()
                .find(|hit| hit.target.instance_id.as_ref() == worker)
                .unwrap()
                .matched_depth,
            depth
        );
    }

    let cleared = rmpv::Value::Array(vec![
        3.0.into(),
        rmpv::Value::Array(vec![rmpv::Value::Map(vec![(
            "type".into(),
            "AllBlocksCleared".into(),
        )])]),
        0.into(),
    ]);
    bytes.clear();
    rmpv::encode::write_value(&mut bytes, &cleared).unwrap();
    let cleared = decode_batch(&bytes).unwrap();
    assert!(matches!(&cleared.events[0], KVEvent::AllBlocksCleared(_)));
    apply(&index, &owner, &cleared);
    let hits = provider.find_tiered_matches(&query);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].target.instance_id.as_ref(), "publisher-B");
    assert_eq!(hits[0].matched_depth, 2);
}
