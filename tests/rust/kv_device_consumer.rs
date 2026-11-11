#![cfg(all(feature = "cpu", feature = "ready"))]

//! Synthetic event consumer using the shared index and read contract.
use std::sync::Arc;
use vllm_router_rs::{
    kv_hash::BlockKeyGenerator,
    kv_index::{
        decode_batch, local_hashes, validate_device_batch,
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
    validate_device_batch(batch, 2).unwrap();
    let mut owner = owner.clone();
    let ResidencyOwner::Worker { dp_rank, .. } = &mut owner else {
        panic!("worker source required")
    };
    *dp_rank = batch.data_parallel_rank.unwrap_or(*dp_rank);
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
                &owner,
                StorageTier::Device,
                &event.block_hashes.iter().map(opaque).collect::<Vec<_>>(),
            ),
            KVEvent::AllBlocksCleared(_) => index.clear(&owner, ClearScope::Worker),
        }
    }
}

fn stored_event() -> rmpv::Value {
    rmpv::Value::Map(vec![
        ("type".into(), "BlockStored".into()),
        (
            "block_hashes".into(),
            rmpv::Value::Array(vec![
                rmpv::Value::Binary(vec![1; 32]),
                rmpv::Value::Binary(vec![2; 32]),
            ]),
        ),
        ("parent_block_hash".into(), rmpv::Value::Nil),
        (
            "token_ids".into(),
            rmpv::Value::Array(vec![1.into(), 2.into(), 3.into(), 4.into()]),
        ),
        ("block_size".into(), 2.into()),
        ("lora_id".into(), rmpv::Value::Nil),
        ("lora_name".into(), rmpv::Value::Nil),
        ("medium".into(), "GPU".into()),
    ])
}

fn removed_event() -> rmpv::Value {
    rmpv::Value::Map(vec![
        ("type".into(), "BlockRemoved".into()),
        (
            "block_hashes".into(),
            rmpv::Value::Array(vec![rmpv::Value::Binary(vec![2; 32])]),
        ),
        ("medium".into(), "GPU".into()),
    ])
}

fn cleared_event() -> rmpv::Value {
    rmpv::Value::Map(vec![("type".into(), "AllBlocksCleared".into())])
}

fn decoded_event(event: rmpv::Value, rank: Option<rmpv::Value>) -> KVEventBatch {
    let mut batch = vec![1.0.into(), rmpv::Value::Array(vec![event])];
    if let Some(rank) = rank {
        batch.push(rank);
    }
    let mut bytes = Vec::new();
    rmpv::encode::write_value(&mut bytes, &rmpv::Value::Array(batch)).unwrap();
    decode_batch(&bytes).unwrap()
}

#[test]
fn stored_shape_admits_absent_zero_and_nonzero_rank() {
    for rank in [None, Some(0_u32), Some(7)] {
        let batch = decoded_event(stored_event(), rank.map(Into::into));
        assert_eq!(batch.data_parallel_rank, rank);
        assert!(validate_device_batch(&batch, 2).is_ok(), "rank {rank:?}");
    }
}

#[test]
fn removed_shape_admits_absent_zero_and_nonzero_rank() {
    for rank in [None, Some(0_u32), Some(7)] {
        let batch = decoded_event(removed_event(), rank.map(Into::into));
        assert_eq!(batch.data_parallel_rank, rank);
        assert!(validate_device_batch(&batch, 2).is_ok(), "rank {rank:?}");
    }
}

#[test]
fn cleared_shape_admits_absent_zero_and_nonzero_rank() {
    for rank in [None, Some(0_u32), Some(7)] {
        let batch = decoded_event(cleared_event(), rank.map(Into::into));
        assert_eq!(batch.data_parallel_rank, rank);
        assert!(validate_device_batch(&batch, 2).is_ok(), "rank {rank:?}");
    }
}

#[test]
fn same_source_rank_remove_and_clear_are_isolated() {
    let index = KvBlockIndexer::new();
    let configured = ResidencyOwner::Worker {
        source: Arc::from("publisher-A"),
        dp_rank: 0,
        incarnation: 9,
    };
    let segments = || {
        let mut hits: Vec<_> = index
            .find_matches(&MatchQuery {
                group_idx: 0,
                local_hashes: local_keys(&[1, 2, 3, 4]),
                tiers_of_interest: vec![StorageTier::Device],
            })
            .into_iter()
            .map(|hit| {
                assert_eq!(hit.target.instance_id.as_ref(), "publisher-A");
                (hit.target.dp_rank, hit.segments)
            })
            .collect();
        hits.sort_unstable();
        hits
    };
    // Both ranks announce the same engine hashes for the same local path;
    // this regression does not depend on different-hash alias support.
    for rank in [0_u32, 7] {
        apply(
            &index,
            &configured,
            &decoded_event(stored_event(), Some(rank.into())),
        );
    }
    assert_eq!(segments(), vec![(0, vec![(0, 2)]), (7, vec![(0, 2)])]);
    apply(
        &index,
        &configured,
        &decoded_event(removed_event(), Some(0.into())),
    );
    assert_eq!(segments(), vec![(0, vec![(0, 1)]), (7, vec![(0, 2)])]);
    apply(
        &index,
        &configured,
        &decoded_event(stored_event(), Some(0.into())),
    );
    apply(
        &index,
        &configured,
        &decoded_event(removed_event(), Some(7.into())),
    );
    assert_eq!(segments(), vec![(0, vec![(0, 2)]), (7, vec![(0, 1)])]);
    apply(
        &index,
        &configured,
        &decoded_event(cleared_event(), Some(7.into())),
    );
    assert_eq!(segments(), vec![(0, vec![(0, 2)])]);
    apply(
        &index,
        &configured,
        &decoded_event(stored_event(), Some(7.into())),
    );
    apply(
        &index,
        &configured,
        &decoded_event(cleared_event(), Some(0.into())),
    );
    assert_eq!(segments(), vec![(7, vec![(0, 2)])]);
}

#[test]
fn absent_rank_uses_configured_source_and_explicit_zero_overrides_it() {
    for wire_rank in [None, Some(rmpv::Value::Nil), Some(0.into())] {
        let index = KvBlockIndexer::new();
        let configured = ResidencyOwner::Worker {
            source: Arc::from("publisher-A"),
            dp_rank: 7,
            incarnation: 9,
        };
        let expected_rank = if wire_rank == Some(0.into()) {
            Some(0)
        } else {
            None
        };
        let batch = decoded_event(stored_event(), wire_rank.clone());
        assert_eq!(batch.data_parallel_rank, expected_rank);
        apply(&index, &configured, &batch);
        let query = MatchQuery {
            group_idx: 0,
            local_hashes: local_keys(&[1, 2, 3, 4]),
            tiers_of_interest: vec![StorageTier::Device],
        };
        let hits = index.find_matches(&query);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].target.dp_rank, expected_rank.unwrap_or(7));
        assert_eq!(hits[0].segments, vec![(0, 2)]);
        apply(
            &index,
            &configured,
            &decoded_event(removed_event(), wire_rank.clone()),
        );
        let hits = index.find_matches(&query);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].target.dp_rank, expected_rank.unwrap_or(7));
        assert_eq!(hits[0].segments, vec![(0, 1)]);
        apply(
            &index,
            &configured,
            &decoded_event(cleared_event(), wire_rank),
        );
        assert!(index.find_matches(&query).is_empty());
    }
}

#[test]
fn extra_keys_decode_losslessly_and_admit_only_supported_shapes() {
    for (wire, expected, admitted) in [
        (None, None, true),
        (Some(rmpv::Value::Nil), None, true),
        (
            Some(rmpv::Value::Array(vec![rmpv::Value::Nil; 2])),
            Some(vec![rmpv::Value::Nil; 2]),
            true,
        ),
        (
            Some(rmpv::Value::Array(vec![rmpv::Value::Nil])),
            Some(vec![rmpv::Value::Nil]),
            false,
        ),
        (
            Some(rmpv::Value::Array(vec![
                rmpv::Value::Nil,
                rmpv::Value::Binary(vec![0xc1, 0xff]),
            ])),
            Some(vec![
                rmpv::Value::Nil,
                rmpv::Value::Binary(vec![0xc1, 0xff]),
            ]),
            false,
        ),
    ] {
        let rmpv::Value::Map(mut event) = stored_event() else {
            unreachable!()
        };
        if let Some(wire) = wire {
            event.push(("extra_keys".into(), wire));
        }
        let batch = decoded_event(rmpv::Value::Map(event), None);
        let KVEvent::BlockStored(stored) = &batch.events[0] else {
            panic!("Store required")
        };
        assert_eq!(stored.extra_keys, expected);
        assert_eq!(
            validate_device_batch(&batch, 2).is_ok(),
            admitted,
            "{expected:?}"
        );
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
    assert_eq!(hits[0].segments, vec![(0, 2)]); // Observation blocks, never reusable tokens.
    assert_eq!(hits[0].target.instance_id.as_ref(), "publisher-A");
    // A routing integration must bind this source to its actual target.
    let local = local_keys(&tokens);
    assert_ne!(
        local[0],
        opaque(&ExternalBlockHash::Bytes(sequences[0].to_vec()))
    );
    apply(&index, &owner, &batch); // Idempotent announcement.
    assert_eq!(
        provider.find_tiered_matches(&query)[0].segments,
        vec![(0, 2)]
    );
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
                .segments,
            vec![(0, 2)]
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
    for (worker, end) in [("publisher-A", 1), ("publisher-B", 2)] {
        assert_eq!(
            hits.iter()
                .find(|hit| hit.target.instance_id.as_ref() == worker)
                .unwrap()
                .segments,
            vec![(0, end)]
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
    assert_eq!(hits[0].segments, vec![(0, 2)]);
}
