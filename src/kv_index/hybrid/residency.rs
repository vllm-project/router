//! What this rank currently holds: the claim set per cache identity.

use std::collections::{HashMap, HashSet};

use crate::kv_hash::SequenceHash;

pub(super) struct State {
    /// The next sequence in a continuous, source-bound event stream.
    pub(super) next_seq: Option<u64>,
    /// Partial replay must never be visible to routing.
    pub(super) visible: bool,
    /// Claims per chain: `None` is the base model, a name is one declared
    /// adapter. The engine seeds an adapter's chain with `(name, path)`, so the
    /// chains never overlap and none can credit another.
    pub(super) claims: HashMap<Option<String>, HashSet<(u32, SequenceHash)>>,
}

impl State {
    pub(super) fn claims_in(
        &self,
        namespace: Option<&str>,
    ) -> Option<&HashSet<(u32, SequenceHash)>> {
        self.claims
            .iter()
            .find(|(identity, _)| identity.as_deref() == namespace)
            .map(|(_, claims)| claims)
    }

    /// Record one claim for `namespace`, refusing only when that chain is full:
    /// an adapter must never evict the base model's evidence.
    pub(super) fn insert_claim(
        &mut self,
        namespace: Option<&str>,
        key: (u32, SequenceHash),
        max_entries: usize,
    ) -> bool {
        let claims = self.claims.entry(namespace.map(str::to_owned)).or_default();
        if claims.contains(&key) {
            return true;
        }
        if claims.len() >= max_entries {
            return false;
        }
        claims.insert(key);
        true
    }

    /// A removal names no chain: the engine evicted those blocks whichever
    /// chain produced them.
    pub(super) fn remove_claim(&mut self, key: (u32, SequenceHash)) {
        for claims in self.claims.values_mut() {
            claims.remove(&key);
        }
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
pub(crate) mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::kv_hash::BlockKeyGenerator;

    use super::super::ingest::HybridEventError;
    use super::super::layout::{HybridEventGroup, HybridGroupKind};
    use super::super::HybridEventIndex;
    use crate::kv_index::wire::ExternalBlockHash;
    use crate::kv_index::wire::{BlockRemoved, BlockStored};
    use crate::kv_index::{KVEvent, KVEventBatch, RoutableTarget};

    fn target() -> RoutableTarget {
        RoutableTarget {
            instance_id: Arc::from("w"),
            dp_rank: 0,
        }
    }

    fn groups() -> Vec<HybridEventGroup> {
        (0..4)
            .map(|group_idx| HybridEventGroup {
                group_idx,
                block_size: 528,
                kind: if group_idx == 3 {
                    HybridGroupKind::FullAttention
                } else {
                    HybridGroupKind::Checkpoint
                },
            })
            .collect()
    }

    fn prompt() -> Vec<u32> {
        (0..1024).map(|i| 100 + i % 100).collect()
    }

    /// A GPU store of `hashes` for one group, as the engine publishes it.
    fn gpu_store(
        group_idx: u32,
        spec: &str,
        block_size: u32,
        token_ids: &[u32],
        hashes: &[SequenceHash],
    ) -> BlockStored {
        BlockStored {
            block_hashes: hashes
                .iter()
                .map(|hash| ExternalBlockHash::Bytes(hash.to_vec()))
                .collect(),
            parent_block_hash: None,
            token_ids: token_ids.to_vec(),
            block_size,
            lora_id: None,
            medium: Some("GPU".into()),
            lora_name: None,
            extra_keys: Some(vec![rmpv::Value::Nil; hashes.len()]),
            group_idx: Some(group_idx),
            kv_cache_spec_kind: Some(spec.into()),
            kv_cache_spec_sliding_window: None,
            locality: None,
            ownership: None,
            session_id: None,
        }
    }

    /// A Qwen3.5 batch: one 528-token block in three Mamba groups and one
    /// full-attention group, hashed with seed 0.
    fn qwen_batch() -> KVEventBatch {
        let tokens = prompt();
        let hash = BlockKeyGenerator::new(528, 0).generate_block_keys(&tokens)[0];
        let events = groups()
            .iter()
            .map(|group| {
                let spec = if group.group_idx == 3 {
                    "full_attention"
                } else {
                    "mamba"
                };
                KVEvent::BlockStored(gpu_store(
                    group.group_idx,
                    spec,
                    528,
                    &tokens[..528],
                    &[hash],
                ))
            })
            .collect();
        KVEventBatch {
            ts: 1.0,
            events,
            data_parallel_rank: Some(0),
        }
    }

    /// The same pages stored under one adapter: the engine hashes them with
    /// the adapter's `(name, path)` extra keys, so this is a different chain.
    fn adapter_batch(name: &str, path: &str) -> KVEventBatch {
        let tokens = prompt();
        let hash =
            BlockKeyGenerator::new(528, 0).generate_block_keys_for_adapter(&tokens, name, path)[0];
        let events = groups()
            .iter()
            .map(|group| {
                let spec = if group.group_idx == 3 {
                    "full_attention"
                } else {
                    "mamba"
                };
                let mut store = gpu_store(group.group_idx, spec, 528, &tokens[..528], &[hash]);
                store.lora_name = Some(name.to_string());
                store.extra_keys = None;
                KVEvent::BlockStored(store)
            })
            .collect();
        KVEventBatch {
            ts: 1.0,
            events,
            data_parallel_rank: Some(0),
        }
    }

    /// `qwen_batch()` encoded as the engine publishes it.
    pub(crate) fn qwen35_payload() -> Vec<u8> {
        let batch = qwen_batch();
        rmp_serde::to_vec_named(&(batch.ts, batch.events, batch.data_parallel_rank)).unwrap()
    }

    fn index(limit: usize) -> HybridEventIndex {
        HybridEventIndex::new(target(), groups(), 528, 0, limit).unwrap()
    }

    #[test]
    fn decoded_qwen_layout_requires_all_groups_and_tracks_removal() {
        let payload = qwen35_payload();
        let batch = crate::kv_index::decode_batch(&payload).unwrap();
        // A payload that does not decode revokes the view.
        let corrupt = index(8);
        corrupt.begin_empty(0);
        assert_eq!(corrupt.apply_payload(0, &payload), Ok(()));
        assert_eq!(
            corrupt.apply_payload(1, &[0xc1]),
            Err(HybridEventError::InvalidBatch)
        );
        assert_eq!(corrupt.reusable_tokens(&prompt(), usize::MAX), 0);
        let index = index(8);
        assert_eq!(
            index.apply_batch(0, &batch),
            Err(HybridEventError::Untrusted)
        );
        index.begin_empty(0);
        assert_eq!(index.apply_batch(0, &batch), Ok(()));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
        assert_eq!(index.reusable_tokens(&prompt(), 527), 0);
        let mut different_prompt = prompt();
        different_prompt[0] += 1;
        assert_eq!(index.reusable_tokens(&different_prompt, usize::MAX), 0);
        let KVEvent::BlockStored(store) = &batch.events[2] else {
            panic!("expected store")
        };
        let removed = KVEventBatch {
            ts: 2.0,
            data_parallel_rank: Some(0),
            events: vec![KVEvent::BlockRemoved(BlockRemoved {
                block_hashes: store.block_hashes.clone(),
                medium: Some("GPU".into()),
                group_idx: Some(2),
                locality: None,
                ownership: None,
            })],
        };
        assert_eq!(index.apply_batch(1, &removed), Ok(()));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
    }

    /// An adapter publishes under its own hash chain. Such a batch must leave
    /// the base model's claims intact instead of revoking the whole view.
    /// An adapter's pages live under their own chain: they are indexed under
    /// the declared adapter, never credited to the base model, and never to
    /// another adapter.
    #[test]
    fn adapter_pages_are_indexed_under_their_own_chain() {
        let index = index(16).with_adapters(vec![
            ("lora1".to_string(), "/adapters/lora1".to_string()),
            ("lora2".to_string(), "/adapters/lora2".to_string()),
        ]);
        index.begin_empty(0);
        assert_eq!(
            index.apply_batch(0, &adapter_batch("lora1", "/adapters/lora1")),
            Ok(())
        );
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
        assert_eq!(
            index.reusable_tokens_in(Some("lora1"), &prompt(), usize::MAX),
            528
        );
        assert_eq!(
            index.reusable_tokens_in(Some("lora2"), &prompt(), usize::MAX),
            0
        );

        // The base model stores the same tokens under its own chain.
        assert_eq!(index.apply_batch(1, &qwen_batch()), Ok(()));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
        assert_eq!(
            index.reusable_tokens_in(Some("lora1"), &prompt(), usize::MAX),
            528
        );

        // An eviction names hashes, not a chain: the engine removes the page
        // from every group, and both views lose it.
        let adapter_hash = BlockKeyGenerator::new(528, 0).generate_block_keys_for_adapter(
            &prompt(),
            "lora1",
            "/adapters/lora1",
        )[0]
        .to_vec();
        let base_hash = BlockKeyGenerator::new(528, 0).generate_block_keys(&prompt())[0].to_vec();
        let removed = KVEventBatch {
            ts: 2.0,
            data_parallel_rank: Some(0),
            events: groups()
                .iter()
                .map(|group| {
                    KVEvent::BlockRemoved(BlockRemoved {
                        block_hashes: vec![
                            ExternalBlockHash::Bytes(adapter_hash.clone()),
                            ExternalBlockHash::Bytes(base_hash.clone()),
                        ],
                        medium: Some("GPU".to_string()),
                        group_idx: Some(group.group_idx),
                        locality: Some("LOCAL".to_string()),
                        ownership: None,
                    })
                })
                .collect(),
        };
        assert_eq!(index.apply_batch(2, &removed), Ok(()));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
        assert_eq!(
            index.reusable_tokens_in(Some("lora1"), &prompt(), usize::MAX),
            0
        );
    }

    /// A page for an adapter this layout does not declare is skipped: it can
    /// neither credit the base model nor revoke the claims the rank holds.
    #[test]
    fn undeclared_adapter_stores_are_skipped() {
        use crate::policies::TieredMatchProvider;

        let index = index(16);
        index.begin_empty(0);
        assert_eq!(index.apply_batch(0, &qwen_batch()), Ok(()));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);

        assert_eq!(
            index.apply_batch(1, &adapter_batch("ghost", "/adapters/ghost")),
            Ok(())
        );
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
        assert_eq!(
            index.reusable_tokens_in(Some("ghost"), &prompt(), usize::MAX),
            0
        );
        assert!(index
            .find_token_matches(&prompt(), 528, Some("ghost"), 0)
            .is_some_and(|hits| hits.is_empty()));
        assert!(index
            .find_token_matches(&prompt(), 528, None, 0)
            .is_some_and(|hits| hits.len() == 1));
    }

    #[test]
    fn adapter_stores_do_not_revoke_the_base_view() {
        let index = index(16);
        index.begin_empty(0);
        assert_eq!(index.apply_batch(0, &qwen_batch()), Ok(()));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);

        let mut adapter_batch = qwen_batch();
        for event in &mut adapter_batch.events {
            if let KVEvent::BlockStored(store) = event {
                store.lora_name = Some("lora1".to_string());
            }
        }
        assert_eq!(index.apply_batch(1, &adapter_batch), Ok(()));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
        // The base chain keeps applying afterwards.
        assert_eq!(index.apply_batch(2, &qwen_batch()), Ok(()));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
    }

    #[test]
    fn gap_and_old_sequence_revoke_the_whole_view() {
        let index = index(8);
        let batch = qwen_batch();
        index.begin_empty(0);
        assert_eq!(index.apply_batch(0, &batch), Ok(()));
        assert_eq!(
            index.apply_batch(2, &batch),
            Err(HybridEventError::Sequence)
        );
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);

        index.begin_empty(0);
        assert_eq!(index.apply_batch(0, &batch), Ok(()));
        assert_eq!(
            index.apply_batch(0, &batch),
            Err(HybridEventError::Sequence)
        );
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
        assert_eq!(
            index.apply_batch(1, &batch),
            Err(HybridEventError::Untrusted)
        );
        index.begin_empty(5);
        assert_eq!(index.apply_batch(5, &batch), Ok(()));
        assert_eq!(
            index.apply_batch(2, &batch),
            Err(HybridEventError::Sequence)
        );
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
    }

    #[test]
    fn replay_stays_hidden_until_the_verified_watermark() {
        let index = index(8);
        let batch = qwen_batch();
        index.begin_replay(0);
        assert_eq!(index.apply_batch(0, &batch), Ok(()));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
        assert_eq!(index.finish_replay(1), Ok(()));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);

        let KVEvent::BlockStored(store) = &batch.events[2] else {
            panic!("expected store")
        };
        let eviction = KVEventBatch {
            ts: 2.0,
            data_parallel_rank: Some(0),
            events: vec![KVEvent::BlockRemoved(BlockRemoved {
                block_hashes: store.block_hashes.clone(),
                medium: Some("GPU".into()),
                group_idx: Some(2),
                locality: None,
                ownership: None,
            })],
        };
        index.begin_replay(0);
        assert_eq!(index.apply_batch(0, &batch), Ok(()));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
        assert_eq!(index.apply_batch(1, &eviction), Ok(()));
        assert_eq!(index.finish_replay(2), Ok(()));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);

        index.begin_replay(0);
        assert_eq!(index.apply_batch(0, &batch), Ok(()));
        assert_eq!(index.finish_replay(2), Err(HybridEventError::Sequence));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
        assert_eq!(
            index.apply_batch(1, &batch),
            Err(HybridEventError::Untrusted)
        );
    }

    #[test]
    fn invalid_event_or_rank_cannot_expose_a_partial_batch() {
        let index = index(8);
        let mut batch = qwen_batch();
        let KVEvent::BlockStored(last) = &mut batch.events[3] else {
            panic!("expected store")
        };
        last.block_hashes = vec![ExternalBlockHash::Int(42)];
        index.begin_empty(0);
        assert_eq!(
            index.apply_batch(0, &batch),
            Err(HybridEventError::InvalidBatch)
        );
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
        assert_eq!(
            index.apply_batch(0, &qwen_batch()),
            Err(HybridEventError::Untrusted)
        );

        let mut wrong_rank = qwen_batch();
        wrong_rank.data_parallel_rank = Some(1);
        index.begin_empty(0);
        assert_eq!(
            index.apply_batch(0, &wrong_rank),
            Err(HybridEventError::InvalidBatch)
        );
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
    }

    #[test]
    fn unmodeled_cache_identity_and_group_metadata_are_rejected() {
        let index = index(8);
        let mut salted = qwen_batch();
        let KVEvent::BlockStored(last) = &mut salted.events[3] else {
            panic!("expected store")
        };
        last.extra_keys = Some(vec![rmpv::Value::from("unknown-salt")]);
        index.begin_empty(0);
        assert_eq!(
            index.apply_batch(0, &salted),
            Err(HybridEventError::InvalidBatch)
        );
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);

        let mut wrong_kind = qwen_batch();
        let KVEvent::BlockStored(last) = &mut wrong_kind.events[3] else {
            panic!("expected store")
        };
        last.kv_cache_spec_kind = Some("mamba".into());
        index.begin_empty(0);
        assert_eq!(
            index.apply_batch(0, &wrong_kind),
            Err(HybridEventError::InvalidBatch)
        );
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
    }

    #[test]
    fn capacity_failure_invalidates_all_groups_and_requires_a_new_barrier() {
        let index = index(3);
        let batch = qwen_batch();
        index.begin_empty(0);
        assert_eq!(
            index.apply_batch(0, &batch),
            Err(HybridEventError::Capacity)
        );
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
        assert_eq!(
            index.apply_batch(1, &batch),
            Err(HybridEventError::Untrusted)
        );
    }

    #[test]
    fn clear_drops_every_group_before_new_events_are_applied() {
        let index = index(8);
        let batch = qwen_batch();
        index.begin_empty(0);
        assert_eq!(index.apply_batch(0, &batch), Ok(()));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
        let cleared = KVEventBatch {
            ts: 2.0,
            data_parallel_rank: Some(0),
            events: vec![KVEvent::AllBlocksCleared(Default::default())],
        };
        assert_eq!(index.apply_batch(1, &cleared), Ok(()));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
        let partial = KVEventBatch {
            ts: 3.0,
            data_parallel_rank: Some(0),
            events: vec![batch.events[0].clone()],
        };
        assert_eq!(index.apply_batch(2, &partial), Ok(()));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
    }

    #[test]
    fn unsupported_or_ambiguous_layouts_are_rejected() {
        assert!(matches!(
            HybridEventIndex::new(target(), groups(), 0, 0, 8),
            Err(HybridEventError::InvalidLayout)
        ));
        let mut duplicate = groups();
        duplicate[0].group_idx = 1;
        assert!(matches!(
            HybridEventIndex::new(target(), duplicate, 528, 0, 8),
            Err(HybridEventError::InvalidLayout)
        ));
    }

    #[test]
    fn sparse_checkpoint_does_not_require_its_parent_to_be_resident() {
        let tokens: Vec<u32> = (0..17).collect();
        let hashes = BlockKeyGenerator::new(4, 0).generate_block_keys(&tokens);
        let layout = vec![
            HybridEventGroup {
                group_idx: 0,
                block_size: 4,
                kind: HybridGroupKind::FullAttention,
            },
            HybridEventGroup {
                group_idx: 1,
                block_size: 8,
                kind: HybridGroupKind::Checkpoint,
            },
        ];
        let mut checkpoint = gpu_store(1, "mamba", 8, &tokens[8..16], &hashes[3..4]);
        checkpoint.parent_block_hash = Some(ExternalBlockHash::Bytes(hashes[1].to_vec()));
        let batch = KVEventBatch {
            ts: 1.0,
            data_parallel_rank: Some(0),
            events: vec![
                KVEvent::BlockStored(gpu_store(0, "full_attention", 4, &tokens[..16], &hashes)),
                KVEvent::BlockStored(checkpoint),
            ],
        };
        let index = HybridEventIndex::new(target(), layout, 4, 0, 8).unwrap();
        index.begin_empty(0);
        assert_eq!(index.apply_batch(0, &batch), Ok(()));
        assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 16);
    }

    #[test]
    fn concurrent_queries_never_see_a_partially_applied_batch() {
        let index = Arc::new(index(8));
        let initial = qwen_batch();
        let KVEvent::BlockStored(store) = &initial.events[2] else {
            panic!("expected store")
        };
        let replacement = KVEventBatch {
            ts: 2.0,
            data_parallel_rank: Some(0),
            events: vec![
                KVEvent::BlockRemoved(BlockRemoved {
                    block_hashes: store.block_hashes.clone(),
                    medium: Some("GPU".into()),
                    group_idx: Some(2),
                    locality: None,
                    ownership: None,
                }),
                KVEvent::BlockStored(store.clone()),
            ],
        };
        index.begin_empty(0);
        assert_eq!(index.apply_batch(0, &initial), Ok(()));
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let reader_index = Arc::clone(&index);
        let reader_barrier = Arc::clone(&barrier);
        let reader = std::thread::spawn(move || {
            let tokens = prompt();
            reader_barrier.wait();
            for _ in 0..128 {
                assert_eq!(reader_index.reusable_tokens(&tokens, usize::MAX), 528);
            }
        });
        barrier.wait();
        for seq in 1..=128 {
            assert_eq!(index.apply_batch(seq, &replacement), Ok(()));
        }
        reader.join().unwrap();
    }

    #[test]
    fn qwen35_checkpoint_groups_may_omit_hashes_on_a_reused_chunk() {
        // Captured from Qwen3.5-4B on vLLM 0.33 (block size 576): re-running a
        // cached prompt emits a stored event whose Mamba groups carry zero
        // hashes and a parent, while full attention carries one hash.
        let tokens: Vec<u32> = (0..2048).map(|i| 100 + i % 100).collect();
        let keys = BlockKeyGenerator::new(576, 0).generate_block_keys(&tokens);
        let groups: Vec<HybridEventGroup> = (0..4)
            .map(|group_idx| HybridEventGroup {
                group_idx,
                block_size: 576,
                kind: if group_idx == 3 {
                    HybridGroupKind::FullAttention
                } else {
                    HybridGroupKind::Checkpoint
                },
            })
            .collect();
        let store = |group_idx: u32, hashes: &[SequenceHash], tokens_len: usize| {
            let spec = if group_idx == 3 {
                "full_attention"
            } else {
                "mamba"
            };
            KVEvent::BlockStored(gpu_store(
                group_idx,
                spec,
                576,
                &tokens[..tokens_len],
                hashes,
            ))
        };
        let index = HybridEventIndex::new(target(), groups, 576, 0, 64).unwrap();
        index.begin_empty(0);
        let first = KVEventBatch {
            ts: 1.0,
            data_parallel_rank: Some(0),
            events: vec![
                store(0, &keys[2..3], 1728),
                store(1, &keys[2..3], 1728),
                store(2, &keys[2..3], 1728),
                store(3, &keys[..3], 1728),
            ],
        };
        assert_eq!(index.apply_batch(0, &first), Ok(()));
        assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 1728);
        // Re-run of the same prompt: checkpoints publish nothing new.
        let rerun = KVEventBatch {
            ts: 2.0,
            data_parallel_rank: Some(0),
            events: vec![
                store(0, &[], 576),
                store(1, &[], 576),
                store(2, &[], 576),
                store(3, &keys[..1], 576),
            ],
        };
        assert_eq!(index.apply_batch(1, &rerun), Ok(()));
        assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 1728);
    }

    #[test]
    fn another_dp_rank_never_answers_for_this_rank() {
        use crate::policies::TieredMatchProvider;

        let index = index(8);
        index.begin_empty(0);
        index.apply_payload(0, &qwen35_payload()).unwrap();
        // Each rank serves independently; its events, ownership and history
        // belong to that rank only.
        assert!(
            !TieredMatchProvider::find_token_matches(&index, &prompt(), 528, None, 0)
                .unwrap()
                .is_empty()
        );
        assert!(
            TieredMatchProvider::find_token_matches(&index, &prompt(), 528, None, 1)
                .unwrap()
                .is_empty()
        );
    }
}
