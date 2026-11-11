//! What this rank currently holds: claims per chain and page evidence.

use std::collections::{HashMap, HashSet};

use crate::kv_hash::SequenceHash;
use crate::kv_index::StorageTier;

pub(super) struct State {
    /// The next sequence in a continuous, source-bound event stream.
    pub(super) next_seq: Option<u64>,
    /// Partial replay must never be visible to routing.
    pub(super) visible: bool,
    /// Claims per cache identity: `None` is the base model's chain, a name is
    /// that adapter's chain. The engine seeds block hashes with the adapter
    /// identity, so the two never share a chain and neither can credit the
    /// other.
    pub(super) claims: HashMap<Option<String>, HashMap<(u32, SequenceHash), TierSet>>,
    /// Groups that have published at least one page hash, per identity. A group
    /// that only publishes key-only stores reports no addressable page, so it
    /// cannot bound the shared prefix.
    pub(super) page_evidence: HashMap<Option<String>, HashSet<u32>>,
}

/// The tiers one page is resident on. A store adds a tier, a removal clears
/// only the tier it names, and the best remaining tier is what the page is
/// credited at.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct TierSet(u8);

impl TierSet {
    const DEVICE: u8 = 1 << 0;
    const HOST: u8 = 1 << 1;
    const DISK: u8 = 1 << 2;
    const EXTERNAL: u8 = 1 << 3;

    fn bit(tier: StorageTier) -> u8 {
        match tier {
            StorageTier::Device => Self::DEVICE,
            StorageTier::HostPinned => Self::HOST,
            StorageTier::Disk => Self::DISK,
            StorageTier::External => Self::EXTERNAL,
        }
    }

    pub(super) fn with(mut self, tier: StorageTier) -> Self {
        self.0 |= Self::bit(tier);
        self
    }

    pub(super) fn without(mut self, tier: StorageTier) -> Self {
        self.0 &= !Self::bit(tier);
        self
    }

    pub(super) fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The tier a hit is credited at: the fastest copy the page still has.
    pub(super) fn best(self) -> Option<StorageTier> {
        if self.0 & Self::DEVICE != 0 {
            Some(StorageTier::Device)
        } else if self.0 & Self::HOST != 0 {
            Some(StorageTier::HostPinned)
        } else if self.0 & Self::DISK != 0 {
            Some(StorageTier::Disk)
        } else if self.0 & Self::EXTERNAL != 0 {
            Some(StorageTier::External)
        } else {
            None
        }
    }
}

impl State {
    /// The claims of one identity. A rank carries a handful of identities, so
    /// the scan is cheaper than allocating a key for every lookup.
    pub(super) fn claims_in(
        &self,
        namespace: Option<&str>,
    ) -> Option<&HashMap<(u32, SequenceHash), TierSet>> {
        self.claims
            .iter()
            .find(|(identity, _)| identity.as_deref() == namespace)
            .map(|(_, claims)| claims)
    }

    /// The groups one identity has page evidence for.
    pub(super) fn page_evidence_in(&self, namespace: Option<&str>) -> Option<&HashSet<u32>> {
        self.page_evidence
            .iter()
            .find(|(identity, _)| identity.as_deref() == namespace)
            .map(|(_, evidence)| evidence)
    }

    /// Record one claim for `namespace`, refusing only when that identity is
    /// full: one adapter must never evict the base model's evidence.
    pub(super) fn insert_claim(
        &mut self,
        namespace: Option<&str>,
        key: (u32, SequenceHash),
        tier: StorageTier,
        max_entries: usize,
    ) -> bool {
        let claims = self.claims.entry(namespace.map(str::to_owned)).or_default();
        if let Some(resident) = claims.get_mut(&key) {
            *resident = resident.with(tier);
            return true;
        }
        if claims.len() >= max_entries {
            return false;
        }
        claims.insert(key, TierSet::default().with(tier));
        true
    }

    /// A removal whose medium names no known tier drops every copy of the page:
    /// keeping a stale claim would over-credit, dropping one costs at most a
    /// re-prefill.
    pub(super) fn remove_claim_all(&mut self, key: (u32, SequenceHash)) {
        for claims in self.claims.values_mut() {
            claims.remove(&key);
        }
    }

    /// A removal names one tier and no identity: the engine evicted that copy
    /// whichever chain produced it. A page that still has another copy stays
    /// claimed at the better remaining tier.
    pub(super) fn remove_claim(&mut self, key: (u32, SequenceHash), tier: StorageTier) {
        for claims in self.claims.values_mut() {
            if let Some(resident) = claims.get_mut(&key) {
                *resident = resident.without(tier);
                if resident.is_empty() {
                    claims.remove(&key);
                }
            }
        }
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
pub(crate) mod tests {
    use std::sync::Arc;

    use super::super::ingest::HybridEventError;
    use super::super::layout::{event_kind_matches, HybridEventGroup, HybridGroupKind};
    use super::super::HybridEventIndex;
    use crate::kv_hash::{BlockKeyGenerator, SequenceHash};
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

    fn host_store(
        batch: &crate::kv_index::KVEventBatch,
        medium: &str,
    ) -> crate::kv_index::KVEventBatch {
        let mut batch = batch.clone();
        for event in &mut batch.events {
            if let KVEvent::BlockStored(store) = event {
                store.medium = Some(medium.to_string());
            }
        }
        batch
    }

    #[test]
    fn host_resident_pages_are_credited_at_their_tier() {
        use crate::policies::TieredMatchProvider;

        let index = index(16);
        index.begin_empty(0);
        assert_eq!(
            index.apply_batch(0, &host_store(&qwen_batch(), "CPU")),
            Ok(())
        );
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
        let matches = index
            .find_token_matches(&prompt(), 528, None, 0)
            .expect("hybrid provider");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].tier, crate::kv_index::StorageTier::HostPinned);
    }

    #[test]
    fn evicting_the_device_copy_keeps_the_host_credit() {
        let index = index(16);
        index.begin_empty(0);
        assert_eq!(index.apply_batch(0, &qwen_batch()), Ok(()));
        assert_eq!(
            index.apply_batch(1, &host_store(&qwen_batch(), "CPU")),
            Ok(())
        );

        let mut removed = qwen_batch();
        for event in &mut removed.events {
            if let KVEvent::BlockStored(store) = event {
                *event = KVEvent::BlockRemoved(BlockRemoved {
                    block_hashes: store.block_hashes.clone(),
                    medium: Some("GPU".into()),
                    group_idx: store.group_idx,
                    locality: None,
                    ownership: None,
                });
            }
        }
        assert_eq!(index.apply_batch(2, &removed), Ok(()));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
    }

    #[test]
    fn a_foreign_owner_is_skipped_without_revoking_the_view() {
        let index = index(16);
        index.begin_empty(0);
        assert_eq!(index.apply_batch(0, &qwen_batch()), Ok(()));
        let mut foreign = qwen_batch();
        for event in &mut foreign.events {
            if let KVEvent::BlockStored(store) = event {
                store.ownership = Some("offload".into());
            }
        }
        assert_eq!(index.apply_batch(1, &foreign), Ok(()));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
    }

    /// A required group that never published a page caps reuse at zero: its
    /// silence is not evidence that the page is unnecessary.
    #[test]
    fn a_required_group_without_page_evidence_caps_reuse_at_zero() {
        let index = index(16);
        index.begin_empty(0);
        let mut batch = qwen_batch();
        batch.events.retain(|event| match event {
            KVEvent::BlockStored(store) => store.group_idx == Some(3),
            _ => false,
        });
        assert_eq!(index.apply_batch(0, &batch), Ok(()));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
    }

    /// An adapter publishes under its own hash chain. Such a batch must leave
    /// the base model's claims intact instead of revoking the whole view.
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
    /// A real adapter store carries its identity as extra keys; that is not a
    /// foreign identity, so the pages still count for the adapter's chain.
    #[test]
    fn adapter_stores_carrying_lora_extra_keys_are_credited() {
        let index =
            index(16).with_adapters(vec![("lora1".to_string(), "/adapters/lora1".to_string())]);
        index.begin_empty(0);
        let mut batch = adapter_batch("lora1", "/adapters/lora1");
        for event in &mut batch.events {
            if let KVEvent::BlockStored(store) = event {
                store.extra_keys = Some(vec![rmpv::Value::String("lora1".into())]);
            }
        }
        assert_eq!(index.apply_batch(0, &batch), Ok(()));
        assert_eq!(
            index.reusable_tokens_in(Some("lora1"), &prompt(), usize::MAX),
            528
        );
    }

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

    /// DeepSeek V4 publishes MLA pages without a state ratio; those restore
    /// from the root, so a full-attention group accepts them.
    #[test]
    fn mla_pages_are_full_attention_stores() {
        let store = gpu_store(0, "mla_attention", 528, &prompt()[..528], &[[7u8; 32]]);
        assert!(event_kind_matches(HybridGroupKind::FullAttention, &store));
        let mut sliding = store;
        sliding.kv_cache_spec_sliding_window = Some(128);
        assert!(!event_kind_matches(
            HybridGroupKind::FullAttention,
            &sliding
        ));
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
        // An integer hash is the engine's default event encoding, so it gets its
        // own error rather than the generic shape mismatch.
        last.block_hashes = vec![ExternalBlockHash::Int(42)];
        index.begin_empty(0);
        assert_eq!(
            index.apply_batch(0, &batch),
            Err(HybridEventError::IntBlockHashes)
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
    fn an_int_parent_hash_is_reported_as_int_block_hashes() {
        let index = index(8);
        let mut batch = qwen_batch();
        let KVEvent::BlockStored(last) = &mut batch.events[3] else {
            panic!("expected store")
        };
        last.parent_block_hash = Some(ExternalBlockHash::Int(7));
        index.begin_empty(0);
        assert_eq!(
            index.apply_batch(0, &batch),
            Err(HybridEventError::IntBlockHashes)
        );
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
        let mut wrong = groups();
        wrong[0].kind = HybridGroupKind::SlidingWindow { window_tokens: 0 };
        assert!(matches!(
            HybridEventIndex::new(target(), wrong, 16, 0, 8),
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

    fn qwen4_groups() -> Vec<HybridEventGroup> {
        let mut groups: Vec<_> = (0..3)
            .map(|group_idx| HybridEventGroup {
                group_idx,
                block_size: 320,
                kind: HybridGroupKind::Checkpoint,
            })
            .collect();
        groups.push(HybridEventGroup {
            group_idx: 3,
            block_size: 320,
            kind: HybridGroupKind::CompressedFullAttention {
                tokens_per_state: 4,
            },
        });
        groups
    }

    fn qwen4_batch(kind_override: Option<&str>) -> KVEventBatch {
        let tokens = prompt();
        let hash = BlockKeyGenerator::new(320, 0).generate_block_keys(&tokens)[0];
        let events = qwen4_groups()
            .iter()
            .map(|group| {
                let compressed =
                    matches!(group.kind, HybridGroupKind::CompressedFullAttention { .. });
                let default = if compressed { "mla_attention" } else { "mamba" };
                let spec = kind_override.unwrap_or(default);
                KVEvent::BlockStored(gpu_store(
                    group.group_idx,
                    spec,
                    320,
                    &tokens[..320],
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

    #[test]
    fn qwen4_compressed_group_accepts_mla_events_and_rejects_other_specs() {
        let index = HybridEventIndex::new(target(), qwen4_groups(), 320, 0, 8).unwrap();
        index.begin_empty(0);
        assert_eq!(index.apply_batch(0, &qwen4_batch(None)), Ok(()));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 320);

        let wrong = HybridEventIndex::new(target(), qwen4_groups(), 320, 0, 8).unwrap();
        wrong.begin_empty(0);
        assert_eq!(
            wrong.apply_batch(0, &qwen4_batch(Some("full_attention"))),
            Err(HybridEventError::InvalidBatch)
        );
    }

    #[test]
    fn compressed_group_must_align_with_its_block_size() {
        let mut groups = qwen4_groups();
        groups[3].kind = HybridGroupKind::CompressedFullAttention {
            tokens_per_state: 7,
        };
        assert!(matches!(
            HybridEventIndex::new(target(), groups, 320, 0, 8),
            Err(HybridEventError::InvalidLayout)
        ));
    }

    #[test]
    fn checkpoint_event_may_cover_the_whole_cached_chunk() {
        let tokens: Vec<u32> = (0..2048).map(|i| 100 + i % 100).collect();
        let keys = BlockKeyGenerator::new(320, 0).generate_block_keys(&tokens);
        // Captured from vLLM 0.33: a Mamba group publishes only the retained
        // checkpoint hash while token_ids spans the whole cached chunk. The
        // QSA attention group still publishes one hash per block.
        let batch = |attention_hashes: usize, checkpoint_hashes: usize| {
            let events = qwen4_groups()
                .iter()
                .map(|group| {
                    let attention =
                        matches!(group.kind, HybridGroupKind::CompressedFullAttention { .. });
                    let count = if attention {
                        attention_hashes
                    } else {
                        checkpoint_hashes
                    };
                    let (spec, hashes) = if attention {
                        ("mla_attention", &keys[..count])
                    } else {
                        ("mamba", &keys[6 - count..6])
                    };
                    KVEvent::BlockStored(gpu_store(
                        group.group_idx,
                        spec,
                        320,
                        &tokens[..1920],
                        hashes,
                    ))
                })
                .collect();
            KVEventBatch {
                ts: 1.0,
                events,
                data_parallel_rank: Some(0),
            }
        };
        let index = HybridEventIndex::new(target(), qwen4_groups(), 320, 0, 16).unwrap();
        index.begin_empty(0);
        assert_eq!(index.apply_batch(0, &batch(6, 1)), Ok(()));
        assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 1920);

        let sparse_attention = HybridEventIndex::new(target(), qwen4_groups(), 320, 0, 16).unwrap();
        sparse_attention.begin_empty(0);
        assert_eq!(
            sparse_attention.apply_batch(0, &batch(1, 1)),
            Err(HybridEventError::InvalidBatch)
        );
        assert_eq!(sparse_attention.reusable_tokens(&tokens, usize::MAX), 0);

        // vLLM can emit a checkpoint group with an empty hash list when all
        // blocks in the chunk are null. The stream stays valid, but that group
        // has no residency and cannot complete a hit.
        let empty_checkpoints =
            HybridEventIndex::new(target(), qwen4_groups(), 320, 0, 16).unwrap();
        empty_checkpoints.begin_empty(0);
        assert_eq!(empty_checkpoints.apply_batch(0, &batch(6, 0)), Ok(()));
        assert_eq!(empty_checkpoints.reusable_tokens(&tokens, usize::MAX), 0);
        assert_eq!(empty_checkpoints.apply_batch(1, &batch(6, 1)), Ok(()));
        assert_eq!(empty_checkpoints.reusable_tokens(&tokens, usize::MAX), 1920);
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

    /// One block per published hash, with the event shape the verified group
    /// kind requires.
    #[allow(clippy::too_many_arguments)]
    fn blocks(
        group: HybridEventGroup,
        spec: &str,
        window: Option<u32>,
        keys: &[SequenceHash],
        block: usize,
        tokens: &[u32],
        from: usize,
        to: usize,
    ) -> KVEvent {
        let mut store = gpu_store(
            group.group_idx,
            spec,
            block as u32,
            &tokens[from * block..to * block],
            &keys[from..to],
        );
        store.kv_cache_spec_sliding_window = window;
        KVEvent::BlockStored(store)
    }

    #[test]
    fn sliding_group_needs_its_window_not_the_whole_prefix() {
        // A bounded-replay sliding group keeps only the trailing window, so a
        // hit must not require its earlier blocks, and a hit must not exceed
        // what the full-attention group can restore.
        let tokens: Vec<u32> = (0..1024).map(|i| 100 + i % 100).collect();
        let keys = BlockKeyGenerator::new(64, 0).generate_block_keys(&tokens);
        let groups = vec![
            HybridEventGroup {
                group_idx: 0,
                block_size: 64,
                kind: HybridGroupKind::FullAttention,
            },
            HybridEventGroup {
                group_idx: 1,
                block_size: 64,
                kind: HybridGroupKind::SlidingWindow { window_tokens: 64 },
            },
        ];
        let index = HybridEventIndex::new(target(), groups.clone(), 64, 0, 64).unwrap();
        index.begin_empty(0);
        let window_only = KVEventBatch {
            ts: 1.0,
            data_parallel_rank: Some(0),
            events: vec![
                blocks(groups[0], "full_attention", None, &keys, 64, &tokens, 0, 8),
                blocks(
                    groups[1],
                    "sliding_window_mla",
                    Some(64),
                    &keys,
                    64,
                    &tokens,
                    7,
                    8,
                ),
            ],
        };
        assert_eq!(index.apply_batch(0, &window_only), Ok(()));
        assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 512);

        // Without the block the window covers, the hit retreats to the last
        // boundary whose window is complete instead of claiming the prefix.
        let gap = HybridEventIndex::new(target(), groups.clone(), 64, 0, 64).unwrap();
        gap.begin_empty(0);
        let missing_window = KVEventBatch {
            ts: 1.0,
            data_parallel_rank: Some(0),
            events: vec![
                blocks(groups[0], "full_attention", None, &keys, 64, &tokens, 0, 8),
                blocks(
                    groups[1],
                    "sliding_window_mla",
                    Some(64),
                    &keys,
                    64,
                    &tokens,
                    0,
                    6,
                ),
            ],
        };
        assert_eq!(gap.apply_batch(0, &missing_window), Ok(()));
        assert_eq!(gap.reusable_tokens(&tokens, usize::MAX), 384);

        // A sliding window of another size is another layout, not a hit.
        let wrong = HybridEventIndex::new(target(), groups.clone(), 64, 0, 64).unwrap();
        wrong.begin_empty(0);
        let other_window = KVEventBatch {
            ts: 1.0,
            data_parallel_rank: Some(0),
            events: vec![
                blocks(groups[0], "full_attention", None, &keys, 64, &tokens, 0, 8),
                blocks(
                    groups[1],
                    "sliding_window_mla",
                    Some(32),
                    &keys,
                    64,
                    &tokens,
                    7,
                    8,
                ),
            ],
        };
        assert_eq!(
            wrong.apply_batch(0, &other_window),
            Err(HybridEventError::InvalidBatch)
        );
        assert_eq!(wrong.reusable_tokens(&tokens, usize::MAX), 0);
    }

    /// The event shape a DeepSeek V4 layout publishes: sliding groups keep only
    /// their resident tail, one group publishes 4-token pages next to its own
    /// 256-token blocks, and the groups use different block sizes that share
    /// one hash chain at the layout's 4-token granularity.
    #[test]
    fn tail_window_events_credit_the_engine_boundary() {
        let tokens: Vec<u32> = (0..1024).map(|i| 100 + i % 100).collect();
        let keys = BlockKeyGenerator::new(4, 0).generate_block_keys(&tokens);
        let view = |block: usize| -> Vec<SequenceHash> {
            let stride = block / 4;
            keys.iter()
                .skip(stride - 1)
                .step_by(stride)
                .copied()
                .collect()
        };
        let groups = vec![
            HybridEventGroup {
                group_idx: 0,
                block_size: 64,
                kind: HybridGroupKind::SlidingWindow { window_tokens: 128 },
            },
            HybridEventGroup {
                group_idx: 1,
                block_size: 64,
                kind: HybridGroupKind::SlidingWindow { window_tokens: 128 },
            },
            HybridEventGroup {
                group_idx: 2,
                block_size: 256,
                kind: HybridGroupKind::CompressedFullAttention {
                    tokens_per_state: 4,
                },
            },
            HybridEventGroup {
                group_idx: 3,
                block_size: 4,
                kind: HybridGroupKind::SlidingWindow { window_tokens: 8 },
            },
            HybridEventGroup {
                group_idx: 4,
                block_size: 8,
                kind: HybridGroupKind::SlidingWindow { window_tokens: 128 },
            },
        ];
        let store = |group: HybridEventGroup,
                     spec: &str,
                     window: Option<u32>,
                     hashes: Vec<SequenceHash>,
                     block: usize,
                     span: usize| {
            let mut store = gpu_store(
                group.group_idx,
                spec,
                block as u32,
                &tokens[..span],
                &hashes,
            );
            store.kv_cache_spec_sliding_window = window;
            KVEvent::BlockStored(store)
        };
        let index = HybridEventIndex::new(target(), groups.clone(), 4, 0, 4096).unwrap();
        index.begin_empty(0);
        // A V4-class layout publishes the sliding groups as key-only stores: the
        // pages are named by the compressed group, and the sliding groups carry
        // no addressable page of their own. They therefore cannot bound the
        // shared prefix, and the compressed group's three restorable blocks do.
        let batch = KVEventBatch {
            ts: 1.0,
            data_parallel_rank: Some(0),
            events: vec![
                store(groups[0], "sliding_window_mla", Some(128), vec![], 64, 1024),
                store(groups[1], "sliding_window_mla", Some(128), vec![], 64, 1024),
                store(groups[2], "mla_attention", None, view(256), 256, 1024),
                // The same group also publishes compressor pages it stores on
                // their own; they carry no boundary the group does not already
                // report.
                store(groups[2], "mla_attention", None, vec![keys[0]], 4, 4),
                store(groups[3], "sliding_window_mla", Some(8), vec![], 4, 1024),
                store(groups[4], "sliding_window_mla", Some(128), vec![], 8, 1024),
            ],
        };
        assert_eq!(index.apply_batch(0, &batch), Ok(()));
        // The compressed group restores three 256-token blocks below the
        // last-token bound, and no other group has page evidence to lower it.
        assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 768);
    }

    /// A sliding store claims exactly the hashes it published. vLLM publishes
    /// the page boundaries it actually holds and skips pages it never stored, so
    /// rebuilding the chain from `token_ids` would claim pages the engine cannot
    /// restore and would exhaust the capacity limit over store/evict cycles.
    #[test]
    fn sliding_claims_only_the_published_page_boundaries() {
        let tokens: Vec<u32> = (0..1024).map(|i| 100 + i % 100).collect();
        let keys = BlockKeyGenerator::new(4, 0).generate_block_keys(&tokens);
        let group = HybridEventGroup {
            group_idx: 0,
            block_size: 64,
            kind: HybridGroupKind::SlidingWindow { window_tokens: 128 },
        };
        let tail: Vec<SequenceHash> = keys.iter().skip(15).step_by(16).copied().collect();
        // The three pages that hold the window ending at the last full block.
        let tail = &tail[tail.len() - 3..];
        let mut store = gpu_store(0, "sliding_window_mla", 64, &tokens[..1024], tail);
        store.kv_cache_spec_sliding_window = Some(128);
        let batch = KVEventBatch {
            ts: 1.0,
            data_parallel_rank: Some(0),
            events: vec![KVEvent::BlockStored(store)],
        };
        // Three published pages are three claims, not sixteen chain steps.
        for (capacity, expected) in [(3, Ok(())), (2, Err(HybridEventError::Capacity))] {
            let index = HybridEventIndex::new(target(), vec![group], 4, 0, capacity).unwrap();
            index.begin_empty(0);
            assert_eq!(index.apply_batch(0, &batch), expected);
        }
        // The published tail still credits its own window.
        let index = HybridEventIndex::new(target(), vec![group], 4, 0, 16).unwrap();
        index.begin_empty(0);
        assert_eq!(index.apply_batch(0, &batch), Ok(()));
        assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 960);
    }

    /// The event shape a DeepSeek V4.1 layout publishes: one 128-token
    /// compressed group carries the whole prefix and the sliding groups publish
    /// nothing, so the layout charges their shared replay window instead.
    #[test]
    fn bounded_replay_credits_one_window_below_the_engine() {
        let tokens: Vec<u32> = (0..1024).map(|i| 100 + i % 100).collect();
        let keys = BlockKeyGenerator::new(128, 0).generate_block_keys(&tokens);
        let groups = vec![HybridEventGroup {
            group_idx: 5,
            block_size: 128,
            kind: HybridGroupKind::CompressedFullAttention {
                tokens_per_state: 1,
            },
        }];
        let index = HybridEventIndex::new(target(), groups, 128, 0, 4096)
            .unwrap()
            .with_replay_tokens(128)
            .unwrap();
        index.begin_empty(0);
        let batch = KVEventBatch {
            ts: 1.0,
            data_parallel_rank: Some(0),
            events: vec![KVEvent::BlockStored(gpu_store(
                5,
                "mla_attention",
                128,
                &tokens,
                &keys,
            ))],
        };
        assert_eq!(index.apply_batch(0, &batch), Ok(()));
        // The engine reports 896 for a 1,024-token repeat (the last whole block
        // under the last-token bound) and replays the trailing window on top of
        // it, so the reusable amount is one window lower.
        assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 768);
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
