//! Atomic, fail-closed hybrid event state for one worker rank.

use std::collections::HashSet;

use parking_lot::RwLock;

use super::hybrid::{common_reusable_tokens, HybridGroupKind, HybridGroupView};
use super::wire::{BlockStored, ExternalBlockHash};
use super::{KVEvent, KVEventBatch, Locality, RoutableTarget, StorageTier, TierMatch};
use crate::kv_hash::{BlockKeyGenerator, SequenceHash};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridEventGroup {
    pub group_idx: u32,
    /// The verified cache block size in tokens, not the hash granularity.
    pub block_size: u32,
    pub kind: HybridGroupKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridEventError {
    InvalidLayout,
    Untrusted,
    Sequence,
    InvalidBatch,
    Capacity,
}

struct State {
    /// The next sequence in a continuous, source-bound event stream.
    next_seq: Option<u64>,
    /// Partial replay must never be visible to routing.
    visible: bool,
    claims: HashSet<(u32, SequenceHash)>,
}

/// A bounded index for one verified identity, worker, rank and incarnation.
/// Stores and removals across all groups share one lock and one sequence.
pub struct HybridEventIndex {
    target: RoutableTarget,
    groups: Vec<HybridEventGroup>,
    keys: BlockKeyGenerator,
    max_entries: usize,
    state: RwLock<State>,
}

impl HybridEventIndex {
    /// Test constructor keyed by a `PYTHONHASHSEED` instead of a root digest.
    #[cfg(test)]
    pub fn new(
        target: RoutableTarget,
        groups: Vec<HybridEventGroup>,
        hash_block_size: u32,
        hash_seed: u32,
        max_entries: usize,
    ) -> Result<Self, HybridEventError> {
        if hash_block_size == 0 {
            return Err(HybridEventError::InvalidLayout);
        }
        Self::with_keys(
            target,
            groups,
            max_entries,
            BlockKeyGenerator::new(hash_block_size as usize, u64::from(hash_seed)),
        )
    }

    /// The root digest and group layout must come from the same verified engine.
    pub fn new_with_root_hash(
        target: RoutableTarget,
        groups: Vec<HybridEventGroup>,
        hash_block_size: u32,
        root_hash: SequenceHash,
        max_entries: usize,
    ) -> Result<Self, HybridEventError> {
        if hash_block_size == 0 {
            return Err(HybridEventError::InvalidLayout);
        }
        Self::with_keys(
            target,
            groups,
            max_entries,
            BlockKeyGenerator::from_root_hash(hash_block_size as usize, root_hash),
        )
    }

    fn with_keys(
        target: RoutableTarget,
        groups: Vec<HybridEventGroup>,
        max_entries: usize,
        keys: BlockKeyGenerator,
    ) -> Result<Self, HybridEventError> {
        let mut ids = HashSet::new();
        if groups.is_empty()
            || groups.len() > 64
            || max_entries == 0
            || groups.iter().any(|group| {
                group.block_size == 0
                    || !(group.block_size as usize).is_multiple_of(keys.block_size())
                    || !ids.insert(group.group_idx)
            })
        {
            return Err(HybridEventError::InvalidLayout);
        }
        Ok(Self {
            target,
            groups,
            keys,
            max_entries,
            state: RwLock::new(State {
                next_seq: None,
                visible: false,
                claims: HashSet::new(),
            }),
        })
    }

    pub fn target(&self) -> &RoutableTarget {
        &self.target
    }

    /// Begin tracking from an externally established empty-cache barrier.
    /// This is not a snapshot or replay implementation. Never call it merely
    /// because a subscription started or a gap occurred.
    pub fn begin_empty(&self, next_seq: u64) {
        let mut state = self.state.write();
        state.claims.clear();
        state.next_seq = Some(next_seq);
        state.visible = true;
    }

    /// Start a full replay from an externally verified empty origin. Queries
    /// remain cold until the caller verifies a continuous replay watermark.
    pub fn begin_replay(&self, next_seq: u64) {
        let mut state = self.state.write();
        state.claims.clear();
        state.next_seq = Some(next_seq);
        state.visible = false;
    }

    /// Publish a complete replay atomically. The caller must obtain the
    /// watermark from the source's replay/live handoff, not infer it locally.
    pub fn finish_replay(&self, expected_next_seq: u64) -> Result<(), HybridEventError> {
        let mut state = self.state.write();
        if state.visible || state.next_seq != Some(expected_next_seq) {
            return Self::fail(&mut state, HybridEventError::Sequence);
        }
        state.visible = true;
        Ok(())
    }

    /// Discard all claims on a gap, restart, lease retirement or lost event.
    pub fn invalidate(&self) {
        let mut state = self.state.write();
        state.claims.clear();
        state.next_seq = None;
        state.visible = false;
    }

    /// Decode a wire payload before applying it. Corrupt payloads invalidate
    /// all groups, including claims from previously accepted batches.
    ///
    pub fn apply_payload(&self, seq: u64, payload: &[u8]) -> Result<(), HybridEventError> {
        let batch = match crate::kv_index::decode_batch(payload) {
            Ok(batch) => batch,
            Err(error) => {
                tracing::warn!(
                    error,
                    bytes = payload.len(),
                    "hybrid payload did not decode"
                );
                self.invalidate();
                return Err(HybridEventError::InvalidBatch);
            }
        };
        self.apply_batch(seq, &batch)
    }

    /// Apply an ordered batch atomically. A sequence mismatch, including a
    /// duplicate, invalidates the view: without an independently verified
    /// publisher epoch, a repeated sequence can also mean a restart.
    pub fn apply_batch(&self, seq: u64, batch: &KVEventBatch) -> Result<(), HybridEventError> {
        let mut state = self.state.write();
        let Some(next_seq) = state.next_seq else {
            return Err(HybridEventError::Untrusted);
        };
        if seq != next_seq || seq == u64::MAX {
            return Self::fail(&mut state, HybridEventError::Sequence);
        }
        if batch.data_parallel_rank != Some(self.target.dp_rank)
            || !batch.ts.is_finite()
            || batch.ts < 0.0
        {
            return Self::fail(&mut state, HybridEventError::InvalidBatch);
        }
        for event in &batch.events {
            let result = match event {
                KVEvent::BlockStored(store) => {
                    // An engine may publish several page sizes for one group
                    // index. Only the declared granularity is addressable by
                    // the layout; the other pages cover the same tokens and
                    // add no boundary information.
                    let Some(group) = self
                        .groups
                        .iter()
                        .find(|group| store.group_idx == Some(group.group_idx))
                    else {
                        return Self::fail(&mut state, HybridEventError::InvalidBatch);
                    };
                    if store.block_size != group.block_size {
                        tracing::debug!(
                            group_idx = group.group_idx,
                            declared = group.block_size,
                            published = store.block_size,
                            "hybrid group published a secondary page size"
                        );
                        continue;
                    }
                    if store.medium.as_deref() != Some("GPU")
                        || store.locality.as_deref().is_some_and(|s| s != "LOCAL")
                        || store.ownership.is_some()
                        || store.lora_id.is_some()
                        || store.lora_name.is_some()
                        || !event_kind_matches(group.kind, store)
                        || !stored_shape_matches(group.kind, store)
                        || store.extra_keys.as_ref().is_some_and(|keys| {
                            keys.len() != store.block_hashes.len()
                                || keys.iter().any(|key| !key.is_nil())
                        })
                        || store
                            .parent_block_hash
                            .as_ref()
                            .is_some_and(|hash| sequence_hash(hash).is_none())
                    {
                        Err(HybridEventError::InvalidBatch)
                    } else {
                        let Some(keys) = store
                            .block_hashes
                            .iter()
                            .map(sequence_hash)
                            .collect::<Option<Vec<_>>>()
                        else {
                            return Self::fail(&mut state, HybridEventError::InvalidBatch);
                        };
                        let group_idx = group.group_idx;
                        for hash in keys {
                            let key = (group_idx, hash);
                            if !state.claims.contains(&key)
                                && state.claims.len() >= self.max_entries
                            {
                                return Self::fail(&mut state, HybridEventError::Capacity);
                            }
                            state.claims.insert(key);
                        }
                        Ok(())
                    }
                }
                KVEvent::BlockRemoved(remove) => {
                    if remove.medium.as_deref() != Some("GPU")
                        || remove.locality.as_deref().is_some_and(|s| s != "LOCAL")
                        || remove.ownership.is_some()
                        || !self
                            .groups
                            .iter()
                            .any(|group| remove.group_idx == Some(group.group_idx))
                        || remove.block_hashes.is_empty()
                    {
                        Err(HybridEventError::InvalidBatch)
                    } else {
                        let group_idx = remove.group_idx.expect("validated group");
                        for hash in &remove.block_hashes {
                            let Some(hash) = sequence_hash(hash) else {
                                return Self::fail(&mut state, HybridEventError::InvalidBatch);
                            };
                            state.claims.remove(&(group_idx, hash));
                        }
                        Ok(())
                    }
                }
                KVEvent::AllBlocksCleared(_) => {
                    state.claims.clear();
                    Ok(())
                }
            };
            if let Err(error) = result {
                return Self::fail(&mut state, error);
            }
        }
        state.next_seq = Some(seq + 1);
        Ok(())
    }

    /// Calculate one consistent local snapshot. Zero means a miss or an
    /// untrusted source.
    pub fn reusable_tokens(&self, tokens: &[u32], max_reusable_tokens: usize) -> u64 {
        let hashes = self.keys.generate_block_keys(tokens);
        let state = self.state.read();
        if state.next_seq.is_none() || !state.visible {
            return 0;
        }
        let matches: Vec<Vec<TierMatch>> = self
            .groups
            .iter()
            .map(|group| {
                let stride = group.block_size as usize / self.keys.block_size();
                let mut segments: Vec<(u32, u32)> = Vec::new();
                for (position, hash) in hashes.iter().skip(stride - 1).step_by(stride).enumerate() {
                    let Ok(end) = u32::try_from(position.saturating_add(1)) else {
                        break;
                    };
                    if !state.claims.contains(&(group.group_idx, *hash)) {
                        continue;
                    }
                    let start = end - 1;
                    if let Some(segment) = segments.last_mut().filter(|segment| segment.1 == start)
                    {
                        segment.1 = end;
                    } else {
                        segments.push((start, end));
                    }
                }
                vec![TierMatch {
                    target: self.target.clone(),
                    tier: StorageTier::Device,
                    locality: Locality::Local,
                    segments,
                }]
            })
            .collect();
        let views: Vec<_> = self
            .groups
            .iter()
            .zip(&matches)
            .map(|(group, matches)| HybridGroupView {
                group_idx: group.group_idx,
                block_size: group.block_size,
                kind: group.kind,
                matches,
            })
            .collect();
        common_reusable_tokens(
            &self.target,
            &views,
            tokens.len().saturating_sub(1).min(max_reusable_tokens) as u64,
        )
    }

    fn fail<T>(state: &mut State, error: HybridEventError) -> Result<T, HybridEventError> {
        state.claims.clear();
        state.next_seq = None;
        state.visible = false;
        Err(error)
    }
}

fn sequence_hash(hash: &ExternalBlockHash) -> Option<SequenceHash> {
    match hash {
        ExternalBlockHash::Bytes(bytes) => bytes.as_slice().try_into().ok(),
        ExternalBlockHash::Int(_) => None,
    }
}

/// A group accepts only the event shape its verified restore rule describes.
fn event_kind_matches(kind: HybridGroupKind, store: &BlockStored) -> bool {
    let spec = store.kv_cache_spec_kind.as_deref();
    match kind {
        HybridGroupKind::FullAttention => {
            spec == Some("full_attention") && store.kv_cache_spec_sliding_window.is_none()
        }
        HybridGroupKind::Checkpoint => {
            spec == Some("mamba") && store.kv_cache_spec_sliding_window.is_none()
        }
    }
}

/// Checkpoint groups may publish fewer hashes than the blocks the token range
/// spans; attention groups publish one per block.
fn stored_shape_matches(kind: HybridGroupKind, store: &BlockStored) -> bool {
    let block_size = store.block_size as usize;
    if block_size == 0 || store.token_ids.is_empty() {
        return false;
    }
    if !store.token_ids.len().is_multiple_of(block_size) {
        return false;
    }
    let blocks = store.token_ids.len() / block_size;
    match kind {
        HybridGroupKind::Checkpoint => store.block_hashes.len() <= blocks,
        _ => store.block_hashes.len() == blocks,
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
pub(crate) mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::kv_index::wire::{BlockRemoved, BlockStored};

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
}
