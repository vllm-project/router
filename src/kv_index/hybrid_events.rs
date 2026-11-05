//! Atomic, fail-closed hybrid event state for one worker rank.

use std::collections::HashSet;

use parking_lot::RwLock;

use super::hybrid::{align_down, common_reusable_tokens, HybridGroupKind, HybridGroupView};
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
    /// Layout-declared restore-time recomputation window (vLLM
    /// `prefix_replay_tokens`). It applies to the whole layout: a group that
    /// publishes no events can still make this window mandatory.
    replay_tokens: u32,
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
                    // Compressed state covers `tokens_per_state` tokens, so the
                    // group boundary has to be a multiple of that ratio.
                    || match group.kind {
                        HybridGroupKind::SlidingWindow { window_tokens } => window_tokens == 0,
                        HybridGroupKind::CompressedFullAttention { tokens_per_state } => {
                            tokens_per_state == 0
                                || !group.block_size.is_multiple_of(tokens_per_state)
                        }
                        _ => false,
                    }
            })
        {
            return Err(HybridEventError::InvalidLayout);
        }
        Ok(Self {
            target,
            groups,
            keys,
            replay_tokens: 0,
            max_entries,
            state: RwLock::new(State {
                next_seq: None,
                visible: false,
                claims: HashSet::new(),
            }),
        })
    }

    /// Declare the layout's restore-time recomputation window
    /// (`prefix_replay_tokens`); it lowers the credit for the whole layout.
    pub fn with_replay_tokens(mut self, replay_tokens: u32) -> Result<Self, HybridEventError> {
        let hash_block_size = self.keys.block_size();
        if replay_tokens > 0 && (hash_block_size == 0 || replay_tokens > u32::MAX / 2) {
            return Err(HybridEventError::InvalidLayout);
        }
        self.replay_tokens = replay_tokens;
        Ok(self)
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
                    // index (V4-class compressors store 4-token pages next to
                    // the group's own pages). Only the declared granularity is
                    // addressable by the layout; the other pages cover the same
                    // tokens and add no boundary information.
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
                        let Ok(keys) = self.stored_keys(group, store) else {
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
        let boundary = common_reusable_tokens(
            &self.target,
            &views,
            tokens.len().saturating_sub(1).min(max_reusable_tokens) as u64,
        );
        // The engine replays the layout's trailing window before prefilling the
        // rest, and rounds the remaining hit down to a block. Charging the same
        // way keeps Router's estimate from exceeding the engine's own cache hit.
        let credited = boundary.saturating_sub(u64::from(self.replay_tokens));
        align_down(credited, self.keys.block_size() as u32)
    }

    /// Every block a store proves. A sliding group publishes only its tail;
    /// the chain from the parent digest recovers the rest, and is used only
    /// when its tail matches the published hashes.
    fn stored_keys(
        &self,
        group: &HybridEventGroup,
        store: &BlockStored,
    ) -> Result<Vec<SequenceHash>, ()> {
        let published = store
            .block_hashes
            .iter()
            .map(|hash| sequence_hash(hash).ok_or(()))
            .collect::<Result<Vec<_>, ()>>()?;
        if !matches!(group.kind, HybridGroupKind::SlidingWindow { .. })
            || store.token_ids.is_empty()
            || !store
                .token_ids
                .len()
                .is_multiple_of(group.block_size as usize)
        {
            return Ok(published);
        }
        let parent = match store.parent_block_hash.as_ref() {
            None => self.keys.none_hash(),
            Some(hash) => sequence_hash(hash).ok_or(())?,
        };
        let expanded = self.keys.keys_after(parent, &store.token_ids);
        // Only the group's own page boundaries are addressable: `reusable_tokens`
        // looks up nothing else and the engine's removals name nothing else, so
        // claiming the intermediate chain steps would leave entries behind on
        // every store/evict cycle until the capacity limit revoked the rank.
        let stride = (group.block_size as usize / self.keys.block_size()).max(1);
        let boundaries: Vec<SequenceHash> = expanded
            .iter()
            .skip(stride - 1)
            .step_by(stride)
            .copied()
            .collect();
        if published.is_empty() {
            return Ok(boundaries);
        }
        if boundaries.ends_with(&published) {
            Ok(boundaries)
        } else {
            Ok(published)
        }
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
/// The sliding-window size is part of the identity: the same model served with
/// a different window is a different layout.
fn event_kind_matches(kind: HybridGroupKind, store: &BlockStored) -> bool {
    let spec = store.kv_cache_spec_kind.as_deref();
    match kind {
        HybridGroupKind::FullAttention => {
            spec == Some("full_attention") && store.kv_cache_spec_sliding_window.is_none()
        }
        HybridGroupKind::CompressedFullAttention { .. } => {
            spec == Some("mla_attention") && store.kv_cache_spec_sliding_window.is_none()
        }
        HybridGroupKind::Checkpoint => {
            spec == Some("mamba") && store.kv_cache_spec_sliding_window.is_none()
        }
        HybridGroupKind::SlidingWindow { window_tokens } => {
            matches!(spec, Some("sliding_window" | "sliding_window_mla"))
                && store.kv_cache_spec_sliding_window.map(u64::from) == Some(window_tokens)
        }
    }
}

/// Checkpoint and sliding groups may publish fewer hashes than the blocks the
/// token range spans; attention groups publish one per block.
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
        HybridGroupKind::Checkpoint | HybridGroupKind::SlidingWindow { .. } => {
            store.block_hashes.len() <= blocks
        }
        _ => store.block_hashes.len() == blocks,
    }
}

// Install this provider only when the engine contract verifies that recomputing
// the last token is a sufficient bound (no unmodeled lookahead/replay rules).
impl crate::policies::TieredMatchProvider for HybridEventIndex {
    fn find_tiered_matches(&self, _query: &super::MatchQuery) -> Vec<TierMatch> {
        // A single group's dense lookup is never sufficient for a hybrid hit.
        Vec::new()
    }

    fn find_token_matches(
        &self,
        tokens: &[u32],
        block_size: u32,
        lora_name: Option<&str>,
        dp_rank: u32,
    ) -> Option<Vec<TierMatch>> {
        // Events, ownership and history are bound to one rank. A hit computed
        // from another rank's stream is never evidence for this one.
        if lora_name.is_some() || block_size == 0 || dp_rank != self.target.dp_rank {
            return Some(Vec::new());
        }
        let tokens = self.reusable_tokens(tokens, usize::MAX);
        if tokens == 0 || !tokens.is_multiple_of(u64::from(block_size)) {
            return Some(Vec::new());
        }
        let Ok(blocks) = u32::try_from(tokens / u64::from(block_size)) else {
            return Some(Vec::new());
        };
        Some(vec![TierMatch {
            target: self.target.clone(),
            tier: StorageTier::Device,
            locality: Locality::Local,
            segments: vec![(0, blocks)],
        }])
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
        let tail = |block: usize, hold: usize| -> Vec<SequenceHash> {
            let hashes = view(block);
            hashes[hashes.len() - hold..].to_vec()
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
        let batch = KVEventBatch {
            ts: 1.0,
            data_parallel_rank: Some(0),
            events: vec![
                store(
                    groups[0],
                    "sliding_window_mla",
                    Some(128),
                    tail(64, 2),
                    64,
                    1024,
                ),
                store(
                    groups[1],
                    "sliding_window_mla",
                    Some(128),
                    tail(64, 2),
                    64,
                    1024,
                ),
                store(groups[2], "mla_attention", None, view(256), 256, 1024),
                // The same group also publishes compressor pages it stores on
                // their own; they carry no boundary the group does not already
                // report.
                store(groups[2], "mla_attention", None, vec![keys[0]], 4, 4),
                store(
                    groups[3],
                    "sliding_window_mla",
                    Some(8),
                    tail(4, 2),
                    4,
                    1024,
                ),
                store(
                    groups[4],
                    "sliding_window_mla",
                    Some(128),
                    tail(8, 16),
                    8,
                    1024,
                ),
            ],
        };
        assert_eq!(index.apply_batch(0, &batch), Ok(()));
        // The compressed group restores three 256-token blocks below the
        // last-token bound, and the engine keeps that smallest hit even though
        // the sliding tails sit at the end of the prefix.
        assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 768);
    }

    /// A sliding claim must stay inside what the engine can later remove. The
    /// store spans 1,024 tokens at a 64-token group page, so the chain has 256
    /// four-token steps but only 16 of them are page boundaries: querying and
    /// removing address nothing else, and retaining the intermediates would
    /// exhaust the capacity limit over repeated store/evict cycles.
    #[test]
    fn sliding_claims_only_the_page_boundaries_its_removals_name() {
        let tokens: Vec<u32> = (0..1024).map(|i| 100 + i % 100).collect();
        let keys = BlockKeyGenerator::new(4, 0).generate_block_keys(&tokens);
        let group = HybridEventGroup {
            group_idx: 0,
            block_size: 64,
            kind: HybridGroupKind::SlidingWindow { window_tokens: 128 },
        };
        let tail: Vec<SequenceHash> = keys.iter().skip(15).step_by(16).copied().collect();
        let tail = &tail[tail.len() - 2..];
        let mut store = gpu_store(0, "sliding_window_mla", 64, &tokens[..1024], tail);
        store.kv_cache_spec_sliding_window = Some(128);
        let batch = KVEventBatch {
            ts: 1.0,
            data_parallel_rank: Some(0),
            events: vec![KVEvent::BlockStored(store)],
        };
        for (capacity, expected) in [(16, Ok(())), (15, Err(HybridEventError::Capacity))] {
            let index = HybridEventIndex::new(target(), vec![group], 4, 0, capacity).unwrap();
            index.begin_empty(0);
            assert_eq!(index.apply_batch(0, &batch), expected);
        }
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
