//! Atomic, fail-closed hybrid event state for one worker rank.
//!
//! This is not connected to runtime worker discovery or serving. Before use,
//! the caller must verify the complete group layout, cache identity, hash
//! algorithm and seed, publisher/rank binding, and engine restore rules. A routing decision also
//! needs the existing subscriber lease and freshness checks.

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
    /// The layout and sha256_cbor seed must come from a verified capability,
    /// never inferred from the model name or from a partial event stream.
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

    /// Declare the layout's restore-time recomputation window.
    ///
    /// vLLM rewinds a prefix hit by this many tokens (rounded down to a block)
    /// before prefilling the rest, and requires every replay group to agree on
    /// one window. A non-zero window therefore reduces the credited prefix for
    /// the complete layout, including groups that never emit block events.
    pub fn with_replay_tokens(mut self, replay_tokens: u32) -> Result<Self, HybridEventError> {
        let hash_block_size = self.keys.block_size();
        if replay_tokens > 0 && (hash_block_size == 0 || replay_tokens > u32::MAX / 2) {
            return Err(HybridEventError::InvalidLayout);
        }
        self.replay_tokens = replay_tokens;
        Ok(self)
    }

    pub fn replay_tokens(&self) -> u32 {
        self.replay_tokens
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
    /// Two wire shapes are accepted: a `(ts, events, rank)` batch, and a bare
    /// single event as published by engines that encode one event per message.
    /// A bare event carries no rank, so it is attributed to the rank whose
    /// endpoint delivered it — never to whichever rank happens to be first.
    pub fn apply_payload(&self, seq: u64, payload: &[u8]) -> Result<(), HybridEventError> {
        let batch = match crate::kv_index::decode_batch(payload) {
            Ok(batch) => batch,
            Err(batch_error) => match crate::kv_index::decode_event_payload(payload) {
                Ok(event) => KVEventBatch {
                    ts: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|elapsed| elapsed.as_secs_f64())
                        .unwrap_or(0.0),
                    events: vec![event],
                    data_parallel_rank: Some(self.target.dp_rank),
                },
                Err(event_error) => {
                    // The two shapes fail for different reasons; naming both is
                    // the only way to diagnose a publisher contract change.
                    tracing::warn!(
                        batch_error,
                        event_error,
                        bytes = payload.len(),
                        "hybrid payload did not decode"
                    );
                    self.invalidate();
                    return Err(HybridEventError::InvalidBatch);
                }
            },
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
                    let group = self.groups.iter().find(|group| {
                        store.group_idx == Some(group.group_idx)
                            && store.block_size == group.block_size
                            && event_kind_matches(group.kind, store)
                    });
                    if group.is_none()
                        || store.medium.as_deref() != Some("GPU")
                        || store.locality.as_deref().is_some_and(|s| s != "LOCAL")
                        || store.ownership.is_some()
                        || store.lora_id.is_some()
                        || store.lora_name.is_some()
                        || !group.is_some_and(|group| stored_shape_matches(group.kind, store))
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
                        let group_idx = store.group_idx.expect("validated group");
                        for hash in &store.block_hashes {
                            let Some(hash) = sequence_hash(hash) else {
                                return Self::fail(&mut state, HybridEventError::InvalidBatch);
                            };
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

    /// Calculate one consistent local snapshot. The caller must verify that
    /// the input uses this index's exact identity and hash contract, with no
    /// unmodeled multimodal data, adapter or cache salt. The bound must account
    /// for engine-specific limits; unsupported speculative/lookahead restore
    /// semantics must receive no credit. Zero can mean a miss or untrusted
    /// source; neither grants routing credit.
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

/// A checkpoint event may describe a whole cached chunk while publishing only
/// the states that survived sparse retention, so its hash count is bounded by
/// the token range instead of equal to it. Attention groups publish one hash
/// per block.
fn stored_shape_matches(kind: HybridGroupKind, store: &BlockStored) -> bool {
    let block_size = store.block_size as usize;
    if block_size == 0 || store.token_ids.is_empty() {
        return false;
    }
    if kind == HybridGroupKind::Checkpoint {
        store.token_ids.len().is_multiple_of(block_size)
            && store.block_hashes.len() <= store.token_ids.len() / block_size
    } else {
        store.block_hashes.len().checked_mul(block_size) == Some(store.token_ids.len())
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
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::kv_index::wire::{BlockRemoved, BlockStored};

    fn hex_root(value: &str) -> [u8; 32] {
        assert_eq!(value.len(), 64);
        let mut root = [0; 32];
        for (out, pair) in root.iter_mut().zip(value.as_bytes().chunks_exact(2)) {
            let nibble = |byte: u8| (byte as char).to_digit(16).unwrap() as u8;
            *out = (nibble(pair[0]) << 4) | nibble(pair[1]);
        }
        root
    }

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

    fn qwen_batch() -> KVEventBatch {
        let tokens = prompt();
        let hash = BlockKeyGenerator::new(528, 0).generate_block_keys(&tokens)[0];
        // Captured from the Qwen3.5-4B vLLM 0.28.0 H20 probe with full-width
        // sha256_cbor events, PYTHONHASHSEED=0 and the synthetic prompt above.
        assert_eq!(
            hash,
            [
                0x62, 0x7d, 0xd1, 0x45, 0x45, 0x65, 0x2f, 0x6a, 0x2d, 0xe0, 0xae, 0x29, 0x21, 0xe9,
                0xed, 0x23, 0x3f, 0xe3, 0x64, 0x87, 0xe1, 0x52, 0xa7, 0xc2, 0xd7, 0x78, 0x2c, 0xeb,
                0xf5, 0xc2, 0x41, 0x81,
            ]
        );
        let events = groups()
            .iter()
            .map(|group| {
                KVEvent::BlockStored(BlockStored {
                    block_hashes: vec![ExternalBlockHash::Bytes(hash.to_vec())],
                    parent_block_hash: None,
                    token_ids: tokens[..528].to_vec(),
                    block_size: 528,
                    lora_id: None,
                    medium: Some("GPU".into()),
                    lora_name: None,
                    extra_keys: Some(vec![rmpv::Value::Nil]),
                    group_idx: Some(group.group_idx),
                    kv_cache_spec_kind: Some(
                        if group.group_idx == 3 {
                            "full_attention"
                        } else {
                            "mamba"
                        }
                        .into(),
                    ),
                    kv_cache_spec_sliding_window: None,
                    locality: None,
                    ownership: None,
                    session_id: None,
                })
            })
            .collect();
        KVEventBatch {
            ts: 1.0,
            events,
            data_parallel_rank: Some(0),
        }
    }

    fn index(limit: usize) -> HybridEventIndex {
        HybridEventIndex::new(target(), groups(), 528, 0, limit).unwrap()
    }

    #[test]
    fn qwen35_real_msgpack_batch_matches_engine_reuse() {
        // Captured from H20 vLLM 0.28.0 using the synthetic prompt from
        // prompt(), sha256_cbor, seed 0 and full-width event hashes.
        let payload = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/rust/fixtures/kv_cbor/qwen35_vllm028.msgpack"
        ));
        let batch = crate::kv_index::decode_batch(payload).unwrap();
        assert_eq!(batch.data_parallel_rank, Some(0));
        assert_eq!(batch.events.len(), 4);
        let index = index(8);
        index.begin_empty(0);
        assert_eq!(index.apply_payload(0, payload), Ok(()));
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
        assert_eq!(
            index.apply_payload(1, &[0xc1]),
            Err(HybridEventError::InvalidBatch)
        );
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
    }

    #[test]
    fn engine_root_hash_reproduces_the_qwen_sequence_without_seed() {
        let payload = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/rust/fixtures/kv_cbor/qwen35_vllm028.msgpack"
        ));
        let root = BlockKeyGenerator::new(528, 0).none_hash();
        let index = HybridEventIndex::new_with_root_hash(target(), groups(), 528, root, 8).unwrap();
        index.begin_empty(0);
        index.apply_payload(0, payload).unwrap();
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
        let wrong =
            HybridEventIndex::new_with_root_hash(target(), groups(), 528, [0; 32], 8).unwrap();
        wrong.begin_empty(0);
        wrong.apply_payload(0, payload).unwrap();
        assert_eq!(wrong.reusable_tokens(&prompt(), usize::MAX), 0);
    }

    /// vLLM 0.28 publishes one event per message; the rank comes from the
    /// endpoint, not from the payload.
    #[test]
    fn bare_event_payload_is_credited_to_the_bound_rank() {
        let batch = qwen_batch();
        let mut credited = 0;
        let index = index(8);
        index.begin_empty(0);
        for (seq, event) in batch.events.iter().enumerate() {
            let payload = rmp_serde::to_vec_named(event).unwrap();
            index.apply_payload(seq as u64, &payload).unwrap();
            credited = index.reusable_tokens(&prompt(), usize::MAX);
        }
        assert_eq!(credited, 528);
        // The same bytes on a rank-1 index are that rank's events, not rank 0's.
        let other = HybridEventIndex::new(
            RoutableTarget {
                dp_rank: 1,
                ..target()
            },
            groups(),
            528,
            0,
            8,
        )
        .unwrap();
        other.begin_empty(0);
        let payload = rmp_serde::to_vec_named(&batch.events[0]).unwrap();
        other.apply_payload(0, &payload).unwrap();
        assert_eq!(other.reusable_tokens(&prompt(), usize::MAX), 0);
    }

    /// Captured from vLLM 0.29 with `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0`:
    /// four `BlockStored` events (three mamba groups plus full attention) with
    /// 32-byte block hashes and `data_parallel_rank: 0`.
    ///
    /// This engine ran its default `--prefix-caching-hash` (`sha256`, which
    /// pickles the tuple) while the declared layout says `sha256_cbor`. Both
    /// produce 32-byte digests, so only recomputing the digest exposes the
    /// disagreement: the batch is a valid hybrid layout with a foreign hash
    /// contract and must credit nothing.
    #[test]
    fn foreign_hash_contract_is_credited_as_nothing() {
        let payload = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/rust/fixtures/kv_cbor/qwen35_vllm029.msgpack"
        ));
        let prompt: Vec<u32> = serde_json::from_slice(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/rust/fixtures/kv_cbor/qwen35_vllm029_prompt.json"
        )))
        .unwrap();
        assert_eq!(prompt.len(), 601);
        let root = hex_root("9bd96a485ad84efdafb72ee48a1d7a69bcead0f8f0433173941b276b9581eef0");
        let index = HybridEventIndex::new_with_root_hash(target(), groups(), 528, root, 8).unwrap();
        index.begin_empty(0);
        index
            .apply_payload(0, payload)
            .expect("the batch is structurally valid");
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 0);
    }

    /// Captured from vLLM 0.29 with `--prefix-caching-hash sha256_cbor` and
    /// `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0`. The engine's `NONE_HASH` is
    /// `sha256_cbor("vllm-none-hash")`, so the layout must declare that root:
    /// the seed-zero digest is a different, incompatible sequence.
    #[test]
    fn cbor_hashed_engine_credits_the_replayed_prefix() {
        let payload = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/rust/fixtures/kv_cbor/qwen35_vllm029_cbor.msgpack"
        ));
        let prompt: Vec<u32> = serde_json::from_slice(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/rust/fixtures/kv_cbor/qwen35_vllm029_prompt.json"
        )))
        .unwrap();
        let root = hex_root("9bd96a485ad84efdafb72ee48a1d7a69bcead0f8f0433173941b276b9581eef0");
        let index = HybridEventIndex::new_with_root_hash(target(), groups(), 528, root, 8).unwrap();
        index.begin_empty(0);
        index
            .apply_payload(0, payload)
            .expect("captured batch applies");
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 528);
        // A wrong root is a different sequence, not a cosmetic field.
        let wrong =
            HybridEventIndex::new_with_root_hash(target(), groups(), 528, [0; 32], 8).unwrap();
        wrong.begin_empty(0);
        wrong.apply_payload(0, payload).unwrap();
        assert_eq!(wrong.reusable_tokens(&prompt, usize::MAX), 0);
    }

    #[test]
    fn decoded_qwen_layout_requires_all_groups_and_tracks_removal() {
        let original = qwen_batch();
        let payload =
            rmp_serde::to_vec_named(&(original.ts, original.events, Some(0_u32))).unwrap();
        let batch = crate::kv_index::decode_batch(&payload).unwrap();
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
        let stored = |group_idx, size, token_ids: &[u32], block_hashes, parent| {
            KVEvent::BlockStored(BlockStored {
                block_hashes,
                parent_block_hash: parent,
                token_ids: token_ids.to_vec(),
                block_size: size,
                lora_id: None,
                medium: Some("GPU".into()),
                lora_name: None,
                extra_keys: None,
                group_idx: Some(group_idx),
                kv_cache_spec_kind: Some(
                    if group_idx == 0 {
                        "full_attention"
                    } else {
                        "mamba"
                    }
                    .into(),
                ),
                kv_cache_spec_sliding_window: None,
                locality: None,
                ownership: None,
                session_id: None,
            })
        };
        let batch = KVEventBatch {
            ts: 1.0,
            data_parallel_rank: Some(0),
            events: vec![
                stored(
                    0,
                    4,
                    &tokens[..16],
                    hashes
                        .iter()
                        .map(|h| ExternalBlockHash::Bytes(h.to_vec()))
                        .collect(),
                    None,
                ),
                stored(
                    1,
                    8,
                    &tokens[8..16],
                    vec![ExternalBlockHash::Bytes(hashes[3].to_vec())],
                    Some(ExternalBlockHash::Bytes(hashes[1].to_vec())),
                ),
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
                KVEvent::BlockStored(BlockStored {
                    block_hashes: vec![ExternalBlockHash::Bytes(hash.to_vec())],
                    parent_block_hash: None,
                    token_ids: tokens[..320].to_vec(),
                    block_size: 320,
                    lora_id: None,
                    medium: Some("GPU".into()),
                    lora_name: None,
                    extra_keys: Some(vec![rmpv::Value::Nil]),
                    group_idx: Some(group.group_idx),
                    kv_cache_spec_kind: Some(
                        kind_override
                            .unwrap_or(if compressed { "mla_attention" } else { "mamba" })
                            .into(),
                    ),
                    kv_cache_spec_sliding_window: None,
                    locality: None,
                    ownership: None,
                    session_id: None,
                })
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
                    KVEvent::BlockStored(BlockStored {
                        block_hashes: (if attention {
                            &keys[..count]
                        } else {
                            &keys[6 - count..6]
                        })
                        .iter()
                        .map(|hash| ExternalBlockHash::Bytes(hash.to_vec()))
                        .collect(),
                        parent_block_hash: None,
                        token_ids: tokens[..1920].to_vec(),
                        block_size: 320,
                        lora_id: None,
                        medium: Some("GPU".into()),
                        lora_name: None,
                        extra_keys: Some(vec![rmpv::Value::Nil; count]),
                        group_idx: Some(group.group_idx),
                        kv_cache_spec_kind: Some(
                            if attention { "mla_attention" } else { "mamba" }.into(),
                        ),
                        kv_cache_spec_sliding_window: None,
                        locality: None,
                        ownership: None,
                        session_id: None,
                    })
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
        let store = |group_idx: u32, hashes: Vec<ExternalBlockHash>, tokens_len: usize| {
            let n = hashes.len();
            KVEvent::BlockStored(BlockStored {
                block_hashes: hashes,
                parent_block_hash: None,
                token_ids: tokens[..tokens_len].to_vec(),
                block_size: 576,
                lora_id: None,
                medium: Some("GPU".into()),
                lora_name: None,
                extra_keys: Some(vec![rmpv::Value::Nil; n]),
                group_idx: Some(group_idx),
                kv_cache_spec_kind: Some(
                    if group_idx == 3 {
                        "full_attention"
                    } else {
                        "mamba"
                    }
                    .into(),
                ),
                kv_cache_spec_sliding_window: None,
                locality: None,
                ownership: None,
                session_id: None,
            })
        };
        let h = |i: usize| ExternalBlockHash::Bytes(keys[i].to_vec());
        let index = HybridEventIndex::new(target(), groups, 576, 0, 64).unwrap();
        index.begin_empty(0);
        let first = KVEventBatch {
            ts: 1.0,
            data_parallel_rank: Some(0),
            events: vec![
                store(0, vec![h(2)], 1728),
                store(1, vec![h(2)], 1728),
                store(2, vec![h(2)], 1728),
                store(3, vec![h(0), h(1), h(2)], 1728),
            ],
        };
        assert_eq!(index.apply_batch(0, &first), Ok(()));
        assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 1728);
        // Re-run of the same prompt: checkpoints publish nothing new.
        let rerun = KVEventBatch {
            ts: 2.0,
            data_parallel_rank: Some(0),
            events: vec![
                store(0, vec![], 576),
                store(1, vec![], 576),
                store(2, vec![], 576),
                store(3, vec![h(0)], 576),
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
        KVEvent::BlockStored(BlockStored {
            block_hashes: keys[from..to]
                .iter()
                .map(|key| ExternalBlockHash::Bytes(key.to_vec()))
                .collect(),
            parent_block_hash: None,
            token_ids: tokens[from * block..to * block].to_vec(),
            block_size: block as u32,
            lora_id: None,
            medium: Some("GPU".into()),
            lora_name: None,
            extra_keys: Some(vec![rmpv::Value::Nil; to - from]),
            group_idx: Some(group.group_idx),
            kv_cache_spec_kind: Some(spec.into()),
            kv_cache_spec_sliding_window: window,
            locality: None,
            ownership: None,
            session_id: None,
        })
    }

    #[test]
    fn layout_replay_window_rounds_the_credited_prefix_down() {
        // vLLM rewinds a prefix hit by the layout's replay window and then
        // rounds to a block. The window belongs to the layout, so it reduces
        // credit even though no group event mentions it.
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
                kind: HybridGroupKind::SlidingWindow { window_tokens: 32 },
            },
        ];
        let batch = KVEventBatch {
            ts: 1.0,
            data_parallel_rank: Some(0),
            events: vec![
                blocks(
                    groups[0],
                    "full_attention",
                    None,
                    &keys,
                    64,
                    &tokens,
                    0,
                    keys.len(),
                ),
                blocks(
                    groups[1],
                    "sliding_window_mla",
                    Some(32),
                    &keys,
                    64,
                    &tokens,
                    0,
                    keys.len(),
                ),
            ],
        };
        let replay = HybridEventIndex::new(target(), groups.clone(), 64, 0, 64)
            .unwrap()
            .with_replay_tokens(128)
            .unwrap();
        replay.begin_empty(0);
        assert_eq!(replay.apply_batch(0, &batch), Ok(()));
        // 1023 tokens bound the hit to block 15 (960); a 128-token replay
        // leaves 832, already block aligned.
        assert_eq!(replay.reusable_tokens(&tokens, usize::MAX), 832);
        assert_eq!(replay.replay_tokens(), 128);

        let plain = HybridEventIndex::new(target(), groups, 64, 0, 64).unwrap();
        plain.begin_empty(0);
        assert_eq!(plain.apply_batch(0, &batch), Ok(()));
        assert_eq!(plain.reusable_tokens(&tokens, usize::MAX), 960);
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

    #[test]
    fn another_dp_rank_never_answers_for_this_rank() {
        use crate::policies::TieredMatchProvider;

        let index = index(8);
        index.begin_empty(0);
        index
            .apply_payload(
                0,
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/rust/fixtures/kv_cbor/qwen35_vllm028.msgpack"
                )),
            )
            .unwrap();
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
