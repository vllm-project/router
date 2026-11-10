//! Hybrid attention prefix index.
//!
//! `HybridIndex` owns the residency lock and is the only type callers see.
//! Event interpretation lives in `ingest`, cache-group rules in `layout`.

mod ingest;
mod layout;
pub(crate) mod residency;
mod subscriber;

use std::collections::{HashMap, HashSet};

use parking_lot::RwLock;

use self::ingest::sequence_hash;
pub use self::ingest::HybridEventError;
use self::layout::align_down;
pub use self::layout::{
    common_reusable_tokens, HybridAdapterCapability, HybridCapabilities, HybridEventGroup,
    HybridGroupCapability, HybridGroupKind, HybridGroupView, HybridRankCapability,
};
use self::residency::State;
pub use self::subscriber::{
    spawn_zmq, HybridEventStream, HybridStreamError, HybridSubscriberHandle,
};
pub use crate::kv_index::subscriber::HybridSourceLease;

#[allow(unused_imports)]
use self::ingest::HybridEventError as _;
use super::wire::BlockStored;
use super::{Locality, RoutableTarget, StorageTier, TierMatch};
use crate::kv_hash::{BlockKeyGenerator, SequenceHash};

/// A bounded index for one verified identity, worker, rank and incarnation.
/// Stores and removals across all groups share one lock and one sequence.
pub struct HybridEventIndex {
    pub(super) target: RoutableTarget,
    pub(super) groups: Vec<HybridEventGroup>,
    pub(super) keys: BlockKeyGenerator,
    /// Layout-declared restore-time recomputation window (vLLM
    /// `prefix_replay_tokens`). It applies to the whole layout: a group that
    /// publishes no events can still make this window mandatory.
    pub(super) replay_tokens: u32,
    /// LoRA adapters this worker serves, as `(name, path)`. The engine seeds an
    /// adapter's chain with both, so a layout that does not declare an adapter
    /// cannot credit it, and one that declares the wrong path credits nothing.
    pub(super) adapters: Vec<(String, String)>,
    pub(super) max_entries: usize,
    state: RwLock<State>,
}

impl HybridEventIndex {
    /// Test constructor keyed by a `PYTHONHASHSEED` instead of a root digest.
    #[cfg(all(test, feature = "cpu", feature = "ready"))]
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
            adapters: Vec::new(),
            max_entries,
            state: RwLock::new(State {
                next_seq: None,
                visible: false,
                claims: HashMap::new(),
                page_evidence: HashMap::new(),
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

    /// Declare the LoRA adapters this worker serves. Adapter pages are a
    /// separate chain keyed by `(name, path)`; without the declaration their
    /// pages are skipped rather than credited to the base model.
    pub fn with_adapters(mut self, adapters: Vec<(String, String)>) -> Self {
        self.adapters = adapters;
        self
    }

    fn adapter_path(&self, name: &str) -> Option<&str> {
        self.adapters
            .iter()
            .find(|(declared, _)| declared == name)
            .map(|(_, path)| path.as_str())
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

    /// Calculate one consistent local snapshot. Zero means a miss or an
    /// untrusted source.
    pub fn reusable_tokens(&self, tokens: &[u32], max_reusable_tokens: usize) -> u64 {
        self.reusable_tokens_in(None, tokens, max_reusable_tokens)
    }

    /// The credited prefix and the tier it is credited at. A page held in host
    /// memory or on storage is still restorable, but the policy weighs it by the
    /// transfer credit of its tier, so the boundary is reported at the slowest
    /// tier that any group needs to satisfy it.
    fn credit_parts(
        &self,
        namespace: Option<&str>,
        tokens: &[u32],
        max_reusable_tokens: usize,
    ) -> (u64, Option<TierMatch>) {
        {
            let state = self.state.read();
            if state.next_seq.is_none() || !state.visible {
                return (0, None);
            }
        }
        let hashes = match namespace {
            None => self.keys.generate_block_keys(tokens),
            Some(name) => match self.adapter_path(name) {
                Some(path) => self
                    .keys
                    .generate_block_keys_for_adapter(tokens, name, path),
                None => return (0, None),
            },
        };
        let state = self.state.read();
        if state.next_seq.is_none() || !state.visible {
            return (0, None);
        }
        let Some(claims) = state.claims_in(namespace) else {
            return (0, None);
        };
        let page_evidence = state.page_evidence_in(namespace);
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
                    if !claims.contains_key(&(group.group_idx, *hash)) {
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
                page_evidence: page_evidence
                    .is_some_and(|evidence| evidence.contains(&group.group_idx)),
            })
            .collect();
        let boundary = common_reusable_tokens(
            &self.target,
            &views,
            tokens.len().saturating_sub(1).min(max_reusable_tokens) as u64,
        );
        let credited = boundary.saturating_sub(u64::from(self.replay_tokens));
        let credited = align_down(credited, self.keys.block_size() as u32);
        if credited == 0 {
            return (0, None);
        }
        // Report the slowest tier any group needs at the credited boundary: the
        // nearest resident page at or below it is the one that completes it.
        let mut worst = StorageTier::Device;
        for group in &self.groups {
            let stride = group.block_size as usize / self.keys.block_size();
            let blocks = (credited / u64::from(group.block_size)) as usize;
            for position in (0..blocks).rev() {
                let Some(hash) = hashes.get(stride * (position + 1) - 1) else {
                    break;
                };
                if let Some(tier) = claims
                    .get(&(group.group_idx, *hash))
                    .and_then(|resident| resident.best())
                {
                    if tier_rank(tier) > tier_rank(worst) {
                        worst = tier;
                    }
                    break;
                }
            }
        }
        let Ok(blocks) = u32::try_from(credited / u64::from(self.keys.block_size() as u32)) else {
            return (0, None);
        };
        (
            credited,
            Some(TierMatch {
                target: self.target.clone(),
                tier: worst,
                locality: Locality::Local,
                segments: vec![(0, blocks)],
            }),
        )
    }
    fn reusable_tokens_in(
        &self,
        namespace: Option<&str>,
        tokens: &[u32],
        max_reusable_tokens: usize,
    ) -> u64 {
        self.credit_parts(namespace, tokens, max_reusable_tokens).0
    }

    /// Every block a store proves. A sliding group publishes only its tail;
    /// the chain from the parent digest recovers the rest, and is used only
    /// when its tail matches the published hashes.
    fn stored_keys(
        &self,
        _group: &HybridEventGroup,
        store: &BlockStored,
    ) -> Result<Vec<SequenceHash>, ()> {
        // Only published digests are residency evidence. vLLM publishes the
        // hashes of the pages it actually holds (it skips null or masked pages
        // while `token_ids` still spans the whole range), so rebuilding the
        // chain from the token ids would claim pages the engine never stored.
        store
            .block_hashes
            .iter()
            .map(|hash| sequence_hash(hash).ok_or(()))
            .collect()
    }
}

/// Prefer the fastest copy when several tiers hold the same page.
fn tier_rank(tier: StorageTier) -> u8 {
    match tier {
        StorageTier::Device => 0,
        StorageTier::HostPinned => 1,
        StorageTier::Disk => 2,
        StorageTier::External => 3,
    }
}

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
        if block_size == 0 || dp_rank != self.target.dp_rank {
            return Some(Vec::new());
        }
        // The policy's page size can be coarser than the hash granularity: the
        // engine restores the aligned prefix, so round down instead of dropping
        // a recoverable hit.
        let (tokens, credit) = self.credit_parts(lora_name, tokens, usize::MAX);
        let tokens = align_down(tokens, block_size);
        if tokens == 0 {
            return Some(Vec::new());
        }
        let Ok(blocks) = u32::try_from(tokens / u64::from(block_size)) else {
            return Some(Vec::new());
        };
        // The engine restores at the tier that completes the boundary; the policy
        // weighs that tier by its transfer credit.
        let Some(credit) = credit else {
            return Some(Vec::new());
        };
        Some(vec![TierMatch {
            segments: vec![(0, blocks)],
            ..credit
        }])
    }
}
