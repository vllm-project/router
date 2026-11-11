//! Hybrid attention prefix index.
//!
//! `HybridIndex` owns the residency lock and is the only type callers see.
//! Event interpretation lives in `ingest`, cache-group rules in `layout`.

mod ingest;
mod layout;
mod residency;
mod subscriber;

use std::collections::HashSet;

use parking_lot::RwLock;

pub use self::ingest::HybridEventError;
pub use self::layout::{
    common_reusable_tokens, HybridCapabilities, HybridEventGroup, HybridGroupCapability,
    HybridGroupKind, HybridGroupView, HybridRankCapability,
};
pub use self::subscriber::{
    spawn_zmq, HybridEventStream, HybridStreamError, HybridSubscriberHandle,
};
pub use crate::kv_index::subscriber::HybridSourceLease;

#[allow(unused_imports)]
use self::ingest::HybridEventError as _;
use self::residency::State;
use super::{Locality, RoutableTarget, StorageTier, TierMatch};
use crate::kv_hash::{BlockKeyGenerator, SequenceHash};

pub struct HybridEventIndex {
    pub(super) target: RoutableTarget,
    pub(super) groups: Vec<HybridEventGroup>,
    pub(super) keys: BlockKeyGenerator,
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
}
