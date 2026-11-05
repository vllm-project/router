//! Common-prefix calculation for hybrid cache groups.
//!
//! Like vLLM, each group's hit is measured on its own and the smallest one is
//! credited. A hit in one group is never evidence for another.

use std::collections::HashSet;

use super::{Locality, RoutableTarget, StorageTier, TierMatch};

/// The verified restore rule for a cache group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridGroupKind {
    /// Every block from the start of the prompt through the boundary is needed.
    FullAttention,
    /// Every compressed state from the start of the prompt through the
    /// boundary is needed. One stored state covers `tokens_per_state` tokens,
    /// so a boundary is only restorable when it divides evenly by that ratio.
    CompressedFullAttention { tokens_per_state: u32 },
    /// Only a checkpoint block ending at the boundary is needed.
    Checkpoint,
}

/// One required group's placement in a single trusted snapshot.
///
/// Segment boundaries in `matches` are in this group's blocks, not tokens.
pub struct HybridGroupView<'a> {
    pub group_idx: u32,
    pub block_size: u32,
    pub kind: HybridGroupKind,
    pub matches: &'a [TierMatch],
}

/// Round `tokens` down to `block_size`.
///
/// Restore-time replay is charged in whole blocks, matching the engine's own
/// `hit_tokens -= hit_tokens % block_size` rewind.
pub fn align_down(tokens: u64, block_size: u32) -> u64 {
    if block_size == 0 {
        return 0;
    }
    tokens - tokens % u64::from(block_size)
}

/// Return the latest jointly restorable boundary, or zero when none is proven.
/// `max_reusable_tokens` is the engine's upper bound; only local GPU claims on
/// `target` count.
pub fn common_reusable_tokens(
    target: &RoutableTarget,
    groups: &[HybridGroupView<'_>],
    max_reusable_tokens: u64,
) -> u64 {
    if groups.is_empty() || max_reusable_tokens == 0 {
        return 0;
    }
    let mut group_ids = HashSet::with_capacity(groups.len());
    if groups
        .iter()
        .any(|group| group.block_size == 0 || !group_ids.insert(group.group_idx))
    {
        return 0;
    }

    // Every group reports its own hit; the smallest one is what the engine
    // restores. A group that proves nothing drops the whole prefix, because the
    // engine takes the minimum across all groups.
    let mut boundary = max_reusable_tokens;
    for group in groups {
        let available = longest_supported(group, target, u64::MAX);
        let hit = align_down(available.min(max_reusable_tokens), group.block_size);
        if hit == 0 {
            return 0;
        }
        boundary = boundary.min(hit);
    }

    // Re-check every group at the reduced boundary until it stops moving, as
    // the engine's coordinator does.
    loop {
        let mut next = boundary;
        for group in groups {
            next = next.min(longest_supported(group, target, boundary));
        }
        if next == 0 {
            return 0;
        }
        if next == boundary {
            return boundary;
        }
        boundary = next;
    }
}

fn eligible(hit: &TierMatch, target: &RoutableTarget) -> bool {
    hit.target == *target && hit.tier == StorageTier::Device && hit.locality == Locality::Local
}

/// The longest boundary at most `cap` that one group's own restore rule
/// accepts, in tokens, over the segments this snapshot holds.
fn longest_supported(group: &HybridGroupView<'_>, target: &RoutableTarget, cap: u64) -> u64 {
    let size = u64::from(group.block_size);
    if size == 0 {
        return 0;
    }
    let cap_blocks = cap / size;
    let mut best = 0;
    for hit in group.matches.iter().filter(|hit| eligible(hit, target)) {
        for &(start, end) in &hit.segments {
            if start >= end {
                continue;
            }
            let (start, end) = (u64::from(start), u64::from(end).min(cap_blocks));
            if end <= start {
                continue;
            }
            let candidate = match group.kind {
                // Dense and compressed state cover every block from the root.
                HybridGroupKind::FullAttention => {
                    if start == 0 {
                        end
                    } else {
                        0
                    }
                }
                HybridGroupKind::CompressedFullAttention { tokens_per_state } => {
                    if start != 0
                        || tokens_per_state == 0
                        || !group.block_size.is_multiple_of(tokens_per_state)
                    {
                        0
                    } else {
                        end
                    }
                }
                // A checkpoint is restorable on its own.
                HybridGroupKind::Checkpoint => end,
            };
            best = best.max(candidate * size);
        }
    }
    best
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::kv_index::{KvBlockIndexer, MatchQuery, ResidencyOwner};

    fn target(name: &str, rank: u32) -> RoutableTarget {
        RoutableTarget {
            instance_id: Arc::from(name),
            dp_rank: rank,
        }
    }

    fn hit(target: &RoutableTarget, segments: &[(u32, u32)]) -> TierMatch {
        TierMatch {
            target: target.clone(),
            tier: StorageTier::Device,
            segments: segments.to_vec(),
            locality: Locality::Local,
        }
    }

    fn group<'a>(
        id: u32,
        block_size: u32,
        kind: HybridGroupKind,
        matches: &'a [TierMatch],
    ) -> HybridGroupView<'a> {
        HybridGroupView {
            group_idx: id,
            block_size,
            kind,
            matches,
        }
    }

    #[test]
    fn sparse_checkpoints_and_different_block_sizes_use_the_same_boundary() {
        let worker = target("w", 0);
        let full = [hit(&worker, &[(0, 6)])]; // 0..96, blocks of 16
        let state = [hit(&worker, &[(0, 2)])]; // checkpoints at 32 and 64
        let groups = [
            group(0, 16, HybridGroupKind::FullAttention, &full),
            group(1, 32, HybridGroupKind::Checkpoint, &state),
        ];
        assert_eq!(common_reusable_tokens(&worker, &groups, 127), 64);
        assert_eq!(common_reusable_tokens(&worker, &groups, 63), 32);
        assert_eq!(common_reusable_tokens(&worker, &groups, 31), 0);
    }

    #[test]
    fn missing_full_blocks_or_checkpoint_falls_back_to_an_earlier_boundary() {
        let worker = target("w", 0);
        let full = [hit(&worker, &[(0, 3), (4, 6)])]; // gap at 48..64
        let state = [hit(&worker, &[(0, 1), (2, 3)])];
        let groups = [
            group(0, 16, HybridGroupKind::FullAttention, &full),
            group(1, 32, HybridGroupKind::Checkpoint, &state),
        ];
        assert_eq!(common_reusable_tokens(&worker, &groups, 128), 32);
    }

    #[test]
    fn state_on_another_rank_or_tier_cannot_complete_a_hit() {
        let worker = target("w", 0);
        let full = [hit(&worker, &[(0, 4)])];
        let mut remote = hit(&worker, &[(0, 4)]);
        remote.tier = StorageTier::HostPinned;
        let mut external = hit(&worker, &[(0, 4)]);
        external.locality = Locality::Remote;
        let state = [hit(&target("w", 1), &[(0, 4)]), remote, external];
        let groups = [
            group(0, 16, HybridGroupKind::FullAttention, &full),
            group(1, 16, HybridGroupKind::Checkpoint, &state),
        ];
        assert_eq!(common_reusable_tokens(&worker, &groups, 64), 0);
    }

    #[test]
    fn incomplete_or_invalid_layout_never_grants_credit() {
        let worker = target("w", 0);
        let hits = [hit(&worker, &[(0, 4)])];
        let invalid = [group(0, 0, HybridGroupKind::FullAttention, &hits)];
        let duplicate = [
            group(0, 16, HybridGroupKind::FullAttention, &hits),
            group(0, 16, HybridGroupKind::Checkpoint, &hits),
        ];
        assert_eq!(common_reusable_tokens(&worker, &[], 64), 0);
        assert_eq!(common_reusable_tokens(&worker, &invalid, 64), 0);
        assert_eq!(common_reusable_tokens(&worker, &duplicate, 64), 0);
    }

    #[test]
    fn group_index_removal_reduces_the_common_restore_point() {
        let worker = target("w", 0);
        let owner = ResidencyOwner::Worker {
            source: worker.instance_id.clone(),
            dp_rank: worker.dp_rank,
            incarnation: 1,
        };
        let index = KvBlockIndexer::new();
        let blocks: Vec<(Arc<str>, Arc<str>)> = (0..4)
            .map(|i| (Arc::from(format!("seq{i}")), Arc::from(format!("block{i}"))))
            .collect();
        for id in [0, 1] {
            index.store(id, owner.clone(), StorageTier::Device, None, &blocks);
        }
        let query = |group_idx| MatchQuery {
            group_idx,
            local_hashes: blocks.iter().map(|(_, local)| local.clone()).collect(),
            tiers_of_interest: vec![StorageTier::Device],
        };
        let full = index.find_matches(&query(0));
        let before = index.find_matches(&query(1));
        let groups = [
            group(0, 16, HybridGroupKind::FullAttention, &full),
            group(1, 16, HybridGroupKind::Checkpoint, &before),
        ];
        assert_eq!(common_reusable_tokens(&worker, &groups, 63), 48);

        index.remove(1, &owner, StorageTier::Device, &[Arc::from("seq2")]);
        let after = index.find_matches(&query(1));
        let groups = [
            group(0, 16, HybridGroupKind::FullAttention, &full),
            group(1, 16, HybridGroupKind::Checkpoint, &after),
        ];
        assert_eq!(common_reusable_tokens(&worker, &groups, 63), 32);
    }
}
