//! Interpret one source snapshot using the cache groups' restore rules.

use super::{align_down, common_reusable_tokens, CacheGroup, GroupRule, GroupView};
use crate::kv_index::types::{push_block_segment, TierMatch};
use crate::kv_index::{Locality, RoutableTarget, StorageTier};

use crate::kv_index::index::TierBits;

/// Restore coverage for one source. Tier costs are read from the original pages.
struct GroupEvidence {
    coverage: Vec<TierMatch>,
    partial_ends: Vec<u64>,
}

impl GroupEvidence {
    fn from_pages(
        target: &RoutableTarget,
        group: &CacheGroup,
        pages: &[TierBits],
        hash_unit: usize,
    ) -> Self {
        debug_assert!(
            hash_unit > 0
                && group.block_size as usize >= hash_unit
                && (group.block_size as usize).is_multiple_of(hash_unit)
        );
        let stride = group.block_size as usize / hash_unit;
        let partial_ends = pages
            .iter()
            .enumerate()
            .filter_map(|(i, bits)| {
                ((i + 1) % stride != 0
                    && bits
                        .best()
                        .is_some_and(|tier| tier != StorageTier::External))
                .then_some((i + 1) as u64 * hash_unit as u64)
            })
            .collect();
        let mut segments = Vec::new();
        for (i, bits) in pages.iter().skip(stride - 1).step_by(stride).enumerate() {
            let Ok(end) = u32::try_from(i + 1) else {
                break;
            };
            if bits
                .best()
                .is_some_and(|tier| tier != StorageTier::External)
            {
                push_block_segment(&mut segments, end - 1, end);
            }
        }
        Self {
            coverage: if segments.is_empty() {
                Vec::new()
            } else {
                vec![TierMatch {
                    target: target.clone(),
                    // Union coverage proves restoration, never GPU residency.
                    tier: StorageTier::Device,
                    locality: Locality::Local,
                    segments,
                }]
            },
            partial_ends,
        }
    }

    fn view<'a>(&'a self, group: &CacheGroup) -> GroupView<'a> {
        GroupView {
            group_idx: group.group_idx,
            block_size: group.block_size,
            kind: group.kind,
            page_evidence: !self.coverage.is_empty() || !self.partial_ends.is_empty(),
            matches: &self.coverage,
            partial_ends: &self.partial_ends,
        }
    }
}

/// Return the credited tokens and slowest required tier for a source snapshot.
pub(in crate::kv_index) fn credit_from_pages(
    target: &RoutableTarget,
    groups: &[CacheGroup],
    claims: &[Vec<TierBits>],
    hash_block_size: usize,
    max_reusable_tokens: u64,
    replay_tokens: u32,
) -> (u64, Option<StorageTier>) {
    let evidence: Vec<_> = groups
        .iter()
        .zip(claims)
        .map(|(group, claims)| GroupEvidence::from_pages(target, group, claims, hash_block_size))
        .collect();
    let views: Vec<_> = groups
        .iter()
        .zip(&evidence)
        .map(|(group, evidence)| evidence.view(group))
        .collect();
    let boundary = common_reusable_tokens(target, &views, max_reusable_tokens);
    let credited = boundary.saturating_sub(u64::from(replay_tokens));
    let credited = align_down(credited, hash_block_size as u32);
    if credited == 0 {
        return (0, None);
    }
    // Replay changes the work credited, not the cached state needed to restore
    // the original hit. Charge the slowest tier at that restore boundary.
    let mut worst = StorageTier::Device;
    for (group_pos, (group, partials)) in groups
        .iter()
        .zip(evidence.iter().map(|e| &e.partial_ends))
        .enumerate()
    {
        let stride = group.block_size as usize / hash_block_size;
        let blocks = (boundary / u64::from(group.block_size)) as usize;
        // Every page the group needs to complete the boundary counts: a
        // dense prefix needs all of them, a checkpoint only the boundary
        // page, and a sliding group the trailing window.
        let required = match group.kind {
            GroupRule::FullAttention | GroupRule::CompressedFullAttention { .. } => blocks.max(1),
            GroupRule::Checkpoint => 1,
            GroupRule::SlidingWindow { window_tokens } => {
                let window_blocks =
                    super::window_blocks(window_tokens, u64::from(group.block_size)) as usize;
                blocks.min(window_blocks).max(1)
            }
        };
        let floor = blocks.saturating_sub(required);
        if partials.contains(&boundary) {
            let index = (boundary as usize / hash_block_size).saturating_sub(1);
            if let Some(tier) = claims[group_pos].get(index).and_then(|bits| bits.best()) {
                if tier.rank() > worst.rank() {
                    worst = tier;
                }
            }
            // A checkpoint needs only the page at this boundary, not an
            // older checkpoint which might reside on a slower tier.
            if matches!(group.kind, GroupRule::Checkpoint) {
                continue;
            }
        }
        for position in (floor..blocks.max(1)).rev() {
            let Some(tier) = claims[group_pos]
                .get(stride * (position + 1) - 1)
                .map(|bits| bits.best())
            else {
                break;
            };
            if let Some(tier) = tier {
                if tier.rank() > worst.rank() {
                    worst = tier;
                }
                if matches!(group.kind, GroupRule::Checkpoint) {
                    break;
                }
            }
        }
    }
    (credited, Some(worst))
}
