//! Interpret one source snapshot using the cache groups' restore rules.

use super::{align_down, restore::restore_boundary, CacheGroup, GroupEvidence, GroupRule};
use crate::kv_index::storage::PageTiers;
use crate::kv_index::types::{TierBits, TierMatch};
use crate::kv_index::{Locality, RoutableTarget, StorageTier};

/// Preserve the full-attention policy's per-tier segments, including gaps and
/// a transfer continuation. Only published entries within the request cap count.
pub(in crate::kv_index) fn full_attention_matches(
    target: &RoutableTarget,
    group: &CacheGroup,
    pages: PageTiers<'_>,
    hash_unit: usize,
    score_unit: u32,
    cap: u64,
) -> Vec<TierMatch> {
    const TIERS: [StorageTier; 3] = [
        StorageTier::Device,
        StorageTier::HostPinned,
        StorageTier::Disk,
    ];
    let aligned = score_unit == group.block_size && hash_unit == score_unit as usize;
    let max_pages = cap / hash_unit as u64;
    let mut segments = [Vec::new(), Vec::new(), Vec::new()];
    // Join adjacent physical coverage at each tier before rounding to scoring blocks.
    let mut covered = [(0_u64, 0_u64); 3];
    let prefix_pages = max_pages.min(pages.prefix_len as u64);
    if prefix_pages != 0 && pages.prefix.0 != 0 {
        let prefix_tokens = prefix_pages * hash_unit as u64;
        let blocks = if aligned {
            prefix_pages
        } else {
            prefix_tokens / u64::from(score_unit)
        };
        let Ok(prefix_end) = u32::try_from(blocks) else {
            return Vec::new();
        };
        for ((tier, hits), run) in TIERS.into_iter().zip(&mut segments).zip(&mut covered) {
            if pages.prefix.contains(tier) {
                run.1 = prefix_tokens;
                if prefix_end != 0 {
                    hits.push((0, prefix_end));
                }
            }
        }
    }
    for (position, bits) in pages.tail.iter().enumerate() {
        let position = (pages.prefix_len + position) as u64;
        if position >= max_pages {
            break;
        }
        if bits.0 == 0 {
            continue;
        }
        let (start, end) = if aligned {
            (position, position + 1)
        } else {
            let end = (position + 1) * hash_unit as u64;
            let start = (end - 1) / u64::from(group.block_size) * u64::from(group.block_size);
            (start, end)
        };
        for ((tier, hits), run) in TIERS.into_iter().zip(&mut segments).zip(&mut covered) {
            if !bits.contains(tier) {
                continue;
            }
            let (start, end) = if aligned {
                (start, end)
            } else {
                if start > run.1 {
                    run.0 = start;
                }
                run.1 = end;
                (
                    run.0.div_ceil(u64::from(score_unit)),
                    end / u64::from(score_unit),
                )
            };
            let (Ok(start), Ok(end)) = (u32::try_from(start), u32::try_from(end)) else {
                return Vec::new();
            };
            if start < end {
                match hits.last_mut() {
                    Some((_, previous_end)) if start <= *previous_end => *previous_end = end,
                    _ => hits.push((start, end)),
                }
            }
        }
    }
    TIERS
        .into_iter()
        .zip(segments)
        .filter(|(_, segments)| !segments.is_empty())
        .map(|(tier, segments)| TierMatch {
            target: target.clone(),
            tier,
            locality: Locality::Local,
            segments,
        })
        .collect()
}

impl GroupEvidence {
    fn from_pages(group: CacheGroup, pages: &PageTiers<'_>, hash_unit: usize) -> Self {
        debug_assert!(
            hash_unit > 0
                && group.block_size as usize >= hash_unit
                && (group.block_size as usize).is_multiple_of(hash_unit)
        );
        let stride = group.block_size as usize / hash_unit;
        let local = |bits: TierBits| {
            bits.best()
                .is_some_and(|tier| tier != StorageTier::External)
        };
        let shared = local(pages.prefix);
        let mut segments = Vec::new();
        let prefix_blocks = u32::try_from(pages.prefix_len / stride).unwrap_or(u32::MAX);
        let mut run_start = (shared && prefix_blocks != 0).then_some(0);
        let partial_ends =
            if stride == 1 {
                Vec::new()
            } else {
                (1..=if shared { pages.prefix_len } else { 0 })
                    .chain(
                        pages.tail.iter().enumerate().filter_map(|(i, bits)| {
                            local(*bits).then_some(pages.prefix_len + i + 1)
                        }),
                    )
                    .filter(|end| end % stride != 0)
                    .map(|end| end as u64 * hash_unit as u64)
                    .collect()
            };
        // The shared run may end inside a physical block. Align the first
        // tail endpoint to the group's stride in the original hash chain.
        let offset = stride - 1 - pages.prefix_len % stride;
        let mut end = prefix_blocks;
        for &bits in pages.tail.iter().skip(offset).step_by(stride) {
            if end == u32::MAX {
                break;
            }
            if local(bits) {
                run_start.get_or_insert(end);
            } else if let Some(start) = run_start.take() {
                segments.push((start, end));
            }
            end += 1;
        }
        if let Some(start) = run_start {
            segments.push((start, end));
        }
        Self {
            group,
            segments,
            partial_ends,
        }
    }
}

/// The verified layout's dense group that must drop one proof unit.
#[derive(Default)]
pub(in crate::kv_index) struct TailDrop {
    pub group: Option<u32>,
    pub fine_grained: bool,
}

/// Return the credited tokens and slowest required tier for a source snapshot.
pub(in crate::kv_index) fn credit_from_pages(
    groups: &[CacheGroup],
    claims: &[PageTiers<'_>],
    hash_block_size: usize,
    max_reusable_tokens: u64,
    replay_tokens: u32,
    tail_drop: TailDrop,
) -> (u64, Option<StorageTier>) {
    let evidence: Vec<_> = groups
        .iter()
        .zip(claims)
        .map(|(&group, claims)| GroupEvidence::from_pages(group, claims, hash_block_size))
        .collect();
    let fine_unit = if tail_drop.fine_grained {
        hash_block_size as u64
    } else {
        0
    };
    let (boundary, tail_proof) =
        restore_boundary(&evidence, max_reusable_tokens, tail_drop.group, fine_unit);
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
        let partial_end = match tail_proof {
            // The dense block already matched by the coordinator may have been
            // published at a later hash boundary. Price the retained physical
            // page rather than a nonexistent hash exactly at `boundary`.
            Some((position, proof))
                if position == group_pos
                    && !boundary.is_multiple_of(u64::from(group.block_size)) =>
            {
                Some(proof.min((blocks as u64 + 1) * u64::from(group.block_size)))
            }
            _ => partials.contains(&boundary).then_some(boundary),
        };
        if let Some(end) = partial_end {
            let index = (end as usize / hash_block_size).saturating_sub(1);
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
            let index = stride * (position + 1) - 1;
            let Some(tier) = claims[group_pos].get(index).map(|bits| bits.best()) else {
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
            if index < claims[group_pos].prefix_len {
                break;
            }
        }
    }
    (credited, Some(worst))
}
