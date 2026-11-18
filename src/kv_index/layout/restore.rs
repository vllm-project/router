//! Compute the common restore boundary from cache-group evidence.

use std::collections::HashSet;

use super::{GroupRule, GroupView};
use crate::kv_index::types::TierMatch;
use crate::kv_index::{Locality, RoutableTarget, StorageTier};

/// Return the latest jointly restorable boundary, or zero when none is proven.
/// `max_reusable_tokens` is the engine's upper bound; only local, addressable
/// device, host, or disk claims on `target` count. Compressed groups must have
/// a nonzero state size that divides their block size.
pub fn common_reusable_tokens(
    target: &RoutableTarget,
    groups: &[GroupView<'_>],
    max_reusable_tokens: u64,
) -> u64 {
    if groups.is_empty() || max_reusable_tokens == 0 {
        return 0;
    }
    let mut group_ids = HashSet::with_capacity(groups.len());
    if groups.iter().any(|group| {
        group.block_size == 0
            || !group_ids.insert(group.group_idx)
            || matches!(group.kind, GroupRule::SlidingWindow { window_tokens: 0 })
    }) {
        return 0;
    }

    // Every cacheable group must supply its own pages. An empty sliding
    // store can mean masked or null pages; it does not make the group optional.
    if groups.iter().any(|group| !group.page_evidence) {
        return 0;
    }

    // Reproduce the engine's coordinator: every group reports the longest
    // boundary it can restore under the current cap, the smallest one wins, and
    // the loop repeats because that smaller cap can invalidate a sliding
    // group's window.
    let Some(mut cursors) = groups
        .iter()
        .map(|group| BoundaryCursor::new(group, target))
        .collect::<Option<Vec<_>>>()
    else {
        return 0;
    };
    let mut boundary = max_reusable_tokens;
    loop {
        let mut next = boundary;
        for cursor in &mut cursors {
            // The group already reports a boundary it can restore; snapping it
            // to the page would discard a partial-page entry the engine
            // published. The caller aligns to the hash granule at the end.
            next = next.min(cursor.predecessor(boundary));
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

/// Temporary, query-local predecessor cursor. Block ranges are inclusive;
/// partial boundaries are exact token offsets and are never interpolated.
struct BoundaryCursor {
    size: u64,
    ranges: Vec<(u64, u64)>,
    range_end: usize,
    partials: Vec<u64>,
    partial_end: usize,
}

impl BoundaryCursor {
    fn new(group: &GroupView<'_>, target: &RoutableTarget) -> Option<Self> {
        let size = u64::from(group.block_size);
        if let GroupRule::CompressedFullAttention { tokens_per_state } = group.kind {
            if tokens_per_state == 0 || !group.block_size.is_multiple_of(tokens_per_state) {
                return None;
            }
        }
        let mut ranges = Vec::new();
        let full_prefix = super::full_attention_prefix(
            group
                .matches
                .iter()
                .filter(|hit| eligible(hit, target))
                .flat_map(|hit| hit.segments.iter().copied()),
        );
        for hit in group.matches.iter().filter(|hit| eligible(hit, target)) {
            for &(start, end) in &hit.segments {
                let (start, end) = (u64::from(start), u64::from(end));
                if start >= end {
                    continue;
                }
                let first = match group.kind {
                    GroupRule::FullAttention | GroupRule::CompressedFullAttention { .. } => {
                        if start != 0 || end != u64::from(full_prefix) {
                            continue;
                        }
                        1
                    }
                    GroupRule::Checkpoint => start + 1,
                    GroupRule::SlidingWindow { window_tokens } => {
                        if start == 0 {
                            1
                        } else {
                            let needed = super::window_blocks(window_tokens, size);
                            start.saturating_add(needed)
                        }
                    }
                };
                if first <= end {
                    ranges.push((first, end));
                }
            }
        }
        ranges.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
        for (start, end) in ranges {
            if let Some(last) = merged.last_mut() {
                if start <= last.1.saturating_add(1) {
                    last.1 = last.1.max(end);
                    continue;
                }
            }
            merged.push((start, end));
        }

        let root_end = merged
            .first()
            .filter(|(start, _)| *start == 1)
            .map_or(0, |(_, end)| end * size);
        let mut partials: Vec<u64> = group
            .partial_ends
            .iter()
            .copied()
            .filter(|end| {
                if *end == 0 || *end % size == 0 {
                    return false;
                }
                let page_start = (*end / size) * size;
                match group.kind {
                    GroupRule::Checkpoint => true,
                    GroupRule::FullAttention => root_end >= page_start,
                    GroupRule::CompressedFullAttention { tokens_per_state } => {
                        *end % u64::from(tokens_per_state) == 0 && root_end >= page_start
                    }
                    GroupRule::SlidingWindow { .. } => false,
                }
            })
            .collect();
        partials.sort_unstable();
        partials.dedup();
        Some(Self {
            size,
            range_end: merged.len(),
            ranges: merged,
            partial_end: partials.len(),
            partials,
        })
    }

    fn predecessor(&mut self, cap: u64) -> u64 {
        let cap_blocks = cap / self.size;
        while self.range_end > 0 && self.ranges[self.range_end - 1].0 > cap_blocks {
            self.range_end -= 1;
        }
        let full = if self.range_end == 0 {
            0
        } else {
            self.ranges[self.range_end - 1].1.min(cap_blocks) * self.size
        };
        while self.partial_end > 0 && self.partials[self.partial_end - 1] > cap {
            self.partial_end -= 1;
        }
        full.max(if self.partial_end == 0 {
            0
        } else {
            self.partials[self.partial_end - 1]
        })
    }
}

fn eligible(hit: &TierMatch, target: &RoutableTarget) -> bool {
    // A page held in host memory or on storage is still restorable; the policy
    // weighs it by the transfer credit of its tier. External residencies belong
    // to another owner and are never claimed by this index.
    matches!(
        hit.tier,
        StorageTier::Device | StorageTier::HostPinned | StorageTier::Disk
    ) && hit.locality == Locality::Local
        && hit.target == *target
}
