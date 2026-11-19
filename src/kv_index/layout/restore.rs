//! Compute the common restore boundary from cache-group evidence.

use std::collections::HashSet;

use super::{GroupEvidence, GroupRule};

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(super) fn common_reusable_tokens(groups: &[GroupEvidence], max_reusable_tokens: u64) -> u64 {
    restore_boundary(groups, max_reusable_tokens, None, 0).0
}

/// Find the common boundary and the raw dense hit for a verified tail-drop group.
/// `fine_unit` is zero for block-aligned lookup.
pub(super) fn restore_boundary(
    groups: &[GroupEvidence],
    max_reusable_tokens: u64,
    tail_drop_group: Option<u32>,
    fine_unit: u64,
) -> (u64, Option<(usize, u64)>) {
    if groups.is_empty() || max_reusable_tokens == 0 {
        return (0, None);
    }
    let mut group_ids = HashSet::with_capacity(groups.len());
    if groups.iter().map(|evidence| evidence.group).any(|group| {
        group.block_size == 0
            || !group_ids.insert(group.group_idx)
            || matches!(group.kind, GroupRule::SlidingWindow { window_tokens: 0 })
    }) {
        return (0, None);
    }

    // Every cacheable group must supply its own pages. An empty sliding
    // store can mean masked or null pages; it does not make the group optional.
    if groups
        .iter()
        .any(|group| group.segments.is_empty() && group.partial_ends.is_empty())
    {
        return (0, None);
    }

    // Reproduce the engine's coordinator: every group reports the longest
    // boundary it can restore under the current cap, the smallest one wins, and
    // the loop repeats because that smaller cap can invalidate a sliding
    // group's window.
    let Some(mut cursors) = groups
        .iter()
        .map(BoundaryCursor::new)
        .collect::<Option<Vec<_>>>()
    else {
        return (0, None);
    };
    let tail = match tail_drop_group {
        None => None,
        Some(group_idx) => {
            let Some(index) = groups.iter().position(|evidence| {
                evidence.group.group_idx == group_idx
                    && evidence.group.kind == GroupRule::FullAttention
            }) else {
                return (0, None);
            };
            let drop_unit = if fine_unit == 0 {
                for cursor in &mut cursors {
                    cursor.partial_end = 0;
                }
                cursors[index].size
            } else {
                fine_unit
            };
            // The hybrid coordinator can look one unit past the cap to prove
            // a dense hit, then drops that unit. The single-group coordinator
            // looks up to the cap before dropping its last block.
            let lookup_cap = if groups.len() == 1 {
                max_reusable_tokens
            } else {
                max_reusable_tokens.saturating_add(drop_unit)
            };
            let proof = cursors[index].predecessor(lookup_cap);
            // vLLM drops the last unit even when the dense hit ends below the cap.
            Some((index, proof, proof.saturating_sub(drop_unit)))
        }
    };
    let mut boundary = max_reusable_tokens;
    loop {
        let mut next = boundary;
        for (index, cursor) in cursors.iter_mut().enumerate() {
            let group_hit = match tail {
                // The engine retains the dense prefix after the first lookup.
                // Later checkpoint limits can shorten it without another drop.
                Some((group, _, retained)) if group == index => retained,
                _ => cursor.predecessor(boundary),
            };
            next = next.min(group_hit);
        }
        if next == 0 {
            return (0, None);
        }
        if next == boundary {
            return (boundary, tail.map(|(group, proof, _)| (group, proof)));
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
    fn new(evidence: &GroupEvidence) -> Option<Self> {
        let group = evidence.group;
        let size = u64::from(group.block_size);
        if let GroupRule::CompressedFullAttention { tokens_per_state } = group.kind {
            if tokens_per_state == 0 || !group.block_size.is_multiple_of(tokens_per_state) {
                return None;
            }
        }
        let segments = evidence.segments.iter().copied();
        let mut ranges = Vec::new();
        match group.kind {
            GroupRule::FullAttention | GroupRule::CompressedFullAttention { .. } => {
                let end = super::full_attention_prefix(segments);
                if end != 0 {
                    ranges.push((1, u64::from(end)));
                }
            }
            GroupRule::Checkpoint | GroupRule::SlidingWindow { .. } => {
                let needed = match group.kind {
                    GroupRule::SlidingWindow { window_tokens } => {
                        super::window_blocks(window_tokens, size)
                    }
                    _ => 1,
                };
                for (start, end) in segments {
                    let first = if start == 0 {
                        1
                    } else {
                        u64::from(start).saturating_add(needed)
                    };
                    let end = u64::from(end);
                    if first <= end {
                        ranges.push((first, end));
                    }
                }
                ranges.sort_unstable();
                ranges.dedup_by(|later, earlier| {
                    if later.0 <= earlier.1.saturating_add(1) {
                        earlier.1 = earlier.1.max(later.1);
                        true
                    } else {
                        false
                    }
                });
            }
        }

        let root_end = ranges
            .first()
            .filter(|(start, _)| *start == 1)
            .map_or(0, |(_, end)| end * size);
        let mut partials: Vec<u64> = evidence
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
            range_end: ranges.len(),
            ranges,
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
