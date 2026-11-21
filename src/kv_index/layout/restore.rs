//! Compute the common restore boundary from cache-group evidence.

use smallvec::SmallVec;

use super::{align_down, GroupEvidence, GroupRule};

pub(in crate::kv_index) struct AlignedTailDrop {
    pub groups: Vec<u32>,
    pub alignment: u32,
}

pub(in crate::kv_index) struct DenseLookup {
    pub positions: Vec<usize>,
    pub drop_tail: bool,
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(super) fn common_reusable_tokens(groups: &[GroupEvidence], max_reusable_tokens: u64) -> u64 {
    restore_boundary(groups, max_reusable_tokens, None, 0, false, &[]).0
}

/// Find the common restore boundary for a layout verified at source binding.
/// `fine_unit` is zero for block-aligned lookup.
pub(super) fn restore_boundary(
    groups: &[GroupEvidence],
    max_reusable_tokens: u64,
    tail_drop_group: Option<u32>,
    fine_unit: u64,
    proof_within_cap: bool,
    dense_lookups: &[DenseLookup],
) -> (u64, SmallVec<[(usize, u64); 1]>) {
    if groups.is_empty() || max_reusable_tokens == 0 {
        return (0, SmallVec::new());
    }
    // Every cacheable group must supply its own pages. An empty sliding
    // store can mean masked or null pages; it does not make the group optional.
    if groups
        .iter()
        .any(|group| group.segments.is_empty() && group.partial_ends.is_empty())
    {
        return (0, SmallVec::new());
    }

    // Reproduce the engine's coordinator: every group reports the longest
    // boundary it can restore under the current cap, the smallest one wins, and
    // the loop repeats because that smaller cap can invalidate a sliding
    // group's window.
    let mut cursors: Vec<_> = groups
        .iter()
        .map(|group| BoundaryCursor::new(group, false))
        .collect();
    let mut boundary = max_reusable_tokens;
    if !dense_lookups.is_empty() {
        let cap = if proof_within_cap {
            max_reusable_tokens
        } else {
            max_reusable_tokens.saturating_add(fine_unit)
        };
        // vLLM looks up dense batches in order, requiring matching endpoints
        // within a batch. Subsequent checkpoint checks can only shorten hits.
        for lookup in dense_lookups {
            let drop = if lookup.drop_tail { fine_unit } else { 0 };
            let mut proof = cap.min(boundary.saturating_add(drop));
            loop {
                let next = lookup.positions.iter().fold(proof, |end, &position| {
                    end.min(cursors[position].predecessor(proof))
                });
                if next == proof {
                    break;
                }
                proof = next;
            }
            let retained = proof.saturating_sub(drop);
            for &position in &lookup.positions {
                cursors[position].dense_proof = Some((proof, retained));
            }
            boundary = boundary.min(retained);
        }
    } else if let Some(group_idx) = tail_drop_group {
        let Some(index) = groups.iter().position(|evidence| {
            evidence.group.group_idx == group_idx && evidence.group.kind.is_dense_attention()
        }) else {
            return (0, SmallVec::new());
        };
        let drop_unit = if fine_unit == 0 {
            for cursor in &mut cursors {
                cursor.partial_end = 0;
            }
            cursors[index].size
        } else {
            fine_unit
        };
        // Starting in vLLM 0.32, the hybrid coordinator can use one
        // extra proof unit from the request's hash chain.
        let lookup_cap = if proof_within_cap || groups.len() == 1 {
            max_reusable_tokens
        } else {
            max_reusable_tokens.saturating_add(drop_unit)
        };
        let proof = cursors[index].predecessor(lookup_cap);
        // vLLM's FullAttentionManager drops one unit even below the cap.
        // https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/core/single_type_kv_cache_manager.py#L825-L835
        cursors[index].dense_proof = Some((proof, proof.saturating_sub(drop_unit)));
    }
    loop {
        let mut next = boundary;
        for cursor in &mut cursors {
            let group_hit = match cursor.dense_proof {
                // The engine retains the dense prefix after the first lookup.
                // Later checkpoint limits can shorten it without another drop.
                Some((_, retained)) => retained,
                None => cursor.predecessor(boundary),
            };
            next = next.min(group_hit);
        }
        if next == 0 {
            return (0, SmallVec::new());
        }
        if next == boundary {
            return (
                boundary,
                cursors
                    .iter()
                    .enumerate()
                    .filter_map(|(position, cursor)| {
                        cursor.dense_proof.map(|(proof, _)| (position, proof))
                    })
                    .collect(),
            );
        }
        boundary = next;
    }
}

/// A dropped page proves that the preceding aligned boundary is reusable.
/// Sliding groups need both their retained window and the extra trailing page.
pub(super) fn aligned_tail_boundary(
    groups: &[GroupEvidence],
    cap: u64,
    tail: &AlignedTailDrop,
    proof_within_cap: bool,
) -> u64 {
    let mut boundary = cap;
    let mut cursors: Vec<_> = groups
        .iter()
        .map(|evidence| {
            let drop = tail.groups.contains(&evidence.group.group_idx);
            if drop && proof_within_cap {
                boundary = boundary.min(cap.saturating_sub(u64::from(evidence.group.block_size)));
            }
            let mut cursor = BoundaryCursor::new(evidence, drop);
            cursor.partial_end = 0;
            cursor
        })
        .collect();
    boundary = align_down(boundary, tail.alignment);
    while boundary != 0 {
        let next = align_down(
            cursors
                .iter_mut()
                .map(|cursor| cursor.predecessor(boundary))
                .min()
                .unwrap_or(0),
            tail.alignment,
        );
        if next == boundary {
            return boundary;
        }
        boundary = next;
    }
    0
}

/// Temporary, query-local predecessor cursor. Block ranges are inclusive;
/// partial boundaries are exact token offsets and are never interpolated.
struct BoundaryCursor {
    size: u64,
    ranges: Vec<(u64, u64)>,
    range_end: usize,
    partials: Vec<u64>,
    partial_end: usize,
    dense_proof: Option<(u64, u64)>,
}

impl BoundaryCursor {
    fn new(evidence: &GroupEvidence, drop_last: bool) -> Self {
        let group = evidence.group;
        let size = u64::from(group.block_size);
        let segments = evidence.segments.iter().copied();
        let tail_pages = u64::from(drop_last);
        let mut ranges = Vec::new();
        match group.kind {
            GroupRule::FullAttention | GroupRule::CompressedFullAttention { .. } => {
                let end =
                    u64::from(super::full_attention_prefix(segments)).saturating_sub(tail_pages);
                if end != 0 {
                    ranges.push((1, end));
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
                    let end = u64::from(end).saturating_sub(tail_pages);
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
        Self {
            size,
            range_end: ranges.len(),
            ranges,
            partial_end: partials.len(),
            partials,
            dense_proof: None,
        }
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
