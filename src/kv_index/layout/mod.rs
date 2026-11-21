//! Cache layouts, restore rules and declared model capabilities.

mod capabilities;
#[cfg(all(test, feature = "cpu", feature = "ready"))]
pub(crate) use capabilities::valid_groups;
pub use capabilities::{AdapterCapability, CacheCapabilities, GroupCapability, RankCapability};

mod evidence;
mod restore;

pub(super) use evidence::full_attention_matches;
pub(super) use evidence::{credit_from_pages, TailDrop};
#[cfg(all(test, feature = "cpu", feature = "ready"))]
use restore::common_reusable_tokens;
pub(super) use restore::{AlignedTailDrop, DenseLookup};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheGroup {
    pub group_idx: u32,
    /// The verified cache block size in tokens, not the hash granularity.
    pub block_size: u32,
    pub kind: GroupRule,
}

/// The verified restore rule for a cache group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupRule {
    /// Every block from the start of the prompt through the boundary is needed.
    FullAttention,
    /// Every compressed state from the start of the prompt through the
    /// boundary is needed. One stored state covers `tokens_per_state` tokens,
    /// so a boundary is only restorable when it divides evenly by that ratio.
    CompressedFullAttention { tokens_per_state: u32 },
    /// Only a checkpoint block ending at the boundary is needed.
    Checkpoint,
    /// The preceding `window_tokens - 1` tokens must be covered by a
    /// contiguous run of cached blocks, or the run must reach the prompt start.
    SlidingWindow { window_tokens: u64 },
}

impl GroupRule {
    // vLLM registers MLAAttentionSpec with FullAttentionManager too. Either
    // kind can drop a lookup unit when the engine reports that group.
    fn uses_full_attention_lookup(self) -> bool {
        matches!(
            self,
            Self::FullAttention | Self::CompressedFullAttention { .. }
        )
    }
}

/// Restore coverage of one group in the source snapshot.
///
/// Segment boundaries use group blocks. Tier costs come from the original pages.
struct GroupEvidence {
    group: CacheGroup,
    segments: Vec<(u32, u32)>,
    /// Hash-aligned partial page boundaries present in this group.
    partial_ends: Vec<u64>,
}

/// Full attention restores from the root, never across a missing page. This
/// primitive is shared by group restoration and the existing tiered scorer.
pub(crate) fn full_attention_prefix(ranges: impl IntoIterator<Item = (u32, u32)>) -> u32 {
    ranges
        .into_iter()
        .filter(|(start, _)| *start == 0)
        .map(|(_, end)| end)
        .max()
        .unwrap_or(0)
}

pub(super) fn window_blocks(window_tokens: u64, block_size: u64) -> u64 {
    // vLLM counts at least one matching block before accepting a window hit.
    window_tokens.saturating_sub(1).div_ceil(block_size).max(1)
}

/// Round `tokens` down to `block_size`.
///
/// Restore-time replay is charged in whole blocks, matching the engine's own
/// `hit_tokens -= hit_tokens % block_size` rewind.
pub(crate) fn align_down(tokens: u64, block_size: u32) -> u64 {
    if block_size == 0 {
        return 0;
    }
    tokens - tokens % u64::from(block_size)
}

pub(super) fn common_block_alignment(sizes: impl IntoIterator<Item = u32>) -> Option<u32> {
    sizes.into_iter().try_fold(1u32, |alignment, size| {
        if size == 0 {
            return None;
        }
        let (mut a, mut b) = (alignment, size);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        (alignment / a).checked_mul(size)
    })
}
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests;
