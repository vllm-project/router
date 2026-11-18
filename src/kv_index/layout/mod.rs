//! Cache layouts, restore rules and declared model capabilities.

use crate::kv_index::types::TierMatch;
use serde::{Deserialize, Serialize};

mod evidence;
mod restore;

pub(super) use evidence::credit_from_pages;
pub use restore::common_reusable_tokens;

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
    /// All blocks intersecting the preceding `window_tokens` are needed.
    SlidingWindow { window_tokens: u64 },
}

/// One required group's placement in a single trusted snapshot.
///
/// Segment boundaries in `matches` are in this group's blocks, not tokens.
pub struct GroupView<'a> {
    pub group_idx: u32,
    pub block_size: u32,
    pub kind: GroupRule,
    pub matches: &'a [TierMatch],
    /// Whether this group has published addressable page evidence.
    /// A missing group cannot inherit evidence from another group.
    pub page_evidence: bool,
    /// Hash-aligned partial page boundaries present in this group.
    pub partial_ends: &'a [u64],
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
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct CacheCapabilities {
    pub schema_version: u32,
    /// LoRA adapters this worker serves. The engine hashes an adapter's pages
    /// with the adapter identity, so the Router can only credit them when the
    /// layout names them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub adapters: Vec<AdapterCapability>,
    pub ranks: Vec<RankCapability>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct AdapterCapability {
    pub name: String,
    /// The path exactly as passed to the engine's `--lora-modules`: the engine
    /// hashes the verbatim string, so an equivalent but different spelling is a
    /// different chain and credits nothing.
    pub path: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct RankCapability {
    pub rank: u32,
    /// Total cache groups the engine declares, including groups that never
    /// publish block events (V4.1 keeps its sparse state in such backends).
    /// `None` means the listed groups are the whole layout.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_groups: Option<u32>,
    pub model: String,
    pub weight_version: String,
    pub weight_identity: Option<String>,
    pub publisher_topic: Option<String>,
    pub instance_topic: bool,
    pub hash_algorithm: String,
    pub root_hash: Option<String>,
    pub full_hashes: bool,
    pub hash_block_size: Option<u32>,
    /// Source publishes hash-granularity stores as partial prefix entries.
    /// Enable only for a verified engine event contract.
    #[serde(default)]
    pub partial_hash_hits: bool,
    pub hash_lookahead_tokens: u32,
    pub speculative_tokens: u32,
    pub prefix_caching: bool,
    pub groups: Vec<GroupCapability>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct GroupCapability {
    pub group_idx: u32,
    pub kind: String,
    pub spec_type: String,
    pub block_size: u32,
    pub sliding_window: Option<u32>,
    pub prefix_cacheable: bool,
    pub prefix_replay_tokens: u32,
    pub mamba_cache_mode: Option<String>,
    pub prefill_checkpoint_alignment: Option<u32>,
    pub tokens_per_state: Option<String>,
    pub cache_role: Option<String>,
}
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests;
