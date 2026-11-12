//! Cache groups, their restore rules, and the declared hybrid layout.
//!
//! Like vLLM, each group's hit is measured on its own and the smallest one is
//! credited. A hit in one group is never evidence for another.

use std::collections::HashSet;

use super::super::wire::BlockStored;
use super::super::{Locality, RoutableTarget, StorageTier, TierMatch};
use super::HybridEventIndex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridEventGroup {
    pub group_idx: u32,
    /// The verified cache block size in tokens, not the hash granularity.
    pub block_size: u32,
    pub kind: HybridGroupKind,
}

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
    /// All blocks intersecting the preceding `window_tokens` are needed.
    SlidingWindow { window_tokens: u64 },
}

/// One required group's placement in a single trusted snapshot.
///
/// Segment boundaries in `matches` are in this group's blocks, not tokens.
pub struct HybridGroupView<'a> {
    pub group_idx: u32,
    pub block_size: u32,
    pub kind: HybridGroupKind,
    pub matches: &'a [TierMatch],
    /// Whether the engine ever published pages for this group. A group that
    /// publishes key-only stores reports no addressable page, so it cannot
    /// constrain the boundary; the layout's other groups do.
    pub page_evidence: bool,
    /// Hash-aligned partial page boundaries present in this group.
    pub partial_ends: &'a [u64],
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
    if groups.iter().any(|group| {
        group.block_size == 0
            || !group_ids.insert(group.group_idx)
            || matches!(
                group.kind,
                HybridGroupKind::SlidingWindow { window_tokens: 0 }
            )
    }) {
        return 0;
    }

    // A sliding group may be key-only: V4-class layouts store its state as
    // key-only events next to the compressed group that names the pages, so its
    // silence cannot contradict another group's evidence. Every other kind must
    // publish pages to count; until it does, its absence caps reuse at zero.
    let constraining: Vec<&HybridGroupView<'_>> = groups
        .iter()
        .filter(|group| {
            group.page_evidence || !matches!(group.kind, HybridGroupKind::SlidingWindow { .. })
        })
        .collect();
    if constraining.is_empty() {
        return 0;
    }

    // Reproduce the engine's coordinator: every group reports the longest
    // boundary it can restore under the current cap, the smallest one wins, and
    // the loop repeats because that smaller cap can invalidate a sliding
    // group's window.
    let mut boundary = max_reusable_tokens;
    loop {
        let mut next = boundary;
        for group in &constraining {
            // The group already reports a boundary it can restore; snapping it
            // to the page would discard a partial-page entry the engine
            // published. The caller aligns to the hash granule at the end.
            next = next.min(supported_at(group, target, boundary));
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

/// The largest boundary at or below `boundary` this group can still restore.
///
/// Dense and compressed groups restore a prefix from the root, so the cap is
/// their only limit. A sliding group does not: it can only restore a boundary
/// whose trailing window is resident, and the credit never exceeds the end of
/// the run that covers it.
fn supported_at(group: &HybridGroupView<'_>, target: &RoutableTarget, boundary: u64) -> u64 {
    let size = u64::from(group.block_size);
    if size == 0 {
        return 0;
    }
    let HybridGroupKind::SlidingWindow { window_tokens } = group.kind else {
        return longest_supported(group, target, boundary);
    };
    let cap_blocks = boundary / size;
    let needed = window_tokens.saturating_sub(1).div_ceil(size);
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
            // A run anchored at the root has no earlier state the window could
            // be missing, so any boundary up to its end is restorable. A run
            // below the root has to cover the window that ends there.
            if start == 0 || end - start >= needed {
                best = best.max(end * size);
            }
        }
    }
    best
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
                        end * size
                    } else {
                        0
                    }
                }
                HybridGroupKind::CompressedFullAttention { tokens_per_state } => {
                    if start != 0 || tokens_per_state == 0 {
                        0
                    } else {
                        // A compressed group restores whole states. When the
                        // page size is not a state multiple the state-aligned
                        // prefix is still restorable, so align the boundary
                        // down instead of dropping the group's evidence.
                        align_down(end * size, tokens_per_state)
                    }
                }
                // A checkpoint is restorable on its own.
                HybridGroupKind::Checkpoint => end * size,
                // The window that ends at the boundary has to fit inside the
                // resident run, except below block zero: a run anchored at the
                // root has no earlier state the window could be missing, so the
                // engine restores it even when it is shorter than the window.
                // Credit stops at the run's end: the engine trims to the run,
                // never to the requested boundary.
                HybridGroupKind::SlidingWindow { window_tokens } => {
                    let needed = window_tokens.saturating_sub(1).div_ceil(size);
                    if start == 0 || end - start >= needed {
                        end * size
                    } else {
                        0
                    }
                }
            };
            best = best.max(candidate);
        }
    }
    // Checkpoints are not downward closed: an entry above `cap` cannot
    // establish an earlier checkpoint. Dense groups need the preceding pages.
    for &end in group.partial_ends {
        if end > cap || end <= best || end % size == 0 {
            continue;
        }
        let page_start = (end / size) * size;
        let restorable = match group.kind {
            HybridGroupKind::Checkpoint => true,
            HybridGroupKind::FullAttention => best >= page_start,
            HybridGroupKind::CompressedFullAttention { tokens_per_state } => {
                tokens_per_state != 0
                    && end % u64::from(tokens_per_state) == 0
                    && best >= page_start
            }
            HybridGroupKind::SlidingWindow { .. } => false,
        };
        if restorable {
            best = end;
        }
    }
    best
}

// ---- Declared layout (`--kv-hybrid-layout`) ----

use serde::{Deserialize, Serialize};

use super::super::subscriber::EventSource;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct HybridCapabilities {
    pub schema_version: u32,
    /// LoRA adapters this worker serves. The engine hashes an adapter's pages
    /// with the adapter identity, so the Router can only credit them when the
    /// layout names them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub adapters: Vec<HybridAdapterCapability>,
    pub ranks: Vec<HybridRankCapability>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct HybridAdapterCapability {
    pub name: String,
    /// The path exactly as passed to the engine's `--lora-modules`: the engine
    /// hashes the verbatim string, so an equivalent but different spelling is a
    /// different chain and credits nothing.
    pub path: String,
}

/// Adapter declarations are the only way to credit adapter pages, so a
/// malformed declaration must fail the layout instead of binding a chain that
/// is not the one the engine stores.
fn validate_adapters(
    adapters: &[HybridAdapterCapability],
) -> Result<Vec<(String, String)>, &'static str> {
    let mut names = HashSet::new();
    if adapters.len() > 64
        || adapters.iter().any(|adapter| {
            adapter.name.is_empty()
                || adapter.path.is_empty()
                || !names.insert(adapter.name.as_str())
        })
    {
        return Err("invalid adapter declaration");
    }
    Ok(adapters
        .iter()
        .map(|adapter| (adapter.name.clone(), adapter.path.clone()))
        .collect())
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct HybridRankCapability {
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
    pub groups: Vec<HybridGroupCapability>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct HybridGroupCapability {
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

impl HybridCapabilities {
    /// The capability declared for one DP rank. Ranks are independent
    /// instances: a rank that is absent here simply gets no hybrid credit and
    /// must not invalidate the other ranks of the same worker.
    pub fn rank_capability(&self, rank: u32) -> Option<&HybridRankCapability> {
        self.ranks.iter().find(|entry| entry.rank == rank)
    }

    /// Construct an index for one rank and its event source. Unknown layouts
    /// get no credit.
    pub fn index_for_rank(
        &self,
        rank: u32,
        source: &EventSource,
        target: RoutableTarget,
        max_entries: usize,
    ) -> Result<HybridEventIndex, &'static str> {
        let mut ranks = HashSet::new();
        if self.schema_version != 1
            || self.ranks.iter().any(|entry| !ranks.insert(entry.rank))
            || target.dp_rank != source.dp_rank
            || rank != source.dp_rank
            || target.instance_id != source.source
            || source.topic.is_empty()
            || source.replay_endpoint.as_deref().is_none_or(str::is_empty)
        {
            return Err("invalid capability or source binding");
        }
        let rank = self
            .rank_capability(rank)
            .ok_or("missing rank capability")?;
        if rank.model.is_empty()
            || rank.weight_version.is_empty()
            || !rank.instance_topic
            || rank.publisher_topic.as_deref() != Some(source.topic.as_str())
            || rank.hash_algorithm != "sha256_cbor"
            || !rank.full_hashes
            || !rank.prefix_caching
            || rank.hash_lookahead_tokens != 0
            || rank.speculative_tokens != 0
        {
            return Err("unsupported engine identity or hash contract");
        }
        let root = rank
            .root_hash
            .as_deref()
            .and_then(parse_hash)
            .ok_or("invalid root hash")?;
        let hash_block_size = rank.hash_block_size.ok_or("missing hash block size")?;
        let mut groups = Vec::with_capacity(rank.groups.len());
        let mut replay_tokens = 0;
        for group in &rank.groups {
            // vLLM rewinds a hit by one replay window shared by every replay
            // group. Disagreeing windows have no single restore point, and a
            // replay window only exists for sliding-window layouts.
            if group.prefix_replay_tokens > 0 {
                if !matches!(group.kind.as_str(), "sliding_window" | "sliding_window_mla") {
                    return Err("unsupported group restore contract");
                }
                if replay_tokens != 0 && replay_tokens != group.prefix_replay_tokens {
                    return Err("unsupported group restore contract");
                }
                replay_tokens = group.prefix_replay_tokens;
            }
            // Scratch groups such as the Qwen4 QSA key ring hold per-request
            // state, never shareable state, and emit no block events. They are
            // excluded from the index only when their shape proves that.
            if !group.prefix_cacheable {
                if group.prefix_replay_tokens > 0
                    && matches!(
                        (group.kind.as_str(), group.spec_type.as_str()),
                        ("sliding_window", "SlidingWindowSpec")
                            | ("sliding_window_mla", "SlidingWindowMLASpec")
                            | ("mla_attention", "MLAAttentionSpec")
                    )
                {
                    // Bounded replay: the engine rebuilds this group's trailing
                    // window from the other groups' state. Its own blocks are
                    // not required to be resident, but the window must still be
                    // charged to the layout.
                    if group.sliding_window.is_none_or(|window| {
                        window == 0 || u64::from(window) != u64::from(group.prefix_replay_tokens)
                    }) {
                        return Err("unsupported bounded replay group");
                    }
                    continue;
                }
                if group.spec_type != "CircularBufferSpec"
                    || group.tokens_per_state.as_deref() != Some("1")
                    || group.block_size == 0
                {
                    return Err("unsupported scratch group");
                }
                continue;
            }
            let kind = match (group.kind.as_str(), group.spec_type.as_str()) {
                ("full_attention", "FullAttentionSpec") if group.sliding_window.is_none() => {
                    HybridGroupKind::FullAttention
                }
                ("mla_attention", "MLAAttentionSpec")
                    if group.sliding_window.is_none()
                        && group
                            .cache_role
                            .as_deref()
                            .is_none_or(|role| role == "sparse") =>
                {
                    // A compressed group names the tokens one state covers.
                    // Without that declaration (DeepSeek V4's MLA pages) the
                    // page restores from the root like a full-attention block,
                    // so it is modeled as one instead of being rejected.
                    match group.tokens_per_state.as_deref() {
                        // Declared but malformed: the engine would not honor
                        // this shape, so the layout is refused.
                        Some(declared) => {
                            let tokens_per_state = declared
                                .parse::<u32>()
                                .ok()
                                .filter(|value| {
                                    *value > 0 && group.block_size.is_multiple_of(*value)
                                })
                                .ok_or("unsupported compressed attention group")?;
                            HybridGroupKind::CompressedFullAttention { tokens_per_state }
                        }
                        // No declaration: the page restores from the root.
                        None => HybridGroupKind::FullAttention,
                    }
                }
                ("sliding_window", "SlidingWindowSpec")
                | ("sliding_window_mla", "SlidingWindowMLASpec") => {
                    let window = group
                        .sliding_window
                        .filter(|window| *window > 0)
                        .ok_or("unsupported sliding window group")?;
                    HybridGroupKind::SlidingWindow {
                        window_tokens: u64::from(window),
                    }
                }
                ("mamba", "MambaSpec")
                    if group.mamba_cache_mode.as_deref() == Some("align")
                        && group.prefill_checkpoint_alignment.is_none_or(|alignment| {
                            alignment > 0 && group.block_size.is_multiple_of(alignment)
                        }) =>
                {
                    HybridGroupKind::Checkpoint
                }
                _ => return Err("unsupported group kind"),
            };
            groups.push(HybridEventGroup {
                group_idx: group.group_idx,
                block_size: group.block_size,
                kind,
            });
        }
        // The declared layout, not just the indexed subset, must describe a
        // hybrid cache: several groups, at least one attention group, and some
        // state that a single attention hit cannot prove (a checkpoint group,
        // a sliding-window group, or a declared replay window). One attention
        // group is enough when the layout says it is the whole layout: a model
        // whose sparse state never publishes block events (V4.1) has no group
        // whose checkpoint the credit could contradict.
        let has_attention = groups.iter().any(|group| {
            matches!(
                group.kind,
                HybridGroupKind::FullAttention
                    | HybridGroupKind::CompressedFullAttention { .. }
                    | HybridGroupKind::SlidingWindow { .. }
            )
        });
        let has_state = groups.iter().any(|group| {
            matches!(
                group.kind,
                HybridGroupKind::Checkpoint | HybridGroupKind::SlidingWindow { .. }
            )
        });
        let declared_single_attention =
            groups.len() == 1 && rank.declared_groups == Some(1) && has_attention;
        if groups.is_empty()
            || !has_attention
            || !(has_state || replay_tokens > 0 || declared_single_attention)
        {
            return Err("not a supported mixed layout");
        }
        let adapters = validate_adapters(&self.adapters)?;
        HybridEventIndex::new_with_root_hash(target, groups, hash_block_size, root, max_entries)
            .map_err(|_| "invalid group layout")?
            .with_replay_tokens(replay_tokens)
            .map_err(|_| "invalid group layout")
            .map(|index| {
                index
                    .with_adapters(adapters)
                    .with_partial_hash_hits(rank.partial_hash_hits)
            })
    }

    /// Bind one event source to its rank's verified layout.
    pub fn index_for_source(
        &self,
        source: &EventSource,
        target: RoutableTarget,
        max_entries: usize,
    ) -> Result<HybridEventIndex, &'static str> {
        self.index_for_rank(source.dp_rank, source, target, max_entries)
    }
}

fn parse_hash(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64 {
        return None;
    }
    fn nibble(value: u8) -> Option<u8> {
        match value {
            b'0'..=b'9' => Some(value - b'0'),
            b'a'..=b'f' => Some(value - b'a' + 10),
            b'A'..=b'F' => Some(value - b'A' + 10),
            _ => None,
        }
    }
    let mut root = [0; 32];
    for (index, out) in root.iter_mut().enumerate() {
        let pair = &value.as_bytes()[index * 2..index * 2 + 2];
        *out = (nibble(pair[0])? << 4) | nibble(pair[1])?;
    }
    Some(root)
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
            page_evidence: true,
            partial_ends: &[],
        }
    }

    /// vLLM also caches a prefix ending inside a page (`cache_partial_block`).
    /// The boundary is the partial token end, not the page end.
    #[test]
    fn a_partial_page_boundary_is_restorable() {
        let worker = target("w", 0);
        // Page 576, a partial entry at 560: no page-level hash exists yet.
        let mut full_group = group(0, 576, HybridGroupKind::FullAttention, &[]);
        full_group.partial_ends = &[560];
        assert_eq!(common_reusable_tokens(&worker, &[full_group], 1024), 560);
    }

    /// A checkpoint needs only itself, so its partial entry alone is enough.
    #[test]
    fn a_partial_checkpoint_is_restorable_on_its_own() {
        let worker = target("w", 0);
        let mut checkpoint = group(0, 576, HybridGroupKind::Checkpoint, &[]);
        checkpoint.partial_ends = &[560];
        assert_eq!(common_reusable_tokens(&worker, &[checkpoint], 1024), 560);
    }

    #[test]
    fn partial_checkpoint_respects_cap_and_does_not_need_earlier_pages() {
        let worker = target("w", 0);
        let mut checkpoint = group(0, 576, HybridGroupKind::Checkpoint, &[]);
        checkpoint.partial_ends = &[560, 1136];
        assert_eq!(common_reusable_tokens(&worker, &[checkpoint], 544), 0);
        let mut checkpoint = group(0, 576, HybridGroupKind::Checkpoint, &[]);
        checkpoint.partial_ends = &[560, 1136];
        assert_eq!(common_reusable_tokens(&worker, &[checkpoint], 1100), 560);
        let mut checkpoint = group(0, 576, HybridGroupKind::Checkpoint, &[]);
        checkpoint.partial_ends = &[560, 1136];
        assert_eq!(common_reusable_tokens(&worker, &[checkpoint], 1152), 1136);
    }

    #[test]
    fn dense_partial_requires_earlier_full_pages() {
        let worker = target("w", 0);
        let mut full = group(0, 576, HybridGroupKind::FullAttention, &[]);
        full.partial_ends = &[1136];
        assert_eq!(common_reusable_tokens(&worker, &[full], 1152), 0);
        let hits = [hit(&worker, &[(0, 1)])];
        let mut full = group(0, 576, HybridGroupKind::FullAttention, &hits);
        full.partial_ends = &[1136];
        assert_eq!(common_reusable_tokens(&worker, &[full], 1152), 1136);
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
    fn sliding_window_needs_only_the_required_local_interval() {
        let worker = target("w", 0);
        let full = [hit(&worker, &[(0, 8)])];
        let window = [hit(&worker, &[(5, 8)])];
        let groups = [
            group(0, 16, HybridGroupKind::FullAttention, &full),
            group(
                1,
                16,
                HybridGroupKind::SlidingWindow { window_tokens: 33 },
                &window,
            ),
        ];
        assert_eq!(common_reusable_tokens(&worker, &groups, 128), 128);
        // The last token cannot be reused, so both groups report 7 blocks. The
        // sliding group still holds two contiguous blocks below that boundary,
        // which is what its window needs, so the reduced boundary stands.
        assert_eq!(common_reusable_tokens(&worker, &groups, 127), 112);
    }

    /// The sliding group's retained tail can sit entirely above a boundary
    /// another group pulled down. The window that ends at that boundary is then
    /// absent, so there is no shared restore point and no credit.
    #[test]
    fn a_reduced_boundary_that_loses_the_sliding_window_credits_nothing() {
        let worker = target("w", 0);
        let full = [hit(&worker, &[(0, 4)])]; // 0..64
        let window = [hit(&worker, &[(6, 8)])]; // 96..128, a 32-token window
        let groups = [
            group(0, 16, HybridGroupKind::FullAttention, &full),
            group(
                1,
                16,
                HybridGroupKind::SlidingWindow { window_tokens: 32 },
                &window,
            ),
        ];
        assert_eq!(common_reusable_tokens(&worker, &groups, 143), 0);
    }

    /// A sliding group whose resident run starts at the root has no earlier
    /// state the window could be missing, so a prefix shorter than the window is
    /// still reusable: the engine restores the whole run.
    #[test]
    fn a_root_anchored_prefix_shorter_than_the_window_is_reusable() {
        let worker = target("w", 0);
        let full = [hit(&worker, &[(0, 1)])];
        let window = [hit(&worker, &[(0, 1)])];
        let groups = [
            group(0, 64, HybridGroupKind::FullAttention, &full),
            group(
                1,
                64,
                HybridGroupKind::SlidingWindow { window_tokens: 128 },
                &window,
            ),
        ];
        // A 65-token request keeps one 64-token block, well inside the window.
        assert_eq!(common_reusable_tokens(&worker, &groups, 64), 64);
        assert_eq!(common_reusable_tokens(&worker, &groups, 128), 64);
    }

    /// A sliding group whose run sits below the boundary still supports it,
    /// because the engine's lookup accepts any run of the window's blocks under
    /// the candidate length.
    #[test]
    fn a_sliding_run_below_the_boundary_keeps_the_credit() {
        let worker = target("w", 0);
        let full = [hit(&worker, &[(0, 6)])]; // 0..96
        let window = [hit(&worker, &[(4, 8)])]; // 64..128, a 32-token window
        let groups = [
            group(0, 16, HybridGroupKind::FullAttention, &full),
            group(
                1,
                16,
                HybridGroupKind::SlidingWindow { window_tokens: 32 },
                &window,
            ),
        ];
        assert_eq!(common_reusable_tokens(&worker, &groups, 143), 96);
    }

    #[test]
    fn state_on_another_rank_or_remote_locality_cannot_complete_a_hit() {
        let worker = target("w", 0);
        let full = [hit(&worker, &[(0, 4)])];
        let mut remote = hit(&worker, &[(0, 4)]);
        remote.locality = Locality::Remote;
        let state = [hit(&target("w", 1), &[(0, 4)]), remote];
        let groups = [
            group(0, 16, HybridGroupKind::FullAttention, &full),
            group(1, 16, HybridGroupKind::Checkpoint, &state),
        ];
        assert_eq!(common_reusable_tokens(&worker, &groups, 64), 0);
    }

    /// A host or storage copy of a checkpoint still completes the hit: the page
    /// is restorable, only its transfer cost differs.
    #[test]
    fn a_host_resident_checkpoint_completes_the_hit() {
        let worker = target("w", 0);
        let full = [hit(&worker, &[(0, 4)])];
        let mut host = hit(&worker, &[(0, 4)]);
        host.tier = StorageTier::HostPinned;
        let state = [host];
        let groups = [
            group(0, 16, HybridGroupKind::FullAttention, &full),
            group(1, 16, HybridGroupKind::Checkpoint, &state),
        ];
        assert_eq!(common_reusable_tokens(&worker, &groups, 64), 64);
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
        let empty_window = [group(
            1,
            16,
            HybridGroupKind::SlidingWindow { window_tokens: 0 },
            &hits,
        )];
        assert_eq!(common_reusable_tokens(&worker, &[], 64), 0);
        assert_eq!(common_reusable_tokens(&worker, &invalid, 64), 0);
        assert_eq!(common_reusable_tokens(&worker, &duplicate, 64), 0);
        assert_eq!(common_reusable_tokens(&worker, &empty_window, 64), 0);
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

    mod declarations {
        use super::*;
        use std::sync::Arc;

        #[test]
        fn a_declared_single_attention_layout_is_admitted() {
            let group = serde_json::json!({
                "group_idx": 5, "kind": "full_attention", "spec_type": "FullAttentionSpec",
                "block_size": 128, "sliding_window": null, "prefix_cacheable": true,
                "prefix_replay_tokens": 0, "mamba_cache_mode": null,
                "prefill_checkpoint_alignment": null,
            });
            let mut value = serde_json::json!({"schema_version": 1, "ranks": [{
                "rank": 0, "model": "v41", "weight_version": "default",
                "weight_identity": "model-sha256", "publisher_topic": "epoch-1",
                "instance_topic": true, "hash_algorithm": "sha256_cbor",
                "full_hashes": true,
                "root_hash": "4e1195df020de59e0d65a33a4279f1183e7ae4e5d980e309f8b55adff2e61c3e",
                "hash_block_size": 128, "hash_lookahead_tokens": 0,
                "speculative_tokens": 0, "prefix_caching": true,
                "declared_groups": 1,
                "groups": [group]
            }]});
            let source = EventSource {
                source: Arc::from("worker"),
                dp_rank: 0,
                pub_endpoint: "inproc://publisher".into(),
                replay_endpoint: Some("inproc://replay".into()),
                topic: "epoch-1".into(),
                hwm: None,
            };
            let target = RoutableTarget {
                instance_id: Arc::from("worker"),
                dp_rank: 0,
            };
            let make = |value: &serde_json::Value| {
                serde_json::from_value::<HybridCapabilities>(value.clone()).unwrap()
            };
            assert!(make(&value)
                .index_for_source(&source, target.clone(), 8)
                .is_ok());
            // Without the explicit total the same layout is indistinguishable
            // from a truncated multi-group declaration, so it stays rejected.
            value["ranks"][0]["declared_groups"] = serde_json::Value::Null;
            assert!(make(&value)
                .index_for_source(&source, target.clone(), 8)
                .is_err());
            // A declared single group that is not an attention group is useless.
            value["ranks"][0]["groups"][0]["kind"] = serde_json::json!("mamba");
            value["ranks"][0]["groups"][0]["spec_type"] = serde_json::json!("MambaSpec");
            value["ranks"][0]["groups"][0]["mamba_cache_mode"] = serde_json::json!("align");
            value["ranks"][0]["declared_groups"] = serde_json::json!(1);
            assert!(make(&value).index_for_source(&source, target, 8).is_err());
        }

        #[test]
        fn capability_requires_complete_supported_layout_and_same_publisher() {
            let group = |idx, kind, spec, mode| {
                serde_json::json!({
                    "group_idx": idx, "kind": kind, "spec_type": spec, "block_size": 528,
                    "sliding_window": null, "prefix_cacheable": true,
                    "prefix_replay_tokens": 0, "mamba_cache_mode": mode,
                    "prefill_checkpoint_alignment": if kind == "mamba" { Some(528) } else { None },
                })
            };
            let mut value = serde_json::json!({"schema_version": 1, "ranks": [{
                "rank": 0, "model": "qwen", "weight_version": "default", "weight_identity": "model-sha256",
                "publisher_topic": "epoch-1", "instance_topic": true,
                "hash_algorithm": "sha256_cbor", "full_hashes": true,
                "root_hash": "4e1195df020de59e0d65a33a4279f1183e7ae4e5d980e309f8b55adff2e61c3e",
                "hash_block_size": 528, "hash_lookahead_tokens": 0,
                "speculative_tokens": 0, "prefix_caching": true,
                "groups": [group(0, "mamba", "MambaSpec", Some("align")),
                    group(1, "mamba", "MambaSpec", Some("align")),
                    group(2, "mamba", "MambaSpec", Some("align")),
                    group(3, "full_attention", "FullAttentionSpec", None)]
            }]});
            let source = EventSource {
                source: Arc::from("worker"),
                dp_rank: 0,
                pub_endpoint: "inproc://publisher".into(),
                replay_endpoint: Some("inproc://replay".into()),
                topic: "epoch-1".into(),
                hwm: None,
            };
            let target = RoutableTarget {
                instance_id: Arc::from("worker"),
                dp_rank: 0,
            };
            let make = |value: &serde_json::Value| {
                serde_json::from_value::<HybridCapabilities>(value.clone()).unwrap()
            };
            let index = make(&value)
                .index_for_source(&source, target.clone(), 8)
                .unwrap();
            index.begin_empty(0);
            index
                .apply_payload(
                    0,
                    &crate::kv_index::hybrid::residency::tests::qwen35_payload(),
                )
                .unwrap();
            let prompt: Vec<u32> = (0..1024).map(|i| 100 + i % 100).collect();
            assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 528);
            value["ranks"][0]["groups"][0]["prefill_checkpoint_alignment"] =
                serde_json::Value::Null;
            assert!(make(&value)
                .index_for_source(&source, target.clone(), 8)
                .is_ok());
            value["ranks"][0]["groups"][0]["prefill_checkpoint_alignment"] = serde_json::json!(7);
            assert!(make(&value)
                .index_for_source(&source, target.clone(), 8)
                .is_err());
            value["ranks"][0]["groups"][0]["prefill_checkpoint_alignment"] = serde_json::json!(528);
            for (field, invalid) in [
                ("publisher_topic", serde_json::json!("other")),
                ("full_hashes", serde_json::json!(false)),
                ("speculative_tokens", serde_json::json!(1)),
                ("root_hash", serde_json::json!("bad")),
            ] {
                let saved = value["ranks"][0][field].clone();
                value["ranks"][0][field] = invalid;
                assert!(
                    make(&value)
                        .index_for_source(&source, target.clone(), 8)
                        .is_err(),
                    "{field}"
                );
                value["ranks"][0][field] = saved;
            }
            value["ranks"][0]["groups"][0]["prefix_replay_tokens"] = serde_json::json!(128);
            assert!(make(&value).index_for_source(&source, target, 8).is_err());
        }

        /// DeepSeek V4 publishes MLA pages (no `tokens_per_state`) next to
        /// sliding-window groups on a 4-token hash chain; that layout has to bind.
        #[test]
        fn deepseek_v4_layout_binds_without_a_checkpoint_group() {
            let group = |idx, kind, spec, block, window: Option<u32>| {
                serde_json::json!({
                    "group_idx": idx, "kind": kind, "spec_type": spec, "block_size": block,
                    "sliding_window": window, "prefix_cacheable": true,
                    "prefix_replay_tokens": 0, "mamba_cache_mode": null,
                    "prefill_checkpoint_alignment": null,
                })
            };
            let value = serde_json::json!({"schema_version": 1, "ranks": [{
                "rank": 0, "model": "deepseek-v4", "weight_version": "default",
                "weight_identity": null, "publisher_topic": "ds4", "instance_topic": true,
                "hash_algorithm": "sha256_cbor", "root_hash":
                    "9bd96a485ad84efdafb72ee48a1d7a69bcead0f8f0433173941b276b9581eef0",
                "full_hashes": true, "hash_block_size": 4, "hash_lookahead_tokens": 0,
                "speculative_tokens": 0, "prefix_caching": true,
                "groups": [
                    group(0, "sliding_window_mla", "SlidingWindowMLASpec", 64, Some(128)),
                    group(1, "sliding_window_mla", "SlidingWindowMLASpec", 64, Some(128)),
                    group(2, "mla_attention", "MLAAttentionSpec", 256, None),
                    group(3, "sliding_window_mla", "SlidingWindowMLASpec", 4, Some(8)),
                    group(4, "sliding_window_mla", "SlidingWindowMLASpec", 8, Some(128)),
                ]
            }]});
            let capabilities: HybridCapabilities = serde_json::from_value(value).expect("layout");
            let source = EventSource {
                source: Arc::from("http://worker"),
                dp_rank: 0,
                pub_endpoint: "tcp://127.0.0.1:1".into(),
                replay_endpoint: Some("tcp://127.0.0.1:2".into()),
                topic: "ds4".into(),
                hwm: None,
            };
            let target = RoutableTarget {
                instance_id: Arc::from("http://worker"),
                dp_rank: 0,
            };
            capabilities
                .index_for_source(&source, target, 16)
                .expect("deepseek v4 layout must bind");
        }

        #[test]
        fn qwen4_layout_admits_compressed_group_and_skips_the_scratch_ring() {
            let group = |idx, kind, spec, extra: serde_json::Value| {
                let mut value = serde_json::json!({
                    "group_idx": idx, "kind": kind, "spec_type": spec, "block_size": 320,
                    "sliding_window": null, "prefix_cacheable": true,
                    "prefix_replay_tokens": 0, "mamba_cache_mode": null,
                    "prefill_checkpoint_alignment": null, "tokens_per_state": "1",
                    "cache_role": null,
                });
                value
                    .as_object_mut()
                    .unwrap()
                    .extend(extra.as_object().unwrap().clone());
                value
            };
            let mamba = || {
                group(
                    0,
                    "mamba",
                    "MambaSpec",
                    serde_json::json!({"mamba_cache_mode": "align", "prefill_checkpoint_alignment": 320, "tokens_per_state": "-1", "block_size": 320}),
                )
            };
            // Captured from a Qwen4Exp (QSA) vLLM 0.33.0 probe: three GDN groups,
            // one per-request key ring that emits no block events, and one
            // compressed MLA group whose states cover four tokens each.
            let mut value = serde_json::json!({"schema_version": 1, "ranks": [{
                "rank": 0, "model": "qwen4", "weight_version": "default",
                "weight_identity": "qwen4-sha256",
                "publisher_topic": "kv-events", "instance_topic": true,
                "hash_algorithm": "sha256_cbor", "full_hashes": true,
                "root_hash": "4e1195df020de59e0d65a33a4279f1183e7ae4e5d980e309f8b55adff2e61c3e",
                "hash_block_size": 320, "hash_lookahead_tokens": 0,
                "speculative_tokens": 0, "prefix_caching": true,
                "groups": [
                    mamba(),
                    group(1, "mamba", "MambaSpec", serde_json::json!({"mamba_cache_mode": "align", "prefill_checkpoint_alignment": 320, "tokens_per_state": "-1"})),
                    group(2, "mamba", "MambaSpec", serde_json::json!({"mamba_cache_mode": "align", "prefill_checkpoint_alignment": 320, "tokens_per_state": "-1"})),
                    group(3, "unknown", "CircularBufferSpec", serde_json::json!({"prefix_cacheable": false, "block_size": 4})),
                    group(4, "mla_attention", "MLAAttentionSpec", serde_json::json!({"tokens_per_state": "4", "cache_role": "sparse"}))
                ]
            }]});
            let source = EventSource {
                source: Arc::from("worker"),
                dp_rank: 0,
                pub_endpoint: "inproc://publisher".into(),
                replay_endpoint: Some("inproc://replay".into()),
                topic: "kv-events".into(),
                hwm: None,
            };
            let target = RoutableTarget {
                instance_id: Arc::from("worker"),
                dp_rank: 0,
            };
            let make = |value: &serde_json::Value| {
                serde_json::from_value::<HybridCapabilities>(value.clone()).unwrap()
            };
            assert!(make(&value)
                .index_for_source(&source, target.clone(), 16)
                .is_ok());

            for (path, invalid) in [(4, serde_json::json!("3")), (4, serde_json::json!("field"))] {
                let saved = value["ranks"][0]["groups"][path]["tokens_per_state"].clone();
                value["ranks"][0]["groups"][path]["tokens_per_state"] = invalid;
                assert!(
                    make(&value)
                        .index_for_source(&source, target.clone(), 16)
                        .is_err(),
                    "{path}"
                );
                value["ranks"][0]["groups"][path]["tokens_per_state"] = saved;
            }
            value["ranks"][0]["groups"][4]["cache_role"] = serde_json::json!("indexer");
            assert!(make(&value)
                .index_for_source(&source, target.clone(), 16)
                .is_err());
            value["ranks"][0]["groups"][4]["cache_role"] = serde_json::json!("sparse");
            value["ranks"][0]["groups"][3]["spec_type"] = serde_json::json!("FullAttentionSpec");
            assert!(make(&value)
                .index_for_source(&source, target.clone(), 16)
                .is_err());
            value["ranks"][0]["groups"][3]["spec_type"] = serde_json::json!("CircularBufferSpec");
            value["ranks"][0]["groups"][4]["kind"] = serde_json::json!("full_attention");
            assert!(make(&value).index_for_source(&source, target, 16).is_err());
        }
    }
}

pub(super) fn stored_shape_matches(kind: HybridGroupKind, store: &BlockStored) -> bool {
    let block_size = store.block_size as usize;
    if block_size == 0 || store.token_ids.is_empty() {
        return false;
    }
    if !store.token_ids.len().is_multiple_of(block_size) {
        return false;
    }
    let blocks = store.token_ids.len() / block_size;
    match kind {
        HybridGroupKind::Checkpoint | HybridGroupKind::SlidingWindow { .. } => {
            store.block_hashes.len() <= blocks
        }
        _ => store.block_hashes.len() == blocks,
    }
}

pub(super) fn event_kind_matches(kind: HybridGroupKind, store: &BlockStored) -> bool {
    let spec = store.kv_cache_spec_kind.as_deref();
    match kind {
        // An MLA page without a state ratio restores from the root exactly like
        // a full-attention block, and DeepSeek V4 publishes it as
        // `mla_attention`.
        HybridGroupKind::FullAttention => {
            matches!(spec, Some("full_attention") | Some("mla_attention"))
                && store.kv_cache_spec_sliding_window.is_none()
        }
        HybridGroupKind::CompressedFullAttention { .. } => {
            spec == Some("mla_attention") && store.kv_cache_spec_sliding_window.is_none()
        }
        HybridGroupKind::Checkpoint => {
            spec == Some("mamba") && store.kv_cache_spec_sliding_window.is_none()
        }
        HybridGroupKind::SlidingWindow { window_tokens } => {
            matches!(spec, Some("sliding_window" | "sliding_window_mla"))
                && store.kv_cache_spec_sliding_window.map(u64::from) == Some(window_tokens)
        }
    }
}
