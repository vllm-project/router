//! Common-prefix calculation for hybrid cache groups.
//!
//! Like vLLM, each group's hit is measured on its own and the smallest one is
//! credited. A hit in one group is never evidence for another.

use std::collections::HashSet;

use super::super::wire::BlockStored;
use super::super::{Locality, RoutableTarget, StorageTier, TierMatch};

/// The verified restore rule for a cache group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridGroupKind {
    /// Every block from the start of the prompt through the boundary is needed.
    FullAttention,
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
                // Dense attention covers every block from the root.
                HybridGroupKind::FullAttention => {
                    if start == 0 {
                        end
                    } else {
                        0
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridEventGroup {
    pub group_idx: u32,
    /// The verified cache block size in tokens, not the hash granularity.
    pub block_size: u32,
    pub kind: HybridGroupKind,
}

pub(super) fn event_kind_matches(kind: HybridGroupKind, store: &BlockStored) -> bool {
    let spec = store.kv_cache_spec_kind.as_deref();
    match kind {
        HybridGroupKind::FullAttention => {
            spec == Some("full_attention") && store.kv_cache_spec_sliding_window.is_none()
        }
        HybridGroupKind::Checkpoint => {
            spec == Some("mamba") && store.kv_cache_spec_sliding_window.is_none()
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
        HybridGroupKind::Checkpoint => store.block_hashes.len() <= blocks,
        _ => store.block_hashes.len() == blocks,
    }
}

// ---- Layout declaration (from `--kv-hybrid-layout`) ----

use serde::{Deserialize, Serialize};

use super::super::subscriber::EventSource;
use super::HybridEventIndex;

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
    /// hashes the verbatim string, so an equivalent but different spelling is
    /// a different chain and credits nothing.
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
    pub model: String,
    pub weight_version: String,
    pub weight_identity: Option<String>,
    pub publisher_topic: Option<String>,
    pub instance_topic: bool,
    pub hash_algorithm: String,
    pub root_hash: Option<String>,
    pub full_hashes: bool,
    pub hash_block_size: Option<u32>,
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
        for group in &rank.groups {
            // A replay window or a group without shareable state is not
            // modeled: such a layout gets no credit rather than one the engine
            // cannot honor.
            if group.prefix_replay_tokens > 0 || !group.prefix_cacheable {
                return Err("unsupported group restore contract");
            }
            let kind = match (group.kind.as_str(), group.spec_type.as_str()) {
                ("full_attention", "FullAttentionSpec") if group.sliding_window.is_none() => {
                    HybridGroupKind::FullAttention
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
        // hybrid cache: several groups, at least one attention group, and a
        // checkpoint group that a single attention hit cannot prove.
        let has_attention = groups
            .iter()
            .any(|group| matches!(group.kind, HybridGroupKind::FullAttention));
        let has_state = groups
            .iter()
            .any(|group| matches!(group.kind, HybridGroupKind::Checkpoint));
        if rank.groups.len() < 2 || groups.is_empty() || !has_attention || !has_state {
            return Err("not a supported mixed layout");
        }
        let adapters = validate_adapters(&self.adapters)?;
        HybridEventIndex::new_with_root_hash(target, groups, hash_block_size, root, max_entries)
            .map(|index| index.with_adapters(adapters))
            .map_err(|_| "invalid group layout")
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

    mod declarations {
        use super::super::*;
        use std::sync::Arc;

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
    }
}
