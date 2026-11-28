//! Verified engine declarations, independent of worker discovery and storage.
use super::{common_block_alignment, AlignedTailDrop, CacheGroup, DenseLookup, GroupRule};
use crate::kv_index::request_hash::{BlockKeyGenerator, SequenceHash};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

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
    /// Number of group declarations in this layout, including listed groups
    /// that do not publish cache events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_groups: Option<u32>,
    #[serde(default)]
    pub model: String,
    pub weight_version: String,
    pub publisher_topic: Option<String>,
    pub instance_topic: bool,
    pub hash_algorithm: String,
    pub root_hash: Option<String>,
    /// Exact resolved vLLM seed; permits deriving the root without hand-written hex.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash_seed: Option<String>,
    pub full_hashes: bool,
    pub hash_block_size: Option<u32>,
    /// Source publishes hash-granularity stores as partial prefix entries.
    /// Enable only for a verified engine event contract.
    #[serde(default)]
    pub partial_hash_hits: bool,
    pub hash_lookahead_tokens: u32,
    pub speculative_tokens: u32,
    /// Groups whose cache lookup actually drops a trailing proof unit for this
    /// worker role. With speculation and zero hash lookahead, report an empty
    /// list if the engine skips the drop. EAGLE group IDs describe model
    /// membership and may differ from this list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tail_drop_groups: Option<Vec<u32>>,
    /// Ordered runtime lookup batches for fine-grained speculation with
    /// multiple dense groups. Groups in one batch must match the same hash.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dense_lookup_groups: Option<Vec<Vec<u32>>>,
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
    pub tokens_per_state: Option<String>,
    pub cache_role: Option<String>,
}

impl RankCapability {
    pub(crate) fn reported_hash_mismatch<'a>(
        &'a self,
        reported: Option<&'a str>,
    ) -> Option<(&'a str, &'a str)> {
        let reported = reported?;
        let expected = self.hash_algorithm.as_str();
        (expected != reported).then_some((expected, reported))
    }

    pub(crate) fn has_plain_full_attention_shape(&self) -> bool {
        self.declared_groups == Some(1)
            && self.hash_lookahead_tokens == 0
            && self.speculative_tokens == 0
            && !self.partial_hash_hits
            && self.tail_drop_groups.as_ref().is_none_or(Vec::is_empty)
            && self.dense_lookup_groups.is_none()
            && matches!(
                self.groups.as_slice(),
                [group] if self.hash_block_size == Some(group.block_size)
                    && group.group_idx == 0
                    && group.kind == "full_attention"
                    && group.spec_type == "FullAttentionSpec"
                    && group.sliding_window.is_none()
                    && group.prefix_cacheable
                    && group.prefix_replay_tokens == 0
            )
    }
}
/// Adapter declarations are the only way to credit adapter pages, so a
/// malformed declaration must fail the layout instead of binding a chain that
/// is not the one the engine stores.
fn validate_adapters(
    adapters: &[AdapterCapability],
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

pub(crate) fn valid_groups(groups: &[CacheGroup], hash_block_size: usize) -> bool {
    let mut ids = HashSet::new();
    hash_block_size != 0
        && !groups.is_empty()
        && groups.len() <= 64
        && groups.iter().all(|group| {
            group.block_size != 0
                && (group.block_size as usize).is_multiple_of(hash_block_size)
                && ids.insert(group.group_idx)
                && match group.kind {
                    GroupRule::SlidingWindow { window_tokens } => window_tokens != 0,
                    GroupRule::CompressedFullAttention { tokens_per_state } => {
                        tokens_per_state != 0 && group.block_size.is_multiple_of(tokens_per_state)
                    }
                    _ => true,
                }
        })
}

/// The decoded cache contract. Validation does not allocate an index or owner.
pub(crate) struct VerifiedLayout {
    pub(crate) groups: Vec<CacheGroup>,
    pub(crate) hash_block_size: u32,
    pub(crate) hash_lookahead_tokens: u32,
    pub(crate) tail_drop_group: Option<u32>,
    pub(in crate::kv_index) aligned_tail_drop: Option<AlignedTailDrop>,
    pub(in crate::kv_index) dense_lookups: Vec<DenseLookup>,
    pub(crate) root: SequenceHash,
    pub(crate) replay_tokens: u32,
    pub(crate) adapters: Vec<(String, String)>,
    pub(crate) partial_hash_hits: bool,
}

impl CacheCapabilities {
    pub(crate) fn is_plain_full_attention(&self) -> bool {
        !self.ranks.is_empty()
            && self
                .ranks
                .iter()
                .all(RankCapability::has_plain_full_attention_shape)
    }

    pub(crate) fn matches_hash_block_size(
        &self,
        rank: Option<u32>,
        size: u32,
        discover: bool,
    ) -> bool {
        let matches = |rank: &RankCapability| {
            rank.hash_block_size
                .map_or(discover, |hash_size| hash_size == size)
        };
        if let Some(rank) = rank {
            return self.rank_capability(rank).is_some_and(matches);
        }
        !self.ranks.is_empty() && self.ranks.iter().all(matches)
    }

    /// The capability declared for one DP rank. Ranks are independent
    /// instances: a rank that is absent here simply gets no hybrid credit and
    /// must not invalidate the other ranks of the same worker.
    pub fn rank_capability(&self, rank: u32) -> Option<&RankCapability> {
        self.ranks.iter().find(|entry| entry.rank == rank)
    }

    pub(crate) fn verify_layout(
        &self,
        dp_rank: u32,
        model: Option<&str>,
        observed_topic: Option<&str>,
        replay_endpoint: Option<&str>,
    ) -> Result<VerifiedLayout, &'static str> {
        let mut ranks = HashSet::new();
        if self.schema_version != 1
            || self.ranks.iter().any(|entry| !ranks.insert(entry.rank))
            || observed_topic.is_some_and(str::is_empty)
            || replay_endpoint.is_none_or(str::is_empty)
        {
            return Err("invalid capability or source binding");
        }
        let rank = self
            .rank_capability(dp_rank)
            .ok_or("missing rank capability")?;
        if rank
            .declared_groups
            .is_some_and(|count| count as usize != rank.groups.len())
        {
            return Err("declared group count does not match layout");
        }
        if model.is_some_and(|model| rank.model != model) {
            return Err("model does not match source model");
        }
        let topic = rank.publisher_topic.as_deref();
        if topic.is_none_or(str::is_empty) {
            return Err("invalid capability or source binding");
        }
        if rank.model.is_empty() || rank.weight_version.is_empty() {
            return Err("missing engine model or weight version");
        }
        if !rank.instance_topic || observed_topic.is_some_and(|observed| topic != Some(observed)) {
            return Err("layout publisher topic does not identify the observed worker instance");
        }
        if rank.hash_algorithm != "sha256_cbor" || !rank.full_hashes || !rank.prefix_caching {
            return Err("KV layout requires prefix caching and full-width sha256_cbor hashes");
        }
        let tail_drop_groups = rank.tail_drop_groups.as_deref().unwrap_or_default();
        if (rank.speculative_tokens == 0 && rank.hash_lookahead_tokens != 0)
            || (rank.speculative_tokens != 0
                && rank.hash_lookahead_tokens == 0
                && rank.tail_drop_groups.is_none())
            || ((rank.hash_lookahead_tokens != 0 || rank.speculative_tokens == 0)
                && !tail_drop_groups.is_empty())
        {
            return Err("unsupported speculative cache contract");
        }
        let hash_block_size = rank.hash_block_size.ok_or("missing hash block size")?;
        if hash_block_size == 0 {
            return Err("invalid hash block size");
        }
        let derived = match rank.hash_seed.as_deref() {
            Some("") => return Err("invalid hash seed"),
            Some(seed) => {
                Some(BlockKeyGenerator::from_seed(hash_block_size as usize, seed).root_hash())
            }
            None => None,
        };
        let root = match (rank.root_hash.as_deref(), derived) {
            (Some(hex), seed_root) => {
                let root = parse_hash(hex).ok_or("invalid root hash")?;
                if seed_root.is_some_and(|seed_root| seed_root != root) {
                    return Err("hash seed and root disagree");
                }
                root
            }
            (None, Some(root)) => root,
            (None, None) => return Err("missing root hash or resolved seed"),
        };
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
                    GroupRule::FullAttention
                }
                ("mla_attention", "MLAAttentionSpec")
                    if group.sliding_window.is_none()
                        && group
                            .cache_role
                            .as_deref()
                            .is_none_or(|role| matches!(role, "sparse" | "indexer")) =>
                {
                    // Sparse MLA and its persistent indexer use the same
                    // full-prefix lookup with an explicit token-to-state ratio.
                    match group.tokens_per_state.as_deref() {
                        // Declared but malformed: the engine would not honor
                        // this shape, so the layout is refused.
                        Some(declared) => {
                            let tokens_per_state = declared
                                .parse::<u32>()
                                .ok()
                                .ok_or("unsupported compressed attention group")?;
                            GroupRule::CompressedFullAttention { tokens_per_state }
                        }
                        None if group.cache_role.is_some() => {
                            return Err("MLA cache role requires tokens_per_state");
                        }
                        None => GroupRule::FullAttention,
                    }
                }
                ("sliding_window", "SlidingWindowSpec")
                | ("sliding_window_mla", "SlidingWindowMLASpec") => {
                    let window = group
                        .sliding_window
                        .filter(|window| *window > 0)
                        .ok_or("unsupported sliding window group")?;
                    GroupRule::SlidingWindow {
                        window_tokens: u64::from(window),
                    }
                }
                ("mamba", "MambaSpec") if group.mamba_cache_mode.as_deref() == Some("align") => {
                    GroupRule::Checkpoint
                }
                _ => return Err("unsupported group kind"),
            };
            groups.push(CacheGroup {
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
                GroupRule::FullAttention
                    | GroupRule::CompressedFullAttention { .. }
                    | GroupRule::SlidingWindow { .. }
            )
        });
        let has_state = groups.iter().any(|group| {
            matches!(
                group.kind,
                GroupRule::Checkpoint | GroupRule::SlidingWindow { .. }
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
        if !valid_groups(&groups, hash_block_size as usize) {
            return Err("invalid group layout");
        }
        // Circular buffers are excluded from vLLM's cache lookup even when
        // the coordinator lists them as draft groups.
        let tail_groups: Vec<_> = tail_drop_groups
            .iter()
            .copied()
            .filter(|id| {
                !rank.groups.iter().any(|group| {
                    group.group_idx == *id
                        && !group.prefix_cacheable
                        && group.spec_type == "CircularBufferSpec"
                })
            })
            .collect();
        let dense_lookups = if let Some(lookups) = &rank.dense_lookup_groups {
            verify_dense_lookups(
                &groups,
                lookups,
                &tail_groups,
                rank.partial_hash_hits,
                replay_tokens,
            )?
        } else {
            Vec::new()
        };
        let (tail_drop_group, aligned_tail_drop) = if tail_groups.is_empty()
            || !dense_lookups.is_empty()
        {
            (None, None)
        } else {
            let unsupported = "unsupported tail-drop layout. Set tail_drop_groups to the effective groups and use [] when the worker skips the drop";
            if replay_tokens != 0 {
                return Err(unsupported);
            }
            let single_dense = tail_groups.as_slice().first().and_then(|id| {
                let group = groups.iter().find(|group| group.group_idx == *id)?;
                (tail_groups.len() == 1
                    && group.kind.uses_full_attention_lookup()
                    && groups.iter().all(|other| {
                        (other.group_idx == *id || other.kind == GroupRule::Checkpoint)
                            && (rank.partial_hash_hits || other.block_size == group.block_size)
                    }))
                .then_some(*id)
            });
            if single_dense.is_some() {
                (single_dense, None)
            } else {
                let mut distinct = HashSet::new();
                // These layouts use whole scheduler units. Every group in the
                // alignment must also participate in the reported cache contract.
                if rank.partial_hash_hits
                    || groups.len() != rank.groups.len()
                    || !tail_groups.iter().all(|id| {
                        distinct.insert(*id)
                            && groups.iter().any(|group| {
                                group.group_idx == *id && group.kind != GroupRule::Checkpoint
                            })
                    })
                {
                    return Err(unsupported);
                }
                let alignment = common_block_alignment(groups.iter().map(|group| group.block_size))
                    .ok_or(unsupported)?;
                (
                    None,
                    Some(AlignedTailDrop {
                        groups: tail_groups,
                        alignment,
                    }),
                )
            }
        };
        Ok(VerifiedLayout {
            groups,
            hash_block_size,
            hash_lookahead_tokens: rank.hash_lookahead_tokens,
            tail_drop_group,
            aligned_tail_drop,
            dense_lookups,
            root,
            replay_tokens,
            adapters: validate_adapters(&self.adapters)?,
            partial_hash_hits: rank.partial_hash_hits,
        })
    }
}

fn verify_dense_lookups(
    groups: &[CacheGroup],
    lookups: &[Vec<u32>],
    tail_groups: &[u32],
    partial_hash_hits: bool,
    replay_tokens: u32,
) -> Result<Vec<DenseLookup>, &'static str> {
    let error =
        "dense_lookup_groups must report every full-attention group once, in engine lookup order";
    let tail: HashSet<_> = tail_groups.iter().copied().collect();
    if !partial_hash_hits
        || replay_tokens != 0
        || tail.is_empty()
        || tail.len() != tail_groups.len()
        || groups
            .iter()
            .any(|group| !matches!(group.kind, GroupRule::FullAttention | GroupRule::Checkpoint))
        || tail.iter().any(|id| {
            !groups
                .iter()
                .any(|group| group.group_idx == *id && group.kind == GroupRule::FullAttention)
        })
    {
        return Err(error);
    }
    let mut seen = HashSet::new();
    let mut verified = Vec::new();
    for ids in lookups {
        let drop_tail = tail.contains(ids.first().ok_or(error)?);
        let mut positions = Vec::new();
        for id in ids {
            let position = groups
                .iter()
                .position(|group| group.group_idx == *id && group.kind == GroupRule::FullAttention)
                .ok_or(error)?;
            if !seen.insert(*id)
                || tail.contains(id) != drop_tail
                || positions.first().is_some_and(|&first: &usize| {
                    groups[first].block_size != groups[position].block_size
                })
            {
                return Err(error);
            }
            positions.push(position);
        }
        verified.push(DenseLookup {
            positions,
            drop_tail,
        });
    }
    if seen.len() < 2
        || seen.len()
            != groups
                .iter()
                .filter(|group| group.kind == GroupRule::FullAttention)
                .count()
    {
        return Err(error);
    }
    Ok(verified)
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
