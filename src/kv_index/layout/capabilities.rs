//! Verified engine declarations, independent of worker discovery and storage.
use super::{CacheGroup, GroupRule};
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
    /// Cache groups whose engine lookup drops a trailing proof unit. When
    /// speculation uses legacy hashes, the engine must report this list, even
    /// when it is empty. The speculative token count alone cannot identify it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tail_drop_groups: Option<Vec<u32>>,
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

impl RankCapability {
    pub(crate) fn is_plain_full_attention(&self) -> bool {
        self.declared_groups.is_none_or(|count| count == 1)
            && self.hash_lookahead_tokens == 0
            && self.speculative_tokens == 0
            && !self.partial_hash_hits
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
    pub(crate) root: SequenceHash,
    pub(crate) replay_tokens: u32,
    pub(crate) adapters: Vec<(String, String)>,
    pub(crate) partial_hash_hits: bool,
}

impl CacheCapabilities {
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
        if model.is_some_and(|model| rank.model != model) {
            return Err("model does not match source model");
        }
        let topic = rank.publisher_topic.as_deref();
        if topic.is_none_or(str::is_empty) {
            return Err("invalid capability or source binding");
        }
        if rank.model.is_empty()
            || rank.weight_version.is_empty()
            || !rank.instance_topic
            || observed_topic.is_some_and(|observed| topic != Some(observed))
            || rank.hash_algorithm != "sha256_cbor"
            || !rank.full_hashes
            || !rank.prefix_caching
        {
            return Err("unsupported engine identity or hash contract");
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
                    GroupRule::FullAttention
                }
                ("mla_attention", "MLAAttentionSpec")
                    if group.sliding_window.is_none()
                        && group
                            .cache_role
                            .as_deref()
                            .is_none_or(|role| role == "sparse") =>
                {
                    // A compressed group names the tokens one state covers.
                    // Without that declaration, restore from the root like a
                    // full-attention block instead of rejecting the layout.
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
                        // No declaration: the page restores from the root.
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
                ("mamba", "MambaSpec")
                    if group.mamba_cache_mode.as_deref() == Some("align")
                        && group.prefill_checkpoint_alignment.is_none_or(|alignment| {
                            alignment > 0 && group.block_size.is_multiple_of(alignment)
                        }) =>
                {
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
        let tail_drop_group = if tail_drop_groups.is_empty() {
            None
        } else {
            let mut dense = groups
                .iter()
                .filter(|group| group.kind == GroupRule::FullAttention);
            let group = dense.next().ok_or("unsupported tail-drop layout")?;
            if dense.next().is_some()
                || tail_drop_groups != [group.group_idx]
                || groups.len() != rank.groups.len()
                || !rank.groups.iter().any(|declared| {
                    declared.group_idx == group.group_idx
                        && declared.spec_type == "FullAttentionSpec"
                })
                || replay_tokens != 0
                || groups.iter().any(|other| {
                    !matches!(other.kind, GroupRule::FullAttention | GroupRule::Checkpoint)
                        || (!rank.partial_hash_hits && other.block_size != group.block_size)
                })
            {
                return Err("unsupported tail-drop layout");
            }
            Some(group.group_idx)
        };
        Ok(VerifiedLayout {
            groups,
            hash_block_size,
            hash_lookahead_tokens: rank.hash_lookahead_tokens,
            tail_drop_group,
            root,
            replay_tokens,
            adapters: validate_adapters(&self.adapters)?,
            partial_hash_hits: rank.partial_hash_hits,
        })
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
