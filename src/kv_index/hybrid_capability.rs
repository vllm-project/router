//! Admission of Router-configured hybrid cache layouts.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::hybrid::HybridGroupKind;
use super::hybrid_events::{HybridEventGroup, HybridEventIndex};
use super::subscriber::EventSource;
use super::RoutableTarget;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct HybridCapabilities {
    pub schema_version: u32,
    pub ranks: Vec<HybridRankCapability>,
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
        HybridEventIndex::new_with_root_hash(target, groups, hash_block_size, root, max_entries)
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
    for (out, pair) in root.iter_mut().zip(value.as_bytes().chunks_exact(2)) {
        *out = (nibble(pair[0])? << 4) | nibble(pair[1])?;
    }
    Some(root)
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
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
            .apply_payload(0, &crate::kv_index::hybrid_events::tests::qwen35_payload())
            .unwrap();
        let prompt: Vec<u32> = (0..1024).map(|i| 100 + i % 100).collect();
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 528);
        value["ranks"][0]["groups"][0]["prefill_checkpoint_alignment"] = serde_json::Value::Null;
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
