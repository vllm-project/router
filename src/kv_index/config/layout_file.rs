//! Operator-supplied layouts and their event-source mappings.

use std::{collections::HashMap, path::Path};

use super::{PdKvEventSources, StaticKvEventSources};
use crate::config::{ConfigError, ConfigResult, RoutingMode};
use crate::kv_index::CacheCapabilities;

/// Fill defaults for operator-supplied layouts. Worker reports use the strict wire schema.
fn parse_kv_layout(
    mut layout: serde_json::Map<String, serde_json::Value>,
) -> serde_json::Result<CacheCapabilities> {
    use serde_json::{json, Value};

    layout.entry("schema_version").or_insert(json!(1));
    if let Some(ranks) = layout.get_mut("ranks").and_then(Value::as_array_mut) {
        for rank in ranks.iter_mut().filter_map(Value::as_object_mut) {
            for (field, value) in [
                ("weight_version", json!("default")),
                ("instance_topic", json!(true)),
                ("hash_algorithm", json!("sha256_cbor")),
                ("full_hashes", json!(true)),
                ("prefix_caching", json!(true)),
            ] {
                rank.entry(field).or_insert(value);
            }
            if !rank.contains_key("root_hash") {
                rank.entry("hash_seed").or_insert(json!("vllm-none-hash"));
            }
            match rank.get("speculative_tokens").and_then(Value::as_u64) {
                Some(0) => {
                    rank.entry("hash_lookahead_tokens").or_insert(json!(0));
                }
                Some(_) if !rank.contains_key("hash_lookahead_tokens") => {
                    return Err(serde::de::Error::custom(
                        "set hash_lookahead_tokens to the worker's block hash lookahead when speculation is enabled",
                    ));
                }
                _ => {}
            }
            if let Some(groups) = rank.get_mut("groups").and_then(Value::as_array_mut) {
                for (idx, group) in groups.iter_mut().enumerate() {
                    let Some(group) = group.as_object_mut() else {
                        continue;
                    };
                    let kind = group.get("kind").and_then(Value::as_str);
                    let spec = match kind {
                        Some("full_attention") => Some("FullAttentionSpec"),
                        Some("mla_attention") => Some("MLAAttentionSpec"),
                        Some("sliding_window") => Some("SlidingWindowSpec"),
                        Some("sliding_window_mla") => Some("SlidingWindowMLASpec"),
                        Some("mamba") => Some("MambaSpec"),
                        Some("scratch") => Some("CircularBufferSpec"),
                        _ => None,
                    };
                    let cacheable = kind != Some("scratch")
                        && group
                            .get("prefix_replay_tokens")
                            .and_then(Value::as_u64)
                            .unwrap_or(0)
                            == 0;
                    if let Some(spec) = spec {
                        group.entry("spec_type").or_insert(json!(spec));
                    }
                    group.entry("group_idx").or_insert(json!(idx));
                    group.entry("prefix_cacheable").or_insert(json!(cacheable));
                    group.entry("prefix_replay_tokens").or_insert(json!(0));
                }
            }
        }
    }
    serde_json::from_value(Value::Object(layout))
}

fn read_layout_file(
    path: &Path,
) -> ConfigResult<HashMap<String, serde_json::Map<String, serde_json::Value>>> {
    let data = std::fs::read(path).map_err(|error| ConfigError::ValidationFailed {
        reason: format!("cannot read KV cache layout {}: {error}", path.display()),
    })?;
    serde_json::from_slice(&data).map_err(|error| ConfigError::ValidationFailed {
        reason: format!("invalid KV cache layout {}: {error}", path.display()),
    })
}

/// Apply a layout file to the event-source maps prepared by the launcher.
#[doc(hidden)]
pub fn apply_kv_layout_file(
    path: &Path,
    mode: &RoutingMode,
    static_kv_event_sources: &mut HashMap<String, StaticKvEventSources>,
    pd_kv_event_sources: &mut HashMap<String, PdKvEventSources>,
) -> ConfigResult<()> {
    let layouts = read_layout_file(path)?;
    let pd = matches!(mode, RoutingMode::VllmPrefillDecode { .. });
    for (worker, fields) in layouts {
        for field in ["endpoints", "replay_endpoints", "compatible_decode_urls"] {
            if fields.contains_key(field) {
                return Err(ConfigError::ValidationFailed {
                    reason: format!("'{field}' for '{worker}' belongs in the KV event source or P/D routing options, not the cache layout"),
                });
            }
        }
        if fields
            .get("ranks")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|ranks| {
                ranks.iter().any(|rank| {
                    rank.get("publisher_topic").is_some() || rank.get("instance_topic").is_some()
                })
            })
        {
            return Err(ConfigError::ValidationFailed {
                reason: format!("publisher topics for '{worker}' belong in --kv-event-topic or worker discovery"),
            });
        }
        let layout = parse_kv_layout(fields).map_err(|error| ConfigError::ValidationFailed {
            reason: format!("invalid KV cache layout for '{worker}': {error}"),
        })?;
        let rank = if let Some(rank) = crate::backend::parse_dp_rank(&worker) {
            layout.rank_capability(rank)
        } else {
            layout.ranks.iter().min_by_key(|rank| rank.rank)
        }
        .ok_or_else(|| ConfigError::ValidationFailed {
            reason: format!("KV cache layout for '{worker}' must declare at least one rank"),
        })?;
        let block_size = rank
            .hash_block_size
            .or_else(|| layout.ranks.iter().find_map(|rank| rank.hash_block_size));
        let model = (!rank.model.is_empty()).then(|| rank.model.clone());
        if pd {
            let source = pd_kv_event_sources
                .entry(worker)
                .or_insert(PdKvEventSources {
                    endpoints: Vec::new(),
                    replay_endpoints: None,
                    block_size: 16,
                    model: None,
                    hybrid: None,
                });
            if let Some(size) = block_size {
                source.block_size = size;
            }
            if model.is_some() {
                source.model = model;
            }
            source.hybrid = Some(layout);
        } else {
            let source =
                static_kv_event_sources
                    .entry(worker)
                    .or_insert_with(|| StaticKvEventSources {
                        endpoints: Vec::new(),
                        replay_endpoints: None,
                        topic: String::new(),
                        block_size: 16,
                        model: String::new(),
                        hybrid: None,
                    });
            if let Some(size) = block_size {
                source.block_size = size;
            }
            if let Some(model) = model {
                source.model = model;
            }
            source.hybrid = Some(layout);
        }
    }
    Ok(())
}
