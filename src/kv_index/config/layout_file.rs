//! Operator-supplied layouts and their event-source mappings.

use std::{collections::HashMap, path::Path};

use super::{PdKvEventSources, StaticKvEventSources};
use crate::config::{ConfigError, ConfigResult, RoutingMode};
use crate::kv_index::CacheCapabilities;

#[derive(serde::Deserialize)]
struct WorkerKvLayout {
    #[serde(flatten)]
    layout: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    endpoints: Vec<String>,
    replay_endpoints: Option<Vec<String>>,
    #[serde(default)]
    compatible_decode_urls: Vec<String>,
}

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
            if rank.get("speculative_tokens").and_then(Value::as_u64) == Some(0) {
                rank.entry("hash_lookahead_tokens").or_insert(json!(0));
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

fn read_layout_file(path: &Path) -> ConfigResult<HashMap<String, WorkerKvLayout>> {
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
    for (worker, config) in layouts {
        let WorkerKvLayout {
            layout,
            endpoints,
            replay_endpoints,
            compatible_decode_urls,
        } = config;
        let layout: Option<CacheCapabilities> =
            if pd && !compatible_decode_urls.is_empty() && layout.is_empty() {
                None
            } else {
                let layout =
                    parse_kv_layout(layout).map_err(|error| ConfigError::ValidationFailed {
                        reason: format!("invalid KV cache layout for '{worker}': {error}"),
                    })?;
                if layout.ranks.is_empty() {
                    return Err(ConfigError::ValidationFailed {
                        reason: format!(
                            "KV cache layout for '{worker}' must declare at least one rank"
                        ),
                    });
                }
                Some(layout)
            };
        let rank = layout.as_ref().and_then(|layout| {
            if let Some(rank) = crate::backend::parse_dp_rank(&worker) {
                layout.rank_capability(rank)
            } else {
                layout.ranks.iter().min_by_key(|rank| rank.rank)
            }
        });
        let block_size = rank
            .and_then(|rank| rank.hash_block_size)
            .or_else(|| {
                layout
                    .as_ref()?
                    .ranks
                    .iter()
                    .find_map(|rank| rank.hash_block_size)
            })
            .unwrap_or(16);
        let model = rank
            .map(|rank| rank.model.clone())
            .filter(|model| !model.is_empty());
        if pd {
            let source = PdKvEventSources {
                endpoints,
                replay_endpoints,
                block_size,
                model,
                hybrid: layout,
                compatible_decode_urls,
            };
            if pd_kv_event_sources.insert(worker.clone(), source).is_some() {
                return Err(ConfigError::ValidationFailed {
                    reason: format!("duplicate PD KV event source for '{worker}'"),
                });
            }
        } else if let Some(source) = static_kv_event_sources.get_mut(&worker) {
            if !endpoints.is_empty() || replay_endpoints.is_some() {
                return Err(ConfigError::ValidationFailed {
                    reason: format!("duplicate KV event endpoints for '{worker}'"),
                });
            }
            source.hybrid = layout;
        } else {
            static_kv_event_sources.insert(
                worker,
                StaticKvEventSources {
                    endpoints,
                    replay_endpoints,
                    topic: rank
                        .and_then(|rank| rank.publisher_topic.clone())
                        .unwrap_or_default(),
                    block_size,
                    model: model.unwrap_or_default(),
                    hybrid: layout,
                },
            );
        }
    }
    Ok(())
}
