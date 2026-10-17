//! Opt-in policy backed by fenced, engine-owned KV event state.
use super::{LoadBalancingPolicy, RequestHeaders};
use crate::config::types::KvAwareConfig;
use crate::core::Worker;
use crate::kv_index::routing::KvRoutingDecision;
use crate::kv_index::runtime::KvRuntime;
use std::sync::Arc;

#[derive(Debug)]
pub struct KvAwarePolicy {
    pub runtime: Arc<KvRuntime>,
}

impl Default for KvAwarePolicy {
    fn default() -> Self {
        Self::with_config(KvAwareConfig::default())
    }
}
impl KvAwarePolicy {
    pub fn with_config(config: KvAwareConfig) -> Self {
        Self {
            runtime: Arc::new(KvRuntime::new(config)),
        }
    }
    pub fn select_with_tokens(
        &self,
        workers: &[Arc<dyn Worker>],
        model: Option<&str>,
        tokens: Option<&[u32]>,
        namespace_safe: bool,
    ) -> Option<(usize, KvRoutingDecision)> {
        self.runtime.select(workers, model, tokens, namespace_safe)
    }
}
impl LoadBalancingPolicy for KvAwarePolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        _: Option<&str>,
        _: Option<&RequestHeaders>,
    ) -> Option<usize> {
        self.select_with_tokens(workers, None, None, false)
            .map(|(idx, _)| idx)
    }
    fn name(&self) -> &'static str {
        "kv_aware"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Request metadata only narrows eligibility. It can never establish trust.
pub(crate) fn namespace_safe(body: &serde_json::Value) -> bool {
    for field in [
        "cache_salt",
        "lora",
        "lora_name",
        "lora_path",
        "multi_modal_data",
        "prompt_embeds",
        "mm_processor_kwargs",
        "truncate_prompt_tokens",
    ] {
        if body.get(field).is_some_and(|v| !v.is_null()) {
            return false;
        }
    }
    if body.get("add_special_tokens").and_then(|v| v.as_bool()) == Some(true) {
        return false;
    }
    body.get("messages")
        .and_then(|v| v.as_array())
        .is_none_or(|messages| {
            messages.iter().all(|message| {
                message.get("content").is_none_or(|content| match content {
                    serde_json::Value::Null | serde_json::Value::String(_) => true,
                    serde_json::Value::Array(parts) => parts
                        .iter()
                        .all(|part| part.get("type").and_then(|v| v.as_str()) == Some("text")),
                    _ => false,
                })
            })
        })
}
