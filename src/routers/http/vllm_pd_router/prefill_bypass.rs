//! Scheduler-independent policy and worker contract for local prefill.

use crate::backend::preprocess::http_cache_read_fallback;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const DEFAULT_LOCAL_PREFILL_LIMIT: usize = 2048;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum PdPrefillBypassPolicy {
    #[default]
    #[value(name = "short_suffix")]
    ShortSuffix,
    #[value(name = "busy_prefill")]
    BusyPrefill,
    #[value(name = "short_suffix_or_busy_prefill")]
    ShortSuffixOrBusyPrefill,
}

impl PdPrefillBypassPolicy {
    fn uses_prefill_load(self) -> bool {
        self != Self::ShortSuffix
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PdPrefillBypassConfig {
    /// Enable Prefill bypass with the default thresholds.
    pub enabled: bool,
    /// Override the local prefill token limit (default 2048 when enabled). Zero disables bypass.
    pub max_tokens: Option<usize>,
    /// Maximum fraction of the input left to prefill in the suffix check.
    pub max_ratio: f64,
    /// Policy used to decide whether to bypass Prefill.
    pub policy: PdPrefillBypassPolicy,
    /// Minimum number of other Prefill requests for Prefill to be busy.
    pub p_busy_requests: Option<usize>,
}

impl Default for PdPrefillBypassConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_tokens: None,
            max_ratio: 0.7,
            policy: PdPrefillBypassPolicy::ShortSuffix,
            p_busy_requests: None,
        }
    }
}

impl PdPrefillBypassConfig {
    fn token_limit(self) -> usize {
        self.max_tokens.unwrap_or(DEFAULT_LOCAL_PREFILL_LIMIT)
    }
}

/// Estimated prompt and remaining work, based on verified Decode-local KV.
#[derive(Clone, Copy)]
pub(in crate::routers::http) struct PrefillEstimate {
    pub prompt_tokens: usize,
    pub remaining_tokens: usize,
}

/// Prefill bypass policy and worker contract. Admission, worker liveness and
/// stream ownership belong to the dispatch adapter.
#[derive(Debug)]
pub(crate) struct PrefillBypass {
    config: PdPrefillBypassConfig,
}

impl PrefillBypass {
    pub(super) fn from_config(config: PdPrefillBypassConfig) -> Result<Option<Self>, String> {
        let enabled = (config.enabled || config.max_tokens.is_some()) && config.token_limit() > 0;
        if !(0.0 < config.max_ratio && config.max_ratio <= 1.0) {
            return Err("pd-prefill-bypass-max-ratio must be in (0, 1]".into());
        }
        if enabled && config.policy.uses_prefill_load() && config.p_busy_requests.unwrap_or(0) == 0
        {
            return Err("the prefill bypass load policy requires a positive pd-prefill-bypass-p-busy-requests".into());
        }
        Ok(enabled.then_some(Self { config }))
    }

    pub(in crate::routers::http) fn needs_prefill_load(&self) -> bool {
        self.config.policy.uses_prefill_load()
    }

    /// A caller should supply a request with compatible local-prefill semantics
    /// before looking up KV.
    pub(in crate::routers::http) fn eligible_request(path: &str, request: &Value) -> bool {
        http_cache_read_fallback(request).is_none() && eligible_request(path, request)
    }

    pub(in crate::routers::http) fn within_limit(&self, estimate: PrefillEstimate) -> bool {
        estimate.remaining_tokens < estimate.prompt_tokens
            && estimate.remaining_tokens < self.config.token_limit()
    }

    pub(in crate::routers::http) fn transfer_params(&self) -> Value {
        // The worker's maximum is inclusive. The router's threshold is exclusive.
        local_params(self.config.token_limit() - 1)
    }

    /// Check the caller's verified local KV and load observed at dispatch.
    /// Load counts exclude this request's own reservations.
    pub(in crate::routers::http) fn allows(
        &self,
        estimate: PrefillEstimate,
        decode_idle: bool,
        other_prefills: Option<usize>,
    ) -> bool {
        if !decode_idle || !self.within_limit(estimate) {
            return false;
        }
        let suffix = (estimate.remaining_tokens as f64)
            < self.config.max_ratio * estimate.prompt_tokens as f64;
        let busy = self
            .config
            .p_busy_requests
            .zip(other_prefills)
            .is_some_and(|(threshold, count)| count >= threshold);
        match self.config.policy {
            PdPrefillBypassPolicy::ShortSuffix => suffix,
            PdPrefillBypassPolicy::BusyPrefill => busy,
            PdPrefillBypassPolicy::ShortSuffixOrBusyPrefill => suffix || busy,
        }
    }
}

fn eligible_request(path: &str, request: &Value) -> bool {
    if ["kv_transfer_params", "ec_transfer_params", "prompt_embeds"]
        .iter()
        .any(|key| request.get(key).is_some_and(|value| !value.is_null()))
        || ["n", "best_of"].iter().any(|key| {
            request
                .get(key)
                .is_some_and(|value| !value.is_null() && value.as_u64() != Some(1))
        })
        || ["use_beam_search", "echo"]
            .iter()
            .any(|key| request.get(key).and_then(Value::as_bool) == Some(true))
        || ["logprobs", "top_logprobs"].iter().any(|key| {
            request
                .get(key)
                .is_some_and(|value| !matches!(value, Value::Null | Value::Bool(false)))
        })
    {
        return false;
    }
    match path {
        "/v1/chat/completions" => request
            .get("messages")
            .and_then(Value::as_array)
            .is_some_and(|messages| {
                !messages.is_empty()
                    && messages.iter().all(|message| {
                        ["tool_calls", "function_call"]
                            .iter()
                            .all(|key| message.get(key).is_none_or(Value::is_null))
                            && message.get("content").is_some_and(|content| {
                                content.is_string()
                                    || content.as_array().is_some_and(|parts| {
                                        !parts.is_empty()
                                            && parts.iter().all(|part| {
                                                part.get("type").and_then(Value::as_str)
                                                    == Some("text")
                                            })
                                    })
                            })
                    })
            }),
        "/v1/completions" => request.get("prompt").is_some_and(|prompt| {
            prompt.is_string()
                || prompt.as_array().is_some_and(|tokens| {
                    !tokens.is_empty() && tokens.iter().all(|token| token.as_u64().is_some())
                })
        }),
        _ => false,
    }
}

const LIMIT_FIELD: &str = "max_local_prefill_tokens";
const ERROR_CODE: &str = "local_prefill_limit_exceeded";
pub(in crate::routers::http) const ERROR_HEADER: &str = "x-vllm-error-code";

pub(in crate::routers::http) fn http_rejected(
    status: http::StatusCode,
    headers: &http::HeaderMap,
) -> bool {
    status == http::StatusCode::UNPROCESSABLE_ENTITY
        && headers
            .get(ERROR_HEADER)
            .and_then(|value| value.to_str().ok())
            == Some(ERROR_CODE)
}

fn local_params(limit: usize) -> Value {
    serde_json::json!({
        "do_remote_prefill": false,
        "do_remote_decode": false,
        (LIMIT_FIELD): limit,
    })
}

pub(in crate::routers::http) fn grpc_rejected(status: &tonic::Status) -> bool {
    status.code() == tonic::Code::FailedPrecondition
        && status
            .metadata()
            .get(ERROR_HEADER)
            .and_then(|value| value.to_str().ok())
            == Some(ERROR_CODE)
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn opt_in_defaults_require_both_suffix_thresholds() {
        let mut config = PdPrefillBypassConfig::default();
        assert!(PrefillBypass::from_config(config).unwrap().is_none());
        config.enabled = true;
        let bypass = PrefillBypass::from_config(config).unwrap().unwrap();
        assert_eq!(bypass.transfer_params()[LIMIT_FIELD], 2047);
        let mut estimate = PrefillEstimate {
            prompt_tokens: 10_000,
            remaining_tokens: 2047,
        };
        assert!(bypass.allows(estimate, true, None));
        assert!(!bypass.allows(estimate, false, None));
        estimate.remaining_tokens = 2048;
        assert!(!bypass.allows(estimate, true, None));
        estimate.prompt_tokens = 1000;
        estimate.remaining_tokens = 699;
        assert!(bypass.allows(estimate, true, None));
        estimate.remaining_tokens = 700;
        assert!(!bypass.allows(estimate, true, None));
        config.max_tokens = Some(0);
        assert!(PrefillBypass::from_config(config).unwrap().is_none());
    }

    #[test]
    fn validates_configuration_before_installation() {
        let config = PdPrefillBypassConfig {
            max_ratio: 0.0,
            ..Default::default()
        };
        assert!(PrefillBypass::from_config(config).is_err());

        let config = PdPrefillBypassConfig {
            policy: PdPrefillBypassPolicy::BusyPrefill,
            max_tokens: Some(8),
            ..Default::default()
        };
        assert!(PrefillBypass::from_config(config).is_err());
        let configured = PdPrefillBypassConfig {
            p_busy_requests: Some(1),
            ..config
        };
        assert!(PrefillBypass::from_config(configured).unwrap().is_some());
    }

    #[test]
    fn policy_respects_ratio_load_and_hard_limit() {
        let mut config = PdPrefillBypassConfig {
            max_tokens: Some(256),
            max_ratio: 0.25,
            p_busy_requests: Some(4),
            ..Default::default()
        };
        let mut estimate = PrefillEstimate {
            prompt_tokens: 400,
            remaining_tokens: 99,
        };
        assert!(PrefillBypass { config }.allows(estimate, true, None));
        estimate.remaining_tokens = 150;
        assert!(!PrefillBypass { config }.allows(estimate, true, Some(4)));
        config.policy = PdPrefillBypassPolicy::BusyPrefill;
        assert!(!PrefillBypass { config }.allows(estimate, true, Some(3)));
        assert!(PrefillBypass { config }.allows(estimate, true, Some(4)));
        assert!(!PrefillBypass { config }.allows(estimate, true, None));
        config.policy = PdPrefillBypassPolicy::ShortSuffixOrBusyPrefill;
        assert!(PrefillBypass { config }.allows(estimate, true, Some(4)));
        estimate.remaining_tokens = 99;
        assert!(PrefillBypass { config }.allows(estimate, true, None));
        estimate.remaining_tokens = 255;
        assert!(PrefillBypass { config }.allows(estimate, true, Some(4)));
        estimate.remaining_tokens = 256;
        assert!(!PrefillBypass { config }.allows(estimate, true, Some(4)));
        estimate.remaining_tokens = estimate.prompt_tokens;
        assert!(!PrefillBypass { config }.allows(estimate, true, Some(4)));
        config.max_tokens = Some(0);
        assert!(PrefillBypass::from_config(config).unwrap().is_none());
    }

    #[test]
    fn bypass_only_accepts_single_text_requests_without_handoff() {
        let request = serde_json::json!({"prompt":[1,2,3]});
        assert!(PrefillBypass::eligible_request("/v1/completions", &request));
        for request in [
            serde_json::json!({"prompt":"text","n":2}),
            serde_json::json!({"prompt":"text","echo":true}),
            serde_json::json!({"prompt":"text","logprobs":0}),
            serde_json::json!({"prompt":["a","b"]}),
            serde_json::json!({"prompt":"text","kv_transfer_params":{}}),
            serde_json::json!({"prompt":"text","prompt_embeds":[]}),
        ] {
            assert!(!PrefillBypass::eligible_request(
                "/v1/completions",
                &request
            ));
        }
        assert!(PrefillBypass::eligible_request(
            "/v1/completions",
            &serde_json::json!({"prompt":"text","echo":false,"logprobs":null})
        ));
        for request in [
            serde_json::json!({"messages":[{"content":"text"}],"logprobs":true}),
            serde_json::json!({"messages":[{"content":"text"}],"top_logprobs":1}),
            serde_json::json!({"messages":[{"role":"assistant","content":"text","tool_calls":[{"id":"call"}]}]}),
        ] {
            assert!(!PrefillBypass::eligible_request(
                "/v1/chat/completions",
                &request
            ));
        }
        for content in [
            serde_json::json!([]),
            serde_json::json!([{"type": "image_url"}]),
        ] {
            assert!(!PrefillBypass::eligible_request(
                "/v1/chat/completions",
                &serde_json::json!({"messages": [{"content": content}]})
            ));
        }
    }
}
