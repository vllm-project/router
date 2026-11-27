//! Policy for allowing local prefill on an already selected Decode worker.
use crate::config::{PdPrefillBypassConfig, PdPrefillBypassPolicy};
use serde_json::Value;

pub(super) const LIMIT_FIELD: &str = "max_local_prefill_tokens";
pub(super) const ERROR_CODE: &str = "local_prefill_limit_exceeded";
pub(super) const ERROR_HEADER: &str = "x-vllm-error-code";

#[derive(Clone, Copy)]
pub(super) struct PrefillEstimate {
    pub prompt_tokens: usize,
    pub remaining_tokens: usize,
    pub worker_revision: u64,
}

impl PrefillEstimate {
    pub fn within_limit(self, limit: usize) -> bool {
        limit > 0 && self.remaining_tokens < self.prompt_tokens && self.remaining_tokens <= limit
    }

    pub fn allows(self, config: &PdPrefillBypassConfig, other_prefills: Option<usize>) -> bool {
        if !self.within_limit(config.max_tokens) {
            return false;
        }
        let suffix = self.remaining_tokens as f64 <= config.max_ratio * self.prompt_tokens as f64;
        let busy = config
            .p_busy_requests
            .zip(other_prefills)
            .is_some_and(|(threshold, count)| count >= threshold);
        match config.policy {
            PdPrefillBypassPolicy::ShortSuffix => suffix,
            PdPrefillBypassPolicy::BusyPrefill => busy,
            PdPrefillBypassPolicy::ShortSuffixOrBusyPrefill => suffix || busy,
        }
    }
}

pub(super) fn eligible_request(path: &str, request: &Value) -> bool {
    if ["kv_transfer_params", "ec_transfer_params", "prompt_embeds"]
        .iter()
        .any(|key| request.get(key).is_some_and(|value| !value.is_null()))
        || ["n", "best_of"].iter().any(|key| {
            request
                .get(key)
                .is_some_and(|value| !value.is_null() && value.as_u64() != Some(1))
        })
        || request.get("use_beam_search").and_then(Value::as_bool) == Some(true)
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
                        message.get("content").is_some_and(|content| {
                            content.is_string()
                                || content.as_array().is_some_and(|parts| {
                                    parts.iter().all(|part| {
                                        part.get("type").and_then(Value::as_str) == Some("text")
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

pub(super) fn http_rejected(status: http::StatusCode, headers: &http::HeaderMap) -> bool {
    status == http::StatusCode::UNPROCESSABLE_ENTITY
        && headers
            .get(ERROR_HEADER)
            .and_then(|value| value.to_str().ok())
            == Some(ERROR_CODE)
}

pub(super) fn local_params(limit: usize) -> Value {
    serde_json::json!({
        "do_remote_prefill": false,
        "do_remote_decode": false,
        (LIMIT_FIELD): limit,
    })
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;

    #[test]
    fn policy_respects_ratio_load_and_hard_limit() {
        let mut config = PdPrefillBypassConfig {
            max_tokens: 256,
            max_ratio: 0.25,
            p_busy_requests: Some(4),
            ..Default::default()
        };
        let mut estimate = PrefillEstimate {
            prompt_tokens: 400,
            remaining_tokens: 100,
            worker_revision: 1,
        };
        assert!(estimate.allows(&config, None));
        estimate.remaining_tokens = 150;
        assert!(!estimate.allows(&config, Some(4)));
        config.policy = PdPrefillBypassPolicy::BusyPrefill;
        assert!(!estimate.allows(&config, Some(3)));
        assert!(estimate.allows(&config, Some(4)));
        assert!(!estimate.allows(&config, None));
        config.policy = PdPrefillBypassPolicy::ShortSuffixOrBusyPrefill;
        assert!(estimate.allows(&config, Some(4)));
        estimate.remaining_tokens = 100;
        assert!(estimate.allows(&config, None));
        estimate.remaining_tokens = 256;
        assert!(estimate.allows(&config, Some(4)));
        estimate.remaining_tokens = 257;
        assert!(!estimate.allows(&config, Some(4)));
        estimate.remaining_tokens = estimate.prompt_tokens;
        assert!(!estimate.allows(&config, Some(4)));
        config.max_tokens = 0;
        estimate.remaining_tokens = 1;
        assert!(!estimate.allows(&config, Some(4)));
    }

    #[test]
    fn bypass_only_accepts_single_text_requests_without_handoff() {
        let request = serde_json::json!({"prompt":[1,2,3]});
        assert!(eligible_request("/v1/completions", &request));
        for request in [
            serde_json::json!({"prompt":"text","n":2}),
            serde_json::json!({"prompt":["a","b"]}),
            serde_json::json!({"prompt":"text","kv_transfer_params":{}}),
            serde_json::json!({"prompt":"text","prompt_embeds":[]}),
        ] {
            assert!(!eligible_request("/v1/completions", &request));
        }
        assert!(!eligible_request(
            "/v1/chat/completions",
            &serde_json::json!({"messages":[{"content":[{"type":"image_url"}]}]})
        ));
    }
}
