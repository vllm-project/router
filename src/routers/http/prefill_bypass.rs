//! Scheduler-independent policy and worker contract for local prefill.

use crate::backend::preprocess::http_cache_read_fallback;
use axum::response::Response;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;

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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, clap::Args)]
#[serde(default)]
pub struct PdPrefillBypassConfig {
    /// Enable Prefill bypass with the default thresholds.
    #[arg(id = "pd_prefill_bypass_enabled", long = "pd-prefill-bypass")]
    pub enabled: bool,
    /// Override the local prefill token limit (default 2048 when enabled). Zero disables bypass.
    #[arg(
        id = "pd_prefill_bypass_max_tokens",
        long = "pd-prefill-bypass-max-tokens"
    )]
    pub max_tokens: Option<usize>,
    /// Maximum fraction of the input left to prefill in the suffix check.
    #[arg(
        id = "pd_prefill_bypass_max_ratio",
        long = "pd-prefill-bypass-max-ratio",
        default_value_t = 0.7
    )]
    pub max_ratio: f64,
    /// Policy used to decide whether to bypass Prefill.
    #[arg(
        id = "pd_prefill_bypass_policy",
        long = "pd-prefill-bypass-policy",
        value_enum,
        default_value = "short_suffix"
    )]
    pub policy: PdPrefillBypassPolicy,
    /// Minimum number of other Prefill requests for Prefill to be busy.
    #[arg(
        id = "pd_prefill_bypass_p_busy_requests",
        long = "pd-prefill-bypass-p-busy-requests"
    )]
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
        self.max_tokens.unwrap_or(2048)
    }

    pub(super) fn is_enabled(self) -> bool {
        (self.enabled || self.max_tokens.is_some()) && self.token_limit() > 0
    }

    pub(super) fn validate(self) -> Result<(), String> {
        if !(0.0 < self.max_ratio && self.max_ratio <= 1.0) {
            return Err("pd-prefill-bypass-max-ratio must be in (0, 1]".into());
        }
        if self.is_enabled()
            && self.policy.uses_prefill_load()
            && self.p_busy_requests.unwrap_or(0) == 0
        {
            return Err("the prefill bypass load policy requires a positive pd-prefill-bypass-p-busy-requests".into());
        }
        Ok(())
    }
}

/// Estimated prompt and remaining work, based on verified Decode-local KV.
#[derive(Clone, Copy)]
pub(super) struct PrefillEstimate {
    pub prompt_tokens: usize,
    pub remaining_tokens: usize,
}

/// Prefill bypass policy and worker contract. Admission, worker liveness and
/// stream ownership belong to the dispatch adapter.
#[derive(Debug)]
pub(super) struct PrefillBypass {
    config: PdPrefillBypassConfig,
}

impl PrefillBypass {
    pub(super) fn enabled(config: PdPrefillBypassConfig) -> Option<Self> {
        config.is_enabled().then_some(Self { config })
    }

    pub(super) fn needs_prefill_load(&self) -> bool {
        self.config.policy.uses_prefill_load()
    }

    /// A caller should supply a request with compatible local-prefill semantics
    /// before looking up KV.
    pub(super) fn eligible_request(path: &str, request: &Value) -> bool {
        http_cache_read_fallback(request).is_none() && eligible_request(path, request)
    }

    pub(super) fn within_limit(&self, estimate: PrefillEstimate) -> bool {
        estimate.remaining_tokens < estimate.prompt_tokens
            && estimate.remaining_tokens <= self.config.token_limit()
    }

    pub(super) fn transfer_params(&self) -> Value {
        local_params(self.config.token_limit())
    }

    /// Check the caller's verified local KV and load observed at dispatch.
    /// Load counts exclude this request's own reservations.
    pub(super) fn allows(
        &self,
        estimate: PrefillEstimate,
        decode_idle: bool,
        other_prefills: Option<usize>,
    ) -> bool {
        if !decode_idle || !self.within_limit(estimate) {
            return false;
        }
        let suffix = estimate.remaining_tokens < self.config.token_limit()
            && (estimate.remaining_tokens as f64)
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

const LIMIT_FIELD: &str = "max_local_prefill_tokens";
const ERROR_CODE: &str = "local_prefill_limit_exceeded";
pub(super) const ERROR_HEADER: &str = "x-vllm-error-code";

pub(super) fn http_rejected(status: http::StatusCode, headers: &http::HeaderMap) -> bool {
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

pub(super) fn grpc_rejected(status: &tonic::Status) -> bool {
    status.code() == tonic::Code::FailedPrecondition
        && status
            .metadata()
            .get(ERROR_HEADER)
            .and_then(|value| value.to_str().ok())
            == Some(ERROR_CODE)
}

pub(super) enum PrefillDispatch<T> {
    Split(T),
    Respond(Response),
}

pub(super) enum BypassAttempt<T> {
    Split(T),
    Respond(Response),
    /// Decode refused before generation and the caller released its attempt.
    Refused,
}

/// The caller owns admission and attempt cleanup. The callback receives `true`
/// only after a typed refusal, and that admission goes straight to the split path.
pub(super) async fn admit_with_bypass<T, A, AFut, B, BFut>(
    mut admit: A,
    bypass: B,
) -> PrefillDispatch<T>
where
    A: FnMut(bool) -> AFut,
    AFut: Future<Output = Result<T, Box<Response>>>,
    B: FnOnce(T) -> BFut,
    BFut: Future<Output = BypassAttempt<T>>,
{
    let initial = match admit(false).await {
        Ok(attempt) => attempt,
        Err(response) => return PrefillDispatch::Respond(*response),
    };
    match bypass(initial).await {
        BypassAttempt::Split(attempt) => PrefillDispatch::Split(attempt),
        BypassAttempt::Respond(response) => PrefillDispatch::Respond(response),
        BypassAttempt::Refused => match admit(true).await {
            Ok(attempt) => PrefillDispatch::Split(attempt),
            Err(response) => PrefillDispatch::Respond(*response),
        },
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn typed_refusal_readmits_without_trying_bypass_again() {
        let mut admissions = Vec::new();
        let outcome = admit_with_bypass(
            |readmit| {
                admissions.push(readmit);
                std::future::ready(Ok(admissions.len()))
            },
            |attempt| async move {
                assert_eq!(attempt, 1);
                BypassAttempt::Refused
            },
        )
        .await;
        assert!(matches!(outcome, PrefillDispatch::Split(2)));
        assert_eq!(admissions, [false, true]);
    }

    #[test]
    fn opt_in_defaults_require_both_suffix_thresholds() {
        let mut config = PdPrefillBypassConfig::default();
        assert!(PrefillBypass::enabled(config).is_none());
        config.enabled = true;
        assert!(PrefillBypass::enabled(config).is_some());
        let bypass = PrefillBypass::enabled(config).unwrap();
        assert_eq!(bypass.transfer_params()[LIMIT_FIELD], 2048);
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
        assert!(PrefillBypass::enabled(config).is_none());
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
        estimate.remaining_tokens = 256;
        assert!(PrefillBypass { config }.allows(estimate, true, Some(4)));
        estimate.remaining_tokens = 257;
        assert!(!PrefillBypass { config }.allows(estimate, true, Some(4)));
        estimate.remaining_tokens = estimate.prompt_tokens;
        assert!(!PrefillBypass { config }.allows(estimate, true, Some(4)));
        config.max_tokens = Some(0);
        assert!(PrefillBypass::enabled(config).is_none());
    }

    #[test]
    fn bypass_only_accepts_single_text_requests_without_handoff() {
        let request = serde_json::json!({"prompt":[1,2,3]});
        assert!(PrefillBypass::eligible_request("/v1/completions", &request));
        for request in [
            serde_json::json!({"prompt":"text","n":2}),
            serde_json::json!({"prompt":["a","b"]}),
            serde_json::json!({"prompt":"text","kv_transfer_params":{}}),
            serde_json::json!({"prompt":"text","prompt_embeds":[]}),
        ] {
            assert!(!PrefillBypass::eligible_request(
                "/v1/completions",
                &request
            ));
        }
        assert!(!PrefillBypass::eligible_request(
            "/v1/chat/completions",
            &serde_json::json!({"messages":[{"content":[{"type":"image_url"}]}]})
        ));
    }
}
