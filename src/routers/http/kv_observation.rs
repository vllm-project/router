//! HTTP adapters for the shared attempt accounting kernel.
use super::usage_merge;
use crate::kv_observation::{CacheObservation, KvRouteAttempt, UnavailableReason, MAX_USAGE_BYTES};
use http::StatusCode;
use serde_json::Value;
use std::sync::Arc;

fn http_failure(status: StatusCode) -> Option<UnavailableReason> {
    if status.is_client_error() {
        Some(UnavailableReason::RequestRejected)
    } else if !status.is_success() {
        Some(UnavailableReason::BackendError)
    } else {
        None
    }
}

/// All colocated JSON forwarding paths use this same observation budget.
/// The existing non-streaming body buffer still returns every response byte.
pub(super) fn observe_json_response(attempt: &KvRouteAttempt, status: StatusCode, body: &[u8]) {
    let observation = if let Some(reason) = http_failure(status) {
        CacheObservation::Unavailable(reason)
    } else if body.len() > MAX_USAGE_BYTES {
        CacheObservation::Unavailable(UnavailableReason::OversizedJson)
    } else {
        match serde_json::from_slice::<Value>(body) {
            Ok(value) => usage_merge::cached_tokens(&value)
                .map(CacheObservation::Observed)
                .unwrap_or(CacheObservation::Unavailable(
                    UnavailableReason::MissingUsage,
                )),
            Err(_) => CacheObservation::Unavailable(UnavailableReason::MalformedJson),
        }
    };
    attempt.record_once(observation);
}

pub(super) fn observe_merged_json(attempt: &KvRouteAttempt, merged: &Value, prefill: u64) {
    let observation = usage_merge::cached_tokens(merged)
        .map(CacheObservation::Observed)
        .unwrap_or(CacheObservation::Unavailable(
            UnavailableReason::MissingUsage,
        ));
    attempt.record_once(check_merged_usage(observation, Some(prefill)));
}

fn check_merged_usage(observation: CacheObservation, expected: Option<u64>) -> CacheObservation {
    // Reconciliation can leave a frame unchanged (for example an incompatible
    // usage dialect). Its count is then the original Decode value, not the
    // selected Prefill's reuse. Do not count that passthrough as a prediction error.
    match (observation, expected) {
        (CacheObservation::Observed(actual), Some(prefill)) if actual != prefill => {
            CacheObservation::Unavailable(UnavailableReason::PdUsageNotComparable)
        }
        _ => observation,
    }
}

pub(super) fn observe_unavailable(
    attempt: &KvRouteAttempt,
    status: StatusCode,
    reason: UnavailableReason,
) {
    let reason = if reason == UnavailableReason::RequestError {
        reason
    } else {
        http_failure(status).unwrap_or(reason)
    };
    attempt.record_once(CacheObservation::Unavailable(reason));
}

/// Receives parsed values from the one HTTP SSE framer. Only KvRouteAttempt's
/// atomic finalization flag owns completion; parsing state is not a second receipt.
pub(super) struct SseUsageObservation {
    merged_prefill: Option<u64>,
    attempt: Arc<KvRouteAttempt>,
    status: StatusCode,
    observed_cached_tokens: Option<u64>,
    fallback_reason: Option<UnavailableReason>,
    incomplete: bool,
}

impl SseUsageObservation {
    pub(super) fn new(
        attempt: Arc<KvRouteAttempt>,
        status: StatusCode,
        merged_prefill: Option<u64>,
    ) -> Self {
        Self {
            merged_prefill,
            attempt,
            status,
            observed_cached_tokens: None,
            fallback_reason: None,
            incomplete: false,
        }
    }
    pub(super) fn reject(&mut self, reason: UnavailableReason) {
        self.fallback_reason = Some(reason);
    }
    pub(super) fn set_incomplete(&mut self, incomplete: bool) {
        self.incomplete = incomplete;
    }
    pub(super) fn observe_value(&mut self, value: &Value) {
        if self.attempt.is_finished() {
            return;
        }
        let Some(cached_tokens) = usage_merge::cached_tokens(value) else {
            if usage_merge::usage_object(value).is_some_and(|usage| {
                ["prompt_tokens_details", "input_tokens_details"]
                    .iter()
                    .any(|key| {
                        usage
                            .get(*key)
                            .and_then(|details| details.get("cached_tokens"))
                            .is_some_and(|cached| !cached.is_null())
                    })
            }) {
                self.reject(UnavailableReason::InvalidUsage);
            }
            return;
        };
        if self
            .observed_cached_tokens
            .is_some_and(|previous| previous != cached_tokens)
        {
            self.reject(UnavailableReason::InvalidUsage);
        } else {
            self.observed_cached_tokens = Some(cached_tokens);
        }
    }
    pub(super) fn finish(&mut self, succeeded: bool) {
        if self.attempt.is_finished() {
            return;
        }
        let observation = if let Some(reason) = http_failure(self.status).or(self.fallback_reason) {
            CacheObservation::Unavailable(reason)
        } else if self.incomplete {
            CacheObservation::Unavailable(UnavailableReason::IncompleteSseFrame)
        } else if let Some(tokens) = self.observed_cached_tokens {
            CacheObservation::Observed(tokens)
        } else if !succeeded {
            CacheObservation::Unavailable(UnavailableReason::StreamError)
        } else {
            CacheObservation::Unavailable(UnavailableReason::MissingUsage)
        };
        self.attempt
            .record_once(check_merged_usage(observation, self.merged_prefill));
    }
}
impl Drop for SseUsageObservation {
    fn drop(&mut self) {
        self.finish(false);
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::kv_observation::test_helpers::*;

    #[test]
    fn json_usage_dialects_and_http_failures_use_the_shared_receipt() {
        for (status, body, reason) in [
            (
                StatusCode::OK,
                r#"{"usage":{"prompt_tokens_details":{"cached_tokens":32}}}"#,
                "none",
            ),
            (
                StatusCode::OK,
                r#"{"response":{"usage":{"input_tokens":64,"input_tokens_details":{"cached_tokens":32}}}}"#,
                "none",
            ),
            (StatusCode::OK, "bad json", "malformed_json"),
            (StatusCode::OK, "{}", "missing_usage"),
            (StatusCode::BAD_REQUEST, "bad json", "request_rejected"),
            (StatusCode::BAD_GATEWAY, "bad json", "backend_error"),
        ] {
            let metrics = render(|| {
                observe_json_response(&KvRouteAttempt::new(prediction()), status, body.as_bytes())
            });
            assert!(
                has_outcome(
                    &metrics,
                    if reason == "none" {
                        "observed"
                    } else {
                        "unavailable"
                    },
                    reason
                ),
                "{metrics}"
            );
            assert_eq!(
                metrics.contains("vllm_router_kv_actual_cached_tokens_count 1"),
                reason == "none",
                "{metrics}"
            );
        }
    }

    #[test]
    fn json_observation_budget_includes_both_boundary_bytes() {
        for size in [MAX_USAGE_BYTES - 1, MAX_USAGE_BYTES, MAX_USAGE_BYTES + 1] {
            let mut body = br#"{"usage":{"prompt_tokens_details":{"cached_tokens":32}}}"#.to_vec();
            body.resize(size, b' ');
            let metrics = render(|| {
                observe_json_response(&KvRouteAttempt::new(prediction()), StatusCode::OK, &body)
            });
            assert!(
                has_outcome(
                    &metrics,
                    if size <= MAX_USAGE_BYTES {
                        "observed"
                    } else {
                        "unavailable"
                    },
                    if size <= MAX_USAGE_BYTES {
                        "none"
                    } else {
                        "oversized_json"
                    }
                ),
                "size={size}: {metrics}"
            );
        }
    }

    #[test]
    fn passthrough_pd_count_must_still_belong_to_prefill() {
        for (actual, reason) in [(32, "none"), (16, "pd_usage_not_comparable")] {
            let metrics = render(|| {
                observe_merged_json(
                    &KvRouteAttempt::new(prediction()),
                    &serde_json::json!({"usage":{"prompt_tokens_details":{"cached_tokens":actual}}}),
                    32,
                )
            });
            assert!(
                has_outcome(
                    &metrics,
                    if actual == 32 {
                        "observed"
                    } else {
                        "unavailable"
                    },
                    reason
                ),
                "{metrics}"
            );
        }
    }
}
