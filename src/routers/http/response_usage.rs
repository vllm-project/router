//! HTTP adapters for the shared attempt accounting kernel.
use super::usage_merge;
use crate::kv_observation::{KvRouteAttempt, UnavailableReason, MAX_USAGE_BYTES};
use bytes::Bytes;
use futures_util::{stream, Stream, StreamExt};
use http::StatusCode;
use serde_json::Value;
use std::io;
use std::pin::Pin;
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
/// Program-tracked responses already buffer their body for completion accounting.
pub(super) fn observe_json_response(attempt: &KvRouteAttempt, status: StatusCode, body: &[u8]) {
    let observation = if let Some(reason) = http_failure(status) {
        Err(reason)
    } else if body.len() > MAX_USAGE_BYTES {
        Err(UnavailableReason::OversizedJson)
    } else {
        match serde_json::from_slice::<Value>(body) {
            Ok(value) => usage_merge::cached_tokens(&value).ok_or(UnavailableReason::MissingUsage),
            Err(_) => Err(UnavailableReason::MalformedJson),
        }
    };
    attempt.record_once(observation);
}

/// Forward each JSON fragment immediately, retaining only the observation budget.
pub(super) fn observe_json_stream<S, E>(
    source: S,
    attempt: Option<Arc<KvRouteAttempt>>,
    status: StatusCode,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, io::Error>> + Send>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: std::error::Error + Send + Sync + 'static,
{
    let Some(attempt) = attempt else {
        return Box::pin(source.map(|item| item.map_err(io::Error::other)));
    };
    let observation = JsonUsageObservation {
        attempt,
        status,
        buffer: Some(Vec::new()),
    };
    Box::pin(stream::unfold(
        (Box::pin(source), observation),
        |(mut source, mut observation)| async move {
            match source.next().await {
                Some(Ok(bytes)) => {
                    observation.push(&bytes);
                    Some((Ok(bytes), (source, observation)))
                }
                Some(Err(error)) => {
                    observe_unavailable(
                        &observation.attempt,
                        observation.status,
                        UnavailableReason::ResponseError,
                    );
                    Some((Err(io::Error::other(error)), (source, observation)))
                }
                None => {
                    if let Some(buffer) = &observation.buffer {
                        observe_json_response(&observation.attempt, observation.status, buffer);
                    } else {
                        observe_unavailable(
                            &observation.attempt,
                            observation.status,
                            UnavailableReason::OversizedJson,
                        );
                    }
                    None
                }
            }
        },
    ))
}

struct JsonUsageObservation {
    attempt: Arc<KvRouteAttempt>,
    status: StatusCode,
    // None means the response exceeded the observation budget.
    buffer: Option<Vec<u8>>,
}

impl JsonUsageObservation {
    fn push(&mut self, bytes: &[u8]) {
        let Some(buffer) = &mut self.buffer else {
            return;
        };
        if bytes.len() > MAX_USAGE_BYTES - buffer.len() {
            self.buffer = None;
            return;
        }
        let needed = buffer.len() + bytes.len();
        if needed > buffer.capacity() {
            let capacity = needed
                .max(buffer.capacity().saturating_mul(2))
                .min(MAX_USAGE_BYTES);
            buffer.reserve_exact(capacity - buffer.len());
        }
        buffer.extend_from_slice(bytes);
    }
}

impl Drop for JsonUsageObservation {
    fn drop(&mut self) {
        observe_unavailable(
            &self.attempt,
            self.status,
            UnavailableReason::AttemptCancelled,
        );
    }
}

pub(super) fn observe_merged_json(
    attempt: &KvRouteAttempt,
    merged: &Value,
    dialect: usage_merge::UsageDialect,
) {
    let observation = usage_merge::cached_tokens_for_dialect(merged, dialect)
        .ok_or(UnavailableReason::MissingUsage);
    attempt.record_once(observation);
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
    attempt.record_once(Err(reason));
}

/// Receives parsed values from the one HTTP SSE framer. Only KvRouteAttempt's
/// atomic finalization flag owns completion; parsing state is not a second receipt.
pub(super) struct SseUsageObservation {
    attempt: Arc<KvRouteAttempt>,
    status: StatusCode,
    observed_cached_tokens: Option<u64>,
    fallback_reason: Option<UnavailableReason>,
    incomplete: bool,
}

impl SseUsageObservation {
    pub(super) fn new(attempt: Arc<KvRouteAttempt>, status: StatusCode) -> Self {
        Self {
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
    pub(super) fn observe_value(
        &mut self,
        value: &Value,
        dialect: Option<usage_merge::UsageDialect>,
    ) {
        if self.attempt.is_finished() {
            return;
        }
        // Keep mode accepts either usage spelling. Reconcile mode supplies its
        // selected dialect so an untouched Decode alias cannot become a sample.
        let dialects = dialect.as_ref().map(std::slice::from_ref).unwrap_or(&[
            usage_merge::UsageDialect::Completions,
            usage_merge::UsageDialect::Responses,
        ]);
        let Some(cached_tokens) = dialects
            .iter()
            .find_map(|dialect| usage_merge::cached_tokens_for_dialect(value, *dialect))
        else {
            if usage_merge::usage_object(value).is_some_and(|usage| {
                dialects.iter().any(|dialect| {
                    usage
                        .get(dialect.keys().1)
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
            Err(reason)
        } else if self.incomplete {
            Err(UnavailableReason::IncompleteSseFrame)
        } else if !succeeded {
            Err(UnavailableReason::StreamError)
        } else if let Some(tokens) = self.observed_cached_tokens {
            Ok(tokens)
        } else {
            Err(UnavailableReason::MissingUsage)
        };
        self.attempt.record_once(observation);
    }
}
impl Drop for SseUsageObservation {
    fn drop(&mut self) {
        self.finish(false);
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
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
    fn json_stream_forwards_before_eof_and_rejects_failure_or_cancellation() {
        use futures_util::FutureExt;
        let body =
            Bytes::from_static(br#"{"usage":{"prompt_tokens_details":{"cached_tokens":32}}}"#);
        for (ending, reason) in [
            ("eof", "none"),
            ("error", "response_error"),
            ("drop", "attempt_cancelled"),
        ] {
            let metrics = render(|| {
                let source = stream::iter([Ok::<_, io::Error>(body.clone())]);
                let source: Pin<Box<dyn Stream<Item = Result<Bytes, io::Error>> + Send>> =
                    match ending {
                        "error" => Box::pin(
                            source.chain(stream::iter([Err(io::Error::other("backend failed"))])),
                        ),
                        "drop" => Box::pin(source.chain(stream::pending())),
                        _ => Box::pin(source),
                    };
                let attempt = KvRouteAttempt::new(prediction());
                let mut output = observe_json_stream(source, Some(attempt.clone()), StatusCode::OK);
                assert_eq!(
                    output.next().now_or_never().unwrap().unwrap().unwrap(),
                    body
                );
                assert!(
                    !attempt.is_finished(),
                    "a complete usage object is provisional until EOF"
                );
                match ending {
                    "error" => assert!(output.next().now_or_never().unwrap().unwrap().is_err()),
                    "drop" => assert!(output.next().now_or_never().is_none()),
                    _ => assert!(output.next().now_or_never().unwrap().is_none()),
                }
                drop(output);
                assert!(attempt.is_finished());
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
                "{ending}: {metrics}"
            );
            assert_eq!(
                metrics.contains("vllm_router_kv_actual_cached_tokens_count 1"),
                reason == "none",
                "{metrics}"
            );
        }
    }

    #[test]
    fn json_stream_budget_bounds_allocation_and_preserves_oversized_bodies() {
        use futures_util::FutureExt;
        for size in [MAX_USAGE_BYTES - 1, MAX_USAGE_BYTES, MAX_USAGE_BYTES + 1] {
            let mut body = br#"{"usage":{"prompt_tokens_details":{"cached_tokens":32}}}"#.to_vec();
            body.resize(size, b' ');
            let mut observation = JsonUsageObservation {
                attempt: KvRouteAttempt::new(prediction()),
                status: StatusCode::OK,
                buffer: Some(Vec::new()),
            };
            for chunk in body.chunks(17_000) {
                observation.push(chunk);
                assert!(observation
                    .buffer
                    .as_ref()
                    .is_none_or(|buffer| buffer.capacity() <= MAX_USAGE_BYTES));
            }
            assert_eq!(observation.buffer.is_some(), size <= MAX_USAGE_BYTES);
            drop(observation);
            let metrics = render(|| {
                let source = stream::iter(
                    body.chunks(17_000)
                        .map(Bytes::copy_from_slice)
                        .map(Ok::<_, io::Error>)
                        .collect::<Vec<_>>(),
                );
                let output = observe_json_stream(
                    source,
                    Some(KvRouteAttempt::new(prediction())),
                    StatusCode::OK,
                );
                let received = output
                    .collect::<Vec<_>>()
                    .now_or_never()
                    .unwrap()
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                assert_eq!(received.concat(), body);
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
            assert_eq!(
                metrics.contains("vllm_router_kv_actual_cached_tokens_count 1"),
                size <= MAX_USAGE_BYTES,
                "{metrics}"
            );
        }
    }
}
