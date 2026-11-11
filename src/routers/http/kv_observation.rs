use super::{sse::sse_lines, sse::FrameScanner, usage_merge};
use crate::kv_index::RoutableTarget;
use crate::metrics::RouterMetrics;
use bytes::Bytes;
use http::StatusCode;
use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const MAX_OBSERVATION_BYTES: usize = 1024 * 1024;

pub(super) struct KvResponseStream<S> {
    inner: std::pin::Pin<Box<S>>,
    attempt: Arc<KvRouteAttempt>,
    sse: Option<KvSseObserver>,
    status: StatusCode,
    json: Vec<u8>,
    oversized: bool,

    finished: bool,
}

impl<S> KvResponseStream<S> {
    pub(super) fn new(
        inner: S,
        attempt: Arc<KvRouteAttempt>,
        status: StatusCode,
        streaming: bool,
    ) -> Self {
        Self {
            inner: Box::pin(inner),
            sse: streaming.then(|| KvSseObserver::new(attempt.clone(), status)),
            attempt,
            status,
            json: Vec::new(),
            oversized: false,

            finished: false,
        }
    }
}

impl<S, E> futures_util::Stream for KvResponseStream<S>
where
    S: futures_util::Stream<Item = Result<Bytes, E>>,
{
    type Item = Result<Bytes, E>;
    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let result = this.inner.as_mut().poll_next(context);
        match &result {
            std::task::Poll::Ready(Some(Ok(bytes))) => {
                if let Some(sse) = &mut this.sse {
                    sse.observe(bytes);
                } else if !this.oversized {
                    if this.json.len().saturating_add(bytes.len()) > MAX_OBSERVATION_BYTES {
                        this.oversized = true;
                        this.json.clear();
                    } else {
                        this.json.extend_from_slice(bytes);
                    }
                }
            }
            std::task::Poll::Ready(Some(Err(_))) => {
                this.finished = true;
                if let Some(sse) = &mut this.sse {
                    sse.finish(false);
                } else {
                    observe_unavailable(
                        &this.attempt,
                        this.status,
                        UnavailableReason::ResponseError,
                    );
                }
            }
            std::task::Poll::Ready(None) if !this.finished => {
                this.finished = true;
                if let Some(sse) = &mut this.sse {
                    sse.finish(true);
                } else if this.oversized {
                    observe_unavailable(
                        &this.attempt,
                        this.status,
                        UnavailableReason::OversizedJson,
                    );
                } else {
                    observe_json_response(&this.attempt, this.status, &this.json);
                }
            }
            _ => {}
        }
        result
    }
}

#[derive(Debug, Clone)]
pub(super) struct KvRoutePrediction {
    pub(super) target: RoutableTarget,
    pub(super) cached_tokens: Option<u64>,
    pub(super) multi_prompt: bool,
}

pub(super) struct KvRouteAttempt {
    prediction: KvRoutePrediction,
    finished: AtomicBool,
}

impl KvRouteAttempt {
    pub(super) fn new(prediction: KvRoutePrediction) -> Arc<Self> {
        Arc::new(Self {
            prediction,
            finished: AtomicBool::new(false),
        })
    }

    fn record_once(&self, observation: CacheObservation) {
        if self
            .finished
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            record_observation(&self.prediction, observation);
        }
    }

    #[cfg(test)]
    fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }
}

impl Drop for KvRouteAttempt {
    fn drop(&mut self) {
        if !self.finished.load(Ordering::Acquire) {
            self.record_once(CacheObservation::Unavailable(
                UnavailableReason::AttemptCancelled,
            ));
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum UnavailableReason {
    AttemptCancelled,
    BackendError,
    RequestRejected,
    RequestError,
    ResponseError,
    MissingUsage,
    MultiPromptUsageNotAggregate,
    MalformedJson,
    OversizedJson,
    OversizedSseFrame,
    IncompleteSseFrame,
    InvalidUsage,
    MalformedUsage,
    StreamError,
    PdUsageNotComparable,
}

impl UnavailableReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::AttemptCancelled => "attempt_cancelled",
            Self::BackendError => "backend_error",
            Self::RequestRejected => "request_rejected",
            Self::RequestError => "request_error",
            Self::ResponseError => "response_error",
            Self::MissingUsage => "missing_usage",
            Self::MultiPromptUsageNotAggregate => "multi_prompt_usage_not_aggregate",
            Self::MalformedJson => "malformed_json",
            Self::OversizedJson => "oversized_json",
            Self::OversizedSseFrame => "oversized_sse_frame",
            Self::IncompleteSseFrame => "incomplete_sse_frame",
            Self::InvalidUsage => "invalid_usage",
            Self::MalformedUsage => "malformed_usage",
            Self::StreamError => "stream_error",
            Self::PdUsageNotComparable => "pd_usage_not_comparable",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheObservation {
    Observed(u64),
    Ambiguous,
    Unavailable(UnavailableReason),
}

fn record_observation(prediction: &KvRoutePrediction, observation: CacheObservation) {
    let observation = if prediction.multi_prompt
        && matches!(
            observation,
            CacheObservation::Observed(_) | CacheObservation::Ambiguous
        ) {
        CacheObservation::Unavailable(UnavailableReason::MultiPromptUsageNotAggregate)
    } else {
        observation
    };
    let (status, reason, actual_cached_tokens) = match observation {
        CacheObservation::Observed(tokens) => ("observed", "none", Some(tokens)),
        CacheObservation::Ambiguous => ("ambiguous", "conflicting_usage", None),
        CacheObservation::Unavailable(reason) => ("unavailable", reason.as_str(), None),
    };

    RouterMetrics::record_kv_route_outcome(
        status,
        reason,
        prediction.cached_tokens,
        actual_cached_tokens,
    );
    tracing::debug!(
        status,
        reason,
        instance_id = %prediction.target.instance_id,
        dp_rank = prediction.target.dp_rank,
        predicted_cached_tokens = ?prediction.cached_tokens,
        actual_cached_tokens,
        "KV route prediction outcome"
    );
}

fn http_failure(status: StatusCode) -> Option<UnavailableReason> {
    if status.is_client_error() {
        Some(UnavailableReason::RequestRejected)
    } else if !status.is_success() {
        Some(UnavailableReason::BackendError)
    } else {
        None
    }
}

/// A Program response may report reuse for a different worker than the selection.
pub(super) fn comparable_attempt(
    attempt: Option<Arc<KvRouteAttempt>>,
    status: StatusCode,
    unattributed: bool,
) -> Option<Arc<KvRouteAttempt>> {
    if unattributed {
        if let Some(attempt) = attempt {
            observe_unavailable(&attempt, status, UnavailableReason::PdUsageNotComparable);
        }
        None
    } else {
        attempt
    }
}

pub(super) fn observe_json_response(attempt: &KvRouteAttempt, status: StatusCode, body: &[u8]) {
    let observation = if let Some(reason) = http_failure(status) {
        CacheObservation::Unavailable(reason)
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

pub(super) struct KvSseObserver {
    scanner: FrameScanner,
    frame: Vec<u8>,
    discarding_frame: bool,
    usage: SseUsageObservation,
}

pub(super) struct SseUsageObservation {
    merged_prefill: Option<u64>,
    attempt: Arc<KvRouteAttempt>,
    status: StatusCode,
    observed_cached_tokens: Option<u64>,
    ambiguous: bool,
    fallback_reason: Option<UnavailableReason>,
    incomplete: bool,
    finished: bool,
}

impl KvSseObserver {
    pub(super) fn new(attempt: Arc<KvRouteAttempt>, status: StatusCode) -> Self {
        Self {
            scanner: FrameScanner::default(),
            frame: Vec::new(),
            discarding_frame: false,
            usage: SseUsageObservation::new(attempt, status, None),
        }
    }

    pub(super) fn observe(&mut self, chunk: &Bytes) {
        if !self.usage.status.is_success() || self.usage.finished {
            return;
        }

        let mut remaining = chunk.as_ref();
        while !remaining.is_empty() {
            let frame_end = self.scanner.frame_end(remaining);
            let consumed = frame_end.unwrap_or(remaining.len());
            if !self.discarding_frame {
                if self.frame.len().saturating_add(consumed) > MAX_OBSERVATION_BYTES {
                    self.frame.clear();
                    self.discarding_frame = true;
                    self.usage.reject(UnavailableReason::OversizedSseFrame);
                } else {
                    self.frame.extend_from_slice(&remaining[..consumed]);
                }
            }
            remaining = &remaining[consumed..];

            if frame_end.is_some() {
                if self.discarding_frame {
                    self.discarding_frame = false;
                } else {
                    let frame = std::mem::take(&mut self.frame);
                    self.observe_frame(&frame);
                }
            }
        }
    }

    fn observe_frame(&mut self, frame: &[u8]) {
        if !frame
            .windows(b"\"usage\"".len())
            .any(|window| window == b"\"usage\"")
        {
            return;
        }

        let mut data = Vec::new();
        for (line, _) in sse_lines(frame) {
            let Some(value) = line.strip_prefix(b"data:") else {
                continue;
            };
            if !data.is_empty() {
                data.push(b'\n');
            }
            data.extend_from_slice(value.strip_prefix(b" ").unwrap_or(value));
        }
        if data.is_empty() || data == b"[DONE]" {
            return;
        }

        match serde_json::from_slice::<Value>(&data) {
            Ok(value) => self.usage.observe_value(&value),
            Err(_) => self.usage.reject(UnavailableReason::MalformedUsage),
        }
    }

    pub(super) fn finish(&mut self, succeeded: bool) {
        self.usage
            .set_incomplete(self.frame.iter().any(|b| !matches!(b, b'\r' | b'\n')));
        self.usage.finish(succeeded);
    }
}

impl SseUsageObservation {
    pub(super) fn new(
        attempt: Arc<KvRouteAttempt>,
        status: StatusCode,
        merged_prefill: Option<u64>,
    ) -> Self {
        Self {
            attempt,
            status,
            merged_prefill,
            observed_cached_tokens: None,
            ambiguous: false,
            fallback_reason: None,
            incomplete: false,
            finished: false,
        }
    }

    pub(super) fn reject(&mut self, reason: UnavailableReason) {
        self.fallback_reason = Some(reason);
    }

    pub(super) fn set_incomplete(&mut self, incomplete: bool) {
        self.incomplete = incomplete;
    }

    pub(super) fn observe_value(&mut self, value: &Value) {
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
                self.fallback_reason = Some(UnavailableReason::InvalidUsage);
            }
            return;
        };
        if self
            .observed_cached_tokens
            .is_some_and(|previous| previous != cached_tokens)
        {
            self.ambiguous = true;
        } else {
            self.observed_cached_tokens = Some(cached_tokens);
        }
    }

    pub(super) fn finish(&mut self, stream_succeeded: bool) {
        let observation = if let Some(reason) = http_failure(self.status) {
            CacheObservation::Unavailable(reason)
        } else if let Some(reason) = self.fallback_reason {
            CacheObservation::Unavailable(reason)
        } else if self.incomplete {
            CacheObservation::Unavailable(UnavailableReason::IncompleteSseFrame)
        } else if self.ambiguous {
            CacheObservation::Ambiguous
        } else if let Some(tokens) = self.observed_cached_tokens {
            CacheObservation::Observed(tokens)
        } else if !stream_succeeded {
            CacheObservation::Unavailable(UnavailableReason::StreamError)
        } else {
            CacheObservation::Unavailable(UnavailableReason::MissingUsage)
        };
        self.record_once(observation);
    }

    fn record_once(&mut self, observation: CacheObservation) {
        if std::mem::replace(&mut self.finished, true) {
            return;
        }
        self.attempt
            .record_once(check_merged_usage(observation, self.merged_prefill));
    }
}

impl Drop for KvSseObserver {
    fn drop(&mut self) {
        if !self.usage.finished {
            self.finish(false);
        }
    }
}

impl Drop for SseUsageObservation {
    fn drop(&mut self) {
        if !self.finished {
            self.finish(false);
        }
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn prediction() -> KvRoutePrediction {
        KvRoutePrediction {
            target: RoutableTarget {
                instance_id: Arc::from("worker-a"),
                dp_rank: 0,
            },
            cached_tokens: Some(32),
            multi_prompt: false,
        }
    }

    #[test]
    fn program_usage_and_http_rejections_do_not_enter_accuracy_histograms() {
        for (status, reason) in [
            (StatusCode::OK, "pd_usage_not_comparable"),
            (StatusCode::BAD_REQUEST, "request_rejected"),
            (StatusCode::BAD_GATEWAY, "backend_error"),
        ] {
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            metrics::with_local_recorder(&recorder, || {
                let attempt = KvRouteAttempt::new(prediction());
                assert!(comparable_attempt(Some(attempt), status, true).is_none());
            });
            let metrics = handle.render();
            assert!(metrics.contains(&format!("reason=\"{reason}\"")));
            assert!(!metrics.contains("vllm_router_kv_predicted_cached_tokens"));
            assert!(!metrics.contains("vllm_router_kv_actual_cached_tokens"));
        }
    }

    #[test]
    fn accuracy_histograms_count_only_pairs_and_preserve_error_direction() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            for actual in [16, 32, 48] {
                record_observation(&prediction(), CacheObservation::Observed(actual));
            }
            drop(KvRouteAttempt::new(prediction()));
        });
        let metrics = handle.render();
        for name in ["predicted", "actual"] {
            assert!(metrics
                .lines()
                .any(|line| line == format!("vllm_router_kv_{name}_cached_tokens_count 3")));
        }
        for (direction, error) in [("underpredicted", 16), ("exact", 0), ("overpredicted", 16)] {
            assert!(metrics.lines().any(|line| line == format!("vllm_router_kv_cached_token_absolute_error_sum{{direction=\"{direction}\"}} {error}")));
        }
    }

    #[test]
    fn extracts_both_usage_dialects() {
        let completions = serde_json::json!({
            "usage": {
                "prompt_tokens": 64,
                "prompt_tokens_details": {"cached_tokens": 32}
            }
        });
        let responses = serde_json::json!({
            "response": {
                "usage": {
                    "input_tokens": 64,
                    "input_tokens_details": {"cached_tokens": 32}
                }
            }
        });
        assert_eq!(usage_merge::cached_tokens(&completions), Some(32));
        assert_eq!(usage_merge::cached_tokens(&responses), Some(32));
    }

    #[test]
    fn transparent_stream_preserves_bytes_and_records_json_and_sse() {
        use futures_util::{FutureExt, StreamExt};
        for streaming in [false, true] {
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            metrics::with_local_recorder(&recorder, || {
                let payload = if streaming {
                    Bytes::from_static(b"data: {\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":32}}}\n\ndata: [DONE]\n\n")
                } else {
                    Bytes::from_static(
                        br#"{"usage":{"prompt_tokens_details":{"cached_tokens":32}}}"#,
                    )
                };
                let chunks = vec![
                    Ok::<_, std::io::Error>(payload.slice(..7)),
                    Ok(payload.slice(7..)),
                ];
                let mut stream = KvResponseStream::new(
                    futures_util::stream::iter(chunks),
                    KvRouteAttempt::new(prediction()),
                    StatusCode::OK,
                    streaming,
                );
                let mut received = Vec::new();
                while let Some(chunk) = stream.next().now_or_never().unwrap() {
                    received.extend_from_slice(&chunk.unwrap());
                }
                assert_eq!(received, payload);
                drop(stream);
            });
            assert!(handle.render().lines().any(|line| line
                .starts_with("vllm_router_kv_route_outcomes_total{")
                && line.contains("status=\"observed\"")
                && line.ends_with(" 1")));
        }
    }

    #[test]
    fn transparent_stream_errors_and_cancellation_finalize_once() {
        use futures_util::{FutureExt, StreamExt};
        for streaming in [false, true] {
            for cancelled in [false, true] {
                let recorder =
                    metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
                let handle = recorder.handle();
                metrics::with_local_recorder(&recorder, || {
                    let chunks = vec![
                        Ok(Bytes::from_static(b"partial")),
                        Err(std::io::Error::other("broken body")),
                    ];
                    let mut stream = KvResponseStream::new(
                        futures_util::stream::iter(chunks),
                        KvRouteAttempt::new(prediction()),
                        StatusCode::OK,
                        streaming,
                    );
                    assert_eq!(
                        stream.next().now_or_never().unwrap().unwrap().unwrap(),
                        Bytes::from_static(b"partial")
                    );
                    if !cancelled {
                        assert!(stream.next().now_or_never().unwrap().unwrap().is_err());
                        assert!(stream.next().now_or_never().unwrap().is_none());
                    }
                    drop(stream);
                });
                let exported = handle.render();
                let outcomes: Vec<_> = exported
                    .lines()
                    .filter(|line| line.starts_with("vllm_router_kv_route_outcomes_total{"))
                    .collect();
                assert_eq!(outcomes.len(), 1);
                assert!(outcomes[0].contains("status=\"unavailable\""));
                assert!(outcomes[0].ends_with(" 1"));
                let reason = if streaming {
                    "incomplete_sse_frame"
                } else if cancelled {
                    "attempt_cancelled"
                } else {
                    "response_error"
                };
                assert!(outcomes[0].contains(&format!("reason=\"{reason}\"")));
                assert!(!exported.contains("vllm_router_kv_cached_token_absolute_error"));
            }
        }
    }

    #[test]
    fn multi_prompt_usage_is_unavailable_for_json_and_sse() {
        for streaming in [false, true] {
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            metrics::with_local_recorder(&recorder, || {
                let mut prediction = prediction();
                prediction.multi_prompt = true;
                prediction.cached_tokens = None;
                let attempt = KvRouteAttempt::new(prediction);
                if streaming {
                    let mut observer = KvSseObserver::new(attempt, StatusCode::OK);
                    observer.observe(&Bytes::from_static(b"data: {\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":32}}}\n\ndata: [DONE]\n\n"));
                    observer.finish(true);
                } else {
                    observe_json_response(
                        &attempt,
                        StatusCode::OK,
                        br#"{"usage":{"prompt_tokens_details":{"cached_tokens":32}}}"#,
                    );
                }
            });
            let exported = handle.render();
            assert!(exported.lines().any(|line| line
                .starts_with("vllm_router_kv_route_outcomes_total{")
                && line.contains("reason=\"multi_prompt_usage_not_aggregate\"")
                && line.contains("status=\"unavailable\"")
                && line.ends_with(" 1")));
            assert!(!exported.contains("vllm_router_kv_cached_token_absolute_error"));
        }
    }

    #[test]
    fn multi_prompt_preserves_failure_and_cancellation_reasons() {
        for reason in [
            UnavailableReason::RequestError,
            UnavailableReason::BackendError,
            UnavailableReason::AttemptCancelled,
        ] {
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            metrics::with_local_recorder(&recorder, || {
                let mut prediction = prediction();
                prediction.multi_prompt = true;
                prediction.cached_tokens = None;
                let attempt = KvRouteAttempt::new(prediction);
                if reason != UnavailableReason::AttemptCancelled {
                    observe_unavailable(&attempt, StatusCode::BAD_GATEWAY, reason);
                }
                drop(attempt);
            });
            let reason = reason.as_str();
            let exported = handle.render();
            assert!(exported.lines().any(|line| line
                .starts_with("vllm_router_kv_route_outcomes_total{")
                && line.contains(&format!("reason=\"{reason}\""))
                && line.ends_with(" 1")));
            assert!(!exported.contains("vllm_router_kv_predicted_cached_tokens"));
            assert!(!exported.contains("vllm_router_kv_cached_token_absolute_error"));
        }
    }

    #[test]
    fn observes_fragmented_sse_usage() {
        let attempt = KvRouteAttempt::new(prediction());
        let mut observer = KvSseObserver::new(attempt, StatusCode::OK);
        observer.observe(&Bytes::from_static(
            b"data: {\"choices\":[]}\n\ndata: {\"usage\":{\"prompt_tokens_details\":{\"cached_",
        ));
        observer.observe(&Bytes::from_static(b"tokens\":32}}}\n\ndata: [DONE]\n\n"));
        assert_eq!(observer.usage.observed_cached_tokens, Some(32));
        assert!(!observer.usage.ambiguous);
    }

    #[test]
    fn conflicting_sse_usage_is_ambiguous() {
        let attempt = KvRouteAttempt::new(prediction());
        let mut observer = KvSseObserver::new(attempt, StatusCode::OK);
        observer.observe(&Bytes::from_static(
            b"data: {\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":16}}}\n\n",
        ));
        observer.observe(&Bytes::from_static(
            b"data: {\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":32}}}\n\n",
        ));
        assert!(observer.usage.ambiguous);
    }

    #[test]
    fn incomplete_sse_parsing_does_not_credit_previous_usage() {
        for oversized in [false, true] {
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            metrics::with_local_recorder(&recorder, || {
                let mut observer =
                    KvSseObserver::new(KvRouteAttempt::new(prediction()), StatusCode::OK);
                observer.observe(&Bytes::from_static(
                    b"data: {\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":32}}}\n\n",
                ));
                if oversized {
                    let mut frame = b"data: ".to_vec();
                    frame.resize(MAX_OBSERVATION_BYTES + 1, b'x');
                    frame.extend_from_slice(b"\n\n");
                    observer.observe(&Bytes::from(frame));
                } else {
                    observer.observe(&Bytes::from_static(b"data: {\"usage\":invalid}\n\n"));
                }
                observer.finish(true);
            });
            let exported = handle.render();
            let reason = if oversized {
                "oversized_sse_frame"
            } else {
                "malformed_usage"
            };
            assert!(exported.lines().any(|line| line
                .starts_with("vllm_router_kv_route_outcomes_total{")
                && line.contains("status=\"unavailable\"")
                && line.contains(&format!("reason=\"{reason}\""))
                && line.ends_with(" 1")));
            assert!(!exported.contains("vllm_router_kv_actual_cached_tokens"));
            assert!(!exported.contains("vllm_router_kv_cached_token_absolute_error"));
        }
    }

    #[test]
    fn invalid_sse_usage_does_not_reuse_previous_count() {
        for cached in [
            serde_json::json!(-1),
            serde_json::json!("32"),
            serde_json::json!(65),
            serde_json::json!(1.5),
        ] {
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            metrics::with_local_recorder(&recorder, || {
                let mut observer =
                    KvSseObserver::new(KvRouteAttempt::new(prediction()), StatusCode::OK);
                observer.observe(&Bytes::from_static(b"data: {\"usage\":{\"prompt_tokens\":64,\"prompt_tokens_details\":{\"cached_tokens\":32}}}\n\n"));
                let value = serde_json::json!({"usage":{"prompt_tokens":64,"prompt_tokens_details":{"cached_tokens":cached}}});
                observer.observe(&Bytes::from(format!("data: {value}\n\n")));
                observer.finish(true);
            });
            let exported = handle.render();
            assert!(exported.lines().any(|line| line
                .starts_with("vllm_router_kv_route_outcomes_total{")
                && line.contains("reason=\"invalid_usage\"")
                && line.ends_with(" 1")));
            assert!(!exported.contains("vllm_router_kv_actual_cached_tokens"));
        }
    }

    #[test]
    fn unfinished_sse_tail_invalidates_usage_on_eof_error_and_drop() {
        for ending in ["eof", "error", "drop"] {
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            metrics::with_local_recorder(&recorder, || {
                let mut observer =
                    KvSseObserver::new(KvRouteAttempt::new(prediction()), StatusCode::OK);
                observer.observe(&Bytes::from_static(
                    b"data: {\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":32}}}\n\n",
                ));
                observer.observe(&Bytes::from_static(
                    b"data: {\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":",
                ));
                match ending {
                    "eof" => observer.finish(true),
                    "error" => observer.finish(false),
                    _ => {}
                }
                drop(observer);
            });
            let exported = handle.render();
            assert!(exported.lines().any(|line| line
                .starts_with("vllm_router_kv_route_outcomes_total{")
                && line.contains("reason=\"incomplete_sse_frame\"")
                && line.contains("status=\"unavailable\"")
                && line.ends_with(" 1")));
            assert!(!exported.contains("vllm_router_kv_actual_cached_tokens"));
            assert!(!exported.contains("vllm_router_kv_cached_token_absolute_error"));
        }
    }

    #[test]
    fn shared_attempt_is_finalized_once_across_response_paths() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let attempt = KvRouteAttempt::new(prediction());
            observe_unavailable(
                &attempt,
                StatusCode::BAD_GATEWAY,
                UnavailableReason::RequestError,
            );
            observe_json_response(
                &attempt,
                StatusCode::OK,
                br#"{"usage":{"prompt_tokens_details":{"cached_tokens":32}}}"#,
            );
            assert!(attempt.is_finished());
            drop(attempt);
            let retry = KvRouteAttempt::new(prediction());
            observe_json_response(
                &retry,
                StatusCode::OK,
                br#"{"usage":{"prompt_tokens_details":{"cached_tokens":32}}}"#,
            );
            drop(retry);
        });
        let exported = handle.render();
        let outcomes: Vec<_> = exported
            .lines()
            .filter(|line| line.starts_with("vllm_router_kv_route_outcomes_total{"))
            .collect();
        assert_eq!(outcomes.len(), 2);
        assert!(outcomes.iter().all(|line| line.ends_with(" 1")));
        assert!(outcomes
            .iter()
            .any(|line| line.contains("status=\"observed\"")));
        assert!(outcomes
            .iter()
            .any(|line| line.contains("reason=\"request_error\"")));
    }

    #[test]
    fn dropping_unfinished_attempt_finalizes_cancellation() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            drop(KvRouteAttempt::new(prediction()));
        });
        let exported = handle.render();
        let outcome = exported
            .lines()
            .find(|line| line.starts_with("vllm_router_kv_route_outcomes_total{"))
            .expect("cancelled attempt outcome metric");
        assert!(outcome.contains("reason=\"attempt_cancelled\""));
        assert!(outcome.contains("status=\"unavailable\""));
        assert!(outcome.ends_with(" 1"));
    }
}
