//! HTTP and gRPC response adaptation for Program lifecycle and token observations.
//!
//! This module understands transport envelopes only. Identity parsing and all
//! scheduling decisions remain in `program_scheduling`.

use crate::backend::EngineOutcomeObserver;
use crate::program_scheduling::{ProgramDispatch, ProgramScheduler, ProgramUsageObservation};
use crate::token_estimator::{MomentumTokenEstimator, TokenEstimateCalibration};
use axum::{
    body::{Body, Bytes},
    response::Response,
};
use futures_util::Stream;
use parking_lot::Mutex;
use serde_json::Value;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, OnceLock,
};
use std::task::{Context, Poll};
use std::time::Instant;

pub(super) const MAX_SSE_LINE_BYTES: usize = 1024 * 1024;

/// Terminal marker required to count an HTTP stream as successful.
#[derive(Clone, Copy)]
pub(super) enum StreamEnd {
    Eof,
    OpenAiDone,
}

impl StreamEnd {
    pub(super) fn for_http_route(route: &str) -> Self {
        match route {
            "/v1/chat/completions" | "/v1/completions" => Self::OpenAiDone,
            _ => Self::Eof,
        }
    }
}

#[derive(Clone)]
pub(super) struct ProgramCompletion {
    pub(super) session_admission: bool,
    inner: Arc<ProgramCompletionInner>,
}

struct ProgramCompletionInner {
    scheduler: Arc<ProgramScheduler>,
    dispatch: ProgramDispatch,
    usage: Mutex<ProgramUsageAccumulator>,
    first_stream_output_at: OnceLock<Instant>,
    token_estimator: Arc<MomentumTokenEstimator>,
    token_estimate_calibration: TokenEstimateCalibration,
    finished: AtomicBool,
    finished_success: AtomicBool,
    observer: Arc<EngineOutcomeObserver>,
    pd_ttl: Mutex<Option<std::time::Duration>>,
    replacement_dispatch: Mutex<Option<ProgramDispatch>>,
}

impl ProgramCompletion {
    pub(super) fn new(
        scheduler: Arc<ProgramScheduler>,
        dispatch: ProgramDispatch,
        token_estimator: Arc<MomentumTokenEstimator>,
        token_estimate_calibration: TokenEstimateCalibration,
    ) -> Self {
        Self {
            session_admission: false,
            inner: Arc::new(ProgramCompletionInner {
                scheduler,
                dispatch,
                usage: Mutex::new(ProgramUsageAccumulator::default()),
                first_stream_output_at: OnceLock::new(),
                token_estimator,
                token_estimate_calibration,
                finished: AtomicBool::new(false),
                finished_success: AtomicBool::new(false),
                observer: Arc::new(EngineOutcomeObserver::default()),
                pd_ttl: Mutex::new(None),
                replacement_dispatch: Mutex::new(None),
            }),
        }
    }

    pub(super) fn dispatch(&self) -> &ProgramDispatch {
        &self.inner.dispatch
    }

    pub(super) fn observe_json(&self, body: &[u8]) {
        self.inner.usage.lock().observe_json(body);
    }

    pub(super) fn observe_sse_chunk(&self, chunk: &[u8]) {
        if !chunk.is_empty() {
            let _ = self.inner.first_stream_output_at.set(Instant::now());
        }
        self.inner.usage.lock().observe_sse_chunk(chunk);
    }

    pub(super) fn restart_pd_attempt(&self) -> Option<ProgramDispatch> {
        let mut replacement = self.inner.replacement_dispatch.lock();
        let current = replacement.as_ref().unwrap_or(&self.inner.dispatch);
        let next = self.inner.scheduler.restart_pd_attempt(current)?;
        *replacement = Some(next.clone());
        Some(next)
    }

    pub(super) fn set_pd_ttl(&self, ttl: std::time::Duration) {
        *self.inner.pd_ttl.lock() = Some(ttl);
    }

    pub(super) fn finish(&self, success: bool) {
        self.inner.finish(success, StreamEnd::Eof);
    }

    pub(super) fn finish_http_stream(&self, transport_success: bool, stream_end: StreamEnd) {
        self.inner.finish(transport_success, stream_end);
    }

    /// Terminal Engine outcome channel for the gRPC producer.
    pub(super) fn engine_observer(&self) -> Arc<EngineOutcomeObserver> {
        self.inner.observer.clone()
    }

    #[cfg(test)]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(super) fn finished_success(&self) -> bool {
        self.inner.finished_success.load(Ordering::Acquire)
    }
}

/// Settle gRPC streams at EOF or cancellation. The buffered gRPC adapter has
/// already recorded the Engine outcome when response headers arrive.
pub(super) fn settle_grpc_response(
    response: Response,
    is_stream: bool,
    completion: ProgramCompletion,
) -> Response {
    if !response.status().is_success() {
        completion.finish(false);
        return response;
    }
    if !is_stream {
        // The gRPC buffered adapter has already read the Engine response and
        // recorded its usage in the completion observer, so the attempt settles
        // at header time exactly as before. The body passes through untouched.
        completion.finish(true);
        return response;
    }
    // Streaming attempts settle on the stream's terminal state: forward every
    // chunk while observing SSE usage/error frames, and finish once at EOF (or
    // failure). The body is never buffered ahead of the client.
    let (parts, body) = response.into_parts();
    let stream = ProgramSettlementBody {
        inner: Box::pin(body.into_data_stream()),
        completion: Some(completion),
    };
    Response::from_parts(parts, Body::from_stream(stream))
}

/// Body adapter that settles a Program attempt when the stream terminates.
struct ProgramSettlementBody {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, axum::Error>> + Send>>,
    completion: Option<ProgramCompletion>,
}

impl Stream for ProgramSettlementBody {
    type Item = Result<Bytes, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.as_mut().get_mut();
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                if let Some(completion) = &this.completion {
                    completion.observe_sse_chunk(&chunk);
                }
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(error))) => {
                if let Some(completion) = this.completion.take() {
                    completion.finish(false);
                }
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(None) => {
                if let Some(completion) = this.completion.take() {
                    completion.finish(true);
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for ProgramSettlementBody {
    fn drop(&mut self) {
        // Client cancellation or an early stream closure still settles once.
        if let Some(completion) = self.completion.take() {
            completion.finish(false);
        }
    }
}

/// Request the final usage frame upstream, preserving original raw options
/// (including unknown fields and explicit nulls) when raw JSON is available.
pub(super) fn prepare_stream_usage_body<T: serde::Serialize>(
    raw_body: Option<&[u8]>,
    request: &T,
) -> Option<Value> {
    let mut value = raw_body
        .and_then(|body| serde_json::from_slice::<Value>(body).ok())
        .or_else(|| serde_json::to_value(request).ok())?;
    let object = value.as_object_mut()?;
    match object.get_mut("stream_options") {
        Some(Value::Object(options)) => {
            options.insert("include_usage".to_string(), Value::Bool(true));
            // Enabling usage must not add statistics to ordinary deltas when
            // the client disabled usage. Only the final usage-only frame can
            // be hidden without changing the client's content frames.
            if let Some(continuous) = options.get_mut("continuous_usage_stats") {
                *continuous = Value::Bool(false);
            }
        }
        _ => {
            object.insert(
                "stream_options".to_string(),
                serde_json::json!({"include_usage": true}),
            );
        }
    }
    Some(value)
}

/// Removes the trailing `usage` SSE data line from a client-visible stream.
///
/// The Router injects `stream_options.include_usage` upstream for exact token
/// facts, but a client that did not request usage must not receive that chunk.
/// Lines are buffered across chunk boundaries; other bytes pass through.
#[derive(Debug, Default)]
pub(super) struct SseUsageStripper {
    pending: Vec<u8>,
}

impl SseUsageStripper {
    pub(super) fn push(&mut self, chunk: &[u8]) -> Result<Option<Bytes>, &'static str> {
        let mut out = Vec::new();
        for part in chunk.split_inclusive(|byte| *byte == b'\n') {
            // Fail the filtered stream before forwarding an oversized line
            // that might contain injected usage.
            if part.len() > MAX_SSE_LINE_BYTES.saturating_sub(self.pending.len()) {
                self.pending.clear();
                return Err("upstream SSE line exceeds the 1 MiB usage-filter limit");
            }
            self.pending.extend_from_slice(part);
            if part.ends_with(b"\n") {
                if !is_usage_sse_line(&self.pending) {
                    out.extend_from_slice(&self.pending);
                }
                self.pending.clear();
            }
        }
        Ok((!out.is_empty()).then(|| Bytes::from(out)))
    }

    pub(super) fn finish(&mut self) -> Option<Bytes> {
        let line = std::mem::take(&mut self.pending);
        (!line.is_empty() && !is_usage_sse_line(&line)).then(|| Bytes::from(line))
    }
}

fn sse_data(line: &[u8]) -> Option<&[u8]> {
    Some(line.strip_prefix(b"data:")?.trim_ascii())
}

fn is_error_frame(value: &Value) -> bool {
    value.get("error").is_some_and(|error| !error.is_null())
        || value.get("object").and_then(Value::as_str) == Some("error")
}

fn top_level_usage(value: &Value) -> Option<&Value> {
    value.get("usage").filter(|usage| !usage.is_null())
}

/// Whether one SSE line is the final usage-only frame.
///
/// Only the trailing frame emitted for `stream_options.include_usage` may be
/// removed. A backend that also attaches `usage` (possibly `null`) to ordinary
/// deltas, or emits usage alongside non-empty `choices`, must keep those
/// frames: dropping them would lose content, tool calls and finish reasons.
/// Error frames must also survive, even when they include usage.
fn is_usage_sse_line(line: &[u8]) -> bool {
    let Some(data) = sse_data(line) else {
        return false;
    };
    let Ok(value) = serde_json::from_slice::<Value>(data) else {
        return false;
    };
    !is_error_frame(&value)
        && top_level_usage(&value).is_some()
        && value
            .get("choices")
            .and_then(Value::as_array)
            .is_none_or(Vec::is_empty)
}

impl ProgramCompletionInner {
    fn finish(&self, mut success: bool, stream_end: StreamEnd) {
        if self.finished.swap(true, Ordering::AcqRel) {
            return;
        }
        let (mut usage, wire_error, saw_done) = {
            let mut accumulator = self.usage.lock();
            accumulator.finish_sse();
            (
                accumulator.observation,
                accumulator.wire_error,
                accumulator.saw_done,
            )
        };
        // The gRPC producer knows the Engine's final counts even when the client
        // did not ask for a wire `usage` chunk; prefer them over missing SSE.
        if let Some(engine_usage) = self.observer.usage() {
            // gRPC wire usage falls back to the Router's prepared count when
            // Engine prompt_info is absent. The producer's observation is
            // authoritative, including None: never verify a prepared count
            // against its own synthetic wire copy. HTTP has no producer usage
            // observation, so its independently reported wire count is kept.
            usage.prompt_tokens = engine_usage.prompt_tokens;
            usage.completion_tokens = usage
                .completion_tokens
                .max(Some(engine_usage.completion_tokens));
        }
        if self.observer.failed() {
            // A producer error frame (timeout, tonic error, premature EOF) is
            // not a successful completion even though the SSE body ended.
            success = false;
        }
        if wire_error || (matches!(stream_end, StreamEnd::OpenAiDone) && !saw_done) {
            success = false;
        }
        self.finished_success.store(success, Ordering::Release);
        if success {
            self.token_estimator
                .observe_feedback(&self.token_estimate_calibration, usage.prompt_tokens);
        }
        let decode_seconds = self.first_stream_output_at.get().map(|first_output| {
            Instant::now()
                .saturating_duration_since(*first_output)
                .as_secs_f64()
        });
        let replacement = self.replacement_dispatch.lock();
        let dispatch = replacement.as_ref().unwrap_or(&self.dispatch);
        // V0 no-return labeling depends on upstream honoring the existing contract:
        // `expected_resume=false` means no later model request is expected after this round.
        let terminate = !dispatch.expected_resume();
        self.scheduler.complete_with_pd_ttl(
            dispatch,
            success,
            terminate,
            usage,
            decode_seconds,
            *self.pd_ttl.lock(),
        );
    }
}

impl Drop for ProgramCompletionInner {
    fn drop(&mut self) {
        self.finish(false, StreamEnd::Eof);
    }
}

#[derive(Debug, Default)]
struct ProgramUsageAccumulator {
    observation: ProgramUsageObservation,
    sse_buffer: Vec<u8>,
    /// A wire-level error payload ended the stream; the body-level EOF must not
    /// be treated as success.
    wire_error: bool,
    saw_done: bool,
}

impl ProgramUsageAccumulator {
    fn observe_json(&mut self, body: &[u8]) {
        if let Ok(value) = serde_json::from_slice::<Value>(body) {
            self.wire_error |= is_error_frame(&value);
            self.observe_value(&value);
        }
    }

    fn observe_sse_chunk(&mut self, chunk: &[u8]) {
        self.sse_buffer.extend_from_slice(chunk);
        while let Some(newline) = self.sse_buffer.iter().position(|byte| *byte == b'\n') {
            let line = self.sse_buffer.drain(..=newline).collect::<Vec<_>>();
            self.observe_sse_line(&line);
        }
        if self.sse_buffer.len() > MAX_SSE_LINE_BYTES {
            self.sse_buffer.clear();
        }
    }

    fn finish_sse(&mut self) {
        if !self.sse_buffer.is_empty() {
            let line = std::mem::take(&mut self.sse_buffer);
            self.observe_sse_line(&line);
        }
    }

    fn observe_sse_line(&mut self, line: &[u8]) {
        let Ok(line) = std::str::from_utf8(line) else {
            return;
        };
        let Some(data) = sse_data(line.trim().as_bytes()) else {
            return;
        };
        if data == b"[DONE]" {
            self.saw_done = true;
            return;
        }
        // Ordinary deltas without usage or error metadata need no JSON decode.
        if data
            .windows(7)
            .any(|window| matches!(window, b"\"error\"" | b"\"usage\""))
        {
            self.observe_json(data);
        }
    }

    fn observe_value(&mut self, value: &Value) {
        for usage in [
            top_level_usage(value),
            value.pointer("/message/usage"),
            value.pointer("/response/usage"),
        ]
        .into_iter()
        .flatten()
        {
            let cache_read = first_u64(
                usage,
                &[
                    "cache_read_input_tokens",
                    "cached_input_tokens",
                    "cached_tokens",
                ],
            );
            let cache_creation = first_u64(usage, &["cache_creation_input_tokens"]);
            let prompt_tokens = usage
                .get("prompt_tokens")
                .and_then(serde_json::Value::as_u64)
                .or_else(|| {
                    usage
                        .get("input_tokens")
                        .and_then(serde_json::Value::as_u64)
                        .map(|tokens| {
                            tokens
                                .saturating_add(cache_read.unwrap_or(0))
                                .saturating_add(cache_creation.unwrap_or(0))
                        })
                })
                .and_then(|tokens| usize::try_from(tokens).ok());
            let completion_tokens = first_usize(usage, &["completion_tokens", "output_tokens"]);
            let cached_prompt_tokens = usage
                .pointer("/prompt_tokens_details/cached_tokens")
                .and_then(serde_json::Value::as_u64)
                .or_else(|| {
                    usage
                        .pointer("/input_tokens_details/cached_tokens")
                        .and_then(serde_json::Value::as_u64)
                })
                .or(cache_read)
                .and_then(|tokens| usize::try_from(tokens).ok());
            self.observation.merge(ProgramUsageObservation {
                prompt_tokens,
                completion_tokens,
                cached_prompt_tokens,
            });
        }
    }
}

fn first_usize(value: &serde_json::Value, fields: &[&str]) -> Option<usize> {
    first_u64(value, fields).and_then(|tokens| usize::try_from(tokens).ok())
}

fn first_u64(value: &serde_json::Value, fields: &[&str]) -> Option<u64> {
    fields
        .iter()
        .find_map(|field| value.get(*field).and_then(serde_json::Value::as_u64))
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::backend::outcome::EngineUsage;

    #[test]
    fn sse_usage_stripper_only_drops_the_final_usage_only_frame() {
        let mut stripper = SseUsageStripper::default();
        let first = b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n";
        assert_eq!(stripper.push(first).unwrap().as_deref(), Some(&first[..]));
        // Ordinary deltas may carry a null `usage`, or even a usage object,
        // next to non-empty choices; both must be preserved.
        let null_usage =
            b"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}],\"usage\":null}\n\n";
        assert_eq!(
            stripper.push(null_usage).unwrap().as_deref(),
            Some(&null_usage[..])
        );
        let usage_with_choices =
            b"data: {\"choices\":[{\"delta\":{\"content\":\"!\"}}],\"usage\":{\"prompt_tokens\":3}}\n\n";
        assert_eq!(
            stripper.push(usage_with_choices).unwrap().as_deref(),
            Some(&usage_with_choices[..])
        );
        // Failure information is never a usage-only event. Cover both error
        // envelopes, including a final line without a newline.
        for error in [
            &b"data: {\"error\":{\"message\":\"failed\"},\"usage\":{\"prompt_tokens\":3}}\n\n"[..],
            &b"data: {\"object\":\"error\",\"message\":\"failed\",\"choices\":[],\"usage\":{\"prompt_tokens\":3}}\r\n\r\n"[..],
        ] {
            assert_eq!(stripper.push(error).unwrap().as_deref(), Some(error));
            let unterminated = error.trim_ascii_end();
            assert!(stripper.push(unterminated).unwrap().is_none());
            assert_eq!(stripper.finish().as_deref(), Some(unterminated));
        }
        // The final usage-only frame is dropped even when split across chunks.
        let part_a = b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3";
        let part_b = b"}}\n\n";
        assert!(stripper.push(part_a).unwrap().is_none());
        assert_eq!(stripper.push(part_b).unwrap().as_deref(), Some(&b"\n"[..]));
        assert!(stripper.finish().is_none());
    }

    #[test]
    fn sse_usage_stripper_bounds_partial_lines_without_limiting_complete_chunks() {
        let limit = MAX_SSE_LINE_BYTES;
        let mut stripper = SseUsageStripper::default();
        assert!(stripper.push(&vec![b'x'; limit]).unwrap().is_none());
        assert_eq!(stripper.pending.len(), limit);
        assert!(stripper.push(b"x").is_err());
        assert!(stripper.pending.is_empty());
        assert!(stripper.push(&vec![b'x'; limit + 1]).is_err());
        assert!(stripper.pending.is_empty());
        let mut line = vec![b'x'; limit - 1];
        line.push(b'\n');
        // The bound is per line, even when one transport chunk exceeds 1 MiB.
        let chunk = line.repeat(2);
        assert_eq!(stripper.push(&chunk).unwrap().unwrap().as_ref(), chunk);
        assert!(stripper.pending.is_empty());
    }

    #[test]
    fn inject_stream_usage_keeps_other_options() {
        let request = serde_json::json!({
            "model": "m",
            "stream": true,
            "stream_options": {"include_usage": false, "continuous_usage_stats": true, "other_option": 7}
        });
        let body = prepare_stream_usage_body(None, &request).unwrap();
        assert_eq!(
            body["stream_options"]["include_usage"],
            serde_json::Value::Bool(true)
        );
        assert_eq!(body["stream_options"]["continuous_usage_stats"], false);
        assert_eq!(body["stream_options"]["other_option"], 7);
        let request = serde_json::json!({"model": "m", "stream": true});
        let without = prepare_stream_usage_body(None, &request).unwrap();
        assert_eq!(
            without["stream_options"]["include_usage"],
            serde_json::Value::Bool(true)
        );
    }
    use crate::program_scheduling::{ProgramIdentity, ProgramSchedulerConfig, ProgramTarget};
    use crate::token_estimator::TokenEstimateScope;

    #[test]
    fn parses_openai_and_anthropic_usage_without_double_counting() {
        let mut accumulator = ProgramUsageAccumulator::default();
        accumulator.observe_json(
            br#"{"usage":{"input_tokens":10,"cache_read_input_tokens":30,"cache_creation_input_tokens":20,"output_tokens":5}}"#,
        );
        assert_eq!(accumulator.observation.prompt_tokens, Some(60));
        assert_eq!(accumulator.observation.cached_prompt_tokens, Some(30));
        assert_eq!(accumulator.observation.completion_tokens, Some(5));
    }

    #[test]
    fn sse_prefilter_preserves_usage_across_chunk_boundaries() {
        let mut accumulator = ProgramUsageAccumulator::default();
        accumulator
            .observe_sse_chunk(b"data: {\"choices\":[{\"delta\":{\"content\":\"usage-free\"}}]}\n");
        assert_eq!(accumulator.observation, ProgramUsageObservation::default());

        accumulator.observe_sse_chunk(b"data: {\"usage\":{\"prompt_tokens\":12,");
        accumulator.observe_sse_chunk(b"\"completion_tokens\":3}}\n\ndata: [DONE]\n");
        assert_eq!(accumulator.observation.prompt_tokens, Some(12));
        assert_eq!(accumulator.observation.completion_tokens, Some(3));
    }

    #[test]
    fn sse_done_is_a_data_marker_and_can_cross_chunk_boundaries() {
        let mut accumulator = ProgramUsageAccumulator::default();
        accumulator.observe_sse_chunk(b": comment containing data: [DONE]\r\n\r\n");
        assert!(!accumulator.saw_done);
        accumulator.observe_sse_chunk(b"data: [DO");
        assert!(!accumulator.saw_done);
        accumulator.observe_sse_chunk(b"NE]\r\n\r\n");
        assert!(accumulator.saw_done);
    }

    #[tokio::test]
    async fn retry_attempts_complete_exactly_once_without_occupancy_leak() {
        let scheduler = Arc::new(ProgramScheduler::new(ProgramSchedulerConfig::default()));
        let target = ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://worker".into(),
            dp_rank: None,
        };
        let identity = ProgramIdentity::from_request(
            None,
            Some(&serde_json::json!({"vllm_xargs":{"agentic_context":{
                "program_id":"program","task_id":null,"expected_resume":true
            }}})),
            Some("model"),
        )
        .unwrap()
        .unwrap();
        let estimator = Arc::new(MomentumTokenEstimator::default());

        let first_dispatch = scheduler
            .acquire(identity.clone(), 100, std::slice::from_ref(&target), None)
            .await
            .unwrap();
        let (_, first_calibration) = estimator.estimate(
            TokenEstimateScope::new("model", "/v1/chat/completions"),
            "first attempt",
        );
        let first = ProgramCompletion::new(
            scheduler.clone(),
            first_dispatch,
            estimator.clone(),
            first_calibration,
        );
        first.finish(false);
        first.finish(false);
        drop(first);

        let second_dispatch = scheduler
            .acquire(identity, 100, std::slice::from_ref(&target), None)
            .await
            .unwrap();
        let (_, second_calibration) = estimator.estimate(
            TokenEstimateScope::new("model", "/v1/chat/completions"),
            "second attempt",
        );
        let second = ProgramCompletion::new(
            scheduler.clone(),
            second_dispatch,
            estimator,
            second_calibration,
        );
        second.finish(true);
        second.finish(true);
        drop(second);

        let snapshot = scheduler.diagnostics();
        assert_eq!(snapshot.ranks.len(), 1);
        assert_eq!(snapshot.ranks[0].programs.len(), 1);
        assert_eq!(snapshot.ranks[0].programs[0].in_flight_requests, 0);
        assert_eq!(snapshot.ranks[0].programs[0].waiting_requests, 0);
        assert_eq!(snapshot.ranks[0].router_in_flight_requests, 0);
    }

    #[tokio::test]
    async fn expected_resume_false_marks_http_completion_terminal() {
        let scheduler = Arc::new(ProgramScheduler::new(ProgramSchedulerConfig::default()));
        let target = ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://worker".into(),
            dp_rank: None,
        };
        let identity = ProgramIdentity::from_request(
            None,
            Some(&serde_json::json!({"vllm_xargs":{"agentic_context":{
                "program_id":"terminal","task_id":null,"expected_resume":false
            }}})),
            Some("model"),
        )
        .unwrap()
        .unwrap();
        let dispatch = scheduler
            .acquire(identity, 100, &[target], None)
            .await
            .unwrap();
        assert!(!dispatch.expected_resume());

        let estimator = Arc::new(MomentumTokenEstimator::default());
        let (_, calibration) = estimator.estimate(
            TokenEstimateScope::new("model", "/v1/chat/completions"),
            "terminal",
        );
        let completion =
            ProgramCompletion::new(scheduler.clone(), dispatch, estimator, calibration);
        completion.finish(true);

        let state = scheduler.state.lock();
        let (_, snapshot) = state.prediction_profiles.snapshot().unwrap();
        assert!(snapshot
            .return_time_profiles
            .iter()
            .any(|entry| entry.no_return_count >= 1.0));
    }

    #[tokio::test]
    async fn engine_usage_from_the_producer_is_used_without_wire_usage() {
        let (scheduler, completion) = settlement_fixture("observer usage").await;
        // The client did not ask for a wire usage chunk; the producer still
        // knows the Engine's counts.
        completion.engine_observer().record_usage(EngineUsage {
            prompt_tokens: Some(123),
            completion_tokens: 7,
        });
        completion.observe_sse_chunk(b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n");
        completion.finish(true);
        assert!(completion.finished_success());
        let snapshot = scheduler.diagnostics();
        assert_eq!(snapshot.ranks[0].programs[0].context_tokens, 123);
    }

    #[tokio::test]
    async fn http_wire_usage_remains_authoritative_without_a_producer_observation() {
        let (scheduler, completion) = settlement_fixture("HTTP usage").await;
        completion.observe_sse_chunk(
            b"data: {\"usage\":{\"prompt_tokens\":155,\"completion_tokens\":7}}\n",
        );
        completion.finish(true);
        assert!(completion.finished_success());
        assert_eq!(
            scheduler.diagnostics().ranks[0].programs[0].context_tokens,
            155
        );
    }

    #[tokio::test]
    async fn missing_engine_prompt_info_discards_synthetic_wire_prompt_count() {
        let (scheduler, completion) = settlement_fixture("no prompt info").await;
        // The producer saw completion tokens but the Engine never reported
        // `prompt_info`; gRPC wire usage contains a synthetic fallback count.
        completion.engine_observer().record_usage(EngineUsage {
            prompt_tokens: None,
            completion_tokens: 7,
        });
        completion.observe_sse_chunk(
            b"data: {\"usage\":{\"prompt_tokens\":155,\"completion_tokens\":7}}\n",
        );
        completion.finish(true);
        assert!(completion.finished_success());
        let snapshot = scheduler.diagnostics();
        assert_eq!(snapshot.ranks[0].programs[0].context_tokens, 100);
    }

    #[tokio::test]
    async fn wire_error_frame_does_not_count_as_a_successful_round() {
        let (_, completion) = settlement_fixture("wire error").await;
        completion
            .observe_sse_chunk(b"data: {\"error\":\"gRPC stream closed before finish_info\"}\n\n");
        completion.finish(true);
        assert!(!completion.finished_success());
    }

    #[tokio::test]
    async fn producer_error_outcome_overrides_body_eof_success() {
        let (_, completion) = settlement_fixture("producer error").await;
        completion.engine_observer().record_error();
        completion.finish(true);
        assert!(!completion.finished_success());
    }

    #[tokio::test]
    async fn error_free_stream_still_succeeds() {
        let (_, completion) = settlement_fixture("clean stream").await;
        completion.observe_sse_chunk(b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n");
        completion.finish(true);
        assert!(completion.finished_success());
    }

    async fn settlement_fixture(input: &str) -> (Arc<ProgramScheduler>, ProgramCompletion) {
        let scheduler = Arc::new(ProgramScheduler::new(ProgramSchedulerConfig::default()));
        let target = ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://worker".into(),
            dp_rank: None,
        };
        let identity = ProgramIdentity::from_request(
            None,
            Some(&serde_json::json!({"vllm_xargs":{"agentic_context":{
                "program_id":"observer","expected_resume":true
            }}})),
            Some("model"),
        )
        .unwrap()
        .unwrap();
        let dispatch = scheduler
            .acquire(
                identity,
                crate::program_scheduling::ContextTokens::estimated(100),
                std::slice::from_ref(&target),
                None,
            )
            .await
            .unwrap();
        let estimator = Arc::new(MomentumTokenEstimator::default());
        let (_, calibration) = estimator.estimate(
            TokenEstimateScope::new("model", "/v1/chat/completions"),
            input,
        );
        let completion =
            ProgramCompletion::new(scheduler.clone(), dispatch, estimator, calibration);
        (scheduler, completion)
    }
}
