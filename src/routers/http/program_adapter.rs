//! HTTP response adaptation for Program lifecycle and token observations.
//!
//! This module understands transport envelopes only. Identity parsing and all
//! scheduling decisions remain in `program_scheduling`.

use crate::backend::grpc::EngineOutcomeObserver;
use crate::program_scheduling::{ProgramDispatch, ProgramScheduler, ProgramUsageObservation};
use crate::token_estimator::{MomentumTokenEstimator, TokenEstimateCalibration};
use parking_lot::Mutex;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, OnceLock,
};
use std::time::Instant;

#[derive(Clone)]
pub(super) struct ProgramCompletion {
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
        self.inner.finish(success);
    }

    /// Terminal Engine outcome channel for the gRPC producer.
    pub(super) fn engine_observer(&self) -> Arc<EngineOutcomeObserver> {
        self.inner.observer.clone()
    }

    #[cfg(test)]
    pub(super) fn finished_success(&self) -> bool {
        self.inner.finished_success.load(Ordering::Acquire)
    }
}

impl ProgramCompletionInner {
    fn finish(&self, mut success: bool) {
        if self.finished.swap(true, Ordering::AcqRel) {
            return;
        }
        let (mut usage, wire_error) = {
            let mut accumulator = self.usage.lock();
            accumulator.finish_sse();
            (accumulator.observation, accumulator.wire_error)
        };
        // The gRPC producer knows the Engine's final counts even when the client
        // did not ask for a wire `usage` chunk; prefer them over missing SSE.
        if let Some((prompt_tokens, completion_tokens)) = self.observer.usage() {
            usage.merge(ProgramUsageObservation {
                // `None` means the Engine did not report `prompt_info`; the
                // wire usage (if any) and the arrival fact remain the only
                // evidence, and reconciliation must not invent a match.
                prompt_tokens,
                completion_tokens: Some(completion_tokens),
                cached_prompt_tokens: None,
            });
        }
        if self.observer.failed() {
            // A producer error frame (timeout, tonic error, premature EOF) is
            // not a successful completion even though the SSE body ended.
            success = false;
        }
        if wire_error {
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
        self.finish(false);
    }
}

#[derive(Debug, Default)]
struct ProgramUsageAccumulator {
    observation: ProgramUsageObservation,
    sse_buffer: Vec<u8>,
    /// A wire-level error payload ended the stream; the body-level EOF must not
    /// be treated as success.
    wire_error: bool,
}

impl ProgramUsageAccumulator {
    const MAX_PARTIAL_SSE_BYTES: usize = 1024 * 1024;

    fn observe_json(&mut self, body: &[u8]) {
        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) {
            self.observe_error(&value);
            self.observe_value(&value);
        }
    }

    /// An `error` payload is the gRPC producer's terminal failure frame and may
    /// also come from an upstream error body; neither is a completed attempt.
    fn observe_error(&mut self, value: &serde_json::Value) {
        if value.get("error").is_some_and(|error| !error.is_null()) {
            self.wire_error = true;
        }
        if value.get("object").and_then(serde_json::Value::as_str) == Some("error") {
            self.wire_error = true;
        }
    }

    fn observe_sse_chunk(&mut self, chunk: &[u8]) {
        self.sse_buffer.extend_from_slice(chunk);
        while let Some(newline) = self.sse_buffer.iter().position(|byte| *byte == b'\n') {
            let mut line = self.sse_buffer.drain(..=newline).collect::<Vec<_>>();
            while line
                .last()
                .is_some_and(|byte| matches!(byte, b'\n' | b'\r'))
            {
                line.pop();
            }
            self.observe_sse_line(&line);
        }
        if self.sse_buffer.len() > Self::MAX_PARTIAL_SSE_BYTES {
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
        let Some(data) = line.trim().strip_prefix("data:") else {
            return;
        };
        let data = data.trim();
        let data = data.as_bytes();
        if data
            .windows(b"\"error\"".len())
            .any(|window| window == b"\"error\"")
        {
            if let Ok(value) = serde_json::from_slice::<serde_json::Value>(data) {
                self.observe_error(&value);
            }
        }
        if data
            .windows(b"\"usage\"".len())
            .any(|window| window == b"\"usage\"")
        {
            self.observe_json(data);
        }
    }

    fn observe_value(&mut self, value: &serde_json::Value) {
        for usage in [
            value.get("usage"),
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
mod tests {
    use super::*;
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
        let (scheduler, dispatch) = settlement_fixture().await;
        let estimator = Arc::new(MomentumTokenEstimator::default());
        let (_, calibration) = estimator.estimate(
            TokenEstimateScope::new("model", "/v1/chat/completions"),
            "observer usage",
        );
        let completion =
            ProgramCompletion::new(scheduler.clone(), dispatch, estimator, calibration);
        // The client did not ask for a wire usage chunk; the producer still
        // knows the Engine's counts.
        completion.engine_observer().record_usage(Some(123), 7);
        completion.observe_sse_chunk(b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n");
        completion.finish(true);
        assert!(completion.finished_success());
        let snapshot = scheduler.diagnostics();
        assert_eq!(snapshot.ranks[0].programs[0].context_tokens, 123);
    }

    #[tokio::test]
    async fn missing_engine_prompt_info_does_not_fabricate_a_prompt_count() {
        let (scheduler, dispatch) = settlement_fixture().await;
        let estimator = Arc::new(MomentumTokenEstimator::default());
        let (_, calibration) = estimator.estimate(
            TokenEstimateScope::new("model", "/v1/chat/completions"),
            "no prompt info",
        );
        let completion =
            ProgramCompletion::new(scheduler.clone(), dispatch, estimator, calibration);
        // The producer saw completion tokens but the Engine never reported
        // `prompt_info`; only wire usage may supply prompt tokens.
        completion.engine_observer().record_usage(None, 7);
        completion.observe_sse_chunk(
            b"data: {\"usage\":{\"prompt_tokens\":155,\"completion_tokens\":7}}\n",
        );
        completion.finish(true);
        assert!(completion.finished_success());
        let snapshot = scheduler.diagnostics();
        assert_eq!(snapshot.ranks[0].programs[0].context_tokens, 155);
    }

    #[tokio::test]
    async fn wire_error_frame_does_not_count_as_a_successful_round() {
        let (scheduler, dispatch) = settlement_fixture().await;
        let estimator = Arc::new(MomentumTokenEstimator::default());
        let (_, calibration) = estimator.estimate(
            TokenEstimateScope::new("model", "/v1/chat/completions"),
            "wire error",
        );
        let completion =
            ProgramCompletion::new(scheduler.clone(), dispatch, estimator, calibration);
        completion
            .observe_sse_chunk(b"data: {\"error\":\"gRPC stream closed before finish_info\"}\n\n");
        completion.finish(true);
        assert!(!completion.finished_success());
    }

    #[tokio::test]
    async fn producer_error_outcome_overrides_body_eof_success() {
        let (scheduler, dispatch) = settlement_fixture().await;
        let estimator = Arc::new(MomentumTokenEstimator::default());
        let (_, calibration) = estimator.estimate(
            TokenEstimateScope::new("model", "/v1/chat/completions"),
            "producer error",
        );
        let completion =
            ProgramCompletion::new(scheduler.clone(), dispatch, estimator, calibration);
        completion.engine_observer().record_error();
        completion.finish(true);
        assert!(!completion.finished_success());
    }

    #[tokio::test]
    async fn error_free_stream_still_succeeds() {
        let (scheduler, dispatch) = settlement_fixture().await;
        let estimator = Arc::new(MomentumTokenEstimator::default());
        let (_, calibration) = estimator.estimate(
            TokenEstimateScope::new("model", "/v1/chat/completions"),
            "clean stream",
        );
        let completion = ProgramCompletion::new(scheduler, dispatch, estimator, calibration);
        completion.observe_sse_chunk(b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n");
        completion.finish(true);
        assert!(completion.finished_success());
    }

    async fn settlement_fixture() -> (
        Arc<ProgramScheduler>,
        crate::program_scheduling::ProgramDispatch,
    ) {
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
        (scheduler, dispatch)
    }
}
