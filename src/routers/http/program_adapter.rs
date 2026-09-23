//! HTTP response adaptation for Program lifecycle and token observations.
//!
//! This module understands transport envelopes only. Identity parsing and all
//! scheduling decisions remain in `program_scheduling`.

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

    pub(super) fn finish(&self, success: bool) {
        self.inner.finish(success);
    }
}

impl ProgramCompletionInner {
    fn finish(&self, success: bool) {
        if self.finished.swap(true, Ordering::AcqRel) {
            return;
        }
        let usage = {
            let mut accumulator = self.usage.lock();
            accumulator.finish_sse();
            accumulator.observation
        };
        if success {
            self.token_estimator
                .observe_feedback(&self.token_estimate_calibration, usage.prompt_tokens);
        }
        let decode_seconds = self.first_stream_output_at.get().map(|first_output| {
            Instant::now()
                .saturating_duration_since(*first_output)
                .as_secs_f64()
        });
        self.scheduler
            .complete(&self.dispatch, success, false, usage, decode_seconds);
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
}

impl ProgramUsageAccumulator {
    const MAX_PARTIAL_SSE_BYTES: usize = 1024 * 1024;

    fn observe_json(&mut self, body: &[u8]) {
        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) {
            self.observe_value(&value);
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

#[cfg(test)]
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
}
