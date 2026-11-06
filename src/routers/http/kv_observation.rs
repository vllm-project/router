use super::{sse::sse_lines, sse::FrameScanner, usage_merge};
use crate::kv_index::RoutableTarget;
use crate::metrics::RouterMetrics;
use bytes::Bytes;
use http::StatusCode;
use serde_json::Value;

const MAX_SSE_FRAME_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone)]
pub(super) struct KvRoutePrediction {
    pub(super) target: RoutableTarget,
    pub(super) cached_tokens: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheObservation {
    Observed(u64),
    Ambiguous,
    Unavailable(&'static str),
}

fn response_cached_tokens(value: &Value) -> Option<u64> {
    let usage = usage_merge::usage_object(value)?;
    [
        ("prompt_tokens_details", "prompt_tokens"),
        ("input_tokens_details", "input_tokens"),
    ]
    .iter()
    .find_map(|(details_key, input_key)| {
        let cached = usage.get(*details_key)?.get("cached_tokens")?.as_u64()?;
        if let Some(input) = usage.get(*input_key) {
            if cached > input.as_u64()? {
                return None;
            }
        }
        Some(cached)
    })
}

fn record_observation(prediction: &KvRoutePrediction, observation: CacheObservation) {
    let (status, reason, actual_cached_tokens) = match observation {
        CacheObservation::Observed(tokens) => ("observed", "none", Some(tokens)),
        CacheObservation::Ambiguous => ("ambiguous", "conflicting_usage", None),
        CacheObservation::Unavailable(reason) => ("unavailable", reason, None),
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
        predicted_cached_tokens = prediction.cached_tokens,
        actual_cached_tokens,
        "KV route prediction outcome"
    );
}

pub(super) fn observe_json_response(
    prediction: &KvRoutePrediction,
    status: StatusCode,
    body: &[u8],
) {
    let observation = if !status.is_success() {
        CacheObservation::Unavailable("backend_error")
    } else {
        match serde_json::from_slice::<Value>(body) {
            Ok(value) => response_cached_tokens(&value)
                .map(CacheObservation::Observed)
                .unwrap_or(CacheObservation::Unavailable("missing_usage")),
            Err(_) => CacheObservation::Unavailable("malformed_json"),
        }
    };
    record_observation(prediction, observation);
}

pub(super) fn observe_unavailable(
    prediction: &KvRoutePrediction,
    status: StatusCode,
    reason: &'static str,
) {
    let reason = if status.is_success() {
        reason
    } else {
        "backend_error"
    };
    record_observation(prediction, CacheObservation::Unavailable(reason));
}

pub(super) struct KvSseObserver {
    prediction: KvRoutePrediction,
    successful_status: bool,
    scanner: FrameScanner,
    frame: Vec<u8>,
    discarding_frame: bool,
    observed_cached_tokens: Option<u64>,
    ambiguous: bool,
    fallback_reason: Option<&'static str>,
    finished: bool,
}

impl KvSseObserver {
    pub(super) fn new(prediction: KvRoutePrediction, status: StatusCode) -> Self {
        Self {
            prediction,
            successful_status: status.is_success(),
            scanner: FrameScanner::default(),
            frame: Vec::new(),
            discarding_frame: false,
            observed_cached_tokens: None,
            ambiguous: false,
            fallback_reason: None,
            finished: false,
        }
    }

    pub(super) fn observe(&mut self, chunk: &Bytes) {
        if !self.successful_status || self.finished {
            return;
        }

        let mut remaining = chunk.as_ref();
        while !remaining.is_empty() {
            let frame_end = self.scanner.frame_end(remaining);
            let consumed = frame_end.unwrap_or(remaining.len());
            if !self.discarding_frame {
                if self.frame.len().saturating_add(consumed) > MAX_SSE_FRAME_BYTES {
                    self.frame.clear();
                    self.discarding_frame = true;
                    self.fallback_reason = Some("oversized_sse_frame");
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
            Ok(value) => {
                let Some(cached_tokens) = response_cached_tokens(&value) else {
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
            Err(_) => self.fallback_reason = Some("malformed_usage"),
        }
    }

    pub(super) fn finish(&mut self, stream_succeeded: bool) {
        let observation = if !self.successful_status {
            CacheObservation::Unavailable("backend_error")
        } else if self.ambiguous {
            CacheObservation::Ambiguous
        } else if let Some(tokens) = self.observed_cached_tokens {
            CacheObservation::Observed(tokens)
        } else if !stream_succeeded {
            CacheObservation::Unavailable("stream_error")
        } else {
            CacheObservation::Unavailable(self.fallback_reason.unwrap_or("missing_usage"))
        };
        self.record_once(observation);
    }

    fn record_once(&mut self, observation: CacheObservation) {
        if std::mem::replace(&mut self.finished, true) {
            return;
        }
        record_observation(&self.prediction, observation);
    }
}

impl Drop for KvSseObserver {
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
            cached_tokens: 32,
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
        assert_eq!(response_cached_tokens(&completions), Some(32));
        assert_eq!(response_cached_tokens(&responses), Some(32));
    }

    #[test]
    fn observes_fragmented_sse_usage() {
        let mut observer = KvSseObserver::new(prediction(), StatusCode::OK);
        observer.observe(&Bytes::from_static(
            b"data: {\"choices\":[]}\n\ndata: {\"usage\":{\"prompt_tokens_details\":{\"cached_",
        ));
        observer.observe(&Bytes::from_static(b"tokens\":32}}}\n\ndata: [DONE]\n\n"));
        assert_eq!(observer.observed_cached_tokens, Some(32));
        assert!(!observer.ambiguous);
    }

    #[test]
    fn conflicting_sse_usage_is_ambiguous() {
        let mut observer = KvSseObserver::new(prediction(), StatusCode::OK);
        observer.observe(&Bytes::from_static(
            b"data: {\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":16}}}\n\n",
        ));
        observer.observe(&Bytes::from_static(
            b"data: {\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":32}}}\n\n",
        ));
        assert!(observer.ambiguous);
    }
}
