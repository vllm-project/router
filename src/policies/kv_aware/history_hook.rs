//! Request and output milestones for Regular completion prefix history.

use std::{ops::ControlFlow, sync::Arc};

use http::{header::CONTENT_ENCODING, HeaderMap, StatusCode};
use parking_lot::Mutex;
use serde_json::Value;

use super::{CompletionHistoryRequest, HistoryObservation};
use crate::{
    backend::{preprocess::completion_cache_read_fallback, WorkerPoolKind},
    core::{Worker, WorkerRegistry},
    policies::{kv_aware_ops, LoadBalancingPolicy},
    protocols::spec::{CompletionRequest, GenerationRequest, GrpcRequestRef, PromptInput},
};

pub(crate) struct KvAwareHistoryHook;

pub(crate) struct HistoryCapture {
    request: CompletionHistoryRequest,
    echo: bool,
}

pub(crate) struct HistoryAttempt {
    policy: Arc<dyn LoadBalancingPolicy>,
    observation: HistoryObservation,
    registry: Arc<WorkerRegistry>,
    worker_url: String,
    echo: bool,
}

impl KvAwareHistoryHook {
    // Freeze the registrations before the first input-preparation await.
    pub(crate) fn capture(
        enabled: bool,
        pool: Option<WorkerPoolKind>,
        request: &impl GenerationRequest,
        model_id: Option<&str>,
        policy: impl FnOnce() -> Arc<dyn LoadBalancingPolicy>,
        workers: impl FnOnce() -> Vec<Arc<dyn Worker>>,
    ) -> Option<HistoryCapture> {
        if !enabled {
            return None;
        }
        let echo = match (pool, request.grpc_request()) {
            (Some(_), Some(GrpcRequestRef::Completion(request)))
                if completion_history_input(request) =>
            {
                request.echo
            }
            // Chat uses the existing identity/render qualification below
            // rather than a second field allowlist.
            (Some(WorkerPoolKind::Http), Some(GrpcRequestRef::Chat(request))) => {
                request.echo == Some(true)
            }
            _ => return None,
        };
        if request.cache_identity() != (None, None) {
            return None;
        }
        let model = request
            .get_model()
            .filter(|m| !m.is_empty() && model_id.is_none_or(|id| id == *m))?;
        let policy = policy();
        let request =
            kv_aware_ops(policy.as_ref())?.capture_completion_history(&workers(), model)?;
        Some(HistoryCapture { request, echo })
    }
}

fn completion_history_input(request: &CompletionRequest) -> bool {
    matches!(
        request.prompt,
        PromptInput::String(_) | PromptInput::IntArray(_)
    ) && request.lora_path.is_none()
        && completion_cache_read_fallback(request).is_none()
        && request
            .other
            .iter()
            .all(|(field, value)| match field.as_str() {
                "add_special_tokens" => value.is_boolean(),
                "priority" => value.is_i64() || value.is_u64(),
                "truncate_prompt_tokens" => true,
                _ => false,
            })
}

impl HistoryCapture {
    pub(crate) fn allow_cache_read(self, fallback: Option<&str>) -> Option<Self> {
        fallback.is_none().then_some(self)
    }

    pub(crate) fn request(
        &self,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
    ) -> Option<&CompletionHistoryRequest> {
        (lora_name.is_none() && cache_salt.is_none()).then_some(&self.request)
    }

    // Only the actual policy selection can mint the observation for an attempt.
    pub(crate) fn bind(
        &self,
        observation: HistoryObservation,
        policy: Arc<dyn LoadBalancingPolicy>,
        registry: Arc<WorkerRegistry>,
        worker_url: &str,
    ) -> HistoryAttempt {
        HistoryAttempt {
            policy,
            observation,
            registry,
            worker_url: worker_url.to_owned(),
            echo: self.echo,
        }
    }
}

impl HistoryAttempt {
    pub(crate) fn for_http_response(
        self,
        route: &str,
        stream: bool,
        status: StatusCode,
        headers: &HeaderMap,
    ) -> Option<HttpHistoryAttempt> {
        // Echo and response encoding affect observation, not history-assisted selection.
        if self.echo
            || !status.is_success()
            || headers
                .get_all(CONTENT_ENCODING)
                .iter()
                .any(|value| !value.as_bytes().eq_ignore_ascii_case(b"identity"))
        {
            return None;
        }
        let output = match (route, stream) {
            ("/v1/completions", _) => HistoryOutput::Completion,
            ("/v1/chat/completions", true) => HistoryOutput::ChatDelta,
            ("/v1/chat/completions", false) => HistoryOutput::ChatMessage,
            _ => return None,
        };
        Some(HttpHistoryAttempt {
            attempt: self,
            output,
        })
    }

    pub(crate) fn grpc_callback(self) -> impl Fn() + Send + Sync + 'static {
        let pending = Mutex::new(Some(self));
        move || {
            let attempt = pending.lock().take();
            if let Some(attempt) = attempt {
                attempt.commit();
            }
        }
    }

    fn commit(self) {
        // The worker at output time can differ from the one selected earlier.
        if let (Some(kv), Some(worker)) = (
            kv_aware_ops(self.policy.as_ref()),
            self.registry.get_by_url(&self.worker_url),
        ) {
            kv.observe_completion_milestone(worker.as_ref(), self.observation);
        }
    }
}

pub(crate) struct HttpHistoryAttempt {
    attempt: HistoryAttempt,
    output: HistoryOutput,
}

impl HttpHistoryAttempt {
    pub(crate) fn inspect(&self, event_type: Option<&[u8]>, data: &[u8]) -> ControlFlow<bool> {
        self.output.inspect(event_type, data)
    }

    pub(crate) fn commit(self) {
        self.attempt.commit();
    }
}

#[derive(Clone, Copy)]
enum HistoryOutput {
    Completion,
    ChatDelta,
    ChatMessage,
}

impl HistoryOutput {
    fn inspect(self, event_type: Option<&[u8]>, data: &[u8]) -> ControlFlow<bool> {
        if matches!(event_type, Some(b"error" | b"abort")) || data.trim_ascii() == b"[DONE]" {
            return ControlFlow::Break(false);
        }
        let Ok(value) = serde_json::from_slice::<Value>(data) else {
            return ControlFlow::Continue(());
        };
        if value.get("error").is_some_and(|error| !error.is_null())
            || value.get("object").and_then(Value::as_str) == Some("error")
        {
            return ControlFlow::Break(false);
        }
        let Some(choices) = value.get("choices").and_then(Value::as_array) else {
            return ControlFlow::Continue(());
        };
        let mut output = false;
        for choice in choices {
            let reason = choice.get("finish_reason").and_then(Value::as_str);
            if matches!(reason, Some("abort" | "error")) {
                return ControlFlow::Break(false);
            }
            // Reasoning, tools, roles and prompt metadata are not generated text.
            let text = match self {
                Self::Completion => choice.get("text"),
                Self::ChatDelta => choice.get("delta").and_then(|delta| delta.get("content")),
                Self::ChatMessage => choice
                    .get("message")
                    .and_then(|message| message.get("content")),
            };
            output |= text
                .and_then(Value::as_str)
                .is_some_and(|text| !text.is_empty());
        }
        if output {
            ControlFlow::Break(true)
        } else {
            ControlFlow::Continue(())
        }
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn history_requires_route_specific_generated_text() {
        for (case, json, expected) in [
            (
                "final completion text",
                r#"{"error":null,"choices":[{"text":" ","finish_reason":"stop"}]}"#,
                [true, true, false, false],
            ),
            (
                "chat delta",
                r#"{"choices":[{"delta":{"content":"generated"}}]}"#,
                [false, false, true, false],
            ),
            (
                "chat message",
                r#"{"choices":[{"message":{"content":"generated"},"finish_reason":"length"}]}"#,
                [false, false, false, true],
            ),
            (
                "wrong types and empty text",
                r#"{"choices":[{"text":"","delta":{"content":42},"message":{"content":[]}}]}"#,
                [false; 4],
            ),
            (
                "role, reasoning, tools and prompt metadata",
                r#"{"choices":[{"prompt_token_ids":[1],"delta":{"role":"assistant","reasoning":"hidden","tool_calls":[{}]},"message":{"role":"assistant","reasoning_content":"hidden","tool_calls":[{}]}}]}"#,
                [false; 4],
            ),
            (
                "usage only",
                r#"{"choices":[],"usage":{"completion_tokens":1}}"#,
                [false; 4],
            ),
            (
                "incomplete JSON",
                r#"{"choices":[{"text":"partial"}"#,
                [false; 4],
            ),
            (
                "abort takes precedence over text in any choice",
                r#"{"choices":[{"text":"not output","delta":{"content":"not output"},"message":{"content":"not output"}},{"finish_reason":"abort"}]}"#,
                [false; 4],
            ),
            (
                "error object takes precedence over text",
                r#"{"object":"error","choices":[{"text":"not output","delta":{"content":"not output"},"message":{"content":"not output"}}]}"#,
                [false; 4],
            ),
            (
                "non-null error field takes precedence over text",
                r#"{"error":{},"choices":[{"text":"not output","delta":{"content":"not output"},"message":{"content":"not output"}}]}"#,
                [false; 4],
            ),
        ] {
            for ((output, mode), expected) in [
                (HistoryOutput::Completion, "completion stream"),
                (HistoryOutput::Completion, "completion body"),
                (HistoryOutput::ChatDelta, "chat stream"),
                (HistoryOutput::ChatMessage, "chat body"),
            ]
            .into_iter()
            .zip(expected)
            {
                let observed = output.inspect(None, json.as_bytes());
                assert_eq!(
                    matches!(observed, ControlFlow::Break(true)),
                    expected,
                    "{case}: {mode}"
                );
            }
        }
    }

    #[test]
    fn history_predicate_marks_abort_error_and_done_terminal() {
        let output = HistoryOutput::Completion;
        for (event, data) in [
            (None, " \t[DONE] \t"),
            (
                Some(b"error".as_slice()),
                r#"{"choices":[{"text":"not output"}]}"#,
            ),
            (Some(b"abort".as_slice()), ""),
            (None, r#"{"error":{}}"#),
            (None, r#"{"object":"error"}"#),
            (
                None,
                r#"{"choices":[{"text":"not output","finish_reason":"abort"}]}"#,
            ),
            (
                None,
                r#"{"choices":[{"text":"not output","finish_reason":"error"}]}"#,
            ),
        ] {
            assert_eq!(
                output.inspect(event, data.as_bytes()),
                ControlFlow::Break(false),
                "{event:?}: {data}"
            );
        }
        // A normal finish belongs to one choice, not the entire response.
        for reason in ["stop", "length"] {
            let data =
                format!(r#"{{"choices":[{{"index":0,"text":"","finish_reason":"{reason}"}}]}}"#);
            assert_eq!(
                output.inspect(None, data.as_bytes()),
                ControlFlow::Continue(())
            );
        }
        assert_eq!(
            output.inspect(None, br#"{"choices":[{"index":1,"text":"generated"}]}"#),
            ControlFlow::Break(true)
        );
    }
}
