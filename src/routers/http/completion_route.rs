//! Request-level generation eligibility and Completion forwarding.

use std::sync::Arc;

use axum::{
    body::Bytes,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
};
use reqwest::{Client, RequestBuilder};

use super::completion_output::CompletionOutputObserver;
use crate::core::Worker;
use crate::protocols::spec::CompletionRequest;

pub(super) struct CompletionRoute {
    pub(super) worker_credit: bool,
    pub(super) raw_body: Option<Bytes>,
    pub(super) echo: bool,
}

impl CompletionRoute {
    pub(super) fn forward_request<T: serde::Serialize>(
        &self,
        client: &Client,
        url: &str,
        headers: Option<&HeaderMap>,
        typed_req: &T,
    ) -> RequestBuilder {
        if let Some(raw) = &self.raw_body {
            client
                .post(url)
                .header(
                    CONTENT_TYPE,
                    headers
                        .and_then(|headers| headers.get(CONTENT_TYPE))
                        .cloned()
                        .unwrap_or_else(|| HeaderValue::from_static("application/json")),
                )
                .body(raw.clone())
        } else {
            client.post(url).json(typed_req)
        }
    }

    pub(super) fn output_observer(&self, status: StatusCode) -> CompletionOutputObserver {
        // Echo cannot distinguish prompt from generated text. Each response gets
        // its own observer; unsuccessful responses remain in prefill as well.
        CompletionOutputObserver::new(self.echo || !status.is_success())
    }
}

pub(super) fn prepare_completion_route(
    request: &CompletionRequest,
    raw_body: Option<Bytes>,
) -> CompletionRoute {
    // Worker preparation models these modifiers. Unknown identity extensions
    // still execute normally but cannot acquire cache credit.
    let worker_credit = supports_completion_credit(request);
    CompletionRoute {
        worker_credit,
        raw_body,
        echo: request.echo,
    }
}

fn supports_completion_credit(request: &CompletionRequest) -> bool {
    request.lora_path.is_none()
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

/// Only the effective, plain generation input can establish a history observation.
pub(super) fn plain_history_input(req: Option<crate::protocols::spec::GrpcRequestRef<'_>>) -> bool {
    use crate::protocols::spec::{ChatMessage, GrpcRequestRef, PromptInput, UserMessageContent};
    fn text(c: &UserMessageContent) -> bool {
        match c {
            UserMessageContent::Text(_) => true,
            UserMessageContent::Parts(parts) => {
                !parts.is_empty()
                    && parts.iter().all(|p| {
                        p.get("type").and_then(|v| v.as_str()) == Some("text")
                            && p.get("text").is_some_and(|v| v.is_string())
                            && p.as_object().is_some_and(|fields| fields.len() == 2)
                    })
            }
        }
    }
    match req {
        Some(GrpcRequestRef::Completion(r)) => {
            supports_completion_credit(r)
                && matches!(r.prompt, PromptInput::String(_) | PromptInput::IntArray(_))
        }
        Some(GrpcRequestRef::Chat(r)) => {
            r.other.is_empty()
                && r.lora_path.is_none()
                && r.tools.is_none()
                && r.tool_choice.is_none()
                && r.functions.is_none()
                && r.function_call.is_none()
                && r.chat_template_kwargs.is_none()
                && r.reasoning_effort.is_none()
                && !r.messages.is_empty()
                && r.messages.iter().all(|m| match m {
                    ChatMessage::User { content, .. } | ChatMessage::System { content, .. } => {
                        text(content)
                    }
                    ChatMessage::Assistant {
                        content: Some(c),
                        tool_calls: None,
                        function_call: None,
                        reasoning: None,
                        ..
                    } => text(c),
                    _ => false,
                })
        }
        None => false,
    }
}

/// The public Responses handler is transparent: inspect its original fields,
/// rather than deserializing away extensions that may change input identity.
pub(super) fn plain_responses_history_input(body: &serde_json::Value) -> bool {
    let Some(fields) = body.as_object() else {
        return false;
    };
    fields.keys().all(|field| {
        matches!(
            field.as_str(),
            "model"
                | "input"
                | "instructions"
                | "store"
                | "stream"
                | "background"
                | "previous_response_id"
                | "tools"
                | "tool_choice"
                | "reasoning"
                | "max_output_tokens"
                | "temperature"
                | "top_p"
        )
    }) && body.get("store").and_then(serde_json::Value::as_bool) == Some(false)
        && body
            .get("background")
            .is_none_or(|v| v.as_bool() == Some(false))
        && body
            .get("previous_response_id")
            .is_none_or(serde_json::Value::is_null)
        && body.get("reasoning").is_none_or(serde_json::Value::is_null)
        && body
            .get("tools")
            .is_none_or(|v| v.as_array().is_some_and(Vec::is_empty))
        && body
            .get("tool_choice")
            .is_none_or(|v| matches!(v.as_str(), Some("auto" | "none")))
        && body.get("input").is_some_and(serde_json::Value::is_string)
        && body
            .get("instructions")
            .is_none_or(|v| v.is_null() || v.is_string())
}

pub(super) fn placement_eligibility(
    completion: &CompletionRoute,
    workers: &[Arc<dyn Worker>],
) -> Option<Vec<bool>> {
    (!completion.worker_credit).then(|| vec![false; workers.len()])
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::plain_responses_history_input;
    use serde_json::json;

    #[test]
    fn responses_history_requires_plain_original_fields() {
        let plain = json!({"model":"model", "input":"hello", "store":false});
        assert!(plain_responses_history_input(&plain));
        for (field, value) in [
            ("unknown_extension", json!(null)),
            ("lora_name", json!("adapter")),
            ("cache_salt", json!("tenant")),
            ("store", json!(true)),
            ("background", json!(true)),
            ("previous_response_id", json!("old-response")),
            ("reasoning", json!({"effort":"low"})),
            ("tools", json!([{"type":"function","name":"tool"}])),
            ("tool_choice", json!("required")),
            ("input", json!([{"role":"user","content":"hello"}])),
        ] {
            let mut body = plain.clone();
            body[field] = value;
            assert!(!plain_responses_history_input(&body), "{field}");
        }
        let mut default_store = plain.clone();
        default_store.as_object_mut().unwrap().remove("store");
        assert!(!plain_responses_history_input(&default_store));
    }
}
