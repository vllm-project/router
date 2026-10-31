//! Anthropic Messages API (`POST /v1/messages`) request types.
//!
//! The router only types the fields it reads for routing (`model`, `stream`,
//! `system`, `messages`, `max_tokens`, `tools`, `metadata`). Every other field
//! is kept in a flattened map, and optional typed fields keep an explicit
//! `null`, so the body forwarded to the worker carries the same json as the
//! client sent. `model`, `max_tokens` and `messages` are required. The router never translates this request to
//! chat completions; engines that speak `/v1/messages` natively get it as is.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::spec::GenerationRequest;

/// Optional typed field that keeps an explicit `null`. Outer `None` means the
/// key was absent, `Some(None)` means the client sent `null`, so the
/// forwarded body round trips exactly.
pub type Nullable<T> = Option<Option<T>>;

fn nullable<'de, D, T>(deserializer: D) -> Result<Nullable<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(Some)
}

fn present<T>(field: &Nullable<T>) -> Option<&T> {
    field.as_ref().and_then(Option::as_ref)
}

/// `POST /v1/messages` request body.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessagesRequest {
    /// Required by the Messages API; a missing field is a 400 from the router.
    pub model: String,

    pub messages: Vec<AnthropicMessage>,

    /// Required by the Messages API; a missing field is a 400 from the router.
    pub max_tokens: u64,

    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub stream: Nullable<bool>,

    /// Top level system prompt, a string or a list of text blocks.
    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub system: Nullable<MessageContent>,

    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub tools: Nullable<Vec<AnthropicTool>>,

    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub metadata: Nullable<MessagesMetadata>,

    /// Fields the router does not read, forwarded unchanged.
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: MessageContent,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

/// Message or system content: a plain string or a list of content blocks.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// A content block. Only `type`, `text` and `name` are read; the rest
/// (image sources, tool inputs, tool results, cache control) is forwarded as is.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub text: Nullable<String>,
    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub name: Nullable<String>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicTool {
    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub name: Nullable<String>,
    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub description: Nullable<String>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessagesMetadata {
    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub user_id: Nullable<String>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl MessageContent {
    fn push_routing_text(&self, out: &mut Vec<String>) {
        match self {
            MessageContent::Text(text) => out.push(text.clone()),
            MessageContent::Blocks(blocks) => {
                for block in blocks {
                    block.push_routing_text(out);
                }
            }
        }
    }
}

impl ContentBlock {
    /// Text blocks contribute their text. Other blocks contribute a short
    /// stable marker so large payloads (images, documents, tool results)
    /// never enter the routing text.
    fn push_routing_text(&self, out: &mut Vec<String>) {
        match self.block_type.as_str() {
            "text" => {
                if let Some(text) = present(&self.text) {
                    out.push(text.clone());
                }
            }
            "tool_use" => out.push(format!(
                "[tool_use:{}]",
                present(&self.name).map(String::as_str).unwrap_or_default()
            )),
            other => out.push(format!("[{other}]")),
        }
    }
}

impl MessagesRequest {
    /// `metadata.user_id`, read from the nested field only.
    pub fn metadata_user_id(&self) -> Option<&str> {
        present(&self.metadata)
            .and_then(|m| present(&m.user_id))
            .map(String::as_str)
            .map(str::trim)
            .filter(|id| !id.is_empty())
    }
}

impl GenerationRequest for MessagesRequest {
    fn is_stream(&self) -> bool {
        present(&self.stream).copied().unwrap_or(false)
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    /// System prompt first, then tool names and descriptions, then message
    /// text in order. The stable parts lead, so turns of one conversation
    /// share a prefix for text keyed policies. Cache aware and kv aware
    /// routing use worker tokens from `/v1/messages/render` (or `/tokenize`)
    /// instead; see `backend::preprocess::http_kv_input`.
    fn extract_text_for_routing(&self) -> String {
        let mut parts = Vec::new();
        if let Some(system) = present(&self.system) {
            system.push_routing_text(&mut parts);
        }
        for tool in present(&self.tools).into_iter().flatten() {
            if let Some(name) = present(&tool.name) {
                parts.push(name.clone());
            }
            if let Some(description) = present(&tool.description) {
                parts.push(description.clone());
            }
        }
        for message in &self.messages {
            message.content.push_routing_text(&mut parts);
        }
        parts.join("\n")
    }

    fn affinity_user_key(&self) -> Option<&str> {
        self.metadata_user_id()
    }

    /// Program identity fields (`vllm_xargs`, `agent_hint`, `session_params`,
    /// `session_id`, `user_id`) live in the forwarded `other` fields, so a
    /// body level Program opt-in keeps working on the typed route.
    fn affinity_identity(&self) -> crate::request_identity::IdentityFields<'_> {
        crate::request_identity::IdentityFields {
            other: Some(&self.other),
            session_params: None,
        }
    }

    fn extract_program_identity_payload(&self) -> Option<Value> {
        crate::protocols::spec::program_identity_payload(Some(&self.other), None)
    }
}

/// Anthropic error `type` for a router generated status code.
pub fn anthropic_error_type(status: u16) -> &'static str {
    match status {
        400 | 422 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 | 529 => "overloaded_error",
        _ => "api_error",
    }
}

/// Anthropic error body: `{"type":"error","error":{"type":...,"message":...}}`.
pub fn anthropic_error_body(status: u16, message: &str) -> Value {
    serde_json::json!({
        "type": "error",
        "error": {
            "type": anthropic_error_type(status),
            "message": message,
        }
    })
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(v: &Value) -> MessagesRequest {
        serde_json::from_value(v.clone()).unwrap()
    }

    /// Parse with the required fields filled in, for tests about other fields.
    fn parse_with_required(v: &Value) -> MessagesRequest {
        let mut v = v.clone();
        let obj = v.as_object_mut().unwrap();
        obj.entry("model").or_insert(json!("m"));
        obj.entry("max_tokens").or_insert(json!(16));
        parse(&v)
    }

    #[test]
    fn round_trip_keeps_unknown_fields() {
        let body = json!({
            "model": "m",
            "max_tokens": 64,
            "temperature": 0.5,
            "top_k": 4,
            "stop_sequences": ["x"],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
            "tools": [{"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {"user_id": {"type": "string"}}}}],
            "tool_choice": {"type": "auto"},
            "metadata": {"user_id": "u1", "extra": 1},
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "tool_use", "id": "a", "name": "t", "input": {"q": 1}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": [{"type": "text", "text": "r"}]},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
                ]}
            ]
        });
        let req = parse(&body);
        assert_eq!(serde_json::to_value(&req).unwrap(), body);
    }

    #[test]
    fn round_trip_minimal_adds_nothing() {
        let body =
            json!({"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]});
        assert_eq!(serde_json::to_value(parse(&body)).unwrap(), body);
    }

    #[test]
    fn stream_and_model() {
        let req = parse_with_required(&json!({"stream": true, "messages": []}));
        assert!(req.is_stream());
        assert_eq!(req.get_model(), Some("m"));
        let req = parse_with_required(&json!({"messages": []}));
        assert!(!req.is_stream());
        let req = parse_with_required(&json!({"stream": null, "messages": []}));
        assert!(!req.is_stream());
    }

    #[test]
    fn routing_text_order_and_markers() {
        let req = parse_with_required(&json!({
            "system": "sys prompt",
            "tools": [{"name": "search", "description": "web search", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
                    {"type": "document", "source": {"type": "text", "data": "long"}}
                ]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "search", "input": {"q": "x"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "secret payload"}]}
            ]
        }));
        assert_eq!(
            req.extract_text_for_routing(),
            "sys prompt\nsearch\nweb search\nhello\n[image]\n[document]\n[tool_use:search]\n[tool_result]"
        );
    }

    #[test]
    fn routing_text_system_blocks() {
        let req = parse_with_required(&json!({
            "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "messages": [{"role": "user", "content": "c"}]
        }));
        assert_eq!(req.extract_text_for_routing(), "a\nb\nc");
    }

    #[test]
    fn routing_prefix_stable_across_turns() {
        let turn1 = parse_with_required(
            &json!({"system": "s", "messages": [{"role": "user", "content": "q1"}]}),
        );
        let turn2 = parse_with_required(&json!({"system": "s", "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}
        ]}));
        assert!(turn2
            .extract_text_for_routing()
            .starts_with(&turn1.extract_text_for_routing()));
    }

    #[test]
    fn metadata_user_id_is_nested_only() {
        let req = parse_with_required(&json!({
            "tools": [{"name": "t", "input_schema": {"properties": {"user_id": "schema"}}}],
            "messages": []
        }));
        assert_eq!(req.affinity_user_key(), None);
        let req = parse_with_required(&json!({"metadata": {"user_id": " u1 "}, "messages": []}));
        assert_eq!(req.affinity_user_key(), Some("u1"));
        let req = parse_with_required(&json!({"metadata": {"user_id": ""}, "messages": []}));
        assert_eq!(req.affinity_user_key(), None);
    }

    #[test]
    fn required_fields_are_enforced() {
        let ok = json!({"model": "m", "max_tokens": 1, "messages": []});
        assert!(serde_json::from_value::<MessagesRequest>(ok.clone()).is_ok());
        for field in ["model", "max_tokens", "messages"] {
            let mut body = ok.clone();
            body.as_object_mut().unwrap().remove(field);
            let err = serde_json::from_value::<MessagesRequest>(body).unwrap_err();
            assert!(err.to_string().contains(field), "{field}: {err}");
        }
    }

    #[test]
    fn round_trip_keeps_explicit_nulls() {
        let body = json!({
            "model": "m",
            "max_tokens": 8,
            "stream": null,
            "system": null,
            "tools": [{"name": "t", "description": null, "input_schema": {"type": "object"}}],
            "metadata": {"user_id": null},
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "name": null}]}]
        });
        let req = parse(&body);
        assert!(!req.is_stream());
        assert_eq!(req.affinity_user_key(), None);
        assert_eq!(serde_json::to_value(&req).unwrap(), body);
        let body =
            json!({"model": "m", "max_tokens": 8, "metadata": null, "tools": null, "messages": []});
        assert_eq!(serde_json::to_value(parse(&body)).unwrap(), body);
    }

    #[test]
    fn error_shape() {
        assert_eq!(
            anthropic_error_body(503, "no workers"),
            json!({"type": "error", "error": {"type": "overloaded_error", "message": "no workers"}})
        );
        assert_eq!(anthropic_error_type(400), "invalid_request_error");
        assert_eq!(anthropic_error_type(401), "authentication_error");
        assert_eq!(anthropic_error_type(404), "not_found_error");
        assert_eq!(anthropic_error_type(500), "api_error");
    }

    #[test]
    fn program_identity_payload_keeps_body_level_opt_in() {
        let req = parse_with_required(&json!({
            "messages": [{"role": "user", "content": "hi"}],
            "vllm_xargs": {"agentic_context": {"program_id": "program-a"}},
            "agent_hint": {"session_id": "session-a"},
            "metadata": {"user_id": "u-1"}
        }));
        let payload = req.extract_program_identity_payload().unwrap();
        assert_eq!(
            payload["vllm_xargs"]["agentic_context"]["program_id"],
            "program-a"
        );
        assert_eq!(payload["agent_hint"]["session_id"], "session-a");
        // Generation content never enters the identity payload.
        assert!(payload.get("messages").is_none());
        assert!(payload.get("metadata").is_none());

        let plain = parse_with_required(&json!({
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert!(plain.extract_program_identity_payload().is_none());
    }
}
