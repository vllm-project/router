//! Proto `GenerateResponse` chunks → OpenAI JSON / SSE for the **client**.
//!
//! Not a southbound hop. The worker already spoke gRPC; this only
//! reshapes `finish_reason`, usage, and `data: [DONE]`.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::backend::pb;
use crate::protocols::spec::{
    ChatChoice, ChatCompletionResponse, ChatCompletionStreamResponse, ChatMessage,
    ChatMessageDelta, ChatStreamChoice, CompletionChoice, CompletionResponse,
    CompletionStreamChoice, CompletionStreamResponse, Usage, UserMessageContent,
};

/// Client-facing response shape. The internal pipeline is always chat-shaped;
/// `/v1/completions` needs the legacy `text_completion` payload instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseShape {
    Chat,
    Completion,
}

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn finish_reason_name(reason: i32) -> Option<String> {
    match pb::finish_info::FinishReason::try_from(reason) {
        Ok(pb::finish_info::FinishReason::Length) => Some("length".to_string()),
        Ok(pb::finish_info::FinishReason::Stop) => Some("stop".to_string()),
        Ok(pb::finish_info::FinishReason::Aborted) => Some("abort".to_string()),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn stream_chunk(
    id: &str,
    model: &str,
    created: u64,
    text: &str,
    include_role: bool,
    finish_reason: Option<String>,
    usage: Option<Usage>,
    token_ids: Option<Vec<u32>>,
) -> ChatCompletionStreamResponse {
    ChatCompletionStreamResponse {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        system_fingerprint: None,
        choices: vec![ChatStreamChoice {
            index: 0,
            delta: ChatMessageDelta {
                role: include_role.then(|| "assistant".to_string()),
                content: (!text.is_empty()).then(|| text.to_string()),
                tool_calls: None,
                function_call: None,
                reasoning: None,
            },
            logprobs: None,
            finish_reason,
            token_ids,
        }],
        usage,
    }
}

pub fn stream_usage_chunk(
    id: &str,
    model: &str,
    created: u64,
    usage: Usage,
) -> ChatCompletionStreamResponse {
    ChatCompletionStreamResponse {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        system_fingerprint: None,
        choices: Vec::new(),
        usage: Some(usage),
    }
}

pub fn format_sse<T: serde::Serialize>(chunk: &T) -> String {
    format!(
        "data: {}\n\n",
        serde_json::to_string(chunk).unwrap_or_default()
    )
}

/// One SSE generation frame in the shape the client asked for.
#[allow(clippy::too_many_arguments)]
pub fn sse_chunk(
    shape: ResponseShape,
    id: &str,
    model: &str,
    created: u64,
    text: &str,
    include_role: bool,
    finish_reason: Option<String>,
    token_ids: Option<Vec<u32>>,
) -> String {
    match shape {
        ResponseShape::Chat => format_sse(&stream_chunk(
            id,
            model,
            created,
            text,
            include_role,
            finish_reason,
            None,
            token_ids,
        )),
        ResponseShape::Completion => format_sse(&completion_stream_chunk(
            id,
            model,
            created,
            text,
            finish_reason,
            token_ids,
        )),
    }
}

/// Final usage frame (`stream_options.include_usage`), shape-matched.
pub fn sse_usage_chunk(
    shape: ResponseShape,
    id: &str,
    model: &str,
    created: u64,
    usage: Usage,
) -> String {
    match shape {
        ResponseShape::Chat => format_sse(&stream_usage_chunk(id, model, created, usage)),
        ResponseShape::Completion => {
            format_sse(&completion_stream_usage_chunk(id, model, created, usage))
        }
    }
}

pub fn completion_stream_chunk(
    id: &str,
    model: &str,
    created: u64,
    text: &str,
    finish_reason: Option<String>,
    token_ids: Option<Vec<u32>>,
) -> CompletionStreamResponse {
    CompletionStreamResponse {
        id: id.to_string(),
        object: "text_completion".to_string(),
        created,
        model: model.to_string(),
        choices: vec![CompletionStreamChoice {
            text: text.to_string(),
            index: 0,
            logprobs: None,
            finish_reason,
            token_ids,
        }],
        usage: None,
        system_fingerprint: None,
    }
}

pub fn completion_stream_usage_chunk(
    id: &str,
    model: &str,
    created: u64,
    usage: Usage,
) -> CompletionStreamResponse {
    CompletionStreamResponse {
        id: id.to_string(),
        object: "text_completion".to_string(),
        created,
        model: model.to_string(),
        choices: Vec::new(),
        usage: Some(usage),
        system_fingerprint: None,
    }
}

pub fn completion_final_response(
    id: &str,
    model: &str,
    created: u64,
    text: String,
    finish_reason: Option<String>,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> CompletionResponse {
    CompletionResponse {
        id: id.to_string(),
        object: "text_completion".to_string(),
        created,
        model: model.to_string(),
        choices: vec![CompletionChoice {
            text,
            index: 0,
            logprobs: None,
            finish_reason,
            matched_stop: None,
            hidden_states: None,
        }],
        usage: Some(Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            completion_tokens_details: None,
        }),
        system_fingerprint: None,
    }
}

pub const SSE_DONE: &str = "data: [DONE]\n\n";

pub fn final_response(
    id: &str,
    model: &str,
    created: u64,
    text: String,
    finish_reason: Option<String>,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: id.to_string(),
        object: "chat.completion".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage::Assistant {
                role: "assistant".to_string(),
                content: Some(UserMessageContent::Text(text)),
                name: None,
                tool_calls: None,
                function_call: None,
                reasoning: None,
            },
            logprobs: None,
            finish_reason,
            matched_stop: None,
            hidden_states: None,
        }],
        usage: Some(Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            completion_tokens_details: None,
        }),
        system_fingerprint: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_chunk_is_openai_shaped() {
        let chunk = stream_chunk("chatcmpl-1", "m", 1, "hi", true, None, None, None);
        let line = format_sse(&chunk);
        assert!(line.starts_with("data: {"));
        assert!(line.contains("chat.completion.chunk"));
        assert!(line.contains("assistant"));
        assert!(line.contains("hi"));
    }

    #[test]
    fn completion_sse_chunk_carries_text_not_delta() {
        let line = sse_chunk(
            ResponseShape::Completion,
            "cmpl-1",
            "m",
            1,
            "hi",
            true,
            None,
            Some(vec![7]),
        );
        assert!(line.contains("\"object\":\"text_completion\""));
        assert!(line.contains("\"choices\":[{\"text\":\"hi\""));
        assert!(line.contains("\"token_ids\":[7]"));
        assert!(!line.contains("\"delta\""));
    }

    #[test]
    fn completion_usage_frame_is_shape_matched() {
        let line = sse_usage_chunk(
            ResponseShape::Completion,
            "cmpl-1",
            "m",
            1,
            Usage {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5,
                completion_tokens_details: None,
            },
        );
        assert!(line.contains("\"object\":\"text_completion\""));
        assert!(line.contains("\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2"));
    }

    #[test]
    fn completion_final_response_carries_text_not_message() {
        let body = completion_final_response(
            "cmpl-1",
            "m",
            1,
            "hi".to_string(),
            Some("stop".to_string()),
            3,
            2,
        );
        let json = serde_json::to_string(&body).unwrap();
        assert!(json.contains("\"object\":\"text_completion\""));
        assert!(json.contains("\"choices\":[{\"text\":\"hi\""));
        assert!(!json.contains("\"message\""));
    }
}
