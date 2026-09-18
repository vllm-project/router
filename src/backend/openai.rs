//! Proto `GenerateResponse` chunks → OpenAI JSON / SSE for the **client**.
//!
//! Not a southbound hop. The worker already spoke gRPC; this only
//! reshapes `finish_reason`, usage, and `data: [DONE]`.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::backend::pb;
use crate::protocols::spec::{
    ChatChoice, ChatCompletionResponse, ChatCompletionStreamResponse, ChatMessage,
    ChatMessageDelta, ChatStreamChoice, Usage,
};

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

pub fn stream_chunk(
    id: &str,
    model: &str,
    created: u64,
    text: &str,
    include_role: bool,
    finish_reason: Option<String>,
    usage: Option<Usage>,
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

pub fn format_sse(chunk: &ChatCompletionStreamResponse) -> String {
    format!(
        "data: {}\n\n",
        serde_json::to_string(chunk).unwrap_or_default()
    )
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
                content: Some(text),
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
        let chunk = stream_chunk("chatcmpl-1", "m", 1, "hi", true, None, None);
        let line = format_sse(&chunk);
        assert!(line.starts_with("data: {"));
        assert!(line.contains("chat.completion.chunk"));
        assert!(line.contains("assistant"));
        assert!(line.contains("hi"));
    }
}
