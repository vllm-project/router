//! Adapter onto `vllm-chat` + `vllm-tokenizer` + `vllm-text` `Prompt`.
//! Not a second renderer. `load_model_backends` is what
//! `TokenizerCache` runs once per model key.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Result};
use serde_json::Value;
use vllm_chat::{
    load_model_backends, ChatContent, ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatTool,
    ChatToolChoice, GenerationPromptMode, LoadModelBackendsOptions, LoadedModelBackends,
    ReasoningEffort, ResolvedToolContext,
};
use vllm_text::Prompt;
use vllm_tokenizer::{IncrementalDecoder, Tokenizer};

pub use vllm_chat::ChatMessage as UpstreamChatMessage;
pub use vllm_tokenizer::DynTokenizer;
pub use vllm_tokenizer::IncrementalDecoder as IncrementalDecoderTrait;

#[derive(Clone)]
pub struct VllmFrontend {
    backends: Arc<LoadedModelBackends>,
}

impl VllmFrontend {
    pub async fn load(
        model_id: &str,
        default_chat_template_kwargs: HashMap<String, Value>,
    ) -> vllm_chat::Result<Self> {
        let backends = load_model_backends(
            model_id,
            LoadModelBackendsOptions {
                language_model_only: true,
                default_chat_template_kwargs,
                ..Default::default()
            },
        )
        .await?;
        Ok(Self {
            backends: Arc::new(backends),
        })
    }

    pub fn tokenizer(&self) -> DynTokenizer {
        self.backends.text_backend.tokenizer()
    }

    pub fn default_temperature(&self) -> Result<Option<f32>> {
        Ok(self
            .backends
            .text_backend
            .sampling_hints()?
            .default_temperature)
    }

    pub fn chat_tokenize_timed(
        &self,
        request: &ChatRequest,
    ) -> vllm_chat::Result<(Vec<u32>, f64, f64)> {
        let t0 = Instant::now();
        let rendered = self.backends.chat_backend.chat_renderer().render(request)?;
        let template_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t1 = Instant::now();
        let ids = match rendered.prompt {
            Prompt::Text(text) => self
                .tokenizer()
                .encode(&text, request.add_special_tokens)
                .map_err(|error| vllm_chat::Error::ChatTemplate(error.to_string()))?,
            Prompt::TokenIds(ids) => ids,
        };
        let encode_ms = t1.elapsed().as_secs_f64() * 1000.0;
        Ok((ids, template_ms, encode_ms))
    }
}

pub struct OpenAiChatRequest {
    pub request_id: String,
    pub messages: Vec<ChatMessage>,
    pub add_generation_prompt: Option<bool>,
    pub continue_final_message: bool,
    pub template_kwargs: HashMap<String, Value>,
    pub tools: Vec<ChatTool>,
    pub tool_choice: Option<ChatToolChoice>,
    pub parallel_tool_calls: bool,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub response_format: Option<Value>,
    pub add_special_tokens: bool,
    pub documents: Option<Vec<Value>>,
    pub priority: i32,
    pub cache_salt: Option<String>,
    pub data_parallel_rank: Option<u32>,
    pub session_id: Option<String>,
}

pub fn chat_request_from_openai(input: OpenAiChatRequest) -> Result<ChatRequest> {
    // Keep this lowering aligned with vLLM 0.29
    // server/routes/openai/chat_completions/convert.rs::prepare_chat_request.
    // That function is pub(super) and route-owned, so importing it would
    // require vllm-server. Replace this adapter if upstream exposes a public,
    // frontend-only OpenAI -> ChatRequest lowering API.
    let generation_prompt_mode = match (
        input.add_generation_prompt,
        input.continue_final_message,
        input.messages.last().map(ChatMessage::role),
    ) {
        (Some(true), true, _) => {
            bail!("cannot set both continue_final_message and add_generation_prompt")
        }
        (_, true, Some(ChatRole::Assistant)) => GenerationPromptMode::ContinueFinalAssistant,
        (_, true, _) => bail!("continue_final_message requires a final assistant message"),
        (Some(false), false, _) => GenerationPromptMode::NoGenerationPrompt,
        (None | Some(true), false, _) => GenerationPromptMode::StartNewAssistant,
    };
    let tool_context = ResolvedToolContext::new(
        &input.messages,
        input.tools,
        input.tool_choice,
        input.parallel_tool_calls,
    )?;

    let mut request = ChatRequest::for_test();
    request.request_id = input.request_id;
    request.messages = input.messages;
    request.add_special_tokens = input.add_special_tokens;
    request.chat_options = ChatOptions {
        generation_prompt_mode,
        reasoning_effort: input.reasoning_effort,
        response_format: input.response_format,
        template_kwargs: input.template_kwargs,
        ..ChatOptions::default()
    };
    request.tool_context = tool_context;
    request.documents = input.documents;
    request.priority = input.priority;
    request.cache_salt = input.cache_salt;
    request.data_parallel_rank = input.data_parallel_rank;
    request.session_id = input.session_id;
    request.validate()?;
    Ok(request)
}

pub fn user_text(text: impl Into<String>) -> ChatMessage {
    ChatMessage::user(ChatContent::Text(text.into()))
}

pub fn system_text(text: impl Into<String>) -> ChatMessage {
    ChatMessage::system(ChatContent::Text(text.into()))
}

pub fn tool_text(content: impl Into<String>, tool_call_id: impl Into<String>) -> ChatMessage {
    ChatMessage::tool_response(ChatContent::Text(content.into()), tool_call_id)
}

pub fn decode_stream<'a>(
    tokenizer: &'a dyn Tokenizer,
    prompt_token_ids: &[u32],
    skip_special_tokens: bool,
) -> Box<dyn IncrementalDecoder + 'a> {
    tokenizer.create_decode_stream(prompt_token_ids, skip_special_tokens, 0)
}

pub fn emit_detok(
    decoder: &mut dyn IncrementalDecoder,
    token_ids: &[u32],
) -> vllm_tokenizer::Result<String> {
    let mut out = String::new();
    for &id in token_ids {
        decoder.push_token(id)?;
        if let Some(chunk) = decoder.next_chunk() {
            out.push_str(&chunk);
        }
    }
    Ok(out)
}
