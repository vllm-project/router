//! Chat+tokenize for gRPC workers: **only** `vllm-chat` + `vllm-tokenizer`.
//!
//! There is no HuggingFace/minijinja fallback. If those crates are not
//! linked, `grpc://` is unavailable.
//!
//! `TokenizerCache` caches loaded frontend objects (`load_model_backends`),
//! not prior-request token ids or engine KV.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use dashmap::DashMap;
use parking_lot::RwLock;

use super::vllm_frontend::{
    chat_request_from_openai, system_text, tool_text, user_text, OpenAiChatRequest, VllmFrontend,
};
use crate::protocols::spec::{
    ChatCompletionRequest, ChatMessage as SpecChatMessage, ToolChoice as SpecToolChoice,
    ToolChoiceValue, UserMessageContent,
};
use vllm_chat::{
    AssistantContentBlock, AssistantToolCall, ChatTool, ChatToolChoice,
    ReasoningEffort as VllmReasoningEffort,
};

#[derive(Clone)]
enum Frontend {
    Vllm(Arc<VllmFrontend>),
    /// Integration-test pin: fake ids, still not this repo’s tokenizer.
    TestIds(Vec<u32>),
}

/// In-process cache of loaded `vllm-chat` / `vllm-tokenizer` objects.
///
/// First request for a model key calls `load_model_backends`; later
/// requests clone the `Arc`. This is **not** reuse of prior-request
/// `token_ids` and **not** engine KV / prefix-cache routing.
#[derive(Clone)]
pub struct TokenizerCache {
    pinned: Arc<RwLock<Option<Frontend>>>,
    by_model: Arc<DashMap<String, Arc<VllmFrontend>>>,
    load_lock: Arc<tokio::sync::Mutex<()>>,
}

impl std::fmt::Debug for TokenizerCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenizerCache").finish()
    }
}

impl Default for TokenizerCache {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenizerCache {
    pub fn new() -> Self {
        Self {
            pinned: Arc::new(RwLock::new(None)),
            by_model: Arc::new(DashMap::new()),
            load_lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    /// Tests only: bypass model loading by returning these fake prompt ids.
    pub fn pin_test_token_ids(&self, token_ids: Vec<u32>) {
        *self.pinned.write() = Some(Frontend::TestIds(token_ids));
    }

    pub async fn resolve(&self, model: Option<&str>) -> Result<FrontendHandle> {
        if let Some(pinned) = self.pinned.read().clone() {
            return Ok(FrontendHandle(pinned));
        }
        let (key, source) = model_resolution(model)?;
        if let Some(hit) = self.by_model.get(&key) {
            return Ok(FrontendHandle(Frontend::Vllm(hit.clone())));
        }
        // Loading a tokenizer is expensive. Re-check under a single-flight
        // lock so a cold burst does not load the same model N times.
        let _load_guard = self.load_lock.lock().await;
        if let Some(hit) = self.by_model.get(&key) {
            return Ok(FrontendHandle(Frontend::Vllm(hit.clone())));
        }
        let frontend = Arc::new(
            VllmFrontend::load(&source, Default::default())
                .await
                .map_err(|e| anyhow!("vllm-chat load {source} for key {key}: {e}"))?,
        );
        self.by_model.insert(key, frontend.clone());
        Ok(FrontendHandle(Frontend::Vllm(frontend)))
    }
}

fn model_resolution(request_model: Option<&str>) -> Result<(String, String)> {
    let request_model = request_model
        .map(str::trim)
        .filter(|model| !model.is_empty());
    let tokenizer_override = std::env::var("VLLM_ROUTER_TOKENIZER")
        .ok()
        .filter(|value| !value.is_empty())
        .map(|tok| {
            let path = Path::new(&tok);
            if path.is_dir() {
                return Ok(tok);
            }
            if path.is_file() {
                return path
                    .parent()
                    .filter(|parent| parent.is_dir())
                    .map(|parent| parent.to_string_lossy().into_owned())
                    .ok_or_else(|| anyhow!("VLLM_ROUTER_TOKENIZER has no model directory: {tok}"));
            }
            Err(anyhow!(
                "VLLM_ROUTER_TOKENIZER is not a local model directory or tokenizer file: {tok}"
            ))
        })
        .transpose()?;
    let model_override = std::env::var("VLLM_ROUTER_MODEL")
        .ok()
        .filter(|value| !value.is_empty());
    let source = tokenizer_override
        .or(model_override)
        .or_else(|| request_model.map(str::to_string))
        .ok_or_else(|| anyhow!("grpc worker requires request.model or VLLM_ROUTER_MODEL"))?;
    // Environment variables select where to load frontend assets from; the
    // request model remains the cache identity so aliases do not collapse into
    // one global key in multi-model routing.
    let key = request_model
        .map(str::to_string)
        .unwrap_or_else(|| source.clone());
    Ok((key, source))
}

#[derive(Clone)]
pub struct FrontendHandle(Frontend);

impl FrontendHandle {
    pub fn tokenizer(&self) -> Option<super::vllm_frontend::DynTokenizer> {
        match &self.0 {
            Frontend::Vllm(frontend) => Some(frontend.tokenizer()),
            Frontend::TestIds(_) => None,
        }
    }

    pub fn default_temperature(&self) -> Result<Option<f32>> {
        match &self.0 {
            Frontend::Vllm(frontend) => frontend.default_temperature(),
            Frontend::TestIds(_) => Ok(None),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenizeOut {
    pub token_ids: Vec<u32>,
    pub adapt_ms: f64,
    pub template_ms: f64,
    pub encode_ms: f64,
    pub encode_backend: &'static str,
}

impl TokenizeOut {
    pub fn frontend_ms(&self) -> f64 {
        self.adapt_ms + self.template_ms + self.encode_ms
    }
}

pub fn tokenize_chat_request_timed(
    request: &ChatCompletionRequest,
    frontend: &FrontendHandle,
) -> Result<TokenizeOut> {
    match &frontend.0 {
        Frontend::TestIds(ids) => {
            if ids.is_empty() {
                return Err(anyhow!("pinned test token_ids are empty"));
            }
            Ok(TokenizeOut {
                token_ids: ids.clone(),
                adapt_ms: 0.0,
                template_ms: 0.0,
                encode_ms: 0.0,
                encode_backend: "test-pin",
            })
        }
        Frontend::Vllm(vllm) => tokenize_vllm(request, vllm),
    }
}

fn tokenize_vllm(request: &ChatCompletionRequest, frontend: &VllmFrontend) -> Result<TokenizeOut> {
    if request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
        && !matches!(
            request.tool_choice,
            Some(SpecToolChoice::Value(ToolChoiceValue::None))
        )
    {
        return Err(anyhow!(
            "active tool calling requires vllm-chat output parsing, which the gRPC response adapter does not yet expose; tool history and tool_choice=none are supported"
        ));
    }
    if request.reasoning_effort.is_some() {
        return Err(anyhow!(
            "reasoning_effort requires vllm-chat output parsing, which the gRPC response adapter does not yet expose"
        ));
    }
    let t0 = Instant::now();
    let kwargs = request.chat_template_kwargs.clone().unwrap_or_default();
    if request.other.contains_key("chat_template") {
        return Err(anyhow!(
            "per-request chat_template is not supported by the router gRPC frontend"
        ));
    }
    let documents = request
        .other
        .get("documents")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|error| anyhow!("invalid documents: {error}"))?;
    let priority = request
        .other
        .get("priority")
        .and_then(serde_json::Value::as_i64)
        .map(i32::try_from)
        .transpose()
        .map_err(|_| anyhow!("priority is outside the i32 range"))?
        .unwrap_or(0);
    let data_parallel_rank = request
        .other
        .get("data_parallel_rank")
        .and_then(serde_json::Value::as_u64)
        .map(u32::try_from)
        .transpose()
        .map_err(|_| anyhow!("data_parallel_rank is outside the u32 range"))?;
    let session_id = request
        .session_params
        .as_ref()
        .and_then(|params| params.get("session_id"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let upstream = chat_request_from_openai(OpenAiChatRequest {
        request_id: format!("router-{}", request.model.as_deref().unwrap_or("model")),
        messages: spec_messages_to_upstream(&request.messages)?,
        add_generation_prompt: request.add_generation_prompt,
        continue_final_message: request.continue_final_message,
        template_kwargs: kwargs,
        tools: spec_tools_to_upstream(request)?,
        tool_choice: spec_tool_choice_to_upstream(request.tool_choice.as_ref())?,
        parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
        reasoning_effort: spec_reasoning_effort(request)?,
        response_format: request
            .response_format
            .as_ref()
            .map(serde_json::to_value)
            .transpose()?,
        add_special_tokens: request
            .other
            .get("add_special_tokens")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        documents,
        priority,
        cache_salt: request
            .other
            .get("cache_salt")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string),
        data_parallel_rank,
        session_id,
    })?;
    let adapt_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (token_ids, template_ms, encode_ms) = frontend
        .chat_tokenize_timed(&upstream)
        .map_err(|e| anyhow!("vllm-chat tokenize: {e}"))?;
    if token_ids.is_empty() {
        return Err(anyhow!("vllm-chat produced empty token_ids"));
    }
    Ok(TokenizeOut {
        token_ids,
        adapt_ms,
        template_ms,
        encode_ms,
        encode_backend: "vllm-tokenizer",
    })
}

fn spec_messages_to_upstream(
    messages: &[SpecChatMessage],
) -> Result<Vec<super::vllm_frontend::UpstreamChatMessage>> {
    messages
        .iter()
        .map(|message| {
            Ok(match message {
                SpecChatMessage::System { content, .. } => system_text(content),
                SpecChatMessage::User { content, .. } => user_text(user_content_text(content)?),
                SpecChatMessage::Assistant {
                    content,
                    tool_calls,
                    reasoning,
                    ..
                } => {
                    let mut blocks = Vec::new();
                    if let Some(reasoning) = reasoning.as_ref().filter(|text| !text.is_empty()) {
                        blocks.push(AssistantContentBlock::Reasoning {
                            text: reasoning.clone(),
                        });
                    }
                    if let Some(content) = content {
                        blocks.push(AssistantContentBlock::Text {
                            text: content.clone(),
                        });
                    }
                    if let Some(tool_calls) = tool_calls {
                        for call in tool_calls {
                            if call.tool_type != "function" {
                                return Err(anyhow!(
                                    "only function assistant tool calls are supported"
                                ));
                            }
                            blocks.push(AssistantContentBlock::ToolCall(AssistantToolCall {
                                id: call.id.clone(),
                                name: call.function.name.clone(),
                                arguments: call
                                    .function
                                    .arguments
                                    .clone()
                                    .unwrap_or_else(|| "{}".to_string()),
                            }));
                        }
                    }
                    if blocks.is_empty() {
                        return Err(anyhow!(
                            "assistant message must contain content, reasoning, or tool_calls"
                        ));
                    }
                    super::vllm_frontend::UpstreamChatMessage::assistant_blocks(blocks)
                }
                SpecChatMessage::Tool {
                    content,
                    tool_call_id,
                    ..
                } => {
                    let text = content.as_str().ok_or_else(|| {
                        anyhow!("tool message content must be a string on the gRPC path")
                    })?;
                    tool_text(text, tool_call_id)
                }
                SpecChatMessage::Function { .. } => {
                    return Err(anyhow!(
                        "legacy function messages are not supported by vllm-chat"
                    ));
                }
            })
        })
        .collect()
}

fn spec_tools_to_upstream(request: &ChatCompletionRequest) -> Result<Vec<ChatTool>> {
    request
        .tools
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|tool| {
            if tool.tool_type != "function" {
                return Err(anyhow!("only function tools are supported"));
            }
            Ok(ChatTool {
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                parameters: tool.function.parameters.clone(),
                strict: None,
            })
        })
        .collect()
}

fn spec_tool_choice_to_upstream(choice: Option<&SpecToolChoice>) -> Result<Option<ChatToolChoice>> {
    choice
        .map(|choice| {
            Ok(match choice {
                SpecToolChoice::Value(ToolChoiceValue::Auto) => ChatToolChoice::Auto,
                SpecToolChoice::Value(ToolChoiceValue::Required) => ChatToolChoice::Required,
                SpecToolChoice::Value(ToolChoiceValue::None) => ChatToolChoice::None,
                SpecToolChoice::Function {
                    tool_type,
                    function,
                } => {
                    if tool_type != "function" {
                        return Err(anyhow!("only function tool_choice is supported"));
                    }
                    ChatToolChoice::Function {
                        name: function.name.clone(),
                    }
                }
            })
        })
        .transpose()
}

fn spec_reasoning_effort(request: &ChatCompletionRequest) -> Result<Option<VllmReasoningEffort>> {
    request
        .reasoning_effort
        .as_ref()
        .map(|effort| {
            let value = serde_json::to_value(effort)?;
            serde_json::from_value(value).map_err(Into::into)
        })
        .transpose()
}

fn user_content_text(content: &UserMessageContent) -> Result<String> {
    match content {
        UserMessageContent::Text(text) => Ok(text.clone()),
        UserMessageContent::Parts(parts) => {
            let mut text = String::new();
            for part in parts {
                match part {
                    crate::protocols::spec::ContentPart::Text { text: part } => {
                        text.push_str(part);
                    }
                    crate::protocols::spec::ContentPart::ImageUrl { .. } => {
                        return Err(anyhow!(
                            "multimodal chat is not supported by the token_ids-only gRPC path"
                        ));
                    }
                }
            }
            Ok(text)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    fn model_dir() -> Option<String> {
        let dir = std::env::var("VLLM_ROUTER_MODEL")
            .ok()
            .filter(|s| !s.is_empty())?;
        Path::new(&dir)
            .join("tokenizer.json")
            .is_file()
            .then_some(dir)
    }

    fn python_bin() -> PathBuf {
        if let Ok(py) = std::env::var("PYTHON") {
            if !py.is_empty() {
                return PathBuf::from(py);
            }
        }
        if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
            let cand = PathBuf::from(venv).join("bin/python");
            if cand.is_file() {
                return cand;
            }
        }
        PathBuf::from("python3")
    }

    fn python_vllm_chat_ids(
        model: &str,
        messages: &serde_json::Value,
    ) -> Result<Option<Vec<u32>>, String> {
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/python_vllm_chat_ids.py");
        let out = Command::new(python_bin())
            .arg(&script)
            .arg(model)
            .arg(messages.to_string())
            .output()
            .map_err(|e| format!("spawn python: {e}"))?;
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            if err.contains("ModuleNotFoundError") || err.contains("No module named") {
                return Ok(None);
            }
            return Err(format!(
                "python vllm tokenize failed ({:?}): {err}",
                out.status
            ));
        }
        serde_json::from_slice(&out.stdout).map(Some).map_err(|e| {
            format!(
                "python stdout not a json id list: {e}; stdout={}",
                String::from_utf8_lossy(&out.stdout)
            )
        })
    }

    async fn rust_vllm_chat_ids(model: &str, messages: serde_json::Value) -> Vec<u32> {
        let frontend = VllmFrontend::load(model, Default::default())
            .await
            .expect("vllm-chat load_model_backends");
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": 8,
            "add_generation_prompt": true
        }))
        .unwrap();
        tokenize_vllm(&req, &frontend)
            .expect("vllm-chat tokenize")
            .token_ids
    }

    #[test]
    fn preserves_reasoning_and_assistant_tool_history() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "messages": [{
                "role": "assistant",
                "reasoning": "check the weather",
                "content": "calling tool",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": "{\"city\":\"Paris\"}"
                    }
                }]
            }]
        }))
        .unwrap();

        let messages = spec_messages_to_upstream(&request.messages).unwrap();
        let vllm_chat::ChatMessage::Assistant { content } = &messages[0] else {
            panic!("expected assistant message");
        };
        assert!(matches!(
            &content[0],
            AssistantContentBlock::Reasoning { text } if text == "check the weather"
        ));
        assert!(matches!(
            &content[1],
            AssistantContentBlock::Text { text } if text == "calling tool"
        ));
        assert!(matches!(
            &content[2],
            AssistantContentBlock::ToolCall(call)
                if call.id == "call-1"
                    && call.name == "weather"
                    && call.arguments == "{\"city\":\"Paris\"}"
        ));
    }

    #[test]
    fn preserves_request_tools_and_named_choice() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "lookup weather",
                    "parameters": {"type": "object"}
                }
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "weather"}
            }
        }))
        .unwrap();

        let messages = spec_messages_to_upstream(&request.messages).unwrap();
        let tools = spec_tools_to_upstream(&request).unwrap();
        let choice = spec_tool_choice_to_upstream(request.tool_choice.as_ref()).unwrap();
        let chat = chat_request_from_openai(OpenAiChatRequest {
            request_id: "request-1".into(),
            messages,
            add_generation_prompt: request.add_generation_prompt,
            continue_final_message: request.continue_final_message,
            template_kwargs: Default::default(),
            tools,
            tool_choice: choice,
            parallel_tool_calls: true,
            reasoning_effort: None,
            response_format: None,
            add_special_tokens: false,
            documents: None,
            priority: 0,
            cache_salt: None,
            data_parallel_rank: None,
            session_id: None,
        })
        .unwrap();
        assert_eq!(chat.tools().len(), 1);
        assert_eq!(chat.tools()[0].name, "weather");
        assert!(matches!(
            chat.tool_choice(),
            ChatToolChoice::Function { name } if name == "weather"
        ));
    }

    #[tokio::test]
    async fn vllm_chat_tokenize_matches_python_vllm_user() {
        let Some(model) = model_dir() else {
            eprintln!("skip: set VLLM_ROUTER_MODEL to a model dir with tokenizer.json");
            return;
        };
        let messages = serde_json::json!([{"role": "user", "content": "hello"}]);
        let Some(py_ids) = python_vllm_chat_ids(&model, &messages).expect("python vllm") else {
            eprintln!("skip: python cannot import vllm (set PYTHON or VIRTUAL_ENV)");
            return;
        };
        let rust_ids = rust_vllm_chat_ids(&model, messages).await;
        assert_eq!(
            rust_ids, py_ids,
            "vllm-chat+vllm-tokenizer must match Python vllm.tokenizers.get_tokenizer"
        );
    }

    #[tokio::test]
    async fn vllm_chat_tokenize_matches_python_vllm_system_user() {
        let Some(model) = model_dir() else {
            eprintln!("skip: set VLLM_ROUTER_MODEL to a model dir with tokenizer.json");
            return;
        };
        let messages = serde_json::json!([
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content": "ping"}
        ]);
        let Some(py_ids) = python_vllm_chat_ids(&model, &messages).expect("python vllm") else {
            eprintln!("skip: python cannot import vllm (set PYTHON or VIRTUAL_ENV)");
            return;
        };
        let rust_ids = rust_vllm_chat_ids(&model, messages).await;
        assert_eq!(
            rust_ids, py_ids,
            "vllm-chat+vllm-tokenizer must match Python vllm on system+user"
        );
    }
}
