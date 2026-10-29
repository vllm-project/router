//! HTTP input preparation and gRPC chat rendering/tokenization.
//! Loaded models share optional L0 (exact) and L1 (prefix) encoding caches.

use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::policies::WorkerTokenInputs;
use http::HeaderMap;
use reqwest::Client;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::debug;

use anyhow::{anyhow, Result};
use dashmap::DashMap;
use parking_lot::RwLock;

use super::encode_cache::FrontendCache;
use super::l1_encode_cache::{L1Cache, L1CacheConfig, L1CacheStats};
use crate::tokenizer::{TokenizerCacheConfig, TokenizerCacheStats};

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

/// HTTP routing uses the worker's own render contract, not the gRPC model
/// frontend: HTTP workers can have different templates and tokenizer settings.
enum HttpKvInput {
    Direct(Arc<[Vec<u32>]>),
    Render { path: &'static str },
}

fn dense_token_ids(value: &serde_json::Value) -> Result<Vec<u32>, &'static str> {
    let ids = value.as_array().ok_or("invalid_token_ids")?;
    if ids.is_empty() {
        return Err("empty_input");
    }
    ids.iter()
        .map(|id| {
            id.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or("invalid_token_ids")
        })
        .collect()
}

fn token_prompts(value: &serde_json::Value) -> Result<Arc<[Vec<u32>]>, &'static str> {
    let items = value.as_array().ok_or("invalid_token_ids")?;
    let prompts = if items.first().is_some_and(serde_json::Value::is_array) {
        items
            .iter()
            .map(dense_token_ids)
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![dense_token_ids(value)?]
    };
    Ok(Arc::from(prompts))
}

fn http_kv_input(body: &serde_json::Value, route: &str) -> Result<HttpKvInput, &'static str> {
    // The event index cannot identify multimodal/embedding or prompt-adapter
    // cache keys from token IDs alone. HTTP adapter names also need an explicit
    // mapping to the names carried by KV events before they can earn credit.
    for field in [
        "features",
        "prompt_embeds",
        "input_embeds",
        "input_embeddings",
        "multi_modal_data",
        "lora_path",
        "lora_name",
        "prompt_adapter_request",
    ] {
        if body.get(field).is_some_and(|value| !value.is_null()) {
            return Err("unsupported_cache_identity");
        }
    }
    if let Some(messages) = body.get("messages").and_then(serde_json::Value::as_array) {
        for message in messages {
            if let Some(parts) = message.get("content").and_then(serde_json::Value::as_array) {
                if parts.iter().any(|part| {
                    part.get("type").and_then(serde_json::Value::as_str) != Some("text")
                }) {
                    return Err("multimodal_input");
                }
            }
        }
    }
    let path = match route {
        "/v1/completions" => {
            if body
                .get("truncate_prompt_tokens")
                .is_none_or(serde_json::Value::is_null)
                && body.get("suffix").is_none_or(serde_json::Value::is_null)
            {
                if let Some(items) = body.get("prompt").and_then(serde_json::Value::as_array) {
                    if items.is_empty() {
                        return Err("empty_input");
                    }
                    if items[0].is_number() || items[0].is_array() {
                        return Ok(HttpKvInput::Direct(token_prompts(&body["prompt"])?));
                    }
                }
            }
            "/v1/completions/render"
        }
        "/v1/chat/completions" => "/v1/chat/completions/render",
        "/v1/responses" => {
            if body
                .get("previous_response_id")
                .is_some_and(|value| !value.is_null())
            {
                return Err("stateful_input");
            }
            "/v1/responses/render"
        }
        "/generate" => {
            // Text/prompt take precedence over input_ids on the native route.
            if ["text", "prompt"]
                .iter()
                .any(|field| body.get(field).is_some_and(|value| !value.is_null()))
            {
                return Err("unsupported_input");
            }
            return Ok(HttpKvInput::Direct(token_prompts(&body["input_ids"])?));
        }
        "/inference/v1/generate" => {
            return Ok(HttpKvInput::Direct(Arc::from(vec![dense_token_ids(
                &body["token_ids"],
            )?])))
        }
        _ => return Err("unsupported_route"),
    };
    Ok(HttpKvInput::Render { path })
}

/// Router-tokenized prompts for every worker. The same cache-identity checks
/// as the HTTP render path apply, so an input the event index cannot key
/// earns no credit on either transport.
pub(crate) fn prepared_token_inputs(
    urls: Vec<String>,
    body: &serde_json::Value,
    route: &str,
    token_ids: &[u32],
) -> std::collections::HashMap<String, Result<HttpTokenInput, &'static str>> {
    let result = http_kv_input(body, route).map(|_| HttpTokenInput {
        prompts: Arc::from(vec![token_ids.to_vec()]),
        source: "prepared",
    });
    urls.into_iter()
        .map(|url| {
            let result = match &result {
                Ok(input) => Ok(HttpTokenInput {
                    prompts: input.prompts.clone(),
                    source: input.source,
                }),
                Err(reason) => Err(*reason),
            };
            (url, result)
        })
        .collect()
}

/// Backend-effective prompts, shared by colocated and PD routing policies.
pub(crate) struct HttpTokenInput {
    pub prompts: Arc<[Vec<u32>]>,
    pub source: &'static str,
}

pub(crate) async fn prepare_http_token_inputs(
    client: &Client,
    mut urls: Vec<String>,
    body: &serde_json::Value,
    route: &str,
    headers: Option<&HeaderMap>,
    api_key: Option<&str>,
    budget: Duration,
) -> std::collections::HashMap<String, Result<HttpTokenInput, &'static str>> {
    use futures_util::{stream, StreamExt};
    urls.sort();
    urls.dedup();
    let input = match http_kv_input(body, route) {
        Ok(input) => input,
        Err(reason) => return urls.into_iter().map(|url| (url, Err(reason))).collect(),
    };
    if let HttpKvInput::Direct(prompts) = input {
        return urls
            .into_iter()
            .map(|url| {
                (
                    url,
                    Ok(HttpTokenInput {
                        prompts: prompts.clone(),
                        source: "token_ids",
                    }),
                )
            })
            .collect();
    }
    let HttpKvInput::Render { path } = input else {
        unreachable!()
    };
    let expected_prompts = if route == "/v1/completions" {
        body.get("prompt")
            .and_then(serde_json::Value::as_array)
            .filter(|items| {
                items
                    .first()
                    .is_some_and(|item| item.is_string() || item.is_array())
            })
            .map_or(1, Vec::len)
    } else {
        1
    };
    let tokenize_fallback = std::sync::OnceLock::new();
    let mut inputs: std::collections::HashMap<_, _> = urls
        .iter()
        .map(|url| (url.clone(), Err("preparation_unavailable")))
        .collect();
    let results = stream::iter(urls)
        .map(|url| {
            let tokenize_fallback = &tokenize_fallback;
            async move {
                let result = async {
                    let (prompts, source) =
                        match http_input_json(client, &url, path, body, headers, api_key).await {
                            Ok(rendered) => (http_rendered_tokens(&rendered)?, "render"),
                            Err("render_unavailable") => {
                                let requests = tokenize_fallback
                                    .get_or_init(|| http_tokenize_fallback(body, route))
                                    .as_ref()
                                    .ok_or("render_unavailable")?;
                                let mut prompts = Vec::with_capacity(requests.len());
                                for request in requests {
                                    let tokenized = http_input_json(
                                        client,
                                        &url,
                                        "/tokenize",
                                        request,
                                        headers,
                                        api_key,
                                    )
                                    .await?;
                                    let ids = dense_token_ids(&tokenized["tokens"])?;
                                    if tokenized["count"].as_u64() != Some(ids.len() as u64)
                                        || tokenized["max_model_len"]
                                            .as_u64()
                                            .is_some_and(|limit| ids.len() as u64 > limit)
                                    {
                                        return Err("invalid_preparation_response");
                                    }
                                    prompts.push(ids);
                                }
                                (Arc::from(prompts), "tokenize")
                            }
                            Err(reason) => return Err(reason),
                        };
                    if prompts.len() != expected_prompts {
                        return Err("invalid_preparation_response");
                    }
                    Ok(HttpTokenInput { prompts, source })
                }
                .await;
                (url, result)
            }
        })
        .buffer_unordered(8);
    futures_util::pin_mut!(results);
    let deadline = tokio::time::Instant::now() + budget;
    while let Ok(Some((url, result))) = tokio::time::timeout_at(deadline, results.next()).await {
        inputs.insert(url, result);
    }
    inputs
}

async fn http_input_json(
    client: &reqwest::Client,
    base: &str,
    path: &str,
    body: &serde_json::Value,
    headers: Option<&http::HeaderMap>,
    api_key: Option<&str>,
) -> Result<serde_json::Value, &'static str> {
    let mut request = client
        .post(format!("{}{path}", base.trim_end_matches('/')))
        .timeout(std::time::Duration::from_secs(5))
        .json(body);
    if let Some(key) = api_key {
        request = request.bearer_auth(key);
    } else if let Some(value) = headers.and_then(|headers| headers.get("authorization")) {
        request = request.header("authorization", value);
    }
    let response = request
        .send()
        .await
        .map_err(|_| "preparation_unavailable")?;
    match response.status() {
        reqwest::StatusCode::NOT_FOUND
        | reqwest::StatusCode::METHOD_NOT_ALLOWED
        | reqwest::StatusCode::NOT_IMPLEMENTED => return Err("render_unavailable"),
        status if !status.is_success() => return Err("preparation_rejected"),
        _ => {}
    }
    response
        .json()
        .await
        .map_err(|_| "invalid_preparation_response")
}

pub(crate) fn http_tokenize_fallback(
    body: &serde_json::Value,
    route: &str,
) -> Option<Vec<serde_json::Value>> {
    // /tokenize has fewer fields than generation in vLLM 0.30. Complex chat
    // must use /render; silently dropping tool/template options is not exact.
    for field in [
        "truncate_prompt_tokens",
        "suffix",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "response_format",
        "reasoning_effort",
        "documents",
    ] {
        if body.get(field).is_some_and(|value| !value.is_null()) {
            return None;
        }
    }
    let mut payload = serde_json::Map::new();
    for field in [
        "model",
        "add_special_tokens",
        "chat_template",
        "chat_template_kwargs",
        "add_generation_prompt",
        "continue_final_message",
    ] {
        if let Some(value) = body.get(field).filter(|value| !value.is_null()) {
            payload.insert(field.into(), value.clone());
        }
    }
    match route {
        "/v1/completions" => {
            let prompts = match body.get("prompt")? {
                serde_json::Value::String(text) => vec![text.as_str()],
                serde_json::Value::Array(items) => items
                    .iter()
                    .map(serde_json::Value::as_str)
                    .collect::<Option<Vec<_>>>()?,
                _ => return None,
            };
            Some(
                prompts
                    .into_iter()
                    .map(|prompt| {
                        let mut payload = payload.clone();
                        payload.insert("prompt".into(), prompt.into());
                        serde_json::Value::Object(payload)
                    })
                    .collect(),
            )
        }
        "/v1/chat/completions" => {
            let messages = body.get("messages")?.as_array()?;
            if messages.iter().any(|message| {
                ["tool_calls", "function_call"]
                    .iter()
                    .any(|field| message.get(field).is_some_and(|value| !value.is_null()))
            }) {
                return None;
            }
            payload.insert("messages".into(), body["messages"].clone());
            Some(vec![serde_json::Value::Object(payload)])
        }
        _ => None,
    }
}

fn http_rendered_tokens(body: &serde_json::Value) -> Result<Arc<[Vec<u32>]>, &'static str> {
    let items = match body {
        serde_json::Value::Array(items) => items.iter().collect::<Vec<_>>(),
        item => vec![item],
    };
    if items.is_empty() {
        return Err("empty_input");
    }
    let prompts = items
        .into_iter()
        .map(|item| {
            if item.get("features").is_some_and(|value| !value.is_null()) {
                return Err("multimodal_input");
            }
            dense_token_ids(&item["token_ids"])
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Arc::from(prompts))
}

pub(crate) async fn prepare_http_kv_inputs(
    client: &Client,
    workers: &[Arc<dyn Worker>],
    headers: Option<&HeaderMap>,
    body: &serde_json::Value,
    route: &str,
    api_key: Option<&str>,
    budget: Duration,
) -> WorkerTokenInputs {
    let urls = workers
        .iter()
        .map(|worker| worker.base_url().to_owned())
        .collect();
    let results =
        prepare_http_token_inputs(client, urls, body, route, headers, api_key, budget).await;
    let mut inputs = WorkerTokenInputs::new();
    for (url, result) in results {
        match result {
            Ok(input) => {
                RouterMetrics::record_kv_routing_input("http", input.source);
                inputs.insert(url, input.prompts);
            }
            Err(reason) => {
                RouterMetrics::record_kv_routing_input("http", reason);
                debug!(worker = %url, reason, "Prefix input preparation unavailable");
            }
        }
    }
    inputs
}

#[derive(Clone)]
enum Frontend {
    Vllm(Arc<VllmFrontend>),
    /// Integration-test pin: fake ids, still not this repo’s tokenizer.
    TestIds(Vec<u32>),
}

/// Loads each model once and optionally shares an L0 budget across models.
#[derive(Clone)]
pub struct TokenizerCache {
    pinned: Arc<RwLock<Option<Frontend>>>,
    by_model: Arc<DashMap<String, Arc<VllmFrontend>>>,
    load_lock: Arc<tokio::sync::Mutex<()>>,
    encoding_cache: Option<Arc<FrontendCache>>,
    l1_cache: Option<Arc<L1Cache>>,
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
            encoding_cache: None,
            l1_cache: None,
        }
    }

    /// Enable L0 with a shared budget across models.
    pub fn with_encoding_cache(config: TokenizerCacheConfig) -> Result<Self> {
        Self::with_caches(Some(config), None)
    }

    /// Enable L0 and/or L1, each with one budget shared across models.
    pub fn with_caches(
        l0: Option<TokenizerCacheConfig>,
        l1: Option<L1CacheConfig>,
    ) -> Result<Self> {
        Ok(Self {
            encoding_cache: l0.map(FrontendCache::new).transpose()?.map(Arc::new),
            l1_cache: l1.map(L1Cache::new).transpose()?.map(Arc::new),
            ..Self::new()
        })
    }

    /// L0 stats across models, or None when disabled.
    pub fn encoding_cache_stats(&self) -> Option<TokenizerCacheStats> {
        self.encoding_cache.as_ref().map(|cache| cache.stats())
    }

    /// L1 stats across models, or None when disabled.
    pub fn l1_cache_stats(&self) -> Option<L1CacheStats> {
        self.l1_cache.as_ref().map(|cache| cache.stats())
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
            VllmFrontend::load(
                &source,
                Default::default(),
                self.encoding_cache.clone(),
                self.l1_cache.clone(),
            )
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
    pub token_ids: Arc<[u32]>,
    pub adapt_ms: f64,
    pub template_ms: f64,
    pub encode_ms: f64,
    pub encode_backend: &'static str,
    /// Prompt tokens served by L0 or L1 instead of the tokenizer.
    pub cached_tokens: usize,
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
                token_ids: Arc::from(ids.clone()),
                adapt_ms: 0.0,
                template_ms: 0.0,
                encode_ms: 0.0,
                encode_backend: "test-pin",
                cached_tokens: 0,
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
    let (token_ids, cached_tokens, template_ms, encode_ms) = frontend
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
        cached_tokens,
    })
}

fn spec_messages_to_upstream(
    messages: &[SpecChatMessage],
) -> Result<Vec<super::vllm_frontend::UpstreamChatMessage>> {
    messages
        .iter()
        .map(|message| {
            Ok(match message {
                SpecChatMessage::System { content, .. } => system_text(user_content_text(content)?),
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
                            text: user_content_text(content)?,
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
                SpecChatMessage::Other(value) => {
                    // HTTP backends can forward unknown roles verbatim, but vllm-chat
                    // cannot represent them, so the gRPC frontend rejects them.
                    let role = value
                        .get("role")
                        .and_then(serde_json::Value::as_str)
                        .expect("ChatMessage::Other must contain a string `role`");
                    return Err(anyhow!(
                        "chat message role `{role}` is not supported by the router gRPC frontend"
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
                match part.get("type").and_then(serde_json::Value::as_str) {
                    Some("text") => {
                        let part_text = part
                            .get("text")
                            .and_then(serde_json::Value::as_str)
                            .ok_or_else(|| {
                                anyhow!("text content part is missing a string `text` field")
                            })?;
                        text.push_str(part_text);
                    }
                    // The token_ids-only gRPC path cannot carry non-text parts.
                    _ => {
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

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    #[tokio::test]
    async fn shared_http_preparation_retains_good_workers_and_exact_tool_payloads() {
        use super::prepare_http_token_inputs;
        use axum::{routing::post, Json, Router};
        use serde_json::json;
        let app = Router::new()
            .route(
                "/good/v1/chat/completions/render",
                post(|Json(body): Json<serde_json::Value>| async move {
                    assert_eq!(body["tools"], json!([]));
                    assert_eq!(body["messages"][0]["tool_calls"][0]["id"], "call-1");
                    assert_eq!(body["truncate_prompt_tokens"], 3);
                    Json(json!({"token_ids":[7,8,9]}))
                }),
            )
            .route(
                "/bad/v1/chat/completions/render",
                post(|| async { axum::http::StatusCode::INTERNAL_SERVER_ERROR }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let good = format!("{base}/good");
        let bad = format!("{base}/bad");
        let body = json!({"messages":[{"role":"assistant","tool_calls":[{"id":"call-1"}]}],
            "tools":[],"truncate_prompt_tokens":3});
        let results = prepare_http_token_inputs(
            &reqwest::Client::new(),
            vec![good.clone(), bad.clone()],
            &body,
            "/v1/chat/completions",
            None,
            None,
            Duration::from_secs(1),
        )
        .await;
        assert_eq!(
            results[&good].as_ref().ok().unwrap().prompts.as_ref(),
            &[vec![7, 8, 9]]
        );
        assert!(results[&bad].is_err());
        server.abort();
    }

    #[test]
    fn http_inputs_keep_prompt_boundaries_and_require_exact_render_for_options() {
        use super::{http_kv_input, http_rendered_tokens, HttpKvInput};
        use serde_json::json;
        let HttpKvInput::Direct(prompts) =
            http_kv_input(&json!({"prompt":[[1,2],[3,4]]}), "/v1/completions").unwrap()
        else {
            panic!("direct ids")
        };
        assert_eq!(prompts.as_ref(), &[vec![1, 2], vec![3, 4]]);
        assert!(http_kv_input(&json!({"prompt":[-1,2]}), "/v1/completions").is_err());
        for ids in [json!([1, 2]), json!([[1, 2], [3, 4]])] {
            let HttpKvInput::Direct(prompts) =
                http_kv_input(&json!({"input_ids":ids}), "/generate").unwrap()
            else {
                panic!("native token ids")
            };
            assert_eq!(prompts[0], vec![1, 2]);
        }
        for body in [
            json!({"input_ids":[]}),
            json!({"input_ids":[-1]}),
            json!({"input_ids":[1], "text":"preferred"}),
            json!({"input_ids":[1], "prompt":"preferred"}),
        ] {
            assert!(http_kv_input(&body, "/generate").is_err());
        }

        for body in [
            json!({"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":"auto"}),
            json!({"messages":[{"role":"assistant","tool_calls":[]}],"response_format":{"type":"json_object"}}),
            json!({"prompt":[1,2,3],"truncate_prompt_tokens":2}),
        ] {
            let route = if body.get("prompt").is_some() {
                "/v1/completions"
            } else {
                "/v1/chat/completions"
            };
            let HttpKvInput::Render { .. } = http_kv_input(&body, route).unwrap() else {
                panic!("render options")
            };
            assert!(super::http_tokenize_fallback(&body, route).is_none());
        }
        assert!(http_kv_input(&json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"image"}}]}]}),"/v1/chat/completions").is_err());
        assert!(http_rendered_tokens(
            &json!({"token_ids":[1,2],"features":{"mm_hashes":{"image":["x"]}}})
        )
        .is_err());
    }
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
        let stdout = String::from_utf8_lossy(&out.stdout);
        let ids = serde_json::from_slice(&out.stdout).or_else(|first_error| {
            stdout
                .lines()
                .rev()
                .map(str::trim)
                .find(|line| line.starts_with('['))
                .ok_or(first_error)
                .and_then(serde_json::from_str)
        });
        ids.map(Some)
            .map_err(|e| format!("python stdout not a json id list: {e}; stdout={}", stdout))
    }

    async fn rust_vllm_chat_ids(model: &str, messages: serde_json::Value) -> Vec<u32> {
        let frontend = VllmFrontend::load(model, Default::default(), None, None)
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
            .to_vec()
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
    fn rejects_unknown_roles_on_grpc_path() {
        for role in ["latest_reminder", "developer"] {
            let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "messages": [{
                    "role": role,
                    "content": "Model-specific instructions."
                }]
            }))
            .unwrap();

            let error = spec_messages_to_upstream(&request.messages).unwrap_err();
            assert_eq!(
                error.to_string(),
                format!("chat message role `{role}` is not supported by the router gRPC frontend")
            );
        }
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

    #[test]
    fn test_user_content_text_rejects_non_text_parts() {
        let parts = UserMessageContent::Parts(vec![
            serde_json::json!({
                "type": "audio_url",
                "audio_url": {"url": "data:audio/wav;base64,AAAA"}
            }),
            serde_json::json!({"type": "text", "text": "hello"}),
        ]);
        let err = user_content_text(&parts).unwrap_err();
        assert!(
            err.to_string().contains("multimodal chat is not supported"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_user_content_text_concatenates_text_parts() {
        let plain = UserMessageContent::Text("plain".to_string());
        assert_eq!(user_content_text(&plain).unwrap(), "plain");

        let parts = UserMessageContent::Parts(vec![
            serde_json::json!({"type": "text", "text": "foo"}),
            serde_json::json!({"type": "text", "text": "bar"}),
        ]);
        assert_eq!(user_content_text(&parts).unwrap(), "foobar");
    }

    #[test]
    fn test_user_content_text_rejects_malformed_text_part() {
        // `type == "text"` but no string `text` field.
        let parts = UserMessageContent::Parts(vec![serde_json::json!({"type": "text"})]);
        let err = user_content_text(&parts).unwrap_err();
        assert!(
            err.to_string().contains("missing a string `text` field"),
            "unexpected error: {err}"
        );
    }
}
