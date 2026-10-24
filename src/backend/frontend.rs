//! Chat frontend for an all-`grpc://` worker pool.
//!
//! `prepare` tokenizes once (outside policy / retry). `dispatch` sends
//! the prepared `token_ids` to a chosen worker (`convert` + GenerateStream
//! + detok). Policy stays outside this type.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::response::Response;

use super::grpc::{FirstOutputCallback, GrpcEngineBackend};
use super::preprocess::{tokenize_chat_request_timed, FrontendHandle, TokenizeOut, TokenizerCache};
use crate::protocols::spec::ChatCompletionRequest;
use crate::tokenizer::{TokenizerCacheConfig, TokenizerCacheStats};

/// Client-facing response shape. The internal pipeline is always chat-shaped;
/// `/v1/completions` needs the legacy `text_completion` payload instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseShape {
    Chat,
    Completion,
}

/// Failure from the gRPC prepare stage, split by who must act on it.
#[derive(Debug, Clone)]
pub enum PrepareError {
    /// Tokenizer / sampling-default failures: model assets are not ready yet.
    Unavailable(String),
    /// The request itself cannot be prepared; the client must change it.
    Invalid(String),
}

impl std::fmt::Display for PrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(message) | Self::Invalid(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for PrepareError {}

/// Tokenized request ready for policy + `dispatch`. Both client routes land
/// here; `shape` records which one, and completions reuses the chat shape.
#[derive(Clone)]
pub struct PreparedRequest {
    pub request: Arc<ChatCompletionRequest>,
    pub tokenized: TokenizeOut,
    pub handle: FrontendHandle,
    pub resolve_ms: f64,
    pub t_req: Instant,
    pub shape: ResponseShape,
    /// Client opted in with `return_token_ids`; ids are withheld otherwise.
    pub emit_token_ids: bool,
}

/// Load-once tokenizer + tonic client. One replica per router process.
#[derive(Clone, Debug)]
pub struct EngineFrontend {
    tokenizer: Arc<TokenizerCache>,
    grpc: GrpcEngineBackend,
}

impl Default for EngineFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineFrontend {
    pub fn new() -> Self {
        Self {
            tokenizer: Arc::new(TokenizerCache::new()),
            grpc: GrpcEngineBackend::new(),
        }
    }

    pub fn with_request_timeout(request_timeout: Duration) -> Self {
        Self {
            tokenizer: Arc::new(TokenizerCache::new()),
            grpc: GrpcEngineBackend::with_timeouts(Duration::from_secs(10), request_timeout),
        }
    }

    /// Enable L0 with a shared budget across models.
    pub fn with_tokenizer_cache(
        request_timeout: Duration,
        config: TokenizerCacheConfig,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            tokenizer: Arc::new(TokenizerCache::with_encoding_cache(config)?),
            ..Self::with_request_timeout(request_timeout)
        })
    }

    pub(crate) fn from_env(request_timeout: Duration) -> anyhow::Result<Self> {
        match super::encode_cache::config_from_env()? {
            Some(config) => Self::with_tokenizer_cache(request_timeout, config),
            None => Ok(Self::with_request_timeout(request_timeout)),
        }
    }

    /// L0 stats across models, or None when disabled.
    pub fn tokenizer_cache_stats(&self) -> Option<TokenizerCacheStats> {
        self.tokenizer.encoding_cache_stats()
    }

    /// Tests only: bypass model loading by returning these fake prompt ids.
    pub fn pin_test_token_ids(&self, token_ids: Vec<u32>) {
        self.tokenizer.pin_test_token_ids(token_ids);
    }

    /// Tokenizer handle + sampling defaults shared by both prepare paths;
    /// returns the request start instant and the resolve-stage milliseconds.
    async fn resolve_with_defaults(
        &self,
        request: &mut ChatCompletionRequest,
    ) -> Result<(FrontendHandle, Instant, f64), PrepareError> {
        let t_req = Instant::now();
        let handle = self
            .tokenizer
            .resolve(request.model.as_deref())
            .await
            .map_err(|e| PrepareError::Unavailable(format!("tokenizer: {e}")))?;
        if request.temperature.is_none() {
            request.temperature = Some(
                handle
                    .default_temperature()
                    .map_err(|e| PrepareError::Unavailable(format!("sampling defaults: {e}")))?
                    .unwrap_or(1.0),
            );
        }
        let resolve_ms = t_req.elapsed().as_secs_f64() * 1000.0;
        Ok((handle, t_req, resolve_ms))
    }

    /// Chat template + encode. Call once, before `policy.select`.
    pub async fn prepare(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Result<PreparedRequest, PrepareError> {
        let (handle, t_req, resolve_ms) = self.resolve_with_defaults(&mut request).await?;
        let mut tokenized = tokenize_chat_request_timed(&request, &handle)
            .map_err(|e| PrepareError::Invalid(format!("preprocess: {e}")))?;
        truncate_grpc_prompt(&request, &mut tokenized)?;
        Ok(PreparedRequest {
            request: Arc::new(request),
            tokenized,
            handle,
            resolve_ms,
            t_req,
            shape: ResponseShape::Chat,
            emit_token_ids: false,
        })
    }

    /// Wire adapter path for `/v1/completions` on the gRPC pool: the client
    /// already sent token ids, so there is no chat template / router tokenize.
    /// Sampling defaults are resolved like `prepare`; encode fields report 0.
    pub async fn prepare_token_ids(
        &self,
        mut request: ChatCompletionRequest,
        token_ids: Vec<u32>,
        emit_token_ids: bool,
    ) -> Result<PreparedRequest, PrepareError> {
        let (handle, t_req, resolve_ms) = self.resolve_with_defaults(&mut request).await?;
        let mut tokenized = TokenizeOut {
            token_ids: Arc::from(token_ids),
            adapt_ms: 0.0,
            template_ms: 0.0,
            encode_ms: 0.0,
            encode_backend: "completions-token-ids",
            cached_tokens: 0,
        };
        truncate_grpc_prompt(&request, &mut tokenized)?;
        Ok(PreparedRequest {
            request: Arc::new(request),
            tokenized,
            handle,
            resolve_ms,
            t_req,
            shape: ResponseShape::Completion,
            emit_token_ids,
        })
    }

    /// Convert + GenerateStream + detok on the selected worker URL.
    pub async fn dispatch(&self, worker_url: &str, prepared: PreparedRequest) -> Response {
        self.dispatch_with_first_output(worker_url, prepared, None)
            .await
    }

    pub(crate) async fn dispatch_with_first_output(
        &self,
        worker_url: &str,
        prepared: PreparedRequest,
        on_first_output: Option<FirstOutputCallback>,
    ) -> Response {
        self.grpc
            .dispatch_prepared(worker_url, prepared, on_first_output)
            .await
    }

    pub fn remove_worker(&self, worker_url: &str) {
        self.grpc.remove_client(worker_url);
    }
}

fn truncate_grpc_prompt(
    request: &ChatCompletionRequest,
    tokenized: &mut TokenizeOut,
) -> Result<(), PrepareError> {
    if let Some(value) = request
        .other
        .get("truncate_prompt_tokens")
        .filter(|value| !value.is_null())
    {
        let limit = value
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| {
                PrepareError::Invalid(
                    "truncate_prompt_tokens must be a non-negative 32-bit integer on gRPC".into(),
                )
            })? as usize;
        // The gRPC protocol's zero means no truncation; positive limits keep
        // the suffix. Sending the already-truncated IDs makes placement and
        // dispatch agree, even when the worker applies the same limit again.
        if limit > 0 && tokenized.token_ids.len() > limit {
            tokenized.token_ids =
                Arc::from(&tokenized.token_ids[tokenized.token_ids.len() - limit..]);
            tokenized.cached_tokens = tokenized.cached_tokens.min(limit);
        }
    }
    Ok(())
}
