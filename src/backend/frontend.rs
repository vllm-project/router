//! Chat frontend for an all-`grpc://` worker pool.
//!
//! `prepare` tokenizes once (outside policy / retry). `dispatch` sends
//! the prepared `token_ids` to a chosen worker (`convert` + GenerateStream
//! + detok). Policy stays outside this type.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::response::Response;

use super::grpc::{FirstOutputCallback, GrpcEngineBackend, KvHandoff, PrefillProgressCallback};
use super::l1_encode_cache::{L1CacheConfig, L1CacheStats};
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

    /// Opt in to worker progress and retain last-known work when updates become stale.
    pub(crate) fn with_prefill_progress_stale_after(
        mut self,
        stale_after: Option<Duration>,
    ) -> Self {
        self.grpc.prefill_progress_stale_after = stale_after;
        self
    }

    /// Enable L0 with a shared budget across models.
    pub fn with_tokenizer_cache(
        request_timeout: Duration,
        config: TokenizerCacheConfig,
    ) -> anyhow::Result<Self> {
        Self::with_encoding_caches(request_timeout, Some(config), None)
    }

    /// Enable L0 (exact) and/or L1 (prefix) encoding caches.
    pub fn with_encoding_caches(
        request_timeout: Duration,
        l0: Option<TokenizerCacheConfig>,
        l1: Option<L1CacheConfig>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            tokenizer: Arc::new(TokenizerCache::with_caches(l0, l1)?),
            ..Self::with_request_timeout(request_timeout)
        })
    }

    pub(crate) fn from_env(request_timeout: Duration) -> anyhow::Result<Self> {
        Self::with_encoding_caches(
            request_timeout,
            super::encode_cache::config_from_env()?,
            super::l1_encode_cache::config_from_env()?,
        )
    }

    /// L0 stats across models, or None when disabled.
    pub fn tokenizer_cache_stats(&self) -> Option<TokenizerCacheStats> {
        self.tokenizer.encoding_cache_stats()
    }

    /// L1 stats across models, or None when disabled.
    pub fn l1_cache_stats(&self) -> Option<L1CacheStats> {
        self.tokenizer.l1_cache_stats()
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
        self.dispatch_with_first_output(worker_url, prepared, None, None)
            .await
    }

    pub(crate) async fn dispatch_with_first_output(
        &self,
        worker_url: &str,
        prepared: PreparedRequest,
        on_first_output: Option<FirstOutputCallback>,
        on_prefill_progress: Option<PrefillProgressCallback>,
    ) -> Response {
        self.grpc
            .dispatch_prepared(
                worker_url,
                prepared,
                on_first_output,
                on_prefill_progress,
                KvHandoff::default(),
            )
            .await
    }

    pub fn remove_worker(&self, worker_url: &str) {
        self.grpc.remove_client(worker_url);
    }

    /// P/D prefill leg: one sampled token on `worker_url`. Returns the
    /// engine's `kv_transfer_params` for the decode leg, if it sent any.
    /// `on_prefill_progress` receives worker progress when it is enabled.
    pub(crate) async fn prefill(
        &self,
        worker_url: &str,
        prepared: &PreparedRequest,
        engine_request_id: String,
        kv_transfer_params: &serde_json::Value,
        on_prefill_progress: Option<PrefillProgressCallback>,
    ) -> Result<Option<serde_json::Value>, Response> {
        self.grpc
            .prefill_prepared(
                worker_url,
                prepared,
                engine_request_id,
                kv_transfer_params,
                on_prefill_progress,
            )
            .await
    }

    /// P/D decode leg: the client's chat on `worker_url`, reading prompt KV
    /// through `kv_transfer_params`.
    pub async fn dispatch_decode(
        &self,
        worker_url: &str,
        prepared: PreparedRequest,
        engine_request_id: String,
        kv_transfer_params: Option<serde_json::Value>,
    ) -> Response {
        let handoff = KvHandoff {
            engine_request_id: Some(engine_request_id),
            kv_transfer_params,
        };
        self.grpc
            .dispatch_prepared(worker_url, prepared, None, None, handoff)
            .await
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
            let removed = tokenized.token_ids.len() - limit;
            tokenized.token_ids = Arc::from(&tokenized.token_ids[removed..]);
            // L1 reuses a prefix; truncation keeps the suffix.
            tokenized.cached_tokens = tokenized.cached_tokens.saturating_sub(removed);
        }
    }
    Ok(())
}

/// HTTP pools proxy the request body, so the router never tokenizes and the
/// L0/L1 switches have no effect. Say so instead of ignoring them silently.
pub(crate) fn warn_ignored_encoding_caches() {
    for name in ignored_cache_switches(|name| std::env::var(name).ok()) {
        tracing::warn!("{name} is ignored: HTTP worker pools do not tokenize in the router");
    }
}

fn ignored_cache_switches(read: impl Fn(&str) -> Option<String>) -> Vec<&'static str> {
    ["VLLM_ROUTER_L0_CACHE", "VLLM_ROUTER_L1_CACHE"]
        .into_iter()
        .filter(|name| !matches!(read(name).as_deref(), None | Some("" | "0" | "false")))
        .collect()
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;

    #[test]
    fn truncation_counts_only_cached_tokens_in_retained_suffix() {
        let request = serde_json::from_value(serde_json::json!({
            "messages": [], "truncate_prompt_tokens": 3
        }))
        .unwrap();
        for (cached_tokens, expected) in [(0, 0), (2, 0), (4, 1), (6, 3)] {
            let mut tokenized = TokenizeOut {
                token_ids: Arc::from([1, 2, 3, 4, 5, 6]),
                adapt_ms: 0.0,
                template_ms: 0.0,
                encode_ms: 0.0,
                encode_backend: "test",
                cached_tokens,
            };
            truncate_grpc_prompt(&request, &mut tokenized).unwrap();
            assert_eq!(tokenized.token_ids.as_ref(), &[4, 5, 6]);
            assert_eq!(tokenized.cached_tokens, expected);
        }
    }

    #[test]
    fn http_pools_report_requested_cache_switches() {
        let read = |values: &'static [(&'static str, &'static str)]| {
            move |name: &str| {
                values
                    .iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, value)| value.to_string())
            }
        };
        assert!(ignored_cache_switches(read(&[])).is_empty());
        assert!(ignored_cache_switches(read(&[("VLLM_ROUTER_L1_CACHE", "0")])).is_empty());
        assert_eq!(
            ignored_cache_switches(read(&[
                ("VLLM_ROUTER_L0_CACHE", "1"),
                ("VLLM_ROUTER_L1_CACHE", "true"),
            ])),
            ["VLLM_ROUTER_L0_CACHE", "VLLM_ROUTER_L1_CACHE"]
        );
    }
}
