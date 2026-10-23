//! Chat frontend for an all-`grpc://` worker pool.
//!
//! `prepare` tokenizes once (outside policy / retry). `dispatch` sends
//! the prepared `token_ids` to a chosen worker (`convert` + GenerateStream
//! + detok). Policy stays outside this type.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::response::Response;

use super::grpc::{FirstOutputCallback, GrpcEngineBackend, KvHandoff};
use super::preprocess::{tokenize_chat_request_timed, FrontendHandle, TokenizeOut, TokenizerCache};
use crate::protocols::spec::ChatCompletionRequest;
use crate::tokenizer::{TokenizerCacheConfig, TokenizerCacheStats};

/// Tokenized chat ready for policy + `dispatch`.
#[derive(Clone)]
pub struct PreparedChat {
    pub request: Arc<ChatCompletionRequest>,
    pub tokenized: TokenizeOut,
    pub handle: FrontendHandle,
    pub resolve_ms: f64,
    pub t_req: Instant,
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

    /// Chat template + encode. Call once, before `policy.select`.
    pub async fn prepare(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Result<PreparedChat, String> {
        let t_req = Instant::now();
        let handle = self
            .tokenizer
            .resolve(request.model.as_deref())
            .await
            .map_err(|e| format!("tokenizer: {e}"))?;
        if request.temperature.is_none() {
            request.temperature = Some(
                handle
                    .default_temperature()
                    .map_err(|e| format!("sampling defaults: {e}"))?
                    .unwrap_or(1.0),
            );
        }
        let resolve_ms = t_req.elapsed().as_secs_f64() * 1000.0;
        let tokenized = tokenize_chat_request_timed(&request, &handle)
            .map_err(|e| format!("preprocess: {e}"))?;
        Ok(PreparedChat {
            request: Arc::new(request),
            tokenized,
            handle,
            resolve_ms,
            t_req,
        })
    }

    /// Convert + GenerateStream + detok on the selected worker URL.
    pub async fn dispatch(&self, worker_url: &str, prepared: PreparedChat) -> Response {
        self.dispatch_with_first_output(worker_url, prepared, None)
            .await
    }

    pub(crate) async fn dispatch_with_first_output(
        &self,
        worker_url: &str,
        prepared: PreparedChat,
        on_first_output: Option<FirstOutputCallback>,
    ) -> Response {
        self.grpc
            .dispatch_prepared(worker_url, prepared, on_first_output, KvHandoff::default())
            .await
    }

    pub fn remove_worker(&self, worker_url: &str) {
        self.grpc.remove_client(worker_url);
    }

    /// P/D prefill leg: one sampled token on `worker_url`. Returns the
    /// engine's `kv_transfer_params` for the decode leg, if it sent any.
    pub async fn prefill(
        &self,
        worker_url: &str,
        prepared: &PreparedChat,
        engine_request_id: String,
        kv_transfer_params: &serde_json::Value,
    ) -> Result<Option<serde_json::Value>, Response> {
        self.grpc
            .prefill_prepared(worker_url, prepared, engine_request_id, kv_transfer_params)
            .await
    }

    /// P/D decode leg: the client's chat on `worker_url`, reading prompt KV
    /// through `kv_transfer_params`.
    pub async fn dispatch_decode(
        &self,
        worker_url: &str,
        prepared: PreparedChat,
        engine_request_id: String,
        kv_transfer_params: Option<serde_json::Value>,
    ) -> Response {
        let handoff = KvHandoff {
            engine_request_id: Some(engine_request_id),
            kv_transfer_params,
        };
        self.grpc
            .dispatch_prepared(worker_url, prepared, None, handoff)
            .await
    }
}
