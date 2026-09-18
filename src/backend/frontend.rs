//! Chat frontend for an all-`grpc://` worker pool.
//!
//! `prepare` tokenizes once (outside policy / retry). `dispatch` sends
//! the prepared `token_ids` to a chosen worker (`convert` + GenerateStream
//! + detok). Policy stays outside this type.

use std::sync::Arc;
use std::time::Instant;

use axum::response::Response;

use super::grpc::GrpcEngineBackend;
use super::preprocess::{tokenize_chat_request_timed, FrontendHandle, TokenizeOut, TokenizerCache};
use crate::protocols::spec::ChatCompletionRequest;

/// Tokenized chat ready for policy + `dispatch`.
#[derive(Clone)]
pub struct PreparedChat {
    pub request: ChatCompletionRequest,
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

    pub fn from_env() -> Self {
        Self {
            tokenizer: Arc::new(TokenizerCache::new()),
            grpc: GrpcEngineBackend::new(),
        }
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
            request,
            tokenized,
            handle,
            resolve_ms,
            t_req,
        })
    }

    /// Convert + GenerateStream + detok on the selected worker URL.
    pub async fn dispatch(&self, worker_url: &str, prepared: PreparedChat) -> Response {
        self.grpc
            .dispatch_prepared(
                worker_url,
                &prepared.request,
                prepared.tokenized,
                prepared.handle,
                prepared.resolve_ms,
                prepared.t_req,
            )
            .await
    }
}
