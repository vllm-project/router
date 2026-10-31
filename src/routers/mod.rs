//! Router implementations

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
};
use std::fmt::Debug;

use crate::protocols::anthropic::MessagesRequest;
use crate::protocols::spec::{
    ChatCompletionRequest, CompletionRequest, EmbeddingRequest, GenerateRequest,
    InferenceGenerateRequest, RerankRequest, ResponsesRequest,
};

pub mod factory;
pub mod header_utils;
pub mod http;
pub mod router_manager;

pub use factory::RouterFactory;
// Re-export HTTP routers for convenience (keeps routers::openai_router path working)
pub use http::{openai_router, pd_router, pd_types, router};

/// Worker management trait for administrative operations
///
/// This trait is separate from RouterTrait to allow Send futures
/// for use in service discovery and other background tasks
/// Response extension set on responses that came from a worker (or from a
/// transparent proxy path). `/v1/messages` rewrites only responses without it
/// into the anthropic error shape, so worker errors always pass through as
/// sent, whatever their content type or size.
#[derive(Debug, Clone, Copy)]
pub struct UpstreamResponse;

#[async_trait]
pub trait WorkerManagement: Send + Sync {
    /// Add a worker to the router
    async fn add_worker(&self, worker_url: &str) -> Result<String, String>;

    /// Remove a worker from the router
    fn remove_worker(&self, worker_url: &str);

    /// Get all worker URLs
    fn get_worker_urls(&self) -> Vec<String>;
}

/// Core trait for all router implementations
///
/// This trait provides a unified interface for routing requests,
/// regardless of whether it's a regular router or PD router.
#[async_trait]
pub trait RouterTrait: Send + Sync + Debug + WorkerManagement {
    /// Get a reference to self as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;

    /// Return an operator snapshot when this Router owns Program scheduling.
    fn scheduling_diagnostics(&self) -> Option<serde_json::Value> {
        None
    }

    /// Route a health check request
    async fn health(&self, req: Request<Body>) -> Response;

    /// Route a health generate request
    async fn health_generate(&self, req: Request<Body>) -> Response;

    /// Get server information
    async fn get_server_info(&self, req: Request<Body>) -> Response;

    /// Get available models
    async fn get_models(&self, req: Request<Body>) -> Response;

    /// Get model information
    async fn get_model_info(&self, req: Request<Body>) -> Response;

    /// Route a generate request
    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response;

    /// Route a generate request to vLLM's disaggregated `/inference/v1/generate`.
    async fn route_inference_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &InferenceGenerateRequest,
        model_id: Option<&str>,
    ) -> Response;

    /// Route a chat completion request
    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response;

    /// Route a completion request
    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response;

    /// Route a responses request
    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response;

    /// Retrieve a stored/background response by id
    async fn get_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response;

    /// Cancel a background response by id
    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response;

    /// Delete a response by id
    async fn delete_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses delete endpoint not implemented",
        )
            .into_response()
    }

    /// List input items of a response by id
    async fn list_response_input_items(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses list input items endpoint not implemented",
        )
            .into_response()
    }

    /// Route embedding requests (OpenAI-compatible /v1/embeddings)
    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response;

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response;

    /// Route an anthropic messages request (`POST /v1/messages`).
    ///
    /// Routers without a typed implementation keep today's behavior and pass
    /// the request to `route_transparent`, which returns 404 when the
    /// transparent proxy is not supported.
    async fn route_messages(
        &self,
        headers: Option<&HeaderMap>,
        body: &MessagesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        match serde_json::to_value(body) {
            Ok(value) => {
                // The transparent path does not separate its own errors from
                // worker responses, so keep its responses untouched as before.
                let mut response = self
                    .route_transparent(headers, "/v1/messages", &Method::POST, value)
                    .await;
                response.extensions_mut().insert(UpstreamResponse);
                response
            }
            Err(e) => (
                StatusCode::BAD_REQUEST,
                format!("Invalid request body: {e}"),
            )
                .into_response(),
        }
    }

    /// Flush cache on all workers
    async fn flush_cache(&self) -> Response;

    /// Get worker loads (for monitoring)
    async fn get_worker_loads(&self) -> Response;

    /// Get router type name
    fn router_type(&self) -> &'static str;

    /// Check if this is a PD router
    fn is_pd_mode(&self) -> bool {
        self.router_type() == "pd"
    }

    /// Server liveness check - is the server process running
    fn liveness(&self) -> Response {
        // Simple liveness check - if we can respond, we're alive
        (StatusCode::OK, "OK").into_response()
    }

    /// Server readiness check - is the server ready to handle requests
    fn readiness(&self) -> Response;

    /// Route a transparent proxy request (any path/body)
    /// Used for catch-all routing of unmatched paths
    /// Returns the response from the backend or an error response
    async fn route_transparent(
        &self,
        _headers: Option<&HeaderMap>,
        _path: &str,
        _method: &Method,
        _body: serde_json::Value,
    ) -> Response {
        // Default: not supported - return 404
        (StatusCode::NOT_FOUND, "Not Found").into_response()
    }
}
