//! Router implementations

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
};
use std::fmt::Debug;

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

    /// Preserve validated ingress bytes where a router supports transparent
    /// Completion forwarding. Other backends retain their typed behavior.
    async fn route_completion_raw(
        &self,
        headers: Option<&HeaderMap>,
        body: std::sync::Arc<CompletionRequest>,
        raw: &axum::body::Bytes,
        model_id: Option<&str>,
    ) -> Response {
        let _ = raw;
        self.route_completion(headers, body.as_ref(), model_id)
            .await
    }

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

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    // An implementation of the pre-existing borrowed API, with no raw override.
    #[derive(Debug)]
    struct LegacyRouter {
        request: Arc<CompletionRequest>,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl WorkerManagement for LegacyRouter {
        async fn add_worker(&self, _: &str) -> Result<String, String> {
            Ok(String::new())
        }
        fn remove_worker(&self, _: &str) {}
        fn get_worker_urls(&self) -> Vec<String> {
            vec![]
        }
    }

    #[async_trait]
    impl RouterTrait for LegacyRouter {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        async fn health(&self, _: Request<Body>) -> Response {
            unreachable!()
        }
        async fn health_generate(&self, _: Request<Body>) -> Response {
            unreachable!()
        }
        async fn get_server_info(&self, _: Request<Body>) -> Response {
            unreachable!()
        }
        async fn get_models(&self, _: Request<Body>) -> Response {
            unreachable!()
        }
        async fn get_model_info(&self, _: Request<Body>) -> Response {
            unreachable!()
        }
        async fn route_generate(
            &self,
            _: Option<&HeaderMap>,
            _: &GenerateRequest,
            _: Option<&str>,
        ) -> Response {
            unreachable!()
        }
        async fn route_inference_generate(
            &self,
            _: Option<&HeaderMap>,
            _: &InferenceGenerateRequest,
            _: Option<&str>,
        ) -> Response {
            unreachable!()
        }
        async fn route_chat(
            &self,
            _: Option<&HeaderMap>,
            _: &ChatCompletionRequest,
            _: Option<&str>,
        ) -> Response {
            unreachable!()
        }
        async fn route_completion(
            &self,
            headers: Option<&HeaderMap>,
            body: &CompletionRequest,
            model_id: Option<&str>,
        ) -> Response {
            assert!(std::ptr::eq(body, self.request.as_ref()));
            assert_eq!(headers.unwrap()["x-legacy-test"], "preserved");
            assert_eq!(model_id, Some("legacy"));
            self.calls.fetch_add(1, Ordering::Relaxed);
            StatusCode::ACCEPTED.into_response()
        }
        async fn route_responses(
            &self,
            _: Option<&HeaderMap>,
            _: &ResponsesRequest,
            _: Option<&str>,
        ) -> Response {
            unreachable!()
        }
        async fn get_response(&self, _: Option<&HeaderMap>, _: &str) -> Response {
            unreachable!()
        }
        async fn cancel_response(&self, _: Option<&HeaderMap>, _: &str) -> Response {
            unreachable!()
        }
        async fn route_embeddings(
            &self,
            _: Option<&HeaderMap>,
            _: &EmbeddingRequest,
            _: Option<&str>,
        ) -> Response {
            unreachable!()
        }
        async fn route_rerank(
            &self,
            _: Option<&HeaderMap>,
            _: &RerankRequest,
            _: Option<&str>,
        ) -> Response {
            unreachable!()
        }
        async fn flush_cache(&self) -> Response {
            unreachable!()
        }
        async fn get_worker_loads(&self) -> Response {
            unreachable!()
        }
        fn router_type(&self) -> &'static str {
            "legacy"
        }
        fn readiness(&self) -> Response {
            unreachable!()
        }
    }

    #[tokio::test]
    async fn legacy_completion_trait_accepts_borrowed_and_owned_raw_dispatch() {
        let raw = axum::body::Bytes::from_static(br#"{ "model": "legacy", "prompt": "hello" }"#);
        let request: Arc<CompletionRequest> = Arc::new(serde_json::from_slice(&raw).unwrap());
        let legacy = Arc::new(LegacyRouter {
            request: request.clone(),
            calls: AtomicUsize::new(0),
        });
        let router: Arc<dyn RouterTrait> = legacy.clone();
        let mut headers = HeaderMap::new();
        headers.insert("x-legacy-test", "preserved".parse().unwrap());
        assert_eq!(
            router
                .route_completion(Some(&headers), request.as_ref(), Some("legacy"))
                .await
                .status(),
            StatusCode::ACCEPTED
        );
        assert_eq!(
            router
                .route_completion_raw(Some(&headers), request.clone(), &raw, Some("legacy"))
                .await
                .status(),
            StatusCode::ACCEPTED
        );

        let manager = router_manager::RouterManager::new(
            crate::config::RouterConfig::default(),
            reqwest::Client::new(),
            Arc::new(crate::core::WorkerRegistry::new()),
            Arc::new(crate::policies::PolicyRegistry::new(
                crate::config::PolicyConfig::RoundRobin,
            )),
        );
        manager.register_router(router_manager::RouterId::new("legacy".into()), router);
        assert_eq!(
            manager
                .route_completion_raw(Some(&headers), request, &raw, None)
                .await
                .status(),
            StatusCode::ACCEPTED
        );
        assert_eq!(legacy.calls.load(Ordering::Relaxed), 3);
    }
}
