// gRPC Router Implementation

use crate::config::types::RetryConfig;
use crate::core::{
    BasicWorker, CircuitBreakerConfig, HealthChecker, HealthConfig, Worker, WorkerType,
};
use crate::grpc::VllmSchedulerClient;
use crate::metrics::RouterMetrics;
use crate::policies::LoadBalancingPolicy;
use crate::routers::{RouterTrait, WorkerManagement};
use crate::tokenizer::traits::Tokenizer;
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tracing::{info, warn};

/// gRPC router implementation for VLLM
#[allow(dead_code)] // Fields will be used once implementation is complete
pub struct GrpcRouter {
    /// Worker connections
    workers: Arc<RwLock<Vec<Arc<dyn Worker>>>>,
    /// gRPC clients for each worker
    grpc_clients: Arc<RwLock<HashMap<String, VllmSchedulerClient>>>,
    /// Load balancing policy
    policy: Arc<dyn LoadBalancingPolicy>,
    /// Tokenizer for handling text encoding/decoding
    tokenizer: Arc<dyn Tokenizer>,
    /// Worker health checker
    _health_checker: Option<HealthChecker>,
    /// Configuration
    timeout_secs: u64,
    interval_secs: u64,
    intra_node_data_parallel_size: usize,
    api_key: Option<String>,
    retry_config: RetryConfig,
    circuit_breaker_config: CircuitBreakerConfig,
}

impl GrpcRouter {
    /// Create a new gRPC router
    pub async fn new(
        worker_urls: Vec<String>,
        policy: Arc<dyn LoadBalancingPolicy>,
        ctx: &Arc<crate::server::AppContext>,
    ) -> Result<Self, String> {
        // Update metrics
        RouterMetrics::set_active_workers(worker_urls.len());

        // Extract necessary components from context
        let tokenizer = ctx
            .tokenizer
            .as_ref()
            .ok_or_else(|| "gRPC router requires tokenizer".to_string())?
            .clone();

        // Convert config CircuitBreakerConfig to core CircuitBreakerConfig
        let circuit_breaker_config = ctx.router_config.effective_circuit_breaker_config();
        let core_cb_config = CircuitBreakerConfig {
            failure_threshold: circuit_breaker_config.failure_threshold,
            success_threshold: circuit_breaker_config.success_threshold,
            timeout_duration: Duration::from_secs(circuit_breaker_config.timeout_duration_secs),
            window_duration: Duration::from_secs(circuit_breaker_config.window_duration_secs),
        };

        // Create gRPC clients for each worker
        let mut grpc_clients = HashMap::new();
        for url in &worker_urls {
            match VllmSchedulerClient::connect(url).await {
                Ok(client) => {
                    grpc_clients.insert(url.clone(), client);
                    info!("Connected to gRPC worker at {}", url);
                }
                Err(e) => {
                    warn!("Failed to connect to gRPC worker at {}: {}", url, e);
                    // Continue with other workers
                }
            }
        }

        if grpc_clients.is_empty() {
            return Err("Failed to connect to any gRPC workers".to_string());
        }

        // Create Worker trait objects with gRPC connection mode
        let mut workers: Vec<Arc<dyn Worker>> = Vec::new();

        // Move clients from the HashMap to the workers
        for url in &worker_urls {
            if let Some(client) = grpc_clients.remove(url) {
                let worker = BasicWorker::with_connection_mode(
                    url.clone(),
                    WorkerType::Regular,
                    crate::core::ConnectionMode::Grpc { port: None },
                )
                .with_circuit_breaker_config(core_cb_config.clone())
                .with_health_config(HealthConfig {
                    timeout_secs: ctx.router_config.health_check.timeout_secs,
                    check_interval_secs: ctx.router_config.health_check.check_interval_secs,
                    endpoint: ctx.router_config.health_check.endpoint.clone(),
                    failure_threshold: ctx.router_config.health_check.failure_threshold,
                    success_threshold: ctx.router_config.health_check.success_threshold,
                })
                .with_grpc_client(client);

                workers.push(Arc::new(worker) as Arc<dyn Worker>);
            } else {
                warn!("No gRPC client for worker {}, skipping", url);
            }
        }

        // Initialize policy with workers if needed
        if let Some(cache_aware) = policy
            .as_any()
            .downcast_ref::<crate::policies::CacheAwarePolicy>()
        {
            cache_aware.init_workers(&workers);
        }

        let workers = Arc::new(RwLock::new(workers));
        let health_checker = crate::core::start_health_checker(
            Arc::clone(&workers),
            ctx.router_config.worker_startup_check_interval_secs,
        );

        Ok(GrpcRouter {
            workers,
            grpc_clients: Arc::new(RwLock::new(grpc_clients)),
            policy,
            tokenizer,
            _health_checker: Some(health_checker),
            timeout_secs: ctx.router_config.worker_startup_timeout_secs,
            interval_secs: ctx.router_config.worker_startup_check_interval_secs,
            intra_node_data_parallel_size: ctx.router_config.intra_node_data_parallel_size,
            api_key: ctx.router_config.api_key.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            circuit_breaker_config: core_cb_config,
        })
    }
}

impl std::fmt::Debug for GrpcRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrpcRouter")
            .field("workers_count", &self.workers.read().unwrap().len())
            .field("timeout_secs", &self.timeout_secs)
            .field("interval_secs", &self.interval_secs)
            .field(
                "intra_node_data_parallel_size",
                &self.intra_node_data_parallel_size,
            )
            .finish()
    }
}

#[async_trait]
impl RouterTrait for GrpcRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_models(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_generate(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_chat(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &serde_json::Value,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::ResponsesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_rerank(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::RerankRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn flush_cache(&self) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_worker_loads(&self) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    fn router_type(&self) -> &'static str {
        "grpc"
    }

    fn readiness(&self) -> Response {
        (StatusCode::SERVICE_UNAVAILABLE).into_response()
    }
}

#[async_trait]
impl WorkerManagement for GrpcRouter {
    async fn add_worker(&self, _worker_url: &str) -> Result<String, String> {
        Err("Not implemented".to_string())
    }

    fn remove_worker(&self, _worker_url: &str) {}

    fn get_worker_urls(&self) -> Vec<String> {
        self.workers
            .read()
            .unwrap()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }
}
