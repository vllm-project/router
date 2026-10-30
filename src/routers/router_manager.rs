//! Router Manager for coordinating multiple routers and workers
//!
//! Provides centralized management based on enable_igw flag:
//! - Single Router Mode (enable_igw=false): Router owns workers directly
//! - Multi-Router Mode (enable_igw=true): RouterManager coordinates everything

use crate::config::RouterConfig;
use crate::core::{CircuitBreakerConfig, Worker, WorkerFactory, WorkerRegistry, WorkerType};
use crate::policies::kv_aware_ops;
use crate::protocols::spec::{
    ChatCompletionRequest, CompletionRequest, EmbeddingRequest, GenerateRequest,
    InferenceGenerateRequest, RerankRequest, ResponsesRequest,
};
use crate::protocols::worker_spec::{
    WorkerApiResponse, WorkerConfigRequest, WorkerErrorResponse, WorkerInfo, WorkerListResponse,
    WorkerStats, WorkerTypeStats,
};
use crate::routers::{RouterTrait, WorkerManagement};
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
};
use dashmap::DashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

/// Router identifier
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct RouterId(String);

impl RouterId {
    pub fn new(id: String) -> Self {
        Self(id)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Router Manager - Central coordinator for routers and workers
pub struct RouterManager {
    /// Worker registry (single source of truth in multi-router mode)
    worker_registry: Arc<WorkerRegistry>,

    /// Policy registry for managing model-to-policy mappings
    policy_registry: Arc<crate::policies::PolicyRegistry>,

    /// Keep management API topology changes and policy notifications in order.
    /// Never held during backend discovery or across an await.
    worker_mutation: parking_lot::Mutex<()>,

    /// All routers managed by this manager
    /// RouterId examples: "http-regular", "http-pd"
    routers: Arc<DashMap<RouterId, Arc<dyn RouterTrait>>>,

    /// Default router for requests without specific routing
    default_router: Arc<std::sync::RwLock<Option<RouterId>>>,

    /// HTTP client for querying worker info
    client: reqwest::Client,

    /// Configuration
    #[allow(dead_code)] // May be used in future enhancements
    config: RouterConfig,
}

impl RouterManager {
    /// Create a new router manager with shared registries
    pub fn new(
        config: RouterConfig,
        client: reqwest::Client,
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<crate::policies::PolicyRegistry>,
    ) -> Self {
        Self {
            worker_registry,
            policy_registry,
            worker_mutation: parking_lot::Mutex::new(()),
            routers: Arc::new(DashMap::new()),
            default_router: Arc::new(std::sync::RwLock::new(None)),
            client,
            config,
        }
    }

    /// Register a router with the manager
    pub fn register_router(&self, id: RouterId, router: Arc<dyn RouterTrait>) {
        // Store router
        self.routers.insert(id.clone(), router);

        // Set as default if first router
        let mut default_router = self.default_router.write().unwrap();
        if default_router.is_none() {
            *default_router = Some(id.clone());
            info!("Set default router to {}", id.as_str());
        }
    }

    /// Set the default router
    pub fn set_default_router(&self, id: RouterId) {
        let mut default_router = self.default_router.write().unwrap();
        *default_router = Some(id);
    }

    /// Get the number of registered routers
    pub fn router_count(&self) -> usize {
        self.routers.len()
    }

    /// Get router for a specific model based on worker types
    pub fn get_router_for_model(&self, model_id: &str) -> Option<Arc<dyn RouterTrait>> {
        // Query workers for this model from registry
        let workers = self.worker_registry.get_consumer_workers(Some(model_id));

        if !workers.is_empty() {
            // Determine router based on worker types
            let has_pd_workers = workers.iter().any(|w| {
                matches!(
                    w.worker_type(),
                    WorkerType::Prefill { .. } | WorkerType::Decode
                )
            });

            let router_id = if has_pd_workers {
                RouterId::new("http-pd".to_string())
            } else {
                RouterId::new("http-regular".to_string())
            };

            // Return the router if it exists
            if let Some(router) = self.routers.get(&router_id) {
                return Some(router.clone());
            }
        }

        // Fall back to default router
        let default_router = self.default_router.read().unwrap();
        if let Some(ref default_id) = *default_router {
            self.routers.get(default_id).map(|r| r.clone())
        } else {
            None
        }
    }

    /// Get workers for routing decision
    pub fn get_workers_for_request(&self, model_id: Option<&str>) -> Vec<Arc<dyn Worker>> {
        self.worker_registry.get_consumer_workers(model_id)
    }

    /// Add a worker to the registry
    pub async fn add_worker(
        &self,
        config: WorkerConfigRequest,
    ) -> Result<WorkerApiResponse, WorkerErrorResponse> {
        // Build labels from configuration
        let mut labels = config.labels.clone();

        // Discover the model through the backend's standard model API if the
        // caller did not provide it.
        let model_id = if let Some(model_id) = config.model_id {
            model_id
        } else {
            match self.query_model_id(&config.url).await {
                Ok(model_id) => model_id,
                Err(e) => {
                    warn!("Failed to discover model from {}: {}", config.url, e);
                    "unknown".to_string()
                }
            }
        };

        // Add configuration to labels
        labels.insert("model_id".to_string(), model_id.clone());

        if let Some(priority) = config.priority {
            labels.insert("priority".to_string(), priority.to_string());
        }

        if let Some(cost) = config.cost {
            labels.insert("cost".to_string(), cost.to_string());
        }

        let worker = match config.worker_type.as_deref() {
            Some("prefill") => WorkerFactory::create_prefill_with_labels(
                config.url.clone(),
                config.bootstrap_port,
                labels.clone(),
                CircuitBreakerConfig::default(),
            ),
            Some("decode") => WorkerFactory::create_decode_with_labels(
                config.url.clone(),
                labels.clone(),
                CircuitBreakerConfig::default(),
            ),
            _ => WorkerFactory::create_regular_with_labels(
                config.url.clone(),
                labels.clone(),
                CircuitBreakerConfig::default(),
            ),
        };

        // Serialize the registry update and its policy notification. Discovery
        // above can await, but this critical section only updates local state.
        let _mutation = self.worker_mutation.lock();
        let worker_arc: Arc<dyn Worker> = Arc::from(worker);
        let registration = self
            .worker_registry
            .register_with_previous(worker_arc.clone());
        let worker_id = registration.worker_id;

        // Notify PolicyRegistry about the new worker
        // Extract policy hint from labels if provided
        let policy_hint = labels.get("policy").map(|s| s.as_str());
        // This API creates consumers. A replaced Encode worker never counted
        // toward consumer policies, even when it shares the new model ID.
        let policy = match registration
            .previous
            .filter(|previous| previous.worker_type().is_consumer())
        {
            Some(previous) if previous.model_id() == model_id => {
                self.policy_registry.get_policy_or_default(&model_id)
            }
            previous => {
                if let Some(previous) = previous {
                    self.policy_registry.on_worker_removed(previous.model_id());
                }
                self.policy_registry.on_worker_added(&model_id, policy_hint)
            }
        };
        if worker_arc.worker_type() == WorkerType::Regular {
            if let Some(kv) = kv_aware_ops(policy.as_ref()) {
                kv.register_worker(worker_arc.as_ref(), None, None);
            }
            let default = self.policy_registry.get_default_policy();
            if !Arc::ptr_eq(&policy, &default) {
                if let Some(kv) = kv_aware_ops(default.as_ref()) {
                    kv.register_worker(worker_arc.as_ref(), None, None);
                }
            }
        }

        // Log which type of router would handle this worker (for debugging)
        let expected_router = match config.worker_type.as_deref() {
            Some("prefill") | Some("decode") => "http-pd",
            _ => "http-regular",
        };

        info!(
            "Worker for model '{}' would be handled by '{}' router based on type",
            model_id, expected_router
        );

        info!(
            "Added worker {} with URL {} for model {} using policy {}",
            worker_id.as_str(),
            config.url,
            model_id,
            policy.name()
        );

        // Return worker info
        let worker_info = self.worker_to_info(worker_id.as_str(), &worker_arc);

        Ok(WorkerApiResponse {
            success: true,
            message: format!("Worker {} added successfully", worker_id.as_str()),
            worker: Some(worker_info),
        })
    }

    /// Remove a worker from the registry
    pub fn remove_worker_from_registry(
        &self,
        url: &str,
    ) -> Result<WorkerApiResponse, WorkerErrorResponse> {
        let _mutation = self.worker_mutation.lock();
        if let Some(worker) = self.worker_registry.remove_by_url(url) {
            if worker.worker_type() == WorkerType::Regular {
                let policy = self
                    .policy_registry
                    .get_policy_or_default(worker.model_id());
                if let Some(kv) = kv_aware_ops(policy.as_ref()) {
                    kv.unregister_worker(url);
                }
                let default = self.policy_registry.get_default_policy();
                if !Arc::ptr_eq(&policy, &default) {
                    if let Some(kv) = kv_aware_ops(default.as_ref()) {
                        kv.unregister_worker(url);
                    }
                }
            }
            // Notify for the object actually removed, not an earlier lookup.
            self.policy_registry
                .notify_consumer_removed(worker.as_ref());
            info!(
                "Removed worker with URL {} for model {}",
                url,
                worker.model_id()
            );

            Ok(WorkerApiResponse {
                success: true,
                message: format!("Worker {} removed successfully", url),
                worker: None,
            })
        } else {
            Err(WorkerErrorResponse {
                error: format!("Worker with URL {} not found", url),
                code: "WORKER_NOT_FOUND".to_string(),
            })
        }
    }

    /// List all workers
    pub fn list_workers(&self) -> WorkerListResponse {
        let workers = self.worker_registry.get_all_with_ids();
        let worker_infos: Vec<WorkerInfo> = workers
            .iter()
            .map(|(id, w)| self.worker_to_info(id.as_str(), w))
            .collect();

        let total = worker_infos.len();

        // Get stats from the worker registry
        let registry_stats = self.worker_registry.stats();

        // Convert WorkerRegistryStats to WorkerStats
        let stats = WorkerStats {
            total_workers: registry_stats.total_workers,
            healthy_workers: registry_stats.healthy_workers,
            total_models: registry_stats.total_models,
            total_load: registry_stats.total_load,
            by_type: WorkerTypeStats {
                regular: registry_stats.regular_workers,
                prefill: registry_stats.prefill_workers,
                decode: registry_stats.decode_workers,
                encode: registry_stats.encode_workers,
            },
        };

        WorkerListResponse {
            workers: worker_infos,
            total,
            stats,
        }
    }

    /// Get worker by URL
    pub fn get_worker(&self, url: &str) -> Option<WorkerInfo> {
        self.worker_registry
            .get_by_url(url)
            .map(|w| self.worker_to_info("unknown", &w))
    }

    /// Query model identity through OpenAI `/v1/models` or the gRPC control API.
    async fn query_model_id(&self, url: &str) -> Result<String, String> {
        if crate::backend::is_grpc_url(url) {
            let info = crate::backend::get_grpc_model_info(url, Duration::from_secs(10)).await?;
            let model_id = if info.served_model_name.is_empty() {
                info.model_id
            } else {
                info.served_model_name
            };
            return (!model_id.trim().is_empty())
                .then_some(model_id)
                .ok_or_else(|| format!("gRPC GetModelInfo returned no model id for {url}"));
        }

        let models_url = format!("{}/v1/models", url.trim_end_matches('/'));
        let response = self
            .client
            .get(&models_url)
            .send()
            .await
            .map_err(|error| format!("request to {models_url} failed: {error}"))?;
        if !response.status().is_success() {
            return Err(format!(
                "{models_url} returned status {}",
                response.status()
            ));
        }
        let models: serde_json::Value = response
            .json()
            .await
            .map_err(|error| format!("invalid model list from {models_url}: {error}"))?;
        models["data"]
            .as_array()
            .and_then(|data| data.first())
            .and_then(|model| model["id"].as_str())
            .filter(|model_id| !model_id.trim().is_empty())
            .map(str::to_string)
            .ok_or_else(|| format!("{models_url} returned no model id"))
    }

    /// Convert Worker to WorkerInfo
    fn worker_to_info(&self, id: &str, worker: &Arc<dyn Worker>) -> WorkerInfo {
        let metadata = worker.metadata();

        WorkerInfo {
            id: id.to_string(),
            url: worker.url().to_string(),
            model_id: worker.model_id().to_string(),
            priority: worker.priority(),
            cost: worker.cost(),
            worker_type: match worker.worker_type() {
                WorkerType::Regular => "regular".to_string(),
                WorkerType::Prefill { .. } => "prefill".to_string(),
                WorkerType::Decode => "decode".to_string(),
                WorkerType::Encode => "encode".to_string(),
            },
            is_healthy: worker.is_healthy(),
            load: worker.load(),
            connection_mode: format!("{:?}", worker.connection_mode()),
            metadata: metadata.labels.clone(),
        }
    }

    /// Get the appropriate router for a request based on headers and request content
    pub fn select_router_for_request(
        &self,
        headers: Option<&HeaderMap>,
        model_id: Option<&str>,
    ) -> Option<Arc<dyn RouterTrait>> {
        // Extract priority and cost preferences from headers if available
        let _priority_threshold = headers.and_then(|h| {
            h.get("x-worker-priority")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u32>().ok())
        });

        let _max_cost = headers.and_then(|h| {
            h.get("x-max-cost")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<f32>().ok())
        });

        // Check if PD (prefill-decode) mode is preferred from headers
        let prefer_pd = headers
            .and_then(|h| {
                h.get("x-prefer-pd")
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s == "true" || s == "1")
            })
            .unwrap_or(false);

        // If model specified, use get_router_for_model
        let candidate_routers = if let Some(model) = model_id {
            if let Some(router) = self.get_router_for_model(model) {
                vec![router]
            } else {
                Vec::new()
            }
        } else {
            // No model specified, consider all routers
            self.routers
                .iter()
                .map(|entry| entry.value().clone())
                .collect::<Vec<_>>()
        };

        if candidate_routers.is_empty() {
            // No routers found for the specified model
            return None;
        }

        // Score routers based on worker attributes and request preferences
        let mut best_router = None;
        let mut best_score = 0.0;

        for router in candidate_routers {
            let mut score = 1.0;

            // Check if this is a PD router
            let is_pd = router.is_pd_mode();
            if prefer_pd && is_pd {
                score += 2.0; // Bonus for matching PD preference
            } else if !prefer_pd && !is_pd {
                score += 1.0; // Bonus for matching regular preference
            }

            // Get workers for this router and evaluate based on priority/cost
            // Note: This would require routers to expose their workers or stats
            // For now, we'll use a simple selection based on router type

            // TODO: Once routers expose worker stats, we can evaluate:
            // - Average worker priority vs priority_threshold
            // - Average worker cost vs max_cost
            // - Current load and health status

            if score > best_score {
                best_score = score;
                best_router = Some(router);
            }
        }

        best_router
    }
}

/// RouterManager implements RouterTrait to act as a meta-router
/// that delegates requests to the appropriate underlying router
#[async_trait]
impl WorkerManagement for RouterManager {
    /// Add a worker - in multi-router mode, this adds to the registry
    async fn add_worker(&self, worker_url: &str) -> Result<String, String> {
        // Create a basic worker config request
        let config = WorkerConfigRequest {
            url: worker_url.to_string(),
            model_id: None,
            worker_type: None,
            priority: None,
            cost: None,
            labels: std::collections::HashMap::new(),
            bootstrap_port: None,
        };

        match self.add_worker(config).await {
            Ok(response) => Ok(response.message),
            Err(e) => Err(e.error),
        }
    }

    /// Remove a worker from the registry
    fn remove_worker(&self, worker_url: &str) {
        let _ = self.remove_worker_from_registry(worker_url);
    }

    /// Get all worker URLs from the registry
    fn get_worker_urls(&self) -> Vec<String> {
        self.worker_registry.get_all_urls()
    }
}

#[async_trait]
impl RouterTrait for RouterManager {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Health check - return 503 if no routers available
    async fn health(&self, _req: Request<Body>) -> Response {
        // Health check should succeed if RouterManager exists, even without routers
        // Individual router health can be checked via specific endpoints
        (StatusCode::OK, "RouterManager is healthy").into_response()
    }

    /// Health generate - check if any router can handle generate requests
    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Return 503 since we have no routers with workers
        // TODO: Should check if any router has healthy workers
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "No routers with healthy workers available",
        )
            .into_response()
    }

    /// Get server information - aggregate from all routers
    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // TODO: Aggregate info from all routers with healthy workers
        // For now, return basic info about the RouterManager
        (
            StatusCode::OK,
            serde_json::json!({
                "router_manager": true,
                "routers_count": self.routers.len(),
                "workers_count": self.worker_registry.get_all().len()
            })
            .to_string(),
        )
            .into_response()
    }

    /// Get available models - query from worker registry
    async fn get_models(&self, _req: Request<Body>) -> Response {
        // Get models from worker registry
        let models = self.worker_registry.get_models();

        if models.is_empty() {
            (StatusCode::SERVICE_UNAVAILABLE, "No models available").into_response()
        } else {
            (
                StatusCode::OK,
                serde_json::json!({
                    "models": models
                })
                .to_string(),
            )
                .into_response()
        }
    }

    /// Get model information
    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        // TODO: Extract model from request and route to appropriate router
        // For now, return not implemented
        (
            StatusCode::NOT_IMPLEMENTED,
            "Model info endpoint not yet implemented in RouterManager",
        )
            .into_response()
    }

    /// Route a generate request
    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        // Select router based on headers
        // GenerateRequest doesn't have a model field
        let router = self.select_router_for_request(headers, None);

        if let Some(router) = router {
            // In multi-model mode, pass None since GenerateRequest doesn't have model field
            router.route_generate(headers, body, None).await
        } else {
            // Return 404 when no router is available for the request
            (
                StatusCode::NOT_FOUND,
                "No router available for this request",
            )
                .into_response()
        }
    }

    async fn route_inference_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &InferenceGenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, None);

        if let Some(router) = router {
            router.route_inference_generate(headers, body, None).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for this request",
            )
                .into_response()
        }
    }

    /// Route a chat completion request
    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        // Select router based on headers and model.
        // When model is None, select_router_for_request considers all registered
        // routers (including PD routers), so a single-model deployment will still
        // route correctly even without the model field.
        let model_id = body.model.as_deref();
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_chat(headers, body, model_id).await
        } else {
            let msg = match model_id {
                Some(m) => format!("Model '{}' not found or no router available", m),
                None => "No routers registered to handle this request".to_string(),
            };
            (StatusCode::NOT_FOUND, msg).into_response()
        }
    }

    /// Route a completion request
    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        // Select router based on headers and model.
        // When model is None, select_router_for_request considers all registered
        // routers (including PD routers), so a single-model deployment will still
        // route correctly even without the model field.
        let model_id = body.model.as_deref();
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_completion(headers, body, model_id).await
        } else {
            let msg = match model_id {
                Some(m) => format!("Model '{}' not found or no router available", m),
                None => "No routers registered to handle this request".to_string(),
            };
            (StatusCode::NOT_FOUND, msg).into_response()
        }
    }

    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ResponsesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "responses api not yet implemented in inference gateway mode",
        )
            .into_response()
    }

    async fn delete_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "responses api not yet implemented in inference gateway mode",
        )
            .into_response()
    }

    async fn list_response_input_items(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "responses api not yet implemented in inference gateway mode",
        )
            .into_response()
    }

    async fn get_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.get_response(headers, response_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("No router available to get response '{}'", response_id),
            )
                .into_response()
        }
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.cancel_response(headers, response_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("No router available to cancel response '{}'", response_id),
            )
                .into_response()
        }
    }

    /// Route embeddings request
    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        // Select router based on headers and model.
        // When model is None, select_router_for_request considers all registered
        // routers, so a single-model deployment will still route correctly.
        let model_id = body.model.as_deref();
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_embeddings(headers, body, model_id).await
        } else {
            let msg = match model_id {
                Some(m) => format!("Model '{}' not found or no router available", m),
                None => "No routers registered to handle this request".to_string(),
            };
            (StatusCode::NOT_FOUND, msg).into_response()
        }
    }

    /// Route rerank request
    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Try to select a router based on headers
        let router = self.select_router_for_request(headers, None);

        if let Some(router) = router {
            router.route_rerank(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for rerank request",
            )
                .into_response()
        }
    }

    /// Flush cache on all routers and workers
    async fn flush_cache(&self) -> Response {
        // TODO: Call flush_cache on all routers that have workers
        // For now, return success if we have any routers
        if self.routers.is_empty() {
            (StatusCode::SERVICE_UNAVAILABLE, "No routers configured").into_response()
        } else {
            // TODO: Actually flush cache on all routers
            (StatusCode::OK, "Cache flush requested").into_response()
        }
    }

    /// Get worker loads from all routers
    async fn get_worker_loads(&self) -> Response {
        // Return worker loads from the registry
        let workers = self.worker_registry.get_all();
        let loads: Vec<serde_json::Value> = workers
            .iter()
            .map(|w| {
                serde_json::json!({
                    "url": w.url(),
                    "model": w.model_id(),
                    "load": w.load(),
                    "is_healthy": w.is_healthy()
                })
            })
            .collect();

        (
            StatusCode::OK,
            serde_json::json!({
                "workers": loads
            })
            .to_string(),
        )
            .into_response()
    }

    /// Get router type name
    fn router_type(&self) -> &'static str {
        "manager"
    }

    /// Server readiness check - check if any router is ready
    fn readiness(&self) -> Response {
        if self.routers.is_empty() {
            (StatusCode::SERVICE_UNAVAILABLE, "No routers configured").into_response()
        } else {
            // TODO: Check readiness of all routers
            (StatusCode::OK, "Ready").into_response()
        }
    }

    /// Route a transparent proxy request through the appropriate router
    async fn route_transparent(
        &self,
        headers: Option<&HeaderMap>,
        path: &str,
        method: &Method,
        body: serde_json::Value,
    ) -> Response {
        // Select router based on headers (no model info available for transparent proxy)
        let router = self.select_router_for_request(headers, None);

        if let Some(router) = router {
            router.route_transparent(headers, path, method, body).await
        } else {
            // Return 404 when no router is available
            (
                StatusCode::NOT_FOUND,
                "No router available for this request",
            )
                .into_response()
        }
    }
}

// Note: get_first_available_router removed - we now properly handle
// router selection based on model and worker availability

impl std::fmt::Debug for RouterManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RouterManager")
            .field("routers_count", &self.routers.len())
            .field("workers_count", &self.worker_registry.get_all().len())
            .field("default_router", &*self.default_router.read().unwrap())
            .finish()
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn inference_views_ignore_encode_only_workers() {
        let registry = Arc::new(WorkerRegistry::new());
        registry.register(Arc::from(WorkerFactory::create_encode(
            "http://encoder:8000".into(),
        )));
        let config = RouterConfig::default();
        let manager = RouterManager::new(
            config.clone(),
            reqwest::Client::new(),
            registry,
            Arc::new(crate::policies::PolicyRegistry::new(config.policy)),
        );
        assert!(manager.get_workers_for_request(None).is_empty());
        assert!(manager.get_workers_for_request(Some("unknown")).is_empty());
        assert!(manager.get_router_for_model("unknown").is_none());
        assert_eq!(manager.list_workers().total, 1);
    }

    #[tokio::test]
    async fn manager_readiness_preserves_registered_router_contract() {
        let config = RouterConfig {
            enable_igw: true,
            ..Default::default()
        };
        let context = Arc::new(
            crate::server::AppContext::new(
                config.clone(),
                reqwest::Client::new(),
                64,
                None,
                vec![],
            )
            .unwrap(),
        );
        let manager = RouterManager::new(
            config,
            context.client.clone(),
            context.worker_registry.clone(),
            context.policy_registry.clone(),
        );
        assert_eq!(
            manager.readiness().status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        let router = crate::routers::RouterFactory::create_regular_router(&[], &context)
            .await
            .unwrap();
        assert_eq!(router.readiness().status(), StatusCode::SERVICE_UNAVAILABLE);
        manager.register_router(RouterId::new("http-regular".into()), Arc::from(router));
        // Keep IGW's existing registration-based readiness; Phase 1 does not
        // introduce child-health readiness semantics for non-EPD deployments.
        assert_eq!(manager.readiness().status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn manager_model_listing_preserves_registry_view() {
        let registry = Arc::new(WorkerRegistry::new());
        for (url, role, model) in [
            ("http://encoder:8000", WorkerType::Encode, "encode-only"),
            ("http://consumer:8000", WorkerType::Regular, "inference"),
            ("http://consumer2:8000", WorkerType::Regular, "inference"),
        ] {
            registry.register(Arc::new(
                crate::core::BasicWorker::new(url.into(), role).with_labels(
                    std::collections::HashMap::from([("model_id".into(), model.into())]),
                ),
            ));
        }
        let expected = registry.get_models();
        let config = RouterConfig::default();
        let manager = RouterManager::new(
            config.clone(),
            reqwest::Client::new(),
            registry,
            Arc::new(crate::policies::PolicyRegistry::new(config.policy)),
        );
        let response = manager.get_models(Request::new(Body::empty())).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let models: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // Preserve registry contents/order rather than adding manager-only sorting.
        assert_eq!(models["models"], serde_json::json!(expected));
    }

    #[test]
    fn list_workers_reports_encode_role_and_count() {
        let registry = Arc::new(WorkerRegistry::new());
        registry.register(Arc::from(WorkerFactory::create_encode(
            "http://encoder:8080".to_string(),
        )));
        registry.register(Arc::from(WorkerFactory::create_regular(
            "http://consumer:8080".to_string(),
        )));
        let config = RouterConfig::default();
        let manager = RouterManager::new(
            config.clone(),
            reqwest::Client::new(),
            registry,
            Arc::new(crate::policies::PolicyRegistry::new(config.policy)),
        );
        let response = serde_json::to_value(manager.list_workers()).unwrap();
        assert_eq!(response["stats"]["by_type"]["encode"], 1);
        assert_eq!(response["stats"]["by_type"]["regular"], 1);
        assert_eq!(response["total"], 2);
        let encoder = response["workers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|worker| worker["url"] == "http://encoder:8080")
            .unwrap();
        assert_eq!(encoder["worker_type"], "encode");
        assert_eq!(encoder["connection_mode"], "Http");
    }
}
