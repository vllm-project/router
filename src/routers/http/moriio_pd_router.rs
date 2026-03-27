// MoRIIO PD (Prefill-Decode) Router Implementation
// Handles AMD MoRIIO connector two-stage request routing in READ and WRITE transfer modes.

use super::moriio_service_discovery::{MoriIOInstance, MoriIOServiceRegistry, TransferMode};
use super::pd_types::error_chain;
use crate::core::{BasicWorker, Worker, WorkerType};
use crate::metrics::RouterMetrics;
use crate::otel_http::{self, ClientRequestOptions};
use crate::policies::PolicyRegistry;
use crate::routers::{RouterTrait, WorkerManagement};
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Parsed host and port extracted from a MoRIIO `request_address` URL
fn extract_host_port(request_address: &str) -> Result<(String, u16), String> {
    let parsed = url::Url::parse(request_address)
        .map_err(|e| format!("Invalid request_address '{}': {}", request_address, e))?;

    let host = parsed
        .host_str()
        .ok_or_else(|| format!("No host in request_address '{}'", request_address))?
        .to_string();

    let port = parsed
        .port()
        .ok_or_else(|| format!("No port in request_address '{}'", request_address))?;

    Ok((host, port))
}

/// Convert a list of MoRIIO instances to Worker objects for policy selection
fn instances_to_workers(instances: &[MoriIOInstance]) -> Vec<Arc<dyn Worker>> {
    instances
        .iter()
        .map(|inst| {
            // Strip path from request_address to get base URL for the worker
            let base_url = match url::Url::parse(&inst.request_address) {
                Ok(parsed) => {
                    let scheme = parsed.scheme();
                    let host = parsed.host_str().unwrap_or("localhost");
                    match parsed.port() {
                        Some(p) => format!("{}://{}:{}", scheme, host, p),
                        None => format!("{}://{}", scheme, host),
                    }
                }
                Err(_) => inst.request_address.clone(),
            };
            Arc::new(BasicWorker::new(base_url, WorkerType::Regular)) as Arc<dyn Worker>
        })
        .collect()
}

/// Select a worker index using the prefill or decode policy
fn select_worker(
    instances: &[MoriIOInstance],
    policy_registry: &Arc<PolicyRegistry>,
    is_prefill: bool,
    request_text: Option<&str>,
) -> Option<usize> {
    if instances.is_empty() {
        return None;
    }
    let workers = instances_to_workers(instances);
    let policy = if is_prefill {
        policy_registry.get_prefill_policy()
    } else {
        policy_registry.get_decode_policy()
    };
    policy.select_worker(&workers, request_text)
}

/// MoRIIO PD Router — routes requests through the MoRIIO connector in READ or WRITE mode
#[derive(Debug)]
pub struct MoriIOPDRouter {
    /// Service registry for ZMQ-based instance discovery
    service_registry: Arc<MoriIOServiceRegistry>,
    /// HTTP client for forwarding requests
    http_client: reqwest::Client,
    /// Policy registry for load balancing
    policy_registry: Arc<PolicyRegistry>,
}

impl MoriIOPDRouter {
    /// Create a new MoRIIO PD router in service discovery mode
    pub async fn new_discovery(
        discovery_address: &str,
        ctx: &Arc<crate::server::AppContext>,
    ) -> Result<Self, String> {
        info!(
            "MoriIOPDRouter: starting service discovery on {}",
            discovery_address
        );
        let mut registry = MoriIOServiceRegistry::new();
        registry
            .start_listener(discovery_address)
            .await
            .map_err(|e| format!("Failed to start MoRIIO service discovery: {}", e))?;

        info!("MoriIOPDRouter created in discovery mode");
        Ok(Self {
            service_registry: Arc::new(registry),
            http_client: reqwest::Client::new(),
            policy_registry: ctx.policy_registry.clone(),
        })
    }

    /// Prepare the prefill-stage request: cap max_tokens to 1 and force stream=false
    fn prepare_prefill_request(mut req: Value) -> Value {
        req["max_tokens"] = json!(1);
        if let Some(mt) = req.get("max_completion_tokens") {
            if mt.as_u64().map_or(true, |v| v > 1) {
                req["max_completion_tokens"] = json!(1);
            }
        }
        if let Some(min) = req.get("min_tokens").and_then(|v| v.as_u64()) {
            if min > 1 {
                req["min_tokens"] = json!(1);
            }
        }
        req["stream"] = json!(false);
        if let Some(obj) = req.as_object_mut() {
            obj.remove("stream_options");
        }
        req
    }

    /// Build kv_transfer_params for the prefill request
    fn build_prefill_kv_params(transfer_id: &str, decode: &MoriIOInstance) -> Value {
        let (decode_host, decode_port) = match extract_host_port(&decode.request_address) {
            Ok(hp) => hp,
            Err(e) => {
                warn!("Could not extract host/port from decode address: {}", e);
                ("".to_string(), 0)
            }
        };

        json!({
            "transfer_id": transfer_id,
            "remote_dp_size": decode.dp_size,
            "remote_tp_size": decode.tp_size,
            "do_remote_decode": true,
            "do_remote_prefill": false,
            "remote_handshake_port": decode.handshake_port,
            "remote_notify_port": decode.notify_port,
            "remote_engine_id": null,
            "remote_block_ids": null,
            "remote_host": decode_host,
            "remote_port": decode_port,
            "max_tokens": 1,
            "stream": false
        })
    }

    /// Build kv_transfer_params for the decode request
    fn build_decode_kv_params(
        transfer_id: &str,
        prefill: &MoriIOInstance,
        prefill_idx: usize,
        remote_engine_id: Option<&Value>,
        remote_block_ids: Option<&Value>,
    ) -> Value {
        let (prefill_host, prefill_port) = match extract_host_port(&prefill.request_address) {
            Ok(hp) => hp,
            Err(e) => {
                warn!("Could not extract host/port from prefill address: {}", e);
                ("".to_string(), 0)
            }
        };

        let mut params = json!({
            "transfer_id": transfer_id,
            "remote_dp_size": prefill.dp_size,
            "remote_tp_size": prefill.tp_size,
            "do_remote_decode": false,
            "do_remote_prefill": true,
            "remote_handshake_port": prefill.handshake_port,
            "remote_notify_port": prefill.notify_port,
            "remote_engine_id": remote_engine_id.cloned().unwrap_or(Value::Null),
            "remote_block_ids": remote_block_ids.cloned().unwrap_or(Value::Null),
            "remote_host": prefill_host,
            "remote_port": prefill_port,
        });

        // Add remote_dp_rank only when prefill has > 1 DP ranks
        if prefill.dp_size > 1 {
            params["remote_dp_rank"] = json!(prefill_idx % prefill.dp_size as usize);
        }

        params
    }

    /// Core two-stage routing: handles both READ and WRITE transfer modes
    async fn process_moriio_request(
        &self,
        request_json: Value,
        path: &str,
        headers: Option<&HeaderMap>,
    ) -> Response {
        // Fetch live instances from service registry
        let prefill_instances = self.service_registry.get_prefill_instances();
        let decode_instances = self.service_registry.get_decode_instances();

        if prefill_instances.is_empty() || decode_instances.is_empty() {
            RouterMetrics::record_pd_error("server_selection");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "No MoRIIO workers available: {} prefill, {} decode",
                    prefill_instances.len(),
                    decode_instances.len()
                ),
            )
                .into_response();
        }

        // Detect transfer mode from service registry
        let transfer_mode = match self.service_registry.transfer_mode() {
            Some(m) => m,
            None => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "MoRIIO transfer_mode not yet determined (no instance registered)",
                )
                    .into_response();
            }
        };

        let request_text = serde_json::to_string(&request_json).ok();
        let request_str = request_text.as_deref();

        let prefill_idx =
            match select_worker(&prefill_instances, &self.policy_registry, true, request_str) {
                Some(idx) => idx,
                None => {
                    RouterMetrics::record_pd_error("server_selection");
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Prefill policy failed to select a worker".to_string(),
                    )
                        .into_response();
                }
            };

        let decode_idx =
            match select_worker(&decode_instances, &self.policy_registry, false, request_str) {
                Some(idx) => idx,
                None => {
                    RouterMetrics::record_pd_error("server_selection");
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Decode policy failed to select a worker".to_string(),
                    )
                        .into_response();
                }
            };

        let prefill = &prefill_instances[prefill_idx];
        let decode = &decode_instances[decode_idx];
        let transfer_id = format!("tx-{}", Uuid::new_v4());

        info!(
            "MoRIIO routing [mode={}]: prefill={}, decode={}, transfer_id={}",
            transfer_mode, prefill.request_address, decode.request_address, transfer_id
        );

        match transfer_mode {
            TransferMode::Read => {
                self.process_read_mode(
                    request_json,
                    prefill,
                    decode,
                    prefill_idx,
                    &transfer_id,
                    path,
                    headers,
                )
                .await
            }
            TransferMode::Write => {
                self.process_write_mode(
                    request_json,
                    prefill,
                    decode,
                    prefill_idx,
                    &transfer_id,
                    path,
                    headers,
                )
                .await
            }
        }
    }

    /// READ mode: send prefill first, extract block metadata, then send decode
    async fn process_read_mode(
        &self,
        request_json: Value,
        prefill: &MoriIOInstance,
        decode: &MoriIOInstance,
        prefill_idx: usize,
        transfer_id: &str,
        path: &str,
        headers: Option<&HeaderMap>,
    ) -> Response {
        let start = Instant::now();

        // --- Stage 1: Prefill ---
        let mut prefill_req = Self::prepare_prefill_request(request_json.clone());
        prefill_req["kv_transfer_params"] =
            Self::build_prefill_kv_params(transfer_id, decode);

        let prefill_body = match serde_json::to_string(&prefill_req) {
            Ok(s) => s,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to serialize prefill request: {}", e),
                )
                    .into_response()
            }
        };

        // Construct proper prefill URL using base (scheme+host+port) + path
        let prefill_base_url = match url::Url::parse(&prefill.request_address) {
            Ok(u) => {
                let host = u.host_str().unwrap_or("localhost");
                let port = u.port().unwrap_or(8000);
                format!("{}://{}:{}", u.scheme(), host, port)
            }
            Err(_) => prefill.request_address.clone(),
        };
        let prefill_request_url = format!("{}{}", prefill_base_url, path);

        debug!("MoRIIO READ Stage 1 — prefill POST {}", prefill_request_url);

        let prefill_resp = match otel_http::send_client_request(
            self.http_client
                .post(&prefill_request_url)
                .header("Content-Type", "application/json")
                .body(prefill_body),
            headers,
            ClientRequestOptions {
                method: "POST",
                url: &prefill_request_url,
                route: Some(path),
                request_phase: Some("prefill"),
            },
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                let duration = start.elapsed();
                RouterMetrics::record_pd_prefill_error(&prefill.request_address);
                RouterMetrics::record_pd_request(path);
                RouterMetrics::record_pd_request_duration(path, duration);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(
                        "Prefill request to {} failed: {}",
                        prefill_request_url,
                        error_chain(&e)
                    ),
                )
                    .into_response();
            }
        };

        let prefill_status = prefill_resp.status();
        if !prefill_status.is_success() {
            let duration = start.elapsed();
            RouterMetrics::record_pd_prefill_error(&prefill.request_address);
            RouterMetrics::record_pd_request(path);
            RouterMetrics::record_pd_request_duration(path, duration);
            let body_text = prefill_resp.text().await.unwrap_or_default();
            return (
                prefill_status,
                format!("Prefill error {}: {}", prefill_status, body_text),
            )
                .into_response();
        }

        // Parse prefill response to extract remote_engine_id and remote_block_ids
        let prefill_text = match prefill_resp.text().await {
            Ok(t) => t,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to read prefill response: {}", error_chain(&e)),
                )
                    .into_response()
            }
        };

        let prefill_json: Value = match serde_json::from_str(&prefill_text) {
            Ok(v) => v,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to parse prefill response JSON: {}", e),
                )
                    .into_response()
            }
        };

        let remote_engine_id = prefill_json
            .get("kv_transfer_params")
            .and_then(|kv| kv.get("remote_engine_id"));
        let remote_block_ids = prefill_json
            .get("kv_transfer_params")
            .and_then(|kv| kv.get("remote_block_ids"));

        debug!(
            "MoRIIO READ: extracted remote_engine_id={:?}, remote_block_ids present={}",
            remote_engine_id,
            remote_block_ids.is_some()
        );

        // --- Stage 2: Decode ---
        let mut decode_req = request_json.clone();
        // Decrement max_tokens by 1 to account for the prefill token
        if let Some(mt) = decode_req.get("max_tokens").and_then(|v| v.as_u64()) {
            if mt > 0 {
                decode_req["max_tokens"] = json!(mt - 1);
            }
        }
        decode_req["kv_transfer_params"] = Self::build_decode_kv_params(
            transfer_id,
            prefill,
            prefill_idx,
            remote_engine_id,
            remote_block_ids,
        );

        self.send_decode_request(
            decode_req,
            decode,
            path,
            headers,
            &start,
            &prefill.request_address,
            &decode.request_address,
        )
        .await
    }

    /// WRITE mode: fire prefill in background, then send decode concurrently
    async fn process_write_mode(
        &self,
        request_json: Value,
        prefill: &MoriIOInstance,
        decode: &MoriIOInstance,
        prefill_idx: usize,
        transfer_id: &str,
        path: &str,
        headers: Option<&HeaderMap>,
    ) -> Response {
        let start = Instant::now();

        // Build prefill request
        let mut prefill_req = Self::prepare_prefill_request(request_json.clone());
        prefill_req["kv_transfer_params"] =
            Self::build_prefill_kv_params(transfer_id, decode);

        let prefill_body = match serde_json::to_string(&prefill_req) {
            Ok(s) => s,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to serialize prefill request: {}", e),
                )
                    .into_response()
            }
        };

        let prefill_base_url = match url::Url::parse(&prefill.request_address) {
            Ok(u) => {
                let host = u.host_str().unwrap_or("localhost");
                let port = u.port().unwrap_or(8000);
                format!("{}://{}:{}", u.scheme(), host, port)
            }
            Err(_) => prefill.request_address.clone(),
        };
        let prefill_request_url = format!("{}{}", prefill_base_url, path);

        // Clone what we need for the background task
        let http_client_clone = self.http_client.clone();
        let prefill_addr_clone = prefill.request_address.clone();
        let prefill_url_clone = prefill_request_url.clone();

        // Fire prefill in background — decode handles synchronisation via ZMQ internally
        tokio::spawn(async move {
            debug!(
                "MoRIIO WRITE: firing prefill in background at {}",
                prefill_url_clone
            );
            match http_client_clone
                .post(&prefill_url_clone)
                .header("Content-Type", "application/json")
                .body(prefill_body)
                .send()
                .await
            {
                Ok(resp) => {
                    debug!(
                        "MoRIIO WRITE: prefill background response status={}",
                        resp.status()
                    );
                    if !resp.status().is_success() {
                        RouterMetrics::record_pd_prefill_error(&prefill_addr_clone);
                        warn!(
                            "MoRIIO WRITE: background prefill to {} returned {}",
                            prefill_url_clone,
                            resp.status()
                        );
                    }
                }
                Err(e) => {
                    RouterMetrics::record_pd_prefill_error(&prefill_addr_clone);
                    error!(
                        "MoRIIO WRITE: background prefill to {} failed: {}",
                        prefill_url_clone,
                        error_chain(&e)
                    );
                }
            }
        });

        // Build decode request immediately — block IDs stay null; decode waits for ZMQ notification
        let mut decode_req = request_json.clone();
        // Decrement max_tokens by 1 to account for the prefill token
        if let Some(mt) = decode_req.get("max_tokens").and_then(|v| v.as_u64()) {
            if mt > 0 {
                decode_req["max_tokens"] = json!(mt - 1);
            }
        }
        decode_req["kv_transfer_params"] =
            Self::build_decode_kv_params(transfer_id, prefill, prefill_idx, None, None);

        self.send_decode_request(
            decode_req,
            decode,
            path,
            headers,
            &start,
            &prefill.request_address,
            &decode.request_address,
        )
        .await
    }

    /// Send the decode request and forward the response to the caller
    async fn send_decode_request(
        &self,
        decode_req: Value,
        decode: &MoriIOInstance,
        path: &str,
        headers: Option<&HeaderMap>,
        start: &Instant,
        prefill_addr: &str,
        decode_addr: &str,
    ) -> Response {
        let decode_base_url = match url::Url::parse(&decode.request_address) {
            Ok(u) => {
                let host = u.host_str().unwrap_or("localhost");
                let port = u.port().unwrap_or(8000);
                format!("{}://{}:{}", u.scheme(), host, port)
            }
            Err(_) => decode.request_address.clone(),
        };
        let decode_request_url = format!("{}{}", decode_base_url, path);

        let decode_body = match serde_json::to_string(&decode_req) {
            Ok(s) => s,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to serialize decode request: {}", e),
                )
                    .into_response()
            }
        };

        debug!("MoRIIO decode POST {}", decode_request_url);

        let decode_resp = match otel_http::send_client_request(
            self.http_client
                .post(&decode_request_url)
                .header("Content-Type", "application/json")
                .body(decode_body),
            headers,
            ClientRequestOptions {
                method: "POST",
                url: &decode_request_url,
                route: Some(path),
                request_phase: Some("decode"),
            },
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                let duration = start.elapsed();
                RouterMetrics::record_pd_decode_error(decode_addr);
                RouterMetrics::record_pd_request(path);
                RouterMetrics::record_pd_request_duration(path, duration);
                RouterMetrics::record_pd_prefill_request(prefill_addr);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(
                        "Decode request to {} failed: {}",
                        decode_request_url,
                        error_chain(&e)
                    ),
                )
                    .into_response();
            }
        };

        let duration = start.elapsed();
        let status = decode_resp.status();
        let resp_headers = decode_resp.headers().clone();

        RouterMetrics::record_pd_request(path);
        RouterMetrics::record_pd_request_duration(path, duration);
        RouterMetrics::record_pd_prefill_request(prefill_addr);
        RouterMetrics::record_pd_decode_request(decode_addr);
        if !status.is_success() {
            RouterMetrics::record_pd_decode_error(decode_addr);
        }

        // Stream decode response back to caller
        let mut builder = axum::http::Response::builder().status(status);
        for (name, value) in resp_headers.iter() {
            if name != "transfer-encoding" && name != "content-length" {
                builder = builder.header(name, value);
            }
        }
        builder
            .body(Body::from_stream(decode_resp.bytes_stream()))
            .unwrap_or_else(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to build response: {}", e),
                )
                    .into_response()
            })
    }
}

// ── RouterTrait implementation ───────────────────────────────────────────────

#[async_trait]
impl RouterTrait for MoriIOPDRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health(&self, _req: Request<Body>) -> Response {
        (StatusCode::OK, "OK").into_response()
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        (StatusCode::OK, "OK").into_response()
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "not implemented").into_response()
    }

    async fn get_models(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "not implemented").into_response()
    }

    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "not implemented").into_response()
    }

    async fn route_generate(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "MoRIIO router does not support the generate API",
        )
            .into_response()
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &crate::protocols::spec::ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        let request_json = match serde_json::to_value(body) {
            Ok(v) => v,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Serialization error: {}", e),
                )
                    .into_response()
            }
        };
        self.process_moriio_request(request_json, "/v1/chat/completions", headers)
            .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &crate::protocols::spec::CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        let request_json = match serde_json::to_value(body) {
            Ok(v) => v,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Serialization error: {}", e),
                )
                    .into_response()
            }
        };
        self.process_moriio_request(request_json, "/v1/completions", headers)
            .await
    }

    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::ResponsesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "MoRIIO router does not support the responses API",
        )
            .into_response()
    }

    async fn get_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "not implemented").into_response()
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "not implemented").into_response()
    }

    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "MoRIIO router does not support the embeddings API",
        )
            .into_response()
    }

    async fn route_rerank(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::RerankRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "MoRIIO router does not support the rerank API",
        )
            .into_response()
    }

    async fn flush_cache(&self) -> Response {
        (StatusCode::OK, "OK").into_response()
    }

    async fn get_worker_loads(&self) -> Response {
        let (prefill_count, decode_count) = self.service_registry.get_instance_counts();
        let body = serde_json::json!({
            "prefill_instances": prefill_count,
            "decode_instances": decode_count,
        });
        (StatusCode::OK, body.to_string()).into_response()
    }

    fn router_type(&self) -> &'static str {
        "moriio_pd"
    }

    fn readiness(&self) -> Response {
        let (p, d) = self.service_registry.get_instance_counts();
        if p == 0 || d == 0 {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "Waiting for MoRIIO workers to register",
            )
                .into_response();
        }
        (StatusCode::OK, "OK").into_response()
    }

    async fn route_transparent(
        &self,
        headers: Option<&HeaderMap>,
        path: &str,
        method: &Method,
        body: Value,
    ) -> Response {
        if *method != Method::POST {
            return (
                StatusCode::METHOD_NOT_ALLOWED,
                "Only POST requests are supported",
            )
                .into_response();
        }
        self.process_moriio_request(body, path, headers).await
    }
}

// ── WorkerManagement (stub — MoRIIO workers are managed via ZMQ discovery) ──

#[async_trait]
impl WorkerManagement for MoriIOPDRouter {
    async fn add_worker(&self, _worker_url: &str) -> Result<String, String> {
        Err("MoRIIO workers register via ZMQ service discovery".to_string())
    }

    fn remove_worker(&self, _worker_url: &str) {}

    fn get_worker_urls(&self) -> Vec<String> {
        let prefill: Vec<_> = self
            .service_registry
            .get_prefill_instances()
            .into_iter()
            .map(|i| i.request_address)
            .collect();
        let decode: Vec<_> = self
            .service_registry
            .get_decode_instances()
            .into_iter()
            .map(|i| i.request_address)
            .collect();
        [prefill, decode].concat()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_host_port_valid() {
        let (host, port) =
            extract_host_port("http://192.168.1.10:8000/v1/completions").unwrap();
        assert_eq!(host, "192.168.1.10");
        assert_eq!(port, 8000);
    }

    #[test]
    fn test_extract_host_port_no_port() {
        assert!(extract_host_port("http://myhost/v1/completions").is_err());
    }

    #[test]
    fn test_extract_host_port_invalid_url() {
        assert!(extract_host_port("not-a-url").is_err());
    }

    #[test]
    fn test_prepare_prefill_request_caps_max_tokens() {
        let req = json!({
            "max_tokens": 512,
            "stream": true,
            "stream_options": {"include_usage": true}
        });
        let result = MoriIOPDRouter::prepare_prefill_request(req);
        assert_eq!(result["max_tokens"], 1);
        assert_eq!(result["stream"], false);
        assert!(result.get("stream_options").is_none());
    }

    #[test]
    fn test_prepare_prefill_request_clamps_min_tokens() {
        let req = json!({"max_tokens": 512, "min_tokens": 100});
        let result = MoriIOPDRouter::prepare_prefill_request(req);
        assert_eq!(result["min_tokens"], 1);
    }

    fn make_instance(addr: &str) -> MoriIOInstance {
        MoriIOInstance {
            request_address: addr.to_string(),
            handshake_port: 8001,
            notify_port: 8002,
            dp_size: 1,
            tp_size: 1,
            expires_at: u64::MAX,
        }
    }

    #[test]
    fn test_build_prefill_kv_params() {
        let decode = make_instance("http://10.0.0.2:9000/v1/completions");
        let params = MoriIOPDRouter::build_prefill_kv_params("tx-abc", &decode);
        assert_eq!(params["transfer_id"], "tx-abc");
        assert_eq!(params["do_remote_decode"], true);
        assert_eq!(params["do_remote_prefill"], false);
        assert_eq!(params["remote_handshake_port"], 8001);
        assert_eq!(params["remote_notify_port"], 8002);
        assert_eq!(params["remote_host"], "10.0.0.2");
        assert_eq!(params["remote_port"], 9000);
        assert_eq!(params["remote_engine_id"], Value::Null);
        assert_eq!(params["remote_block_ids"], Value::Null);
    }

    #[test]
    fn test_build_decode_kv_params_read_mode_with_engine_id() {
        let prefill = make_instance("http://10.0.0.1:8000/v1/completions");
        let engine_id = json!("engine-xyz");
        let block_ids = json!([1, 2, 3]);
        let params = MoriIOPDRouter::build_decode_kv_params(
            "tx-abc",
            &prefill,
            0,
            Some(&engine_id),
            Some(&block_ids),
        );
        assert_eq!(params["transfer_id"], "tx-abc");
        assert_eq!(params["do_remote_prefill"], true);
        assert_eq!(params["do_remote_decode"], false);
        assert_eq!(params["remote_engine_id"], engine_id);
        assert_eq!(params["remote_block_ids"], block_ids);
        assert_eq!(params["remote_host"], "10.0.0.1");
        assert_eq!(params["remote_port"], 8000);
        // dp_size == 1, so remote_dp_rank should not be present
        assert!(params.get("remote_dp_rank").is_none());
    }

    #[test]
    fn test_build_decode_kv_params_dp_rank_included_when_dp_gt_1() {
        let mut prefill =
            make_instance("http://10.0.0.1:8000/v1/completions");
        prefill.dp_size = 4;
        let params =
            MoriIOPDRouter::build_decode_kv_params("tx-abc", &prefill, 2, None, None);
        assert!(params.get("remote_dp_rank").is_some());
        // rank = prefill_idx % dp_size = 2 % 4 = 2
        assert_eq!(params["remote_dp_rank"], 2);
    }

    #[test]
    fn test_build_decode_kv_params_write_mode_null_blocks() {
        let prefill = make_instance("http://10.0.0.1:8000/v1/completions");
        let params =
            MoriIOPDRouter::build_decode_kv_params("tx-write", &prefill, 0, None, None);
        assert_eq!(params["remote_engine_id"], Value::Null);
        assert_eq!(params["remote_block_ids"], Value::Null);
    }
}
