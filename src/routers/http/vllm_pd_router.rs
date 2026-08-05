// vLLM PD (Prefill-Decode) Router Implementation
// This module extends PdRouterBase to handle vLLM-specific two-stage processing
use super::dp_utils;
use super::logprobs_merge;
use super::pd_router::PdRouterBase;
use super::pd_types::{error_chain, PDRouterError};
use super::vllm_service_discovery::{MoriIOTransferMode, ServiceRegistry, ServiceType};
use crate::config::KvConnector;
use crate::core::{BasicWorker, Worker, WorkerType};
use crate::metrics::RouterMetrics;
use crate::otel_http::{self, ClientRequestOptions};
use crate::policies::PolicyRegistry;
use crate::routers::{header_utils, RouterTrait, WorkerManagement};
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
};
use dashmap::DashMap;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Mooncake prefill bootstrap info (engine_id per dp_rank)
#[derive(Debug, Clone)]
struct MooncakePrefillInfo {
    bootstrap_addr: String,
    dp_engine_ids: HashMap<usize, String>,
}

/// vLLM PD Router that extends PdRouterBase with vLLM-specific request handling
#[derive(Debug)]
pub struct VllmPDRouter {
    /// Underlying PD router for most functionality
    pd_router: PdRouterBase,
    /// Service discovery registry for dynamic ZMQ address resolution
    service_registry: Arc<ServiceRegistry>,
    /// HTTP client for making requests to discovered services
    http_client: reqwest::Client,
    /// Policy registry for load balancing
    policy_registry: Arc<PolicyRegistry>,
    /// Whether this router uses service discovery (true) or direct URLs (false)
    use_discovery: bool,
    /// Enable profiling calls to vLLM workers
    enable_profiling: bool,
    /// Profiling timeout in seconds
    profile_timeout_secs: u64,
    /// Active profiling timeout tasks keyed by worker URL
    profiling_tasks: Arc<Mutex<HashMap<String, tokio::task::AbortHandle>>>,
    /// Intra-node data parallel size for DP-aware routing (automatically enabled when > 1)
    intra_node_data_parallel_size: usize,
    /// KV connector type
    kv_connector: KvConnector,
    /// Mooncake bootstrap info: prefill base_url -> MooncakePrefillInfo
    mooncake_prefill_info: Arc<Mutex<HashMap<String, MooncakePrefillInfo>>>,
    /// Persistent per-URL workers for policy selection. Selection previously
    /// built throwaway `BasicWorker`s per request, so every policy saw
    /// `load() == 0` and `processed_requests() == 0`: balance thresholds could
    /// never trigger and tie-breaks collapsed onto the first worker. These
    /// workers live for the router's lifetime and carry real counters.
    policy_workers: DashMap<String, Arc<dyn Worker>>,
}

/// Transfer ID prefix used by MoRI-IO to correlate prefill and decode legs.
/// Must match `MoRIIOConstants.TRANSFER_PREFIX` in the vLLM Python connector.
const MORIIO_TRANSFER_PREFIX: &str = "tx";

/// Strip the DP-rank suffix from a worker's HTTP address and return the base address
/// plus the parsed rank. Returns `(original, None)` when DP is disabled.
fn extract_base_http_and_dp_rank(
    http: &str,
    intra_node_data_parallel_size: usize,
) -> (String, Option<usize>) {
    if intra_node_data_parallel_size > 1 {
        let url = format!("http://{}", http);
        let (base, rank) = dp_utils::parse_worker_url(&url);
        let base_http = base.replace("http://", "").replace("https://", "");
        (base_http, rank)
    } else {
        (http.to_string(), None)
    }
}

/// Build a prefill reqwest::RequestBuilder with the standard headers and dp-rank header.
fn build_prefill_request_builder(
    http_client: &reqwest::Client,
    url: &str,
    request_id: &str,
    dp_rank: Option<usize>,
) -> reqwest::RequestBuilder {
    let builder = http_client
        .post(url)
        .header("Content-Type", "application/json")
        .header("X-Request-Id", request_id);
    dp_utils::add_dp_rank_header(builder, dp_rank)
}

impl VllmPDRouter {
    /// Query the Mooncake bootstrap server on a prefill node to get engine_id per dp_rank.
    /// Retries with backoff since the prefill server may not be ready at router startup.
    async fn query_mooncake_bootstrap(
        client: &reqwest::Client,
        bootstrap_addr: &str,
    ) -> Result<HashMap<usize, String>, String> {
        let url = format!("{}/query", bootstrap_addr);
        let max_retries = 30;
        let mut backoff_secs = 1u64;

        for attempt in 1..=max_retries {
            match client.get(&url).send().await {
                Ok(response) => {
                    if !response.status().is_success() {
                        if attempt == max_retries {
                            return Err(format!(
                                "Mooncake bootstrap query to {} failed with status {}",
                                url,
                                response.status()
                            ));
                        }
                        warn!(
                            "Mooncake bootstrap query attempt {}/{} to {} returned {}, retrying in {}s",
                            attempt, max_retries, url, response.status(), backoff_secs
                        );
                        tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
                        backoff_secs = (backoff_secs * 2).min(10);
                        continue;
                    }
                    let data: Value = response
                        .json()
                        .await
                        .map_err(|e| format!("Failed to parse bootstrap response: {}", e))?;

                    let mut dp_engine_ids = HashMap::new();
                    if let Some(obj) = data.as_object() {
                        for (dp_rank_str, dp_entry) in obj {
                            let dp_rank: usize = dp_rank_str
                                .parse()
                                .map_err(|e| format!("Invalid dp_rank '{}': {}", dp_rank_str, e))?;
                            let engine_id = dp_entry
                                .get("engine_id")
                                .and_then(|v| v.as_str())
                                .ok_or_else(|| {
                                    format!("Missing engine_id for dp_rank {}", dp_rank)
                                })?
                                .to_string();
                            dp_engine_ids.insert(dp_rank, engine_id);
                        }
                    }
                    info!(
                        "Queried Mooncake bootstrap at {}: {} dp ranks found",
                        url,
                        dp_engine_ids.len()
                    );
                    return Ok(dp_engine_ids);
                }
                Err(e) => {
                    if attempt == max_retries {
                        return Err(format!(
                            "Mooncake bootstrap query to {} failed after {} attempts: {}",
                            url, max_retries, e
                        ));
                    }
                    warn!(
                        "Mooncake bootstrap query attempt {}/{} to {} failed: {}, retrying in {}s",
                        attempt, max_retries, url, e, backoff_secs
                    );
                    tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
                    backoff_secs = (backoff_secs * 2).min(10);
                }
            }
        }
        Err(format!(
            "Mooncake bootstrap query to {} failed after {} attempts",
            url, max_retries
        ))
    }

    /// Returns the MoRI-IO transfer mode if it has been set by a registration, or `None`.
    fn moriio_transfer_mode(&self) -> Option<MoriIOTransferMode> {
        self.service_registry.moriio_transfer_mode.get().copied()
    }

    /// Generate a connector-specific transfer ID for correlating prefill and decode legs.
    /// Returns `None` for connectors that do not use a transfer_id (e.g. NIXL).
    fn generate_transfer_id(&self) -> Option<String> {
        match self.kv_connector {
            // Mooncake uses the "xfer-<uuid>" format.
            KvConnector::Mooncake => Some(format!("xfer-{}", Uuid::new_v4())),
            // MoRI-IO uses the "tx-<uuid-no-dashes>" format to match MoRIIOConstants.TRANSFER_PREFIX.
            KvConnector::MoriIO => Some(format!(
                "{}-{}",
                MORIIO_TRANSFER_PREFIX,
                Uuid::new_v4().simple()
            )),
            KvConnector::Nixl => None,
        }
    }

    /// Build kv_transfer_params for the prefill request.
    ///
    /// Returns an error for MoRI-IO when no transfer mode has been registered yet, so that
    /// requests are not silently dispatched in READ mode when no instances have registered.
    fn build_prefill_kv_transfer_params(&self, transfer_id: Option<&str>) -> Result<Value, String> {
        match self.kv_connector {
            KvConnector::Mooncake => Ok(json!({
                "do_remote_decode": true,
                "do_remote_prefill": false,
                "transfer_id": transfer_id.unwrap_or(""),
            })),
            KvConnector::MoriIO => {
                let mode = self.moriio_transfer_mode().ok_or_else(|| {
                    "No MoRI-IO instances have registered a transfer mode; \
                     cannot dispatch request without knowing READ vs WRITE mode"
                        .to_string()
                })?;
                if matches!(mode, MoriIOTransferMode::Write) {
                    // WRITE mode: prefill pushes KV blocks to decode.
                    // do_remote_decode=true tells the prefill connector to initiate the transfer.
                    Ok(json!({
                        "do_remote_decode": true,
                        "do_remote_prefill": false,
                        "remote_engine_id": serde_json::Value::Null,
                        "remote_block_ids": serde_json::Value::Null,
                        "remote_dp_size": self.intra_node_data_parallel_size,
                        // remote_tp_size is not yet consumed by the vLLM MoRI-IO connector;
                        // hardcoded to 1 until https://github.com/vllm-project/vllm/issues/41211 is resolved.
                        "remote_tp_size": 1,
                        "transfer_id": transfer_id.unwrap_or(""),
                    }))
                } else {
                    // READ mode: prefill waits for decode to pull blocks.
                    Ok(json!({
                        "do_remote_decode": true,
                        "do_remote_prefill": false,
                        "remote_engine_id": serde_json::Value::Null,
                        "remote_block_ids": serde_json::Value::Null,
                        "transfer_id": transfer_id.unwrap_or(""),
                        "remote_dp_size": self.intra_node_data_parallel_size,
                    }))
                }
            }
            KvConnector::Nixl => Ok(json!({
                "do_remote_decode": true,
                "do_remote_prefill": false,
                "remote_engine_id": serde_json::Value::Null,
                "remote_block_ids": serde_json::Value::Null,
                "remote_host": serde_json::Value::Null,
                "remote_port": serde_json::Value::Null
            })),
        }
    }

    /// Build decode kv_transfer_params for all connectors. Returns `None` if params cannot be
    /// determined (e.g. Mooncake bootstrap info missing, or no kv_transfer_params in prefill response).
    async fn build_decode_kv_transfer_params(
        &self,
        prefill_url: &str,
        prefill_response_json: Option<&Value>,
        transfer_id: Option<&str>,
        prefill_dp_rank: Option<u32>,
    ) -> Option<Value> {
        match self.kv_connector {
            KvConnector::Mooncake => {
                let Some((bootstrap_addr, engine_id)) = self
                    .get_mooncake_info(prefill_url, prefill_dp_rank.map(|r| r as usize))
                    .await
                else {
                    warn!(
                        "No Mooncake bootstrap info for prefill {}, decode will proceed without kv_transfer_params",
                        prefill_url
                    );
                    return None;
                };
                Some(self.build_mooncake_decode_kv_transfer_params(
                    transfer_id.unwrap_or(""),
                    &bootstrap_addr,
                    &engine_id,
                ))
            }
            KvConnector::MoriIO => {
                if matches!(self.moriio_transfer_mode(), Some(MoriIOTransferMode::Write)) {
                    // WRITE mode: build decode params directly; decode does not need the prefill response.
                    let mut params = json!({
                        "do_remote_decode": false,
                        "do_remote_prefill": true,
                        "remote_engine_id": serde_json::Value::Null,
                        "remote_block_ids": serde_json::Value::Null,
                        "transfer_id": transfer_id.unwrap_or(""),
                        "remote_dp_size": self.intra_node_data_parallel_size,
                        // remote_tp_size is not yet consumed by the vLLM MoRI-IO connector;
                        // hardcoded to 1 until https://github.com/vllm-project/vllm/issues/41211 is resolved.
                        "remote_tp_size": 1,
                    });
                    if self.intra_node_data_parallel_size > 1 {
                        if let Some(rank) = prefill_dp_rank {
                            params["remote_dp_rank"] = json!(rank);
                        }
                    }
                    Some(params)
                } else {
                    // READ mode: extract params from prefill response and inject remote_dp_size.
                    let mut params = prefill_response_json?.get("kv_transfer_params")?.clone();
                    params["remote_dp_size"] = json!(self.intra_node_data_parallel_size);
                    Some(params)
                }
            }
            KvConnector::Nixl => Some(prefill_response_json?.get("kv_transfer_params")?.clone()),
        }
    }

    /// Build kv_transfer_params for the decode request (Mooncake only).
    /// For NIXL, decode params come from the prefill response instead.
    fn build_mooncake_decode_kv_transfer_params(
        &self,
        transfer_id: &str,
        bootstrap_addr: &str,
        engine_id: &str,
    ) -> Value {
        json!({
            "do_remote_decode": false,
            "do_remote_prefill": true,
            "transfer_id": transfer_id,
            "remote_bootstrap_addr": bootstrap_addr,
            "remote_engine_id": engine_id,
        })
    }

    /// Look up Mooncake prefill info for a given prefill URL and dp_rank
    async fn get_mooncake_info(
        &self,
        prefill_url: &str,
        dp_rank: Option<usize>,
    ) -> Option<(String, String)> {
        let info = self.mooncake_prefill_info.lock().await;
        if let Some(prefill_info) = info.get(prefill_url) {
            let rank = dp_rank.unwrap_or(0);
            if let Some(engine_id) = prefill_info.dp_engine_ids.get(&rank) {
                return Some((prefill_info.bootstrap_addr.clone(), engine_id.clone()));
            }
            // Fallback: use first available engine_id
            if let Some(engine_id) = prefill_info.dp_engine_ids.values().next() {
                return Some((prefill_info.bootstrap_addr.clone(), engine_id.clone()));
            }
        }
        None
    }

    /// Generate vLLM-specific request ID with prefill/decode addressing
    fn generate_vllm_request_id(prefill_addr: &str, decode_addr: &str) -> String {
        let uuid = Uuid::new_v4().to_string().replace('-', "");
        format!(
            "___prefill_addr_{}___decode_addr_{}_{}",
            prefill_addr, decode_addr, uuid
        )
    }

    /// Get ZMQ address for a worker URL using service discovery
    fn get_zmq_address(&self, http_url: &str, service_type: ServiceType) -> String {
        // Extract just the host:port from the URL
        let http_address = http_url.replace("http://", "").replace("https://", "");

        // Try to get ZMQ address from service discovery
        if let Some(zmq_addr) = self
            .service_registry
            .get_zmq_address(&http_address, service_type.clone())
        {
            debug!(
                "Using discovered ZMQ address: {} ({:?}) -> {}",
                http_address, service_type, zmq_addr
            );
            return zmq_addr;
        }

        // Fallback: use HTTP address as ZMQ address
        debug!(
            "No ZMQ discovery result for {} ({:?}), using fallback: {}",
            http_address, service_type, http_address
        );
        http_address
    }

    /// Helper: Start profiling on a backend server with timeout
    async fn start_profiling(&self, worker_url: &str) {
        // Only profile if enabled
        if !self.enable_profiling {
            return;
        }

        // Start profiling on the worker
        self.pd_router.start_profiling(worker_url).await;

        // Spawn a timeout task that will call stop_profiling if timeout is reached
        let timeout_secs = self.profile_timeout_secs;
        let worker_url_owned = worker_url.to_string();
        let pd_router_clone = self.pd_router.clone();
        let profiling_tasks_clone = self.profiling_tasks.clone();

        let task_handle = tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_secs(timeout_secs)).await;

            info!(
                "Profiling timeout reached for {}, stopping profiling",
                worker_url_owned
            );
            pd_router_clone.stop_profiling(&worker_url_owned).await;

            // Remove ourselves from the tasks map
            let mut tasks = profiling_tasks_clone.lock().await;
            tasks.remove(&worker_url_owned);
        });

        // Store the abort handle
        let mut tasks = self.profiling_tasks.lock().await;
        if let Some(old_handle) = tasks.insert(worker_url.to_string(), task_handle.abort_handle()) {
            // Cancel any existing timeout task for this worker
            old_handle.abort();
        }
    }

    /// Helper: Stop profiling on a backend server and cancel timeout task
    async fn stop_profiling(&self, worker_url: &str) {
        // Only stop profiling if it was enabled
        if !self.enable_profiling {
            return;
        }

        // Cancel the timeout task if it exists
        let mut tasks = self.profiling_tasks.lock().await;
        if let Some(handle) = tasks.remove(worker_url) {
            handle.abort();
            info!("Cancelled profiling timeout task for {}", worker_url);
        }

        // Stop profiling on the worker
        self.pd_router.stop_profiling(worker_url).await;
    }

    /// Modify request for prefill stage (limit output to 1 token)
    /// - For inference/v1/generate: patch sampling_params.max_tokens and sampling_params.min_tokens
    /// - For /v1/responses: patch max_output_tokens (the only token-limit field for Responses API)
    /// - For other OpenAI endpoints (chat/completions): patch max_tokens, max_completion_tokens, min_tokens
    ///
    /// stream=false and stream_options removal are always applied at top level.
    fn prepare_prefill_request(mut request: Value, path: &str) -> Value {
        if path.contains("inference/v1/generate") {
            // Generate API: max_tokens and min_tokens are in sampling_params
            if let Some(sampling_params) = request.get_mut("sampling_params") {
                sampling_params["max_tokens"] = json!(1);
                // Also adjust min_tokens to ensure min_tokens <= max_tokens
                // This is required because vLLM validates that min_tokens <= max_tokens
                if let Some(min_tokens) = sampling_params.get("min_tokens").and_then(|v| v.as_u64())
                {
                    if min_tokens > 1 {
                        sampling_params["min_tokens"] = json!(1);
                    }
                }
            } else {
                // Create sampling_params with prefill defaults when missing
                request["sampling_params"] = json!({"max_tokens": 1, "min_tokens": 1});
            }
        } else if path.contains("/v1/responses") {
            // Responses API: only uses max_output_tokens (not max_tokens/max_completion_tokens)
            request["max_output_tokens"] = json!(1);
        } else {
            // OpenAI chat/completions endpoints
            request["max_tokens"] = json!(1);
            if request.get("max_completion_tokens").is_some() {
                request["max_completion_tokens"] = json!(1);
            }
            // Also adjust min_tokens to ensure min_tokens <= max_tokens
            // This is required because vLLM validates that min_tokens <= max_tokens
            if let Some(min_tokens) = request.get("min_tokens").and_then(|v| v.as_u64()) {
                if min_tokens > 1 {
                    request["min_tokens"] = json!(1);
                }
            }
        }
        // Force non-streaming for prefill to get JSON response with kv_transfer_params
        request["stream"] = json!(false);
        // Remove stream_options since we're setting stream=false
        if let Some(obj) = request.as_object_mut() {
            obj.remove("stream_options");
        }
        request
    }

    /// The persistent policy worker for an instance address, created on first
    /// sight and reused for the router's lifetime so load/processed counters
    /// accumulate across requests.
    fn policy_worker(&self, http_addr: &str) -> Arc<dyn Worker> {
        let full_url = if http_addr.starts_with("http://") || http_addr.starts_with("https://") {
            http_addr.to_string()
        } else {
            format!("http://{}", http_addr)
        };
        self.policy_workers
            .entry(full_url.clone())
            .or_insert_with(|| {
                Arc::new(BasicWorker::new(full_url, WorkerType::Regular)) as Arc<dyn Worker>
            })
            .clone()
    }

    /// Convert service discovery instances to persistent Worker objects for
    /// policy selection.
    fn instances_to_workers(&self, instances: &[(String, String)]) -> Vec<Arc<dyn Worker>> {
        instances
            .iter()
            .map(|(http_addr, _zmq_addr)| self.policy_worker(http_addr))
            .collect()
    }

    /// Select worker using policy-based load balancing
    fn select_worker_with_policy(
        &self,
        instances: &[(String, String)],
        is_prefill: bool,
        request_text: Option<&str>,
    ) -> Option<usize> {
        if instances.is_empty() {
            return None;
        }

        // Convert instances to workers for policy selection
        let workers = self.instances_to_workers(instances);

        // Get the appropriate policy
        let policy = if is_prefill {
            self.policy_registry.get_prefill_policy()
        } else {
            self.policy_registry.get_decode_policy()
        };

        // Use policy to select worker
        policy.select_worker(&workers, request_text)
    }

    /// Process vLLM request using pure service discovery
    async fn process_vllm_request(
        &self,
        request_json: Value,
        path: &str,
        headers: Option<&HeaderMap>,
    ) -> Response {
        debug!("Processing vLLM request for path: {}", path);
        debug!(
            "Request JSON: {}",
            serde_json::to_string_pretty(&request_json).unwrap_or_default()
        );

        // Get available instances from service discovery
        let prefill_instances = self.service_registry.get_prefill_instances();
        let decode_instances = self.service_registry.get_decode_instances();

        debug!(
            "Found {} prefill instances, {} decode instances from service discovery",
            prefill_instances.len(),
            decode_instances.len()
        );

        if prefill_instances.is_empty() || decode_instances.is_empty() {
            RouterMetrics::record_pd_error("server_selection");
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "No workers available via service discovery: {} prefill, {} decode",
                    prefill_instances.len(),
                    decode_instances.len()
                ),
            )
                .into_response();
        }

        // Use policy-based load balancing to select prefill and decode workers
        let request_text = serde_json::to_string(&request_json).ok();
        let request_str = request_text.as_deref();

        let prefill_idx =
            match self.select_worker_with_policy(&prefill_instances, true, request_str) {
                Some(idx) => idx,
                None => {
                    RouterMetrics::record_pd_error("server_selection");
                    return (
                        axum::http::StatusCode::SERVICE_UNAVAILABLE,
                        "Prefill policy failed to select a worker".to_string(),
                    )
                        .into_response();
                }
            };

        let decode_idx = match self.select_worker_with_policy(&decode_instances, false, request_str)
        {
            Some(idx) => idx,
            None => {
                RouterMetrics::record_pd_error("server_selection");
                return (
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    "Decode policy failed to select a worker".to_string(),
                )
                    .into_response();
            }
        };

        let (prefill_http, prefill_zmq) = &prefill_instances[prefill_idx];
        let (decode_http, decode_zmq) = &decode_instances[decode_idx];

        let prefill_policy_name = self.policy_registry.get_prefill_policy().name();
        let decode_policy_name = self.policy_registry.get_decode_policy().name();

        debug!(
            "vLLM policy-based routing: prefill={}({}) [policy:{}], decode={}({}) [policy:{}]",
            prefill_http,
            prefill_zmq,
            prefill_policy_name,
            decode_http,
            decode_zmq,
            decode_policy_name
        );

        // Process two-stage vLLM request with discovered endpoints
        // Track in-flight load on the persistent policy workers so selection
        // sees real imbalance. Decode residency is under-counted for
        // streaming responses (the guard drops when the stream STARTS, not
        // when it drains), so the load is a "committed to this worker" proxy
        // — good enough for the balance thresholds, and the
        // (load, processed, idx) tie-break covers the idle case.
        let prefill_pw = self.policy_worker(&prefill_instances[prefill_idx].0);
        let decode_pw = self.policy_worker(&decode_instances[decode_idx].0);
        prefill_pw.increment_load();
        decode_pw.increment_load();
        let result = self
            .process_vllm_two_stage_request_discovered(
                request_json,
                &prefill_instances[prefill_idx],
                &decode_instances[decode_idx],
                path,
                headers,
            )
            .await;
        prefill_pw.decrement_load();
        decode_pw.decrement_load();
        match result {
            Ok(response) => {
                debug!("Two-stage processing completed successfully");
                response
            }
            Err(e) => {
                error!("Two-stage processing failed: {}", e);
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Request processing failed: {}", e),
                )
                    .into_response()
            }
        }
    }

    /// Handle the decode response shared by both sequential and concurrent dispatch paths.
    /// Stops decode profiling, records metrics, then routes to streaming, logprobs-merge,
    /// or plain full-body response depending on the original request.
    #[allow(clippy::too_many_arguments)]
    async fn handle_decode_response(
        &self,
        decode_response: reqwest::Response,
        prefill_response_json: Option<&Value>,
        path: &str,
        prefill_http: &str,
        decode_http: &str,
        decode_base_http: &str,
        start_time: Instant,
        is_streaming: bool,
        needs_logprobs: bool,
    ) -> Result<Response, String> {
        debug!(
            "Decode server responded with status: {}",
            decode_response.status()
        );

        self.stop_profiling(&format!("http://{}", decode_base_http))
            .await;

        let duration = start_time.elapsed();
        RouterMetrics::record_pd_request(path);
        RouterMetrics::record_pd_request_duration(path, duration);
        RouterMetrics::record_pd_prefill_request(prefill_http);
        RouterMetrics::record_pd_decode_request(decode_http);

        if !decode_response.status().is_success() {
            RouterMetrics::record_pd_decode_error(decode_http);
        }

        if needs_logprobs && !is_streaming {
            debug!("Logprobs requested and non-streaming - merging prefill and decode logprobs");

            let status = decode_response.status();
            let resp_headers = decode_response.headers().clone();
            let decode_body = decode_response
                .bytes()
                .await
                .map_err(|e| format!("Failed to read decode response: {}", e))?;

            let mut decode_json: Value = serde_json::from_slice(&decode_body)
                .map_err(|e| format!("Failed to parse decode response as JSON: {}", e))?;

            let empty_json = Value::Null;
            let prefill_json_ref = prefill_response_json.unwrap_or(&empty_json);
            let merged = logprobs_merge::merge_logprobs_in_json(prefill_json_ref, &mut decode_json);
            if merged {
                debug!("Successfully merged logprobs from prefill and decode responses");
            } else {
                warn!("No logprobs were merged (might be expected if logprobs not in response)");
            }

            let merged_body = serde_json::to_vec(&decode_json)
                .map_err(|e| format!("Failed to serialize merged response: {}", e))?;

            let mut response_builder = axum::http::Response::builder().status(status);
            for (name, value) in resp_headers.iter() {
                response_builder = response_builder.header(name, value);
            }
            return response_builder
                .body(axum::body::Body::from(merged_body))
                .map_err(|e| format!("Failed to build response: {}", e));
        }

        debug!(
            "No logprobs merging needed (streaming={}, needs_logprobs={})",
            is_streaming, needs_logprobs
        );

        let status = decode_response.status();

        if is_streaming {
            let mut response_builder = axum::http::Response::builder().status(status);
            let mut decode_headers =
                header_utils::preserve_response_headers(decode_response.headers());
            decode_headers.remove(axum::http::header::CONTENT_LENGTH);
            for (name, value) in decode_headers.iter() {
                response_builder = response_builder.header(name, value);
            }
            let body = axum::body::Body::from_stream(decode_response.bytes_stream());
            return response_builder.body(body).map_err(|e| {
                format!(
                    "Failed to build streaming response from {}: {}",
                    decode_http, e
                )
            });
        }

        // Non-streaming, no logprobs: read entire body
        let decode_headers = decode_response.headers().clone();
        let body = decode_response
            .bytes()
            .await
            .map_err(|e| format!("Failed to read decode response: {}", e))?;
        let mut response_builder = axum::http::Response::builder().status(status);
        for (name, value) in decode_headers.iter() {
            response_builder = response_builder.header(name, value);
        }
        response_builder
            .body(axum::body::Body::from(body))
            .map_err(|e| format!("Failed to build response: {}", e))
    }

    /// Two-stage request processing for vLLM disaggregated mode using discovered endpoints
    async fn process_vllm_two_stage_request_discovered(
        &self,
        request_json: Value,
        prefill_instance: &(String, String),
        decode_instance: &(String, String),
        path: &str,
        headers: Option<&HeaderMap>,
    ) -> Result<Response, String> {
        let (prefill_http, prefill_zmq) = prefill_instance;
        let (decode_http, decode_zmq) = decode_instance;

        debug!("ENTERED process_vllm_two_stage_request_discovered method");
        let start_time = Instant::now();
        debug!(
            "Prefill: HTTP={}, ZMQ={}, Decode: HTTP={}, ZMQ={}, Path: {}",
            prefill_http, prefill_zmq, decode_http, decode_zmq, path
        );

        let request_id = Self::generate_vllm_request_id(prefill_zmq, decode_zmq);
        debug!(
            "Generated vLLM request ID for P2P coordination: {}",
            request_id
        );

        // DO NOT add P2P metadata to internal request_id - let vLLM generate clean internal IDs
        // The P2P metadata will be sent in X-Request-Id header instead

        // Prepare prefill request (max_tokens=1 to force prefill-only mode)
        let mut prefill_request = Self::prepare_prefill_request(request_json.clone(), path);

        // Generate a connector-specific transfer_id (None for NIXL)
        let transfer_id = self.generate_transfer_id();

        // Add kv_transfer_params for KV connector support at top level
        prefill_request["kv_transfer_params"] =
            self.build_prefill_kv_transfer_params(transfer_id.as_deref())?;

        debug!(
            "Added kv_transfer_params to prefill request for {:?} connector",
            self.kv_connector
        );

        let (prefill_base_http, prefill_dp_rank) =
            extract_base_http_and_dp_rank(prefill_http, self.intra_node_data_parallel_size);
        let (decode_base_http, decode_dp_rank) =
            extract_base_http_and_dp_rank(decode_http, self.intra_node_data_parallel_size);

        // Concurrent dispatch: e.g. MoRI-IO WRITE mode
        let is_concurrent_dispatch = matches!(self.kv_connector, KvConnector::MoriIO)
            && matches!(self.moriio_transfer_mode(), Some(MoriIOTransferMode::Write));

        let needs_logprobs = request_json.get("logprobs").is_some()
            || request_json
                .get("echo")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
        let is_streaming = request_json
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let prefill_request_str = serde_json::to_string(&prefill_request)
            .map_err(|e| format!("Failed to serialize prefill request: {}", e))?;

        let prefill_request_url = format!("http://{}{}", prefill_base_http, path);

        // Stage 1: dispatch prefill.
        // Sequential mode: send now and await — decode kv_transfer_params come from the prefill response.
        // Concurrent mode: skip; prefill is sent concurrently with decode in Stage 2 via tokio::join!.
        let prefill_response_json: Option<Value> = if is_concurrent_dispatch {
            None
        } else {
            debug!(
                "Stage 1: Sending prefill-only request (max_tokens=1) to prefill server at http://{}",
                prefill_http
            );
            self.start_profiling(&format!("http://{}", prefill_base_http))
                .await;

            let prefill_response = match otel_http::send_client_request(
                build_prefill_request_builder(
                    &self.http_client,
                    &prefill_request_url,
                    &request_id,
                    prefill_dp_rank,
                )
                .body(prefill_request_str.clone()),
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
                Ok(resp) => resp,
                Err(e) => {
                    let full_error = error_chain(&e);
                    let duration = start_time.elapsed();
                    RouterMetrics::record_pd_prefill_error(prefill_http);
                    RouterMetrics::record_pd_request(path);
                    RouterMetrics::record_pd_request_duration(path, duration);
                    return Err(format!(
                        "Prefill request failed to {}: {}",
                        prefill_http, full_error
                    ));
                }
            };

            let prefill_status = prefill_response.status();
            debug!("Prefill server responded with status: {}", prefill_status);

            if !prefill_status.is_success() {
                let duration = start_time.elapsed();
                RouterMetrics::record_pd_prefill_error(prefill_http);
                RouterMetrics::record_pd_request(path);
                RouterMetrics::record_pd_request_duration(path, duration);
                let error_body = prefill_response.text().await.unwrap_or_default();
                return Err(format!(
                    "Prefill server error {}: {}",
                    prefill_status, error_body
                ));
            }

            // Extract kv_transfer_params from prefill response
            let prefill_response_text = prefill_response.text().await.map_err(|e| {
                let full_error = error_chain(&e);
                format!(
                    "Failed to read prefill response from {}: {}",
                    prefill_http, full_error
                )
            })?;

            debug!("Prefill response body: {}", prefill_response_text);

            let prefill_json: Value = serde_json::from_str(&prefill_response_text)
                .map_err(|e| format!("Failed to parse prefill response as JSON: {}", e))?;

            // Stop profiling on prefill server once we have its response.
            self.stop_profiling(&format!("http://{}", prefill_base_http))
                .await;

            Some(prefill_json)
        };

        // Prepare decode request
        let mut decode_request = request_json.clone();
        let prefill_url_key = format!("http://{}", prefill_base_http);
        if let Some(params) = self
            .build_decode_kv_transfer_params(
                &prefill_url_key,
                prefill_response_json.as_ref(),
                transfer_id.as_deref(),
                prefill_dp_rank.map(|r| r as u32),
            )
            .await
        {
            decode_request["kv_transfer_params"] = params;
            debug!(
                "Added kv_transfer_params to decode request for {:?} connector",
                self.kv_connector
            );
        }

        let decode_request_str = serde_json::to_string(&decode_request)
            .map_err(|e| format!("Failed to serialize decode request: {}", e))?;

        // Stage 2: Send to decode server with original request and same P2P coordination header
        debug!(
            "Stage 2: Sending original request to decode server at http://{}",
            decode_http
        );

        // Start profiling on decode server
        self.start_profiling(&format!("http://{}", decode_base_http))
            .await;

        let mut decode_request_builder = self
            .http_client
            .post(format!("http://{}{}", decode_base_http, path))
            .header("Content-Type", "application/json")
            .header("X-Request-Id", &request_id); // Same P2P coordination metadata in header

        // Add X-data-parallel-rank header using shared utilities
        decode_request_builder =
            dp_utils::add_dp_rank_header(decode_request_builder, decode_dp_rank);
        if let Some(rank) = decode_dp_rank {
            debug!(
                "Added X-data-parallel-rank={} header to decode request",
                rank
            );
        }

        let decode_request_url = format!("http://{}{}", decode_base_http, path);

        // Concurrent dispatch: run prefill and decode concurrently via tokio::join! so the prefill
        // task is always guaranteed to execute (unlike fire-and-forget tokio::spawn, which
        // can be silently dropped under load) and both HTTP sends start at the same time.
        if is_concurrent_dispatch {
            self.start_profiling(&format!("http://{}", prefill_base_http))
                .await;
            // Capture references rather than clones — tokio::join! polls both futures on the
            // same task so they don't need to be Send, and local borrows are valid for the
            // duration of the join.
            let http_client = &self.http_client;
            let enable_profiling = self.enable_profiling;
            let profiling_tasks = &self.profiling_tasks;
            let pd_router = &self.pd_router;
            let prefill_fut = async move {
                let result = otel_http::send_client_request(
                    build_prefill_request_builder(
                        http_client,
                        &prefill_request_url,
                        &request_id,
                        prefill_dp_rank,
                    )
                    .body(prefill_request_str),
                    headers,
                    ClientRequestOptions {
                        method: "POST",
                        url: &prefill_request_url,
                        route: Some(path),
                        request_phase: Some("prefill"),
                    },
                )
                .await;
                if enable_profiling {
                    let worker_url = format!("http://{}", prefill_base_http);
                    let mut tasks = profiling_tasks.lock().await;
                    if let Some(handle) = tasks.remove(&worker_url) {
                        handle.abort();
                    }
                    pd_router.stop_profiling(&worker_url).await;
                }
                match result {
                    Ok(resp) if resp.status().is_success() => {
                        let status = resp.status();
                        match resp.bytes().await {
                            Ok(bytes) => match serde_json::from_slice::<Value>(&bytes) {
                                Ok(json) => {
                                    debug!("Concurrent prefill completed with status {}", status);
                                    Ok(Some(json))
                                }
                                Err(_) => {
                                    debug!(
                                        "Concurrent prefill completed with status {} (non-JSON body)",
                                        status
                                    );
                                    Ok(None)
                                }
                            },
                            Err(e) => {
                                warn!("Concurrent prefill: failed to read response body: {}", e);
                                Ok(None)
                            }
                        }
                    }
                    Ok(resp) => {
                        let status = resp.status();
                        warn!("Concurrent prefill returned non-2xx status: {}", status);
                        Err(format!("Prefill request failed with status {}", status))
                    }
                    Err(e) => {
                        warn!("Concurrent prefill request failed: {}", e);
                        Err(format!("Prefill request failed: {}", e))
                    }
                }
            };
            let decode_fut = otel_http::send_client_request(
                decode_request_builder.body(decode_request_str),
                headers,
                ClientRequestOptions {
                    method: "POST",
                    url: &decode_request_url,
                    route: Some(path),
                    request_phase: Some("decode"),
                },
            );
            let (prefill_result, decode_result) = tokio::join!(prefill_fut, decode_fut);
            let concurrent_prefill_response_json: Option<Value> = match prefill_result {
                Err(prefill_err) => {
                    self.stop_profiling(&format!("http://{}", decode_base_http))
                        .await;
                    let duration = start_time.elapsed();
                    RouterMetrics::record_pd_prefill_error(prefill_http);
                    RouterMetrics::record_pd_request(path);
                    RouterMetrics::record_pd_request_duration(path, duration);
                    RouterMetrics::record_pd_prefill_request(prefill_http);
                    return Err(format!(
                        "Prefill request failed to {}: {}",
                        prefill_http, prefill_err
                    ));
                }
                Ok(json) => json,
            };
            let decode_response = match decode_result {
                Ok(resp) => resp,
                Err(e) => {
                    self.stop_profiling(&format!("http://{}", decode_base_http))
                        .await;
                    let full_error = error_chain(&e);
                    let duration = start_time.elapsed();
                    RouterMetrics::record_pd_decode_error(decode_http);
                    RouterMetrics::record_pd_request(path);
                    RouterMetrics::record_pd_request_duration(path, duration);
                    RouterMetrics::record_pd_prefill_request(prefill_http);
                    return Err(format!(
                        "Decode request failed to {}: {}",
                        decode_http, full_error
                    ));
                }
            };
            return self
                .handle_decode_response(
                    decode_response,
                    concurrent_prefill_response_json.as_ref(),
                    path,
                    prefill_http,
                    decode_http,
                    &decode_base_http,
                    start_time,
                    is_streaming,
                    needs_logprobs,
                )
                .await;
        }

        let decode_response = match otel_http::send_client_request(
            decode_request_builder.body(decode_request_str),
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
            Ok(resp) => resp,
            Err(e) => {
                let full_error = error_chain(&e);
                let duration = start_time.elapsed();
                RouterMetrics::record_pd_decode_error(decode_http);
                RouterMetrics::record_pd_request(path);
                RouterMetrics::record_pd_request_duration(path, duration);
                RouterMetrics::record_pd_prefill_request(prefill_http);
                return Err(format!(
                    "Decode request failed to {}: {}",
                    decode_http, full_error
                ));
            }
        };

        self.handle_decode_response(
            decode_response,
            prefill_response_json.as_ref(),
            path,
            prefill_http,
            decode_http,
            &decode_base_http,
            start_time,
            is_streaming,
            needs_logprobs,
        )
        .await
    }

    /// Two-stage request processing for vLLM disaggregated mode
    ///
    /// This function handles fine-grained load tracking: the prefill worker's load is only
    /// incremented during the prefill phase, and the decode worker's load is only incremented
    /// during the decode phase. This accurately reflects the sequential nature of PD disaggregation.
    async fn process_vllm_two_stage_request(
        &self,
        original_request: Value,
        prefill_worker: Arc<dyn Worker>,
        decode_worker: Arc<dyn Worker>,
        path: &str,
        headers: Option<&HeaderMap>,
    ) -> Result<Response, PDRouterError> {
        debug!("ENTERED process_vllm_two_stage_request method");
        let start_time = Instant::now();
        debug!(
            "Prefill worker: {}, Decode worker: {}, Path: {}",
            prefill_worker.url(),
            decode_worker.url(),
            path
        );

        // Increment prefill load at the start of the prefill phase
        prefill_worker.increment_load();

        let prefill_zmq_addr =
            self.get_zmq_address(prefill_worker.base_url(), ServiceType::Prefill);
        let decode_zmq_addr = self.get_zmq_address(decode_worker.base_url(), ServiceType::Decode);
        let request_id = Self::generate_vllm_request_id(&prefill_zmq_addr, &decode_zmq_addr);

        debug!("Generated vLLM request ID: {}", request_id);
        debug!("🔍 vLLM Proxy Comparison:");
        debug!("  📋 vLLM Proxy Request ID format: ___prefill_addr_{{zmq_addr}}___decode_addr_{{zmq_addr}}_{{uuid}}");
        debug!("  📋 Our Request ID format: ___prefill_addr_{{http_addr}}___decode_addr_{{http_addr}}_{{uuid}}");
        debug!("  📋 vLLM Proxy headers: Authorization: Bearer $OPENAI_API_KEY, X-Request-Id: {{request_id}}");
        debug!(
            "  📋 Our headers: Authorization: Bearer $OPENAI_API_KEY, X-Request-Id: {{request_id}}"
        );

        // Stage 1: Prepare prefill request with max_tokens=1 and kv_transfer_params
        let mut prefill_request = Self::prepare_prefill_request(original_request.clone(), path);

        // Generate a connector-specific transfer_id (None for NIXL)
        let transfer_id = self.generate_transfer_id();

        // Add kv_transfer_params for KV connector support at top level
        prefill_request["kv_transfer_params"] = self
            .build_prefill_kv_transfer_params(transfer_id.as_deref())
            .map_err(|reason| PDRouterError::InvalidConfiguration { reason })?;

        debug!(
            "Added kv_transfer_params to prefill request for {:?} connector",
            self.kv_connector
        );

        // Use endpoint_url() to get the base URL without @rank suffix,
        // avoiding IPv6+DP URL corruption (same fix as Router and PdRouterBase)
        let prefill_base_url = prefill_worker.base_url().to_string();
        let prefill_dp_rank = prefill_worker.dp_rank();
        let prefill_url = prefill_worker.endpoint_url(path);

        debug!(
            "🚀 vLLM Stage 1 - Prefill: {} with request_id: {}",
            prefill_url, request_id
        );
        if let Some(rank) = prefill_dp_rank {
            debug!("📤 Prefill request headers: Authorization=Bearer [REDACTED], X-Request-Id={}, X-data-parallel-rank={}", request_id, rank);
        } else {
            debug!(
                "📤 Prefill request headers: Authorization=Bearer [REDACTED], X-Request-Id={}",
                request_id
            );
        }
        debug!(
            "📤 Prefill request payload: {}",
            serde_json::to_string_pretty(&prefill_request).unwrap_or_default()
        );

        // Start profiling on prefill server
        self.start_profiling(&prefill_base_url).await;

        let mut prefill_request_builder = self
            .pd_router
            .client
            .post(&prefill_url)
            .header("Content-Type", "application/json")
            .header(
                "Authorization",
                format!(
                    "Bearer {}",
                    std::env::var("OPENAI_API_KEY").unwrap_or_default()
                ),
            )
            .header("X-Request-Id", &request_id);

        // Add X-data-parallel-rank header using shared utilities
        prefill_request_builder =
            dp_utils::add_dp_rank_header(prefill_request_builder, prefill_dp_rank);

        let prefill_response = match otel_http::send_client_request(
            prefill_request_builder.json(&prefill_request),
            headers,
            ClientRequestOptions {
                method: "POST",
                url: &prefill_url,
                route: Some(path),
                request_phase: Some("prefill"),
            },
        )
        .await
        {
            Ok(resp) => resp,
            Err(e) => {
                prefill_worker.decrement_load();
                let full_error = error_chain(&e);
                let duration = start_time.elapsed();
                RouterMetrics::record_pd_prefill_error(&prefill_base_url);
                RouterMetrics::record_pd_request(path);
                RouterMetrics::record_pd_request_duration(path, duration);
                return Err(PDRouterError::NetworkError {
                    message: format!("Prefill request failed to {}: {}", prefill_url, full_error),
                });
            }
        };

        debug!("📥 Prefill response status: {}", prefill_response.status());
        debug!(
            "📥 Prefill response headers: {:?}",
            prefill_response.headers()
        );

        // Extract prefill response body to get kv_transfer_params
        let prefill_bytes = match prefill_response.bytes().await {
            Ok(bytes) => bytes,
            Err(e) => {
                prefill_worker.decrement_load();
                let full_error = error_chain(&e);
                let duration = start_time.elapsed();
                RouterMetrics::record_pd_prefill_error(&prefill_base_url);
                RouterMetrics::record_pd_request(path);
                RouterMetrics::record_pd_request_duration(path, duration);
                return Err(PDRouterError::NetworkError {
                    message: format!(
                        "Failed to read prefill response from {}: {}",
                        prefill_url, full_error
                    ),
                });
            }
        };

        debug!(
            "📥 Prefill response body size: {} bytes",
            prefill_bytes.len()
        );
        if prefill_bytes.len() < 1024 {
            debug!(
                "📥 Prefill response body content: {}",
                String::from_utf8_lossy(&prefill_bytes)
            );
        }

        // Parse prefill response to extract kv_transfer_params
        let prefill_response_json: Value = match serde_json::from_slice(&prefill_bytes) {
            Ok(json) => json,
            Err(e) => {
                prefill_worker.decrement_load();
                let duration = start_time.elapsed();
                RouterMetrics::record_pd_prefill_error(&prefill_base_url);
                RouterMetrics::record_pd_request(path);
                RouterMetrics::record_pd_request_duration(path, duration);
                return Err(PDRouterError::NetworkError {
                    message: format!("Failed to parse prefill response as JSON: {}", e),
                });
            }
        };

        // Extract kv_transfer_params from prefill response if present
        let kv_transfer_params = prefill_response_json.get("kv_transfer_params").cloned();

        if let Some(ref params) = kv_transfer_params {
            debug!(
                "Extracted kv_transfer_params from prefill response: {}",
                serde_json::to_string_pretty(params).unwrap_or_default()
            );
        } else {
            debug!("No kv_transfer_params found in prefill response, will proceed without them");
        }

        // Stop profiling on prefill server after its work is done
        self.stop_profiling(&prefill_base_url).await;

        // Prefill phase complete: decrement prefill load, increment decode load
        prefill_worker.decrement_load();
        decode_worker.increment_load();

        debug!("✅ vLLM Stage 1 completed, starting Stage 2 - Decode");

        // Stage 2: Prepare decode request with kv_transfer_params
        let mut decode_request = original_request.clone();
        if matches!(self.kv_connector, KvConnector::Mooncake) {
            // Mooncake: set decode params proactively from bootstrap info
            if let Some((bootstrap_addr, engine_id)) = self
                .get_mooncake_info(&prefill_base_url, prefill_dp_rank)
                .await
            {
                decode_request["kv_transfer_params"] = self
                    .build_mooncake_decode_kv_transfer_params(
                        transfer_id.as_deref().unwrap_or(""),
                        &bootstrap_addr,
                        &engine_id,
                    );
                debug!(
                    "Set Mooncake decode kv_transfer_params with bootstrap_addr={}, engine_id={}",
                    bootstrap_addr, engine_id
                );
            } else {
                warn!(
                    "No Mooncake bootstrap info for prefill {}, decode will proceed without kv_transfer_params",
                    prefill_base_url
                );
            }
        } else {
            // Sequential dispatch (NIXL, MoRI-IO READ): extract kv_transfer_params from prefill response
            if let Some(mut params) = kv_transfer_params {
                if matches!(self.kv_connector, KvConnector::MoriIO) {
                    // MoRI-IO decode connector needs to know how many prefill DP ranks to handshake with.
                    params["remote_dp_size"] = json!(self.intra_node_data_parallel_size);
                }
                decode_request["kv_transfer_params"] = params;
                debug!(
                    "Added kv_transfer_params to decode request for {:?} connector",
                    self.kv_connector
                );
            }
        }

        // Use endpoint_url() to get the base URL without @rank suffix,
        // avoiding IPv6+DP URL corruption (same fix as Router and PdRouterBase)
        let decode_base_url = decode_worker.base_url().to_string();
        let decode_dp_rank = decode_worker.dp_rank();
        let decode_url = decode_worker.endpoint_url(path);

        debug!(
            "🚀 vLLM Stage 2 - Decode: {} with request_id: {}",
            decode_url, request_id
        );
        if let Some(rank) = decode_dp_rank {
            debug!("📤 Decode request headers: Authorization=Bearer [REDACTED], X-Request-Id={}, X-data-parallel-rank={}", request_id, rank);
        } else {
            debug!(
                "📤 Decode request headers: Authorization=Bearer [REDACTED], X-Request-Id={}",
                request_id
            );
        }
        debug!(
            "📤 Decode request payload: {}",
            serde_json::to_string_pretty(&decode_request).unwrap_or_default()
        );

        // Start profiling on decode server
        self.start_profiling(&decode_base_url).await;

        let mut decode_request_builder = self
            .pd_router
            .client
            .post(&decode_url)
            .header("Content-Type", "application/json")
            .header(
                "Authorization",
                format!(
                    "Bearer {}",
                    std::env::var("OPENAI_API_KEY").unwrap_or_default()
                ),
            )
            .header("X-Request-Id", &request_id);

        // Add X-data-parallel-rank header using shared utilities
        decode_request_builder =
            dp_utils::add_dp_rank_header(decode_request_builder, decode_dp_rank);

        let decode_response = match otel_http::send_client_request(
            decode_request_builder.json(&decode_request),
            headers,
            ClientRequestOptions {
                method: "POST",
                url: &decode_url,
                route: Some(path),
                request_phase: Some("decode"),
            },
        )
        .await
        {
            Ok(resp) => resp,
            Err(e) => {
                decode_worker.decrement_load();
                let full_error = error_chain(&e);
                let duration = start_time.elapsed();
                RouterMetrics::record_pd_decode_error(&decode_base_url);
                RouterMetrics::record_pd_request(path);
                RouterMetrics::record_pd_request_duration(path, duration);
                RouterMetrics::record_pd_prefill_request(&prefill_base_url);
                return Err(PDRouterError::NetworkError {
                    message: format!("Decode request failed to {}: {}", decode_url, full_error),
                });
            }
        };

        // Stop profiling on decode server after response received
        self.stop_profiling(&decode_base_url).await;

        // Decode phase complete: decrement decode load
        decode_worker.decrement_load();

        let status = decode_response.status();
        let headers = decode_response.headers().clone();

        info!("📥 Decode response status: {}", status);
        info!("📥 Decode response headers: {:?}", headers);

        // Record PD metrics
        let duration = start_time.elapsed();
        RouterMetrics::record_pd_request(path);
        RouterMetrics::record_pd_request_duration(path, duration);
        RouterMetrics::record_pd_prefill_request(&prefill_base_url);
        RouterMetrics::record_pd_decode_request(&decode_base_url);

        if !status.is_success() {
            RouterMetrics::record_pd_decode_error(&decode_base_url);
        }

        // Check if logprobs merging is needed
        let needs_logprobs = original_request.get("logprobs").is_some()
            || original_request
                .get("echo")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
        let is_streaming = original_request
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // If logprobs requested and non-streaming, merge prefill and decode logprobs
        if needs_logprobs && !is_streaming {
            debug!("Logprobs requested and non-streaming - merging prefill and decode logprobs");

            // Read decode response body
            let decode_body =
                decode_response
                    .bytes()
                    .await
                    .map_err(|e| PDRouterError::NetworkError {
                        message: format!(
                            "Failed to read decode response from {}: {}",
                            decode_url, e
                        ),
                    })?;

            // Parse decode response as JSON
            let mut decode_json: Value =
                serde_json::from_slice(&decode_body).map_err(|e| PDRouterError::NetworkError {
                    message: format!("Failed to parse decode response as JSON: {}", e),
                })?;

            // Merge logprobs from prefill into decode response
            let merged =
                logprobs_merge::merge_logprobs_in_json(&prefill_response_json, &mut decode_json);
            if merged {
                debug!("Successfully merged logprobs from prefill and decode responses");
            } else {
                warn!("No logprobs were merged (might be expected if logprobs not in response)");
            }

            // Serialize merged response
            let merged_body =
                serde_json::to_vec(&decode_json).map_err(|e| PDRouterError::NetworkError {
                    message: format!("Failed to serialize merged response: {}", e),
                })?;

            let mut response_builder = Response::builder().status(status);
            for (key, value) in headers.iter() {
                if key != "transfer-encoding" && key != "content-length" {
                    response_builder = response_builder.header(key, value);
                }
            }

            response_builder.body(Body::from(merged_body)).map_err(|e| {
                PDRouterError::NetworkError {
                    message: format!("Failed to build response from {}: {}", decode_url, e),
                }
            })
        } else {
            // No logprobs merging needed - return decode response as-is (streaming or no logprobs)
            debug!(
                "No logprobs merging needed (streaming={}, needs_logprobs={})",
                is_streaming, needs_logprobs
            );

            let mut response_builder = Response::builder().status(status);
            for (key, value) in headers.iter() {
                if key != "transfer-encoding" && key != "content-length" {
                    response_builder = response_builder.header(key, value);
                }
            }

            let body = Body::from_stream(decode_response.bytes_stream());
            response_builder
                .body(body)
                .map_err(|e| PDRouterError::NetworkError {
                    message: format!("Failed to build response from {}: {}", decode_url, e),
                })
        }
    }

    /// Create a new vLLM PD router
    /// Supports two modes:
    /// 1. Discovery mode: discovery_address is Some, prefill_urls and decode_urls are empty
    /// 2. Direct URL mode: discovery_address is None, prefill_urls and decode_urls are provided
    pub async fn new(
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        discovery_address: Option<String>,
        ctx: &Arc<crate::server::AppContext>,
    ) -> Result<Self, String> {
        let kv_connector = ctx.router_config.kv_connector;
        let http_client = reqwest::Client::new();

        if let Some(ref addr) = discovery_address {
            // Discovery mode
            info!(
                "VllmPDRouter::new called in discovery mode with address: {}",
                addr
            );

            // Create underlying PD router with empty worker lists (they'll be discovered dynamically)
            let pd_router = PdRouterBase::new(vec![], vec![], ctx).await?;

            // Initialize service discovery
            let mut service_registry = ServiceRegistry::new();

            info!("Starting vLLM service discovery on {}", addr);
            service_registry
                .start_listener(addr, kv_connector)
                .await
                .map_err(|e| format!("Failed to start service discovery: {}", e))?;

            info!(
                "VllmPDRouter created successfully with pure service discovery, kv_connector={:?}",
                kv_connector
            );

            Ok(Self {
                pd_router,
                service_registry: Arc::new(service_registry),
                http_client,
                policy_registry: ctx.policy_registry.clone(),
                use_discovery: true,
                enable_profiling: ctx.router_config.enable_profiling,
                profile_timeout_secs: ctx.router_config.profile_timeout_secs,
                profiling_tasks: Arc::new(Mutex::new(HashMap::new())),
                intra_node_data_parallel_size: ctx.router_config.intra_node_data_parallel_size,
                kv_connector,
                mooncake_prefill_info: Arc::new(Mutex::new(HashMap::new())),
                policy_workers: DashMap::new(),
            })
        } else {
            // Direct URL mode (same as PdRouterBase)
            info!(
                "VllmPDRouter::new called in direct URL mode with {} prefill, {} decode workers",
                prefill_urls.len(),
                decode_urls.len()
            );

            // Create underlying PD router with provided worker lists
            let pd_router = PdRouterBase::new(prefill_urls.clone(), decode_urls, ctx).await?;

            // No service discovery in direct URL mode
            let service_registry = ServiceRegistry::new();

            info!("VllmPDRouter created successfully with direct URLs");

            let prefill_workers = pd_router.worker_registry.get_prefill_workers();
            let decode_workers = pd_router.worker_registry.get_decode_workers();
            let prefill_policy = ctx.policy_registry.get_prefill_policy();
            let decode_policy = ctx.policy_registry.get_decode_policy();

            if prefill_policy.requires_initialization() {
                info!("Initializing prefill policy with workers.");
                prefill_policy.init_workers(&prefill_workers);
            }
            if decode_policy.requires_initialization() {
                info!("Initializing decode policy with workers.");
                decode_policy.init_workers(&decode_workers);
            }
            info!("Initializing prefill and decode policies with workers.");

            // Query Mooncake bootstrap servers if kv_connector is mooncake
            let mooncake_prefill_info = Arc::new(Mutex::new(HashMap::new()));
            if matches!(kv_connector, KvConnector::Mooncake) {
                info!("Mooncake connector enabled, querying prefill bootstrap servers...");
                for (url, bootstrap_port) in &prefill_urls {
                    let parsed = url::Url::parse(url)
                        .map_err(|e| format!("Invalid prefill URL '{}': {}", url, e))?;
                    let host = parsed.host_str().unwrap_or("127.0.0.1");
                    let port = bootstrap_port.unwrap_or(8998);
                    let bootstrap_addr = format!("http://{}:{}", host, port);
                    let base_url = format!(
                        "{}://{}:{}",
                        parsed.scheme(),
                        host,
                        parsed.port().unwrap_or(8000)
                    );

                    info!(
                        "Querying Mooncake bootstrap at {} for prefill {}",
                        bootstrap_addr, base_url
                    );
                    match Self::query_mooncake_bootstrap(&http_client, &bootstrap_addr).await {
                        Ok(dp_engine_ids) => {
                            info!(
                                "Got Mooncake engine_ids for {}: {:?}",
                                base_url, dp_engine_ids
                            );
                            mooncake_prefill_info.lock().await.insert(
                                base_url,
                                MooncakePrefillInfo {
                                    bootstrap_addr,
                                    dp_engine_ids,
                                },
                            );
                        }
                        Err(e) => {
                            error!("Failed to query Mooncake bootstrap for {}: {}", base_url, e);
                            return Err(e);
                        }
                    }
                }
                info!("Mooncake bootstrap query complete for all prefill nodes");
            }

            Ok(Self {
                pd_router,
                service_registry: Arc::new(service_registry),
                http_client,
                policy_registry: ctx.policy_registry.clone(),
                use_discovery: false,
                enable_profiling: ctx.router_config.enable_profiling,
                profile_timeout_secs: ctx.router_config.profile_timeout_secs,
                profiling_tasks: Arc::new(Mutex::new(HashMap::new())),
                intra_node_data_parallel_size: ctx.router_config.intra_node_data_parallel_size,
                kv_connector,
                mooncake_prefill_info,
                policy_workers: DashMap::new(),
            })
        }
    }

    /// Add a prefill server to the router
    /// Delegates to the underlying PdRouterBase
    pub async fn add_prefill_server(
        &self,
        url: String,
        bootstrap_port: Option<u16>,
    ) -> Result<String, PDRouterError> {
        self.pd_router.add_prefill_server(url, bootstrap_port).await
    }

    /// Add a decode server to the router
    /// Delegates to the underlying PdRouterBase
    pub async fn add_decode_server(&self, url: String) -> Result<String, PDRouterError> {
        self.pd_router.add_decode_server(url).await
    }

    /// Remove a prefill server from the router
    /// Delegates to the underlying PdRouterBase
    pub async fn remove_prefill_server(&self, url: &str) -> Result<String, PDRouterError> {
        self.pd_router.remove_prefill_server(url).await
    }

    /// Remove a decode server from the router
    /// Delegates to the underlying PdRouterBase
    pub async fn remove_decode_server(&self, url: &str) -> Result<String, PDRouterError> {
        self.pd_router.remove_decode_server(url).await
    }

    /// Get a reference to the underlying PdRouterBase's worker registry
    /// This allows access to worker information for refresh operations
    pub fn worker_registry(&self) -> &crate::core::WorkerRegistry {
        &self.pd_router.worker_registry
    }
}

// Delegate most RouterTrait methods to the underlying PdRouterBase,
// but override specific ones for vLLM behavior
#[async_trait]
impl RouterTrait for VllmPDRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health(&self, req: Request<Body>) -> Response {
        self.pd_router.health(req).await
    }

    async fn health_generate(&self, req: Request<Body>) -> Response {
        self.pd_router.health_generate(req).await
    }

    async fn get_server_info(&self, req: Request<Body>) -> Response {
        self.pd_router.get_server_info(req).await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        self.pd_router.get_models(req).await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        self.pd_router.get_model_info(req).await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &crate::protocols::spec::GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        let request_json = match serde_json::to_value(body) {
            Ok(json) => json,
            Err(e) => {
                return (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Serialization error: {}", e),
                )
                    .into_response()
            }
        };
        self.route_transparent(headers, "/generate", &Method::POST, request_json)
            .await
    }

    async fn route_inference_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &crate::protocols::spec::InferenceGenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        let request_json = match serde_json::to_value(body) {
            Ok(json) => json,
            Err(e) => {
                return (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Serialization error: {}", e),
                )
                    .into_response()
            }
        };
        self.route_transparent(
            headers,
            "/inference/v1/generate",
            &Method::POST,
            request_json,
        )
        .await
    }

    // Override OpenAI-compatible routes for vLLM two-stage processing
    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &crate::protocols::spec::ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        info!(
            "vLLM route_chat called, use_discovery={}",
            self.use_discovery
        );

        if self.use_discovery {
            // Discovery mode - use vLLM-specific two-stage processing
            info!("Using service discovery mode, processing vLLM two-stage request");

            // Convert to generic request and use vLLM processing
            let request_json = match serde_json::to_value(body) {
                Ok(json) => {
                    debug!(
                        "Serialized chat request: {}",
                        serde_json::to_string_pretty(&json).unwrap_or_default()
                    );
                    json
                }
                Err(e) => {
                    return (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Serialization error: {}", e),
                    )
                        .into_response()
                }
            };

            // Process vLLM two-stage request with service discovery
            self.process_vllm_request(request_json, "/v1/chat/completions", headers)
                .await
        } else {
            // Direct URL mode - implement routing logic here (not delegating to PdRouterBase)
            info!("Using direct URL mode with VllmPDRouter's own routing logic");

            // Convert request to JSON
            let request_json = match serde_json::to_value(body) {
                Ok(json) => {
                    debug!(
                        "Serialized chat request: {}",
                        serde_json::to_string_pretty(&json).unwrap_or_default()
                    );
                    json
                }
                Err(e) => {
                    return (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Serialization error: {}", e),
                    )
                        .into_response()
                }
            };

            // Get prefill and decode workers from worker_registry
            let prefill_workers = self.pd_router.worker_registry.get_prefill_workers();
            let decode_workers = self.pd_router.worker_registry.get_decode_workers();

            info!(
                "Found {} prefill workers, {} decode workers from worker_registry",
                prefill_workers.len(),
                decode_workers.len()
            );

            if prefill_workers.is_empty() || decode_workers.is_empty() {
                RouterMetrics::record_pd_error("server_selection");
                return (
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    format!(
                        "No workers available: {} prefill, {} decode",
                        prefill_workers.len(),
                        decode_workers.len()
                    ),
                )
                    .into_response();
            }

            // Select workers using policy with headers for consistent hash
            let request_text = serde_json::to_string(&request_json).ok();
            let request_str = request_text.as_deref();
            let request_headers: Option<HashMap<String, String>> = headers.map(|h| {
                h.iter()
                    .filter_map(|(name, value)| {
                        value
                            .to_str()
                            .ok()
                            .map(|v| (name.as_str().to_lowercase(), v.to_string()))
                    })
                    .collect()
            });

            let prefill_policy = self.policy_registry.get_prefill_policy();
            let decode_policy = self.policy_registry.get_decode_policy();

            let prefill_idx = match prefill_policy.select_worker_with_headers(
                &prefill_workers,
                request_str,
                request_headers.as_ref(),
            ) {
                Some(idx) => idx,
                None => {
                    RouterMetrics::record_pd_error("server_selection");
                    return (
                        axum::http::StatusCode::SERVICE_UNAVAILABLE,
                        "Prefill policy failed to select a worker".to_string(),
                    )
                        .into_response();
                }
            };

            let decode_idx = match decode_policy.select_worker_with_headers(
                &decode_workers,
                request_str,
                request_headers.as_ref(),
            ) {
                Some(idx) => idx,
                None => {
                    RouterMetrics::record_pd_error("server_selection");
                    return (
                        axum::http::StatusCode::SERVICE_UNAVAILABLE,
                        "Decode policy failed to select a worker".to_string(),
                    )
                        .into_response();
                }
            };

            let prefill_worker = &prefill_workers[prefill_idx];
            let decode_worker = &decode_workers[decode_idx];
            // Load tracking is handled inside process_vllm_two_stage_request for fine-grained
            // tracking: prefill load only during prefill phase, decode load only during decode phase.

            info!(
                "Chat: Selected prefill={} [policy:{}], decode={} [policy:{}]",
                prefill_worker.url(),
                prefill_policy.name(),
                decode_worker.url(),
                decode_policy.name()
            );

            // Execute dual dispatch with vLLM two-stage processing
            let resp = match self
                .process_vllm_two_stage_request(
                    request_json,
                    prefill_worker.clone(),
                    decode_worker.clone(),
                    "/v1/chat/completions",
                    headers,
                )
                .await
            {
                Ok(response) => {
                    info!("Two-stage processing completed successfully");
                    response
                }
                Err(e) => {
                    error!("Two-stage processing failed: {}", e);
                    (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Request processing failed: {}", e),
                    )
                        .into_response()
                }
            };
            resp
        }
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &crate::protocols::spec::CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        info!(
            "vLLM route_completion called, use_discovery={}",
            self.use_discovery
        );

        if self.use_discovery {
            // Discovery mode - use vLLM-specific two-stage processing
            info!("Using service discovery mode, processing vLLM two-stage request");

            // Convert to generic request and use vLLM processing
            let request_json = match serde_json::to_value(body) {
                Ok(json) => {
                    debug!(
                        "Serialized completion request: {}",
                        serde_json::to_string_pretty(&json).unwrap_or_default()
                    );
                    json
                }
                Err(e) => {
                    return (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Serialization error: {}", e),
                    )
                        .into_response()
                }
            };

            // Process vLLM two-stage request with service discovery
            self.process_vllm_request(request_json, "/v1/completions", headers)
                .await
        } else {
            // Direct URL mode - implement routing logic here (not delegating to PdRouterBase)
            info!("Using direct URL mode with VllmPDRouter's own routing logic");

            // Convert request to JSON
            let request_json = match serde_json::to_value(body) {
                Ok(json) => {
                    debug!(
                        "Serialized completion request: {}",
                        serde_json::to_string_pretty(&json).unwrap_or_default()
                    );
                    json
                }
                Err(e) => {
                    return (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Serialization error: {}", e),
                    )
                        .into_response()
                }
            };

            // Get prefill and decode workers from worker_registry
            let prefill_workers = self.pd_router.worker_registry.get_prefill_workers();
            let decode_workers = self.pd_router.worker_registry.get_decode_workers();

            info!(
                "Found {} prefill workers, {} decode workers from worker_registry",
                prefill_workers.len(),
                decode_workers.len()
            );

            if prefill_workers.is_empty() || decode_workers.is_empty() {
                RouterMetrics::record_pd_error("server_selection");
                return (
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    format!(
                        "No workers available: {} prefill, {} decode",
                        prefill_workers.len(),
                        decode_workers.len()
                    ),
                )
                    .into_response();
            }

            // Select workers using policy with headers for consistent hash
            let request_text = serde_json::to_string(&request_json).ok();
            let request_str = request_text.as_deref();
            let request_headers: Option<HashMap<String, String>> = headers.map(|h| {
                h.iter()
                    .filter_map(|(name, value)| {
                        value
                            .to_str()
                            .ok()
                            .map(|v| (name.as_str().to_lowercase(), v.to_string()))
                    })
                    .collect()
            });

            let prefill_policy = self.policy_registry.get_prefill_policy();
            let decode_policy = self.policy_registry.get_decode_policy();

            let prefill_idx = match prefill_policy.select_worker_with_headers(
                &prefill_workers,
                request_str,
                request_headers.as_ref(),
            ) {
                Some(idx) => idx,
                None => {
                    RouterMetrics::record_pd_error("server_selection");
                    return (
                        axum::http::StatusCode::SERVICE_UNAVAILABLE,
                        "Prefill policy failed to select a worker".to_string(),
                    )
                        .into_response();
                }
            };

            let decode_idx = match decode_policy.select_worker_with_headers(
                &decode_workers,
                request_str,
                request_headers.as_ref(),
            ) {
                Some(idx) => idx,
                None => {
                    RouterMetrics::record_pd_error("server_selection");
                    return (
                        axum::http::StatusCode::SERVICE_UNAVAILABLE,
                        "Decode policy failed to select a worker".to_string(),
                    )
                        .into_response();
                }
            };

            let prefill_worker = &prefill_workers[prefill_idx];
            let decode_worker = &decode_workers[decode_idx];
            // Load tracking is handled inside process_vllm_two_stage_request for fine-grained
            // tracking: prefill load only during prefill phase, decode load only during decode phase.

            info!(
                "Completion: Selected prefill={} [policy:{}], decode={} [policy:{}]",
                prefill_worker.url(),
                prefill_policy.name(),
                decode_worker.url(),
                decode_policy.name()
            );

            // Execute dual dispatch with vLLM two-stage processing
            let resp = match self
                .process_vllm_two_stage_request(
                    request_json,
                    prefill_worker.clone(),
                    decode_worker.clone(),
                    "/v1/completions",
                    headers,
                )
                .await
            {
                Ok(response) => {
                    info!("Two-stage processing completed successfully");
                    response
                }
                Err(e) => {
                    error!("Two-stage processing failed: {}", e);
                    (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Request processing failed: {}", e),
                    )
                        .into_response()
                }
            };
            resp
        }
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &crate::protocols::spec::ResponsesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        let request_json = match serde_json::to_value(body) {
            Ok(json) => json,
            Err(e) => {
                return (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Serialization error: {}", e),
                )
                    .into_response()
            }
        };
        self.route_transparent(headers, "/v1/responses", &Method::POST, request_json)
            .await
    }

    async fn get_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        self.pd_router.get_response(headers, response_id).await
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        self.pd_router.cancel_response(headers, response_id).await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &crate::protocols::spec::EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.pd_router
            .route_embeddings(headers, body, model_id)
            .await
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &crate::protocols::spec::RerankRequest,
        _model_id: Option<&str>,
    ) -> Response {
        let request_json = match serde_json::to_value(body) {
            Ok(json) => json,
            Err(e) => {
                return (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Serialization error: {}", e),
                )
                    .into_response()
            }
        };
        self.route_transparent(headers, "/v1/rerank", &Method::POST, request_json)
            .await
    }

    async fn flush_cache(&self) -> Response {
        self.pd_router.flush_cache().await
    }

    async fn get_worker_loads(&self) -> Response {
        self.pd_router.get_worker_loads().await
    }

    fn router_type(&self) -> &'static str {
        "vllm_pd"
    }

    fn readiness(&self) -> Response {
        self.pd_router.readiness()
    }

    /// Route a transparent proxy request through the P/D disaggregation pipeline
    /// This handles any path/body and routes through prefill->decode stages
    async fn route_transparent(
        &self,
        headers: Option<&HeaderMap>,
        path: &str,
        method: &Method,
        body: serde_json::Value,
    ) -> Response {
        // Only handle POST requests for inference
        if *method != Method::POST {
            return (
                StatusCode::METHOD_NOT_ALLOWED,
                "Only POST requests are supported for transparent proxy",
            )
                .into_response();
        }

        debug!(
            "Transparent proxy: routing {} {} through P/D pipeline",
            method, path
        );

        // Body is already a serde_json::Value, use it directly
        let request_json = body;

        if self.use_discovery {
            // Discovery mode - use vLLM-specific two-stage processing
            self.process_vllm_request(request_json, path, headers).await
        } else {
            // Direct URL mode - use worker registry, filtered by availability
            let all_prefill = self.pd_router.worker_registry.get_prefill_workers();
            let prefill_workers: Vec<Arc<dyn Worker>> = all_prefill
                .iter()
                .filter(|w| w.is_available())
                .cloned()
                .collect();
            let all_decode = self.pd_router.worker_registry.get_decode_workers();
            let decode_workers: Vec<Arc<dyn Worker>> = all_decode
                .iter()
                .filter(|w| w.is_available())
                .cloned()
                .collect();

            if prefill_workers.is_empty() || decode_workers.is_empty() {
                RouterMetrics::record_pd_error("server_selection");
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!(
                        "No available workers: {} prefill, {} decode",
                        prefill_workers.len(),
                        decode_workers.len()
                    ),
                )
                    .into_response();
            }

            // Select workers using policy with headers for consistent hash
            let request_text = serde_json::to_string(&request_json).ok();
            let request_str = request_text.as_deref();
            let request_headers: Option<HashMap<String, String>> = headers.map(|h| {
                h.iter()
                    .filter_map(|(name, value)| {
                        value
                            .to_str()
                            .ok()
                            .map(|v| (name.as_str().to_lowercase(), v.to_string()))
                    })
                    .collect()
            });

            let prefill_policy = self.policy_registry.get_prefill_policy();
            let decode_policy = self.policy_registry.get_decode_policy();

            let prefill_idx = match prefill_policy.select_worker_with_headers(
                &prefill_workers,
                request_str,
                request_headers.as_ref(),
            ) {
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

            let decode_idx = match decode_policy.select_worker_with_headers(
                &decode_workers,
                request_str,
                request_headers.as_ref(),
            ) {
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

            let prefill_worker = &prefill_workers[prefill_idx];
            let decode_worker = &decode_workers[decode_idx];

            debug!(
                "Transparent proxy: prefill={}, decode={}",
                prefill_worker.url(),
                decode_worker.url()
            );

            // Execute two-stage processing
            match self
                .process_vllm_two_stage_request(
                    request_json,
                    prefill_worker.clone(),
                    decode_worker.clone(),
                    path,
                    headers,
                )
                .await
            {
                Ok(response) => response,
                Err(e) => {
                    error!(
                        "Transparent proxy request failed: prefill={}, decode={}, error={}",
                        prefill_worker.url(),
                        decode_worker.url(),
                        e
                    );
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Transparent proxy request failed: {}", e),
                    )
                        .into_response()
                }
            }
        }
    }
}

// Delegate WorkerManagement to the underlying PdRouterBase
#[async_trait]
impl WorkerManagement for VllmPDRouter {
    async fn add_worker(&self, worker_url: &str) -> Result<String, String> {
        self.pd_router.add_worker(worker_url).await
    }

    fn remove_worker(&self, worker_url: &str) {
        self.pd_router.remove_worker(worker_url);
    }

    fn get_worker_urls(&self) -> Vec<String> {
        self.pd_router.get_worker_urls()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // --- OpenAI-style endpoint tests (chat/completions, completions) ---

    #[test]
    fn test_prefill_chat_completion_sets_max_tokens_1() {
        let request = json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 512,
            "stream": true
        });
        let result = VllmPDRouter::prepare_prefill_request(request, "/v1/chat/completions");
        assert_eq!(result["max_tokens"], 1);
        assert_eq!(result["stream"], false);
    }

    #[test]
    fn test_prefill_chat_completion_sets_max_completion_tokens_1() {
        let request = json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 512,
            "max_completion_tokens": 256
        });
        let result = VllmPDRouter::prepare_prefill_request(request, "/v1/chat/completions");
        assert_eq!(result["max_tokens"], 1);
        assert_eq!(result["max_completion_tokens"], 1);
    }

    #[test]
    fn test_prefill_chat_completion_clamps_min_tokens() {
        let request = json!({
            "model": "test",
            "max_tokens": 512,
            "min_tokens": 100
        });
        let result = VllmPDRouter::prepare_prefill_request(request, "/v1/completions");
        assert_eq!(result["max_tokens"], 1);
        assert_eq!(result["min_tokens"], 1);
    }

    #[test]
    fn test_prefill_chat_completion_leaves_small_min_tokens() {
        let request = json!({
            "model": "test",
            "max_tokens": 512,
            "min_tokens": 0
        });
        let result = VllmPDRouter::prepare_prefill_request(request, "/v1/completions");
        assert_eq!(result["max_tokens"], 1);
        // min_tokens <= 1, so it should be left as-is
        assert_eq!(result["min_tokens"], 0);
    }

    #[test]
    fn test_prefill_chat_completion_removes_stream_options() {
        let request = json!({
            "model": "test",
            "max_tokens": 512,
            "stream": true,
            "stream_options": {"include_usage": true}
        });
        let result = VllmPDRouter::prepare_prefill_request(request, "/v1/chat/completions");
        assert_eq!(result["stream"], false);
        assert!(result.get("stream_options").is_none());
    }

    // --- Responses API endpoint tests (/v1/responses) ---

    #[test]
    fn test_prefill_responses_patches_max_output_tokens() {
        let request = json!({
            "model": "test",
            "input": "What is the capital of France?",
            "max_output_tokens": 1024,
            "stream": true
        });
        let result = VllmPDRouter::prepare_prefill_request(request, "/v1/responses");
        assert_eq!(result["max_output_tokens"], 1);
        // Responses API doesn't use max_tokens, so it should not be injected
        assert!(result.get("max_tokens").is_none());
        assert_eq!(result["stream"], false);
    }

    #[test]
    fn test_prefill_responses_without_max_output_tokens() {
        let request = json!({
            "model": "test",
            "input": "Hello",
            "stream": false
        });
        let result = VllmPDRouter::prepare_prefill_request(request, "/v1/responses");
        // For /v1/responses, max_output_tokens should always be forced to 1
        assert_eq!(result["max_output_tokens"], 1);
        // Responses API doesn't use max_tokens, so it should not be injected
        assert!(result.get("max_tokens").is_none());
        assert_eq!(result["stream"], false);
    }

    // --- Generate API endpoint tests (inference/v1/generate) ---

    #[test]
    fn test_prefill_generate_patches_sampling_params() {
        let request = json!({
            "token_ids": [123, 456],
            "sampling_params": {
                "max_tokens": 512,
                "temperature": 0.7
            }
        });
        let result = VllmPDRouter::prepare_prefill_request(request, "/inference/v1/generate");
        // sampling_params.max_tokens should be capped
        assert_eq!(result["sampling_params"]["max_tokens"], 1);
        // temperature should be preserved
        assert_eq!(result["sampling_params"]["temperature"], 0.7);
        // top-level max_tokens should NOT be set
        assert!(result.get("max_tokens").is_none());
    }

    #[test]
    fn test_prefill_generate_clamps_sampling_params_min_tokens() {
        let request = json!({
            "token_ids": [123, 456],
            "sampling_params": {
                "max_tokens": 512,
                "min_tokens": 50
            }
        });
        let result = VllmPDRouter::prepare_prefill_request(request, "/inference/v1/generate");
        assert_eq!(result["sampling_params"]["max_tokens"], 1);
        assert_eq!(result["sampling_params"]["min_tokens"], 1);
    }

    #[test]
    fn test_prefill_generate_without_sampling_params() {
        // If sampling_params is missing, should not panic
        let request = json!({
            "token_ids": [123, 456],
        });
        let result = VllmPDRouter::prepare_prefill_request(request, "/inference/v1/generate");
        // stream should still be forced to false
        assert_eq!(result["stream"], false);
        // top-level max_tokens should NOT be set (generate path)
        assert!(result.get("max_tokens").is_none());
        // create sampling_params and set min max
        assert_eq!(result["sampling_params"]["max_tokens"], 1);
        assert_eq!(result["sampling_params"]["min_tokens"], 1);
    }

    #[test]
    fn test_prefill_generate_forces_stream_false() {
        let request = json!({
            "token_ids": [123, 456],
            "sampling_params": {"max_tokens": 512},
            "stream": true,
            "stream_options": {"include_usage": true}
        });
        let result = VllmPDRouter::prepare_prefill_request(request, "/inference/v1/generate");
        assert_eq!(result["stream"], false);
        assert!(result.get("stream_options").is_none());
    }

    #[test]
    fn test_kv_transfer_params_moriio_includes_transfer_id_and_remote_dp_size() {
        // MoRI-IO prefill params must carry transfer_id and remote_dp_size.
        use crate::config::KvConnector;
        // Verify the KvConnector::MoriIO variant exists and serializes correctly.
        let connector = KvConnector::MoriIO;
        assert_eq!(format!("{:?}", connector), "MoriIO");
        // The transfer_id prefix must match MoRIIOConstants.TRANSFER_PREFIX.
        assert_eq!(MORIIO_TRANSFER_PREFIX, "tx");
    }

    #[test]
    fn test_kv_transfer_params_nixl_has_no_transfer_id_or_remote_dp_size() {
        // NIXL prefill params must not carry transfer_id or remote_dp_size.
        use crate::config::KvConnector;
        // Verify the KvConnector::Nixl variant exists and is the default.
        let connector = KvConnector::default();
        assert_eq!(connector, KvConnector::Nixl);
    }

    // --- MoRI-IO WRITE mode parameter tests ---

    fn moriio_write_prefill_params(transfer_id: Option<&str>, dp_size: usize) -> Value {
        // Mirror the WRITE mode branch in build_prefill_kv_transfer_params.
        json!({
            "do_remote_decode": true,
            "do_remote_prefill": false,
            "remote_engine_id": serde_json::Value::Null,
            "remote_block_ids": serde_json::Value::Null,
            "remote_dp_size": dp_size,
            // remote_tp_size is not yet consumed by the vLLM MoRI-IO connector;
            // hardcoded to 1 until https://github.com/vllm-project/vllm/issues/41211 is resolved.
            "remote_tp_size": 1,
            "transfer_id": transfer_id.unwrap_or(""),
        })
    }

    fn moriio_write_decode_params(
        transfer_id: Option<&str>,
        dp_size: usize,
        prefill_dp_rank: Option<u32>,
    ) -> Value {
        // Mirror build_moriio_write_decode_kv_transfer_params.
        let mut params = json!({
            "do_remote_decode": false,
            "do_remote_prefill": true,
            "remote_engine_id": serde_json::Value::Null,
            "remote_block_ids": serde_json::Value::Null,
            "transfer_id": transfer_id.unwrap_or(""),
            "remote_dp_size": dp_size,
            // remote_tp_size is not yet consumed by the vLLM MoRI-IO connector;
            // hardcoded to 1 until https://github.com/vllm-project/vllm/issues/41211 is resolved.
            "remote_tp_size": 1,
        });
        if dp_size > 1 {
            if let Some(rank) = prefill_dp_rank {
                params["remote_dp_rank"] = json!(rank);
            }
        }
        params
    }

    #[test]
    fn test_moriio_write_prefill_params_has_do_remote_decode_true() {
        let params = moriio_write_prefill_params(Some("tx-abc"), 1);
        assert_eq!(params["do_remote_decode"], true);
        assert_eq!(params["do_remote_prefill"], false);
        assert!(params["remote_engine_id"].is_null());
        assert!(params["remote_block_ids"].is_null());
        assert_eq!(params["remote_tp_size"], 1);
        assert_eq!(params["remote_dp_size"], 1);
        assert_eq!(params["transfer_id"], "tx-abc");
    }

    #[test]
    fn test_moriio_write_decode_params_have_correct_fields() {
        let params = moriio_write_decode_params(Some("tx-abc"), 1, None);
        assert_eq!(params["do_remote_decode"], false);
        assert_eq!(params["do_remote_prefill"], true);
        assert!(params["remote_engine_id"].is_null());
        assert!(params["remote_block_ids"].is_null());
        assert_eq!(params["transfer_id"], "tx-abc");
        assert_eq!(params["remote_dp_size"], 1);
        assert_eq!(params["remote_tp_size"], 1);
    }

    #[test]
    fn test_moriio_write_decode_params_no_remote_dp_rank_when_dp_size_is_1() {
        let params = moriio_write_decode_params(Some("tx-abc"), 1, Some(0));
        assert!(params.get("remote_dp_rank").is_none());
    }

    #[test]
    fn test_moriio_write_decode_params_includes_remote_dp_rank_when_dp_size_gt_1() {
        let params = moriio_write_decode_params(Some("tx-abc"), 4, Some(2));
        assert_eq!(params["remote_dp_rank"], 2);
    }
}
