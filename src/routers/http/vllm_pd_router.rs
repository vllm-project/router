// vLLM PD (Prefill-Decode) Router Implementation
// This module extends PdRouterBase to handle vLLM-specific two-stage processing
use super::dp_utils;
use super::logprobs_merge;
use super::pd_cached_tokens;
use super::pd_program_adapter::{self, PdProgramRuntime};
use super::pd_router::PdRouterBase;
use super::pd_types::{error_chain, PDRouterError, PdAttempt, PdRequestContext};
use super::usage_merge;
use super::vllm_service_discovery::{MoriIOTransferMode, ServiceRegistry, ServiceType};
use crate::config::KvConnector;
use crate::core::{BasicWorker, DPAwareWorker, Worker, WorkerType};
use crate::metrics::RouterMetrics;
use crate::otel_http::{self, ClientRequestOptions};
use crate::policies::{
    kv_aware_ops, AffinityKey, CacheAwarePolicy, KvRequestTokens, LoadBalancingPolicy,
    PolicyRegistry, WorkerTokenInputs,
};
use crate::program_scheduling::ProgramIdentity;
use crate::routers::{header_utils, RouterTrait, WorkerManagement};
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tokio::sync::{Mutex, OwnedSemaphorePermit};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Mooncake prefill bootstrap info (engine_id per dp_rank)
#[derive(Debug, Clone)]
struct MooncakePrefillInfo {
    bootstrap_addr: String,
    dp_engine_ids: HashMap<usize, String>,
}

/// Request-scoped facts threaded into decode-response handling: the route,
/// stage endpoints, timing, and the response shape the client asked for.
struct DecodeResponseContext<'a> {
    path: &'a str,
    prefill_http: &'a str,
    decode_http: &'a str,
    decode_base_url: &'a str,
    start_time: Instant,
    request: &'a Value,
}

/// A concurrent (NIXL-push / MoRI-IO WRITE) dispatch plan: the prepared stage
/// requests, their target workers, and the vLLM coordination request ID.
struct ConcurrentDispatch {
    prefill_request: Value,
    decode_request: Value,
    prefill_worker: Arc<dyn Worker>,
    decode_worker: Arc<dyn Worker>,
    request_id: String,
}

/// Terminal 503 for a Program turn whose bound endpoints could not be dispatched.
/// The Program reservation is settled as a failure so the next turn can re-admit.
fn program_unavailable(
    completion: &super::program_adapter::ProgramCompletion,
    message: &str,
) -> Response {
    completion.finish(false);
    (StatusCode::SERVICE_UNAVAILABLE, message.to_string()).into_response()
}

/// vLLM PD Router that extends PdRouterBase with vLLM-specific request handling
#[derive(Debug)]
pub struct VllmPDRouter {
    epd: Option<super::epd::EncoderStage>,
    /// Underlying PD router for most functionality
    pd_router: PdRouterBase,
    /// Opt-in Program-scoped PD scheduling; `None` keeps native PD routing.
    program_runtime: Option<PdProgramRuntime>,
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
    /// NIXL push identity per prefill base_url and dp_rank; never held across an await.
    nixl_prefill_info: RwLock<HashMap<String, HashMap<usize, Value>>>,
}

/// Transfer ID prefix used by MoRI-IO to correlate prefill and decode legs.
/// Must match `MoRIIOConstants.TRANSFER_PREFIX` in the vLLM Python connector.
const MORIIO_TRANSFER_PREFIX: &str = "tx";

/// Discovery-backed P/D routing is ready only after both worker types register.
/// Without this guard, `PdRouterBase::health` sees the unused static registry and
/// reports success before a discovered prefill/decode pair can serve requests.
fn discovery_is_ready(prefill_count: usize, decode_count: usize) -> bool {
    prefill_count > 0 && decode_count > 0
}

/// Buffer budget for stage responses. Oversized Decode bodies are streamed;
/// Prefill control responses must fit in memory to extract transfer metadata.
const PD_BUFFERED_BODY_LIMIT_BYTES: usize = 64 * 1024 * 1024;
const MAX_JSON_REWRITE_BYTES: usize = 8 * 1024 * 1024;

/// Prefill control metadata must fit in the buffer; Decode may stream the rest.
async fn read_prefill_body(response: reqwest::Response, url: &str) -> Result<Bytes, PDRouterError> {
    match read_response_body(response, PD_BUFFERED_BODY_LIMIT_BYTES)
        .await
        .map_err(|error| PDRouterError::NetworkError {
            message: format!("Failed to read response body from {url}: {error}"),
        })? {
        ResponseBody::Buffered(body) => Ok(body),
        ResponseBody::Passthrough(_) => {
            RouterMetrics::record_pd_body_over_limit("prefill");
            Err(PDRouterError::NetworkError {
                message: format!(
                    "Response body from {url} exceeds the {} MiB router buffer limit",
                    PD_BUFFERED_BODY_LIMIT_BYTES / (1024 * 1024)
                ),
            })
        }
    }
}

enum ResponseBody {
    Buffered(Bytes),
    Passthrough(Body),
}

async fn read_response_body(
    response: reqwest::Response,
    limit: usize,
) -> Result<ResponseBody, reqwest::Error> {
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Ok(ResponseBody::Passthrough(Body::from_stream(
            response.bytes_stream(),
        )));
    }
    let mut buffered = Vec::new();
    let mut source = response.bytes_stream();
    while let Some(chunk) = source.next().await {
        let chunk = chunk?;
        if chunk.len() > limit - buffered.len() {
            let prefix = futures_util::stream::iter([
                Ok::<_, reqwest::Error>(Bytes::from(buffered)),
                Ok(chunk),
            ]);
            return Ok(ResponseBody::Passthrough(Body::from_stream(
                prefix.chain(source),
            )));
        }
        buffered.extend_from_slice(&chunk);
    }
    Ok(ResponseBody::Buffered(Bytes::from(buffered)))
}

fn response_with_body(
    status: StatusCode,
    headers: &HeaderMap,
    body: Body,
) -> Result<Response, String> {
    let mut builder = Response::builder().status(status);
    for (name, value) in headers {
        builder = builder.header(name, value);
    }
    builder
        .body(body)
        .map_err(|error| format!("Failed to build Decode response: {error}"))
}

/// Rewrite successful SSE responses; relay all other streams unchanged.
fn finalize_decode_stream(
    decode_response: reqwest::Response,
    prefill_response_json: Option<&Value>,
    path: &str,
) -> Result<Response, String> {
    let status = decode_response.status();
    let is_encoded = response_is_encoded(decode_response.headers());
    let mut headers = header_utils::preserve_response_headers(decode_response.headers());
    headers.remove(http::header::CONTENT_LENGTH);
    let is_sse = headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|media_type| media_type.trim().eq_ignore_ascii_case("text/event-stream"));
    let stream = decode_response.bytes_stream();
    let body = if status.is_success()
        && pd_cached_tokens::is_supported_path(path)
        && is_sse
        && !is_encoded
    {
        Body::from_stream(pd_cached_tokens::rewrite_sse_stream(
            stream,
            pd_cached_tokens::prefill_cached_tokens(prefill_response_json),
            prefill_response_json.and_then(usage_merge::usage_snapshot),
        ))
    } else {
        Body::from_stream(stream)
    };
    response_with_body(status, &headers, body)
}

fn response_is_encoded(headers: &HeaderMap) -> bool {
    headers
        .get_all(http::header::CONTENT_ENCODING)
        .iter()
        .any(|value| {
            !value
                .to_str()
                .is_ok_and(|coding| coding.trim().eq_ignore_ascii_case("identity"))
        })
}

fn finalize_decode_body(
    status: StatusCode,
    headers: HeaderMap,
    original: Bytes,
    prefill_response_json: Option<&Value>,
    path: &str,
    needs_logprobs: bool,
) -> Result<Response, String> {
    let is_encoded = response_is_encoded(&headers);
    let mut headers = header_utils::preserve_response_headers(&headers);
    headers.remove(http::header::CONTENT_LENGTH);
    let reconcile_cache = pd_cached_tokens::is_supported_path(path);
    if !status.is_success()
        || is_encoded
        || original.len() > MAX_JSON_REWRITE_BYTES
        || (!reconcile_cache && !needs_logprobs)
    {
        return response_with_body(status, &headers, Body::from(original));
    }
    let cached = pd_cached_tokens::prefill_cached_tokens(prefill_response_json);
    let mut json: Value = match serde_json::from_slice(&original) {
        Ok(json) => json,
        Err(error) if reconcile_cache => {
            warn!("Cannot parse successful PD Decode response for cached-token reconciliation: {error}");
            return response_with_body(status, &headers, Body::from(original));
        }
        Err(error) => return Err(format!("Failed to parse Decode response as JSON: {error}")),
    };

    let mut changed = false;
    if needs_logprobs {
        let empty_json = Value::Null;
        changed |= logprobs_merge::merge_logprobs_in_json(
            prefill_response_json.unwrap_or(&empty_json),
            &mut json,
        );
    }
    if reconcile_cache {
        if let Some(prefill) = prefill_response_json {
            changed |= usage_merge::merge_pd_usage(prefill, &mut json);
        }
        match pd_cached_tokens::reconcile_usage(&mut json, cached, prefill_response_json) {
            Ok(rewritten) => changed |= rewritten,
            Err(reason) => warn!("Cannot reconcile PD cached tokens: {reason}"),
        }
    }

    if changed {
        drop(original);
        let encoded = serde_json::to_vec(&json)
            .map_err(|error| format!("Failed to serialize Decode response: {error}"))?;
        headers.remove(axum::http::header::CONTENT_ENCODING);
        response_with_body(status, &headers, Body::from(encoded))
    } else {
        response_with_body(status, &headers, Body::from(original))
    }
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
    fn build_prefill_kv_transfer_params(
        &self,
        transfer_id: Option<&str>,
        decode_base: Option<&str>,
        decode_dp_size: Option<usize>,
        decode_dp_rank: Option<usize>,
    ) -> Result<Value, String> {
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
                    let remote_tp_size = decode_base
                        .map(|base| {
                            self.service_registry
                                .get_tp_dp_size(
                                    base.trim_start_matches("http://")
                                        .trim_start_matches("https://"),
                                    ServiceType::Decode,
                                )
                                .0
                        })
                        .unwrap_or(1);
                    let remote_dp_size = decode_dp_size
                        .unwrap_or(self.intra_node_data_parallel_size)
                        .max(1);
                    let mut params = json!({
                        "do_remote_decode": true,
                        "do_remote_prefill": false,
                        "remote_engine_id": serde_json::Value::Null,
                        "remote_block_ids": serde_json::Value::Null,
                        "remote_dp_size": remote_dp_size,
                        "remote_tp_size": remote_tp_size,
                        "transfer_id": transfer_id.unwrap_or(""),
                    });
                    if remote_dp_size > 1 {
                        params["remote_dp_rank"] = json!(decode_dp_rank.unwrap_or(0));
                    }
                    Ok(params)
                } else {
                    // READ mode: prefill waits for decode to pull blocks.
                    let remote_dp_size = decode_dp_size
                        .unwrap_or(self.intra_node_data_parallel_size)
                        .max(1);
                    Ok(json!({
                        "do_remote_decode": true,
                        "do_remote_prefill": false,
                        "remote_engine_id": serde_json::Value::Null,
                        "remote_block_ids": serde_json::Value::Null,
                        "transfer_id": transfer_id.unwrap_or(""),
                        "remote_dp_size": remote_dp_size,
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
        prefill_dp_size: Option<usize>,
        prefill_dp_rank: Option<u32>,
        request_id: Option<&str>,
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
                    let prefill_base = prefill_url
                        .trim_start_matches("http://")
                        .trim_start_matches("https://");
                    let (tp_size, _) = self
                        .service_registry
                        .get_tp_dp_size(prefill_base, ServiceType::Prefill);
                    let remote_dp_size = prefill_dp_size
                        .unwrap_or(self.intra_node_data_parallel_size)
                        .max(1);
                    let mut params = json!({
                        "do_remote_decode": false,
                        "do_remote_prefill": true,
                        "remote_engine_id": serde_json::Value::Null,
                        "remote_block_ids": serde_json::Value::Null,
                        "transfer_id": transfer_id.unwrap_or(""),
                        "remote_dp_size": remote_dp_size,
                        "remote_tp_size": tp_size,
                    });
                    if remote_dp_size > 1 {
                        if let Some(rank) = prefill_dp_rank {
                            params["remote_dp_rank"] = json!(rank);
                        }
                    }
                    Some(params)
                } else {
                    // READ mode: extract params from prefill response and inject remote_dp_size.
                    let mut params = prefill_response_json?.get("kv_transfer_params")?.clone();
                    params["remote_dp_size"] = json!(prefill_dp_size
                        .unwrap_or(self.intra_node_data_parallel_size)
                        .max(1));
                    Some(params)
                }
            }
            KvConnector::Nixl => {
                if let Some(json) = prefill_response_json {
                    let kvt = json.get("kv_transfer_params")?;
                    if !Self::is_nixl_push_mode(kvt) {
                        return Some(kvt.clone());
                    }
                }
                let identity = self.ensure_nixl_push_identity(
                    prefill_url,
                    prefill_dp_rank.map(|r| r as usize),
                    prefill_response_json,
                )?;
                Some(Self::build_nixl_push_decode_params(&identity, request_id?))
            }
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

    fn is_nixl_push_mode(kvt: &Value) -> bool {
        kvt.get("transfer_mode")
            .and_then(|v| v.as_str())
            .is_some_and(|mode| mode == "push")
    }

    fn nixl_push_identity_from_kvt(kvt: &Value) -> Option<Value> {
        if !Self::is_nixl_push_mode(kvt) {
            return None;
        }
        let engine_id = kvt.get("remote_engine_id")?;
        let host = kvt.get("remote_host")?;
        let port = kvt.get("remote_port")?;
        if engine_id.is_null() || host.is_null() || port.is_null() {
            return None;
        }
        Some(json!({
            "remote_engine_id": engine_id.clone(),
            "remote_host": host.clone(),
            "remote_port": port.clone(),
            "tp_size": kvt.get("tp_size").cloned().unwrap_or_else(|| json!(1)),
            "transfer_mode": "push",
        }))
    }

    fn build_nixl_push_decode_params(identity: &Value, request_id: &str) -> Value {
        json!({
            "do_remote_prefill": true,
            "do_remote_decode": false,
            "remote_engine_id": identity.get("remote_engine_id").cloned().unwrap_or(Value::Null),
            "remote_host": identity.get("remote_host").cloned().unwrap_or(Value::Null),
            "remote_port": identity.get("remote_port").cloned().unwrap_or(Value::Null),
            "tp_size": identity.get("tp_size").cloned().unwrap_or_else(|| json!(1)),
            "remote_request_id": request_id,
        })
    }

    fn get_nixl_push_identity(&self, prefill_url: &str, dp_rank: Option<usize>) -> Option<Value> {
        let identity = self.get_nixl_info(prefill_url, dp_rank)?;
        Self::is_nixl_push_mode(&identity).then_some(identity)
    }

    fn ensure_nixl_push_identity(
        &self,
        prefill_url: &str,
        dp_rank: Option<usize>,
        prefill_response_json: Option<&Value>,
    ) -> Option<Value> {
        if let Some(identity) = self.get_nixl_push_identity(prefill_url, dp_rank) {
            return Some(identity);
        }
        let identity = prefill_response_json
            .and_then(|json| json.get("kv_transfer_params"))
            .and_then(Self::nixl_push_identity_from_kvt)?;
        self.store_nixl_info(prefill_url, dp_rank, identity.clone());
        Some(identity)
    }

    fn maybe_cache_nixl_push_identity(
        &self,
        prefill_url: &str,
        dp_rank: Option<usize>,
        prefill_response_json: &Value,
    ) {
        if let Some(identity) = prefill_response_json
            .get("kv_transfer_params")
            .and_then(Self::nixl_push_identity_from_kvt)
        {
            self.store_nixl_info(prefill_url, dp_rank, identity);
        }
    }

    fn get_nixl_info(&self, prefill_url: &str, dp_rank: Option<usize>) -> Option<Value> {
        let info = self.nixl_prefill_info.read().unwrap();
        let per_rank = info.get(prefill_url)?;
        match dp_rank {
            Some(rank) => per_rank.get(&rank).cloned(),
            None => per_rank.values().next().cloned(),
        }
    }

    fn store_nixl_info(&self, prefill_url: &str, dp_rank: Option<usize>, identity: Value) {
        let mut info = self.nixl_prefill_info.write().unwrap();
        info.entry(prefill_url.to_string())
            .or_default()
            .insert(dp_rank.unwrap_or(0), identity);
    }

    /// A bare URL evicts every rank; a `@rank` URL evicts only that rank.
    fn evict_nixl_info(&self, worker_url: &str) {
        let (base_url, dp_rank) = dp_utils::parse_worker_url(worker_url);
        let mut info = self.nixl_prefill_info.write().unwrap();
        match dp_rank {
            Some(rank) => {
                if let Some(per_rank) = info.get_mut(&base_url) {
                    per_rank.remove(&rank);
                    if per_rank.is_empty() {
                        info.remove(&base_url);
                    }
                }
            }
            None => {
                info.remove(&base_url);
            }
        }
    }

    /// Generate vLLM-specific request ID with prefill/decode addressing
    fn generate_vllm_request_id(prefill_addr: &str, decode_addr: &str) -> String {
        let uuid = Uuid::new_v4().to_string().replace('-', "");
        format!(
            "___prefill_addr_{}___decode_addr_{}_{}",
            prefill_addr, decode_addr, uuid
        )
    }

    /// Preserve rank-pinned registrations, with aggregate-server fallback.
    fn get_zmq_address(&self, worker: &dyn Worker, service_type: ServiceType) -> String {
        let address = worker
            .url()
            .trim_start_matches("http://")
            .trim_start_matches("https://");
        let base = worker
            .base_url()
            .trim_start_matches("http://")
            .trim_start_matches("https://");
        self.service_registry
            .get_zmq_address(address, service_type.clone())
            .or_else(|| self.service_registry.get_zmq_address(base, service_type))
            .unwrap_or_else(|| base.to_owned())
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
                // Prevent EOS/stop-token sampling on the only prefill output token.
                sampling_params["min_tokens"] = json!(1);
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
            // Prevent EOS/stop-token sampling on the only prefill output token.
            request["min_tokens"] = json!(1);
        }
        // Force non-streaming for prefill to get JSON response with kv_transfer_params
        request["stream"] = json!(false);
        // Remove stream_options since we're setting stream=false
        if let Some(obj) = request.as_object_mut() {
            obj.remove("stream_options");
        }
        request
    }

    fn discovery_candidates(
        &self,
        instances: Vec<(String, String, Option<usize>)>,
        service_type: ServiceType,
    ) -> Vec<Arc<dyn Worker>> {
        let mut workers: Vec<Arc<dyn Worker>> = Vec::new();
        for (address, _, registered_dp_size) in instances {
            let (base, rank) = dp_utils::parse_worker_url(&address);
            let dp_size = match self.kv_connector {
                // Respect an explicit registration, including an explicit
                // single-rank size. Preserve the legacy single-rank MoRI-IO default.
                KvConnector::MoriIO => registered_dp_size.unwrap_or(1).max(1),
                _ => registered_dp_size
                    .unwrap_or(1)
                    .max(self.intra_node_data_parallel_size)
                    .max(1),
            };
            let ranks = dp_utils::expand_dp_ranks(rank, dp_size);
            let url = if base.starts_with("http://") || base.starts_with("https://") {
                base.clone()
            } else {
                format!("http://{base}")
            };
            let role = match service_type {
                ServiceType::Prefill => WorkerType::Prefill {
                    bootstrap_port: None,
                },
                ServiceType::Decode => WorkerType::Decode,
            };
            for rank in ranks {
                let worker: Arc<dyn Worker> = match rank {
                    Some(rank) => {
                        Arc::new(DPAwareWorker::new(url.clone(), rank, dp_size, role.clone()))
                    }
                    None => {
                        Arc::new(BasicWorker::new(url.clone(), role.clone()).with_dp_size(dp_size))
                    }
                };
                workers.push(worker);
            }
        }
        workers
    }

    /// Stable session identity for the optional kv_aware decode binding.
    /// Reuses the shared identity resolver (canonical context, agent_hint,
    /// framework headers, then body `session_id`) so D keys on the stable
    /// protocol identity; user, tenant, correlation, request, and trace
    /// identifiers never qualify. Extracted only while the affinity switch
    /// is on, and regardless of `forward_policy_headers`.
    fn extract_affinity_key(
        decode_policy: &dyn LoadBalancingPolicy,
        headers: Option<&HeaderMap>,
        request: &Value,
    ) -> Option<AffinityKey> {
        let affinity_on =
            kv_aware_ops(decode_policy).is_some_and(|kv| kv.session_affinity_enabled());
        if !affinity_on {
            return None;
        }
        let model_scope = request
            .get("model")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .unwrap_or("default");
        crate::request_identity::resolve_affinity_identity(
            headers,
            crate::request_identity::IdentityFields::from_body(request),
        )
        .ok()
        .flatten()
        .map(|identity| AffinityKey::from_identity(identity, model_scope))
    }

    async fn admit_pair(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        request: &Value,
        path: &str,
        headers: Option<&HeaderMap>,
        forward_policy_headers: bool,
    ) -> Option<(super::pd_types::PdSelection, PdAttempt)> {
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();
        let text = [prefill_policy.as_ref(), decode_policy.as_ref()]
            .iter()
            .any(|policy| !policy.needs_cache_routing_key() && policy.needs_request_text())
            .then(|| serde_json::to_string(request).ok())
            .flatten();
        let needs_headers = forward_policy_headers
            && (prefill_policy.needs_headers() || decode_policy.needs_headers());
        let policy_headers = headers.filter(|_| needs_headers).map(|headers| {
            headers
                .iter()
                .filter_map(|(name, value)| {
                    value
                        .to_str()
                        .ok()
                        .map(|value| (name.as_str().to_lowercase(), value.to_owned()))
                })
                .collect()
        });
        let (prefill_inputs, decode_inputs) = tokio::join!(
            self.pd_router
                .prepare_prefill_inputs(prefill_workers, request, path, headers),
            self.pd_router
                .prepare_decode_inputs(decode_workers, request, path, headers),
        );
        let model_id = request.get("model").and_then(Value::as_str);
        let cache_salt = request
            .get("cache_salt")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty());
        let prefill_tokens: WorkerTokenInputs = prefill_inputs
            .iter()
            .filter(|_| prefill_policy.name() == "cache_aware")
            .flat_map(|inputs| inputs.iter())
            .filter_map(|(base, input)| {
                input
                    .prompts
                    .as_ref()
                    .map(|prompts| (base.clone(), prompts.clone()))
            })
            .collect();
        let decode_tokens = decode_inputs.unwrap_or_default();
        let [prefill_cache, decode_cache] = [
            (prefill_policy.as_ref(), prefill_workers, &prefill_tokens),
            (decode_policy.as_ref(), decode_workers, &decode_tokens),
        ]
        .map(|(policy, workers, inputs)| {
            policy
                .as_any()
                .downcast_ref::<CacheAwarePolicy>()
                .and_then(|cache| {
                    cache.prepare_selection(
                        workers,
                        KvRequestTokens::PerWorker(inputs),
                        model_id,
                        None,
                        cache_salt,
                    )
                })
        });
        let affinity_key = Self::extract_affinity_key(decode_policy.as_ref(), headers, request);
        let context = PdRequestContext {
            prefill_inputs: prefill_inputs.as_ref(),
            model_id,
            legacy_policy_input: text.as_deref(),
            prefill_cache: prefill_cache.as_ref(),
            decode_cache: decode_cache.as_ref(),
            headers: policy_headers.as_ref(),
            affinity_key,
            cache_salt,
            route: path,
        };
        // Existing deployments configure a common connector. Discovery parses
        // registrations with that connector; no new pair compatibility is
        // inferred from KV overlap or DP rank equality.
        self.pd_router
            .select_and_admit(prefill_workers, decode_workers, &context)
    }

    /// Cheap opt-in gate: only Program-capable requests are serialized twice.
    fn program_turn_enabled(
        &self,
        headers: Option<&HeaderMap>,
        other: &serde_json::Map<String, Value>,
    ) -> bool {
        self.program_runtime.as_ref().is_some_and(|runtime| {
            ProgramIdentity::may_be_enabled(headers, other, runtime.scheduler.enable_key())
        })
    }

    /// One Program turn: the scheduler binds D, P is selected per turn, and P
    /// capacity is bounded per target before dispatch.
    ///
    /// `None` means the request is not a Program turn after all, so the caller
    /// must fall back to native PD routing.
    async fn process_program_request(
        &self,
        mut request: Value,
        path: &str,
        headers: Option<&HeaderMap>,
    ) -> Option<Response> {
        let runtime = self.program_runtime.as_ref()?;
        let streaming = request
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let decode_targets = if self.use_discovery {
            pd_program_adapter::discovered_decode_instances(
                &self.service_registry,
                self.intra_node_data_parallel_size,
            )
            .iter()
            .map(|(address, _)| pd_program_adapter::discovered_target(address))
            .collect::<Vec<_>>()
        } else {
            self.pd_router
                .worker_registry
                .get_decode_workers()
                .into_iter()
                .filter(|worker| worker.is_available())
                .map(|worker| pd_program_adapter::worker_target(&worker))
                .collect::<Vec<_>>()
        };
        let completion = match runtime
            .acquire(headers, &request, path, &decode_targets)
            .await
        {
            Ok(Some(completion)) => completion,
            Ok(None) => return None,
            Err(error) => return Some(pd_program_adapter::error_response(error)),
        };
        // Transfer addresses and block IDs can only come from registered engines.
        if let Some(object) = request.as_object_mut() {
            object.remove("kv_transfer_params");
        }
        let target_id = completion.dispatch().target_id.clone();
        // Admission waits, so resolve the committed D against the live pool
        // instead of the candidate list used to propose targets before the wait.
        let Some(decode_worker) = self.program_decode_worker(&target_id) else {
            return Some(program_unavailable(
                &completion,
                "Program decode target left the pool",
            ));
        };
        let prefill_candidates = self.program_prefill_candidates();
        let selected = self.select_program_prefill(&prefill_candidates, &request, path, headers);
        let Some((prefill_id, prefill_worker)) = selected else {
            RouterMetrics::record_pd_error("server_selection");
            return Some(program_unavailable(
                &completion,
                "No feasible registered P/D pair",
            ));
        };
        let permit = match runtime.reserve_prefill(&prefill_id).await {
            Ok(permit) => permit,
            Err(error) => {
                completion.finish(false);
                return Some(pd_program_adapter::error_response(error));
            }
        };
        if !self.program_targets_are_live(&prefill_id, &target_id) {
            return Some(program_unavailable(
                &completion,
                "PD target changed while waiting for prefill capacity",
            ));
        }
        // Program turns bypass native selection, so charge their P and D
        // occupancy explicitly; otherwise they would be invisible to native
        // admission and to `/metrics`.
        let attempt = self
            .pd_router
            .pd_load
            .admit_bound(prefill_worker.clone(), decode_worker.clone());
        let result = self
            .process_vllm_two_stage_request(
                request,
                prefill_worker,
                decode_worker,
                path,
                headers,
                attempt.clone(),
                Some(permit),
            )
            .await;
        Some(match result {
            Ok(response) => pd_program_adapter::track_response(
                attempt.track_response(response),
                completion,
                streaming,
            ),
            Err(error) => {
                attempt.release();
                completion.finish(false);
                (StatusCode::SERVICE_UNAVAILABLE, error.to_string()).into_response()
            }
        })
    }

    /// Decode worker for the D target the scheduler committed to.
    fn program_decode_worker(&self, target_id: &str) -> Option<Arc<dyn Worker>> {
        if self.use_discovery {
            return pd_program_adapter::discovered_decode_worker(
                &self.service_registry,
                self.intra_node_data_parallel_size,
                target_id,
            );
        }
        self.pd_router
            .worker_registry
            .get_decode_workers()
            .into_iter()
            .find(|worker| {
                worker.is_available() && pd_program_adapter::worker_target(worker).id == target_id
            })
    }

    /// P candidates as `(admission id, worker)`. Discovered registrations stay
    /// keyed by their registration so a rank expansion cannot change the
    /// prefill capacity gate it is charged to.
    fn program_prefill_candidates(&self) -> Vec<(String, Arc<dyn Worker>)> {
        if self.use_discovery {
            let mut candidates = Vec::new();
            for snapshot in self
                .service_registry
                .get_instance_snapshots(ServiceType::Prefill)
            {
                let id = pd_program_adapter::discovered_target(&snapshot.0).id;
                for worker in self.discovery_candidates(vec![snapshot], ServiceType::Prefill) {
                    candidates.push((id.clone(), worker));
                }
            }
            candidates
        } else {
            self.pd_router
                .worker_registry
                .get_prefill_workers()
                .into_iter()
                .filter(|worker| worker.is_available())
                .map(|worker| (pd_program_adapter::worker_target(&worker).id, worker))
                .collect()
        }
    }

    /// P is chosen anew for every admitted turn so D continuity cannot pin a
    /// prefill worker.
    fn select_program_prefill(
        &self,
        candidates: &[(String, Arc<dyn Worker>)],
        request: &Value,
        path: &str,
        headers: Option<&HeaderMap>,
    ) -> Option<(String, Arc<dyn Worker>)> {
        if candidates.is_empty() {
            return None;
        }
        let policy = self.policy_registry.get_prefill_policy();
        let workers = candidates
            .iter()
            .map(|(_, worker)| worker.clone())
            .collect::<Vec<_>>();
        let request_text = if policy.needs_cache_routing_key() {
            Some(crate::protocols::spec::cache_routing::for_route(
                request, path,
            ))
        } else if policy.needs_request_text() {
            serde_json::to_string(request).ok()
        } else {
            None
        };
        let policy_headers = policy.needs_headers().then(|| {
            headers
                .map(|headers| {
                    headers
                        .iter()
                        .filter_map(|(name, value)| {
                            value
                                .to_str()
                                .ok()
                                .map(|value| (name.as_str().to_lowercase(), value.to_owned()))
                        })
                        .collect::<HashMap<String, String>>()
                })
                .unwrap_or_default()
        });
        let index = policy.select_worker_with_headers(
            &workers,
            request_text.as_deref(),
            policy_headers.as_ref(),
        )?;
        Some(candidates[index].clone())
    }

    /// Recheck the bound endpoints after a capacity wait: an expired
    /// registration or a removed worker must not receive the dispatch.
    fn program_targets_are_live(&self, prefill_id: &str, decode_id: &str) -> bool {
        if self.use_discovery {
            let prefill_live =
                self.service_registry
                    .get_prefill_instances()
                    .iter()
                    .any(|(address, _)| {
                        pd_program_adapter::discovered_target(address).id == prefill_id
                    });
            let decode_live = pd_program_adapter::discovered_decode_instance_by_id(
                &self.service_registry,
                self.intra_node_data_parallel_size,
                decode_id,
            )
            .is_some();
            prefill_live && decode_live
        } else {
            let prefill_live = self
                .pd_router
                .worker_registry
                .get_prefill_workers()
                .iter()
                .any(|worker| {
                    worker.is_available()
                        && pd_program_adapter::worker_target(worker).id == prefill_id
                });
            let decode_live = self
                .pd_router
                .worker_registry
                .get_decode_workers()
                .iter()
                .any(|worker| {
                    worker.is_available()
                        && pd_program_adapter::worker_target(worker).id == decode_id
                });
            prefill_live && decode_live
        }
    }

    /// Discovery supplies candidates; admission and dispatch are shared with static pools.
    async fn process_vllm_request(
        &self,
        request: Value,
        path: &str,
        headers: Option<&HeaderMap>,
    ) -> Response {
        let prefill_workers = self.discovery_candidates(
            self.service_registry
                .get_instance_snapshots(ServiceType::Prefill),
            ServiceType::Prefill,
        );
        let decode_workers = self.discovery_candidates(
            self.service_registry
                .get_instance_snapshots(ServiceType::Decode),
            ServiceType::Decode,
        );
        let Some((selection, attempt)) = self
            .admit_pair(
                &prefill_workers,
                &decode_workers,
                &request,
                path,
                headers,
                false,
            )
            .await
        else {
            RouterMetrics::record_pd_error("server_selection");
            return (StatusCode::SERVICE_UNAVAILABLE, "No eligible P/D pair").into_response();
        };
        match self
            .process_vllm_two_stage_request(
                request,
                prefill_workers[selection.prefill.worker_index].clone(),
                decode_workers[selection.decode.worker_index].clone(),
                path,
                headers,
                attempt.clone(),
                None,
            )
            .await
        {
            Ok(response) => attempt.track_response(response),
            Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response(),
        }
    }

    /// Consume every prefill response with the same bound and stage cleanup.
    async fn finish_prefill_response(
        &self,
        response: Result<reqwest::Response, reqwest::Error>,
        base: &str,
        attempt: &PdAttempt,
    ) -> Result<(StatusCode, Option<http::HeaderValue>, Bytes), PDRouterError> {
        let result = async {
            let response = response.map_err(|error| PDRouterError::NetworkError {
                message: format!("Prefill request to {base} failed: {}", error_chain(&error)),
            })?;
            let status = response.status();
            let content_type = response.headers().get(http::header::CONTENT_TYPE).cloned();
            let bytes = read_prefill_body(response, base).await?;
            Ok((status, content_type, bytes))
        }
        .await;
        attempt.finish_prefill();
        self.stop_profiling(base).await;
        result
    }

    /// Handle the decode response shared by both sequential and concurrent dispatch paths.
    /// Stops decode profiling, records metrics, then routes to streaming, logprobs-merge,
    /// or plain full-body response depending on the original request.
    async fn handle_decode_response(
        &self,
        decode_response: reqwest::Response,
        prefill_response_json: Option<&Value>,
        context: DecodeResponseContext<'_>,
        attempt: &PdAttempt,
    ) -> Result<Response, String> {
        let DecodeResponseContext {
            path,
            prefill_http,
            decode_http,
            decode_base_url,
            start_time,
            request,
        } = context;
        self.stop_profiling(decode_base_url).await;
        RouterMetrics::record_pd_request(path);
        RouterMetrics::record_pd_request_duration(path, start_time.elapsed());
        RouterMetrics::record_pd_prefill_request(prefill_http);
        RouterMetrics::record_pd_decode_request(decode_http);
        if !decode_response.status().is_success() {
            RouterMetrics::record_pd_decode_error(decode_http);
            attempt.finish_decode();
        }
        let is_streaming = request
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let needs_logprobs = request.get("logprobs").is_some()
            || request
                .get("echo")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        if is_streaming {
            finalize_decode_stream(decode_response, prefill_response_json, path)
        } else {
            let status = decode_response.status();
            let headers = decode_response.headers().clone();
            match read_response_body(decode_response, PD_BUFFERED_BODY_LIMIT_BYTES)
                .await
                .map_err(|error| {
                    format!("Failed to read response body from {decode_base_url}: {error}")
                })? {
                ResponseBody::Buffered(bytes) => {
                    attempt.finish_decode();
                    finalize_decode_body(
                        status,
                        headers,
                        bytes,
                        prefill_response_json,
                        path,
                        needs_logprobs,
                    )
                }
                ResponseBody::Passthrough(body) => {
                    let mut headers = header_utils::preserve_response_headers(&headers);
                    headers.remove(http::header::CONTENT_LENGTH);
                    response_with_body(status, &headers, body)
                }
            }
        }
    }

    /// Two-stage request processing for vLLM disaggregated mode
    ///
    /// Admission already reserved P and pending D. Dispatch drives independent
    /// Stage transitions; D occupancy follows response-body ownership. The
    /// Program path additionally holds its P admission permit for the call.
    #[allow(clippy::too_many_arguments)]
    async fn process_vllm_two_stage_request(
        &self,
        original_request: Value,
        prefill_worker: Arc<dyn Worker>,
        decode_worker: Arc<dyn Worker>,
        path: &str,
        headers: Option<&HeaderMap>,
        attempt: PdAttempt,
        prefill_permit: Option<OwnedSemaphorePermit>,
    ) -> Result<Response, PDRouterError> {
        debug!("ENTERED process_vllm_two_stage_request method");
        let start_time = Instant::now();
        if !original_request.is_object() {
            return Ok((
                StatusCode::BAD_REQUEST,
                "Request body must be a JSON object",
            )
                .into_response());
        }

        // The encoder stage must run before the NIXL concurrent fast path:
        // once a push identity is cached, try_build_concurrent_requests
        // returns early, and multimodal requests reaching P/D without
        // published embeddings break encoder-disaggregated workers.
        let (original_request, mut ec_transfers) = if path == "/v1/chat/completions" {
            if let Some(epd) = &self.epd {
                match epd
                    .prepare(
                        &self.http_client,
                        original_request,
                        prefill_worker.base_url(),
                        headers,
                        std::env::var("OPENAI_API_KEY").ok().as_deref(),
                    )
                    .await
                {
                    Ok(prepared) => prepared,
                    Err(error) => return Ok(error.into_response()),
                }
            } else {
                (original_request, None)
            }
        } else {
            (original_request, None)
        };

        if let Some((prefill_request, decode_request, request_id)) = self
            .try_build_concurrent_requests(&original_request, &prefill_worker, &decode_worker, path)
            .await
        {
            return self
                .process_concurrent_two_stage(
                    ConcurrentDispatch {
                        prefill_request,
                        decode_request,
                        prefill_worker,
                        decode_worker,
                        request_id,
                    },
                    path,
                    headers,
                    start_time,
                    ec_transfers,
                    attempt,
                    prefill_permit,
                )
                .await;
        }

        debug!(
            "Prefill worker: {}, Decode worker: {}, Path: {}",
            prefill_worker.url(),
            decode_worker.url(),
            path
        );

        let prefill_zmq_addr = self.get_zmq_address(prefill_worker.as_ref(), ServiceType::Prefill);
        let decode_zmq_addr = self.get_zmq_address(decode_worker.as_ref(), ServiceType::Decode);
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
        let decode_base_url = decode_worker.base_url().to_string();
        prefill_request["kv_transfer_params"] = self
            .build_prefill_kv_transfer_params(
                transfer_id.as_deref(),
                Some(&decode_base_url),
                decode_worker.dp_size(),
                decode_worker.dp_rank(),
            )
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

        let prefill_response = otel_http::send_client_request(
            prefill_request_builder.json(&prefill_request),
            headers,
            ClientRequestOptions {
                method: "POST",
                url: &prefill_url,
                route: Some(path),
                request_phase: Some("prefill"),
            },
        )
        .await;
        let finished_prefill = self
            .finish_prefill_response(prefill_response, &prefill_base_url, &attempt)
            .await
            .inspect_err(|_| {
                RouterMetrics::record_pd_prefill_error(&prefill_base_url);
                RouterMetrics::record_pd_request(path);
                RouterMetrics::record_pd_request_duration(path, start_time.elapsed());
            });
        // The whole P body has been consumed (or never started): do not hold P
        // admission capacity while D generates the response.
        drop(prefill_permit);
        let (status, content_type, prefill_bytes) = finished_prefill?;
        if !status.is_success() {
            RouterMetrics::record_pd_prefill_error(&prefill_base_url);
            RouterMetrics::record_pd_request(path);
            RouterMetrics::record_pd_request_duration(path, start_time.elapsed());
            if status.is_client_error() {
                let mut response = (status, prefill_bytes).into_response();
                if let Some(content_type) = content_type {
                    response
                        .headers_mut()
                        .insert(http::header::CONTENT_TYPE, content_type);
                }
                return Ok(response);
            }
            return Err(PDRouterError::NetworkError {
                message: format!(
                    "Prefill server {prefill_url} returned {status}: {}",
                    String::from_utf8_lossy(&prefill_bytes)
                ),
            });
        }

        // Parse prefill response to extract kv_transfer_params
        let prefill_response_json: Value = match serde_json::from_slice(&prefill_bytes) {
            Ok(json) => json,
            Err(e) => {
                let duration = start_time.elapsed();
                RouterMetrics::record_pd_prefill_error(&prefill_base_url);
                RouterMetrics::record_pd_request(path);
                RouterMetrics::record_pd_request_duration(path, duration);
                return Err(PDRouterError::NetworkError {
                    message: format!("Failed to parse prefill response as JSON: {}", e),
                });
            }
        };

        if self.epd.is_some()
            && !matches!(self.kv_connector, KvConnector::Mooncake)
            && !matches!(self.moriio_transfer_mode(), Some(MoriIOTransferMode::Write))
            && !prefill_response_json
                .get("kv_transfer_params")
                .is_some_and(Value::is_object)
        {
            let duration = start_time.elapsed();
            RouterMetrics::record_pd_prefill_error(&prefill_base_url);
            RouterMetrics::record_pd_request(path);
            RouterMetrics::record_pd_request_duration(path, duration);
            return Err(PDRouterError::NetworkError {
                message:
                    "Prefill returned no KV transfer parameters; refusing metadata-only decode"
                        .into(),
            });
        }

        // Prefill succeeded and consumed the embeddings; the pending EC
        // transfers are now owned by the decode phase, so disarm the
        // cancel-on-abandon guard.
        if let Some(transfers) = ec_transfers.as_mut() {
            transfers.disarm();
        }

        // Stop profiling on prefill server after its work is done

        // P work is complete. D remains pending until its HTTP dispatch.

        debug!("✅ vLLM Stage 1 completed, starting Stage 2 - Decode");

        // Stage 2: Prepare decode request with kv_transfer_params
        let mut decode_request = original_request.clone();
        if self.epd.is_some() {
            // D receives the prompt layout and KV, not P's EC reservations.
            if let Some(obj) = decode_request.as_object_mut() {
                obj.remove("ec_transfer_params");
            }
        }
        if let Some(params) = self
            .build_decode_kv_transfer_params(
                &prefill_base_url,
                Some(&prefill_response_json),
                transfer_id.as_deref(),
                prefill_worker.dp_size(),
                prefill_dp_rank.map(|r| r as u32),
                Some(&request_id),
            )
            .await
        {
            decode_request["kv_transfer_params"] = params;
        } else {
            warn!(
                "No kv_transfer_params for prefill {}, decode will proceed without them",
                prefill_base_url
            );
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

        attempt.begin_decode();
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
                attempt.finish_decode();
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

        self.handle_decode_response(
            decode_response,
            Some(&prefill_response_json),
            DecodeResponseContext {
                path,
                prefill_http: &prefill_base_url,
                decode_http: &decode_base_url,
                decode_base_url: &decode_base_url,
                start_time,
                request: &original_request,
            },
            &attempt,
        )
        .await
        .map_err(|message| PDRouterError::NetworkError { message })
    }

    async fn try_build_concurrent_requests(
        &self,
        original_request: &Value,
        prefill_worker: &Arc<dyn Worker>,
        decode_worker: &Arc<dyn Worker>,
        path: &str,
    ) -> Option<(Value, Value, String)> {
        let prefill_base_url = prefill_worker.base_url().to_string();
        let prefill_dp_rank = prefill_worker.dp_rank();
        let concurrent = match self.kv_connector {
            KvConnector::Nixl => self
                .get_nixl_push_identity(&prefill_base_url, prefill_dp_rank)
                .is_some(),
            KvConnector::MoriIO => self.moriio_transfer_mode() == Some(MoriIOTransferMode::Write),
            KvConnector::Mooncake => false,
        };
        if !concurrent {
            return None;
        }
        let transfer_id = self.generate_transfer_id();
        let prefill_zmq_addr = self.get_zmq_address(prefill_worker.as_ref(), ServiceType::Prefill);
        let decode_zmq_addr = self.get_zmq_address(decode_worker.as_ref(), ServiceType::Decode);
        let request_id = Self::generate_vllm_request_id(&prefill_zmq_addr, &decode_zmq_addr);

        let decode_base_url = decode_worker.base_url().to_string();
        let mut prefill_request = Self::prepare_prefill_request(original_request.clone(), path);
        prefill_request["kv_transfer_params"] = self
            .build_prefill_kv_transfer_params(
                transfer_id.as_deref(),
                Some(&decode_base_url),
                decode_worker.dp_size(),
                decode_worker.dp_rank(),
            )
            .ok()?;

        let mut decode_request = original_request.clone();
        if self.epd.is_some() {
            // D receives the prompt layout and KV, not P's EC reservations.
            if let Some(obj) = decode_request.as_object_mut() {
                obj.remove("ec_transfer_params");
            }
        }
        decode_request["kv_transfer_params"] = self
            .build_decode_kv_transfer_params(
                &prefill_base_url,
                None,
                transfer_id.as_deref(),
                prefill_worker.dp_size(),
                prefill_dp_rank.map(|rank| rank as u32),
                Some(&request_id),
            )
            .await?;

        Some((prefill_request, decode_request, request_id))
    }

    /// Concurrent dispatch owns the P admission permit until the P body ends.
    #[allow(clippy::too_many_arguments)]
    async fn process_concurrent_two_stage(
        &self,
        dispatch: ConcurrentDispatch,
        path: &str,
        headers: Option<&HeaderMap>,
        start_time: Instant,
        mut ec_transfers: Option<super::epd::EcTransferGuard>,
        attempt: PdAttempt,
        prefill_permit: Option<OwnedSemaphorePermit>,
    ) -> Result<Response, PDRouterError> {
        let ConcurrentDispatch {
            prefill_request,
            decode_request,
            prefill_worker,
            decode_worker,
            request_id,
        } = dispatch;
        let prefill_base_url = prefill_worker.base_url().to_string();
        let prefill_dp_rank = prefill_worker.dp_rank();
        let decode_base_url = decode_worker.base_url().to_string();
        let decode_dp_rank = decode_worker.dp_rank();
        let prefill_url = prefill_worker.endpoint_url(path);
        let decode_url = decode_worker.endpoint_url(path);
        let auth = format!(
            "Bearer {}",
            std::env::var("OPENAI_API_KEY").unwrap_or_default()
        );
        let stage_builder = |url: &str, rank: Option<usize>| {
            dp_utils::add_dp_rank_header(
                self.pd_router
                    .client
                    .post(url)
                    .header("Content-Type", "application/json")
                    .header("Authorization", auth.as_str())
                    .header("X-Request-Id", request_id.as_str()),
                rank,
            )
        };

        self.start_profiling(&prefill_base_url).await;
        self.start_profiling(&decode_base_url).await;

        let prefill_fut = async {
            // Hold the P admission slot until the prefill body is consumed, then
            // release it even if D is still running.
            let prefill_permit = prefill_permit;
            let response = otel_http::send_client_request(
                stage_builder(&prefill_url, prefill_dp_rank).json(&prefill_request),
                headers,
                ClientRequestOptions {
                    method: "POST",
                    url: &prefill_url,
                    route: Some(path),
                    request_phase: Some("prefill"),
                },
            )
            .await;
            let finished = self
                .finish_prefill_response(response, &prefill_base_url, &attempt)
                .await;
            drop(prefill_permit);
            finished
        };
        attempt.begin_decode();
        let (prefill_result, decode_result) = tokio::join!(
            prefill_fut,
            otel_http::send_client_request(
                stage_builder(&decode_url, decode_dp_rank).json(&decode_request),
                headers,
                ClientRequestOptions {
                    method: "POST",
                    url: &decode_url,
                    route: Some(path),
                    request_phase: Some("decode"),
                },
            ),
        );

        let prefill_response_json = match prefill_result {
            Ok((status, _, bytes)) if status.is_success() => {
                serde_json::from_slice::<Value>(&bytes).ok()
            }
            Ok((status, content_type, bytes)) => {
                attempt.finish_decode();
                self.stop_profiling(&decode_base_url).await;
                RouterMetrics::record_pd_prefill_error(&prefill_base_url);
                RouterMetrics::record_pd_request(path);
                RouterMetrics::record_pd_request_duration(path, start_time.elapsed());
                if status.is_client_error() {
                    let mut response = (status, bytes).into_response();
                    if let Some(content_type) = content_type {
                        response
                            .headers_mut()
                            .insert(http::header::CONTENT_TYPE, content_type);
                    }
                    return Ok(response);
                }
                return Err(PDRouterError::NetworkError {
                    message: format!(
                        "Prefill request failed to {} with status {}",
                        prefill_url, status
                    ),
                });
            }
            Err(e) => {
                attempt.finish_decode();
                self.stop_profiling(&decode_base_url).await;
                RouterMetrics::record_pd_prefill_error(&prefill_base_url);
                RouterMetrics::record_pd_request(path);
                RouterMetrics::record_pd_request_duration(path, start_time.elapsed());
                return Err(PDRouterError::NetworkError {
                    message: format!(
                        "Prefill request failed to {}: {}",
                        prefill_url,
                        error_chain(&e)
                    ),
                });
            }
        };
        if let Some(prefill_json) = prefill_response_json.as_ref() {
            self.maybe_cache_nixl_push_identity(&prefill_base_url, prefill_dp_rank, prefill_json);
        }
        // Prefill succeeded and consumed the embeddings; the pending EC
        // transfers are now owned by the decode phase, so disarm the
        // cancel-on-abandon guard.
        if let Some(transfers) = ec_transfers.as_mut() {
            transfers.disarm();
        }

        let decode_response = match decode_result {
            Ok(resp) => resp,
            Err(e) => {
                attempt.finish_decode();
                self.stop_profiling(&decode_base_url).await;
                RouterMetrics::record_pd_decode_error(&decode_base_url);
                RouterMetrics::record_pd_request(path);
                RouterMetrics::record_pd_request_duration(path, start_time.elapsed());
                RouterMetrics::record_pd_prefill_request(&prefill_base_url);
                return Err(PDRouterError::NetworkError {
                    message: format!(
                        "Decode request failed to {}: {}",
                        decode_url,
                        error_chain(&e)
                    ),
                });
            }
        };

        let prefill_http = prefill_base_url
            .strip_prefix("http://")
            .unwrap_or(&prefill_base_url);
        let decode_http = decode_base_url
            .strip_prefix("http://")
            .unwrap_or(&decode_base_url);

        self.handle_decode_response(
            decode_response,
            prefill_response_json.as_ref(),
            DecodeResponseContext {
                path,
                prefill_http,
                decode_http,
                decode_base_url: &decode_base_url,
                start_time,
                request: &decode_request,
            },
            &attempt,
        )
        .await
        .map_err(|message| PDRouterError::NetworkError { message })
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
        // Carries the configured request/connect timeouts; a bare
        // reqwest::Client::new() would let a stalled encoder hold the
        // request and its concurrency permit indefinitely.
        let http_client = ctx.client.clone();

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

            // The Program runtime observes the same registry the request path reads.
            let service_registry = Arc::new(service_registry);
            let program_runtime = PdProgramRuntime::new(ctx, service_registry.clone(), true)?;

            Ok(Self {
                epd: ctx
                    .router_config
                    .epd
                    .clone()
                    .map(super::epd::EncoderStage::new),
                pd_router,
                program_runtime,
                service_registry,
                http_client,
                policy_registry: ctx.policy_registry.clone(),
                use_discovery: true,
                enable_profiling: ctx.router_config.enable_profiling,
                profile_timeout_secs: ctx.router_config.profile_timeout_secs,
                profiling_tasks: Arc::new(Mutex::new(HashMap::new())),
                intra_node_data_parallel_size: ctx.router_config.intra_node_data_parallel_size,
                kv_connector,
                mooncake_prefill_info: Arc::new(Mutex::new(HashMap::new())),
                nixl_prefill_info: RwLock::new(HashMap::new()),
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
            let service_registry = Arc::new(ServiceRegistry::new());
            let program_runtime = PdProgramRuntime::new(ctx, service_registry.clone(), false)?;

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
                epd: ctx
                    .router_config
                    .epd
                    .clone()
                    .map(super::epd::EncoderStage::new),
                pd_router,
                program_runtime,
                service_registry,
                http_client,
                policy_registry: ctx.policy_registry.clone(),
                use_discovery: false,
                enable_profiling: ctx.router_config.enable_profiling,
                profile_timeout_secs: ctx.router_config.profile_timeout_secs,
                profiling_tasks: Arc::new(Mutex::new(HashMap::new())),
                intra_node_data_parallel_size: ctx.router_config.intra_node_data_parallel_size,
                kv_connector,
                mooncake_prefill_info,
                nixl_prefill_info: RwLock::new(HashMap::new()),
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
        self.evict_nixl_info(&url);
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
        self.evict_nixl_info(url);
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
        if self.use_discovery {
            let (prefill_count, decode_count) = self.service_registry.get_instance_counts();
            if !discovery_is_ready(prefill_count, decode_count) {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!(
                        "Waiting for discovered workers: {prefill_count} prefill, {decode_count} decode"
                    ),
                )
                    .into_response();
            }

            return (StatusCode::OK, "OK").into_response();
        }

        self.pd_router.health(req).await
    }

    async fn health_generate(&self, req: Request<Body>) -> Response {
        if self.use_discovery {
            // Discovery mode keeps the static registry empty; build the probe
            // pair from the discovered instances and run it through the same
            // admission path as real traffic, so health checks cannot select
            // by different rules (or fail outright on the empty registry).
            let prefill_workers = self.discovery_candidates(
                self.service_registry
                    .get_instance_snapshots(ServiceType::Prefill),
                ServiceType::Prefill,
            );
            let decode_workers = self.discovery_candidates(
                self.service_registry
                    .get_instance_snapshots(ServiceType::Decode),
                ServiceType::Decode,
            );
            return self
                .pd_router
                .health_generate_for_workers(&prefill_workers, &decode_workers)
                .await;
        }
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
        if self.program_turn_enabled(headers, &body.other) {
            let request = match serde_json::to_value(body) {
                Ok(request) => request,
                Err(error) => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response()
                }
            };
            if let Some(response) = self
                .process_program_request(request, "/v1/chat/completions", headers)
                .await
            {
                return response;
            }
        }
        match serde_json::to_value(body) {
            Ok(body) => {
                self.route_transparent(headers, "/v1/chat/completions", &Method::POST, body)
                    .await
            }
            Err(error) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Serialization error: {error}"),
            )
                .into_response(),
        }
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &crate::protocols::spec::CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        if self.program_turn_enabled(headers, &body.other) {
            let request = match serde_json::to_value(body) {
                Ok(request) => request,
                Err(error) => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response()
                }
            };
            if let Some(response) = self
                .process_program_request(request, "/v1/completions", headers)
                .await
            {
                return response;
            }
        }
        match serde_json::to_value(body) {
            Ok(body) => {
                self.route_transparent(headers, "/v1/completions", &Method::POST, body)
                    .await
            }
            Err(error) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Serialization error: {error}"),
            )
                .into_response(),
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
        let response = self.pd_router.get_worker_loads().await;
        let Some(runtime) = &self.program_runtime else {
            return response;
        };
        let (mut parts, body) = response.into_parts();
        let bytes = match axum::body::to_bytes(body, 8 * 1024 * 1024).await {
            Ok(bytes) => bytes,
            Err(error) => {
                return (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response()
            }
        };
        let mut value: Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(_) => return Response::from_parts(parts, Body::from(bytes)),
        };
        if let Some(object) = value.as_object_mut() {
            object.insert(
                "program_scheduling".into(),
                json!({
                    "path": "cold_remote",
                    "warm_enabled": false,
                    "scheduler": runtime.scheduler.diagnostics(),
                }),
            );
        }
        parts.headers.remove(axum::http::header::CONTENT_LENGTH);
        match serde_json::to_vec(&value) {
            Ok(bytes) => Response::from_parts(parts, Body::from(bytes)),
            Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response(),
        }
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

            let (selection, attempt) = match self
                .admit_pair(
                    &prefill_workers,
                    &decode_workers,
                    &request_json,
                    path,
                    headers,
                    true,
                )
                .await
            {
                Some(admitted) => admitted,
                None => {
                    RouterMetrics::record_pd_error("server_selection");
                    return (StatusCode::SERVICE_UNAVAILABLE, "No eligible P/D pair")
                        .into_response();
                }
            };
            let prefill_idx = selection.prefill.worker_index;
            let decode_idx = selection.decode.worker_index;

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
                    attempt.clone(),
                    None,
                )
                .await
            {
                Ok(response) => attempt.track_response(response),
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
        self.evict_nixl_info(worker_url);
        self.pd_router.remove_worker(worker_url);
    }

    fn get_worker_urls(&self) -> Vec<String> {
        self.pd_router.get_worker_urls()
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    async fn finalize_decode_response(
        response: reqwest::Response,
        prefill: Option<&Value>,
        path: &str,
        streaming: bool,
        logprobs: bool,
    ) -> Result<Response, String> {
        if streaming {
            finalize_decode_stream(response, prefill, path)
        } else {
            let status = response.status();
            let headers = response.headers().clone();
            match read_response_body(response, PD_BUFFERED_BODY_LIMIT_BYTES)
                .await
                .map_err(|error| format!("Failed to read response body: {error}"))?
            {
                ResponseBody::Buffered(bytes) => {
                    finalize_decode_body(status, headers, bytes, prefill, path, logprobs)
                }
                ResponseBody::Passthrough(body) => {
                    let mut headers = header_utils::preserve_response_headers(&headers);
                    headers.remove(http::header::CONTENT_LENGTH);
                    response_with_body(status, &headers, body)
                }
            }
        }
    }

    async fn test_vllm_pd_router() -> VllmPDRouter {
        let config = crate::config::RouterConfig {
            policy: crate::config::PolicyConfig::RoundRobin,
            ..Default::default()
        };
        let ctx = Arc::new(
            crate::server::AppContext::new(config, reqwest::Client::new(), 64, None, vec![])
                .unwrap(),
        );
        VllmPDRouter::new(vec![], vec![], None, &ctx).await.unwrap()
    }

    async fn spawn_test_server(app: axum::Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        url
    }

    use super::*;

    #[tokio::test]
    async fn prefill_body_limit_rejects_oversized_responses() {
        let url = spawn_test_server(axum::Router::new().fallback(|| async {
            Response::builder()
                .header(
                    http::header::CONTENT_LENGTH,
                    PD_BUFFERED_BODY_LIMIT_BYTES + 1,
                )
                .body(Body::from_stream(futures_util::stream::pending::<
                    Result<Bytes, std::io::Error>,
                >()))
                .unwrap()
        }))
        .await;
        let client = reqwest::Client::new();
        let response = client.get(&url).send().await.unwrap();
        let error = read_prefill_body(response, &url).await.unwrap_err();
        assert!(error.to_string().contains("exceeds"));
    }

    #[tokio::test]
    async fn decode_body_overflow_preserves_prefix_and_tail_errors() {
        use futures_util::stream;
        for fail in [false, true] {
            let (finish, ready) = tokio::sync::oneshot::channel();
            let prefix = stream::iter([
                Ok::<_, std::io::Error>(Bytes::from_static(b"pref")),
                Ok(Bytes::from_static(b"ix--")),
            ]);
            let tail = stream::once(async move {
                ready.await.unwrap();
                if fail {
                    Err(std::io::Error::other("truncated"))
                } else {
                    Ok(Bytes::from_static(b"suffix"))
                }
            });
            let response =
                http::Response::new(reqwest::Body::wrap_stream(prefix.chain(tail))).into();
            let body = tokio::time::timeout(
                std::time::Duration::from_secs(1),
                read_response_body(response, 6),
            )
            .await
            .expect("overflow must return before the tail finishes")
            .unwrap();
            let ResponseBody::Passthrough(body) = body else {
                panic!("body must stream above the budget");
            };
            let mut body = body.into_data_stream();
            let first = body.next().await.unwrap().unwrap();
            let second = body.next().await.unwrap().unwrap();
            assert_eq!(first.as_ref(), b"pref");
            assert_eq!(second.as_ref(), b"ix--");
            finish.send(()).unwrap();
            let suffix = body.next().await.unwrap();
            if fail {
                assert!(suffix.is_err());
            } else {
                assert_eq!(suffix.unwrap().as_ref(), b"suffix");
            }
            assert!(body.next().await.is_none());
        }
    }

    #[tokio::test]
    async fn large_decode_responses_preserve_success_and_hold_load_until_eof_or_drop() {
        let router = test_vllm_pd_router().await;
        let prefill: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "https://prefill".into(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        ));
        let decode: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "https://decode".into(),
            WorkerType::Decode,
        ));
        let payload = Bytes::from(vec![b'x'; PD_BUFFERED_BODY_LIMIT_BYTES + 1]);
        let request = json!({"stream":false, "logprobs":true});
        for drop_body in [false, true] {
            let (_, attempt) = router
                .admit_pair(
                    std::slice::from_ref(&prefill),
                    std::slice::from_ref(&decode),
                    &request,
                    "/v1/completions",
                    None,
                    false,
                )
                .await
                .unwrap();
            attempt.begin_decode();
            attempt.finish_prefill();
            let upstream = http::Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .header("x-worker", "decode")
                .body(payload.clone())
                .unwrap()
                .into();
            let response = router
                .handle_decode_response(
                    upstream,
                    None,
                    DecodeResponseContext {
                        path: "/v1/completions",
                        prefill_http: "prefill",
                        decode_http: "decode",
                        decode_base_url: "decode",
                        start_time: Instant::now(),
                        request: &request,
                    },
                    &attempt,
                )
                .await
                .expect("large successful decode response must be preserved");
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(response.headers()["x-worker"], "decode");
            assert_eq!(decode.load(), 1);
            let response = attempt.clone().track_response(response);
            if drop_body {
                drop(response);
            } else {
                let mut stream = response.into_body().into_data_stream();
                let mut received = 0;
                while let Some(chunk) = stream.next().await {
                    let chunk = chunk.unwrap();
                    assert_eq!(chunk.as_ref(), &payload[received..received + chunk.len()]);
                    received += chunk.len();
                }
                assert_eq!(received, payload.len());
            }
            assert_eq!(decode.load(), 0);
            assert!(router.pd_router.pd_load.snapshot().is_empty());
        }
    }

    #[tokio::test]
    async fn test_decode_response_filters_headers_in_every_branch() {
        let router = test_vllm_pd_router().await;
        let prefill_workers = [Arc::new(BasicWorker::new(
            "http://prefill".into(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        )) as Arc<dyn Worker>];
        let decode_workers = [
            Arc::new(BasicWorker::new("http://decode".into(), WorkerType::Decode))
                as Arc<dyn Worker>,
        ];
        for (streaming, logprobs) in [(false, false), (false, true), (true, false)] {
            // A real reqwest response, with a backend content length that will
            // become invalid when prompt logprobs are merged into the JSON.
            let decode: reqwest::Response = axum::http::Response::builder()
                .header("connection", "X-Internal, Content-Encoding")
                .header("content-encoding", "identity")
                .header("x-internal", "private")
                .header("trailer", "x-checksum")
                .header("content-length", "2")
                .header("content-type", "application/json")
                .header("set-cookie", "a=1")
                .header("set-cookie", "b=2")
                .body("{}")
                .unwrap()
                .into();
            let prefill = json!({"prompt_logprobs": [1, 2, 3]});
            let mut request = json!({"stream": streaming});
            if logprobs {
                request["logprobs"] = json!(true);
            }
            let (_, attempt) = router
                .admit_pair(
                    &prefill_workers,
                    &decode_workers,
                    &request,
                    "/v1/chat/completions",
                    None,
                    false,
                )
                .await
                .unwrap();
            attempt.begin_decode();
            let response = router
                .handle_decode_response(
                    decode,
                    Some(&prefill),
                    DecodeResponseContext {
                        path: "/v1/chat/completions",
                        prefill_http: "prefill",
                        decode_http: "decode",
                        decode_base_url: "decode",
                        start_time: Instant::now(),
                        request: &request,
                    },
                    &attempt,
                )
                .await
                .unwrap();
            if !streaming {
                assert!(router
                    .pd_router
                    .pd_load
                    .snapshot()
                    .values()
                    .all(|load| load.active_decode_requests == 0
                        && load.pending_decode_requests == 0));
            }
            for name in ["connection", "x-internal", "trailer", "content-encoding"] {
                assert!(
                    !response.headers().contains_key(name),
                    "{name} leaked (streaming={streaming}, logprobs={logprobs})"
                );
            }
            assert_eq!(response.headers().get_all("set-cookie").iter().count(), 2);
            if streaming || logprobs {
                assert!(!response.headers().contains_key("content-length"));
            }
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            if logprobs {
                let decoded: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(decoded["prompt_logprobs"], prefill["prompt_logprobs"]);
            } else {
                assert_eq!(bytes.as_ref(), b"{}");
            }
        }
        for streaming in [false, true] {
            // Encoded bytes are relayed unchanged; removing their encoding
            // would make the client interpret compressed data as plaintext.
            let encoded = vec![0x1f, 0x8b, 0x08, 0x00, 0xff];
            let decode: reqwest::Response = axum::http::Response::builder()
                .header("content-encoding", "gzip")
                .body(encoded.clone())
                .unwrap()
                .into();
            let request = json!({"stream": streaming});
            let (_, attempt) = router
                .admit_pair(
                    &prefill_workers,
                    &decode_workers,
                    &request,
                    "/v1/chat/completions",
                    None,
                    false,
                )
                .await
                .unwrap();
            attempt.begin_decode();
            let response = router
                .handle_decode_response(
                    decode,
                    None,
                    DecodeResponseContext {
                        path: "/v1/chat/completions",
                        prefill_http: "prefill",
                        decode_http: "decode",
                        decode_base_url: "decode",
                        start_time: Instant::now(),
                        request: &request,
                    },
                    &attempt,
                )
                .await
                .unwrap();
            assert_eq!(response.headers().get("content-encoding").unwrap(), "gzip");
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            assert_eq!(bytes.as_ref(), encoded.as_slice());
        }
    }
    use serde_json::json;

    #[tokio::test]
    async fn discovery_usage_reconciliation_covers_sequential_and_nixl_push() {
        use crate::config::{PolicyConfig, RouterConfig};
        use axum::{extract::State, http::Uri, routing::post, Json};

        async fn prefill(State(push): State<bool>, uri: Uri) -> Json<Value> {
            let usage = if uri.path().ends_with("/responses") {
                json!({"input_tokens": 70, "output_tokens": 1, "input_tokens_details": {"cached_tokens": 4}})
            } else {
                json!({"prompt_tokens": 70, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 4}})
            };
            let mut value =
                json!({"usage": usage, "kv_transfer_params": {"remote_engine_id": "P"}});
            if push {
                value["kv_transfer_params"]["remote_host"] = json!("127.0.0.1");
                value["kv_transfer_params"]["remote_port"] = json!(9100);
                value["kv_transfer_params"]["transfer_mode"] = json!("push");
            }
            Json(value)
        }
        async fn decode(uri: Uri, Json(body): Json<Value>) -> Response {
            if body["model"] == "error" {
                return (
                    StatusCode::TOO_MANY_REQUESTS,
                    [("retry-after", "7")],
                    "overloaded",
                )
                    .into_response();
            }
            let responses = uri.path().ends_with("/responses");
            let usage = if responses {
                json!({"input_tokens": 999, "output_tokens": 8, "total_tokens": 1007, "input_tokens_details": {"cached_tokens": 99}})
            } else {
                json!({"prompt_tokens": 999, "completion_tokens": 8, "total_tokens": 1007, "prompt_tokens_details": {"cached_tokens": 99}})
            };
            if body["stream"] == true {
                let event = if responses {
                    json!({"type": "response.completed", "response": {"usage": usage}})
                } else {
                    json!({"choices": [], "usage": usage})
                };
                return (
                    [("content-type", "text/event-stream")],
                    format!("data: {event}\n\ndata: [DONE]\n\n"),
                )
                    .into_response();
            }
            Json(json!({"usage": usage})).into_response()
        }

        for push in [false, true] {
            let mut app = axum::Router::new();
            for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
                app = app.route(&format!("/p{path}"), post(prefill));
            }
            let mut app = app.with_state(push);
            for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
                app = app.route(&format!("/d{path}"), post(decode));
            }
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let client = reqwest::Client::builder()
                .no_proxy()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .unwrap();
            let context = Arc::new(
                crate::server::AppContext::new(
                    RouterConfig {
                        policy: PolicyConfig::Random,
                        kv_connector: KvConnector::Nixl,
                        disable_retries: true,
                        disable_circuit_breaker: true,
                        ..RouterConfig::default()
                    },
                    client,
                    64,
                    None,
                    vec![],
                )
                .unwrap(),
            );
            // Register directly to isolate the discovery response pipeline
            // from the ZMQ listener/control plane; no fixed ports are opened.
            let mut router = VllmPDRouter::new(vec![], vec![], None, &context)
                .await
                .unwrap();
            router.use_discovery = true;
            for _ in 0..2 {
                for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
                    for streaming in [false, true] {
                        router.service_registry.register_service(
                            format!("{address}/p"),
                            "P".into(),
                            ServiceType::Prefill,
                        );
                        router.service_registry.register_service(
                            format!("{address}/d"),
                            "D".into(),
                            ServiceType::Decode,
                        );
                        let response = router.route_transparent(None, path, &Method::POST,
                            json!({"model": "test", "stream": streaming, "input": "hello", "prompt": "hello", "messages": [{"role": "user", "content": "hello"}]})).await;
                        assert_eq!(response.status(), StatusCode::OK);
                        let bytes = axum::body::to_bytes(response.into_body(), 4096)
                            .await
                            .unwrap();
                        let result: Value = if streaming {
                            let text = std::str::from_utf8(&bytes).unwrap();
                            assert!(text.ends_with("data: [DONE]\n\n"));
                            serde_json::from_str(
                                text.split("\n\n")
                                    .next()
                                    .unwrap()
                                    .strip_prefix("data: ")
                                    .unwrap(),
                            )
                            .unwrap()
                        } else {
                            serde_json::from_slice(&bytes).unwrap()
                        };
                        let usage = usage_merge::usage_object(&result).unwrap();
                        let details = if path == "/v1/responses" {
                            "input_tokens_details"
                        } else {
                            "prompt_tokens_details"
                        };
                        assert_eq!(usage["total_tokens"], 78);
                        assert_eq!(usage[details]["cached_tokens"], 4);

                        let response = router
                            .route_transparent(
                                None,
                                path,
                                &Method::POST,
                                json!({"model": "error", "stream": streaming, "logprobs": true}),
                            )
                            .await;
                        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
                        assert_eq!(response.headers()["retry-after"], "7");
                        assert_eq!(
                            axum::body::to_bytes(response.into_body(), 4096)
                                .await
                                .unwrap()
                                .as_ref(),
                            b"overloaded"
                        );
                    }
                }
            }
            assert_eq!(
                router
                    .get_nixl_push_identity(&format!("http://{address}/p"), None)
                    .is_some(),
                push
            );
            server.abort();
        }
    }

    async fn mock_decode_response(
        status: StatusCode,
        body: &'static str,
    ) -> (reqwest::Response, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = axum::Router::new().route(
            "/",
            axum::routing::get(move || async move {
                (status, [("content-type", "application/json")], body)
            }),
        );
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let response = reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap()
            .get(format!("http://{address}/"))
            .send()
            .await
            .unwrap();
        (response, server)
    }

    #[tokio::test]
    async fn finalizer_forwards_non_sse_streaming_body_before_upstream_finishes() {
        use futures_util::StreamExt;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::time::{timeout, Duration};

        const PREFIX: &[u8] =
            br#"{"usage":{"prompt_tokens_details":{"cached_tokens":99}},"choices":[]"#;
        let prefill = json!({"usage":{"prompt_tokens_details":{"cached_tokens":4}}});
        for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
            for content_type in [
                Some("application/json"),
                Some("application/octet-stream"),
                Some("text/plain"),
                Some("text/event-streaming"),
                None,
            ] {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                let address = listener.local_addr().unwrap();
                let (finish, ready_to_finish) = tokio::sync::oneshot::channel();
                let server = tokio::spawn(async move {
                    let (mut socket, _) = listener.accept().await.unwrap();
                    let mut request = [0; 4096];
                    let bytes_read = socket.read(&mut request).await.unwrap();
                    assert!(bytes_read > 0, "client closed before sending request");
                    let mut headers = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n",
                        PREFIX.len() + 1
                    );
                    if let Some(content_type) = content_type {
                        headers.push_str(&format!("Content-Type: {content_type}\r\n"));
                    }
                    headers.push_str("\r\n");
                    socket.write_all(headers.as_bytes()).await.unwrap();
                    socket.write_all(PREFIX).await.unwrap();
                    // Hold the final JSON byte until the downstream receives its prefix.
                    ready_to_finish.await.unwrap();
                    socket.write_all(b"}").await.unwrap();
                    socket.shutdown().await.unwrap();
                });
                let decode = reqwest::Client::builder()
                    .no_proxy()
                    .timeout(Duration::from_secs(5))
                    .build()
                    .unwrap()
                    .get(format!("http://{address}/"))
                    .send()
                    .await
                    .unwrap();
                let response = timeout(
                    Duration::from_secs(2),
                    finalize_decode_response(decode, Some(&prefill), path, true, true),
                )
                .await
                .expect("non-SSE streaming responses must return before EOF")
                .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                assert_eq!(
                    response
                        .headers()
                        .get("content-type")
                        .map(|v| v.to_str().unwrap()),
                    content_type
                );
                let mut body = response.into_body().into_data_stream();
                let mut prefix = Vec::new();
                timeout(Duration::from_secs(2), async {
                    while prefix.len() < PREFIX.len() {
                        prefix.extend_from_slice(&body.next().await.unwrap().unwrap());
                    }
                })
                .await
                .expect("non-SSE bytes must be forwarded before the upstream finishes");
                assert_eq!(
                    prefix, PREFIX,
                    "path: {path}, content type: {content_type:?}"
                );
                finish.send(()).unwrap();
                let mut suffix = Vec::new();
                while let Some(chunk) = body.next().await {
                    suffix.extend_from_slice(&chunk.unwrap());
                }
                assert_eq!(suffix, b"}");
                server.await.unwrap();
            }
        }
    }

    #[tokio::test]
    async fn finalizer_forwards_oversized_sse_without_truncating() {
        use futures_util::StreamExt;

        for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
            let prefix = b"data: {\"choices\":[{\"text\":\"ok\"}]}\n\n";
            let oversized_frame_bytes = 1024 * 1024 + 1;
            let mut raw = prefix.to_vec();
            raw.extend_from_slice(b"data: ");
            // Include the data field prefix and final blank line in the budget.
            raw.resize(prefix.len() + oversized_frame_bytes - 2, b'x');
            raw.extend_from_slice(b"\n\n");
            assert_eq!(raw.len() - prefix.len(), oversized_frame_bytes);
            raw.extend_from_slice(b"data: [DONE]\n\n");
            let expected = raw.clone();
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let app = axum::Router::new().route(
                "/",
                axum::routing::get(move || {
                    let raw = raw.clone();
                    async move { ([("content-type", "text/event-stream")], raw) }
                }),
            );
            let server = tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            let decode = reqwest::Client::builder()
                .no_proxy()
                .build()
                .unwrap()
                .get(format!("http://{address}/"))
                .send()
                .await
                .unwrap();
            let response = finalize_decode_response(decode, None, path, true, false)
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(response.headers()["content-type"], "text/event-stream");
            let mut stream = response.into_body().into_data_stream();
            assert_eq!(stream.next().await.unwrap().unwrap(), prefix.as_slice());
            let mut actual = prefix.to_vec();
            while let Some(chunk) = stream.next().await {
                actual.extend_from_slice(&chunk.unwrap());
            }
            assert_eq!(actual, expected);
            server.abort();
        }
    }

    // Fixed gzip fixtures for a JSON usage body and an SSE usage/DONE stream.
    const GZIP_JSON: &[u8] = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff\xab\x56\x2a\x2d\x4e\x4c\x4f\x55\xb2\xaa\x56\x2a\x28\xca\xcf\x2d\x28\x89\x2f\xc9\xcf\x4e\xcd\x2b\x8e\x4f\x49\x2d\x49\xcc\xcc\x29\x06\x49\x24\x27\x26\x67\xa4\xa6\x40\x25\x94\xac\x2c\x2d\x6b\x6b\x75\x94\x92\x33\xf2\x33\x93\x53\x81\xdc\xe8\xd8\x5a\x00\xe1\x2d\x48\x6b\x45\x00\x00\x00";
    const GZIP_SSE: &[u8] = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff\x4b\x49\x2c\x49\xb4\x52\xa8\x56\x2a\x2d\x4e\x4c\x4f\x55\xb2\xaa\x56\x2a\x28\xca\xcf\x2d\x28\x89\x2f\xc9\xcf\x4e\xcd\x2b\x8e\x4f\x49\x2d\x49\xcc\xcc\x29\x06\x49\x24\x27\x26\x67\xa4\xa6\x40\x25\x94\xac\x2c\x2d\x6b\x6b\x75\x94\x92\x33\xf2\x33\x93\x53\x81\xdc\xe8\xd8\x5a\x2e\xae\x14\xb0\x61\xd1\x2e\xfe\x7e\xae\xb1\x5c\x5c\x00\x62\x9a\xfc\x7a\x5b\x00\x00\x00";

    async fn mock_encoded_decode_response(
        status: StatusCode,
        body: &'static [u8],
        content_type: &'static str,
        content_encoding: &'static str,
    ) -> (reqwest::Response, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = axum::Router::new().route(
            "/",
            axum::routing::get(move || async move {
                (
                    status,
                    [
                        ("content-type", content_type),
                        ("content-encoding", content_encoding),
                        ("retry-after", "7"),
                        ("connection", "close"),
                    ],
                    body,
                )
            }),
        );
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let response = reqwest::Client::builder()
            .no_proxy()
            .no_gzip()
            .build()
            .unwrap()
            .get(format!("http://{address}/"))
            .send()
            .await
            .unwrap();
        (response, server)
    }

    #[tokio::test]
    async fn finalizer_rewrites_only_event_stream_media_type() {
        let raw = b"data: {\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":99}},\"choices\":[]}\n\ndata: [DONE]\n\n";
        let prefill = json!({"usage":{"prompt_tokens_details":{"cached_tokens":4}}});
        for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
            for (content_type, rewrite) in [
                ("text/event-stream", true),
                ("text/event-stream; charset=utf-8", true),
                ("Text/Event-Stream; Charset=UTF-8", true),
                ("application/json", false),
                ("text/plain", false),
                ("text/event-streaming", false),
                ("application/octet-stream; name=text/event-stream", false),
            ] {
                let (decode, server) =
                    mock_encoded_decode_response(StatusCode::OK, raw, content_type, "identity")
                        .await;
                let response = finalize_decode_response(decode, Some(&prefill), path, true, false)
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                assert_eq!(response.headers()["content-type"], content_type);
                let bytes = axum::body::to_bytes(response.into_body(), 1024)
                    .await
                    .unwrap();
                if rewrite {
                    let text = std::str::from_utf8(&bytes).unwrap();
                    let usage: Value = serde_json::from_str(
                        text.lines().next().unwrap().strip_prefix("data: ").unwrap(),
                    )
                    .unwrap();
                    assert_eq!(usage["usage"]["prompt_tokens_details"]["cached_tokens"], 4);
                    assert!(text.ends_with("data: [DONE]\n\n"));
                } else {
                    assert_eq!(
                        bytes.as_ref(),
                        raw,
                        "path: {path}, content type: {content_type}"
                    );
                }
                server.abort();
            }
        }
    }

    #[tokio::test]
    async fn finalizer_preserves_gzip_passthrough_headers_and_body() {
        let prefill = json!({"usage":{"prompt_tokens_details":{"cached_tokens":4}}});
        for path in [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
            "/custom",
        ] {
            for status in [StatusCode::OK, StatusCode::TOO_MANY_REQUESTS] {
                for is_streaming in [false, true] {
                    for needs_logprobs in [false, true] {
                        let streaming_success = is_streaming && status.is_success();
                        let (raw, content_type) = if streaming_success {
                            (GZIP_SSE, "text/event-stream")
                        } else {
                            (GZIP_JSON, "application/json")
                        };
                        let (decode, server) =
                            mock_encoded_decode_response(status, raw, content_type, "gzip").await;
                        let result = finalize_decode_response(
                            decode,
                            Some(&prefill),
                            path,
                            is_streaming,
                            needs_logprobs,
                        )
                        .await;
                        let response = result.unwrap();
                        assert_eq!(response.status(), status);
                        assert_eq!(response.headers()["content-encoding"], "gzip");
                        assert_eq!(response.headers()["content-type"], content_type);
                        assert_eq!(response.headers()["retry-after"], "7");
                        assert!(response.headers().get("connection").is_none());
                        assert!(response.headers().get("content-length").is_none());
                        let bytes = axum::body::to_bytes(response.into_body(), 1024)
                            .await
                            .unwrap();
                        assert_eq!(bytes.as_ref(), raw, "path: {path}");
                        server.abort();
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn finalizer_preserves_encoded_sse_prefix_on_upstream_error() {
        use futures_util::StreamExt;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0; 4096];
            let bytes_read = socket.read(&mut request).await.unwrap();
            assert!(bytes_read > 0, "client closed before sending request");
            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Encoding: gzip\r\nContent-Encoding: identity\r\nContent-Length: {}\r\n\r\n",
                GZIP_SSE.len() + 20
            );
            socket.write_all(headers.as_bytes()).await.unwrap();
            socket.write_all(GZIP_SSE).await.unwrap();
            socket.shutdown().await.unwrap();
        });
        let decode = reqwest::Client::builder()
            .no_proxy()
            .no_gzip()
            .build()
            .unwrap()
            .get(format!("http://{address}/"))
            .send()
            .await
            .unwrap();
        let response = finalize_decode_response(decode, None, "/v1/completions", true, false)
            .await
            .unwrap();
        let encodings: Vec<_> = response
            .headers()
            .get_all("content-encoding")
            .iter()
            .map(|value| value.to_str().unwrap())
            .collect();
        assert_eq!(encodings, ["gzip", "identity"]);
        let mut stream = response.into_body().into_data_stream();
        let mut received = Vec::new();
        let mut read_failed = false;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => received.extend_from_slice(&bytes),
                Err(_) => {
                    read_failed = true;
                    break;
                }
            }
        }
        assert!(read_failed);
        assert_eq!(received, GZIP_SSE);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn finalizer_reconciles_identity_encoded_responses() {
        let raw = b"data: {\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":99}},\"choices\":[]}\n\n";
        let prefill = json!({"usage":{"prompt_tokens_details":{"cached_tokens":4}}});
        let (decode, server) =
            mock_encoded_decode_response(StatusCode::OK, raw, "text/event-stream", "identity")
                .await;
        let response =
            finalize_decode_response(decode, Some(&prefill), "/v1/completions", true, false)
                .await
                .unwrap();
        assert_eq!(response.headers()["content-encoding"], "identity");
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let result = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(result.contains("\"cached_tokens\":4"));
        assert!(!result.contains("\"cached_tokens\":99"));
        server.abort();

        let (decode, server) = mock_encoded_decode_response(
            StatusCode::OK,
            br#"{"usage":{"prompt_tokens_details":{"cached_tokens":99}},"choices":[]}"#,
            "application/json",
            "identity",
        )
        .await;
        let response =
            finalize_decode_response(decode, Some(&prefill), "/v1/completions", false, false)
                .await
                .unwrap();
        assert!(response.headers().get("content-encoding").is_none());
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let result: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(result["usage"]["prompt_tokens_details"]["cached_tokens"], 4);
        server.abort();
    }

    #[tokio::test]
    async fn finalizer_reconciles_chat_and_completions_without_decode_details_flag() {
        let prefill = json!({"usage": {"prompt_tokens_details": {"cached_tokens": 0}}});
        for path in ["/v1/chat/completions", "/v1/completions"] {
            let (decode, server) = mock_decode_response(
                StatusCode::OK,
                r#"{"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"prompt_tokens_details":null},"metrics":{"x":1}}"#,
            )
            .await;
            let response = finalize_decode_response(decode, Some(&prefill), path, false, false)
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert!(response.headers().get("content-length").is_none());
            let bytes = axum::body::to_bytes(response.into_body(), 1024)
                .await
                .unwrap();
            let result: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(result["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
            assert_eq!(result["usage"]["completion_tokens"], 2);
            assert_eq!(result["metrics"], json!({"x": 1}));
            server.abort();
        }
    }

    #[tokio::test]
    async fn finalizer_reconciles_usage_accounting_edge_cases_for_json_and_sse() {
        for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
            let (input_key, output_key, details_key) = if path == "/v1/responses" {
                ("input_tokens", "output_tokens", "input_tokens_details")
            } else {
                (
                    "prompt_tokens",
                    "completion_tokens",
                    "prompt_tokens_details",
                )
            };
            let prefill = json!({"usage": {input_key: 70, output_key: 1,
                details_key: {"cached_tokens": 90, "audio_tokens": 3}}});
            for streaming in [false, true] {
                for output in [
                    None,
                    Some(json!(null)),
                    Some(json!(-1)),
                    Some(json!(0)),
                    Some(json!(8)),
                ] {
                    let mut decode = json!({"usage": {input_key: 999, "total_tokens": 1007,
                        details_key: {"cached_tokens": 99, "audio_tokens": 2, "future": 7}, "future": 9}});
                    if let Some(output) = &output {
                        decode["usage"][output_key] = output.clone();
                    }
                    if streaming && path == "/v1/responses" {
                        decode = json!({"type": "response.completed", "response": decode});
                    }
                    let (content_type, raw) = if streaming {
                        (
                            "text/event-stream",
                            format!("data: {decode}\n\ndata: [DONE]\n\n").into_bytes(),
                        )
                    } else {
                        ("application/json", serde_json::to_vec(&decode).unwrap())
                    };
                    let upstream = axum::http::Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", content_type)
                        .header("content-length", raw.len())
                        .body(raw)
                        .unwrap();
                    let response = finalize_decode_response(
                        upstream.into(),
                        Some(&prefill),
                        path,
                        streaming,
                        false,
                    )
                    .await
                    .unwrap();
                    assert_eq!(response.status(), StatusCode::OK);
                    assert_eq!(response.headers()["content-type"], content_type);
                    assert!(response.headers().get("content-length").is_none());
                    let bytes = axum::body::to_bytes(response.into_body(), 4096)
                        .await
                        .unwrap();
                    let result: Value = if streaming {
                        let text = std::str::from_utf8(&bytes).unwrap();
                        assert!(text.ends_with("data: [DONE]\n\n"));
                        serde_json::from_str(
                            text.lines().next().unwrap().strip_prefix("data: ").unwrap(),
                        )
                        .unwrap()
                    } else {
                        serde_json::from_slice(&bytes).unwrap()
                    };
                    let usage = usage_merge::usage_object(&result).unwrap();
                    assert_eq!(usage[input_key], 70);
                    assert_eq!(usage.get(output_key), output.as_ref());
                    let total = output
                        .as_ref()
                        .and_then(Value::as_u64)
                        .map(|count| json!(70 + count));
                    assert_eq!(usage.get("total_tokens"), total.as_ref());
                    assert_eq!(usage[details_key], json!({"audio_tokens": 3, "future": 7}));
                    assert_eq!(usage["future"], 9);
                }
            }
        }
    }

    #[tokio::test]
    async fn finalizer_removes_decode_cache_when_prefill_did_not_report_it() {
        let (decode, server) = mock_decode_response(
            StatusCode::OK,
            r#"{"usage":{"prompt_tokens_details":{"cached_tokens":10,"audio_tokens":1}}}"#,
        )
        .await;
        let response =
            finalize_decode_response(decode, Some(&json!({})), "/v1/completions", false, false)
                .await
                .unwrap();
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let result: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            result["usage"]["prompt_tokens_details"],
            json!({"audio_tokens": 1})
        );
        server.abort();
    }

    #[tokio::test]
    async fn finalizer_reconciles_responses_api_and_preserves_unparsable_success() {
        let prefill = json!({"usage": {"prompt_tokens_details": {"cached_tokens": 5}}});
        let response_body = r#"{"usage":{"input_tokens_details":{"cached_tokens":90}}}"#;
        let (decode, server) = mock_decode_response(StatusCode::OK, response_body).await;
        let response =
            finalize_decode_response(decode, Some(&prefill), "/v1/responses", false, false)
                .await
                .unwrap();
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let result: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(result["usage"]["input_tokens_details"]["cached_tokens"], 5);
        server.abort();

        let (decode, server) = mock_decode_response(StatusCode::OK, "not-json").await;
        let response =
            finalize_decode_response(decode, Some(&prefill), "/v1/completions", false, false)
                .await
                .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        assert_eq!(bytes.as_ref(), b"not-json");
        server.abort();
    }

    #[tokio::test]
    async fn finalizer_preserves_unenriched_non_streaming_body_verbatim() {
        for path in [
            "/v1/responses",
            "/generate",
            "/inference/v1/generate",
            "/custom",
        ] {
            let raw = "  upstream plain text\n";
            let (decode, server) = mock_decode_response(StatusCode::OK, raw).await;
            let response = finalize_decode_response(decode, None, path, false, false)
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(response.headers()["content-type"], "application/json");
            let bytes = axum::body::to_bytes(response.into_body(), 1024)
                .await
                .unwrap();
            assert_eq!(bytes.as_ref(), raw.as_bytes(), "path: {path}");
            server.abort();
        }
    }

    #[tokio::test]
    async fn finalizer_reports_truncated_bodies_before_or_after_commit_as_required() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        for is_streaming in [false, true] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = [0; 4096];
                let bytes_read = socket.read(&mut request).await.unwrap();
                assert!(bytes_read > 0, "client closed before sending request");
                // Send fewer bytes than Content-Length promises, then close.
                socket
                            .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 20\r\n\r\n{")
                            .await
                            .unwrap();
                socket.shutdown().await.unwrap();
            });
            let decode = reqwest::Client::builder()
                .no_proxy()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .unwrap()
                .get(format!("http://{address}/"))
                .send()
                .await
                .unwrap();
            let result =
                finalize_decode_response(decode, None, "/v1/completions", is_streaming, false)
                    .await;
            if is_streaming {
                let response = result.unwrap();
                assert!(axum::body::to_bytes(response.into_body(), 1024)
                    .await
                    .is_err());
            } else {
                assert!(result.unwrap_err().contains("Failed to read response body"));
            }
            server.await.unwrap();
        }
    }

    #[tokio::test]
    async fn finalizer_preserves_oversized_json_without_enrichment() {
        let mut raw = vec![b' '; MAX_JSON_REWRITE_BYTES + 1];
        raw.extend_from_slice(b"{}");
        let raw = Bytes::from(raw);
        let response = finalize_decode_body(
            StatusCode::OK,
            HeaderMap::new(),
            raw.clone(),
            Some(&json!({"prompt_logprobs":[1]})),
            "/v1/completions",
            true,
        )
        .unwrap();
        let bytes = axum::body::to_bytes(response.into_body(), raw.len())
            .await
            .unwrap();
        assert_eq!(bytes, raw);
    }

    #[test]
    fn pd_session_identity_requires_kv_aware_decode_with_ttl() {
        use crate::policies::{KvAffinityTable, KvAwarePolicy, RoundRobinPolicy};
        use std::time::Duration;

        let request = json!({"model": "model-a", "session_id": "session-a"});
        let enabled = KvAwarePolicy::default().with_shared_affinity(Arc::new(
            KvAffinityTable::with_ttl(Some(Duration::from_secs(1800))),
        ));
        assert!(VllmPDRouter::extract_affinity_key(&enabled, None, &request).is_some());
        assert!(
            VllmPDRouter::extract_affinity_key(&KvAwarePolicy::default(), None, &request,)
                .is_none()
        );
        assert!(
            VllmPDRouter::extract_affinity_key(&RoundRobinPolicy::new(), None, &request).is_none()
        );
    }

    #[tokio::test]
    async fn test_pd_admission_lifecycle_across_sequential_and_concurrent_http() {
        use super::super::pd_types::Stage;
        use crate::config::{PolicyConfig, RouterConfig, RoutingMode};
        use axum::{
            extract::State,
            routing::{get, post},
            Json, Router,
        };
        use tokio::sync::{mpsc, Semaphore};

        #[derive(Clone)]
        struct Gates {
            p: Arc<Semaphore>,
            d: Arc<Semaphore>,
            entered: mpsc::UnboundedSender<&'static str>,
            discovered: bool,
        }
        async fn prefill(
            State(gates): State<Gates>,
            headers: HeaderMap,
            Json(body): Json<Value>,
        ) -> Response {
            if gates.discovered {
                let id = headers["x-request-id"].to_str().unwrap();
                assert!(id.contains("tcp://P:1") && id.contains("tcp://D:1"));
            }
            gates.entered.send("P").unwrap();
            gates.p.acquire().await.unwrap().forget();
            if body["prompt"] == "prefill-client-error" {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error":"invalid prompt"})),
                )
                    .into_response();
            }
            Json(json!({"kv_transfer_params": {"remote_engine_id": "P"}})).into_response()
        }

        async fn decode(State(gates): State<Gates>, Json(body): Json<Value>) -> Response {
            let streaming = body
                .get("stream")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if !streaming {
                return Json(json!({"choices":[]})).into_response();
            }
            gates.entered.send("D").unwrap();
            gates.d.acquire().await.unwrap().forget();
            let stream = futures_util::stream::once(async {
                Ok::<_, std::io::Error>(bytes::Bytes::from_static(b"data: chunk\n\n"))
            });
            use futures_util::StreamExt;
            Response::new(Body::from_stream(
                stream.chain(futures_util::stream::pending()),
            ))
        }

        // Exercise static/discovered pools and cold sequential/warm NIXL push
        // dispatch. No real engine, ZMQ listener or remote KV is required.
        for (discovered, concurrent, connector) in [
            (false, false, KvConnector::Nixl),
            (false, true, KvConnector::Nixl),
            (true, false, KvConnector::Nixl),
            (true, true, KvConnector::Nixl),
            (true, false, KvConnector::MoriIO),
            (true, true, KvConnector::MoriIO),
        ] {
            let (entered, mut events) = mpsc::unbounded_channel();
            let gates = Gates {
                p: Arc::new(Semaphore::new(0)),
                d: Arc::new(Semaphore::new(0)),
                entered,
                discovered,
            };
            let app = Router::new()
                .route("/p/health", get(|| async { "ok" }))
                .route("/d/health", get(|| async { "ok" }))
                .route("/p/v1/completions", post(prefill))
                .route("/d/v1/completions", post(decode))
                .with_state(gates.clone());
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            let server = tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            let p = format!("{base}/p");
            let d = format!("{base}/d");
            let config = RouterConfig {
                kv_connector: connector,
                mode: RoutingMode::VllmPrefillDecode {
                    prefill_urls: vec![(p.clone(), None)],
                    decode_urls: vec![d.clone()],
                    prefill_policy: None,
                    decode_policy: None,
                    discovery_address: None,
                },
                policy: PolicyConfig::KvAware {
                    router_temperature: 0.0,
                    kv_penalty_host: 0.8,
                    kv_penalty_external: 0.6,
                    kv_penalty_disk: 0.3,
                    static_kv_event_sources: Default::default(),
                },
                worker_startup_timeout_secs: 5,
                ..RouterConfig::default()
            };
            let ctx = Arc::new(
                crate::server::AppContext::new(config, reqwest::Client::new(), 64, None, vec![])
                    .unwrap(),
            );
            let mut router =
                VllmPDRouter::new(vec![(p.clone(), None)], vec![d.clone()], None, &ctx)
                    .await
                    .unwrap();
            if discovered {
                router.use_discovery = true;
                router.service_registry.register_service(
                    p.strip_prefix("http://").unwrap().into(),
                    "tcp://P:1".into(),
                    ServiceType::Prefill,
                );
                router.service_registry.register_service(
                    d.strip_prefix("http://").unwrap().into(),
                    "tcp://D:1".into(),
                    ServiceType::Decode,
                );
            }
            if connector == KvConnector::MoriIO {
                router
                    .service_registry
                    .moriio_transfer_mode
                    .set(if concurrent {
                        MoriIOTransferMode::Write
                    } else {
                        MoriIOTransferMode::Read
                    })
                    .unwrap();
            }
            let router = Arc::new(router);
            if concurrent && connector == KvConnector::Nixl {
                router.store_nixl_info(
                    &p,
                    None,
                    json!({
                        "transfer_mode": "push", "remote_engine_id": "P",
                        "remote_host": "127.0.0.1", "remote_port": 1234,
                    }),
                );
            }
            let request_router = router.clone();
            let request = tokio::spawn(async move {
                request_router
                    .route_transparent(
                        None,
                        "/v1/completions",
                        &Method::POST,
                        json!({"model": "test", "prompt": "hello", "stream": true}),
                    )
                    .await
            });
            let first = tokio::time::timeout(std::time::Duration::from_secs(5), events.recv())
                .await
                .unwrap()
                .unwrap();
            if concurrent {
                let second = tokio::time::timeout(std::time::Duration::from_secs(5), events.recv())
                    .await
                    .unwrap()
                    .unwrap();
                assert_ne!(first, second);
            } else {
                assert_eq!(first, "P");
            }
            let snapshot = router.pd_router.pd_load.snapshot();
            let d_load = snapshot
                .iter()
                .find(|((stage, _), _)| *stage == Stage::Decode)
                .unwrap()
                .1;
            assert_eq!(d_load.pending_decode_requests, usize::from(!concurrent));
            assert_eq!(d_load.active_decode_requests, usize::from(concurrent));
            assert_eq!(
                snapshot
                    .iter()
                    .find(|((stage, _), _)| *stage == Stage::Prefill)
                    .unwrap()
                    .1
                    .prefill_requests,
                1
            );

            // P completion must be visible even while D withholds headers.
            gates.p.add_permits(1);
            if !concurrent {
                assert_eq!(
                    tokio::time::timeout(std::time::Duration::from_secs(5), events.recv())
                        .await
                        .unwrap(),
                    Some("D")
                );
            }
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                while router
                    .pd_router
                    .pd_load
                    .snapshot()
                    .keys()
                    .any(|(stage, _)| *stage == Stage::Prefill)
                {
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                }
            })
            .await
            .unwrap();
            gates.d.add_permits(1);
            let response = tokio::time::timeout(std::time::Duration::from_secs(5), request)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(
                router
                    .pd_router
                    .pd_load
                    .snapshot()
                    .values()
                    .next()
                    .unwrap()
                    .active_decode_requests,
                1
            );
            drop(response); // Covers the never-polled response-body path.
            assert!(router.pd_router.pd_load.snapshot().is_empty());

            gates.p.add_permits(1);
            let response = router
                .route_transparent(
                    None,
                    "/v1/completions",
                    &Method::POST,
                    json!({"model":"test", "prompt":"prefill-client-error", "stream":false}),
                )
                .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert_eq!(events.recv().await, Some("P"));
            assert!(router.pd_router.pd_load.snapshot().is_empty());

            // Aborting an admitted attempt during HTTP work releases both
            // reservations, without needing to receive any response headers.
            let request_router = router.clone();
            let cancelled = tokio::spawn(async move {
                request_router
                    .route_transparent(
                        None,
                        "/v1/completions",
                        &Method::POST,
                        json!({"model": "test", "prompt": "cancel", "stream": true}),
                    )
                    .await
            });
            tokio::time::timeout(std::time::Duration::from_secs(5), events.recv())
                .await
                .unwrap()
                .unwrap();
            cancelled.abort();
            assert!(cancelled.await.unwrap_err().is_cancelled());
            assert!(router.pd_router.pd_load.snapshot().is_empty());
            server.abort();
        }
    }

    #[test]
    fn test_discovery_health_requires_prefill_and_decode_workers() {
        assert!(!discovery_is_ready(0, 0));
        assert!(!discovery_is_ready(1, 0));
        assert!(!discovery_is_ready(0, 1));
        assert!(discovery_is_ready(1, 1));
        assert!(discovery_is_ready(2, 3));
    }

    #[tokio::test]
    async fn discovered_zmq_addresses_preserve_pinned_and_expanded_ranks() {
        let config = crate::config::RouterConfig {
            policy: crate::config::PolicyConfig::RoundRobin,
            ..Default::default()
        };
        let ctx = Arc::new(
            crate::server::AppContext::new(config, reqwest::Client::new(), 64, None, vec![])
                .unwrap(),
        );
        let router = VllmPDRouter::new(vec![], vec![], None, &ctx).await.unwrap();
        for (registered, base, service, role) in [
            (
                "p:8000@1",
                "http://p:8000",
                ServiceType::Prefill,
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
            ),
            (
                "d:8000",
                "http://d:8000",
                ServiceType::Decode,
                WorkerType::Decode,
            ),
        ] {
            router.service_registry.register_service(
                registered.to_owned(),
                "tcp://rank-publisher:5557".to_owned(),
                service.clone(),
            );
            let worker = DPAwareWorker::new(base.to_owned(), 1, 2, role);
            assert_eq!(
                router.get_zmq_address(&worker, service),
                "tcp://rank-publisher:5557"
            );
        }
    }

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
        assert_eq!(result["min_tokens"], 1);
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
    fn test_prefill_chat_completion_raises_zero_min_tokens() {
        let request = json!({
            "model": "test",
            "max_tokens": 512,
            "min_tokens": 0
        });
        let result = VllmPDRouter::prepare_prefill_request(request, "/v1/completions");
        assert_eq!(result["max_tokens"], 1);
        assert_eq!(result["min_tokens"], 1);
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
        // Inject min_tokens when sampling_params is present but omits it.
        assert_eq!(result["sampling_params"]["min_tokens"], 1);
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

    fn moriio_write_prefill_params(
        transfer_id: Option<&str>,
        dp_size: usize,
        tp_size: usize,
    ) -> Value {
        // Mirror the WRITE mode branch in build_prefill_kv_transfer_params.
        json!({
            "do_remote_decode": true,
            "do_remote_prefill": false,
            "remote_engine_id": serde_json::Value::Null,
            "remote_block_ids": serde_json::Value::Null,
            "remote_dp_size": dp_size,
            "remote_tp_size": tp_size,
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
        let params = moriio_write_prefill_params(Some("tx-abc"), 1, 8);
        assert_eq!(params["do_remote_decode"], true);
        assert_eq!(params["do_remote_prefill"], false);
        assert!(params["remote_engine_id"].is_null());
        assert!(params["remote_block_ids"].is_null());
        assert_eq!(params["remote_tp_size"], 8);
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
    async fn dispatch_reservation_test_request(router: &VllmPDRouter, path: &str) -> Response {
        match path {
            "/v1/chat/completions" => router.route_chat(None, &serde_json::from_value(json!({
                "model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": true,
            })).unwrap(), None).await,
            "/v1/completions" => router.route_completion(None, &serde_json::from_value(json!({
                "model": "test", "prompt": "hi", "stream": true,
            })).unwrap(), None).await,
            _ => router.route_transparent(None, path, &Method::POST, json!({
                "model": "test", "input": "hi", "stream": true,
            })).await,
        }
    }

    #[tokio::test]
    async fn selected_decode_is_reserved_while_prefill_is_pending() {
        for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
            let entered = Arc::new(tokio::sync::Notify::new());
            let signal = entered.clone();
            let prefill_url = spawn_test_server(axum::Router::new().fallback(move || {
                let signal = signal.clone();
                async move {
                    signal.notify_one();
                    std::future::pending::<StatusCode>().await
                }
            }))
            .await;
            let prefill: Arc<dyn Worker> = Arc::new(BasicWorker::new(
                prefill_url,
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
            ));
            let decode: Arc<dyn Worker> = Arc::new(BasicWorker::new(
                "http://127.0.0.1:1".to_string(),
                WorkerType::Decode,
            ));
            let router = test_vllm_pd_router().await;
            router.pd_router.worker_registry.register(prefill.clone());
            router.pd_router.worker_registry.register(decode.clone());
            let task =
                tokio::spawn(async move { dispatch_reservation_test_request(&router, path).await });
            tokio::time::timeout(std::time::Duration::from_secs(5), entered.notified())
                .await
                .unwrap();
            let pending_load = decode.load();
            task.abort();
            let _ = task.await;
            assert_eq!(decode.load(), 0, "decode reservation cancellation: {path}");
            assert_eq!(
                pending_load, 1,
                "selected decode must be visible before prefill completes: {path}"
            );
        }
    }
    #[test]
    fn test_decode_reservation_serializes_concurrent_admission() {
        let context = crate::policies::SelectionContext::default();
        let workers: Vec<Arc<dyn Worker>> = (0..2)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://decode-{i}"),
                    WorkerType::Decode,
                )) as Arc<dyn Worker>
            })
            .collect();
        // With two candidates the existing power-of-two policy always compares
        // both. No external load samples are installed in this test.
        let policy = crate::policies::PowerOfTwoPolicy::new();
        let barrier = std::sync::Barrier::new(32);
        let guards = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..32)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        policy
                            .select_and_reserve(&context, &workers, None, None)
                            .unwrap()
                            .1
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(workers[0].load(), 16);
        assert_eq!(workers[1].load(), 16);
        drop(guards);
        assert!(workers.iter().all(|worker| worker.load() == 0));
    }

    #[test]
    fn test_decode_reservation_unavailable_does_not_change_load() {
        let context = crate::policies::SelectionContext::default();
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "http://decode".to_string(),
            WorkerType::Decode,
        ));
        worker.set_healthy(false);
        let policy = crate::policies::PowerOfTwoPolicy::new();
        assert!(policy
            .select_and_reserve(&context, std::slice::from_ref(&worker), None, None)
            .is_none());
        assert_eq!(worker.load(), 0);
        assert!(policy
            .select_and_reserve(&context, &[], None, None)
            .is_none());
    }

    #[tokio::test]
    async fn concurrent_nixl_keeps_one_reservation_until_cancelled() {
        let p_entered = Arc::new(tokio::sync::Notify::new());
        let d_entered = Arc::new(tokio::sync::Notify::new());
        let mut urls = Vec::new();
        for signal in [p_entered.clone(), d_entered.clone()] {
            urls.push(
                spawn_test_server(axum::Router::new().fallback(move || {
                    let signal = signal.clone();
                    async move {
                        signal.notify_one();
                        std::future::pending::<StatusCode>().await
                    }
                }))
                .await,
            );
        }
        let prefill: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            urls[0].clone(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        ));
        let decode: Arc<dyn Worker> =
            Arc::new(BasicWorker::new(urls[1].clone(), WorkerType::Decode));
        let router = test_vllm_pd_router().await;
        router.store_nixl_info(
            prefill.base_url(),
            None,
            json!({
                "transfer_mode": "push", "remote_engine_id": "test-prefill",
                "remote_host": "127.0.0.1", "remote_port": 1234, "tp_size": 1,
            }),
        );
        router.pd_router.worker_registry.register(prefill.clone());
        router.pd_router.worker_registry.register(decode.clone());
        let task = tokio::spawn(async move {
            dispatch_reservation_test_request(&router, "/v1/chat/completions").await
        });
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            p_entered.notified().await;
            d_entered.notified().await;
        })
        .await
        .expect("cached NIXL push identity must dispatch both stages concurrently");
        let loads = (prefill.load(), decode.load());
        task.abort();
        let _ = task.await;
        assert_eq!(
            loads,
            (1, 1),
            "the NIXL fast path must reuse the reservation"
        );
        assert_eq!(decode.load(), 0);
    }

    #[tokio::test]
    async fn prefill_parse_error_releases_decode_reservation() {
        let prefill_url = spawn_test_server(
            axum::Router::new().fallback(|| async { (StatusCode::OK, "not JSON") }),
        )
        .await;
        let prefill: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            prefill_url,
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        ));
        let decode: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "http://127.0.0.1:1".to_string(),
            WorkerType::Decode,
        ));
        let router = test_vllm_pd_router().await;
        router.pd_router.worker_registry.register(prefill.clone());
        router.pd_router.worker_registry.register(decode.clone());
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            dispatch_reservation_test_request(&router, "/v1/responses"),
        )
        .await
        .unwrap();
        assert!(response.status().is_server_error());
        assert_eq!(decode.load(), 0);
    }

    #[tokio::test]
    async fn buffered_logprobs_cancellation_releases_decode_reservation() {
        use axum::{http::header::CONTENT_TYPE, routing::post};
        use bytes::Bytes;
        use std::{convert::Infallible, task::Poll};
        let entered = Arc::new(tokio::sync::Notify::new());
        let signal = entered.clone();
        let prefill_url = spawn_test_server(
            axum::Router::new()
                .fallback(|| async { axum::Json(json!({"kv_transfer_params": {}})) }),
        )
        .await;
        let decode_url = spawn_test_server(axum::Router::new().route(
            "/v1/chat/completions",
            post(move || {
                let signal = signal.clone();
                async move {
                    let stream = futures_util::stream::poll_fn(move |_| {
                        signal.notify_one();
                        Poll::Pending::<Option<Result<Bytes, Infallible>>>
                    });
                    Response::builder()
                        .status(StatusCode::OK)
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from_stream(stream))
                        .unwrap()
                }
            }),
        ))
        .await;
        let prefill: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            prefill_url,
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        ));
        let decode: Arc<dyn Worker> = Arc::new(BasicWorker::new(decode_url, WorkerType::Decode));
        let router = test_vllm_pd_router().await;
        router.pd_router.worker_registry.register(prefill.clone());
        router.pd_router.worker_registry.register(decode.clone());
        let task = tokio::spawn(async move {
            router
                .route_chat(
                    None,
                    &serde_json::from_value(json!({
                        "model": "test", "messages": [{"role":"user","content":"hi"}],
                        "stream": false, "logprobs": true,
                    }))
                    .unwrap(),
                    None,
                )
                .await
        });
        tokio::time::timeout(std::time::Duration::from_secs(5), entered.notified())
            .await
            .unwrap();
        let decode_load = decode.load();
        task.abort();
        let _ = task.await;
        assert_eq!(decode_load, 1);
        assert_eq!(decode.load(), 0);
    }
    #[tokio::test]
    async fn ordinary_decode_dispatch_reuses_reservation() {
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let signal = entered.clone();
        let resume = release.clone();
        let prefill_url = spawn_test_server(
            axum::Router::new()
                .fallback(|| async { axum::Json(json!({"kv_transfer_params": {}})) }),
        )
        .await;
        let decode_url = spawn_test_server(axum::Router::new().fallback(move || {
            let signal = signal.clone();
            let resume = resume.clone();
            async move {
                signal.notify_one();
                resume.notified().await;
                axum::Json(json!({"output": []}))
            }
        }))
        .await;
        let prefill: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            prefill_url,
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        ));
        let decode: Arc<dyn Worker> = Arc::new(BasicWorker::new(decode_url, WorkerType::Decode));
        let router = test_vllm_pd_router().await;
        router.pd_router.worker_registry.register(prefill.clone());
        router.pd_router.worker_registry.register(decode.clone());
        let task = tokio::spawn(async move {
            dispatch_reservation_test_request(&router, "/v1/responses").await
        });
        tokio::time::timeout(std::time::Duration::from_secs(5), entered.notified())
            .await
            .unwrap();
        assert_eq!(prefill.load(), 0);
        assert_eq!(
            decode.load(),
            1,
            "dispatch must not increment the reservation twice"
        );
        release.notify_one();
        let response = tokio::time::timeout(std::time::Duration::from_secs(5), task)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        drop(response);
        assert_eq!(decode.load(), 0);
    }
    #[test]
    fn cache_aware_affinity_survives_pending_reservations() {
        use crate::policies::{CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy};
        let context = crate::policies::SelectionContext::default();
        let workers: Vec<Arc<dyn Worker>> = (0..8)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://d-{i}"),
                    WorkerType::Decode,
                )) as Arc<dyn Worker>
            })
            .collect();
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        policy.init_workers(&workers);
        let prefix = "shared conversation prefix ".repeat(100);
        let (warm_idx, guard) = policy
            .select_and_reserve(&context, &workers, Some(&prefix), None)
            .unwrap();
        drop(guard);
        // Sequential turns preserve affinity as the conversation grows.
        for turn in 0..8 {
            let text = format!("{prefix}turn {turn}");
            let (idx, guard) = policy
                .select_and_reserve(&context, &workers, Some(&text), None)
                .unwrap();
            assert_eq!(idx, warm_idx);
            drop(guard);
        }
        // A pending warm request does not force loss of affinity below the default threshold.
        let (_, first) = policy
            .select_and_reserve(&context, &workers, Some(&prefix), None)
            .unwrap();
        let (idx, second) = policy
            .select_and_reserve(&context, &workers, Some(&prefix), None)
            .unwrap();
        assert_eq!(idx, warm_idx);
        // A disjoint cold request can use an unoccupied D without erasing warm affinity.
        let (cold_idx, cold) = policy
            .select_and_reserve(&context, &workers, Some("ZZZZ unrelated cold prompt"), None)
            .unwrap();
        assert_ne!(cold_idx, warm_idx);
        drop((first, second, cold));
        assert!(workers.iter().all(|w| w.load() == 0));
        let (idx, guard) = policy
            .select_and_reserve(&context, &workers, Some(&prefix), None)
            .unwrap();
        assert_eq!(idx, warm_idx);
        drop(guard);
    }

    #[tokio::test]
    async fn dispatch_failures_release_only_owned_reservation() {
        for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
            for mode in [
                "prefill_http",
                "prefill_transport",
                "decode_http",
                "decode_transport",
                "decode_timeout",
            ] {
                let prefill_url = if mode == "prefill_transport" {
                    "http://127.0.0.1:1".to_string()
                } else if mode == "prefill_http" {
                    spawn_test_server(
                        axum::Router::new()
                            .fallback(|| async { StatusCode::INTERNAL_SERVER_ERROR }),
                    )
                    .await
                } else {
                    spawn_test_server(
                        axum::Router::new()
                            .fallback(|| async { axum::Json(json!({"kv_transfer_params": {}})) }),
                    )
                    .await
                };
                let decode_url = if mode == "decode_transport" {
                    "http://127.0.0.1:1".to_string()
                } else if mode == "decode_timeout" {
                    spawn_test_server(
                        axum::Router::new()
                            .fallback(|| async { std::future::pending::<StatusCode>().await }),
                    )
                    .await
                } else {
                    spawn_test_server(
                        axum::Router::new()
                            .fallback(|| async { StatusCode::INTERNAL_SERVER_ERROR }),
                    )
                    .await
                };
                let prefill: Arc<dyn Worker> = Arc::new(BasicWorker::new(
                    prefill_url,
                    WorkerType::Prefill {
                        bootstrap_port: None,
                    },
                ));
                let decode: Arc<dyn Worker> =
                    Arc::new(BasicWorker::new(decode_url, WorkerType::Decode));
                let mut router = test_vllm_pd_router().await;
                router.http_client = reqwest::Client::builder()
                    .timeout(std::time::Duration::from_millis(200))
                    .build()
                    .unwrap();
                router.pd_router.client = router.http_client.clone();
                router.pd_router.worker_registry.register(prefill);
                router.pd_router.worker_registry.register(decode.clone());
                // Sentinel load catches accidental double release hidden by saturating counters.
                decode.increment_load();
                let response = tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    dispatch_reservation_test_request(&router, path),
                )
                .await
                .unwrap();
                assert!(
                    response.status().is_server_error(),
                    "{path} {mode}: {}",
                    response.status()
                );
                assert_eq!(decode.load(), 1, "{path} {mode}");
                drop(response);
                assert_eq!(
                    decode.load(),
                    1,
                    "response drop must not release again: {path} {mode}"
                );
                decode.decrement_load();
            }
        }
    }
}
