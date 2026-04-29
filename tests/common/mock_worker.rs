// Mock worker for testing - these functions are used by integration tests
#![allow(dead_code)]

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::sse::{Event, KeepAlive},
    response::{IntoResponse, Response, Sse},
    routing::{get, post},
    Router,
};
use futures_util::stream::{self, StreamExt};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Configuration for mock worker behavior
#[derive(Clone)]
pub struct MockWorkerConfig {
    pub port: u16,
    pub worker_type: WorkerType,
    pub health_status: HealthStatus,
    pub response_delay_ms: u64,
    pub fail_rate: f32,
}

#[derive(Clone, Debug)]
pub enum WorkerType {
    Regular,
    Prefill,
    Decode,
}

#[derive(Clone, Debug)]
pub enum HealthStatus {
    Healthy,
    Unhealthy,
    Degraded,
}

/// Supported deterministic response modes for router integration tests.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MockHttpResponseMode {
    #[default]
    Default,
    Ok,
    TooManyRequests,
    ServiceUnavailable,
}

impl MockHttpResponseMode {
    fn forced_http_status(self) -> Option<StatusCode> {
        match self {
            Self::Default => None,
            Self::Ok => Some(StatusCode::OK),
            Self::TooManyRequests => Some(StatusCode::TOO_MANY_REQUESTS),
            Self::ServiceUnavailable => Some(StatusCode::SERVICE_UNAVAILABLE),
        }
    }
}

#[derive(Clone)]
struct MockWorkerState {
    config: Arc<RwLock<MockWorkerConfig>>,
    forced_http_status: Arc<RwLock<Option<StatusCode>>>,
}

/// Mock worker server for testing
pub struct MockWorker {
    config: Arc<RwLock<MockWorkerConfig>>,
    forced_http_status: Arc<RwLock<Option<StatusCode>>>,
    shutdown_handle: Option<tokio::task::JoinHandle<()>>,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl MockWorker {
    pub fn new(config: MockWorkerConfig) -> Self {
        Self::new_with_forced_http_status(config, None)
    }

    pub fn with_http_response_mode(
        config: MockWorkerConfig,
        response_mode: MockHttpResponseMode,
    ) -> Self {
        Self::new_with_forced_http_status(config, response_mode.forced_http_status())
    }

    pub fn with_forced_http_status(config: MockWorkerConfig, status: StatusCode) -> Self {
        Self::new_with_forced_http_status(config, Some(status))
    }

    fn new_with_forced_http_status(
        config: MockWorkerConfig,
        forced_http_status: Option<StatusCode>,
    ) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            forced_http_status: Arc::new(RwLock::new(forced_http_status)),
            shutdown_handle: None,
            shutdown_tx: None,
        }
    }

    pub async fn set_forced_http_status(&self, status: Option<StatusCode>) {
        *self.forced_http_status.write().await = status;
    }

    pub async fn set_http_response_mode(&self, response_mode: MockHttpResponseMode) {
        self.set_forced_http_status(response_mode.forced_http_status())
            .await;
    }

    pub async fn clear_http_response_mode(&self) {
        self.set_http_response_mode(MockHttpResponseMode::Default)
            .await;
    }

    /// Start the mock worker server
    pub async fn start(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        let config = self.config.clone();
        let port = config.read().await.port;
        let state = MockWorkerState {
            config: self.config.clone(),
            forced_http_status: self.forced_http_status.clone(),
        };

        // If port is 0, find an available port
        let port = if port == 0 {
            let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
            let port = listener.local_addr()?.port();
            drop(listener);
            config.write().await.port = port;
            port
        } else {
            port
        };

        let app = Router::new()
            .route("/health", get(health_handler))
            .route("/health_generate", get(health_generate_handler))
            .route("/get_server_info", get(server_info_handler))
            .route("/get_model_info", get(model_info_handler))
            .route("/generate", post(generate_handler))
            .route("/v1/chat/completions", post(chat_completions_handler))
            .route("/v1/completions", post(completions_handler))
            .route("/v1/rerank", post(rerank_handler))
            .route("/v1/responses", post(responses_handler))
            .route("/v1/responses/{response_id}", get(responses_get_handler))
            .route(
                "/v1/responses/{response_id}/cancel",
                post(responses_cancel_handler),
            )
            .route("/flush_cache", post(flush_cache_handler))
            .route("/v1/models", get(v1_models_handler))
            .with_state(state);

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        self.shutdown_tx = Some(shutdown_tx);

        // Spawn the server in a separate task
        let handle = tokio::spawn(async move {
            let listener = match tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("Failed to bind to port {}: {}", port, e);
                    return;
                }
            };

            let server = axum::serve(listener, app).with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            });

            if let Err(e) = server.await {
                eprintln!("Server error: {}", e);
            }
        });

        self.shutdown_handle = Some(handle);

        // Wait for the server to start
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let url = format!("http://127.0.0.1:{}", port);
        Ok(url)
    }

    /// Stop the mock worker server
    pub async fn stop(&mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }

        if let Some(handle) = self.shutdown_handle.take() {
            // Wait for the server to shut down
            let _ = tokio::time::timeout(tokio::time::Duration::from_secs(5), handle).await;
        }
    }
}

impl Drop for MockWorker {
    fn drop(&mut self) {
        // Clean shutdown when dropped
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
    }
}

// Handler implementations

async fn forced_http_status(state: &MockWorkerState) -> Option<StatusCode> {
    *state.forced_http_status.read().await
}

/// Keep legacy randomized failures unless a deterministic forced status is configured.
fn should_fail(config: &MockWorkerConfig, forced_http_status: Option<StatusCode>) -> bool {
    forced_http_status.is_none() && rand::random::<f32>() < config.fail_rate
}

fn forced_json_error_response(status: StatusCode) -> Response {
    (
        status,
        Json(json!({
            "error": format!("Forced {} response for testing", status.as_u16())
        })),
    )
        .into_response()
}

fn forced_openai_error_response(status: StatusCode) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": format!("Forced {} response for testing", status.as_u16()),
                "type": status.canonical_reason().unwrap_or("error").to_ascii_lowercase(),
                "code": status.as_u16().to_string(),
            }
        })),
    )
        .into_response()
}

async fn health_handler(State(state): State<MockWorkerState>) -> Response {
    let config = state.config.read().await;

    match config.health_status {
        HealthStatus::Healthy => Json(json!({
            "status": "healthy",
            "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            "worker_type": format!("{:?}", config.worker_type),
        }))
        .into_response(),
        HealthStatus::Unhealthy => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "unhealthy",
                "error": "Worker is not responding"
            })),
        )
            .into_response(),
        HealthStatus::Degraded => Json(json!({
            "status": "degraded",
            "warning": "High load detected"
        }))
        .into_response(),
    }
}

async fn health_generate_handler(State(state): State<MockWorkerState>) -> Response {
    let config = state.config.read().await;

    if should_fail(&config, None) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": "Random failure for testing"
            })),
        )
            .into_response();
    }

    if matches!(config.health_status, HealthStatus::Healthy) {
        Json(json!({
            "status": "ok",
            "queue_length": 0,
            "processing_time_ms": config.response_delay_ms
        }))
        .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": "Generation service unavailable"
            })),
        )
            .into_response()
    }
}

async fn server_info_handler(State(state): State<MockWorkerState>) -> Response {
    let config = state.config.read().await;

    if should_fail(&config, None) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": "Random failure for testing"
            })),
        )
            .into_response();
    }

    Json(json!({
        "model_path": "mock-model-path",
        "tokenizer_path": "mock-tokenizer-path",
        "port": config.port,
        "host": "127.0.0.1",
        "max_num_batched_tokens": 32768,
        "max_prefill_tokens": 16384,
        "mem_fraction_static": 0.88,
        "tp_size": 1,
        "dp_size": 1,
        "stream_interval": 8,
        "dtype": "float16",
        "device": "cuda",
        "enable_flashinfer": true,
        "enable_p2p_check": true,
        "context_length": 32768,
        "chat_template": null,
        "disable_radix_cache": false,
        "enable_torch_compile": false,
        "trust_remote_code": false,
        "show_time_cost": false,
        "waiting_queue_size": 0,
        "running_queue_size": 0,
        "req_to_token_ratio": 1.2,
        "min_running_requests": 0,
        "max_running_requests": 2048,
        "max_req_num": 8192,
        "max_batch_tokens": 32768,
        "schedule_policy": "lpm",
        "schedule_conservativeness": 1.0,
        "version": "0.3.0",
        "internal_states": [{
            "waiting_queue_size": 0,
            "running_queue_size": 0
        }]
    }))
    .into_response()
}

async fn model_info_handler(State(state): State<MockWorkerState>) -> Response {
    let config = state.config.read().await;

    if should_fail(&config, None) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": "Random failure for testing"
            })),
        )
            .into_response();
    }

    Json(json!({
        "model_path": "mock-model-path",
        "tokenizer_path": "mock-tokenizer-path",
        "is_generation": true,
        "preferred_sampling_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 2048
        }
    }))
    .into_response()
}

async fn generate_handler(
    State(state): State<MockWorkerState>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<serde_json::Value>,
) -> Response {
    let forced_http_status = forced_http_status(&state).await;
    let config = state.config.read().await;

    // Capture request for test inspection
    capture_request(config.port, "/generate", &headers);

    if let Some(status) = forced_http_status.filter(|status| *status != StatusCode::OK) {
        return forced_json_error_response(status);
    }

    if should_fail(&config, forced_http_status) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": "Random failure for testing"
            })),
        )
            .into_response();
    }

    if config.response_delay_ms > 0 {
        tokio::time::sleep(tokio::time::Duration::from_millis(config.response_delay_ms)).await;
    }

    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if is_stream {
        let stream_delay = config.response_delay_ms;

        // Check if it's a batch request
        let is_batch = payload.get("text").and_then(|t| t.as_array()).is_some();

        let batch_size = if is_batch {
            payload
                .get("text")
                .and_then(|t| t.as_array())
                .map(|arr| arr.len())
                .unwrap_or(1)
        } else {
            1
        };

        let mut events = Vec::new();

        // Generate events for each item in batch
        for i in 0..batch_size {
            let timestamp_start = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();

            let data = json!({
                "text": format!("Mock response {}", i + 1),
                "meta_info": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "completion_tokens_wo_jump_forward": 5,
                    "input_token_logprobs": null,
                    "output_token_logprobs": null,
                    "first_token_latency": stream_delay as f64 / 1000.0,
                    "time_to_first_token": stream_delay as f64 / 1000.0,
                    "time_per_output_token": 0.01,
                    "end_time": timestamp_start + (stream_delay as f64 / 1000.0),
                    "start_time": timestamp_start,
                    "finish_reason": {
                        "type": "stop",
                        "reason": "length"
                    }
                },
                "stage": "mid"
            });

            events.push(Ok::<_, Infallible>(Event::default().data(data.to_string())));
        }

        // Add [DONE] event
        events.push(Ok(Event::default().data("[DONE]")));

        let stream = stream::iter(events);

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        Json(json!({
            "text": "This is a mock response.",
            "meta_info": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "completion_tokens_wo_jump_forward": 5,
                "input_token_logprobs": null,
                "output_token_logprobs": null,
                "first_token_latency": config.response_delay_ms as f64 / 1000.0,
                "time_to_first_token": config.response_delay_ms as f64 / 1000.0,
                "time_per_output_token": 0.01,
                "finish_reason": {
                    "type": "stop",
                    "reason": "length"
                }
            }
        }))
        .into_response()
    }
}

async fn chat_completions_handler(
    State(state): State<MockWorkerState>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<serde_json::Value>,
) -> Response {
    let forced_http_status = forced_http_status(&state).await;
    let config = state.config.read().await;

    // Capture request for test inspection
    capture_request(config.port, "/v1/chat/completions", &headers);

    if let Some(status) = forced_http_status.filter(|status| *status != StatusCode::OK) {
        return forced_openai_error_response(status);
    }

    if should_fail(&config, forced_http_status) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": "Random failure for testing",
                    "type": "internal_error",
                    "code": "internal_error"
                }
            })),
        )
            .into_response();
    }

    if config.response_delay_ms > 0 {
        tokio::time::sleep(tokio::time::Duration::from_millis(config.response_delay_ms)).await;
    }

    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    if is_stream {
        let request_id = format!("chatcmpl-{}", Uuid::new_v4());

        let stream = stream::once(async move {
            let chunk = json!({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": timestamp,
                "model": "mock-model",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": "This is a mock chat response."
                    },
                    "finish_reason": null
                }]
            });

            Ok::<_, Infallible>(Event::default().data(chunk.to_string()))
        })
        .chain(stream::once(async { Ok(Event::default().data("[DONE]")) }));

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        Json(json!({
            "id": format!("chatcmpl-{}", Uuid::new_v4()),
            "object": "chat.completion",
            "created": timestamp,
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock chat response."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }))
        .into_response()
    }
}

async fn completions_handler(
    State(state): State<MockWorkerState>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<serde_json::Value>,
) -> Response {
    let forced_http_status = forced_http_status(&state).await;
    let config = state.config.read().await;

    // Capture request for test inspection
    capture_request(config.port, "/v1/completions", &headers);

    if let Some(status) = forced_http_status.filter(|status| *status != StatusCode::OK) {
        return forced_openai_error_response(status);
    }

    if should_fail(&config, forced_http_status) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": "Random failure for testing",
                    "type": "internal_error",
                    "code": "internal_error"
                }
            })),
        )
            .into_response();
    }

    if config.response_delay_ms > 0 {
        tokio::time::sleep(tokio::time::Duration::from_millis(config.response_delay_ms)).await;
    }

    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    if is_stream {
        let request_id = format!("cmpl-{}", Uuid::new_v4());

        let stream = stream::once(async move {
            let chunk = json!({
                "id": request_id,
                "object": "text_completion",
                "created": timestamp,
                "model": "mock-model",
                "choices": [{
                    "text": "This is a mock completion.",
                    "index": 0,
                    "logprobs": null,
                    "finish_reason": null
                }]
            });

            Ok::<_, Infallible>(Event::default().data(chunk.to_string()))
        })
        .chain(stream::once(async { Ok(Event::default().data("[DONE]")) }));

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        Json(json!({
            "id": format!("cmpl-{}", Uuid::new_v4()),
            "object": "text_completion",
            "created": timestamp,
            "model": "mock-model",
            "choices": [{
                "text": "This is a mock completion.",
                "index": 0,
                "logprobs": null,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }))
        .into_response()
    }
}

async fn responses_handler(
    State(state): State<MockWorkerState>,
    Json(payload): Json<serde_json::Value>,
) -> Response {
    let forced_http_status = forced_http_status(&state).await;
    let config = state.config.read().await;

    if let Some(status) = forced_http_status.filter(|status| *status != StatusCode::OK) {
        return forced_openai_error_response(status);
    }

    if should_fail(&config, forced_http_status) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": "Random failure for testing",
                    "type": "internal_error",
                    "code": "internal_error"
                }
            })),
        )
            .into_response();
    }

    if config.response_delay_ms > 0 {
        tokio::time::sleep(tokio::time::Duration::from_millis(config.response_delay_ms)).await;
    }

    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    // Background storage simulation
    let is_background = payload
        .get("background")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let req_id = payload
        .get("request_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    if is_background {
        if let Some(id) = &req_id {
            store_response_for_port(config.port, id);
        }
    }

    if is_stream {
        let request_id = format!("resp-{}", Uuid::new_v4());

        let stream = stream::once(async move {
            let chunk = json!({
                "id": request_id,
                "object": "response",
                "created_at": timestamp,
                "model": "mock-model",
                "status": "in_progress",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": "This is a mock responses streamed output."
                    }]
                }]
            });
            Ok::<_, Infallible>(Event::default().data(chunk.to_string()))
        })
        .chain(stream::once(async { Ok(Event::default().data("[DONE]")) }));

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else if is_background {
        let rid = req_id.unwrap_or_else(|| format!("resp-{}", Uuid::new_v4()));
        Json(json!({
            "id": rid,
            "object": "response",
            "created_at": timestamp,
            "model": "mock-model",
            "output": [],
            "status": "queued",
            "usage": null
        }))
        .into_response()
    } else {
        Json(json!({
            "id": format!("resp-{}", Uuid::new_v4()),
            "object": "response",
            "created_at": timestamp,
            "model": "mock-model",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "This is a mock responses output."
                }]
            }],
            "status": "completed",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15
            }
        }))
        .into_response()
    }
}

async fn flush_cache_handler(State(state): State<MockWorkerState>) -> Response {
    let forced_http_status = forced_http_status(&state).await;
    let config = state.config.read().await;

    if let Some(status) = forced_http_status.filter(|status| *status != StatusCode::OK) {
        return forced_json_error_response(status);
    }

    if should_fail(&config, forced_http_status) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": "Random failure for testing"
            })),
        )
            .into_response();
    }

    Json(json!({
        "message": "Cache flushed successfully"
    }))
    .into_response()
}

async fn v1_models_handler(State(state): State<MockWorkerState>) -> Response {
    let forced_http_status = forced_http_status(&state).await;
    let config = state.config.read().await;

    if let Some(status) = forced_http_status.filter(|status| *status != StatusCode::OK) {
        return forced_openai_error_response(status);
    }

    if should_fail(&config, forced_http_status) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": "Random failure for testing",
                    "type": "internal_error",
                    "code": "internal_error"
                }
            })),
        )
            .into_response();
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    Json(json!({
        "object": "list",
        "data": [{
            "id": "mock-model",
            "object": "model",
            "created": timestamp,
            "owned_by": "organization-owner"
        }]
    }))
    .into_response()
}

async fn responses_get_handler(
    State(state): State<MockWorkerState>,
    Path(response_id): Path<String>,
) -> Response {
    let forced_http_status = forced_http_status(&state).await;
    let config = state.config.read().await;
    if let Some(status) = forced_http_status.filter(|status| *status != StatusCode::OK) {
        return forced_json_error_response(status);
    }
    if should_fail(&config, forced_http_status) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "Random failure for testing" })),
        )
            .into_response();
    }
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    // Only return 200 if this worker "stores" the response id
    if response_exists_for_port(config.port, &response_id) {
        Json(json!({
            "id": response_id,
            "object": "response",
            "created_at": timestamp,
            "model": "mock-model",
            "output": [],
            "status": "completed",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }
        }))
        .into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

async fn responses_cancel_handler(
    State(state): State<MockWorkerState>,
    Path(response_id): Path<String>,
) -> Response {
    let forced_http_status = forced_http_status(&state).await;
    let config = state.config.read().await;
    if let Some(status) = forced_http_status.filter(|status| *status != StatusCode::OK) {
        return forced_json_error_response(status);
    }
    if should_fail(&config, forced_http_status) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "Random failure for testing" })),
        )
            .into_response();
    }
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    if response_exists_for_port(config.port, &response_id) {
        Json(json!({
            "id": response_id,
            "object": "response",
            "created_at": timestamp,
            "model": "mock-model",
            "output": [],
            "status": "cancelled",
            "usage": null
        }))
        .into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

// --- Simple in-memory response store per worker port (for tests) ---
static RESP_STORE: OnceLock<Mutex<HashMap<u16, HashSet<String>>>> = OnceLock::new();

fn get_store() -> &'static Mutex<HashMap<u16, HashSet<String>>> {
    RESP_STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn store_response_for_port(port: u16, response_id: &str) {
    let mut map = get_store().lock().unwrap();
    map.entry(port).or_default().insert(response_id.to_string());
}

fn response_exists_for_port(port: u16, response_id: &str) -> bool {
    let map = get_store().lock().unwrap();
    map.get(&port)
        .map(|set| set.contains(response_id))
        .unwrap_or(false)
}

// Minimal rerank handler returning mock results; router shapes final response
async fn rerank_handler(
    State(state): State<MockWorkerState>,
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    let forced_http_status = forced_http_status(&state).await;
    let config = state.config.read().await;

    // Simulate response delay
    if config.response_delay_ms > 0 {
        tokio::time::sleep(tokio::time::Duration::from_millis(config.response_delay_ms)).await;
    }

    if let Some(status) = forced_http_status.filter(|status| *status != StatusCode::OK) {
        return forced_json_error_response(status);
    }

    // Simulate failure rate
    if should_fail(&config, forced_http_status) {
        return (StatusCode::INTERNAL_SERVER_ERROR, "Simulated failure").into_response();
    }

    // Extract documents from the request to create mock results
    let empty_vec = vec![];
    let documents = payload
        .get("documents")
        .and_then(|d| d.as_array())
        .unwrap_or(&empty_vec);

    // Create mock rerank results with scores based on document index
    let mut mock_results = Vec::new();
    for (i, doc) in documents.iter().enumerate() {
        let score = 0.95 - (i as f32 * 0.1); // Decreasing scores
        let result = serde_json::json!({
            "score": score,
            "document": doc.as_str().unwrap_or(""),
            "index": i,
            "meta_info": {
                "confidence": if score > 0.9 { "high" } else { "medium" }
            }
        });
        mock_results.push(result);
    }

    // Sort by score (highest first) to simulate proper ranking
    mock_results.sort_by(|a, b| {
        b["score"]
            .as_f64()
            .unwrap()
            .partial_cmp(&a["score"].as_f64().unwrap())
            .unwrap()
    });

    (StatusCode::OK, Json(mock_results)).into_response()
}

impl Default for MockWorkerConfig {
    fn default() -> Self {
        Self {
            port: 0,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }
    }
}

// --- Request header capture for verifying router behavior (e.g., X-data-parallel-rank) ---

/// A captured request with headers and path
#[derive(Debug, Clone)]
pub struct CapturedRequest {
    pub path: String,
    pub headers: HashMap<String, String>,
}

static REQ_CAPTURE_STORE: OnceLock<Mutex<HashMap<u16, Vec<CapturedRequest>>>> = OnceLock::new();

fn get_capture_store() -> &'static Mutex<HashMap<u16, Vec<CapturedRequest>>> {
    REQ_CAPTURE_STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record a request for a given worker port
pub fn capture_request(port: u16, path: &str, headers: &axum::http::HeaderMap) {
    let captured = CapturedRequest {
        path: path.to_string(),
        headers: headers
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|v| (name.as_str().to_string(), v.to_string()))
            })
            .collect(),
    };
    let mut store = get_capture_store().lock().unwrap();
    store.entry(port).or_default().push(captured);
}

/// Get all captured requests for a given worker port
pub fn get_captured_requests(port: u16) -> Vec<CapturedRequest> {
    let store = get_capture_store().lock().unwrap();
    store.get(&port).cloned().unwrap_or_default()
}

/// Clear captured requests for a given worker port
pub fn clear_captured_requests(port: u16) {
    let mut store = get_capture_store().lock().unwrap();
    store.remove(&port);
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    async fn post_chat_completion(url: &str) -> reqwest::Response {
        reqwest::Client::new()
            .post(format!("{}/v1/chat/completions", url))
            .json(&json!({
                "model": "mock-model",
                "messages": [{
                    "role": "user",
                    "content": "hello"
                }]
            }))
            .send()
            .await
            .expect("mock worker request should succeed")
    }

    #[tokio::test]
    async fn forced_http_status_returns_deterministic_retryable_errors() {
        for (response_mode, status) in [
            (
                super::MockHttpResponseMode::TooManyRequests,
                axum::http::StatusCode::TOO_MANY_REQUESTS,
            ),
            (
                super::MockHttpResponseMode::ServiceUnavailable,
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
            ),
        ] {
            let mut worker = super::MockWorker::with_http_response_mode(
                super::MockWorkerConfig::default(),
                response_mode,
            );
            let url = worker.start().await.expect("mock worker should start");

            let response = post_chat_completion(&url).await;
            assert_eq!(response.status(), status);

            let body: serde_json::Value = response
                .json()
                .await
                .expect("forced error body should be valid json");
            assert_eq!(body["error"]["code"], status.as_u16().to_string());

            worker.stop().await;
        }
    }

    #[tokio::test]
    async fn forced_http_status_ok_disables_random_failures() {
        let mut worker = super::MockWorker::with_http_response_mode(
            super::MockWorkerConfig {
                fail_rate: 1.0,
                ..Default::default()
            },
            super::MockHttpResponseMode::Ok,
        );
        let url = worker.start().await.expect("mock worker should start");

        let response = post_chat_completion(&url).await;
        assert_eq!(response.status(), axum::http::StatusCode::OK);

        let body: serde_json::Value = response
            .json()
            .await
            .expect("forced success body should be valid json");
        assert_eq!(body["object"], "chat.completion");

        worker.stop().await;
    }
}
