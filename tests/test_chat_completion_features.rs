/// Regression tests for chat completion routing features.
/// These tests verify the features provided by route_typed_request for /v1/chat/completions:
/// retry with worker re-selection, circuit breaker, model-specific routing, streaming SSE,
/// header forwarding, and JSON field pass-through.
mod common;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use common::mock_worker::{
    clear_captured_requests, clear_deterministic_failures, get_captured_requests,
    set_deterministic_failures, HealthStatus, MockWorker, MockWorkerConfig, WorkerType,
};
use reqwest::Client;
use serde_json::json;
use std::sync::Arc;
use tower::ServiceExt;
use vllm_router_rs::config::{
    CircuitBreakerConfig, ConnectionMode, PolicyConfig, RetryConfig, RouterConfig, RoutingMode,
};
use vllm_router_rs::routers::{RouterFactory, RouterTrait};

/// Test context that manages mock workers and provides helper methods
struct TestContext {
    workers: Vec<MockWorker>,
    router: Arc<dyn RouterTrait>,
    client: Client,
    config: RouterConfig,
    ports: Vec<u16>,
}

impl TestContext {
    async fn new_with_config(
        mut config: RouterConfig,
        worker_configs: Vec<MockWorkerConfig>,
    ) -> Self {
        let mut workers = Vec::new();
        let mut worker_urls = Vec::new();
        let mut ports = Vec::new();

        for worker_config in worker_configs {
            let port = worker_config.port;
            ports.push(port);
            let mut worker = MockWorker::new(worker_config);
            let url = worker.start().await.unwrap();
            worker_urls.push(url);
            workers.push(worker);
        }

        if !workers.is_empty() {
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        }

        if let RoutingMode::Regular {
            worker_urls: ref mut urls,
        } = config.mode
        {
            if urls.is_empty() {
                *urls = worker_urls.clone();
            }
        }

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.request_timeout_secs))
            .build()
            .unwrap();

        let app_context = common::create_test_context(config.clone());
        let router = RouterFactory::create_router(&app_context).await.unwrap();
        let router = Arc::from(router);

        if !workers.is_empty() {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }

        Self {
            workers,
            router,
            client,
            config,
            ports,
        }
    }

    async fn create_app(&self) -> axum::Router {
        common::test_app::create_test_app(
            Arc::clone(&self.router),
            self.client.clone(),
            &self.config,
        )
    }

    async fn shutdown(mut self) {
        for worker in &mut self.workers {
            worker.stop().await;
        }
        for port in &self.ports {
            clear_captured_requests(*port);
            clear_deterministic_failures(*port);
        }
    }
}

fn default_config() -> RouterConfig {
    RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![],
        },
        policy: PolicyConfig::RoundRobin,
        host: "127.0.0.1".to_string(),
        port: 3010,
        max_payload_size: 256 * 1024 * 1024,
        request_timeout_secs: 30,
        worker_startup_timeout_secs: 10,
        worker_startup_check_interval_secs: 1,
        discovery: None,
        intra_node_data_parallel_size: 1,
        api_key: None,
        api_key_validation_urls: vec![],
        metrics: None,
        log_dir: None,
        log_level: None,
        request_id_headers: None,
        max_concurrent_requests: 64,
        queue_size: 0,
        queue_timeout_secs: 60,
        rate_limit_tokens_per_second: None,
        cors_allowed_origins: vec![],
        retry: RetryConfig::default(),
        circuit_breaker: CircuitBreakerConfig::default(),
        disable_retries: false,
        disable_circuit_breaker: false,
        health_check: vllm_router_rs::config::HealthCheckConfig::default(),
        enable_igw: false,
        connection_mode: ConnectionMode::Http,
        model_path: None,
        tokenizer_path: None,
        history_backend: vllm_router_rs::config::HistoryBackend::Memory,
        enable_profiling: false,
        profile_timeout_secs: 30,
    }
}

fn chat_payload() -> serde_json::Value {
    json!({
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": false
    })
}

fn make_chat_request(payload: &serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_string(payload).unwrap()))
        .unwrap()
}

// =============================================================================
// Test 1: Retry selects a different worker on failure
// =============================================================================

#[tokio::test]
async fn test_retry_selects_different_worker_on_failure() {
    let port1 = 21001;
    let port2 = 21002;

    // Both workers fail their first request so that regardless of which worker
    // round-robin selects first, a retry is triggered.
    set_deterministic_failures(port1, vec![1]);
    set_deterministic_failures(port2, vec![1]);

    let config = RouterConfig {
        retry: RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 10,
            max_backoff_ms: 100,
            backoff_multiplier: 2.0,
            jitter_factor: 0.0,
        },
        ..default_config()
    };

    let ctx = TestContext::new_with_config(
        config,
        vec![
            MockWorkerConfig {
                port: port1,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            },
            MockWorkerConfig {
                port: port2,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            },
        ],
    )
    .await;

    let app = ctx.create_app().await;
    let resp = app
        .oneshot(make_chat_request(&chat_payload()))
        .await
        .unwrap();

    // The request should succeed after retry (first attempt fails, second succeeds)
    assert_eq!(resp.status(), StatusCode::OK);

    // Verify that a retry occurred: total requests across both workers should be >= 2
    let reqs1 = get_captured_requests(port1);
    let reqs2 = get_captured_requests(port2);
    let total = reqs1.len() + reqs2.len();
    assert!(
        total >= 2,
        "Expected at least 2 total requests (1 failure + 1 retry success), got {}",
        total
    );

    ctx.shutdown().await;
}

// =============================================================================
// Test 2: Circuit breaker opens after repeated failures
// =============================================================================

#[tokio::test]
async fn test_circuit_breaker_opens_after_repeated_failures() {
    let port1 = 21003;
    let port2 = 21004;

    let config = RouterConfig {
        circuit_breaker: CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 1,
            timeout_duration_secs: 60,
            window_duration_secs: 120,
        },
        retry: RetryConfig {
            max_retries: 0, // No retries - we want to see the circuit breaker in action
            ..RetryConfig::default()
        },
        ..default_config()
    };

    let ctx = TestContext::new_with_config(
        config,
        vec![
            MockWorkerConfig {
                port: port1,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 1.0, // Always fails
            },
            MockWorkerConfig {
                port: port2,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0, // Always succeeds
            },
        ],
    )
    .await;

    // Send enough requests to trip the circuit breaker on worker 1
    // With failure_threshold=2, after 2 failures worker 1's circuit should open
    for _ in 0..5 {
        let app = ctx.create_app().await;
        let _resp = app
            .oneshot(make_chat_request(&chat_payload()))
            .await
            .unwrap();
    }

    // Clear capture and send more requests
    clear_captured_requests(port1);
    clear_captured_requests(port2);

    // After circuit breaker opens, requests should mostly go to worker 2
    for _ in 0..3 {
        let app = ctx.create_app().await;
        let resp = app
            .oneshot(make_chat_request(&chat_payload()))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    let reqs2 = get_captured_requests(port2);
    assert!(
        reqs2.len() >= 2,
        "Expected most requests to go to worker 2 after circuit breaker opened on worker 1, got {} requests to worker 2",
        reqs2.len()
    );

    ctx.shutdown().await;
}

// =============================================================================
// Test 3: Streaming response has SSE content type
// =============================================================================

#[tokio::test]
async fn test_streaming_response_has_sse_content_type() {
    let port = 21005;

    let ctx = TestContext::new_with_config(
        default_config(),
        vec![MockWorkerConfig {
            port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }],
    )
    .await;

    let payload = json!({
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": true
    });

    let app = ctx.create_app().await;
    let resp = app.oneshot(make_chat_request(&payload)).await.unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.contains("text/event-stream"),
        "Expected text/event-stream content type for streaming, got: {}",
        content_type
    );

    // Read the body and verify it contains SSE events
    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_str = String::from_utf8_lossy(&body_bytes);
    assert!(
        body_str.contains("[DONE]"),
        "Expected [DONE] in streaming response body"
    );

    ctx.shutdown().await;
}

// =============================================================================
// Test 4: Header forwarding to backend
// =============================================================================

#[tokio::test]
async fn test_header_forwarding_to_backend() {
    let port = 21006;

    let ctx = TestContext::new_with_config(
        default_config(),
        vec![MockWorkerConfig {
            port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }],
    )
    .await;

    let app = ctx.create_app().await;

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(CONTENT_TYPE, "application/json")
        .header("x-custom-header", "test-value-123")
        .header("authorization", "Bearer test-token")
        .body(Body::from(serde_json::to_string(&chat_payload()).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Check that the mock worker received our custom headers
    let captured = get_captured_requests(port);
    assert!(
        !captured.is_empty(),
        "Expected at least one captured request"
    );

    let req = &captured[0];
    assert_eq!(
        req.headers.get("x-custom-header").map(|s| s.as_str()),
        Some("test-value-123"),
        "Custom header should be forwarded to backend"
    );
    assert_eq!(
        req.headers.get("authorization").map(|s| s.as_str()),
        Some("Bearer test-token"),
        "Authorization header should be forwarded to backend"
    );

    ctx.shutdown().await;
}

// =============================================================================
// Test 5: Unknown JSON fields pass through to backend
// This test validates the core value of the migration: fields the router
// doesn't know about should arrive at the backend unchanged.
// NOTE: This test will FAIL before the migration (typed struct strips unknown
// fields) and PASS after (Value pass-through preserves them).
// =============================================================================

#[tokio::test]
async fn test_unknown_fields_pass_through() {
    let port = 21007;

    let ctx = TestContext::new_with_config(
        default_config(),
        vec![MockWorkerConfig {
            port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }],
    )
    .await;

    let app = ctx.create_app().await;

    // Send a chat request with extra fields that ChatCompletionRequest doesn't know about
    let payload = json!({
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": false,
        "reasoning": {"effort": "high"},
        "custom_vllm_param": 42,
        "future_field": "should_survive"
    });

    let resp = app.oneshot(make_chat_request(&payload)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Check that the mock worker received the complete JSON including unknown fields
    let captured = get_captured_requests(port);
    assert!(
        !captured.is_empty(),
        "Expected at least one captured request"
    );

    let body = captured[0]
        .body
        .as_ref()
        .expect("Expected captured request to have a body");

    assert_eq!(
        body.get("custom_vllm_param").and_then(|v| v.as_i64()),
        Some(42),
        "Unknown field 'custom_vllm_param' should be preserved in pass-through"
    );
    assert_eq!(
        body.get("future_field").and_then(|v| v.as_str()),
        Some("should_survive"),
        "Unknown field 'future_field' should be preserved in pass-through"
    );
    assert!(
        body.get("reasoning").is_some(),
        "Unknown field 'reasoning' should be preserved in pass-through"
    );

    ctx.shutdown().await;
}

// =============================================================================
// Test 6: Non-streaming chat completion returns valid response
// =============================================================================

#[tokio::test]
async fn test_non_streaming_chat_completion_success() {
    let port = 21008;

    let ctx = TestContext::new_with_config(
        default_config(),
        vec![MockWorkerConfig {
            port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }],
    )
    .await;

    let app = ctx.create_app().await;
    let resp = app
        .oneshot(make_chat_request(&chat_payload()))
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();

    assert_eq!(
        body.get("object").and_then(|v| v.as_str()),
        Some("chat.completion")
    );
    assert!(
        body.get("choices").is_some(),
        "Response should have choices"
    );

    ctx.shutdown().await;
}

// =============================================================================
// Test 7: Request without model field routes successfully
// =============================================================================

#[tokio::test]
async fn test_request_without_model_field_routes_successfully() {
    let port = 21009;

    let ctx = TestContext::new_with_config(
        default_config(),
        vec![MockWorkerConfig {
            port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }],
    )
    .await;

    // Send request WITHOUT a model field — should fall back to get_all() workers
    let payload = json!({
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": false
    });

    let app = ctx.create_app().await;
    let resp = app.oneshot(make_chat_request(&payload)).await.unwrap();

    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "Request without model field should succeed by routing to any available worker"
    );

    ctx.shutdown().await;
}

// =============================================================================
// Test 8: Request with unknown model returns 503
// =============================================================================

#[tokio::test]
async fn test_request_with_unknown_model_returns_503() {
    let port = 21010;

    let ctx = TestContext::new_with_config(
        default_config(),
        vec![MockWorkerConfig {
            port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }],
    )
    .await;

    // Send request with a model name that no worker is registered for.
    // Workers are registered with model_id=None, so get_by_model_fast("nonexistent")
    // returns empty, and route_chat returns 503.
    let payload = json!({
        "model": "nonexistent-model",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": false
    });

    let app = ctx.create_app().await;
    let resp = app.oneshot(make_chat_request(&payload)).await.unwrap();

    assert_eq!(
        resp.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "Request with unknown model should return 503 (model-specific routing active in route_chat)"
    );

    ctx.shutdown().await;
}

// =============================================================================
// Test 9: route_chat retries on failure, route_transparent does not
// =============================================================================

#[tokio::test]
async fn test_route_chat_retries_on_failure_unlike_transparent() {
    let port1 = 21011;
    let port2 = 21012;

    let config = RouterConfig {
        retry: RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 10,
            max_backoff_ms: 100,
            backoff_multiplier: 2.0,
            jitter_factor: 0.0,
        },
        ..default_config()
    };

    let ctx = TestContext::new_with_config(
        config,
        vec![
            MockWorkerConfig {
                port: port1,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            },
            MockWorkerConfig {
                port: port2,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            },
        ],
    )
    .await;

    // Part 1: route_chat (/v1/chat/completions) SHOULD retry on failure
    set_deterministic_failures(port1, vec![1]);
    set_deterministic_failures(port2, vec![1]);

    let chat_payload = json!({
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": false
    });

    let app = ctx.create_app().await;
    let chat_req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_string(&chat_payload).unwrap()))
        .unwrap();
    let chat_resp = app.oneshot(chat_req).await.unwrap();

    assert_eq!(
        chat_resp.status(),
        StatusCode::OK,
        "route_chat should succeed via retry after initial failure"
    );

    // Part 2: route_transparent (/v1/responses) should NOT retry
    clear_deterministic_failures(port1);
    clear_deterministic_failures(port2);
    set_deterministic_failures(port1, vec![1]);
    set_deterministic_failures(port2, vec![1]);

    let responses_payload = json!({
        "model": "test",
        "input": "Hello!"
    });

    let app = ctx.create_app().await;
    let responses_req = Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::to_string(&responses_payload).unwrap(),
        ))
        .unwrap();
    let responses_resp = app.oneshot(responses_req).await.unwrap();

    assert!(
        responses_resp.status().is_server_error(),
        "route_transparent should return server error (no retry): got {}",
        responses_resp.status()
    );

    ctx.shutdown().await;
}
