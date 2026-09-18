//! This-version wire: HTTP forwards `messages`, gRPC forwards `token_ids`.
//!
//! In-process mocks (no GPU, no live `vllm-rs`). HTTP worker body must
//! contain `messages` and must not contain `token_ids`. gRPC
//! `GenerateRequest` must contain `token_ids` and must not be a text
//! prompt. That locks **this PR's default contract**, not a forever ban
//! on a gRPC text fallback or HTTP token-id input.
//!
//! Uses `tests/common/mock_vllm_rs.rs`. Pin fake ids via
//! `Router::pin_test_token_ids` so CI does not load a real tokenizer.
//!
//!   cargo test --test grpc_vs_http_e2e

mod common;

use std::sync::{Arc, Mutex};

use axum::{extract::Json, routing::get, routing::post, Router as AxumRouter};
use common::create_test_context;
use common::mock_vllm_rs::MockVllmRsServer;
use common::test_app::create_test_app;
use serde_json::json;
use tokio::net::TcpListener;
use tower::ServiceExt;
use vllm_router_rs::config::{
    CircuitBreakerConfig, ConnectionMode, HistoryBackend, PolicyConfig, RetryConfig, RouterConfig,
    RoutingMode,
};
use vllm_router_rs::core::WorkerFactory;
use vllm_router_rs::routers::http::router::Router;

fn test_config(worker_urls: Vec<String>) -> RouterConfig {
    RouterConfig {
        mode: RoutingMode::Regular { worker_urls },
        policy: PolicyConfig::RoundRobin,
        host: "127.0.0.1".to_string(),
        port: 3002,
        max_payload_size: 16 * 1024 * 1024,
        request_timeout_secs: 30,
        worker_startup_timeout_secs: 5,
        worker_startup_check_interval_secs: 1,
        intra_node_data_parallel_size: 1,
        api_key: None,
        api_key_validation_urls: vec![],
        discovery: None,
        metrics: None,
        log_dir: None,
        log_level: None,
        request_id_headers: None,
        max_concurrent_requests: 64,
        queue_size: 0,
        queue_timeout_secs: 60,
        rate_limit_tokens_per_second: None,
        cors_allowed_origins: vec![],
        retry: RetryConfig {
            max_retries: 1,
            ..RetryConfig::default()
        },
        circuit_breaker: CircuitBreakerConfig::default(),
        disable_retries: true,
        disable_circuit_breaker: true,
        health_check: vllm_router_rs::config::HealthCheckConfig::default(),
        enable_igw: false,
        connection_mode: ConnectionMode::Http,
        history_backend: HistoryBackend::Memory,
        enable_profiling: false,
        profile_timeout_secs: 10,
        kv_connector: vllm_router_rs::config::KvConnector::Nixl,
    }
}

async fn capturing_http_worker() -> (String, Arc<Mutex<Vec<serde_json::Value>>>) {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let state = captured.clone();
    let app = AxumRouter::new()
        .route("/health", get(|| async { "ok" }))
        .route(
            "/v1/chat/completions",
            post(move |Json(body): Json<serde_json::Value>| {
                let state = state.clone();
                async move {
                    state.lock().unwrap().push(body);
                    axum::Json(json!({
                        "id": "chatcmpl-http",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "test-model",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "hello from worker"},
                            "finish_reason": "stop"
                        }],
                        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
                    }))
                }
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    (format!("http://{addr}"), captured)
}

const CHAT_BODY: &str = r#"{
    "model": "test-model",
    "messages": [{"role": "user", "content": "Hello world"}],
    "max_tokens": 8,
    "stream": false
}"#;

#[test]
fn worker_factory_auto_detects_grpc_scheme() {
    let grpc = WorkerFactory::create_regular("grpc://127.0.0.1:50051".into());
    let http = WorkerFactory::create_regular("http://127.0.0.1:8000".into());
    assert_eq!(
        grpc.connection_mode(),
        vllm_router_rs::core::ConnectionMode::Grpc
    );
    assert_eq!(
        http.connection_mode(),
        vllm_router_rs::core::ConnectionMode::Http
    );
}

#[tokio::test]
async fn system_grpc_chat_sends_token_ids_not_messages() {
    let grpc = MockVllmRsServer::spawn().await;
    let config = test_config(vec![grpc.grpc_url.clone()]);
    let ctx = create_test_context(config.clone());
    let router = Router::new(vec![grpc.grpc_url.clone()], &ctx)
        .await
        .expect("grpc worker should pass grpc.health.v1");
    router.pin_test_token_ids(vec![1, 2, 3]);

    let app = create_test_app(Arc::new(router), reqwest::Client::new(), &config);
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(CHAT_BODY))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        response.status().is_success(),
        "status {}",
        response.status()
    );
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(
        json["choices"][0]["message"]["content"],
        "hello from worker"
    );

    let captured = grpc.captured();
    assert_eq!(captured.len(), 1);
    assert!(
        !captured[0].had_text_prompt,
        "worker must not receive a text prompt"
    );
    assert!(
        !captured[0].token_ids.is_empty(),
        "worker must receive token_ids from router preprocess"
    );
}

#[tokio::test]
async fn e2e_http_forwards_messages_grpc_forwards_token_ids() {
    let (http_url, http_bodies) = capturing_http_worker().await;
    let grpc = MockVllmRsServer::spawn().await;

    let http_config = test_config(vec![http_url.clone()]);
    let http_ctx = create_test_context(http_config.clone());
    let http_router = Router::new(vec![http_url], &http_ctx).await.unwrap();
    let http_app = create_test_app(Arc::new(http_router), reqwest::Client::new(), &http_config);

    let grpc_config = test_config(vec![grpc.grpc_url.clone()]);
    let grpc_ctx = create_test_context(grpc_config.clone());
    let grpc_router = Router::new(vec![grpc.grpc_url.clone()], &grpc_ctx)
        .await
        .unwrap();
    grpc_router.pin_test_token_ids(vec![1, 2, 3]);
    let grpc_app = create_test_app(Arc::new(grpc_router), reqwest::Client::new(), &grpc_config);

    let http_resp = http_app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(CHAT_BODY))
                .unwrap(),
        )
        .await
        .unwrap();
    let grpc_resp = grpc_app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(CHAT_BODY))
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(http_resp.status().is_success());
    assert!(grpc_resp.status().is_success());

    let http_json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(http_resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    let grpc_json: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(grpc_resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        http_json["choices"][0]["message"]["content"],
        grpc_json["choices"][0]["message"]["content"]
    );
    assert_eq!(
        http_json["choices"][0]["message"]["content"],
        "hello from worker"
    );

    let http_seen = http_bodies.lock().unwrap();
    assert_eq!(http_seen.len(), 1);
    assert!(
        http_seen[0].get("messages").is_some(),
        "HTTP worker must see OpenAI messages: {}",
        http_seen[0]
    );
    assert!(
        http_seen[0].get("token_ids").is_none(),
        "HTTP chat must not carry token_ids"
    );

    let grpc_seen = grpc.captured();
    assert_eq!(grpc_seen.len(), 1);
    assert!(!grpc_seen[0].had_text_prompt);
    assert!(!grpc_seen[0].token_ids.is_empty());
}

#[tokio::test]
async fn dynamically_added_grpc_worker_uses_grpc_health() {
    let grpc = MockVllmRsServer::spawn().await;
    let config = test_config(Vec::new());
    let ctx = create_test_context(config);
    let router = Router::new(Vec::new(), &ctx).await.unwrap();

    let result = router.add_worker(&grpc.grpc_url).await.unwrap();
    assert!(result.contains("Successfully added worker"));
    assert_eq!(router.get_worker_urls(), vec![grpc.grpc_url]);
}

#[tokio::test]
async fn grpc_stream_emits_openai_compatible_usage_chunk() {
    let grpc = MockVllmRsServer::spawn().await;
    let config = test_config(vec![grpc.grpc_url.clone()]);
    let ctx = create_test_context(config.clone());
    let router = Router::new(vec![grpc.grpc_url.clone()], &ctx)
        .await
        .unwrap();
    router.pin_test_token_ids(vec![1, 2, 3]);
    let app = create_test_app(Arc::new(router), reqwest::Client::new(), &config);
    let body = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 8,
        "stream": true,
        "stream_options": {"include_usage": true}
    });
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(response.status().is_success());
    let text = String::from_utf8(
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    let chunks: Vec<serde_json::Value> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    let usage_chunks: Vec<_> = chunks
        .iter()
        .filter(|chunk| chunk.get("usage").is_some_and(|usage| !usage.is_null()))
        .collect();
    assert_eq!(usage_chunks.len(), 1, "{text}");
    assert_eq!(usage_chunks[0]["choices"], json!([]));
    assert_eq!(
        usage_chunks[0]["usage"],
        json!({
            "prompt_tokens": 3,
            "completion_tokens": 3,
            "total_tokens": 6
        })
    );
    assert!(chunks
        .iter()
        .filter(|chunk| chunk["choices"]
            .as_array()
            .is_some_and(|items| !items.is_empty()))
        .all(|chunk| chunk.get("usage").is_none() || chunk["usage"].is_null()));
    assert!(text.contains("data: [DONE]"));
}
