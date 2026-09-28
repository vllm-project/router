//! Integration tests for WASM OnRequest middleware with mock HTTP workers.
//!
//! Builds the example component via `examples/wasm_middleware/build.sh` on demand.

mod common;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use common::mock_worker::{self, HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use common::test_app::create_test_app_with_wasm;
use http_body_util::BodyExt;
use reqwest::Client;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use tower::ServiceExt;
use vllm_router_rs::config::{RouterConfig, RoutingMode};
use vllm_router_rs::routers::RouterFactory;
use vllm_router_rs::wasm_middleware::{
    ensure_example_component_artifact, WasmMiddlewareConfig, WasmMiddlewareRuntime,
};

fn require_component_path() -> PathBuf {
    ensure_example_component_artifact().unwrap_or_else(|| {
        panic!(
            "failed to build/find the example WASM component via \
             `examples/wasm_middleware/build.sh` (wasm32-wasip2 target required)"
        )
    })
}

fn header_values<'a>(
    captured: &'a mock_worker::CapturedRequest,
    name: &str,
) -> Option<&'a Vec<String>> {
    captured
        .headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v)
}

#[tokio::test]
async fn wasm_middleware_modify_reject_and_path_isolation() {
    let component = require_component_path();
    let wasm_runtime = Arc::new(
        WasmMiddlewareRuntime::load(WasmMiddlewareConfig::from_path(component))
            .expect("load wasm runtime"),
    );

    let mut worker = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });
    let worker_url = worker.start().await.expect("start mock worker");
    let worker_port: u16 = worker_url
        .rsplit(':')
        .next()
        .unwrap()
        .parse()
        .expect("parse worker port");
    mock_worker::clear_captured_requests(worker_port);

    let config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![worker_url],
        },
        worker_startup_timeout_secs: 1,
        worker_startup_check_interval_secs: 1,
        ..Default::default()
    };
    let app_context = common::create_test_context(config.clone());
    let router = RouterFactory::create_router(&app_context)
        .await
        .expect("create router");
    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

    let app = create_test_app_with_wasm(
        Arc::from(router),
        Client::new(),
        &config,
        false,
        Some(wasm_runtime),
    );

    // Modify: chat should forward with the example header.
    let chat = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "mock-model",
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(chat.status(), StatusCode::OK);
    let _ = chat.into_body().collect().await.unwrap().to_bytes();

    let captured = mock_worker::get_captured_requests(worker_port);
    let chat_req = captured
        .iter()
        .find(|r| r.path == "/v1/chat/completions")
        .expect("mock worker should receive chat request");
    let marker = header_values(chat_req, "x-wasm-middleware")
        .and_then(|v| v.first())
        .map(String::as_str);
    assert_eq!(marker, Some("example"));

    // Reject: fail closed at the Router before the worker.
    mock_worker::clear_captured_requests(worker_port);
    let rejected = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "mock-model",
                        "messages": [{"role": "user", "content": "nope"}],
                        "note": "__wasm_reject__"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(rejected.status(), StatusCode::BAD_REQUEST);
    assert!(
        mock_worker::get_captured_requests(worker_port).is_empty(),
        "rejected requests must not reach the worker"
    );

    // Path isolation: completions is not attached by default.
    mock_worker::clear_captured_requests(worker_port);
    let completions = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "mock-model",
                        "prompt": "hi"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(completions.status(), StatusCode::OK);
    let _ = completions.into_body().collect().await.unwrap().to_bytes();

    let captured = mock_worker::get_captured_requests(worker_port);
    let completion_req = captured
        .iter()
        .find(|r| r.path == "/v1/completions")
        .expect("mock worker should receive completions request");
    assert!(
        header_values(completion_req, "x-wasm-middleware").is_none(),
        "default WASM attach point must not modify /v1/completions"
    );

    worker.stop().await;
}
