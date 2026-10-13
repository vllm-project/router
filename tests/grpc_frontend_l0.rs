//! L0 integration tests with local models and a mock gRPC worker.
mod common;
#[path = "common/grpc_frontend_fixture.rs"]
mod fixture;

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use common::mock_vllm_rs::MockVllmRsServer;
use fixture::{chat_request, model_fixture};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serde_json::{json, Value};
use tower::ServiceExt;
use vllm_router_rs::backend::EngineFrontend;
use vllm_router_rs::config::{PolicyConfig, RetryConfig, RouterConfig, RoutingMode};
use vllm_router_rs::routers::http::router::Router;
use vllm_router_rs::tokenizer::TokenizerCacheConfig;

fn metric(metrics: &PrometheusHandle, name: &str) -> f64 {
    metrics
        .render()
        .lines()
        .find_map(|line| {
            line.strip_prefix(&format!("{name} "))
                .and_then(|n| n.parse().ok())
        })
        .unwrap_or(0.0)
}

#[tokio::test]
async fn grpc_frontend_cache_parity_metrics_streaming_and_retries() {
    // Keep one test: the recorder and env are process-wide.
    let metrics = PrometheusBuilder::new().install_recorder().unwrap();
    let model = model_fixture();
    let other_model = model_fixture();
    let worker = MockVllmRsServer::spawn().await;
    let off = EngineFrontend::new();
    let on =
        EngineFrontend::with_tokenizer_cache(Duration::from_secs(30), Default::default()).unwrap();
    let request = chat_request(&model, "Hello 世界", false);
    let expected = off.prepare(request.clone()).await.unwrap();
    let first = on.prepare(request.clone()).await.unwrap();
    let second = on.prepare(request.clone()).await.unwrap();
    assert_eq!(first.tokenized.token_ids, expected.tokenized.token_ids);
    assert_eq!(second.tokenized.token_ids, expected.tokenized.token_ids);
    let stats = on.tokenizer_cache_stats().unwrap();
    assert_eq!((stats.misses, stats.hits, stats.entries), (1, 1, 1));
    assert_eq!(
        metric(&metrics, "vllm_tokenizer_cache_bytes"),
        stats.bytes as f64
    );
    assert_eq!(metric(&metrics, "vllm_tokenizer_cache_entries"), 1.0);
    assert_eq!(metric(&metrics, "vllm_tokenizer_cache_hits_total"), 1.0);
    assert_eq!(metric(&metrics, "vllm_tokenizer_cache_misses_total"), 1.0);

    let mut special = request.clone();
    special
        .other
        .insert("add_special_tokens".into(), json!(true));
    let special_ids = on
        .prepare(special.clone())
        .await
        .unwrap()
        .tokenized
        .token_ids;
    assert_ne!(special_ids, first.tokenized.token_ids);
    assert_eq!(
        special_ids,
        off.prepare(special).await.unwrap().tokenized.token_ids
    );
    let mut changed = request.clone();
    changed.chat_template_kwargs =
        Some(serde_json::from_value(json!({"suffix": "changed"})).unwrap());
    assert_ne!(
        on.prepare(changed).await.unwrap().tokenized.token_ids,
        first.tokenized.token_ids
    );
    on.prepare(chat_request(&other_model, "Hello 世界", false))
        .await
        .unwrap();
    assert_eq!(on.tokenizer_cache_stats().unwrap().misses, 4);

    for salt in ["tenant-a", "tenant-b"] {
        let mut salted = request.clone();
        salted.other.insert("cache_salt".into(), json!(salt));
        let before = on.tokenizer_cache_stats().unwrap();
        for _ in 0..2 {
            let prepared = on.prepare(salted.clone()).await.unwrap();
            assert_eq!(prepared.tokenized.token_ids, expected.tokenized.token_ids);
            let response = on.dispatch(&worker.grpc_url, prepared).await;
            assert_eq!(response.status(), 200);
            to_bytes(response.into_body(), usize::MAX).await.unwrap();
            assert_eq!(worker.captured().last().unwrap().cache_salt, salt);
        }
        let after = on.tokenizer_cache_stats().unwrap();
        assert_eq!(after.misses - before.misses, 1);
        assert_eq!(after.hits - before.hits, 1);
    }

    // Cache hits must not skip validation.
    let before = on.tokenizer_cache_stats().unwrap();
    let mut invalid = request.clone();
    invalid.continue_final_message = true;
    assert!(on.prepare(invalid.clone()).await.is_err());
    assert!(off.prepare(invalid).await.is_err());
    assert_eq!(on.tokenizer_cache_stats().unwrap(), before);

    for frontend in [&off, &on] {
        for stream in [false, true] {
            let prepared = frontend
                .prepare(chat_request(&model, "Hello 世界", stream))
                .await
                .unwrap();
            let response = frontend.dispatch(&worker.grpc_url, prepared).await;
            assert_eq!(response.status(), 200);
            let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            if stream {
                let text = std::str::from_utf8(&body).unwrap();
                assert!(text.contains("[DONE]"));
                assert!(text.contains("chat.completion.chunk"));
            } else {
                let value: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(
                    value["usage"]["prompt_tokens"],
                    expected.tokenized.token_ids.len()
                );
            }
        }
    }
    for captured in worker.captured() {
        assert_eq!(captured.token_ids, expected.tokenized.token_ids.as_ref());
    }
    drop(first);
    drop(second);
    drop(on);
    assert_eq!(metric(&metrics, "vllm_tokenizer_cache_bytes"), 0.0);
    assert_eq!(metric(&metrics, "vllm_tokenizer_cache_entries"), 0.0);

    // BPE dropout must bypass L0.
    let stochastic = model_fixture();
    let path = stochastic.path().join("tokenizer.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["model"]["dropout"] = json!(0.1);
    std::fs::write(path, config.to_string()).unwrap();
    let frontend =
        EngineFrontend::with_tokenizer_cache(Duration::from_secs(30), Default::default()).unwrap();
    for _ in 0..2 {
        frontend
            .prepare(chat_request(&stochastic, "Hello", false))
            .await
            .unwrap();
    }
    assert_eq!(
        frontend.tokenizer_cache_stats().unwrap(),
        Default::default()
    );

    // Retries must reuse prepared tokens.
    std::env::set_var("VLLM_ROUTER_L0_CACHE", "1");
    let config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![worker.grpc_url.clone()],
        },
        policy: PolicyConfig::RoundRobin,
        disable_circuit_breaker: true,
        retry: RetryConfig {
            max_retries: 2,
            initial_backoff_ms: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let context = common::create_test_context(config.clone());
    let router = Router::new(vec![worker.grpc_url.clone()], &context)
        .await
        .unwrap();
    std::env::remove_var("VLLM_ROUTER_L0_CACHE");
    let app = common::test_app::create_test_app(Arc::new(router), reqwest::Client::new(), &config);
    let misses = metric(&metrics, "vllm_tokenizer_cache_misses_total");
    let hits = metric(&metrics, "vllm_tokenizer_cache_hits_total");
    let captured = worker.captured().len();
    worker.state.failures_remaining.store(1, Ordering::Relaxed);
    for _ in 0..2 {
        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        to_bytes(response.into_body(), usize::MAX).await.unwrap();
    }
    assert_eq!(
        metric(&metrics, "vllm_tokenizer_cache_misses_total") - misses,
        1.0
    );
    assert_eq!(
        metric(&metrics, "vllm_tokenizer_cache_hits_total") - hits,
        1.0
    );
    let attempts = worker.captured();
    assert_eq!(attempts.len() - captured, 3);
    for attempt in &attempts[captured..] {
        assert_eq!(attempt.token_ids, expected.tokenized.token_ids.as_ref());
    }

    http_ignores_l0_config(&request).await;

    // Oversized inputs must preserve cached entries.
    let small = EngineFrontend::with_tokenizer_cache(
        Duration::from_secs(30),
        TokenizerCacheConfig {
            max_entry_bytes: 512,
            ..Default::default()
        },
    )
    .unwrap();
    small
        .prepare(chat_request(&model, "tiny", false))
        .await
        .unwrap();
    let before = small.tokenizer_cache_stats().unwrap();
    let oversized = metric(&metrics, "vllm_tokenizer_cache_oversized_total");
    small
        .prepare(chat_request(&model, &"large ".repeat(1000), false))
        .await
        .unwrap();
    assert_eq!(small.tokenizer_cache_stats().unwrap().bytes, before.bytes);
    assert_eq!(
        metric(&metrics, "vllm_tokenizer_cache_oversized_total") - oversized,
        1.0
    );
}

async fn http_ignores_l0_config(request: &vllm_router_rs::protocols::spec::ChatCompletionRequest) {
    let worker = axum::Router::new()
        .route("/health", axum::routing::get(|| async { "ok" }))
        .route(
            "/v1/chat/completions",
            axum::routing::post(|axum::Json(body): axum::Json<Value>| async move {
                assert!(body.get("messages").is_some());
                assert!(body.get("token_ids").is_none());
                axum::Json(json!({"from": "http-worker"}))
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, worker).await.unwrap();
    });
    // HTTP ignores invalid L0 settings.
    std::env::set_var("VLLM_ROUTER_L0_CACHE", "1");
    std::env::set_var("VLLM_ROUTER_L0_MAX_BYTES", "invalid");
    let config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![url.clone()],
        },
        ..Default::default()
    };
    let router = Router::new(vec![url], &common::create_test_context(config.clone()))
        .await
        .unwrap();
    std::env::remove_var("VLLM_ROUTER_L0_CACHE");
    std::env::remove_var("VLLM_ROUTER_L0_MAX_BYTES");
    let app = common::test_app::create_test_app(Arc::new(router), reqwest::Client::new(), &config);
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let value: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(value, json!({"from": "http-worker"}));
    server.abort();
}
