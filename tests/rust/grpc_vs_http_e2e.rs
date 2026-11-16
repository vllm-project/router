#![cfg(all(feature = "cpu", feature = "ready"))]

//! This-version wire: HTTP forwards `messages`, gRPC forwards `token_ids`.
//!
//! In-process mocks (no GPU, no live `vllm-rs`). HTTP worker body must
//! contain `messages` and must not contain `token_ids`. gRPC
//! `GenerateRequest` must contain `token_ids` and must not be a text
//! prompt. That locks **this PR's default contract**, not a forever ban
//! on a gRPC text fallback or HTTP token-id input.
//!
//! Uses `tests/rust/common/mock_vllm_rs.rs`. Pin fake ids via
//! `Router::pin_test_token_ids` so CI does not load a real tokenizer.
//!
//!   cargo test --test grpc_vs_http_e2e

mod common;

use std::sync::{Arc, Mutex};

use axum::{extract::Json, routing::get, routing::post, Router as AxumRouter};
use common::create_test_context;
use common::mock_vllm_rs::{MockVllmRs, MockVllmRsServer};
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

/// Startup uses real static source configuration and the normal gRPC health
/// probe. No fixture-only policy registration stands in for discovery.
async fn history_app(
    state: MockVllmRs,
) -> (
    Vec<MockVllmRsServer>,
    Arc<vllm_router_rs::server::AppContext>,
    AxumRouter,
    Vec<zmq::Socket>,
) {
    let mut workers = Vec::new();
    let mut publishers = Vec::new();
    let mut sources = serde_json::Map::new();
    for _ in 0..2 {
        let worker = MockVllmRsServer::spawn_with_state(MockVllmRs {
            captured: Default::default(),
            ..state.clone()
        })
        .await;
        let publisher = zmq::Context::new().socket(zmq::XPUB).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        sources.insert(
            worker.grpc_url.clone(),
            json!({
                "endpoints": [publisher.get_last_endpoint().unwrap().unwrap()],
                "block_size": 16, "model": "test-model"
            }),
        );
        workers.push(worker);
        publishers.push(publisher);
    }
    let urls: Vec<_> = workers
        .iter()
        .map(|worker| worker.grpc_url.clone())
        .collect();
    let mut config = test_config(urls.clone());
    config.kv_aware_completion_history = true;
    config.policy = serde_json::from_value(json!({
        "type": "kv_aware", "router_temperature": 0.0,
        "static_kv_event_sources": sources
    }))
    .unwrap();
    let ctx = create_test_context(config.clone());
    let router = Router::new(urls, &ctx).await.unwrap();
    router.pin_test_token_ids(vec![42; 33]);
    let app = create_test_app(Arc::new(router), reqwest::Client::new(), &config);
    (workers, ctx, app, publishers)
}

async fn history_request(app: AxumRouter, chat: bool, stream: bool) -> axum::response::Response {
    let (path, mut body) = if chat {
        (
            "/v1/chat/completions",
            json!({
                "model": "test-model", "messages": [{"role":"user", "content":"plain text"}]
            }),
        )
    } else {
        (
            "/v1/completions",
            json!({"model":"test-model", "prompt":vec![42;33]}),
        )
    };
    body["stream"] = stream.into();
    body["max_tokens"] = 8.into();
    app.oneshot(common::http::post(path, body.to_string()))
        .await
        .unwrap()
}

fn history_reason_count(
    handle: &metrics_exporter_prometheus::PrometheusHandle,
    reasons: &[&str],
) -> u64 {
    handle
        .render()
        .lines()
        .filter(|line| {
            line.starts_with("vllm_router_history_decisions_total{")
                && reasons
                    .iter()
                    .any(|reason| line.contains(&format!("reason=\"{reason}\"")))
        })
        .map(|line| {
            line.split_whitespace()
                .last()
                .unwrap()
                .parse::<u64>()
                .unwrap()
        })
        .sum()
}

/// Real frontend preparation -> GenerateStream -> valid output -> history.
/// The pinned tokenizer is a CPU fixture, not model/Worker compatibility proof.
#[tokio::test]
async fn grpc_history_observes_valid_output_for_completion_and_text_chat() {
    use common::mock_vllm_rs::HoldGate;
    use vllm_router_rs::kv_index::RoutableTarget;
    use vllm_router_rs::policies::KvAwarePolicy;
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    for (chat, stream) in [(false, false), (false, true), (true, false), (true, true)] {
        let gate = HoldGate::hold_all();
        let (workers, ctx, app, _publishers) = history_app(MockVllmRs {
            output_gate: Some(gate.clone()),
            ..Default::default()
        })
        .await;
        let policy = ctx.policy_registry.get_default_policy();
        let slots = policy
            .as_any()
            .downcast_ref::<KvAwarePolicy>()
            .unwrap()
            .slot_manager()
            .unwrap();
        let first = tokio::spawn(history_request(app.clone(), chat, stream));
        let selected = tokio::time::timeout(std::time::Duration::from_secs(3), async {
            loop {
                if let Some(index) = workers
                    .iter()
                    .position(|worker| !worker.captured().is_empty())
                {
                    break index;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let target = RoutableTarget {
            instance_id: workers[selected].grpc_url.clone().into(),
            dp_rank: 0,
        };
        let before = slots.sample(&target);
        assert_eq!(
            (before.active_prefill_tokens, before.active_requests),
            (33, 1),
            "chat={chat} stream={stream}"
        );
        let eligible = history_reason_count(&handle, &["eligible_ordinary", "override"]);
        gate.release();
        let response = first.await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let response = body_text(response).await;
        assert!(response.contains(if stream {
            "[DONE]"
        } else if chat {
            "chat.completion"
        } else {
            "text_completion"
        }));
        assert_eq!(slots.sample(&target), Default::default());
        assert_eq!(
            ctx.worker_registry
                .get_by_url(&workers[selected].grpc_url)
                .unwrap()
                .load(),
            0
        );
        assert_eq!(workers[selected].captured()[0].token_ids, vec![42; 33]);
        assert!(!workers[selected].captured()[0].had_text_prompt);
        let second = history_request(app, chat, stream).await;
        assert_eq!(second.status(), axum::http::StatusCode::OK);
        body_text(second).await;
        assert_eq!(
            workers[selected].captured().len(),
            2,
            "same-cost history choice chat={chat} stream={stream}"
        );
        assert_eq!(workers[1 - selected].captured().len(), 0);
        assert_eq!(
            history_reason_count(&handle, &["eligible_ordinary", "override"]),
            eligible + 1
        );
        assert_eq!(slots.sample(&target), Default::default());
    }
}

#[tokio::test]
async fn grpc_history_does_not_observe_open_error_abort_or_cancelled_stream() {
    use common::mock_vllm_rs::HoldGate;
    use http_body_util::BodyExt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use vllm_router_rs::kv_index::RoutableTarget;
    use vllm_router_rs::policies::KvAwarePolicy;
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    for case in ["error", "abort", "eof", "cancel"] {
        let gate = HoldGate::hold_all();
        let state = MockVllmRs {
            stream_failures_remaining: Arc::new(AtomicUsize::new(usize::from(case == "error"))),
            aborted_finishes_remaining: Arc::new(AtomicUsize::new(usize::from(case == "abort"))),
            frames: (case == "eof").then(Vec::new),
            output_gate: (case == "cancel").then(|| gate.clone()),
            ..Default::default()
        };
        let (workers, ctx, app, _publishers) = history_app(state).await;
        let policy = ctx.policy_registry.get_default_policy();
        let slots = policy
            .as_any()
            .downcast_ref::<KvAwarePolicy>()
            .unwrap()
            .slot_manager()
            .unwrap();
        let eligible = history_reason_count(&handle, &["eligible_ordinary", "override"]);
        let response = history_request(app.clone(), true, case != "abort").await;
        if case == "cancel" {
            assert_eq!(response.status(), axum::http::StatusCode::OK);
            let mut body = response.into_body();
            let frame = tokio::time::timeout(std::time::Duration::from_secs(3), body.frame())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            let bytes = frame.into_data().unwrap();
            assert!(String::from_utf8_lossy(&bytes).contains("assistant"));
            assert!(!String::from_utf8_lossy(&bytes).contains("hello from worker"));
            drop(body);
            tokio::time::timeout(std::time::Duration::from_secs(3), async {
                while workers[0].state.cancelled.load(Ordering::Relaxed) == 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            gate.release();
        } else {
            let text = body_text(response).await;
            assert!(
                text.contains(if case == "abort" { "abort" } else { "error" }),
                "{case}: {text}"
            );
        }
        for worker in &workers {
            let target = RoutableTarget {
                instance_id: worker.grpc_url.clone().into(),
                dp_rank: 0,
            };
            tokio::time::timeout(std::time::Duration::from_secs(3), async {
                while slots.sample(&target) != Default::default() {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            assert_eq!(
                ctx.worker_registry
                    .get_by_url(&worker.grpc_url)
                    .unwrap()
                    .load(),
                0
            );
        }
        let no_overlap = history_reason_count(&handle, &["no_overlap"]);
        let probe = history_request(app, true, false).await;
        body_text(probe).await;
        assert_eq!(
            history_reason_count(&handle, &["no_overlap"]),
            no_overlap + 1,
            "{case}"
        );
        assert_eq!(
            history_reason_count(&handle, &["eligible_ordinary", "override"]),
            eligible,
            "{case}"
        );
    }
}

fn test_config(worker_urls: Vec<String>) -> RouterConfig {
    RouterConfig {
        epd: None,
        mode: RoutingMode::Regular { worker_urls },
        pd_kv_event_sources: Default::default(),
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
        kv_aware_session_affinity_ttl_secs: None,
        kv_aware_completion_history: false,
        health_check: vllm_router_rs::config::HealthCheckConfig::default(),
        enable_igw: false,
        connection_mode: ConnectionMode::Http,
        history_backend: HistoryBackend::Memory,
        enable_profiling: false,
        profile_timeout_secs: 10,
        kv_connector: vllm_router_rs::config::KvConnector::Nixl,
        program_scheduling: None,
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
                .body(axum::body::Body::from({
                    let mut body: serde_json::Value = serde_json::from_str(CHAT_BODY).unwrap();
                    body["vllm_xargs"] = json!({"agentic_context": {"session_id": "session"}});
                    body["agent_hint"] = json!({"session_id": "hint"});
                    body["session_id"] = json!("fallback");
                    body.to_string()
                }))
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
async fn grpc_top_k_minus_one_is_client_error_without_worker_call() {
    let grpc = MockVllmRsServer::spawn().await;
    let config = test_config(vec![grpc.grpc_url.clone()]);
    let ctx = create_test_context(config.clone());
    let router = Router::new(vec![grpc.grpc_url.clone()], &ctx)
        .await
        .expect("grpc worker should pass grpc.health.v1");
    router.pin_test_token_ids(vec![1, 2, 3]);

    let app = create_test_app(Arc::new(router), reqwest::Client::new(), &config);
    let body = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 8,
        "top_k": -1,
        "stream": false
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

    assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
    let body = String::from_utf8(
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    assert!(body.contains("top_k=-1"), "{body}");
    assert!(
        grpc.captured().is_empty(),
        "conversion failure must not call the worker"
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

/// Non-streaming `/v1/completions` body shared by several tests.
fn completion_body() -> serde_json::Value {
    json!({"model": "test-model", "prompt": [1, 2, 3], "max_tokens": 8, "stream": false})
}

/// Router pinned to one in-process worker, plus that worker.
async fn completion_app(state: MockVllmRs) -> (MockVllmRsServer, AxumRouter) {
    let grpc = MockVllmRsServer::spawn_with_state(state).await;
    let config = test_config(vec![grpc.grpc_url.clone()]);
    let ctx = create_test_context(config.clone());
    let router = Router::new(vec![grpc.grpc_url.clone()], &ctx)
        .await
        .expect("grpc worker should pass grpc.health.v1");
    // Pinned ids differ from the client prompt ids below, so an assertion on
    // captured [1, 2, 3] proves the client ids reached the worker unchanged.
    router.pin_test_token_ids(vec![9, 9, 9]);
    (
        grpc,
        create_test_app(Arc::new(router), reqwest::Client::new(), &config),
    )
}

async fn post_completion(app: AxumRouter, body: serde_json::Value) -> axum::response::Response {
    app.oneshot(
        axum::http::Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap(),
    )
    .await
    .unwrap()
}

async fn body_text(response: axum::response::Response) -> String {
    String::from_utf8(
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap()
}

async fn body_json(response: axum::response::Response) -> serde_json::Value {
    serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap()
}

/// SSE `data:` payloads in order, `[DONE]` dropped.
fn sse_events(body: &str) -> Vec<serde_json::Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

/// What a completion client reads across chunks: joined `text` and every id.
fn streamed(chunks: &[serde_json::Value]) -> (String, Vec<u64>) {
    let text = chunks
        .iter()
        .map(|chunk| chunk["choices"][0]["text"].as_str().unwrap_or_default())
        .collect();
    let token_ids = chunks
        .iter()
        .flat_map(|chunk| {
            chunk["choices"][0]["token_ids"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .filter_map(|id| id.as_u64())
        .collect();
    (text, token_ids)
}

/// `/v1/completions` on a gRPC pool must answer with `text_completion`, not
/// the internal chat shape: Completion clients read `choices[].text`.
#[tokio::test]
async fn grpc_completions_reply_is_text_completion_shaped() {
    let (grpc, app) = completion_app(MockVllmRs::default()).await;

    let response = post_completion(app, completion_body()).await;
    assert!(
        response.status().is_success(),
        "status {}",
        response.status()
    );
    let json = body_json(response).await;
    assert_eq!(json["object"], "text_completion");
    assert_eq!(json["choices"][0]["text"], "hello from worker");
    assert!(
        json["choices"][0].get("message").is_none(),
        "completion clients read choices[].text: {json}"
    );
    assert!(
        json["choices"][0].get("prompt_token_ids").is_none(),
        "prompt_token_ids must follow return_token_ids: {json}"
    );
    assert!(
        json["id"].as_str().unwrap_or_default().starts_with("cmpl-"),
        "unexpected id {}",
        json["id"]
    );

    let captured = grpc.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].token_ids, vec![1, 2, 3]);
    assert!(!captured[0].had_text_prompt);
}

/// Streaming `/v1/completions` must emit `choices[].text` (plus the vLLM
/// `token_ids` used for token-exact replay validation), never `delta`.
#[tokio::test]
async fn grpc_completions_stream_carries_text_and_token_ids() {
    let (_grpc, app) = completion_app(MockVllmRs::default()).await;

    let response = post_completion(
        app,
        json!({
            "model": "test-model",
            "prompt": [1, 2, 3],
            "max_tokens": 8,
            "return_token_ids": true,
            "stream": true,
            "stream_options": {"include_usage": true}
        }),
    )
    .await;
    assert!(
        response.status().is_success(),
        "status {}",
        response.status()
    );
    let text = body_text(response).await;
    assert!(text.contains("data: [DONE]"), "{text}");

    let chunks = sse_events(&text);
    let choice_chunks: Vec<serde_json::Value> = chunks
        .iter()
        .filter(|chunk| {
            chunk["choices"]
                .as_array()
                .is_some_and(|items| !items.is_empty())
        })
        .cloned()
        .collect();
    assert!(!choice_chunks.is_empty(), "{text}");
    assert!(
        !choice_chunks[0]["choices"][0]["text"]
            .as_str()
            .unwrap_or_default()
            .is_empty(),
        "the first choice chunk must carry text, not a priming placeholder: {text}"
    );

    let (streamed_text, streamed_ids) = streamed(&choice_chunks);
    assert_eq!(streamed_text, "hello from worker");
    assert_eq!(streamed_ids, vec![7, 8, 9]);
    for chunk in &choice_chunks {
        assert_eq!(chunk["object"], "text_completion");
        assert!(
            chunk["choices"][0].get("delta").is_none(),
            "completion chunks must not use chat deltas: {chunk}"
        );
    }

    let usage_chunks: Vec<&serde_json::Value> = chunks
        .iter()
        .filter(|chunk| chunk.get("usage").is_some_and(|usage| !usage.is_null()))
        .collect();
    assert_eq!(usage_chunks.len(), 1, "{text}");
    assert_eq!(usage_chunks[0]["object"], "text_completion");
    assert_eq!(usage_chunks[0]["choices"], json!([]));
    assert_eq!(
        usage_chunks[0]["usage"],
        json!({
            "prompt_tokens": 3,
            "completion_tokens": 3,
            "total_tokens": 6
        })
    );
}

/// The decoder buffers byte-fallback fragments, so a frame can carry ids while
/// its text is empty. `return_token_ids` clients must still receive every id.
#[tokio::test]
async fn grpc_completions_stream_keeps_ids_from_textless_frames() {
    let (_grpc, app) = completion_app(MockVllmRs {
        frames: Some(vec![
            ("hello ".to_string(), vec![7]),
            (String::new(), vec![8]),
            ("from worker".to_string(), vec![9]),
        ]),
        ..Default::default()
    })
    .await;

    let response = post_completion(
        app,
        json!({
            "model": "test-model",
            "prompt": [1, 2, 3],
            "max_tokens": 8,
            "return_token_ids": true,
            "stream": true
        }),
    )
    .await;
    assert!(response.status().is_success());
    let text = body_text(response).await;

    let (streamed_text, streamed_ids) = streamed(&sse_events(&text));
    assert_eq!(streamed_text, "hello from worker", "{text}");
    assert_eq!(streamed_ids, vec![7, 8, 9], "{text}");
}

/// `token_ids` is opt-in, exactly as in the OpenAI schema.
#[tokio::test]
async fn grpc_completions_hides_token_ids_unless_requested() {
    let (_grpc, app) = completion_app(MockVllmRs::default()).await;

    let response = post_completion(
        app.clone(),
        json!({
            "model": "test-model",
            "prompt": [1, 2, 3],
            "max_tokens": 8,
            "stream": true
        }),
    )
    .await;
    let text = body_text(response).await;
    assert!(!text.contains("token_ids"), "{text}");

    let response = post_completion(app, completion_body()).await;
    let json = body_json(response).await;
    assert!(json["choices"][0].get("token_ids").is_none(), "{json}");
}

/// The non-streaming reply honors `return_token_ids` too.
#[tokio::test]
async fn grpc_completions_reply_carries_token_ids_when_requested() {
    let (_grpc, app) = completion_app(MockVllmRs::default()).await;

    let response = post_completion(
        app,
        json!({
            "model": "test-model",
            "prompt": [1, 2, 3],
            "max_tokens": 8,
            "return_token_ids": true
        }),
    )
    .await;
    assert!(response.status().is_success());
    let json = body_json(response).await;
    assert_eq!(json["choices"][0]["token_ids"], json!([7, 8, 9]));
    assert_eq!(json["choices"][0]["prompt_token_ids"], json!([1, 2, 3]));
    assert_eq!(json["choices"][0]["text"], "hello from worker");
}

#[tokio::test]
async fn grpc_completions_rejects_unsupported_requests() {
    let (grpc, app) = completion_app(MockVllmRs::default()).await;

    for body in [
        json!({"model": "test-model", "prompt": [1, -2, 3], "max_tokens": 8}),
        json!({"model": "test-model", "prompt": "hello", "max_tokens": 8}),
        json!({"model": "test-model", "prompt": [[1, 2]], "max_tokens": 8}),
        json!({"model": "test-model", "prompt": [1, 2, 3], "suffix": "tail"}),
        json!({"model": "test-model", "prompt": [1, 2, 3], "documents": [{"content": "x"}]}),
        json!({"model": "test-model", "prompt": [1, 2, 3], "guided_decoding_backend": "xgrammar"}),
    ] {
        let response = post_completion(app.clone(), body.clone()).await;
        assert_eq!(
            response.status(),
            axum::http::StatusCode::BAD_REQUEST,
            "{body}"
        );
    }
    assert!(
        grpc.captured().is_empty(),
        "an unusable request must not reach a worker"
    );
}
