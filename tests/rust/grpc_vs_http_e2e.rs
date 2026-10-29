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

mod prefill_progress_regressions {
    use super::*;
    use std::time::Duration;
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;
    use vllm_router_rs::backend::pb::{
        finish_info, FinishInfo, GenerateResponse, PrefillProgress, SequenceOutput,
    };
    use vllm_router_rs::core::Worker;
    use vllm_router_rs::kv_index::{CacheKey, HashMode, KvBlockIndexer, RoutableTarget};
    use vllm_router_rs::policies::{KvAwarePolicy, LoadSample, SlotManager};
    use vllm_router_rs::protocols::spec::ChatCompletionRequest;

    struct Harness {
        router: Arc<Router>,
        server: MockVllmRsServer,
        worker: Arc<dyn Worker>,
        slots: Arc<dyn SlotManager>,
        target: RoutableTarget,
        tx: mpsc::Sender<Result<GenerateResponse, tonic::Status>>,
    }

    async fn wait_until(mut condition: impl FnMut() -> bool) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while !condition() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("gRPC dispatch/accounting did not reach the expected state");
    }

    impl Harness {
        async fn new(enabled: bool, supported: Option<bool>) -> Self {
            let (tx, rx) = mpsc::channel(8);
            let state = MockVllmRs {
                prefill_progress_supported: supported,
                scripted_stream: Arc::new(Mutex::new(Some(Box::pin(ReceiverStream::new(rx))))),
                ..Default::default()
            };
            let server = MockVllmRsServer::spawn_with_state(state).await;
            let mut config = test_config(vec![server.grpc_url.clone()]);
            config.policy = PolicyConfig::KvAware {
                router_temperature: 0.0,
                kv_penalty_host: 0.8,
                kv_penalty_external: 0.6,
                kv_penalty_disk: 0.3,
                static_kv_event_sources: Default::default(),
            };
            config.prefill_progress_stale_after_secs = enabled.then_some(10);
            let ctx = create_test_context(config);
            let router = Router::new(vec![server.grpc_url.clone()], &ctx)
                .await
                .unwrap();
            router.pin_test_token_ids(vec![1; 100]);
            let policy = ctx.policy_registry.get_default_policy();
            let policy = policy.as_any().downcast_ref::<KvAwarePolicy>().unwrap();
            let target = RoutableTarget {
                instance_id: Arc::from("progress-worker"),
                dp_rank: 0,
            };
            policy.register_target_with_cache_identity(
                &server.grpc_url,
                target.clone(),
                CacheKey {
                    model: Arc::from("model"),
                    hash_mode: HashMode::Sha256,
                    block_size: 2,
                },
                Arc::new(KvBlockIndexer::new()),
            );
            Self {
                router: Arc::new(router),
                worker: ctx.worker_registry.get_by_url(&server.grpc_url).unwrap(),
                slots: policy.slot_manager().unwrap(),
                target,
                server,
                tx,
            }
        }

        async fn dispatch(
            &self,
            stream: bool,
        ) -> tokio::task::JoinHandle<axum::response::Response> {
            let router = self.router.clone();
            let request: ChatCompletionRequest = serde_json::from_value(json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": stream,
                "stream_options": {"include_usage": true},
                "max_tokens": 8,
            }))
            .unwrap();
            let task = tokio::spawn(async move {
                router
                    .route_typed_request(None, &request, "/v1/chat/completions", None)
                    .await
            });
            wait_until(|| !self.server.captured().is_empty()).await;
            task
        }

        fn assert_load(&self, remaining: u64, requests: usize) {
            assert_eq!(self.worker.load(), requests);
            assert_eq!(
                self.slots.sample(&self.target),
                LoadSample {
                    active_prefill_tokens: remaining,
                    active_requests: requests,
                }
            );
        }

        async fn progress(&self, sequence: u64, remaining: u32) {
            self.tx
                .send(Ok(GenerateResponse {
                    prefill_progress: Some(PrefillProgress {
                        request_id: self.server.captured()[0].request_id.clone(),
                        sequence,
                        prompt_tokens: 100,
                        remaining_compute_tokens: remaining,
                        ..Default::default()
                    }),
                    ..Default::default()
                }))
                .await
                .unwrap();
        }

        async fn output(&self, text: &str, finished: bool) {
            self.tx
                .send(Ok(GenerateResponse {
                    outputs: Some(SequenceOutput {
                        text: text.into(),
                        num_tokens: 1,
                        finish_info: finished.then_some(FinishInfo {
                            num_output_tokens: 2,
                            finish_reason: finish_info::FinishReason::Stop as i32,
                            ..Default::default()
                        }),
                        ..Default::default()
                    }),
                    ..Default::default()
                }))
                .await
                .unwrap();
        }
    }

    #[tokio::test]
    async fn grpc_progress_fallback_retains_admission_until_output() {
        use http_body_util::BodyExt;
        for stream in [false, true] {
            for (enabled, supported) in [(false, Some(true)), (true, Some(false)), (true, None)] {
                let h = Harness::new(enabled, supported).await;
                let mut task = Some(h.dispatch(stream).await);
                assert!(!h.server.captured()[0].include_prefill_progress);
                h.assert_load(100, 1);
                let mut response = if stream {
                    Some(task.take().unwrap().await.unwrap())
                } else {
                    None
                };
                h.output("hello", false).await;
                if let Some(response) = &mut response {
                    response.body_mut().frame().await.unwrap().unwrap();
                }
                wait_until(|| h.slots.sample(&h.target).active_prefill_tokens == 0).await;
                h.assert_load(0, 1);
                h.output(" world", true).await;
                drop(h.tx);
                let response = match response {
                    Some(response) => response,
                    None => task.take().unwrap().await.unwrap(),
                };
                assert!(response.status().is_success());
                axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap();
                wait_until(|| h.worker.load() == 0).await;
                assert_eq!(h.slots.sample(&h.target), LoadSample::default());
            }
        }
    }

    #[tokio::test]
    async fn grpc_progress_consumers_track_reprefill_and_hide_progress_frames() {
        use http_body_util::BodyExt;
        for stream in [false, true] {
            let h = Harness::new(true, Some(true)).await;
            let mut task = Some(h.dispatch(stream).await);
            assert!(h.server.captured()[0].include_prefill_progress);
            h.progress(2, 40).await;
            wait_until(|| h.slots.sample(&h.target).active_prefill_tokens == 40).await;
            h.assert_load(40, 1);
            h.progress(1, 90).await; // An older snapshot must not replace seq 2.
            let mut response = if stream {
                let mut response = task.take().unwrap().await.unwrap();
                assert!(response.status().is_success());
                assert!(
                    tokio::time::timeout(Duration::from_millis(50), response.body_mut().frame())
                        .await
                        .is_err(),
                    "progress-only frames must not produce SSE chunks"
                );
                Some(response)
            } else {
                None
            };
            h.output("hello", false).await;
            wait_until(|| h.slots.sample(&h.target).active_prefill_tokens == 0).await;
            h.assert_load(0, 1);
            h.progress(3, 25).await;
            wait_until(|| h.slots.sample(&h.target).active_prefill_tokens == 25).await;
            h.assert_load(25, 1);
            h.output(" world", true).await;
            drop(h.tx);
            let response = match response.take() {
                Some(response) => response,
                None => task.take().unwrap().await.unwrap(),
            };
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let text = std::str::from_utf8(&bytes).unwrap();
            assert!(!text.contains("prefill_progress"));
            if stream {
                let events: Vec<serde_json::Value> = text
                    .lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter(|line| *line != "[DONE]")
                    .map(|line| serde_json::from_str(line).unwrap())
                    .collect();
                assert_eq!(
                    events.len(),
                    3,
                    "only two output chunks and usage are expected: {text}"
                );
                assert_eq!(events[0]["choices"][0]["delta"]["content"], "hello");
                assert_eq!(events[1]["choices"][0]["delta"]["content"], " world");
                assert_eq!(events[2]["usage"]["prompt_tokens"], 100);
            } else {
                let reply: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(reply["choices"][0]["message"]["content"], "hello world");
                assert_eq!(reply["usage"]["prompt_tokens"], 100);
            }
            wait_until(|| h.worker.load() == 0).await;
            assert_eq!(h.slots.sample(&h.target), LoadSample::default());
        }
    }

    #[tokio::test]
    async fn grpc_progress_cancellation_releases_latest_balance() {
        for stream in [false, true] {
            let h = Harness::new(true, Some(true)).await;
            let task = h.dispatch(stream).await;
            h.progress(2, 40).await;
            wait_until(|| h.slots.sample(&h.target).active_prefill_tokens == 40).await;
            h.assert_load(40, 1);
            if stream {
                drop(task.await.unwrap());
            } else {
                task.abort();
                assert!(task.await.unwrap_err().is_cancelled());
            }
            wait_until(|| h.worker.load() == 0).await;
            assert_eq!(h.slots.sample(&h.target), LoadSample::default());
        }
    }
}

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
        prefill_progress_stale_after_secs: None,
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
