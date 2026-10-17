//! Real TCP ZMQ ingestion, HTTP forwarding, lifecycle and configuration checks.
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json,
};
use serde_json::{json, Value};
use vllm_router_rs::{
    config::{ConfigValidator, KvAwareConfig, PolicyConfig, RouterConfig, RoutingMode},
    core::{BasicWorker, Worker, WorkerType},
    kv_index::{
        routing::KvRoutingMode, wire::ExternalBlockHash, AllBlocksCleared, BlockRemoved,
        BlockStored, KVEvent,
    },
    policies::{KvAwarePolicy, PolicyFactory},
    protocols::spec::CompletionRequest,
    routers::http::router::Router,
    server::AppContext,
};

struct Publisher {
    _context: zmq::Context,
    socket: zmq::Socket,
    endpoint: String,
}
impl Publisher {
    fn new() -> Self {
        let context = zmq::Context::new();
        let socket = context.socket(zmq::XPUB).unwrap();
        socket.set_linger(0).unwrap();
        socket.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = socket.get_last_endpoint().unwrap().unwrap();
        Self {
            _context: context,
            socket,
            endpoint,
        }
    }
    async fn subscribed(&self) {
        let until = Instant::now() + Duration::from_secs(3);
        loop {
            if let Ok(message) = self.socket.recv_bytes(zmq::DONTWAIT) {
                if message == b"\x01kv" {
                    return;
                }
            }
            assert!(Instant::now() < until, "subscriber did not attach");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
    fn send(&self, rank: u32, sequence: u64, events: Vec<KVEvent>) {
        self.send_at(
            rank,
            sequence,
            events,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        );
    }
    fn send_at(&self, rank: u32, sequence: u64, events: Vec<KVEvent>, timestamp: f64) {
        let bytes = rmp_serde::to_vec_named(&(timestamp, events, rank)).unwrap();
        self.raw(sequence, &bytes);
    }
    fn raw(&self, sequence: u64, bytes: &[u8]) {
        self.socket
            .send_multipart([b"kv".as_slice(), &sequence.to_be_bytes(), bytes], 0)
            .unwrap();
    }
}
fn clear() -> KVEvent {
    KVEvent::AllBlocksCleared(AllBlocksCleared::default())
}
fn stored() -> KVEvent {
    KVEvent::BlockStored(BlockStored {
        extra_keys: None,
        block_hashes: vec![
            ExternalBlockHash::Bytes(vec![1; 32]),
            ExternalBlockHash::Bytes(vec![2; 32]),
        ],
        parent_block_hash: None,
        token_ids: vec![1, 2, 3, 4],
        block_size: 2,
        lora_id: None,
        medium: Some("GPU".into()),
        lora_name: None,
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
        locality: None,
        ownership: None,
        session_id: None,
    })
}
fn removed() -> KVEvent {
    KVEvent::BlockRemoved(BlockRemoved {
        block_hashes: vec![ExternalBlockHash::Bytes(vec![1; 32])],
        medium: Some("GPU".into()),
        group_idx: None,
        locality: None,
        ownership: None,
    })
}

#[derive(Clone)]
struct MockState {
    info: Value,
    sources: Value,
    captured: Arc<Mutex<Vec<(HeaderMap, Value)>>>,
}
struct Engine {
    url: String,
    state: MockState,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Engine {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Engine {
    async fn new(instance: &str, publishers: &[(u32, &Publisher)]) -> Self {
        let sources: serde_json::Map<String, Value> = publishers.iter().map(|(rank,publisher)| (rank.to_string(), json!({"endpoint": publisher.endpoint,"topic":"kv","publisher":"zmq","enable_kv_cache_events":true}))).collect();
        let state = MockState {
            info: json!({"instance_id":instance,"model_id":"m","kv_block_size":2}),
            sources: Value::Object(sources),
            captured: Arc::new(Mutex::new(Vec::new())),
        };
        let app = axum::Router::new()
            .route("/health", get(|| async { "ok" }))
            .route(
                "/get_server_info",
                get(|State(state): State<MockState>| async move { Json(state.info) }),
            )
            .route(
                "/kv_event_sources",
                get(|State(state): State<MockState>| async move { Json(state.sources) }),
            )
            .route("/v1/completions", post(complete))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self { url, state, task }
    }
    fn worker(&self) -> Arc<dyn Worker> {
        Arc::new(BasicWorker::new(self.url.clone(), WorkerType::Regular))
    }
}
async fn complete(
    State(state): State<MockState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    state
        .captured
        .lock()
        .unwrap()
        .push((headers.clone(), body.clone()));
    if body["prompt"][0] == 999 {
        return std::future::pending().await;
    }
    if body["prompt"][0] == 500 {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }
    if body["stream"] == true {
        let pending = futures_util::stream::pending::<Result<bytes::Bytes, std::io::Error>>();
        return (
            [("content-type", "text/event-stream")],
            Body::from_stream(pending),
        )
            .into_response();
    }
    Json(json!({"worker":state.info["instance_id"],"rank":headers.get("x-data-parallel-rank").and_then(|h|h.to_str().ok()),"prompt":body["prompt"]})).into_response()
}
fn policy() -> KvAwarePolicy {
    KvAwarePolicy::with_config(KvAwareConfig {
        cache_weight: 0.9,
        ..Default::default()
    })
}
async fn wait_for(
    policy: &KvAwarePolicy,
    workers: &[Arc<dyn Worker>],
    expected: KvRoutingMode,
) -> (usize, vllm_router_rs::kv_index::routing::KvRoutingDecision) {
    let until = Instant::now() + Duration::from_secs(3);
    loop {
        let result = policy
            .select_with_tokens(workers, Some("m"), Some(&[1, 2, 3, 4, 5]), true)
            .unwrap();
        if result.1.mode == expected {
            return result;
        }
        assert!(
            Instant::now() < until,
            "mode {:?}, expected {expected:?}",
            result.1
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}
async fn attach(policy: &KvAwarePolicy, engine: &Engine, publishers: &[&Publisher]) {
    policy
        .runtime
        .discover(&reqwest::Client::new(), &engine.url, None)
        .await
        .unwrap();
    for publisher in publishers {
        publisher.subscribed().await;
    }
}
async fn router(engines: &[&Engine], dp_size: usize) -> (Arc<Router>, Arc<AppContext>) {
    let urls: Vec<_> = engines.iter().map(|e| e.url.clone()).collect();
    let config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: urls.clone(),
        },
        policy: PolicyConfig::KvAware {
            config: KvAwareConfig {
                cache_weight: 0.9,
                ..Default::default()
            },
        },
        intra_node_data_parallel_size: dp_size,
        worker_startup_timeout_secs: 3,
        worker_startup_check_interval_secs: 1,
        disable_retries: true,
        ..Default::default()
    };
    let ctx =
        Arc::new(AppContext::new(config, reqwest::Client::new(), 64, None, Vec::new()).unwrap());
    (Arc::new(Router::new(urls, &ctx).await.unwrap()), ctx)
}
fn request(prompt: Value, stream: bool) -> CompletionRequest {
    serde_json::from_value(json!({"model":"m","prompt":prompt,"max_tokens":1,"stream":stream}))
        .unwrap()
}

#[tokio::test]
async fn real_event_ingestion_requires_clear_then_prefers_holding_worker_and_honors_eviction() {
    let a = Publisher::new();
    let b = Publisher::new();
    let e0 = Engine::new("idle", &[(0, &a)]).await;
    let e1 = Engine::new("holder", &[(0, &b)]).await;
    let policy = policy();
    let workers = vec![e0.worker(), e1.worker()];
    attach(&policy, &e0, &[&a]).await;
    attach(&policy, &e1, &[&b]).await;
    b.send(0, 1, vec![stored()]);
    assert_eq!(
        wait_for(&policy, &workers, KvRoutingMode::Fallback)
            .await
            .1
            .mode,
        KvRoutingMode::Fallback
    );
    a.send(0, 1, vec![clear()]);
    b.send(0, 2, vec![clear(), stored()]);
    assert_eq!(wait_for(&policy, &workers, KvRoutingMode::Exact).await.0, 1);
    // Advertised model aliases cannot broaden an explicit registry model pool.
    let other_label: Arc<dyn Worker> = Arc::new(
        BasicWorker::new(e1.url.clone(), WorkerType::Regular).with_labels(
            std::collections::HashMap::from([("model_id".into(), "other".into())]),
        ),
    );
    assert!(policy
        .select_with_tokens(&[other_label], Some("m"), Some(&[1, 2, 3, 4, 5]), true)
        .is_none());
    b.send(0, 3, vec![removed()]);
    let until = Instant::now() + Duration::from_secs(3);
    loop {
        let d = wait_for(&policy, &workers, KvRoutingMode::Exact).await.1;
        if d.scores.iter().all(|s| s.credited_blocks == 0.0) {
            break;
        }
        assert!(Instant::now() < until);
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[tokio::test]
async fn gaps_corruption_wrong_rank_and_old_timestamps_need_a_new_clear_barrier() {
    let publisher = Publisher::new();
    let engine = Engine::new("w", &[(0, &publisher)]).await;
    let policy = policy();
    let workers = vec![engine.worker()];
    attach(&policy, &engine, &[&publisher]).await;
    publisher.send(0, 1, vec![clear(), stored()]);
    wait_for(&policy, &workers, KvRoutingMode::Exact).await;
    publisher.send(0, 3, vec![stored()]);
    wait_for(&policy, &workers, KvRoutingMode::Fallback).await;
    publisher.send(0, 4, vec![stored()]);
    publisher.send(0, 5, vec![clear()]);
    wait_for(&policy, &workers, KvRoutingMode::Exact).await;
    publisher.raw(6, b"corrupt");
    wait_for(&policy, &workers, KvRoutingMode::Fallback).await;
    publisher.send(0, 7, vec![clear()]);
    wait_for(&policy, &workers, KvRoutingMode::Exact).await;
    publisher.send(1, 8, vec![stored()]);
    wait_for(&policy, &workers, KvRoutingMode::Fallback).await;
    publisher.send(0, 9, vec![clear()]);
    wait_for(&policy, &workers, KvRoutingMode::Exact).await;
    publisher.send_at(0, 10, vec![stored()], 1.0);
    wait_for(&policy, &workers, KvRoutingMode::Fallback).await;
    publisher.send(0, 1, vec![stored()]);
    publisher.send(0, 2, vec![clear()]);
    wait_for(&policy, &workers, KvRoutingMode::Exact).await;
}

#[tokio::test]
async fn freshness_duplicates_budget_and_removal_do_not_preserve_exact_credit() {
    let publisher = Publisher::new();
    let engine = Engine::new("w", &[(0, &publisher)]).await;
    let policy = KvAwarePolicy::with_config(KvAwareConfig {
        max_event_age_secs: 1,
        max_blocks: 2,
        ..Default::default()
    });
    let workers = vec![engine.worker()];
    attach(&policy, &engine, &[&publisher]).await;
    publisher.send(0, 1, vec![clear(), stored()]);
    wait_for(&policy, &workers, KvRoutingMode::Exact).await;
    tokio::time::sleep(Duration::from_millis(1100)).await;
    publisher.send(0, 1, vec![clear(), stored()]);
    wait_for(&policy, &workers, KvRoutingMode::Fallback).await;
    publisher.send(0, 2, vec![clear()]);
    wait_for(&policy, &workers, KvRoutingMode::Exact).await;
    publisher.send(0, 3, vec![stored()]);
    wait_for(&policy, &workers, KvRoutingMode::Fallback).await;
    publisher.send(0, 4, vec![clear()]);
    wait_for(&policy, &workers, KvRoutingMode::Exact).await;
    policy.runtime.remove_worker(&engine.url);
    let decision = policy
        .select_with_tokens(&workers, Some("m"), Some(&[1, 2, 3, 4, 5]), true)
        .unwrap()
        .1;
    assert_eq!(decision.fallback_reason, Some("undiscovered_worker"));
}

#[tokio::test]
async fn real_http_forwarding_selects_dp_rank_and_replaces_client_rank_header() {
    let p0 = Publisher::new();
    let p1 = Publisher::new();
    let engine = Engine::new("dp", &[(0, &p0), (1, &p1)]).await;
    let (router, ctx) = router(&[&engine], 2).await;
    p0.subscribed().await;
    p1.subscribed().await;
    let policy = ctx.policy_registry.get_default_policy();
    let kv = policy.as_any().downcast_ref::<KvAwarePolicy>().unwrap();
    let workers = ctx.worker_registry.get_all();
    p0.send(0, 1, vec![clear()]);
    p1.send(1, 1, vec![clear(), stored()]);
    wait_for(kv, &workers, KvRoutingMode::Exact).await;
    let mut headers = HeaderMap::new();
    headers.insert("x-data-parallel-rank", "0".parse().unwrap());
    let response = router
        .route_typed_request(
            Some(&headers),
            &request(json!([1, 2, 3, 4, 5]), false),
            "/v1/completions",
            Some("m"),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-router-kv-mode"], "exact");
    let body = axum::body::to_bytes(response.into_body(), 4096)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["rank"], "1");
    assert_eq!(body["prompt"], json!([1, 2, 3, 4, 5]));
    assert!(workers.iter().all(|w| w.load() == 0));
}

#[tokio::test]
async fn http_text_and_salted_prompts_fall_back_without_claiming_exact() {
    let publisher = Publisher::new();
    let engine = Engine::new("w", &[(0, &publisher)]).await;
    let (router, ctx) = router(&[&engine], 1).await;
    publisher.subscribed().await;
    let policy = ctx.policy_registry.get_default_policy();
    let kv = policy.as_any().downcast_ref::<KvAwarePolicy>().unwrap();
    publisher.send(0, 1, vec![clear(), stored()]);
    wait_for(kv, &ctx.worker_registry.get_all(), KvRoutingMode::Exact).await;
    let response = router
        .route_typed_request(
            None,
            &request(json!("hello"), false),
            "/v1/completions",
            Some("m"),
        )
        .await;
    assert_eq!(
        response.headers()["x-router-kv-reason"],
        "missing_prepared_tokens"
    );
    let salted: CompletionRequest =
        serde_json::from_value(json!({"model":"m","prompt":[1,2,3,4,5],"cache_salt":"tenant-b"}))
            .unwrap();
    let response = router
        .route_typed_request(None, &salted, "/v1/completions", Some("m"))
        .await;
    assert_eq!(
        response.headers()["x-router-kv-reason"],
        "unsupported_request_namespace"
    );
}

#[tokio::test]
async fn stream_drop_and_cancellation_before_headers_release_exactly_one_load_slot() {
    let engine = Engine::new("w", &[]).await;
    let (router, ctx) = router(&[&engine], 1).await;
    let worker = ctx.worker_registry.get_all()[0].clone();
    let response = router
        .route_typed_request(
            None,
            &request(json!([1, 2, 3]), true),
            "/v1/completions",
            Some("m"),
        )
        .await;
    assert_eq!(worker.load(), 1);
    drop(response);
    let until = Instant::now() + Duration::from_secs(2);
    while worker.load() != 0 {
        assert!(Instant::now() < until);
        tokio::task::yield_now().await;
    }
    let pending_router = router.clone();
    let task = tokio::spawn(async move {
        pending_router
            .route_typed_request(
                None,
                &request(json!([999, 2, 3]), false),
                "/v1/completions",
                Some("m"),
            )
            .await
    });
    let until = Instant::now() + Duration::from_secs(2);
    while engine.state.captured.lock().unwrap().len() < 2 {
        assert!(Instant::now() < until);
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(worker.load(), 1);
    task.abort();
    let _ = task.await;
    assert_eq!(worker.load(), 0);
    let response = router
        .route_typed_request(
            None,
            &request(json!([500, 2, 3]), false),
            "/v1/completions",
            Some("m"),
        )
        .await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(worker.load(), 0);
}

#[test]
fn config_is_registered_and_invalid_limits_and_disaggregated_modes_are_rejected() {
    let policy: PolicyConfig = serde_json::from_value(json!({"type":"kv_aware"})).unwrap();
    assert_eq!(
        PolicyFactory::create_from_config(&policy).name(),
        "kv_aware"
    );
    assert!(PolicyFactory::create_by_name("kv_aware").is_some());
    for config in [
        KvAwareConfig {
            cache_weight: f64::NAN,
            ..Default::default()
        },
        KvAwareConfig {
            max_sources: 0,
            ..Default::default()
        },
        KvAwareConfig {
            max_event_age_secs: 0,
            ..Default::default()
        },
        KvAwareConfig {
            max_blocks: 0,
            ..Default::default()
        },
    ] {
        let config = RouterConfig {
            policy: PolicyConfig::KvAware { config },
            mode: RoutingMode::Regular {
                worker_urls: vec!["http://worker:8000".into()],
            },
            ..Default::default()
        };
        assert!(ConfigValidator::validate(&config).is_err());
    }
    let config = RouterConfig {
        policy,
        enable_igw: true,
        ..Default::default()
    };
    assert!(ConfigValidator::validate(&config).is_err());
}

#[test]
fn discovery_failure_does_not_erase_registry_model_boundaries() {
    let worker = |url: &str, model: &str| -> Arc<dyn Worker> {
        Arc::new(
            BasicWorker::new(url.into(), WorkerType::Regular).with_labels(
                std::collections::HashMap::from([("model_id".into(), model.into())]),
            ),
        )
    };
    let workers = vec![
        worker("http://a:8000", "m"),
        worker("http://b:8000", "other"),
    ];
    workers[0].increment_load();
    let policy = policy();
    assert_eq!(
        policy
            .select_with_tokens(&workers, Some("m"), None, true)
            .unwrap()
            .0,
        0
    );
    assert!(policy
        .select_with_tokens(&workers, Some("unknown-model"), None, true)
        .is_none());
}

#[tokio::test]
async fn removed_worker_cannot_be_reattached_by_an_old_discovery_response() {
    let publisher = Publisher::new();
    let ready = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let (ready_handler, release_handler) = (ready.clone(), release.clone());
    let endpoint = publisher.endpoint.clone();
    let app =
        axum::Router::new()
            .route(
                "/get_server_info",
                get(move || {
                    let (ready, release) = (ready_handler.clone(), release_handler.clone());
                    async move {
                        ready.notify_one();
                        release.notified().await;
                        Json(json!({"instance_id":"delayed","model_id":"m","kv_block_size":2}))
                    }
                }),
            )
            .route(
                "/kv_event_sources",
                get(move || {
                    let endpoint = endpoint.clone();
                    async move {
                        Json(json!({"0":{"endpoint":endpoint,"topic":"kv","publisher":"zmq"}}))
                    }
                }),
            );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let policy = policy();
    let runtime = policy.runtime.clone();
    let pending_url = url.clone();
    let discovery = tokio::spawn(async move {
        runtime
            .discover(&reqwest::Client::new(), &pending_url, None)
            .await
    });
    tokio::time::timeout(Duration::from_secs(2), ready.notified())
        .await
        .unwrap();
    policy.runtime.remove_worker(&url);
    release.notify_one();
    assert!(discovery
        .await
        .unwrap()
        .unwrap_err()
        .contains("lease revoked"));
    let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(url, WorkerType::Regular))];
    assert_eq!(
        policy
            .select_with_tokens(&workers, Some("m"), Some(&[1, 2, 3]), true)
            .unwrap()
            .1
            .fallback_reason,
        Some("undiscovered_worker")
    );
    server.abort();
}

#[tokio::test]
async fn dynamic_add_and_remove_drive_subscription_lifecycle() {
    let p0 = Publisher::new();
    let e0 = Engine::new("first", &[(0, &p0)]).await;
    let (router, ctx) = router(&[&e0], 1).await;
    p0.subscribed().await;
    let p1 = Publisher::new();
    let e1 = Engine::new("added", &[(0, &p1)]).await;
    router.add_worker(&e1.url).await.unwrap();
    p1.subscribed().await;
    p0.send(0, 1, vec![clear()]);
    p1.send(0, 1, vec![clear(), stored()]);
    let policy = ctx.policy_registry.get_default_policy();
    let kv = policy.as_any().downcast_ref::<KvAwarePolicy>().unwrap();
    wait_for(kv, &ctx.worker_registry.get_all(), KvRoutingMode::Exact).await;
    let response = router
        .route_typed_request(
            None,
            &request(json!([1, 2, 3, 4, 5]), false),
            "/v1/completions",
            Some("m"),
        )
        .await;
    let body = axum::body::to_bytes(response.into_body(), 4096)
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&body).unwrap()["worker"],
        "added"
    );
    router.remove_worker(&e1.url);
    let response = router
        .route_typed_request(
            None,
            &request(json!([1, 2, 3, 4, 5]), false),
            "/v1/completions",
            Some("m"),
        )
        .await;
    let body = axum::body::to_bytes(response.into_body(), 4096)
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&body).unwrap()["worker"],
        "first"
    );
    assert_eq!(ctx.worker_registry.get_all().len(), 1);
}

#[tokio::test]
async fn duplicate_publisher_mapping_and_source_capacity_are_rejected_without_credit() {
    let publisher = Publisher::new();
    let duplicate = Engine::new("duplicate", &[(0, &publisher), (1, &publisher)]).await;
    let policy = policy();
    assert!(policy
        .runtime
        .discover(&reqwest::Client::new(), &duplicate.url, None)
        .await
        .is_err());
    let decision = policy
        .select_with_tokens(
            &[duplicate.worker()],
            Some("m"),
            Some(&[1, 2, 3, 4, 5]),
            true,
        )
        .unwrap()
        .1;
    assert_eq!(decision.mode, KvRoutingMode::Fallback);
    let p0 = Publisher::new();
    let p1 = Publisher::new();
    let two = Engine::new("two", &[(0, &p0), (1, &p1)]).await;
    let limited = KvAwarePolicy::with_config(KvAwareConfig {
        max_sources: 1,
        ..Default::default()
    });
    assert!(limited
        .runtime
        .discover(&reqwest::Client::new(), &two.url, None)
        .await
        .unwrap_err()
        .contains("capacity"));
}

#[path = "common/mock_vllm_rs.rs"]
mod mock_vllm_rs;
use vllm_router_rs::backend::pb;

#[derive(Clone)]
struct ControlStub {
    instance: String,
    endpoints: Vec<String>,
}
#[tonic::async_trait]
impl pb::control_server::Control for ControlStub {
    async fn get_server_info(
        &self,
        _request: tonic::Request<pb::GetServerInfoRequest>,
    ) -> Result<tonic::Response<pb::ServerInfo>, tonic::Status> {
        Ok(tonic::Response::new(pb::ServerInfo {
            instance_id: self.instance.clone(),
            kv_block_size: 2,
            ..Default::default()
        }))
    }
    async fn get_model_info(
        &self,
        _request: tonic::Request<pb::GetModelInfoRequest>,
    ) -> Result<tonic::Response<pb::ModelInfo>, tonic::Status> {
        Ok(tonic::Response::new(pb::ModelInfo {
            model_id: "m".into(),
            served_model_name: "m".into(),
            supports_token_ids_input: true,
            ..Default::default()
        }))
    }
    async fn abort(
        &self,
        _request: tonic::Request<pb::AbortRequest>,
    ) -> Result<tonic::Response<pb::AbortResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn load_lora(
        &self,
        _request: tonic::Request<pb::LoadLoraRequest>,
    ) -> Result<tonic::Response<pb::LoadLoraResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn unload_lora(
        &self,
        _request: tonic::Request<pb::UnloadLoraRequest>,
    ) -> Result<tonic::Response<pb::UnloadLoraResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn list_loras(
        &self,
        _request: tonic::Request<pb::ListLorasRequest>,
    ) -> Result<tonic::Response<pb::ListLorasResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn get_kv_event_sources(
        &self,
        _request: tonic::Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<tonic::Response<pb::GetKvEventSourcesResponse>, tonic::Status> {
        Ok(tonic::Response::new(pb::GetKvEventSourcesResponse {
            sources: self
                .endpoints
                .iter()
                .enumerate()
                .map(|(rank, endpoint)| pb::KvEventSource {
                    data_parallel_rank: Some(rank as u32),
                    endpoint: endpoint.clone(),
                    topic: "kv".into(),
                    transport: "zmq".into(),
                    ..Default::default()
                })
                .collect(),
        }))
    }
    async fn pause_generation(
        &self,
        _request: tonic::Request<pb::PauseGenerationRequest>,
    ) -> Result<tonic::Response<pb::PauseGenerationResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn resume_generation(
        &self,
        _request: tonic::Request<pb::ResumeGenerationRequest>,
    ) -> Result<tonic::Response<pb::ResumeGenerationResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn is_paused(
        &self,
        _request: tonic::Request<pb::IsPausedRequest>,
    ) -> Result<tonic::Response<pb::IsPausedResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn sleep(
        &self,
        _request: tonic::Request<pb::SleepRequest>,
    ) -> Result<tonic::Response<pb::SleepResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn wake_up(
        &self,
        _request: tonic::Request<pb::WakeUpRequest>,
    ) -> Result<tonic::Response<pb::WakeUpResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn is_sleeping(
        &self,
        _request: tonic::Request<pb::IsSleepingRequest>,
    ) -> Result<tonic::Response<pb::IsSleepingResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn init_weight_transfer_engine(
        &self,
        _request: tonic::Request<pb::InitWeightTransferEngineRequest>,
    ) -> Result<tonic::Response<pb::InitWeightTransferEngineResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn start_weight_update(
        &self,
        _request: tonic::Request<pb::StartWeightUpdateRequest>,
    ) -> Result<tonic::Response<pb::StartWeightUpdateResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn start_draft_weight_update(
        &self,
        _request: tonic::Request<pb::StartDraftWeightUpdateRequest>,
    ) -> Result<tonic::Response<pb::StartDraftWeightUpdateResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn update_weights(
        &self,
        _request: tonic::Request<pb::UpdateWeightsRequest>,
    ) -> Result<tonic::Response<pb::UpdateWeightsResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn finish_weight_update(
        &self,
        _request: tonic::Request<pb::FinishWeightUpdateRequest>,
    ) -> Result<tonic::Response<pb::FinishWeightUpdateResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn update_weight_version(
        &self,
        _request: tonic::Request<pb::UpdateWeightVersionRequest>,
    ) -> Result<tonic::Response<pb::UpdateWeightVersionResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn get_weight_version(
        &self,
        _request: tonic::Request<pb::GetWeightVersionRequest>,
    ) -> Result<tonic::Response<pb::GetWeightVersionResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
}

#[derive(Clone)]
struct InferenceStub {
    engine: mock_vllm_rs::MockVllmRs,
    ranks: Arc<Mutex<Vec<String>>>,
}
#[tonic::async_trait]
impl pb::inference_server::Inference for InferenceStub {
    type GenerateStreamStream =
        <mock_vllm_rs::MockVllmRs as pb::inference_server::Inference>::GenerateStreamStream;
    async fn generate(
        &self,
        request: tonic::Request<pb::GenerateRequest>,
    ) -> Result<tonic::Response<pb::GenerateResponse>, tonic::Status> {
        pb::inference_server::Inference::generate(&self.engine, request).await
    }
    async fn generate_stream(
        &self,
        request: tonic::Request<pb::GenerateRequest>,
    ) -> Result<tonic::Response<Self::GenerateStreamStream>, tonic::Status> {
        self.ranks.lock().unwrap().push(
            request
                .metadata()
                .get("x-data-parallel-rank")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("absent")
                .to_string(),
        );
        pb::inference_server::Inference::generate_stream(&self.engine, request).await
    }
}

#[tokio::test]
async fn grpc_control_discovery_and_prepared_tokens_select_the_holding_rank_on_the_wire() {
    let p0 = Publisher::new();
    let p1 = Publisher::new();
    let engine = mock_vllm_rs::MockVllmRs::default();
    let ranks = Arc::new(Mutex::new(Vec::new()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("grpc://{}", listener.local_addr().unwrap());
    let (reporter, health) = tonic_health::server::health_reporter();
    reporter
        .set_service_status("", tonic_health::ServingStatus::Serving)
        .await;
    let inference = pb::inference_server::InferenceServer::new(InferenceStub {
        engine: engine.clone(),
        ranks: ranks.clone(),
    });
    let control = pb::control_server::ControlServer::new(ControlStub {
        instance: "native-dp".into(),
        endpoints: vec![p0.endpoint.clone(), p1.endpoint.clone()],
    });
    let task = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(health)
            .add_service(inference)
            .add_service(control)
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    let config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![url.clone()],
        },
        policy: PolicyConfig::KvAware {
            config: KvAwareConfig {
                cache_weight: 0.9,
                ..Default::default()
            },
        },
        intra_node_data_parallel_size: 2,
        disable_retries: true,
        worker_startup_timeout_secs: 3,
        worker_startup_check_interval_secs: 1,
        ..Default::default()
    };
    let ctx =
        Arc::new(AppContext::new(config, reqwest::Client::new(), 64, None, Vec::new()).unwrap());
    let router = Router::new(vec![url], &ctx).await.unwrap();
    router.pin_test_token_ids(vec![1, 2, 3, 4, 5]);
    p0.subscribed().await;
    p1.subscribed().await;
    p0.send(0, 1, vec![clear()]);
    p1.send(1, 1, vec![clear(), stored()]);
    let policy = ctx.policy_registry.get_default_policy();
    let kv = policy.as_any().downcast_ref::<KvAwarePolicy>().unwrap();
    wait_for(kv, &ctx.worker_registry.get_all(), KvRoutingMode::Exact).await;
    let request: vllm_router_rs::protocols::spec::ChatCompletionRequest = serde_json::from_value(
        json!({"model":"m","messages":[{"role":"user","content":"hello"}],"max_tokens":1}),
    )
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/chat/completions", Some("m"))
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-router-kv-mode"], "exact");
    let _ = axum::body::to_bytes(response.into_body(), 16384)
        .await
        .unwrap();
    let captures = engine.captured.lock().unwrap();
    assert_eq!(captures.len(), 1);
    assert_eq!(captures[0].token_ids, vec![1, 2, 3, 4, 5]);
    assert!(!captures[0].had_text_prompt);
    assert_eq!(*ranks.lock().unwrap(), vec!["1"]);
    assert!(ctx.worker_registry.get_all().iter().all(|w| w.load() == 0));
    task.abort();
}
