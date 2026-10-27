#![cfg(all(feature = "cpu", feature = "ready"))]

use axum::{
    body::{to_bytes, Body},
    http::{HeaderMap, Request, StatusCode},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tower::ServiceExt;
use vllm_router_rs::{
    config::{PolicyConfig, RouterConfig},
    routers::{router_manager::RouterManager, RouterFactory, RouterTrait},
    server::{build_app_with_request_tracing, AppContext, AppState},
};

async fn test_app(igw: bool) -> (Router, Arc<AppContext>) {
    test_app_with_policy(igw, PolicyConfig::RoundRobin, false).await
}

async fn test_app_with_policy(
    igw: bool,
    policy: PolicyConfig,
    pd_routing: bool,
) -> (Router, Arc<AppContext>) {
    let config = RouterConfig {
        enable_igw: igw,
        policy,
        disable_retries: true,
        ..Default::default()
    };
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let context = Arc::new(AppContext::new(config.clone(), client, 64, None, vec![]).unwrap());
    let regular: Arc<dyn RouterTrait> = Arc::from(
        RouterFactory::create_regular_router(&[], &context)
            .await
            .unwrap(),
    );
    let manager = igw.then(|| {
        Arc::new(RouterManager::new(
            config.clone(),
            context.client.clone(),
            context.worker_registry.clone(),
            context.policy_registry.clone(),
        ))
    });
    let router: Arc<dyn RouterTrait> = if let Some(manager) = &manager {
        manager.register_router(
            vllm_router_rs::routers::router_manager::RouterId::new("http-regular".into()),
            regular,
        );
        if pd_routing {
            let pd = RouterFactory::create_vllm_pd_router(
                &[],
                &[],
                None,
                None,
                None,
                &config.policy,
                &context,
            )
            .await
            .unwrap();
            manager.register_router(
                vllm_router_rs::routers::router_manager::RouterId::new("http-pd".into()),
                Arc::from(pd),
            );
        }
        manager.clone()
    } else {
        regular
    };
    let state = Arc::new(AppState {
        router,
        context: context.clone(),
        concurrency_queue_tx: None,
        router_manager: manager,
    });
    (
        build_app_with_request_tracing(state, config.max_payload_size, vec![], vec![], true, false),
        context,
    )
}

async fn request(app: &Router, method: &str, path: &str, body: Value) -> (StatusCode, Value) {
    let request = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = tokio::time::timeout(Duration::from_secs(10), app.clone().oneshot(request))
        .await
        .unwrap()
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let body = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| json!({"body": String::from_utf8_lossy(&bytes)}));
    (status, body)
}

struct TestWorker {
    url: String,
    ranks: Arc<Mutex<Vec<Option<String>>>>,
    model_available: Arc<AtomicBool>,
    model_queries: Arc<AtomicUsize>,
    server: tokio::task::JoinHandle<()>,
}

impl TestWorker {
    async fn new() -> Self {
        let ranks = Arc::new(Mutex::new(Vec::new()));
        let model_available = Arc::new(AtomicBool::new(true));
        let model_queries = Arc::new(AtomicUsize::new(0));
        let available = model_available.clone();
        let queries = model_queries.clone();
        let observed = ranks.clone();
        let respond = move |headers: HeaderMap| {
            let observed = observed.clone();
            async move {
                observed.lock().unwrap().push(
                    headers
                        .get("x-data-parallel-rank")
                        .map(|v| v.to_str().unwrap().to_owned()),
                );
                Json(json!({
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "kv_transfer_params": {"remote_engine_id": "mock", "do_remote_prefill": true}
                }))
            }
        };
        let app = Router::new()
            .route("/health", get(|| async { StatusCode::OK }))
            .route(
                "/v1/models",
                get(move || {
                    let available = available.clone();
                    let queries = queries.clone();
                    async move {
                        queries.fetch_add(1, Ordering::SeqCst);
                        let status = if available.load(Ordering::SeqCst) {
                            StatusCode::OK
                        } else {
                            StatusCode::SERVICE_UNAVAILABLE
                        };
                        (status, Json(json!({"data": [{"id": "test"}]})))
                    }
                }),
            )
            .route("/v1/chat/completions", post(respond.clone()))
            .route("/v1/responses/{id}", get(respond.clone()))
            .route("/v1/responses/{id}/cancel", post(respond));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            url,
            ranks,
            model_available,
            model_queries,
            server,
        }
    }
}

impl Drop for TestWorker {
    fn drop(&mut self) {
        self.server.abort();
    }
}

fn worker_path(url: &str) -> String {
    let encoded: String = url::form_urlencoded::byte_serialize(url.as_bytes()).collect();
    format!("/workers/{encoded}")
}

fn chat() -> Value {
    json!({"model": "test", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1})
}

#[tokio::test]
async fn registered_rank_reaches_the_worker() {
    let worker = TestWorker::new().await;
    let (app, context) = test_app(true).await;
    let (status, registration) = request(
        &app,
        "POST",
        "/workers",
        json!({"url": worker.url, "model_id": "test", "dp_rank": 1, "dp_size": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{registration}");
    let (status, response) = request(&app, "POST", "/v1/chat/completions", chat()).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(*worker.ranks.lock().unwrap(), vec![Some("1".into())]);
    let registered = context.worker_registry.get_all();
    assert_eq!(registered.len(), 1);
    assert_eq!(registered[0].dp_rank(), Some(1));
    registered[0].check_health_async().await.unwrap();
}

#[tokio::test]
async fn ranks_share_an_endpoint_and_can_be_removed_independently() {
    let worker = TestWorker::new().await;
    let (app, context) = test_app(true).await;
    for rank in 0..2 {
        let (status, body) = request(
            &app,
            "POST",
            "/workers",
            json!({"url": worker.url, "model_id": "test", "dp_rank": rank, "dp_size": 2}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["worker"]["url"], format!("{}@{rank}", worker.url));
        assert_eq!(body["worker"]["base_url"], worker.url);
        assert_eq!(body["worker"]["dp_rank"], rank);
        assert_eq!(body["worker"]["dp_size"], 2);
    }
    let (status, list) = request(&app, "GET", "/workers", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(list["total"], 2);
    for rank in 0..2 {
        let (status, info) = request(
            &app,
            "GET",
            &worker_path(&format!("{}@{rank}", worker.url)),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{info}");
        assert_eq!(info["dp_rank"], rank);
        let (status, body) = request(&app, "POST", "/v1/chat/completions", chat()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
    let mut ranks = worker.ranks.lock().unwrap().clone();
    ranks.sort();
    assert_eq!(ranks, vec![Some("0".into()), Some("1".into())]);
    let (status, body) = request(
        &app,
        "DELETE",
        &worker_path(&format!("{}@1", worker.url)),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(context.worker_registry.get_all().len(), 1);
    assert_eq!(context.worker_registry.get_by_model_fast("test").len(), 1);
    let (status, body) = request(&app, "POST", "/v1/chat/completions", chat()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(worker.ranks.lock().unwrap().last(), Some(&Some("0".into())));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn repeated_registration_is_idempotent() {
    let (app, context) = test_app(true).await;
    let config =
        json!({"url": "http://worker:8000", "model_id": "test", "dp_rank": 1, "dp_size": 2});
    let mut registrations = Vec::new();
    for _ in 0..16 {
        let app = app.clone();
        let config = config.clone();
        registrations.push(tokio::spawn(async move {
            let (status, body) = request(&app, "POST", "/workers", config).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            body["worker"]["id"].clone()
        }));
    }
    let mut ids = Vec::new();
    for registration in registrations {
        ids.push(registration.await.unwrap());
    }
    assert!(ids.iter().all(|id| id == &ids[0]));
    assert_eq!(context.worker_registry.get_all().len(), 1);
    assert_eq!(context.worker_registry.get_by_model("test").len(), 1);
    assert_eq!(context.worker_registry.get_by_model_fast("test").len(), 1);
    assert_eq!(context.policy_registry.get_worker_counts()["test"], 1);
    let (status, body) = request(
        &app,
        "DELETE",
        &worker_path("http://worker:8000@1"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(context.worker_registry.get_all().is_empty());
    assert!(context.worker_registry.get_by_model_fast("test").is_empty());
    assert!(context.policy_registry.get_worker_counts().is_empty());
}

#[tokio::test]
async fn conflicting_registration_preserves_existing_workers() {
    let (app, context) = test_app(true).await;
    let config =
        json!({"url": "http://worker:8000", "model_id": "test", "dp_rank": 1, "dp_size": 2});
    assert_eq!(
        request(&app, "POST", "/workers", config.clone()).await.0,
        StatusCode::OK
    );
    for change in [
        json!({"model_id": "other"}),
        json!({"dp_size": 3}),
        json!({"worker_type": "prefill"}),
        json!({"priority": 20}),
        json!({"labels": {"policy": "random"}}),
        json!({"dp_rank": null, "dp_size": null}),
        json!({"url": "http://worker:8000@1", "dp_rank": null, "dp_size": null}),
    ] {
        let mut conflicting = config.clone();
        conflicting
            .as_object_mut()
            .unwrap()
            .extend(change.as_object().unwrap().clone());
        let (status, body) = request(&app, "POST", "/workers", conflicting).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["code"], "WORKER_CONFLICT");
    }
    assert_eq!(context.worker_registry.get_all().len(), 1);
    assert_eq!(context.worker_registry.get_by_model_fast("test").len(), 1);
    assert_eq!(context.policy_registry.get_worker_counts()["test"], 1);
}

#[tokio::test]
async fn invalid_dp_configuration_never_registers_a_worker() {
    let (app, context) = test_app(true).await;
    for change in [
        json!({"dp_size": null}),
        json!({"dp_rank": null}),
        json!({"dp_rank": 0, "dp_size": 0}),
        json!({"dp_rank": 2, "dp_size": 2}),
        json!({"url": "grpc://worker:50051"}),
        json!({"url": "http://worker:8000@1"}),
        json!({"url": "http://user:password@worker:8000"}),
        json!({"url": "http://@worker:8000"}),
        json!({"url": "http://worker:8000?rank=1"}),
        json!({"url": "http://worker:8000#1"}),
        json!({"url": "not-a-url"}),
        json!({"worker_type": "unknown"}),
    ] {
        let mut config =
            json!({"url": "http://worker:8000", "model_id": "test", "dp_rank": 1, "dp_size": 2});
        config
            .as_object_mut()
            .unwrap()
            .extend(change.as_object().unwrap().clone());
        let (status, body) = request(&app, "POST", "/workers", config).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["code"], "INVALID_DP_CONFIG");
    }
    assert!(context.worker_registry.get_all().is_empty());
    assert!(context.policy_registry.get_worker_counts().is_empty());
}

#[tokio::test]
async fn explicit_ranks_require_igw() {
    let (app, context) = test_app(false).await;
    let (status, body) = request(
        &app,
        "POST",
        "/workers",
        json!({"url": "http://worker:8000", "model_id": "test", "dp_rank": 1, "dp_size": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["code"], "DP_REGISTRATION_UNSUPPORTED");
    assert!(context.worker_registry.get_all().is_empty());
}

#[tokio::test]
async fn legacy_registration_keeps_a_basic_worker() {
    let worker = TestWorker::new().await;
    let (app, context) = test_app(true).await;
    let (status, registration) = request(
        &app,
        "POST",
        "/workers",
        json!({"url": worker.url, "model_id": "test"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{registration}");
    assert!(registration["worker"].get("dp_rank").is_none());
    assert!(registration["worker"].get("dp_size").is_none());
    assert!(registration["worker"].get("base_url").is_none());
    assert!(!context.worker_registry.get_all()[0].is_dp_aware());
    let (status, body) = request(&app, "POST", "/v1/chat/completions", chat()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(*worker.ranks.lock().unwrap(), vec![None]);
    let (status, body) = request(
        &app,
        "POST",
        "/workers",
        json!({"url": worker.url, "model_id": "test", "dp_rank": 1, "dp_size": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["code"], "WORKER_CONFLICT");
}

#[tokio::test]
async fn dp_registration_preserves_metadata_and_discovers_model_from_base_url() {
    let worker = TestWorker::new().await;
    let (app, context) = test_app(true).await;
    let (status, body) = request(
        &app,
        "POST",
        "/workers",
        json!({"url": format!("{}/", worker.url), "worker_type": "prefill", "bootstrap_port": 9000,
            "dp_rank": 1, "dp_size": 2, "priority": 75, "cost": 1.5, "labels": {"zone": "a"}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["worker"]["model_id"], "test");
    assert_eq!(body["worker"]["base_url"], worker.url);
    assert_eq!(body["worker"]["priority"], 75);
    assert_eq!(body["worker"]["cost"], 1.5);
    assert_eq!(body["worker"]["metadata"]["zone"], "a");
    let registered = context.worker_registry.get_all();
    assert_eq!(
        registered[0].worker_type(),
        vllm_router_rs::core::WorkerType::Prefill {
            bootstrap_port: Some(9000)
        }
    );
    registered[0].check_health_async().await.unwrap();
}

#[tokio::test]
async fn prefill_and_decode_use_independent_ranks_and_sizes() {
    let prefill = TestWorker::new().await;
    let decode = TestWorker::new().await;
    let (app, _) = test_app_with_policy(true, PolicyConfig::RoundRobin, true).await;
    for config in [
        json!({"url": prefill.url, "worker_type": "prefill", "model_id": "test", "dp_rank": 3, "dp_size": 4}),
        json!({"url": decode.url, "worker_type": "decode", "model_id": "test", "dp_rank": 1, "dp_size": 2}),
    ] {
        let (status, body) = request(&app, "POST", "/workers", config).await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
    let (status, body) = request(&app, "POST", "/v1/chat/completions", chat()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(*prefill.ranks.lock().unwrap(), vec![Some("3".into())]);
    assert_eq!(*decode.ranks.lock().unwrap(), vec![Some("1".into())]);
}

#[tokio::test]
async fn ipv6_and_single_rank_endpoints_keep_their_base_url() {
    let (app, context) = test_app(true).await;
    let (status, body) = request(
        &app,
        "POST",
        "/workers",
        json!({"url": "https://[::1]:8000", "model_id": "test", "dp_rank": 0, "dp_size": 1}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let registered = context.worker_registry.get_all();
    assert_eq!(registered[0].url(), "https://[::1]:8000@0");
    assert_eq!(
        registered[0].endpoint_url("/health"),
        "https://[::1]:8000/health"
    );
}

#[tokio::test]
async fn response_lookup_and_cancellation_use_the_registered_rank() {
    let worker = TestWorker::new().await;
    let (app, _) = test_app(true).await;
    let (status, body) = request(
        &app,
        "POST",
        "/workers",
        json!({"url": worker.url, "model_id": "test", "dp_rank": 1, "dp_size": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    for (method, path) in [
        ("GET", "/v1/responses/test-response"),
        ("POST", "/v1/responses/test-response/cancel"),
    ] {
        let (status, body) = request(&app, method, path, Value::Null).await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
    assert_eq!(
        *worker.ranks.lock().unwrap(),
        vec![Some("1".into()), Some("1".into())]
    );
}

#[tokio::test]
async fn client_rank_header_does_not_override_the_selected_rank() {
    let worker = TestWorker::new().await;
    let (app, _) = test_app(true).await;
    let (status, body) = request(
        &app,
        "POST",
        "/workers",
        json!({"url": worker.url, "model_id": "test", "dp_rank": 1, "dp_size": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("x-data-parallel-rank", "99")
        .body(Body::from(chat().to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    assert_eq!(*worker.ranks.lock().unwrap(), vec![Some("1".into())]);
}

#[tokio::test]
async fn consistent_hash_keeps_session_affinity_across_pd_ranks() {
    let prefill = TestWorker::new().await;
    let decode = TestWorker::new().await;
    let (app, _) = test_app_with_policy(
        true,
        PolicyConfig::ConsistentHash { virtual_nodes: 160 },
        true,
    )
    .await;
    for (worker, role, size) in [(&prefill, "prefill", 4), (&decode, "decode", 2)] {
        for rank in 0..size {
            let (status, body) = request(
                &app,
                "POST",
                "/workers",
                json!({"url": worker.url, "worker_type": role, "model_id": "test", "dp_rank": rank, "dp_size": size}),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{body}");
        }
    }
    for session in 0..8 {
        for turn in 0..3 {
            let mut body = chat();
            body["messages"][0]["content"] = json!(format!("session {session}, turn {turn}"));
            let request = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("x-session-id", format!("session-{session}"))
                .body(Body::from(body.to_string()))
                .unwrap();
            let response = app.clone().oneshot(request).await.unwrap();
            let status = response.status();
            let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
            assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
        }
    }
    for worker in [&prefill, &decode] {
        let ranks = worker.ranks.lock().unwrap();
        assert_eq!(ranks.len(), 24);
        for turns in ranks.chunks_exact(3) {
            assert!(turns[0].is_some());
            assert!(turns.iter().all(|rank| rank == &turns[0]));
        }
    }
}

#[tokio::test]
async fn regression_legacy_dp_suffix_forwarding_and_health() {
    let worker = TestWorker::new().await;
    let config = RouterConfig {
        enable_igw: true,
        intra_node_data_parallel_size: 2,
        policy: PolicyConfig::RoundRobin,
        disable_retries: true,
        ..Default::default()
    };
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let context =
        Arc::new(AppContext::new(config.clone(), client.clone(), 64, None, vec![]).unwrap());
    let router = vllm_router_rs::routers::http::router::Router::new(vec![], &context)
        .await
        .unwrap();
    let manager = RouterManager::new(
        config,
        client,
        context.worker_registry.clone(),
        context.policy_registry.clone(),
    );
    let identity = format!("{}@1", worker.url);
    manager
        .add_worker(serde_json::from_value(json!({"url": identity, "model_id": "test"})).unwrap())
        .await
        .unwrap();
    assert!(!context.worker_registry.get_all()[0].is_dp_aware());
    let response = router
        .route_chat(None, &serde_json::from_value(chat()).unwrap(), Some("test"))
        .await;
    let status = response.status();
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert_eq!(*worker.ranks.lock().unwrap(), vec![Some("1".into())]);
    assert_eq!(
        router.send_health_check(&identity).await.status(),
        StatusCode::OK
    );
}

async fn registration_survives_discovery_change(initially_available: bool) {
    let worker = TestWorker::new().await;
    worker
        .model_available
        .store(initially_available, Ordering::SeqCst);
    let (app, context) = test_app(true).await;
    let config = json!({"url": worker.url, "dp_rank": 1, "dp_size": 2});
    let (status, first) = request(&app, "POST", "/workers", config.clone()).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let model = if initially_available {
        "test"
    } else {
        "unknown"
    };
    assert_eq!(first["worker"]["model_id"], model);
    worker
        .model_available
        .store(!initially_available, Ordering::SeqCst);
    let (status, second) = request(&app, "POST", "/workers", config.clone()).await;
    assert_eq!(status, StatusCode::OK, "{second}");
    assert_eq!(second["worker"]["id"], first["worker"]["id"]);
    assert_eq!(second["worker"]["model_id"], model);
    assert_eq!(worker.model_queries.load(Ordering::SeqCst), 1);
    assert_eq!(context.worker_registry.get_all().len(), 1);
    assert_eq!(context.worker_registry.get_by_model_fast(model).len(), 1);
    assert_eq!(context.policy_registry.get_worker_counts()[model], 1);
    let mut conflict = config;
    conflict["model_id"] = json!("another-model");
    let (status, body) = request(&app, "POST", "/workers", conflict).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["code"], "WORKER_CONFLICT");
}

#[tokio::test]
async fn regression_rank_registration_survives_discovery_outage() {
    registration_survives_discovery_change(true).await;
}

#[tokio::test]
async fn regression_rank_registration_survives_discovery_recovery() {
    registration_survives_discovery_change(false).await;
}

#[tokio::test]
async fn regression_prefill_ranks_allow_distinct_bootstrap_ports() {
    let (app, context) = test_app(true).await;
    for (rank, port) in [(0, 9000), (1, 9001)] {
        let config = json!({"url": "http://worker:8000", "model_id": "test", "worker_type": "prefill",
            "dp_rank": rank, "dp_size": 2, "bootstrap_port": port});
        let (status, body) = request(&app, "POST", "/workers", config.clone()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            request(&app, "POST", "/workers", config).await.0,
            StatusCode::OK
        );
        assert_eq!(
            context
                .worker_registry
                .get_by_url(&format!("http://worker:8000@{rank}"))
                .unwrap()
                .worker_type(),
            vllm_router_rs::core::WorkerType::Prefill {
                bootstrap_port: Some(port)
            }
        );
    }
    let (status, body) = request(
        &app,
        "POST",
        "/workers",
        json!({"url": "http://worker:8000", "model_id": "test", "worker_type": "prefill",
            "dp_rank": 1, "dp_size": 2, "bootstrap_port": 9002}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["code"], "WORKER_CONFLICT");
    assert_eq!(context.worker_registry.get_all().len(), 2);
    assert_eq!(context.policy_registry.get_worker_counts()["test"], 2);
}

#[tokio::test]
async fn concurrent_discovery_reuses_the_model_registered_first() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let queries = Arc::new(AtomicUsize::new(0));
    let worker_app = Router::new().route(
        "/v1/models",
        get({
            let started = started.clone();
            let release = release.clone();
            move || {
                let started = started.clone();
                let release = release.clone();
                let queries = queries.clone();
                async move {
                    if queries.fetch_add(1, Ordering::SeqCst) == 0 {
                        started.notify_one();
                        release.notified().await;
                        (StatusCode::SERVICE_UNAVAILABLE, Json(json!({})))
                    } else {
                        (StatusCode::OK, Json(json!({"data": [{"id": "test"}]})))
                    }
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, worker_app).await.unwrap() });
    let (app, context) = test_app(true).await;
    let config = json!({"url": url, "dp_rank": 1, "dp_size": 2});
    let first = tokio::spawn({
        let app = app.clone();
        let config = config.clone();
        async move { request(&app, "POST", "/workers", config).await }
    });
    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .unwrap();
    let (status, second) = request(&app, "POST", "/workers", config).await;
    release.notify_one();
    let (first_status, first_body) = first.await.unwrap();
    server.abort();
    assert_eq!(status, StatusCode::OK, "{second}");
    assert_eq!(first_status, StatusCode::OK, "{first_body}");
    assert_eq!(first_body["worker"]["id"], second["worker"]["id"]);
    assert_eq!(first_body["worker"]["model_id"], "test");
    assert_eq!(context.worker_registry.get_all().len(), 1);
    assert_eq!(context.worker_registry.get_by_model_fast("test").len(), 1);
    assert_eq!(context.policy_registry.get_worker_counts()["test"], 1);
}
