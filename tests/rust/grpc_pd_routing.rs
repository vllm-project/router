#![cfg(all(feature = "cpu", feature = "ready"))]

//! P/D disaggregation over `grpc://` prefill and decode pools.
//!
//! Two in-process `Inference` mocks stand in for vLLM Rust workers. The
//! router must send P a unary one-token `Generate` with the connector's
//! `kv_transfer_params`, then send D a `GenerateStream` that carries P's
//! `FinishInfo.kv_transfer_params`. No GPU or KV connector is involved.
//!
//!   cargo test --features cpu,ready --test grpc_pd_routing

mod common;
#[path = "common/grpc_frontend_fixture.rs"]
mod fixture;

use std::collections::BTreeSet;
use std::sync::atomic::Ordering;
use std::sync::OnceLock;

use common::create_test_context;
use common::mock_vllm_rs::{MockVllmRs, MockVllmRsServer};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serde_json::{json, Value};
use vllm_router_rs::backend::EngineFrontend;
use vllm_router_rs::config::{KvConnector, PolicyConfig, RouterConfig, RoutingMode};
use vllm_router_rs::policies::PolicyFactory;
use vllm_router_rs::protocols::spec::{ChatCompletionRequest, CompletionRequest};
use vllm_router_rs::routers::http::vllm_pd_router::VllmPDRouter;
use vllm_router_rs::routers::RouterTrait;

fn prefill_kv() -> Value {
    json!({
        "do_remote_prefill": true,
        "do_remote_decode": false,
        "remote_engine_id": "P",
        "remote_block_ids": [[7, 8]],
        "remote_host": "prefill",
        "remote_port": 5600,
        "tp_size": 1,
    })
}

/// Process-wide recorder; series are labelled by each test's own mock URL.
fn metrics() -> &'static PrometheusHandle {
    static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
    HANDLE.get_or_init(|| PrometheusBuilder::new().install_recorder().unwrap())
}

/// A P/D stage gauge for `worker`, whatever its rank label.
fn worker_gauge(name: &str, worker: &str) -> f64 {
    let label = format!("worker=\"{worker}\"");
    metrics()
        .render()
        .lines()
        .filter(|line| line.starts_with(&format!("{name}{{")) && line.contains(&label))
        .find_map(|line| line.rsplit(' ').next().and_then(|n| n.parse().ok()))
        .unwrap_or(0.0)
}

async fn wait_for_gauge(name: &str, worker: &str, expected: f64) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while worker_gauge(name, worker) != expected {
        assert!(
            std::time::Instant::now() < deadline,
            "{name} for {worker} stayed at {}, expected {expected}",
            worker_gauge(name, worker)
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

fn kv_aware() -> PolicyConfig {
    PolicyConfig::KvAware {
        router_temperature: 0.0,
        kv_penalty_host: 0.8,
        kv_penalty_external: 0.6,
        kv_penalty_disk: 0.3,
        static_kv_event_sources: Default::default(),
    }
}

fn worker_counter(name: &str, worker: &str) -> f64 {
    let prefix = format!("{name}{{worker=\"{worker}\"}} ");
    metrics()
        .render()
        .lines()
        .find_map(|line| line.strip_prefix(&prefix).and_then(|n| n.parse().ok()))
        .unwrap_or(0.0)
}

async fn pd_router(prefill: &MockVllmRsServer, decode: &MockVllmRsServer) -> VllmPDRouter {
    pd_router_with_dp(prefill, decode, 1).await
}

async fn pd_router_with_dp(
    prefill: &MockVllmRsServer,
    decode: &MockVllmRsServer,
    dp_size: usize,
) -> VllmPDRouter {
    let router = pd_router_with_pools(&[prefill], decode, dp_size, PolicyConfig::RoundRobin).await;
    router.pin_test_token_ids(vec![1, 2, 3]);
    router
}

async fn pd_router_with_pools(
    prefills: &[&MockVllmRsServer],
    decode: &MockVllmRsServer,
    dp_size: usize,
    prefill_policy: PolicyConfig,
) -> VllmPDRouter {
    pd_router_with_progress(prefills, decode, dp_size, prefill_policy, None).await
}

async fn pd_router_with_progress(
    prefills: &[&MockVllmRsServer],
    decode: &MockVllmRsServer,
    dp_size: usize,
    prefill_policy: PolicyConfig,
    prefill_progress_stale_after_secs: Option<u64>,
) -> VllmPDRouter {
    metrics();
    let prefill_urls: Vec<(String, Option<u16>)> = prefills
        .iter()
        .map(|prefill| (prefill.grpc_url.clone(), None))
        .collect();
    let config = RouterConfig {
        mode: RoutingMode::VllmPrefillDecode {
            prefill_urls: prefill_urls.clone(),
            decode_urls: vec![decode.grpc_url.clone()],
            prefill_policy: Some(prefill_policy.clone()),
            decode_policy: Some(PolicyConfig::RoundRobin),
            discovery_address: None,
        },
        policy: PolicyConfig::RoundRobin,
        kv_connector: KvConnector::Nixl,
        worker_startup_timeout_secs: 5,
        intra_node_data_parallel_size: dp_size,
        prefill_progress_stale_after_secs,
        ..RouterConfig::default()
    };
    vllm_router_rs::config::ConfigValidator::validate(&config).unwrap();
    let ctx = create_test_context(config);
    // Separate P and D policy instances, as `RouterFactory` installs them.
    let policies = &ctx.policy_registry;
    policies.set_prefill_policy(PolicyFactory::create_from_config(&prefill_policy, None));
    policies.set_decode_policy(PolicyFactory::create_from_config(
        &PolicyConfig::RoundRobin,
        None,
    ));
    let router = VllmPDRouter::new(prefill_urls, vec![decode.grpc_url.clone()], None, &ctx)
        .await
        .expect("grpc P/D workers should pass grpc.health.v1");
    router
}

fn chat(stream: bool) -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 8,
        "stream": stream,
    }))
    .unwrap()
}

fn completion(prompt: Value, stream: bool) -> CompletionRequest {
    serde_json::from_value(json!({
        "model": "test-model",
        "prompt": prompt,
        "max_tokens": 8,
        "stream": stream,
    }))
    .unwrap()
}

/// AgentX-style replay sends token ids, so the router must accept them
/// without a tokenizer round trip on either leg.
#[tokio::test]
async fn grpc_pd_completion_passes_client_token_ids_to_both_legs() {
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(prefill_kv()),
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router(&prefill, &decode).await;
    router.pin_test_token_ids(vec![9]); // must not be used: ids come from the client

    let response = router
        .route_completion(None, &completion(json!([11, 12, 13, 14]), false), None)
        .await;
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["object"], "text_completion", "{json}");
    assert!(json["choices"][0].get("message").is_none(), "{json}");
    assert!(json["id"].as_str().unwrap_or_default().starts_with("cmpl-"));

    assert_eq!(prefill.captured()[0].token_ids, vec![11, 12, 13, 14]);
    assert_eq!(decode.captured()[0].token_ids, vec![11, 12, 13, 14]);
}

/// A prompt the gRPC wire cannot carry is rejected before any worker is used.
#[tokio::test]
async fn grpc_pd_completion_rejects_text_prompts() {
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(prefill_kv()),
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router(&prefill, &decode).await;

    let response = router
        .route_completion(None, &completion(json!("hello"), false), None)
        .await;
    assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
    assert!(prefill.captured().is_empty());
    assert!(decode.captured().is_empty());
}

#[tokio::test]
async fn grpc_pd_hands_prefill_kv_params_to_decode() {
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(prefill_kv()),
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router(&prefill, &decode).await;

    for stream in [false, true] {
        let response = router.route_chat(None, &chat(stream), None).await;
        assert!(response.status().is_success(), "stream={stream}");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8_lossy(&bytes);
        assert!(
            body.contains("hello from worker"),
            "stream={stream}: {body}"
        );
    }

    let p = prefill.captured();
    let d = decode.captured();
    assert_eq!((p.len(), d.len()), (2, 2));
    for (p, d) in p.iter().zip(&d) {
        assert!(p.unary, "prefill must be unary Generate");
        assert_eq!((p.max_new_tokens, p.min_new_tokens), (1, 1));
        let sent = p.kv_transfer_params.as_ref().expect("prefill kv params");
        assert_eq!(sent["do_remote_decode"], true);
        assert_eq!(sent["do_remote_prefill"], false);

        assert!(!d.unary, "decode must stream");
        assert_eq!(d.max_new_tokens, 8);
        assert_eq!(d.kv_transfer_params.as_ref(), Some(&prefill_kv()));
        assert_eq!(d.token_ids, p.token_ids);
        assert_eq!(d.request_id, p.request_id);
        assert!(!d.had_text_prompt && !p.had_text_prompt);
    }
    assert!(prefill.aborts().is_empty() && decode.aborts().is_empty());
}

#[tokio::test]
async fn grpc_pd_aborted_prefill_skips_decode() {
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(prefill_kv()),
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router(&prefill, &decode).await;
    prefill
        .state
        .aborted_finishes_remaining
        .store(1, Ordering::Relaxed);

    let response = router.route_chat(None, &chat(false), None).await;
    assert_eq!(response.status(), axum::http::StatusCode::BAD_GATEWAY);
    assert!(decode.captured().is_empty());
    assert_eq!(
        worker_counter("vllm_router_pd_prefill_errors_total", &prefill.grpc_url),
        1.0
    );
}

#[tokio::test]
async fn grpc_pd_non_streaming_decode_releases_load_before_body_is_read() {
    let (_prefill, decode, router) = push_pair().await;
    let d = router
        .worker_registry()
        .get_by_url(&decode.grpc_url)
        .unwrap();

    // Sequential first, then concurrent once P's push identity is cached.
    for _ in 0..2 {
        let response = router.route_chat(None, &chat(false), None).await;
        assert!(response.status().is_success());
        assert_eq!(d.load(), 0, "a collected reply holds no decode load");
        drop(response);
    }
    assert_eq!(decode.captured().len(), 2);
}

#[tokio::test]
async fn grpc_pd_prefill_failure_skips_decode() {
    let prefill = MockVllmRsServer::spawn().await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router(&prefill, &decode).await;
    prefill.state.failures_remaining.store(1, Ordering::Relaxed);

    let response = router.route_chat(None, &chat(false), None).await;
    assert_eq!(
        response.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(prefill.captured().len(), 1);
    assert!(decode.captured().is_empty());
    let errors = "vllm_router_pd_prefill_errors_total";
    assert_eq!(worker_counter(errors, &prefill.grpc_url), 1.0);
}

#[tokio::test]
async fn grpc_pd_decode_failure_after_prefill_records_decode_error() {
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(prefill_kv()),
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router(&prefill, &decode).await;
    decode.state.failures_remaining.store(1, Ordering::Relaxed);

    let response = router.route_chat(None, &chat(false), None).await;
    assert_eq!(
        response.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );
    let (p, d) = (prefill.captured(), decode.captured());
    assert_eq!((p.len(), d.len()), (1, 1));
    assert_eq!(d[0].kv_transfer_params.as_ref(), Some(&prefill_kv()));
    assert_eq!(
        worker_counter("vllm_router_pd_decode_errors_total", &decode.grpc_url),
        1.0
    );
    assert_eq!(
        worker_counter("vllm_router_pd_prefill_errors_total", &prefill.grpc_url),
        0.0
    );
}

#[tokio::test]
async fn grpc_pd_sends_each_leg_its_dp_rank() {
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(prefill_kv()),
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router_with_dp(&prefill, &decode, 2).await;

    for _ in 0..4 {
        let response = router.route_chat(None, &chat(false), None).await;
        assert!(response.status().is_success());
    }

    let ranks = |captured: &[common::mock_vllm_rs::CapturedGrpcGenerate]| {
        captured
            .iter()
            .map(|c| c.dp_rank.clone().expect("x-data-parallel-rank"))
            .collect::<BTreeSet<_>>()
    };
    let (p, d) = (prefill.captured(), decode.captured());
    assert_eq!(p.len(), 4);
    assert_eq!(d.len(), 4);
    let expected = BTreeSet::from(["0".to_string(), "1".to_string()]);
    assert_eq!(ranks(&p), expected, "prefill ranks");
    assert_eq!(ranks(&d), expected, "decode ranks");
    for (p, d) in p.iter().zip(&d) {
        assert_eq!(p.request_id, d.request_id);
        assert_eq!(d.kv_transfer_params.as_ref(), Some(&prefill_kv()));
    }
}

/// P's `FinishInfo` under `NixlPushConnector`: its push identity.
fn push_kv() -> Value {
    json!({
        "do_remote_prefill": true,
        "do_remote_decode": false,
        "remote_engine_id": "P",
        "remote_block_ids": [[7, 8]],
        "remote_host": "prefill",
        "remote_port": 5600,
        "tp_size": 1,
        "transfer_mode": "push",
    })
}

fn push_decode_kv(request_id: &str) -> Value {
    json!({
        "do_remote_prefill": true,
        "do_remote_decode": false,
        "remote_engine_id": "P",
        "remote_host": "prefill",
        "remote_port": 5600,
        "tp_size": 1,
        "remote_request_id": request_id,
    })
}

const PREFILL_DELAY: std::time::Duration = std::time::Duration::from_millis(300);

async fn push_pair() -> (MockVllmRsServer, MockVllmRsServer, VllmPDRouter) {
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(push_kv()),
        reply_delay: PREFILL_DELAY,
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router(&prefill, &decode).await;
    (prefill, decode, router)
}

/// P/D admission scores P by the router's in-flight occupancy, which a
/// `grpc://` worker's missing HTTP `/get_load` must not hide.
#[tokio::test]
async fn grpc_pd_power_of_two_avoids_occupied_prefill() {
    let busy = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(prefill_kv()),
        reply_delay: std::time::Duration::from_secs(5),
        ..Default::default()
    })
    .await;
    let idle = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(prefill_kv()),
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn().await;
    let policy = PolicyConfig::PowerOfTwo {
        load_check_interval_secs: 1,
    };
    let router =
        std::sync::Arc::new(pd_router_with_pools(&[&busy, &idle], &decode, 1, policy).await);
    router.pin_test_token_ids(vec![1, 2, 3]);
    // Let the load monitor run its first pass.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    // Park one slow request on `busy`; those that land on `idle` finish at once.
    let mut parked = Vec::new();
    while busy.captured().is_empty() {
        let router = router.clone();
        parked.push(tokio::spawn(async move {
            router.route_chat(None, &chat(false), None).await.status()
        }));
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    let busy_before = busy.captured().len();
    let idle_before = idle.captured().len();

    for _ in 0..8 {
        let response = router.route_chat(None, &chat(false), None).await;
        assert!(response.status().is_success());
    }
    assert_eq!(
        busy.captured().len(),
        busy_before,
        "the occupied prefill was picked"
    );
    assert_eq!(idle.captured().len(), idle_before + 8);
    for task in parked {
        task.abort();
    }
}

#[tokio::test]
async fn grpc_pd_nixl_push_dispatches_both_legs_once_identity_is_cached() {
    let (prefill, decode, router) = push_pair().await;

    for stream in [false, false, true] {
        let response = router.route_chat(None, &chat(stream), None).await;
        assert!(response.status().is_success(), "stream={stream}");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&bytes).contains("hello from worker"));
    }

    let (p, d) = (prefill.captured(), decode.captured());
    assert_eq!((p.len(), d.len()), (3, 3));
    for (i, (p, d)) in p.iter().zip(&d).enumerate() {
        assert!(p.unary);
        assert_eq!(p.request_id, d.request_id);
        assert_eq!(
            d.kv_transfer_params.as_ref(),
            Some(&push_decode_kv(&d.request_id))
        );
        let gap = d.received_at.unwrap() - p.received_at.unwrap();
        if i == 0 {
            assert!(gap >= PREFILL_DELAY, "first request learns the identity");
        } else {
            assert!(gap < PREFILL_DELAY / 2, "request {i}: D waited {gap:?}");
        }
    }
}

#[tokio::test]
async fn grpc_pd_nixl_push_prefill_failure_fails_the_request() {
    let (prefill, decode, router) = push_pair().await;
    let warmup = router.route_chat(None, &chat(false), None).await;
    assert!(warmup.status().is_success());
    prefill.state.failures_remaining.store(1, Ordering::Relaxed);

    let response = router.route_chat(None, &chat(true), None).await;
    assert_eq!(
        response.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(decode.captured().len(), 2, "D was sent alongside P");
    assert_eq!(
        worker_counter("vllm_router_pd_prefill_errors_total", &prefill.grpc_url),
        1.0
    );
    assert_eq!(
        worker_counter("vllm_router_pd_decode_errors_total", &decode.grpc_url),
        0.0
    );
}

#[tokio::test]
async fn grpc_pd_nixl_push_decode_stream_failure_cancels_prefill() {
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(push_kv()),
        reply_delay: std::time::Duration::from_secs(1),
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router(&prefill, &decode).await;
    let warmup = router.route_chat(None, &chat(false), None).await;
    assert!(warmup.status().is_success());
    axum::body::to_bytes(warmup.into_body(), usize::MAX)
        .await
        .unwrap();
    decode
        .state
        .stream_failures_remaining
        .store(1, Ordering::Relaxed);

    // D opens its stream, so the response is a 200 whose body carries the error.
    let response = router.route_chat(None, &chat(true), None).await;
    assert!(response.status().is_success());
    wait_for_cancel(&prefill, 0).await;
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(String::from_utf8_lossy(&bytes).contains("decode failed"));
    for worker in [&prefill, &decode] {
        let worker = router
            .worker_registry()
            .get_by_url(&worker.grpc_url)
            .unwrap();
        assert_eq!(worker.load(), 0, "{}", worker.url());
    }
    assert_eq!(
        worker_counter("vllm_router_pd_decode_errors_total", &decode.grpc_url),
        1.0
    );
}

#[tokio::test]
async fn grpc_pd_nixl_push_aborted_decode_cancels_prefill() {
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(push_kv()),
        reply_delay: std::time::Duration::from_secs(1),
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router(&prefill, &decode).await;
    let warmup = router.route_chat(None, &chat(false), None).await;
    assert!(warmup.status().is_success());
    decode
        .state
        .aborted_finishes_remaining
        .store(1, Ordering::Relaxed);

    let response = router.route_chat(None, &chat(true), None).await;
    assert!(response.status().is_success());
    wait_for_cancel(&prefill, 0).await;
    let request_id = prefill.captured()[1].request_id.clone();
    wait_for_abort(&prefill, &request_id).await;
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(String::from_utf8_lossy(&bytes).contains("\"finish_reason\":\"abort\""));
    let p = router
        .worker_registry()
        .get_by_url(&prefill.grpc_url)
        .unwrap();
    assert_eq!(p.load(), 0);
    assert_eq!(
        worker_counter("vllm_router_pd_decode_errors_total", &decode.grpc_url),
        1.0
    );
}

#[tokio::test]
async fn grpc_pd_nixl_push_aborted_non_streaming_decode_cancels_prefill() {
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(push_kv()),
        reply_delay: std::time::Duration::from_secs(1),
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router(&prefill, &decode).await;
    let warmup = router.route_chat(None, &chat(false), None).await;
    assert!(warmup.status().is_success());
    decode
        .state
        .aborted_finishes_remaining
        .store(1, Ordering::Relaxed);

    let started = std::time::Instant::now();
    let response = router.route_chat(None, &chat(false), None).await;
    assert_eq!(response.status(), axum::http::StatusCode::BAD_GATEWAY);
    assert!(
        started.elapsed() < std::time::Duration::from_millis(500),
        "the router waited for P"
    );
    wait_for_cancel(&prefill, 0).await;
    let request_id = prefill.captured()[1].request_id.clone();
    wait_for_abort(&prefill, &request_id).await;
    for worker in [&prefill, &decode] {
        let worker = router
            .worker_registry()
            .get_by_url(&worker.grpc_url)
            .unwrap();
        assert_eq!(worker.load(), 0, "{}", worker.url());
    }
    assert_eq!(
        worker_counter("vllm_router_pd_decode_errors_total", &decode.grpc_url),
        1.0
    );
}

/// Drops a streaming response before D's first output, as a disconnecting
/// client does, and expects D's RPC to be cancelled and its load released.
async fn disconnect_before_first_token(
    router: &VllmPDRouter,
    decode: &MockVllmRsServer,
    cancelled_before: usize,
) {
    let response = router.route_chat(None, &chat(true), None).await;
    assert!(response.status().is_success());
    let d = router
        .worker_registry()
        .get_by_url(&decode.grpc_url)
        .unwrap();
    assert_eq!(d.load(), 1, "D stays loaded while its stream is open");

    drop(response);
    wait_for_cancel(decode, cancelled_before).await;
    assert_eq!(d.load(), 0);
    let request_id = decode.captured().last().unwrap().request_id.clone();
    wait_for_abort(decode, &request_id).await;
}

/// `Control.Abort` is sent from a spawned task after the leg is dropped.
async fn wait_for_abort(server: &MockVllmRsServer, request_id: &str) {
    wait_for_aborts(server, request_id, 1).await;
}

/// A leg dropped before its RPC was answered gets a second, delayed Abort.
async fn wait_for_aborts(server: &MockVllmRsServer, request_id: &str, count: usize) {
    let sent = || {
        server
            .aborts()
            .iter()
            .filter(|id| *id == request_id)
            .count()
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while sent() < count && std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(
        sent() >= count,
        "{} got aborts {:?}, expected {request_id} {count} time(s)",
        server.grpc_url,
        server.aborts()
    );
}

/// The decode error for a stream that fails after its 200 is recorded once
/// the producer stops.
async fn wait_for_decode_errors(decode: &MockVllmRsServer, count: f64) {
    let errors = || worker_counter("vllm_router_pd_decode_errors_total", &decode.grpc_url);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    while errors() < count && std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(errors(), count);
}

async fn wait_for_cancel(server: &MockVllmRsServer, cancelled_before: usize) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    while server.state.cancelled.load(Ordering::Relaxed) == cancelled_before
        && std::time::Instant::now() < deadline
    {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(
        server.state.cancelled.load(Ordering::Relaxed),
        cancelled_before + 1,
        "the RPC must be cancelled, not left waiting for its next output"
    );
}

/// Drops the request while P is still prefilling, as a disconnecting client
/// does, and expects P's RPC to be cancelled and its load released.
async fn disconnect_during_prefill(router: &VllmPDRouter, prefill: &MockVllmRsServer) {
    let p = router
        .worker_registry()
        .get_by_url(&prefill.grpc_url)
        .unwrap();
    let cancelled_before = prefill.state.cancelled.load(Ordering::Relaxed);
    let request = chat(true);
    let mut pending = Box::pin(router.route_chat(None, &request, None));
    tokio::select! {
        _ = &mut pending => panic!("request finished before prefill did"),
        _ = tokio::time::sleep(PREFILL_DELAY / 3) => {}
    }
    assert_eq!(p.load(), 1, "P stays loaded while prefilling");

    drop(pending);
    wait_for_cancel(prefill, cancelled_before).await;
    assert_eq!(p.load(), 0);
    let request_id = prefill.captured().last().unwrap().request_id.clone();
    wait_for_aborts(prefill, &request_id, 2).await;
}

fn slow_decode() -> MockVllmRs {
    MockVllmRs {
        reply_delay: std::time::Duration::from_secs(5),
        ..Default::default()
    }
}

#[tokio::test]
async fn grpc_pd_client_disconnect_cancels_decode_stream() {
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(prefill_kv()),
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn_with_state(slow_decode()).await;
    let router = pd_router(&prefill, &decode).await;

    disconnect_before_first_token(&router, &decode, 0).await;
    let p = router
        .worker_registry()
        .get_by_url(&prefill.grpc_url)
        .unwrap();
    assert_eq!(p.load(), 0);
    // D answered before the drop, so vLLM had registered it: one Abort.
    tokio::time::sleep(std::time::Duration::from_millis(800)).await;
    assert_eq!(decode.aborts().len(), 1, "{:?}", decode.aborts());
    assert_eq!(
        worker_counter("vllm_router_pd_decode_errors_total", &decode.grpc_url),
        0.0,
        "a client disconnect is not a decode error"
    );
}

#[tokio::test]
async fn grpc_pd_aborted_decode_stream_records_decode_error() {
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(prefill_kv()),
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router(&prefill, &decode).await;
    decode
        .state
        .aborted_finishes_remaining
        .store(1, Ordering::Relaxed);

    let response = router.route_chat(None, &chat(true), None).await;
    assert!(response.status().is_success());
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(String::from_utf8_lossy(&bytes).contains("\"finish_reason\":\"abort\""));
    wait_for_decode_errors(&decode, 1.0).await;

    let response = router.route_chat(None, &chat(true), None).await;
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    assert_eq!(
        worker_counter("vllm_router_pd_decode_errors_total", &decode.grpc_url),
        1.0,
        "a clean stream records no decode error"
    );
}

#[tokio::test]
async fn grpc_pd_info_endpoints_use_grpc_control() {
    let prefill = MockVllmRsServer::spawn().await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router(&prefill, &decode).await;
    let get = |uri: &str| {
        axum::http::Request::builder()
            .uri(uri)
            .body(axum::body::Body::empty())
            .unwrap()
    };
    let json_body = |response: axum::response::Response| async move {
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice::<Value>(&bytes).unwrap()
    };

    let models = json_body(router.get_models(get("/v1/models")).await).await;
    assert_eq!(models["data"][0]["id"], "mock-model");
    let model_info = json_body(router.get_model_info(get("/get_model_info")).await).await;
    assert_eq!(model_info["model_id"], "mock-model");
    let server_info = json_body(router.get_server_info(get("/get_server_info")).await).await;
    assert_eq!(server_info["engine_version"], "mock");

    let health_generate = router.health_generate(get("/health_generate")).await;
    assert_eq!(
        health_generate.status(),
        axum::http::StatusCode::NOT_IMPLEMENTED
    );
    assert_eq!(
        router.flush_cache().await.status(),
        axum::http::StatusCode::NOT_IMPLEMENTED
    );
    assert!(prefill.captured().is_empty() && decode.captured().is_empty());
}

#[tokio::test]
async fn grpc_pd_nixl_push_client_disconnect_cancels_decode_stream() {
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(push_kv()),
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn_with_state(slow_decode()).await;
    let router = pd_router(&prefill, &decode).await;

    // The first request runs sequentially and caches P's push identity;
    // the second takes the concurrent path.
    disconnect_before_first_token(&router, &decode, 0).await;
    disconnect_before_first_token(&router, &decode, 1).await;
    assert_eq!(decode.captured().len(), 2);
    let p = router
        .worker_registry()
        .get_by_url(&prefill.grpc_url)
        .unwrap();
    assert_eq!(p.load(), 0);
}

#[tokio::test]
async fn grpc_pd_client_disconnect_during_prefill_cancels_prefill() {
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(prefill_kv()),
        reply_delay: PREFILL_DELAY,
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router(&prefill, &decode).await;

    disconnect_during_prefill(&router, &prefill).await;
    assert!(decode.captured().is_empty(), "D is never reached");
}

#[tokio::test]
async fn grpc_pd_nixl_push_client_disconnect_during_prefill_releases_both_legs() {
    let (prefill, decode, router) = push_pair().await;
    let warmup = router.route_chat(None, &chat(false), None).await;
    assert!(warmup.status().is_success());
    axum::body::to_bytes(warmup.into_body(), usize::MAX)
        .await
        .unwrap();

    disconnect_during_prefill(&router, &prefill).await;
    assert_eq!(decode.captured().len(), 2, "D was sent alongside P");
    let d = router
        .worker_registry()
        .get_by_url(&decode.grpc_url)
        .unwrap();
    assert_eq!(d.load(), 0);
}

#[tokio::test]
async fn grpc_pd_rejects_non_chat_routes_before_dispatch() {
    let prefill = MockVllmRsServer::spawn().await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router(&prefill, &decode).await;

    let response = router
        .route_transparent(
            None,
            "/v1/completions",
            &axum::http::Method::POST,
            json!({"model": "test-model", "prompt": "hi", "max_tokens": 4}),
        )
        .await;
    assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(String::from_utf8_lossy(&bytes).contains("/v1/chat/completions"));
    assert!(prefill.captured().is_empty() && decode.captured().is_empty());
}

/// Turns `VLLM_ROUTER_L0_CACHE` and `VLLM_ROUTER_L1_CACHE` on for one test
/// and puts the process environment back on drop, including after a panic.
struct EncodingCacheEnv(Vec<(&'static str, Option<std::ffi::OsString>)>);

impl EncodingCacheEnv {
    fn enable() -> Self {
        let keys = [
            "VLLM_ROUTER_L0_CACHE",
            "VLLM_ROUTER_L1_CACHE",
            "VLLM_ROUTER_MODEL",
            "VLLM_ROUTER_TOKENIZER",
        ];
        let saved = keys
            .into_iter()
            .map(|key| (key, std::env::var_os(key)))
            .collect();
        std::env::set_var("VLLM_ROUTER_L0_CACHE", "1");
        std::env::set_var("VLLM_ROUTER_L1_CACHE", "1");
        // The request model is the fixture directory. A shell override would
        // send the tokenizer at a different checkpoint.
        std::env::remove_var("VLLM_ROUTER_MODEL");
        std::env::remove_var("VLLM_ROUTER_TOKENIZER");
        Self(saved)
    }
}

impl Drop for EncodingCacheEnv {
    fn drop(&mut self) {
        for (key, value) in &self.0 {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }
}

fn cache_chat(model: &std::path::Path, messages: Value) -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": model.to_str().unwrap(),
        "messages": messages,
        "max_tokens": 8,
        "stream": false,
    }))
    .unwrap()
}

/// L0 and L1 on, with a real tokenizer. A repeated chat hits L0. A longer
/// chat that shares the prefix hits L1. Both legs still receive the same
/// token ids as an uncached frontend, plus the usual KV hand-off.
/// The other tests in this file pin token ids and leave both caches off.
#[tokio::test]
async fn grpc_pd_encoding_caches_keep_token_ids_on_both_legs() {
    let _env = EncodingCacheEnv::enable();
    let model = fixture::qwen_model_fixture();
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(prefill_kv()),
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router_with_pools(&[&prefill], &decode, 1, PolicyConfig::RoundRobin).await;
    let (l0, l1) = router.test_encoding_cache_stats();
    assert!(l0.is_some() && l1.is_some());

    let off = EngineFrontend::new();
    let short_messages = json!([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "question 0"},
    ]);
    let short = cache_chat(model.path(), short_messages.clone());
    let expected_short = off
        .prepare(short.clone())
        .await
        .unwrap()
        .tokenized
        .token_ids
        .to_vec();

    let (p_ids, d_ids) = send_chat(&router, &prefill, &decode, &short).await;
    assert_eq!(p_ids, expected_short);
    assert_eq!(d_ids, expected_short);
    let (l0, l1) = router.test_encoding_cache_stats();
    let l0 = l0.unwrap();
    let l1 = l1.unwrap();
    assert!(l0.misses >= 1, "first prompt must miss L0");
    assert_eq!(l0.hits, 0);
    assert!(l1.lookups >= 1, "first prompt must enter L1");
    let l1_lookups = l1.lookups;
    let l1_hits = l1.hits;

    let (p_ids, d_ids) = send_chat(&router, &prefill, &decode, &short).await;
    assert_eq!(p_ids, expected_short);
    assert_eq!(d_ids, expected_short);
    let (l0, l1) = router.test_encoding_cache_stats();
    let l0 = l0.unwrap();
    let l1 = l1.unwrap();
    assert!(l0.hits >= 1, "repeat must hit L0");
    assert_eq!(l1.lookups, l1_lookups, "an L0 hit does not consult L1");

    let grown = cache_chat(
        model.path(),
        json!([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question 0"},
            {"role": "assistant", "content": "answer 0"},
            {"role": "user", "content": "question 1"},
        ]),
    );
    let expected_grown = off
        .prepare(grown.clone())
        .await
        .unwrap()
        .tokenized
        .token_ids
        .to_vec();
    assert_ne!(expected_grown, expected_short);
    let (p_ids, d_ids) = send_chat(&router, &prefill, &decode, &grown).await;
    assert_eq!(p_ids, expected_grown);
    assert_eq!(d_ids, expected_grown);
    let (_, l1) = router.test_encoding_cache_stats();
    let l1 = l1.unwrap();
    assert!(l1.hits > l1_hits, "the shared prefix must hit L1");
    assert!(l1.cached_tokens > 0);

    let p = prefill.captured();
    let d = decode.captured();
    assert_eq!((p.len(), d.len()), (3, 3));
    for (p, d) in p.iter().zip(&d) {
        assert!(p.unary);
        assert_eq!((p.max_new_tokens, p.min_new_tokens), (1, 1));
        assert_eq!(
            p.kv_transfer_params.as_ref().unwrap()["do_remote_decode"],
            true
        );
        assert!(!d.unary);
        assert_eq!(d.kv_transfer_params.as_ref(), Some(&prefill_kv()));
        assert_eq!(d.token_ids, p.token_ids);
        assert_eq!(d.request_id, p.request_id);
        assert!(!d.had_text_prompt && !p.had_text_prompt);
    }
    assert!(prefill.aborts().is_empty() && decode.aborts().is_empty());
}

async fn send_chat(
    router: &VllmPDRouter,
    prefill: &MockVllmRsServer,
    decode: &MockVllmRsServer,
    request: &ChatCompletionRequest,
) -> (Vec<u32>, Vec<u32>) {
    let before = prefill.captured().len();
    let response = router.route_chat(None, request, None).await;
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(
        status.is_success(),
        "{status}: {}",
        String::from_utf8_lossy(&bytes)
    );
    let p = prefill.captured();
    let d = decode.captured();
    assert_eq!(p.len(), before + 1);
    assert_eq!(d.len(), before + 1);
    (
        p.last().unwrap().token_ids.clone(),
        d.last().unwrap().token_ids.clone(),
    )
}

/// kv_aware admission on gRPC pools charges P the router's own prompt
/// tokens; it no longer falls back to occupancy for want of an HTTP render.
#[tokio::test]
async fn grpc_pd_kv_aware_admits_prefill_by_router_tokens() {
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(prefill_kv()),
        reply_delay: std::time::Duration::from_secs(1),
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn().await;
    let router = std::sync::Arc::new(
        pd_router_with_progress(&[&prefill], &decode, 1, kv_aware(), None).await,
    );
    router.pin_test_token_ids(vec![1, 2, 3, 4, 5]);

    let in_flight = {
        let router = router.clone();
        tokio::spawn(async move { router.route_chat(None, &chat(false), None).await.status() })
    };
    let url = &prefill.grpc_url;
    wait_for_gauge("vllm_router_pd_active_prefill_tokens", url, 5.0).await;
    assert_eq!(
        worker_gauge("vllm_router_pd_unknown_prefill_requests", url),
        0.0
    );
    assert!(in_flight.await.unwrap().is_success());
    wait_for_gauge("vllm_router_pd_active_prefill_tokens", url, 0.0).await;
    let p = &prefill.captured()[0];
    assert!(p.unary, "without progress opt-in P stays unary");
    assert_eq!(p.token_ids, vec![1, 2, 3, 4, 5]);
}

/// With progress on, P streams and its snapshots replace the admission
/// estimate while it runs.
#[tokio::test]
async fn grpc_pd_prefill_progress_replaces_admitted_work() {
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(prefill_kv()),
        supports_prefill_progress: true,
        prefill_progress: vec![4, 2],
        reply_delay: std::time::Duration::from_secs(1),
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn().await;
    let router = std::sync::Arc::new(
        pd_router_with_progress(&[&prefill], &decode, 1, kv_aware(), Some(5)).await,
    );
    router.pin_test_token_ids(vec![1, 2, 3, 4, 5]);

    let in_flight = {
        let router = router.clone();
        tokio::spawn(async move { router.route_chat(None, &chat(false), None).await.status() })
    };
    let url = &prefill.grpc_url;
    wait_for_gauge("vllm_router_pd_active_prefill_tokens", url, 2.0).await;
    assert_eq!(worker_gauge("vllm_router_pd_prefill_occupancy", url), 1.0);
    assert!(in_flight.await.unwrap().is_success());
    wait_for_gauge("vllm_router_pd_active_prefill_tokens", url, 0.0).await;
    wait_for_gauge("vllm_router_pd_prefill_occupancy", url, 0.0).await;

    let p = &prefill.captured()[0];
    assert!(!p.unary, "progress needs GenerateStream");
    assert!(p.include_prefill_progress);
    assert_eq!((p.max_new_tokens, p.min_new_tokens), (1, 1));
    // P's handoff still reaches D over the streaming leg.
    let d = &decode.captured()[0];
    assert_eq!(
        d.kv_transfer_params.as_ref().unwrap()["remote_engine_id"],
        "P"
    );
}

/// A worker without the capability keeps the unary leg and the admission
/// estimate.
#[tokio::test]
async fn grpc_pd_prefill_progress_falls_back_for_unsupported_workers() {
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(prefill_kv()),
        prefill_progress: vec![1],
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router_with_progress(&[&prefill], &decode, 1, kv_aware(), Some(5)).await;
    router.pin_test_token_ids(vec![1, 2, 3]);

    let response = router.route_chat(None, &chat(false), None).await;
    assert!(response.status().is_success());
    let p = &prefill.captured()[0];
    assert!(p.unary);
    assert!(!p.include_prefill_progress);
    assert_eq!(
        worker_gauge("vllm_router_pd_active_prefill_tokens", &prefill.grpc_url),
        0.0
    );
}
