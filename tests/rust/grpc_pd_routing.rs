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

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::Ordering;
use std::sync::OnceLock;

use common::create_test_context;
use common::mock_vllm_rs::{MockVllmRs, MockVllmRsServer};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serde_json::{json, Value};
use vllm_router_rs::backend::EngineFrontend;
use vllm_router_rs::config::{
    KvConnector, PdKvEventSources, PolicyConfig, RouterConfig, RoutingMode,
};
use vllm_router_rs::policies::PolicyFactory;
use vllm_router_rs::protocols::spec::ChatCompletionRequest;
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
    pd_router_with_kv_sources(prefills, decode, dp_size, prefill_policy, HashMap::new()).await
}

async fn pd_router_with_kv_sources(
    prefills: &[&MockVllmRsServer],
    decode: &MockVllmRsServer,
    dp_size: usize,
    prefill_policy: PolicyConfig,
    pd_kv_event_sources: HashMap<String, PdKvEventSources>,
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
            decode_policy: None,
            discovery_address: None,
        },
        policy: PolicyConfig::RoundRobin,
        pd_kv_event_sources,
        kv_connector: KvConnector::Nixl,
        worker_startup_timeout_secs: 5,
        intra_node_data_parallel_size: dp_size,
        ..RouterConfig::default()
    };
    vllm_router_rs::config::ConfigValidator::validate(&config).unwrap();
    let ctx = create_test_context(config);
    // Separate P and D policy instances, as `RouterFactory` installs them.
    let policies = &ctx.policy_registry;
    let affinity = policies.kv_affinity_table();
    policies.set_prefill_policy(PolicyFactory::create_from_config(
        &prefill_policy,
        Some(&affinity),
    ));
    policies.set_decode_policy(PolicyFactory::create_from_config(
        &PolicyConfig::RoundRobin,
        Some(&affinity),
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

#[tokio::test]
async fn grpc_pd_power_of_two_uses_local_prefill_load() {
    let busy = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(prefill_kv()),
        reply_delay: std::time::Duration::from_secs(30),
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
    let router = pd_router_with_pools(&[&busy, &idle], &decode, 1, policy).await;
    router.pin_test_token_ids(vec![1, 2, 3]);
    // Let the load monitor run its first pass; it caches 0 for grpc:// URLs.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    // Hold one admitted request on the slow prefill. Requests that land on
    // the idle prefill finish right away.
    let request = chat(false);
    let mut held = None;
    for _ in 0..32 {
        let mut pending = Box::pin(router.route_chat(None, &request, None));
        tokio::select! {
            response = &mut pending => assert!(response.status().is_success()),
            _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                held = Some(pending);
                break;
            }
        }
    }
    assert!(held.is_some(), "no request reached the slow prefill");
    assert_eq!(busy.captured().len(), 1);
    let idle_before = idle.captured().len();

    for _ in 0..8 {
        let response = router.route_chat(None, &chat(false), None).await;
        assert!(response.status().is_success());
    }
    assert_eq!(busy.captured().len(), 1, "the busy prefill was picked");
    assert_eq!(idle.captured().len(), idle_before + 8);
    drop(held);
}

/// One prefill's KV event publisher, as `--pd-kv-events` would name it.
struct KvPublisher {
    _context: zmq::Context,
    socket: zmq::Socket,
    endpoint: String,
}

impl KvPublisher {
    fn bind() -> Self {
        let context = zmq::Context::new();
        let socket = context.socket(zmq::PUB).unwrap();
        socket.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = socket.get_last_endpoint().unwrap().unwrap();
        Self {
            _context: context,
            socket,
            endpoint,
        }
    }

    /// vLLM's `BlockStored` batch for two 16-token blocks.
    fn store_blocks(&self, token_ids: &[u32]) {
        let payload = rmp_serde::to_vec(&(
            1.0,
            vec![json!({
                "type": "BlockStored", "block_hashes": [1, 2], "parent_block_hash": null,
                "token_ids": token_ids, "block_size": 16,
                "lora_id": null, "lora_name": null, "medium": "GPU",
            })],
            0_u32,
        ))
        .unwrap();
        let seq = 0_u64.to_be_bytes();
        self.socket
            .send_multipart([b"".as_slice(), seq.as_slice(), payload.as_slice()], 0)
            .unwrap();
    }
}

/// kv_aware on `grpc://` prefills scores the router's own token ids against
/// each prefill's KV events, and sends a request to the prefill that already
/// holds its prefix. The sources carry no model, as with `--pd-kv-events`.
#[tokio::test]
async fn grpc_pd_kv_aware_prefers_the_prefill_holding_the_prefix() {
    // Run with each prefill warm so a cold tie-break cannot pass the test.
    for warm_index in [0, 1] {
        let prefills = [
            MockVllmRsServer::spawn_with_state(MockVllmRs {
                finish_kv_transfer_params: Some(prefill_kv()),
                ..Default::default()
            })
            .await,
            MockVllmRsServer::spawn_with_state(MockVllmRs {
                finish_kv_transfer_params: Some(prefill_kv()),
                ..Default::default()
            })
            .await,
        ];
        let publishers = [KvPublisher::bind(), KvPublisher::bind()];
        let decode = MockVllmRsServer::spawn().await;
        let sources = prefills
            .iter()
            .zip(&publishers)
            .map(|(prefill, publisher)| {
                (
                    prefill.grpc_url.clone(),
                    PdKvEventSources {
                        endpoints: vec![publisher.endpoint.clone()],
                        replay_endpoints: None,
                        block_size: 16,
                        model: None,
                    },
                )
            })
            .collect();
        let policy = PolicyConfig::KvAware {
            router_temperature: 0.0,
            kv_penalty_host: 0.8,
            kv_penalty_external: 0.6,
            kv_penalty_disk: 0.4,
            static_kv_event_sources: Default::default(),
        };
        let router =
            pd_router_with_kv_sources(&[&prefills[0], &prefills[1]], &decode, 1, policy, sources)
                .await;
        let prompt: Vec<u32> = (0..33).collect();
        router.pin_test_token_ids(prompt.clone());

        let warm = &prefills[warm_index];
        let cold = &prefills[1 - warm_index];
        // The subscriber connects asynchronously; republish until a request
        // lands on the warm prefill.
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                publishers[warm_index].store_blocks(&prompt[..32]);
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                let before = warm.captured().len();
                send_chat_ok(&router).await;
                if warm.captured().len() > before {
                    break;
                }
            }
        })
        .await
        .expect("kv_aware never picked the warm prefill");

        let (warm_before, cold_before) = (warm.captured().len(), cold.captured().len());
        for _ in 0..8 {
            send_chat_ok(&router).await;
        }
        assert_eq!(
            warm.captured().len(),
            warm_before + 8,
            "warm_index={warm_index}"
        );
        assert_eq!(cold.captured().len(), cold_before);
        assert!(warm
            .captured()
            .iter()
            .all(|request| request.token_ids == prompt));
    }
}

/// Names the model the mocks report from `Control.GetModelInfo`.
async fn send_chat_ok(router: &VllmPDRouter) {
    let mut request = chat(false);
    request.model = Some("mock-model".into());
    let response = router.route_chat(None, &request, None).await;
    assert!(response.status().is_success(), "{}", response.status());
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
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
