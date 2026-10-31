#![cfg(all(feature = "cpu", feature = "ready"))]

//! gRPC counterparts of the PD-disaggregation policy cases in
//! `tests/python/integration/test_pd_disagg_routing.py`.
//!
//! vLLM PD mode talks to prefill and decode workers over HTTP. These cases
//! keep the same scheduling behavior and send it through a regular pool of
//! in-process `grpc://` Inference workers.
//!
//!   cargo test --features cpu,ready --test grpc_policy_routing

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::create_test_context;
use common::mock_vllm_rs::{MockVllmRs, MockVllmRsServer};
use serde_json::{json, Value};
use vllm_router_rs::config::{PolicyConfig, RouterConfig, RoutingMode};
use vllm_router_rs::protocols::spec::CompletionRequest;
use vllm_router_rs::routers::http::router::Router;
use vllm_router_rs::routers::RouterTrait;

fn config(urls: Vec<String>, policy: PolicyConfig) -> RouterConfig {
    RouterConfig {
        mode: RoutingMode::Regular { worker_urls: urls },
        policy,
        connection_mode: vllm_router_rs::config::ConnectionMode::Grpc,
        worker_startup_timeout_secs: 5,
        worker_startup_check_interval_secs: 30,
        request_timeout_secs: 30,
        disable_retries: true,
        disable_circuit_breaker: true,
        ..RouterConfig::default()
    }
}

fn cache_aware() -> PolicyConfig {
    PolicyConfig::CacheAware {
        cache_threshold: 0.3,
        balance_abs_threshold: 64,
        balance_rel_threshold: 1.5,
        eviction_interval_secs: 60,
        max_tree_size: 1 << 24,
    }
}

fn kv_aware() -> PolicyConfig {
    PolicyConfig::KvAware {
        router_temperature: 0.0,
        kv_penalty_host: 0.8,
        kv_penalty_external: 0.6,
        kv_penalty_disk: 0.3,
        static_kv_event_sources: Box::default(),
    }
}

async fn workers(n: usize, delay: Duration) -> Vec<MockVllmRsServer> {
    let mut servers = Vec::with_capacity(n);
    for _ in 0..n {
        servers.push(
            MockVllmRsServer::spawn_with_state(MockVllmRs {
                response_delay: delay,
                ..MockVllmRs::default()
            })
            .await,
        );
    }
    servers
}

async fn start_router(servers: &[MockVllmRsServer], policy: PolicyConfig) -> Router {
    let urls: Vec<String> = servers
        .iter()
        .map(|server| server.grpc_url.clone())
        .collect();
    let cfg = config(urls.clone(), policy);
    let router = Router::new(urls, &create_test_context(cfg))
        .await
        .expect("grpc workers should pass grpc.health.v1");
    // Completions already carry token ids. Pinning skips the Hugging Face
    // tokenizer lookup that `prepare_token_ids` still performs for sampling defaults.
    router.pin_test_token_ids(vec![1]);
    router
}

fn hits(servers: &[MockVllmRsServer]) -> Vec<usize> {
    servers
        .iter()
        .map(|server| server.captured().len())
        .collect()
}

fn grew(before: &[usize], after: &[usize]) -> usize {
    let grown: Vec<usize> = after
        .iter()
        .zip(before)
        .enumerate()
        .filter(|(_, (now, was))| *now > *was)
        .map(|(index, _)| index)
        .collect();
    assert_eq!(grown.len(), 1, "before={before:?} after={after:?}");
    grown[0]
}

async fn complete(router: &Router, servers: &[MockVllmRsServer], body: Value) -> usize {
    let before = hits(servers);
    let request: CompletionRequest = serde_json::from_value(body).expect("completion request");
    let response = router.route_completion(None, &request, None).await;
    let status = response.status();
    let payload = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body");
    assert!(
        status.is_success(),
        "status={status} body={}",
        String::from_utf8_lossy(&payload)
    );
    grew(&before, &hits(servers))
}

fn prompt(ids: &[i32], extra: Value) -> Value {
    let mut body = json!({
        "model": "test-model",
        "prompt": ids,
        "max_tokens": 1,
        "stream": false,
    });
    if let (Some(dest), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
        for (key, value) in extra {
            dest.insert(key.clone(), value.clone());
        }
    }
    body
}

async fn wait_for_hit(servers: &[MockVllmRsServer], before: &[usize]) {
    let start = std::time::Instant::now();
    loop {
        if hits(servers).iter().zip(before).any(|(now, was)| now > was) {
            return;
        }
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "in-flight gRPC request was not recorded"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn round_robin_spreads_across_grpc_workers() {
    let servers = workers(3, Duration::ZERO).await;
    let router = start_router(&servers, PolicyConfig::RoundRobin).await;
    let mut counts = [0usize; 3];
    for i in 0..30 {
        let index = complete(&router, &servers, prompt(&[i], json!({}))).await;
        counts[index] += 1;
    }
    assert_eq!(counts, [10, 10, 10], "hits={:?}", hits(&servers));
}

#[tokio::test]
async fn random_spreads_across_grpc_workers() {
    let servers = workers(4, Duration::ZERO).await;
    let router = start_router(&servers, PolicyConfig::Random).await;
    let total = 200usize;
    for i in 0..total {
        complete(&router, &servers, prompt(&[i as i32], json!({}))).await;
    }
    let mean = total as f64 / servers.len() as f64;
    for count in hits(&servers) {
        assert!(
            (0.5 * mean) <= count as f64 && count as f64 <= (1.5 * mean),
            "hits={:?} mean={mean}",
            hits(&servers)
        );
    }
}

#[tokio::test]
async fn consistent_hash_sticks_to_session() {
    let servers = workers(3, Duration::ZERO).await;
    let router = start_router(
        &servers,
        PolicyConfig::ConsistentHash { virtual_nodes: 160 },
    )
    .await;
    let mut chosen = Vec::new();
    for session in ["alpha", "beta", "gamma", "delta"] {
        let mut workers_for_session = Vec::new();
        for turn in 0..8 {
            workers_for_session.push(
                complete(
                    &router,
                    &servers,
                    prompt(&[turn], json!({"session_params": {"session_id": session}})),
                )
                .await,
            );
        }
        assert_eq!(
            workers_for_session
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            1,
            "{session}: {workers_for_session:?}"
        );
        chosen.push(workers_for_session[0]);
    }
    assert!(
        chosen
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            >= 2,
        "sessions collapsed onto one worker: {chosen:?}"
    );
}

#[tokio::test]
async fn cache_aware_prefix_overrides_load() {
    let servers = workers(2, Duration::from_millis(800)).await;
    let router = start_router(&servers, cache_aware()).await;
    let cached_prompt = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let cached = complete(&router, &servers, prompt(&cached_prompt, json!({}))).await;

    let before = hits(&servers);
    let router_for_holder = Arc::new(router);
    let holder = {
        let router = Arc::clone(&router_for_holder);
        let body = prompt(
            &[100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            json!({}),
        );
        tokio::spawn(async move {
            let request: CompletionRequest = serde_json::from_value(body).unwrap();
            let response = router.route_completion(None, &request, None).await;
            assert!(response.status().is_success());
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
        })
    };
    wait_for_hit(&servers, &before).await;
    let busy = grew(&before, &hits(&servers));

    let diverted = complete(
        &router_for_holder,
        &servers,
        prompt(
            &[200, 201, 202, 203, 204, 205, 206, 207, 208, 209],
            json!({}),
        ),
    )
    .await;
    let replay = complete(
        &router_for_holder,
        &servers,
        prompt(&cached_prompt, json!({})),
    )
    .await;
    holder.await.unwrap();

    assert_eq!(busy, cached, "holder should occupy the cached worker");
    assert_ne!(
        diverted, cached,
        "unrelated prompt should avoid the busy worker"
    );
    assert_eq!(replay, cached, "cached prefix should stay despite the load");
}

#[tokio::test]
async fn kv_aware_prefers_the_less_loaded_worker() {
    let servers = workers(2, Duration::ZERO).await;
    let router = start_router(&servers, kv_aware()).await;
    let mut first = None;
    for i in 0..8 {
        let index = complete(&router, &servers, prompt(&[i], json!({}))).await;
        first.get_or_insert(index);
        assert_eq!(index, first.unwrap());
    }

    let delayed = workers(2, Duration::from_millis(800)).await;
    let router = start_router(&delayed, kv_aware()).await;
    let cached = complete(&router, &delayed, prompt(&[1, 2, 3], json!({}))).await;
    let before = hits(&delayed);
    let router = Arc::new(router);
    let holder = {
        let router = Arc::clone(&router);
        tokio::spawn(async move {
            let request: CompletionRequest =
                serde_json::from_value(prompt(&[9, 9, 9], json!({}))).unwrap();
            let response = router.route_completion(None, &request, None).await;
            assert!(response.status().is_success());
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
        })
    };
    wait_for_hit(&delayed, &before).await;
    assert_eq!(grew(&before, &hits(&delayed)), cached);
    let diverted = complete(&router, &delayed, prompt(&[4, 5, 6], json!({}))).await;
    holder.await.unwrap();
    assert_ne!(diverted, cached);
}

#[tokio::test]
async fn kv_aware_affinity_sticks_to_session() {
    let servers = workers(2, Duration::from_millis(800)).await;
    let urls: Vec<String> = servers
        .iter()
        .map(|server| server.grpc_url.clone())
        .collect();
    let mut cfg = config(urls.clone(), kv_aware());
    cfg.kv_aware_session_affinity_ttl_secs = Some(60);
    let router = Router::new(urls, &create_test_context(cfg)).await.unwrap();
    router.pin_test_token_ids(vec![1]);
    let router = Arc::new(router);

    let session =
        |id: &str, turn: i32| prompt(&[turn], json!({"session_params": {"session_id": id}}));
    let alpha = complete(&router, &servers, session("alpha", 1)).await;
    let before = hits(&servers);
    let holder = {
        let router = Arc::clone(&router);
        tokio::spawn(async move {
            let request: CompletionRequest =
                serde_json::from_value(prompt(&[7, 7, 7], json!({}))).unwrap();
            let response = router.route_completion(None, &request, None).await;
            assert!(response.status().is_success());
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
        })
    };
    wait_for_hit(&servers, &before).await;
    let beta = complete(&router, &servers, session("beta", 1)).await;
    let alpha_again = complete(&router, &servers, session("alpha", 2)).await;
    let beta_again = complete(&router, &servers, session("beta", 2)).await;
    holder.await.unwrap();

    assert_ne!(
        beta, alpha,
        "a busy session binding should leave the other worker"
    );
    assert_eq!(alpha_again, alpha);
    assert_eq!(beta_again, beta);
}

#[tokio::test]
async fn program_scheduling_sticks_to_program() {
    let servers = workers(2, Duration::ZERO).await;
    let urls: Vec<String> = servers
        .iter()
        .map(|server| server.grpc_url.clone())
        .collect();
    let mut cfg = config(urls.clone(), PolicyConfig::RoundRobin);
    cfg.program_scheduling = Some(
        serde_json::from_value(json!({
            "binding_only": true,
            "queue_timeout_seconds": 2,
            "metrics_interval_seconds": 0.05
        }))
        .unwrap(),
    );
    let router = Router::new(urls, &create_test_context(cfg)).await.unwrap();
    router.pin_test_token_ids(vec![1]);
    let program = |id: &str| {
        prompt(
            &[1, 2, 3, 4],
            json!({"max_tokens": 8, "vllm_xargs": {"agentic_context": {
                "program_id": id,
                "expected_resume": true
            }}}),
        )
    };
    let mut alpha = Vec::new();
    let mut beta = Vec::new();
    for _ in 0..4 {
        alpha.push(complete(&router, &servers, program("program-alpha")).await);
        beta.push(complete(&router, &servers, program("program-beta")).await);
    }
    assert_eq!(
        alpha.iter().collect::<std::collections::HashSet<_>>().len(),
        1,
        "{alpha:?}"
    );
    assert_eq!(
        beta.iter().collect::<std::collections::HashSet<_>>().len(),
        1,
        "{beta:?}"
    );
}

#[tokio::test]
async fn power_of_two_prefers_the_less_loaded_worker() {
    let slow = MockVllmRsServer::spawn_with_state(MockVllmRs {
        response_delay: Duration::from_secs(3),
        ..MockVllmRs::default()
    })
    .await;
    let fast = MockVllmRsServer::spawn_with_state(MockVllmRs::default()).await;
    let servers = vec![slow, fast];
    let router = Arc::new(
        start_router(
            &servers,
            PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 30,
            },
        )
        .await,
    );

    // Leave one request in flight on the slow worker. A fast hit is drained.
    let holder = loop {
        let before = hits(&servers);
        let router_for_task = Arc::clone(&router);
        let task = tokio::spawn(async move {
            let request: CompletionRequest =
                serde_json::from_value(prompt(&[1, 2, 3], json!({}))).unwrap();
            let response = router_for_task.route_completion(None, &request, None).await;
            assert!(response.status().is_success());
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
        });
        wait_for_hit(&servers, &before).await;
        if grew(&before, &hits(&servers)) == 0 {
            break task;
        }
        task.await.unwrap();
    };
    let mut avoided = 0usize;
    for i in 0..12 {
        if complete(&router, &servers, prompt(&[20 + i], json!({}))).await != 0 {
            avoided += 1;
        }
    }
    holder.await.unwrap();
    assert_eq!(
        avoided, 12,
        "power-of-two should avoid the in-flight worker"
    );
}
