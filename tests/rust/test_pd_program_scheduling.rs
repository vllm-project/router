//! End-to-end HTTP PD tests: two turns, actual rank headers, opt-out and body drop.
#![cfg(all(feature = "cpu", feature = "ready"))]
use axum::{
    extract::{Json, State},
    http::HeaderMap,
    routing::{get, post},
    Router,
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use vllm_router_rs::{
    config::{ConfigValidator, PolicyConfig, ProgramSchedulingConfig, RouterConfig, RoutingMode},
    routers::{http::vllm_pd_router::VllmPDRouter, RouterTrait},
    server::AppContext,
};

#[derive(Clone)]
struct MockState {
    prefill: bool,
    calls: Arc<Mutex<Vec<(String, Value)>>>,
    body_gate: Option<Arc<tokio::sync::Semaphore>>,
    push: bool,
}
struct Mock {
    url: String,
    calls: Arc<Mutex<Vec<(String, Value)>>>,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Mock {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn mock(prefill: bool) -> Mock {
    controlled_mock(prefill, None).await
}
async fn controlled_mock(prefill: bool, body_gate: Option<Arc<tokio::sync::Semaphore>>) -> Mock {
    mock_with_options(prefill, body_gate, false).await
}
async fn mock_with_options(
    prefill: bool,
    body_gate: Option<Arc<tokio::sync::Semaphore>>,
    push: bool,
) -> Mock {
    let dp_size = if body_gate.is_some() { 1 } else { 2 };
    let calls = Arc::new(Mutex::new(Vec::new()));
    let state = MockState {
        prefill,
        calls: calls.clone(),
        body_gate,
        push,
    };
    let app = Router::new().route("/health", get(|| async { "ok" }))
        .route("/get_server_info", get(move || async move { Json(json!({"model_id":"test","data_parallel_size":dp_size})) }))
        .route("/v1/models", get(|| async { Json(json!({"data":[{"id":"test"}]})) }))
        .route("/metrics", get(move || async move {
            (0..dp_size).map(|rank| format!("vllm:kv_cache_usage_perc{{engine=\"{rank}\"}} 0\nvllm:num_requests_running{{engine=\"{rank}\"}} 0\nvllm:num_requests_waiting{{engine=\"{rank}\"}} 0\n")).collect::<String>()
        }))
        .route("/v1/completions", post(generate))
        .route("/v1/chat/completions", post(generate)).with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    Mock { url, calls, task }
}
async fn generate(
    State(state): State<MockState>,
    headers: HeaderMap,
    Json(request): Json<Value>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let rank = headers
        .get("x-data-parallel-rank")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("none")
        .to_string();
    state.calls.lock().unwrap().push((rank, request));
    let mut value = if state.prefill {
        json!({"kv_transfer_params":{"do_remote_prefill":true,"do_remote_decode":false,
            "remote_engine_id":"registered-prefill","remote_block_ids":[1]},
            "choices":[],"usage":{"prompt_tokens":24,"completion_tokens":1}})
    } else {
        json!({"id":"mock","choices":[{"text":"done"}],
            "usage":{"prompt_tokens":24,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":24}}})
    };
    if state.prefill && state.push {
        let params = &mut value["kv_transfer_params"];
        params["transfer_mode"] = json!("push");
        params["remote_host"] = json!("127.0.0.1");
        params["remote_port"] = json!(49999);
    }
    if let Some(gate) = state.body_gate {
        // P sends headers immediately, then stalls its body. D stalls before headers.
        if state.prefill {
            let stream = futures_util::stream::once(async move {
                gate.acquire().await.unwrap().forget();
                Ok::<_, std::convert::Infallible>(serde_json::to_vec(&value).unwrap())
            });
            return axum::http::Response::builder()
                .header("content-type", "application/json")
                .body(axum::body::Body::from_stream(stream))
                .unwrap();
        }
        gate.acquire().await.unwrap().forget();
    }
    Json(value).into_response()
}

async fn wait_calls(mock: &Mock, count: usize) {
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while mock.calls.lock().unwrap().len() < count {
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn prefill_slot_covers_full_p_body_but_not_decode_wait() {
    check_prefill_slot(false).await;
}
#[tokio::test]
async fn concurrent_push_releases_p_slot_before_decode_headers() {
    check_prefill_slot(true).await;
}
async fn check_prefill_slot(push: bool) {
    let p_gate = Arc::new(tokio::sync::Semaphore::new(0));
    let d_gate = Arc::new(tokio::sync::Semaphore::new(0));
    let p = mock_with_options(true, Some(p_gate.clone()), push).await;
    let d = controlled_mock(false, Some(d_gate.clone())).await;
    let router = Arc::new(router_with_dp(&p, &d, true, 1).await);
    let first_router = router.clone();
    let first = tokio::spawn(async move {
        first_router
            .route_completion(None, &request(true), Some("test"))
            .await
    });
    wait_calls(&p, 1).await;
    let second_router = router.clone();
    let second = tokio::spawn(async move {
        let mut value = serde_json::to_value(request(true)).unwrap();
        value["vllm_xargs"] = json!({"agentic_context":{"program_id":"other-task"}});
        let req = serde_json::from_value(value).unwrap();
        second_router
            .route_completion(None, &req, Some("test"))
            .await
    });
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    assert_eq!(
        p.calls.lock().unwrap().len(),
        1,
        "P headers cannot release the slot"
    );
    p_gate.add_permits(1);
    wait_calls(&d, 1).await;
    wait_calls(&p, 2).await;
    assert!(
        !first.is_finished(),
        "D is still waiting for response headers"
    );
    p_gate.add_permits(1);
    wait_calls(&d, 2).await;
    // The second turn can now use concurrent push; its P body must release the
    // slot while join is still waiting for D headers.
    let third_router = router.clone();
    let third = tokio::spawn(async move {
        let mut value = serde_json::to_value(request(true)).unwrap();
        value["vllm_xargs"] = json!({"agentic_context":{"program_id":"third-task"}});
        let req = serde_json::from_value(value).unwrap();
        third_router
            .route_completion(None, &req, Some("test"))
            .await
    });
    wait_calls(&p, 3).await;
    assert!(!second.is_finished(), "D headers cannot hold the P slot");
    p_gate.add_permits(1);
    wait_calls(&d, 3).await;
    d_gate.add_permits(3);
    for task in [first, second, third] {
        let response = tokio::time::timeout(std::time::Duration::from_secs(2), task)
            .await
            .unwrap()
            .unwrap();
        assert!(response.status().is_success());
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
    }
}
async fn router(p: &Mock, d: &Mock, enabled: bool) -> VllmPDRouter {
    router_with_dp(p, d, enabled, 2).await
}
async fn router_with_dp(p: &Mock, d: &Mock, enabled: bool, dp_size: usize) -> VllmPDRouter {
    router_with_capacity(p, d, enabled, dp_size, 100000).await
}
async fn router_with_capacity(
    p: &Mock,
    d: &Mock,
    enabled: bool,
    dp_size: usize,
    capacity: usize,
) -> VllmPDRouter {
    let mut config = RouterConfig {
        mode: RoutingMode::VllmPrefillDecode {
            prefill_urls: vec![(p.url.clone(), None)],
            decode_urls: vec![d.url.clone()],
            prefill_policy: Some(PolicyConfig::RoundRobin),
            decode_policy: Some(PolicyConfig::RoundRobin),
            discovery_address: None,
        },
        intra_node_data_parallel_size: dp_size,
        worker_startup_timeout_secs: 2,
        disable_circuit_breaker: true,
        ..RouterConfig::default()
    };
    if enabled {
        config.program_scheduling = Some(
            serde_json::from_value(json!({
                "token_capacity_per_dp_rank":capacity,"enable_batch_gain_admission":false,
                "decode_buffer_tokens":8,
                "metrics_interval_seconds":0.02,"queue_timeout_seconds":2,"force_resume_timeout_seconds":1,
                "pd":{"enabled":true,"explicit_ttl":true}
            }))
            .unwrap(),
        );
    }
    ConfigValidator::validate(&config).unwrap();
    let ctx = Arc::new(AppContext::new(config, reqwest::Client::new(), 64, None, vec![]).unwrap());
    let kv_affinity_table = ctx.policy_registry.kv_affinity_table();
    ctx.policy_registry.set_prefill_policy(
        vllm_router_rs::policies::PolicyFactory::create_from_config(
            &PolicyConfig::RoundRobin,
            Some(&kv_affinity_table),
        ),
    );
    ctx.policy_registry.set_decode_policy(
        vllm_router_rs::policies::PolicyFactory::create_from_config(
            &PolicyConfig::RoundRobin,
            Some(&kv_affinity_table),
        ),
    );
    VllmPDRouter::new(vec![(p.url.clone(), None)], vec![d.url.clone()], None, &ctx)
        .await
        .unwrap()
}
fn request(program: bool) -> vllm_router_rs::protocols::spec::CompletionRequest {
    let mut request = json!({"model":"test","prompt":"hello","max_tokens":8});
    if program {
        request["vllm_xargs"] = json!({"agentic_context":{"program_id":"agent-task",
        "expected_resume":true,"kv_retention_ttl_seconds":1.0}});
    }
    serde_json::from_value(request).unwrap()
}

#[tokio::test]
async fn token_id_input_fits_capacity_without_reserving_metadata_bytes() {
    let p = mock(true).await;
    let d = mock(false).await;
    let router = router_with_capacity(&p, &d, true, 2, 64).await;
    let mut value = serde_json::to_value(request(true)).unwrap();
    value["prompt"] = json!([100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007]);
    let request = serde_json::from_value(value).unwrap();
    let response = router.route_completion(None, &request, Some("test")).await;
    assert!(
        response.status().is_success(),
        "already-tokenized input must fit its actual token budget"
    );
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(p.calls.lock().unwrap().len(), 1);
    assert_eq!(d.calls.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn token_id_batch_uses_token_count_and_reserves_each_completion() {
    let p = mock(true).await;
    let d = mock(false).await;
    let router = router_with_capacity(&p, &d, true, 2, 64).await;
    let mut value = serde_json::to_value(request(true)).unwrap();
    value["prompt"] = json!([[100000, 100001], [100002, 100003]]);
    let request = serde_json::from_value(value.clone()).unwrap();
    let response = router.route_completion(None, &request, Some("test")).await;
    assert!(
        response.status().is_success(),
        "token IDs should count as four input tokens"
    );
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();

    value["vllm_xargs"]["agentic_context"]["program_id"] = json!("another-batch");
    value["max_tokens"] = json!(40);
    let request = serde_json::from_value(value).unwrap();
    let response = router.route_completion(None, &request, Some("test")).await;
    assert_eq!(response.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(p.calls.lock().unwrap().len(), 1);
    assert_eq!(d.calls.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn multiple_generated_choices_must_fit_rank_capacity() {
    let p = mock(true).await;
    let d = mock(false).await;
    let router = router_with_capacity(&p, &d, true, 2, 512).await;
    let mut value = serde_json::to_value(request(true)).unwrap();
    value["max_tokens"] = json!(256);
    value["n"] = json!(4);
    let request: vllm_router_rs::protocols::spec::CompletionRequest =
        serde_json::from_value(value).unwrap();
    let response = router.route_completion(None, &request, Some("test")).await;
    assert_eq!(
        response.status(),
        axum::http::StatusCode::TOO_MANY_REQUESTS,
        "n=4 with max_tokens=256 must reserve four output budgets"
    );
    assert!(p.calls.lock().unwrap().is_empty());
    assert!(d.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn program_keeps_actual_decode_rank_and_selects_prefill_each_turn() {
    let p = mock(true).await;
    let d = mock(false).await;
    let router = router(&p, &d, true).await;
    for _ in 0..2 {
        let response = router
            .route_completion(None, &request(true), Some("test"))
            .await;
        assert!(response.status().is_success());
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["choices"][0]["text"], "done");
    }
    let pc = p.calls.lock().unwrap();
    let dc = d.calls.lock().unwrap();
    assert_eq!(pc.len(), 2);
    assert_eq!(dc.len(), 2);
    assert_ne!(pc[0].0, pc[1].0, "P policy runs per turn");
    assert_eq!(dc[0].0, dc[1].0, "D rank is bound across turns");
    assert_ne!(dc[0].0, "none");
    assert_eq!(pc[0].1["max_tokens"], 1);
    assert_eq!(dc[0].1["max_tokens"], 8);
    assert_eq!(
        dc[1].1["kv_transfer_params"]["remote_engine_id"],
        "registered-prefill"
    );
}

#[tokio::test]
async fn no_identity_and_disabled_feature_keep_native_round_robin() {
    for (enabled, program) in [(true, false), (false, true)] {
        let p = mock(true).await;
        let d = mock(false).await;
        let router = router(&p, &d, enabled).await;
        for _ in 0..2 {
            let response = router
                .route_completion(None, &request(program), Some("test"))
                .await;
            assert!(response.status().is_success());
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
        }
        let dc = d.calls.lock().unwrap();
        assert_eq!(dc.len(), 2);
        assert_ne!(dc[0].0, dc[1].0, "native D policy still runs per request");
    }
}

#[test]
fn pd_requires_explicit_capacity_and_rejects_unverified_modes() {
    let valid: ProgramSchedulingConfig =
        serde_json::from_value(json!({"pd":{"enabled":true}})).unwrap();
    assert!(valid.pd.enabled);
    let mut config = RouterConfig {
        mode: RoutingMode::VllmPrefillDecode {
            prefill_urls: vec![("http://p:8000".into(), None)],
            decode_urls: vec!["http://d:8000".into()],
            prefill_policy: None,
            decode_policy: None,
            discovery_address: None,
        },
        program_scheduling: Some(valid),
        ..RouterConfig::default()
    };
    assert!(ConfigValidator::validate(&config).is_err());
    let program = config.program_scheduling.as_mut().unwrap();
    program.token_capacity_per_dp_rank = Some(100000);
    program.enable_batch_gain_admission = false;
    assert!(ConfigValidator::validate(&config).is_ok());
    for mutation in 0..4 {
        let mut bad = config.clone();
        let program = bad.program_scheduling.as_mut().unwrap();
        match mutation {
            0 => program.binding_only = true,
            1 => program.enable_batch_gain_admission = true,
            2 => program.pd.max_prefill_requests_per_target = 0,
            _ => program.pd.kv_hash_mode = "unknown".into(),
        }
        assert!(ConfigValidator::validate(&bad).is_err());
    }
    assert!(serde_json::from_value::<ProgramSchedulingConfig>(
        json!({"pd":{"enabled":true,"local_append":true}})
    )
    .is_err());
    assert!(serde_json::from_value::<ProgramSchedulingConfig>(
        json!({"pd":{"enabled":true,"bidirectional_multiturn":true}})
    )
    .is_err());
}

#[tokio::test]
async fn dropping_decode_body_unblocks_only_the_waiting_turn() {
    let p = mock(true).await;
    let d = mock(false).await;
    let router = Arc::new(router(&p, &d, true).await);
    let first = router
        .route_completion(None, &request(true), Some("test"))
        .await;
    assert!(first.status().is_success());
    let second_router = router.clone();
    let mut second = tokio::spawn(async move {
        second_router
            .route_completion(None, &request(true), Some("test"))
            .await
    });
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(20), &mut second)
            .await
            .is_err()
    );
    assert_eq!(p.calls.lock().unwrap().len(), 1);
    drop(first);
    let second = tokio::time::timeout(std::time::Duration::from_secs(2), second)
        .await
        .unwrap()
        .unwrap();
    assert!(second.status().is_success());
    axum::body::to_bytes(second.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(p.calls.lock().unwrap().len(), 2);
    let diagnostic = router.get_worker_loads().await;
    let bytes = axum::body::to_bytes(diagnostic.into_body(), usize::MAX)
        .await
        .unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["program_scheduling"]["warm_enabled"], false);
    let programs = value["program_scheduling"]["scheduler"]["ranks"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| r["programs"].as_array().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(programs.len(), 1);
    assert_eq!(programs[0]["in_flight_requests"], 0);
    assert_eq!(programs[0]["waiting_requests"], 0);
}

#[tokio::test]
async fn chat_program_turns_use_the_same_decode_rank() {
    let p = mock(true).await;
    let d = mock(false).await;
    let router = router(&p, &d, true).await;
    let request = serde_json::from_value(
        json!({"model":"test","messages":[{"role":"user","content":"hello"}],
        "max_tokens":8,"vllm_xargs":{"agentic_context":{"program_id":"chat-agent",
        "expected_resume":true,"kv_retention_ttl_seconds":1.0}}}),
    )
    .unwrap();
    for _ in 0..2 {
        let response = router.route_chat(None, &request, Some("test")).await;
        assert!(response.status().is_success());
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
    }
    let calls = d.calls.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].0, calls[1].0);
}
