#![cfg(all(feature = "cpu", feature = "ready"))]

//! Exact-token multi-turn transport regressions. Mocks prove protocol rules, not GPU savings.
use axum::{
    extract::{Json, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc, Mutex,
};
use vllm_router_rs::{
    config::{ConfigValidator, PolicyConfig, RouterConfig, RoutingMode},
    core::Worker,
    policies::{LoadBalancingPolicy, PolicyFactory, RequestHeaders},
    protocols::spec::{CompletionRequest, GenerationRequest},
    routers::{http::vllm_pd_router::VllmPDRouter, RouterTrait},
    server::AppContext,
};
/// What a [`RecordingPrefill`] saw: request text and headers per call.
type RecordedCalls = Arc<Mutex<Vec<(Option<String>, Option<RequestHeaders>)>>>;

#[derive(Clone)]
struct StateData {
    p: bool,
    calls: Arc<Mutex<Vec<Value>>>,
    epoch: Arc<AtomicU64>,
    fail_warm: bool,
    partial: bool,
    fail_local: Arc<AtomicBool>,
    capability_probes: Arc<AtomicU64>,
    reject_status: Arc<AtomicU64>,
    large_text: Arc<AtomicUsize>,
    metrics_kv_milli: Arc<AtomicU64>,
    metrics_running: Arc<AtomicU64>,
    auth_gate: Arc<AtomicBool>,
}
struct Engine {
    url: String,
    state: StateData,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Engine {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn engine(p: bool, fail_warm: bool, partial: bool) -> Engine {
    let state = StateData {
        p,
        calls: Arc::new(Mutex::new(vec![])),
        epoch: Arc::new(AtomicU64::new(1)),
        fail_warm,
        partial,
        fail_local: Arc::new(AtomicBool::new(false)),
        capability_probes: Arc::new(AtomicU64::new(0)),
        reject_status: Arc::new(AtomicU64::new(0)),
        large_text: Arc::new(AtomicUsize::new(0)),
        metrics_kv_milli: Arc::new(AtomicU64::new(0)),
        metrics_running: Arc::new(AtomicU64::new(0)),
        auth_gate: Arc::new(AtomicBool::new(false)),
    };
    let app=Router::new().route("/health",get(||async{"ok"}))
        .route("/get_server_info",get(||async{Json(json!({"model_id":"test","data_parallel_size":1}))}))
        .route("/v1/models",get(||async{Json(json!({"data":[{"id":"test"}]}))}))
        .route("/metrics",get(|State(s):State<StateData>|async move{
            let kv = s.metrics_kv_milli.load(Ordering::SeqCst) as f64 / 1000.0;
            let running = s.metrics_running.load(Ordering::SeqCst);
            format!("vllm:kv_cache_usage_perc{{engine=\"0\"}} {kv}\nvllm:num_requests_running{{engine=\"0\"}} {running}\nvllm:num_requests_waiting{{engine=\"0\"}} 0\n")
        }))
        .route("/pd_capabilities",get(|State(s):State<StateData>,headers:HeaderMap|async move{
            s.capability_probes.fetch_add(1,Ordering::SeqCst);
            if s.auth_gate.load(Ordering::SeqCst)
                && headers.get(header::AUTHORIZATION).and_then(|value| value.to_str().ok()) != Some("Bearer op-key")
            {
                return (StatusCode::UNAUTHORIZED, "unauthorized").into_response();
            }
            Json(json!({"protocol_revision":"vllm-0.28.0-pd-feedback-v1","instance_epoch":s.epoch.load(Ordering::SeqCst).to_string(),"profile":"test-profile","dp_rank":0,"bidirectional":true,"fail_on_load_error":true,"read_margin_millis":5000,"local_miss_guard":true,"local_prefix_probe":true,"store":false})).into_response()
        }))
        .route("/v1/completions",post(generate)).with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    Engine { url, state, task }
}
async fn generate(
    State(s): State<StateData>,
    _headers: HeaderMap,
    Json(r): Json<Value>,
) -> axum::response::Response {
    s.calls.lock().unwrap().push(r.clone());
    if r.pointer("/kv_transfer_params/pd_probe_only") == Some(&json!(true)) {
        return Json(json!({"choices":[],"usage":{"prompt_tokens":r["prompt"].as_array().unwrap().len(),"completion_tokens":0},"kv_transfer_params":{"pd_probe":true,"pd_instance_epoch":s.epoch.load(Ordering::SeqCst).to_string(),"pd_feedback":{"local_cached_tokens":6}}})).into_response();
    }
    let reject = s.reject_status.load(Ordering::SeqCst);
    if reject != 0 {
        let status = StatusCode::from_u16(reject as u16).unwrap();
        let mut response = (
            status,
            Json(json!({"error":{"message":"upstream rejected","code":reject}})),
        )
            .into_response();
        if status == StatusCode::TOO_MANY_REQUESTS {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("7"));
        }
        return response;
    }
    if !s.p
        && s.fail_local.load(Ordering::SeqCst)
        && r.pointer("/kv_transfer_params/pd_local_max_miss_tokens")
            .is_some()
    {
        if r["stream"] == true {
            let data = format!(
                "data: {}\n\ndata: [DONE]\n\n",
                json!({"choices":[],"usage":{"prompt_tokens":r["prompt"].as_array().unwrap().len(),"completion_tokens":0}})
            );
            return ([("content-type", "text/event-stream")], data).into_response();
        }
        return Json(json!({"choices":[],"usage":{"prompt_tokens":r["prompt"].as_array().unwrap().len(),"completion_tokens":0}})).into_response();
    }
    if s.p && s.fail_warm && r.pointer("/kv_transfer_params/remote_request_id").is_some() {
        return (StatusCode::INTERNAL_SERVER_ERROR, "read failed").into_response();
    }
    let prompt = r["prompt"].as_array().unwrap();
    let output = if s.p {
        vec![json!(42)]
    } else {
        vec![json!(42), json!(43), json!(44), json!(45)]
    };
    let kvt = json!({"do_remote_prefill":s.p,"do_remote_decode":!s.p,"remote_engine_id":if s.p{"p-engine"}else{"d-engine"},"remote_request_id":"engine-owned-request","remote_host":"127.0.0.1","remote_port":39510,"remote_block_ids":[[1,2]],"remote_num_tokens":prompt.len()+if s.p{0}else{3},"transfer_mode":"pull","remote_blocks_ttl_remaining_seconds":30.0,"pd_protocol_revision":"vllm-0.28.0-pd-feedback-v1","pd_instance_epoch":s.epoch.load(Ordering::SeqCst).to_string(),"pd_feedback":{"computed_tokens":prompt.len(),"local_cached_tokens":0,"external_cached_tokens":if r.pointer("/kv_transfer_params/remote_request_id").is_some(){3}else{0}}});
    let mut value = json!({"id":"mock","choices":[{"index":0,"text":"done","prompt_token_ids":prompt,"token_ids":output}],"usage":{"prompt_tokens":prompt.len(),"completion_tokens":output.len()},"kv_transfer_params":kvt});
    let large = s.large_text.load(Ordering::SeqCst);
    if large > 0 && !s.p {
        value["choices"][0]["text"] = json!("x".repeat(large));
    }
    if r["stream"] == true && !s.p {
        // Native vLLM explicitly emits null prompt IDs after the first chunk.
        let first =
            json!({"choices":[{"index":0,"text":"d","prompt_token_ids":prompt,"token_ids":[42]}]});
        let later = json!({"choices":[{"index":0,"text":"one","prompt_token_ids":null,"token_ids":[43,44,45]}],"kv_transfer_params":null});
        let final_usage = json!({"choices":[],"usage":value["usage"],"kv_transfer_params":kvt});
        let data = format!(
            "data: {}\n\ndata: {}\n\ndata: {}\n\n{}",
            first,
            later,
            final_usage,
            if s.partial { "" } else { "data: [DONE]\n\n" }
        );
        return ([("content-type", "text/event-stream")], data).into_response();
    }
    Json(value).into_response()
}
async fn router(p: &Engine, d: &Engine) -> VllmPDRouter {
    router_with_local(p, d, false).await
}
async fn router_with_local(p: &Engine, d: &Engine, local: bool) -> VllmPDRouter {
    router_with_warm_remote(p, d, local, json!({})).await
}
async fn router_with_warm_remote(
    p: &Engine,
    d: &Engine,
    local: bool,
    pd_extra: serde_json::Value,
) -> VllmPDRouter {
    router_with_prefills(&[p], d, PolicyConfig::RoundRobin, local, pd_extra).await
}
/// Same router, but with an explicit prefill pool and prefill policy so a test
/// can observe which P each turn actually lands on.
async fn router_with_prefills(
    prefills: &[&Engine],
    d: &Engine,
    policy: PolicyConfig,
    local: bool,
    pd_extra: serde_json::Value,
) -> VllmPDRouter {
    let (cfg, prefill_urls) = warm_router_config(prefills, d, &policy, local, pd_extra);
    let ctx = Arc::new(AppContext::new(cfg, reqwest::Client::new(), 64, None, vec![]).unwrap());
    let table = ctx.policy_registry.kv_affinity_table();
    let select = PolicyFactory::create_from_config(&policy, Some(&table));
    router_with_selection(&ctx, select, prefill_urls, d).await
}

/// Same router, but with a test-supplied prefill policy: the only way to assert
/// what the policy was handed on each path.
async fn router_with_custom_prefill_policy(
    prefills: &[&Engine],
    d: &Engine,
    select: Arc<dyn LoadBalancingPolicy>,
    local: bool,
    pd_extra: serde_json::Value,
) -> VllmPDRouter {
    let (cfg, prefill_urls) =
        warm_router_config(prefills, d, &PolicyConfig::RoundRobin, local, pd_extra);
    let ctx = Arc::new(AppContext::new(cfg, reqwest::Client::new(), 64, None, vec![]).unwrap());
    router_with_selection(&ctx, select, prefill_urls, d).await
}

async fn router_with_selection(
    ctx: &Arc<AppContext>,
    select: Arc<dyn LoadBalancingPolicy>,
    prefill_urls: Vec<(String, Option<u16>)>,
    d: &Engine,
) -> VllmPDRouter {
    let table = ctx.policy_registry.kv_affinity_table();
    ctx.policy_registry.set_prefill_policy(select);
    ctx.policy_registry
        .set_decode_policy(PolicyFactory::create_from_config(
            &PolicyConfig::RoundRobin,
            Some(&table),
        ));
    VllmPDRouter::new(prefill_urls, vec![d.url.clone()], None, ctx)
        .await
        .unwrap()
}

/// Config shared by every router in this file; returns the URLs it configured.
fn warm_router_config(
    prefills: &[&Engine],
    d: &Engine,
    policy: &PolicyConfig,
    local: bool,
    pd_extra: serde_json::Value,
) -> (RouterConfig, Vec<(String, Option<u16>)>) {
    let prefill_urls = prefills
        .iter()
        .map(|p| (p.url.clone(), None))
        .collect::<Vec<_>>();
    let mut cfg = RouterConfig {
        mode: RoutingMode::VllmPrefillDecode {
            prefill_urls: prefill_urls.clone(),
            decode_urls: vec![d.url.clone()],
            prefill_policy: Some(policy.clone()),
            decode_policy: Some(PolicyConfig::RoundRobin),
            discovery_address: None,
        },
        ..Default::default()
    };
    cfg.worker_startup_timeout_secs = 2;
    cfg.disable_circuit_breaker = true;
    let mut pd = json!({"warm_remote": true, "feedback": true, "explicit_ttl": true});
    if let (Some(target), Some(extra)) = (pd.as_object_mut(), pd_extra.as_object()) {
        for (key, value) in extra {
            target.insert(key.clone(), value.clone());
        }
    }
    cfg.program_scheduling=Some(serde_json::from_value(json!({"token_capacity_per_dp_rank":100000,"enable_batch_gain_admission":false,"decode_buffer_tokens":4,"metrics_interval_seconds":0.02,"queue_timeout_seconds":2,"force_resume_timeout_seconds":1,"pd":pd})).unwrap());
    if local {
        let pd = &mut cfg.program_scheduling.as_mut().unwrap().pd;
        pd.local_append = true;
        pd.calibration=Some(serde_json::from_value(json!({
            "revision":"test-r1","profile":"test-profile","metric":"ttft_seconds",
            "local_tpot_slo_seconds":0.1,"min_route_gain_seconds":0.01,"max_prediction_error_seconds":10.0,
            "cells":[
                {"path":"cold_remote","context_min":1,"context_max":100,"miss_min":0,"miss_max":100,"running_min":0,"running_max":0,"ttft_seconds":1.0,"tpot_p95_seconds":0.01,"uncertainty_seconds":0.0},
                {"path":"warm_remote","context_min":1,"context_max":100,"miss_min":0,"miss_max":100,"running_min":0,"running_max":0,"ttft_seconds":0.5,"tpot_p95_seconds":0.01,"uncertainty_seconds":0.0},
                {"path":"local_append","context_min":1,"context_max":100,"miss_min":0,"miss_max":10,"running_min":0,"running_max":0,"ttft_seconds":0.05,"tpot_p95_seconds":0.01,"uncertainty_seconds":0.0}
            ]})).unwrap());
    }
    ConfigValidator::validate(&cfg).unwrap();
    (cfg, prefill_urls)
}
async fn call(router: &VllmPDRouter, prompt: Vec<u32>, stream: bool) -> (StatusCode, Vec<u8>) {
    let (status, _, body) = call_with_headers(router, prompt, stream).await;
    (status, body)
}
async fn call_with_headers(
    router: &VllmPDRouter,
    prompt: Vec<u32>,
    stream: bool,
) -> (StatusCode, HeaderMap, Vec<u8>) {
    let body = completion_body("task", &prompt, true, stream);
    send_completion(router, None, body).await
}

/// The one request builder: token-id prompt, Program identity, and (for
/// `warm_shape`) the exact-token fields the warm/local protocol needs.
fn completion_body(program: &str, prompt: &[u32], warm_shape: bool, stream: bool) -> Value {
    let mut value = json!({"model":"test","prompt":prompt,"max_tokens":4,
        "kv_transfer_params":{"remote_host":"client-untrusted"},
        "vllm_xargs":{"agentic_context":{"program_id":program,"kv_retention_ttl_seconds":1}}});
    if warm_shape {
        value["return_token_ids"] = json!(true);
        value["stream_options"] = json!({"include_usage": true});
    }
    value["stream"] = json!(stream);
    value
}

/// The one sender: status, response headers and the buffered body.
async fn send_completion(
    router: &VllmPDRouter,
    headers: Option<&HeaderMap>,
    body: Value,
) -> (StatusCode, HeaderMap, Vec<u8>) {
    let request = serde_json::from_value(body).unwrap();
    let result = router
        .route_completion(headers, &request, Some("test"))
        .await;
    let status = result.status();
    let response_headers = result.headers().clone();
    (
        status,
        response_headers,
        axum::body::to_bytes(result.into_body(), 16 * 1048576)
            .await
            .unwrap()
            .to_vec(),
    )
}
fn next_prompt() -> Vec<u32> {
    vec![1, 2, 3, 42, 43, 44, 45, 99]
}
#[tokio::test]
async fn exact_two_turns_take_engine_metadata_once_and_reject_rewrite() {
    let p = engine(true, false, false).await;
    let d = engine(false, false, false).await;
    let r = router(&p, &d).await;
    assert_eq!(call(&r, vec![1, 2, 3], false).await.0, StatusCode::OK);
    assert_eq!(call(&r, next_prompt(), false).await.0, StatusCode::OK);
    {
        let calls = p.state.calls.lock().unwrap();
        assert!(calls[0]
            .pointer("/kv_transfer_params/remote_host")
            .is_none());
        assert_eq!(
            calls[1].pointer("/kv_transfer_params/remote_engine_id"),
            Some(&json!("d-engine"))
        );
        // The D advertised prompt+generated coverage; only the prompt prefix
        // is verified, so the P pull must be bounded to the prompt length.
        assert_eq!(
            calls[1].pointer("/kv_transfer_params/remote_num_tokens"),
            Some(&json!(3))
        );
    }
    let mut rewritten = next_prompt();
    rewritten[0] = 9;
    assert_eq!(call(&r, rewritten, false).await.0, StatusCode::OK);
    assert!(p.state.calls.lock().unwrap()[2]
        .pointer("/kv_transfer_params/remote_request_id")
        .is_none());
}
#[tokio::test]
async fn capability_probe_is_cached_across_turns() {
    let p = engine(true, false, false).await;
    let d = engine(false, false, false).await;
    let r = router(&p, &d).await;
    assert_eq!(call(&r, vec![1, 2, 3], false).await.0, StatusCode::OK);
    let probes_after_first = p.state.capability_probes.load(Ordering::SeqCst);
    assert!(probes_after_first > 0);
    assert_eq!(call(&r, next_prompt(), false).await.0, StatusCode::OK);
    // The prefill identity is reused from the router-side cache; the decode
    // engine may still be re-probed as the candidate's source.
    assert_eq!(
        p.state.capability_probes.load(Ordering::SeqCst),
        probes_after_first
    );
}
#[tokio::test]
async fn warm_gate_respects_prefill_running_ceiling() {
    let p = engine(true, false, false).await;
    let d = engine(false, false, false).await;
    p.state.metrics_running.store(3, Ordering::SeqCst);
    let r = router_with_warm_remote(&p, &d, false, json!({"warm_max_p_running": 1})).await;
    // Let the P metrics scrape record the per-target sample.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    assert_eq!(call(&r, vec![1, 2, 3], false).await.0, StatusCode::OK);
    assert_eq!(call(&r, next_prompt(), false).await.0, StatusCode::OK);
    // P is above the running limit, so the successor turn must go cold.
    assert!(p.state.calls.lock().unwrap()[1]
        .pointer("/kv_transfer_params/remote_engine_id")
        .is_none());
}

#[tokio::test]
async fn warm_gate_respects_prefill_kv_floor() {
    let p = engine(true, false, false).await;
    let d = engine(false, false, false).await;
    p.state.metrics_kv_milli.store(900, Ordering::SeqCst);
    let r = router_with_warm_remote(&p, &d, false, json!({"warm_min_p_kv_usage": 0.95})).await;
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    assert_eq!(call(&r, vec![1, 2, 3], false).await.0, StatusCode::OK);
    assert_eq!(call(&r, next_prompt(), false).await.0, StatusCode::OK);
    // P is below the KV floor, so the successor turn must go cold.
    assert!(p.state.calls.lock().unwrap()[1]
        .pointer("/kv_transfer_params/remote_engine_id")
        .is_none());
}

#[tokio::test]
async fn large_non_streaming_completion_still_publishes_warm_candidate() {
    let p = engine(true, false, false).await;
    let d = engine(false, false, false).await;
    // Larger than the metadata bound (256 KiB by default): the response body
    // must be bounded separately from the kv_transfer_params inspection.
    d.state.large_text.store(320 * 1024, Ordering::SeqCst);
    let r = router(&p, &d).await;
    assert_eq!(call(&r, vec![1, 2, 3], false).await.0, StatusCode::OK);
    assert_eq!(call(&r, next_prompt(), false).await.0, StatusCode::OK);
    assert_eq!(
        p.state.calls.lock().unwrap()[1].pointer("/kv_transfer_params/remote_engine_id"),
        Some(&json!("d-engine"))
    );
}

#[tokio::test]
async fn multi_megabyte_completion_still_publishes_warm_candidate() {
    let p = engine(true, false, false).await;
    let d = engine(false, false, false).await;
    // The pre-output gate allows up to max_response_bytes (8 MiB), so the
    // response tracker must inspect the same range: 2 MiB is above its old
    // fixed 1 MiB buffer.
    d.state.large_text.store(2 * 1024 * 1024, Ordering::SeqCst);
    let r = router(&p, &d).await;
    assert_eq!(call(&r, vec![1, 2, 3], false).await.0, StatusCode::OK);
    assert_eq!(call(&r, next_prompt(), false).await.0, StatusCode::OK);
    assert_eq!(
        p.state.calls.lock().unwrap()[1].pointer("/kv_transfer_params/remote_engine_id"),
        Some(&json!("d-engine"))
    );
}

#[tokio::test]
async fn upstream_decode_rejection_is_forwarded_unchanged() {
    let p = engine(true, false, false).await;
    let d = engine(false, false, false).await;
    d.state.reject_status.store(429, Ordering::SeqCst);
    let r = router(&p, &d).await;
    let (status, headers, body) = call_with_headers(&r, vec![1, 2, 3], false).await;
    // The decode target's status, body and Retry-After survive.
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        headers.get(header::RETRY_AFTER),
        Some(&HeaderValue::from_static("7"))
    );
    assert!(String::from_utf8_lossy(&body).contains("upstream rejected"));
}

#[tokio::test]
async fn warm_prefill_client_error_is_forwarded_without_cold_retry() {
    let p = engine(true, false, false).await;
    let d = engine(false, false, false).await;
    let r = router(&p, &d).await;
    assert_eq!(call(&r, vec![1, 2, 3], false).await.0, StatusCode::OK);
    p.state.reject_status.store(400, Ordering::SeqCst);
    let (status, _, body) = call_with_headers(&r, next_prompt(), false).await;
    // A rejected warm prefill is an ordinary upstream error, not a read failure.
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(String::from_utf8_lossy(&body).contains("upstream rejected"));
    assert_eq!(p.state.calls.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn replayed_successor_without_generated_tokens_still_reads_warm() {
    // Trace replay rebuilds the next prompt from history: it starts with the
    // previous prompt but does not contain this turn's live output tokens.
    let p = engine(true, false, false).await;
    let d = engine(false, false, false).await;
    let r = router(&p, &d).await;
    assert_eq!(call(&r, vec![1, 2, 3], false).await.0, StatusCode::OK);
    assert_eq!(call(&r, vec![1, 2, 3, 99], false).await.0, StatusCode::OK);
    let calls = p.state.calls.lock().unwrap();
    assert_eq!(
        calls[1].pointer("/kv_transfer_params/remote_engine_id"),
        Some(&json!("d-engine"))
    );
    assert_eq!(
        calls[1].pointer("/kv_transfer_params/remote_num_tokens"),
        Some(&json!(3))
    );
}
#[tokio::test]
async fn warm_remote_turns_do_not_leak_worker_load() {
    let p = engine(true, false, false).await;
    let d = engine(false, false, false).await;
    let r = router(&p, &d).await;
    assert_eq!(call(&r, vec![1, 2, 3], false).await.0, StatusCode::OK);
    assert_eq!(call(&r, next_prompt(), false).await.0, StatusCode::OK);
    // admit_bound charges the workers; a second attach would leak +1 per turn.
    let prefill = r.worker_registry().get_prefill_workers()[0].load();
    let decode = r.worker_registry().get_decode_workers()[0].load();
    assert_eq!(
        (prefill, decode),
        (0, 0),
        "warm-remote turns leaked worker load (prefill={prefill}, decode={decode})"
    );
}

#[tokio::test]
async fn capability_probe_uses_operator_key_when_configured() {
    struct EnvGuard;
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            std::env::remove_var("OPENAI_API_KEY");
        }
    }
    let _guard = EnvGuard;
    std::env::set_var("OPENAI_API_KEY", "op-key");
    let p = engine(true, false, false).await;
    let d = engine(false, false, false).await;
    // The engine now requires the operator key on the capability endpoint,
    // exactly like real operator-key deployments.
    p.state.auth_gate.store(true, Ordering::SeqCst);
    let r = router(&p, &d).await;
    assert_eq!(call(&r, vec![1, 2, 3], false).await.0, StatusCode::OK);
    assert_eq!(call(&r, next_prompt(), false).await.0, StatusCode::OK);
    assert!(p.state.capability_probes.load(Ordering::SeqCst) > 0);
    assert_eq!(
        p.state.calls.lock().unwrap()[1].pointer("/kv_transfer_params/remote_engine_id"),
        Some(&json!("d-engine")),
        "warm read never engaged although completions authenticate with the operator key"
    );
}

#[tokio::test]
async fn automatic_ttl_without_warm_reads_publishes_no_candidates() {
    let p = engine(true, false, false).await;
    let d = engine(false, false, false).await;
    // Only the automatic TTL is requested: the executor is needed for seat
    // sizing, but no candidate can ever be consumed.
    let r = router_with_warm_remote(
        &p,
        &d,
        false,
        json!({"warm_remote": false, "automatic_ttl": true, "ttl_online": true}),
    )
    .await;
    assert_eq!(call(&r, vec![1, 2, 3], false).await.0, StatusCode::OK);
    assert_eq!(call(&r, next_prompt(), false).await.0, StatusCode::OK);
    let response = r.get_worker_loads().await;
    let bytes = axum::body::to_bytes(response.into_body(), 8 * 1024 * 1024)
        .await
        .unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    let counters = value
        .pointer("/program_scheduling/warm_remote/counters")
        .expect("warm-remote diagnostics");
    assert!(counters.get("candidate_published").is_none());
    assert!(counters.get("candidate_claimed").is_none());
}

#[tokio::test]
async fn warm_read_failure_retries_cold_without_reusing_source() {
    let p = engine(true, true, false).await;
    let d = engine(false, false, false).await;
    let r = router(&p, &d).await;
    assert_eq!(call(&r, vec![1, 2, 3], false).await.0, StatusCode::OK);
    assert_eq!(call(&r, next_prompt(), false).await.0, StatusCode::OK);
    let c = p.state.calls.lock().unwrap();
    assert_eq!(c.len(), 3);
    assert!(c[1]
        .pointer("/kv_transfer_params/remote_request_id")
        .is_some());
    assert!(c[2]
        .pointer("/kv_transfer_params/remote_request_id")
        .is_none());
    // Every prefill attempt, including the warm read and the bounded cold
    // retry, forces min_tokens=1 so handoff metadata can always be produced.
    assert!(c
        .iter()
        .all(|call| call.get("min_tokens") == Some(&json!(1))));
}
#[tokio::test]
async fn restart_at_same_url_invalidates_candidate() {
    let p = engine(true, false, false).await;
    let d = engine(false, false, false).await;
    let r = router(&p, &d).await;
    call(&r, vec![1, 2, 3], false).await;
    d.state.epoch.store(2, Ordering::SeqCst);
    assert_eq!(call(&r, next_prompt(), false).await.0, StatusCode::OK);
    assert!(p.state.calls.lock().unwrap()[1]
        .pointer("/kv_transfer_params/remote_request_id")
        .is_none());
}
#[tokio::test]
async fn completed_sse_publishes_but_partial_sse_never_publishes() {
    for partial in [false, true] {
        let p = engine(true, false, false).await;
        let d = engine(false, false, partial).await;
        let r = router(&p, &d).await;
        let first = call(&r, vec![1, 2, 3], true).await;
        assert_eq!(first.0, StatusCode::OK);
        assert_eq!(call(&r, next_prompt(), true).await.0, StatusCode::OK);
        assert_eq!(
            p.state.calls.lock().unwrap()[1]
                .pointer("/kv_transfer_params/remote_request_id")
                .is_some(),
            !partial
        );
    }
}

#[tokio::test]
async fn calibrated_local_bypasses_p_and_actual_miss_rejection_retries_remote() {
    for (rejected, streaming) in [(false, false), (true, false), (true, true)] {
        let p = engine(true, false, false).await;
        let d = engine(false, false, false).await;
        let r = router_with_local(&p, &d, true).await;
        assert_eq!(call(&r, vec![1, 2, 3], false).await.0, StatusCode::OK);
        d.state.fail_local.store(rejected, Ordering::SeqCst);
        assert_eq!(call(&r, next_prompt(), streaming).await.0, StatusCode::OK);
        assert_eq!(
            p.state.calls.lock().unwrap().len(),
            if rejected { 2 } else { 1 }
        );
        let all = d.state.calls.lock().unwrap();
        let calls = all
            .iter()
            .filter(|r| r.pointer("/kv_transfer_params/pd_probe_only").is_none())
            .collect::<Vec<_>>();
        assert!(calls[1]
            .pointer("/kv_transfer_params/pd_local_max_miss_tokens")
            .is_some());
        if rejected {
            assert!(calls[2]
                .pointer("/kv_transfer_params/pd_local_max_miss_tokens")
                .is_none());
        }
    }
}

/// Issue #499: a turn that falls back to the cold path must consume exactly one
/// prefill selection. Before the fix the warm-remote probe asked the policy
/// first and the cold path asked again, so with `round_robin` and two prefills
/// every actual prefill landed on the second one.
#[tokio::test]
async fn cold_fallback_consumes_one_prefill_selection_per_turn() {
    let p0 = engine(true, false, false).await;
    let p1 = engine(true, false, false).await;
    let d = engine(false, false, false).await;
    let r = router_with_prefills(&[&p0, &p1], &d, PolicyConfig::RoundRobin, false, json!({})).await;
    for _ in 0..2 {
        let body = completion_body("task", &[1, 2, 3], false, false);
        assert_eq!(send_completion(&r, None, body).await.0, StatusCode::OK);
    }
    let first = p0.state.calls.lock().unwrap().len();
    let second = p1.state.calls.lock().unwrap().len();
    assert_eq!(
        (first, second),
        (1, 1),
        "round_robin must alternate across turns, not spend two selections on the cold fallback \
         (p0={first}, p1={second})"
    );
}

/// Prefill calls an engine received for one prompt.
fn prefill_calls(engine: &Engine, prompt: &[u32]) -> Vec<Value> {
    engine
        .state
        .calls
        .lock()
        .unwrap()
        .iter()
        .filter(|call| call.get("prompt") == Some(&json!(prompt)))
        .cloned()
        .collect()
}

/// Issue #499: the warm-remote path used to call `select_worker(_, None)`, so a
/// hash policy received no request inputs and every Program turn collapsed onto
/// the constant fallback key `"request:"`. A Program's warm turn must land on
/// the prefill its cold turn selected for the same request text.
///
/// The hash ring is built from the engine URLs and the mocks bind random ports,
/// so which prefill a given text maps to changes per run: fixed expectations
/// would be flaky. Calibrate against the ring actually built — keep starting
/// Programs with distinct prompts until each prefill has one, then assert that
/// Program's warm turn returns to it. With the old constant key, cold turns used
/// the same key, so calibration never finds a second prefill.
#[tokio::test]
async fn warm_remote_selection_uses_the_request_text() {
    let p0 = engine(true, false, false).await;
    let p1 = engine(true, false, false).await;
    let d = engine(false, false, false).await;
    let r = router_with_prefills(
        &[&p0, &p1],
        &d,
        PolicyConfig::ConsistentHash { virtual_nodes: 160 },
        false,
        json!({}),
    )
    .await;
    let mut affined: [Option<(String, Vec<u32>)>; 2] = [None, None];
    for index in 0..32u32 {
        if affined.iter().all(Option::is_some) {
            break;
        }
        let program = format!("calibrate-{index}");
        let prompt = vec![3000 + index, 42, 43];
        let body = completion_body(&program, &prompt, true, false);
        assert_eq!(send_completion(&r, None, body).await.0, StatusCode::OK);
        let engine = match (
            prefill_calls(&p0, &prompt).is_empty(),
            prefill_calls(&p1, &prompt).is_empty(),
        ) {
            (false, true) => 0,
            (true, false) => 1,
            _ => panic!("one cold turn must reach exactly one prefill"),
        };
        affined[engine].get_or_insert((program, prompt));
    }
    let [Some((program0, prompt0)), Some((program1, prompt1))] = affined else {
        let (first, second) = (
            p0.state.calls.lock().unwrap().len(),
            p1.state.calls.lock().unwrap().len(),
        );
        panic!(
            "could not find a request text for both prefills: every Program turn was pinned to \
             one engine (p0={first}, p1={second})"
        );
    };
    for (engine, program, prompt) in [(0, program0, prompt0), (1, program1, prompt1)] {
        let body = completion_body(&program, &prompt, true, false);
        assert_eq!(send_completion(&r, None, body).await.0, StatusCode::OK);
        let (turns0, turns1) = (prefill_calls(&p0, &prompt), prefill_calls(&p1, &prompt));
        let (affined_turns, other) = if engine == 0 {
            (&turns0, &turns1)
        } else {
            (&turns1, &turns0)
        };
        assert_eq!(
            affined_turns.len(),
            2,
            "the warm turn of {program} left its affined prefill: selection ignored the request text"
        );
        assert!(other.is_empty(), "{program} reached both prefills");
        assert!(
            affined_turns[1]
                .pointer("/kv_transfer_params/remote_engine_id")
                .is_some(),
            "turn 2 of {program} did not take the warm read"
        );
    }
}

/// Records the inputs the prefill policy is handed on every call, and rotates
/// the selection so both prefills stay in play.
#[derive(Debug)]
struct RecordingPrefill {
    needs_text: bool,
    needs_cache_key: bool,
    needs_headers: bool,
    calls: RecordedCalls,
    counter: AtomicUsize,
}

impl RecordingPrefill {
    fn new(needs_text: bool, needs_cache_key: bool, needs_headers: bool) -> Arc<Self> {
        Arc::new(Self {
            needs_text,
            needs_cache_key,
            needs_headers,
            calls: Arc::new(Mutex::new(Vec::new())),
            counter: AtomicUsize::new(0),
        })
    }

    fn calls(&self) -> Vec<(Option<String>, Option<RequestHeaders>)> {
        self.calls.lock().unwrap().clone()
    }
}

impl LoadBalancingPolicy for RecordingPrefill {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        self.calls
            .lock()
            .unwrap()
            .push((request_text.map(str::to_string), headers.cloned()));
        Some(self.counter.fetch_add(1, Ordering::SeqCst) % workers.len())
    }

    fn name(&self) -> &'static str {
        "recording"
    }

    fn needs_request_text(&self) -> bool {
        self.needs_text
    }

    fn needs_cache_routing_key(&self) -> bool {
        self.needs_cache_key
    }

    fn needs_headers(&self) -> bool {
        self.needs_headers
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Issue #499: warm and cold dispatch must feed the policy the same request
/// inputs, and a cold fallback must not ask it a second time. A recording policy
/// checks both directly, without depending on the hash ring.
#[tokio::test]
async fn warm_and_cold_selection_receive_the_same_request_inputs() {
    let p0 = engine(true, false, false).await;
    let p1 = engine(true, false, false).await;
    let d = engine(false, false, false).await;
    let policy = RecordingPrefill::new(true, false, true);
    let r =
        router_with_custom_prefill_policy(&[&p0, &p1], &d, policy.clone(), false, json!({})).await;
    let mut headers = HeaderMap::new();
    headers.insert("x-session-id", HeaderValue::from_static("session-abc"));

    // A shape the exact-token protocol cannot serve: one selection, and the cold
    // path reuses it.
    let body = completion_body("cold-only", &[11, 12, 13], false, false);
    let (status, _, _) = send_completion(&r, Some(&headers), body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        policy.calls().len(),
        1,
        "the cold fallback asked the policy again"
    );

    // A shape it can serve: the cold turn, then the warm turn of one Program.
    for _ in 0..2 {
        let body = completion_body("warm-turn", &[11, 12, 13], true, false);
        let (status, _, _) = send_completion(&r, Some(&headers), body).await;
        assert_eq!(status, StatusCode::OK);
    }
    // One prefill call per turn: the cold-only dispatch, the cold turn of
    // `warm-turn`, and its warm read. The warm turn lands on the other P than
    // its cold turn (the recording policy rotates), which is safe because the
    // candidate's source is D; a warm read that failed and silently retried cold
    // would add a fourth call, and no-warm-at-all would leave three without
    // metadata.
    let mut prefills = Vec::new();
    for engine in [&p0, &p1] {
        prefills.extend(engine.state.calls.lock().unwrap().iter().cloned());
    }
    assert_eq!(
        prefills.len(),
        3,
        "expected cold-only dispatch + cold turn + warm read: {prefills:?}"
    );
    let warm_reads = prefills
        .iter()
        .filter(|call| {
            // A cold dispatch carries the key with a null value, so test the
            // value, not the key.
            call.pointer("/kv_transfer_params/remote_engine_id")
                .and_then(Value::as_str)
                .is_some()
        })
        .count();
    assert_eq!(
        warm_reads, 1,
        "the second turn must be a real warm read on the cross-P prefill"
    );
    let calls = policy.calls();
    assert_eq!(calls.len(), 3, "one selection per turn: {calls:?}");
    let (cold_text, cold_headers) = &calls[1];
    let (warm_text, warm_headers) = &calls[2];
    assert_eq!(
        cold_text, warm_text,
        "warm and cold saw a different request text"
    );
    assert_eq!(
        cold_headers, warm_headers,
        "warm and cold saw different headers"
    );
    assert_eq!(
        cold_headers
            .as_ref()
            .and_then(|headers| headers.get("x-session-id"))
            .map(String::as_str),
        Some("session-abc"),
        "needs_headers() did not carry the request headers"
    );
}

/// `needs_cache_routing_key()` is a different input from the serialized request;
/// the policy must receive the content-preserving key.
#[tokio::test]
async fn selection_uses_the_cache_routing_key_when_requested() {
    let p0 = engine(true, false, false).await;
    let p1 = engine(true, false, false).await;
    let d = engine(false, false, false).await;
    let policy = RecordingPrefill::new(true, true, false);
    let r =
        router_with_custom_prefill_policy(&[&p0, &p1], &d, policy.clone(), false, json!({})).await;
    let body = completion_body("cache-key", &[21, 22, 23], false, false);
    let expected = serde_json::from_value::<CompletionRequest>(body.clone())
        .unwrap()
        .extract_cache_routing_key();
    let (status, _, _) = send_completion(&r, None, body).await;
    assert_eq!(status, StatusCode::OK);
    let calls = policy.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].0.as_deref(),
        Some(expected.as_str()),
        "needs_cache_routing_key() must receive the cache routing key, not the body"
    );
    assert!(
        calls[0].1.is_none(),
        "headers were not requested but were attached"
    );
}
