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
    routers::{http::vllm_pd_router::VllmPDRouter, RouterTrait},
    server::AppContext,
};
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
    };
    let app=Router::new().route("/health",get(||async{"ok"}))
        .route("/get_server_info",get(||async{Json(json!({"model_id":"test","data_parallel_size":1}))}))
        .route("/v1/models",get(||async{Json(json!({"data":[{"id":"test"}]}))}))
        .route("/metrics",get(|State(s):State<StateData>|async move{
            let kv = s.metrics_kv_milli.load(Ordering::SeqCst) as f64 / 1000.0;
            let running = s.metrics_running.load(Ordering::SeqCst);
            format!("vllm:kv_cache_usage_perc{{engine=\"0\"}} {kv}\nvllm:num_requests_running{{engine=\"0\"}} {running}\nvllm:num_requests_waiting{{engine=\"0\"}} 0\n")
        }))
        .route("/pd_capabilities",get(|State(s):State<StateData>|async move{s.capability_probes.fetch_add(1,Ordering::SeqCst);Json(json!({"protocol_revision":"vllm-0.28.0-pd-feedback-v1","instance_epoch":s.epoch.load(Ordering::SeqCst).to_string(),"profile":"test-profile","dp_rank":0,"bidirectional":true,"fail_on_load_error":true,"read_margin_millis":5000,"local_miss_guard":true,"local_prefix_probe":true,"store":false}))}))
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
    let mut cfg = RouterConfig {
        mode: RoutingMode::VllmPrefillDecode {
            prefill_urls: vec![(p.url.clone(), None)],
            decode_urls: vec![d.url.clone()],
            prefill_policy: Some(PolicyConfig::RoundRobin),
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
    let ctx = Arc::new(AppContext::new(cfg, reqwest::Client::new(), 64, None, vec![]).unwrap());
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
async fn call(router: &VllmPDRouter, prompt: Vec<u32>, stream: bool) -> (StatusCode, Vec<u8>) {
    let (status, _, body) = call_with_headers(router, prompt, stream).await;
    (status, body)
}
async fn call_with_headers(
    router: &VllmPDRouter,
    prompt: Vec<u32>,
    stream: bool,
) -> (StatusCode, HeaderMap, Vec<u8>) {
    let r=serde_json::from_value(json!({"model":"test","prompt":prompt,"max_tokens":4,"return_token_ids":true,"stream":stream,"stream_options":{"include_usage":true},"kv_transfer_params":{"remote_host":"client-untrusted"},"vllm_xargs":{"agentic_context":{"program_id":"task","kv_retention_ttl_seconds":1}}})).unwrap();
    let result = router.route_completion(None, &r, Some("test")).await;
    let status = result.status();
    let headers = result.headers().clone();
    (
        status,
        headers,
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
