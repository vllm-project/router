#![cfg(all(feature = "cpu", feature = "ready"))]

use std::{sync::Arc, time::Duration};

use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{header::CONTENT_TYPE, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post as axum_post},
    Json, Router,
};
use serde_json::{json, Value};
use tokio::{
    net::TcpListener,
    sync::{mpsc, oneshot},
    task::JoinHandle,
};
use tokio_stream::wrappers::ReceiverStream;
use vllm_router_rs::{
    config::{PolicyConfig, RouterConfig, RoutingMode},
    metrics::{start_prometheus, PrometheusConfig},
    middleware::QueuedRequest,
    routers::{RouterFactory, RouterTrait},
    server::{build_app, AppContext, AppState},
};

const COMPLETIONS: &str = "/v1/completions";
const CHAT: &str = "/v1/chat/completions";
const RESPONSES: &str = "/v1/responses";

struct Server {
    url: String,
    task: JoinHandle<()>,
}

impl Server {
    async fn start(app: Router) -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self { url, task }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}

struct Call {
    request: Value,
    respond: oneshot::Sender<Response>,
}

struct Worker {
    server: Server,
    calls: mpsc::Receiver<Call>,
}

impl Worker {
    async fn start() -> Self {
        let (tx, calls) = mpsc::channel(8);
        let app = Router::new()
            .route("/health", get(|| async { StatusCode::OK }))
            .route(
                "/v1/models",
                get(|| async {
                    Json(json!({"object":"list","data":[{"id":"test-model","object":"model"}]}))
                }),
            )
            .route(COMPLETIONS, axum_post(mock))
            .route(CHAT, axum_post(mock))
            .route(RESPONSES, axum_post(mock))
            .route("/v1/completions/render", axum_post(mock))
            .with_state(tx);
        Self {
            server: Server::start(app).await,
            calls,
        }
    }

    async fn call(&mut self) -> Call {
        self.calls.recv().await.unwrap()
    }
}

async fn mock(State(tx): State<mpsc::Sender<Call>>, Json(request): Json<Value>) -> Response {
    let (respond, rx) = oneshot::channel();
    tx.send(Call { request, respond }).await.unwrap();
    rx.await
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

async fn router(config: RouterConfig, queue: Option<mpsc::Sender<QueuedRequest>>) -> Server {
    let context = Arc::new(
        AppContext::new(
            config.clone(),
            reqwest::Client::builder()
                .timeout(Duration::from_secs(20))
                .build()
                .unwrap(),
            config.max_concurrent_requests,
            Some(1),
            vec![],
        )
        .unwrap(),
    );
    let router: Arc<dyn RouterTrait> =
        Arc::from(RouterFactory::create_router(&context).await.unwrap());
    Server::start(build_app(
        Arc::new(AppState {
            router,
            context,
            concurrency_queue_tx: queue,
            router_manager: None,
        }),
        1024 * 1024,
        vec![],
        vec![],
        false,
    ))
    .await
}

fn config(mode: RoutingMode) -> RouterConfig {
    RouterConfig {
        mode,
        policy: PolicyConfig::RoundRobin,
        worker_startup_timeout_secs: 5,
        retry: vllm_router_rs::config::RetryConfig {
            max_retries: 2,
            initial_backoff_ms: 1,
            jitter_factor: 0.,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn post(server: &Server, path: &str, body: Value) -> JoinHandle<reqwest::Response> {
    let url = format!("{}{path}", server.url);
    tokio::spawn(async move {
        reqwest::Client::new()
            .post(url)
            .json(&body)
            .send()
            .await
            .unwrap()
    })
}

type SseTx = mpsc::Sender<Result<Bytes, std::io::Error>>;

fn stream_reply(call: Call) -> SseTx {
    let (tx, rx) = mpsc::channel(8);
    call.respond
        .send(
            (
                [(CONTENT_TYPE, "text/event-stream")],
                Body::from_stream(ReceiverStream::new(rx)),
            )
                .into_response(),
        )
        .unwrap();
    tx
}

async fn send(tx: &SseTx, bytes: &'static str) {
    tx.send(Ok(Bytes::from_static(bytes.as_bytes())))
        .await
        .unwrap();
}

async fn read_until(response: &mut reqwest::Response, until: &str) {
    let mut content = Vec::new();
    while !content.ends_with(until.as_bytes()) {
        content.extend(response.chunk().await.unwrap().expect("SSE data"));
    }
}

fn sample(scrape: &str, suffix: &str, labels: &[(&str, &str)]) -> f64 {
    let metric = format!("vllm_router_{suffix}");
    scrape
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(' ')?;
            if key.split('{').next() != Some(metric.as_str())
                || !labels
                    .iter()
                    .all(|(label, value)| key.contains(&format!("{label}=\"{value}\"")))
            {
                return None;
            }
            Some(value.parse::<f64>().unwrap())
        })
        .sum()
}

struct Metrics(String);

impl Metrics {
    fn start() -> Self {
        let port = portpicker::pick_unused_port().unwrap();
        start_prometheus(PrometheusConfig {
            port,
            host: "127.0.0.1".into(),
        });
        Self(format!("http://127.0.0.1:{port}/metrics"))
    }

    async fn scrape(&self) -> String {
        reqwest::get(&self.0)
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .text()
            .await
            .unwrap()
    }

    async fn wait(&self, cond: impl Fn(&str) -> bool) {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if cond(&self.scrape().await) {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
}

async fn ordinary(metrics: &Metrics) {
    let mut worker = Worker::start().await;
    let router = router(
        config(RoutingMode::Regular {
            worker_urls: vec![worker.server.url.clone()],
        }),
        None,
    )
    .await;

    let request = post(
        &router,
        COMPLETIONS,
        json!({"model":"user-generated-label","prompt":"hi","stream":true,"stream_options":{"include_usage":true}}),
    );
    let call = worker.call().await;
    assert_eq!(call.request["stream"], true);
    let tx = stream_reply(call);
    let mut response = request.await.unwrap();

    send(&tx, "data: {\"choices\":[{\"index\":0,\"text\":\"f").await;
    read_until(&mut response, "\"f").await;
    let scrape = metrics.scrape().await;
    assert_eq!(sample(&scrape, "active_requests", &[]), 1.);
    assert!(!scrape.contains("model=\"user-generated-label\""));
    assert_eq!(sample(&scrape, "ttft_seconds_count", &[]), 0.);

    send(&tx, "irst\",\"token_ids\":[42]}]}\r\n\r\n").await;
    read_until(&mut response, "}]}\r\n\r\n").await;
    let scrape = metrics.scrape().await;
    let route = ("route", COMPLETIONS);
    assert_eq!(
        sample(&scrape, "ttft_seconds_count", &[route, ("stream", "true")]),
        1.
    );
    assert_eq!(
        sample(&scrape, "ttft_seconds_bucket", &[route, ("le", "120")]),
        1.
    );

    send(
        &tx,
        "data: {\"choices\":[{\"index\":0,\"text\":\"two\",\"token_ids\":[43]}]}\n\n",
    )
    .await;
    read_until(&mut response, "\n\n").await;
    let scrape = metrics.scrape().await;
    assert_eq!(sample(&scrape, "itl_seconds_count", &[route]), 1.);
    assert_eq!(
        sample(&scrape, "itl_seconds_bucket", &[route, ("le", "5")]),
        1.
    );

    send(
        &tx,
        concat!(
            "data: {\"choices\":[{\"index\":0,\"text\":\"a\",\"token_ids\":[44]}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"text\":\"b\",\"token_ids\":[45]}]}\n\n",
        ),
    )
    .await;
    read_until(&mut response, "[45]}]}\n\n").await;
    send(&tx, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":17,\"completion_tokens\":4}}\n\ndata: [DONE]\n\n").await;
    drop(tx);
    response.bytes().await.unwrap();
    let scrape = metrics.scrape().await;
    assert_eq!(sample(&scrape, "itl_seconds_count", &[route]), 1.);
    assert_eq!(sample(&scrape, "active_requests", &[]), 0.);
    assert_eq!(sample(&scrape, "input_tokens_count", &[route]), 1.);
    assert_eq!(
        sample(&scrape, "input_tokens_bucket", &[route, ("le", "16")]),
        0.
    );
    assert_eq!(
        sample(&scrape, "input_tokens_bucket", &[route, ("le", "32")]),
        1.
    );
    assert_eq!(
        sample(&scrape, "output_tokens_bucket", &[route, ("le", "4")]),
        1.
    );

    let request = post(
        &router,
        COMPLETIONS,
        json!({"model":"test-model","prompt":"hi","stream":true}),
    );
    let tx = stream_reply(worker.call().await);
    // A success-looking payload, even a terminal marker, is not enough until EOF.
    send(&tx, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":99,\"completion_tokens\":9}}\n\ndata: [DONE]\n\n").await;
    let mut response = request.await.unwrap();
    read_until(&mut response, "[DONE]\n\n").await;
    send(
        &tx,
        "event: error\ndata: {\"error\":{\"message\":\"failed\"}}\n\n",
    )
    .await;
    drop(tx);
    response.bytes().await.unwrap();
    let scrape = metrics.scrape().await;
    assert_eq!(sample(&scrape, "input_tokens_count", &[route]), 1.);
    assert_eq!(sample(&scrape, "requests_cancelled_total", &[]), 0.);

    let request = post(
        &router,
        COMPLETIONS,
        json!({"model":"test-model","prompt":"hi"}),
    );
    worker
        .call()
        .await
        .respond
        .send(StatusCode::SERVICE_UNAVAILABLE.into_response())
        .unwrap();
    let call = worker.call().await;
    let scrape = metrics.scrape().await;
    assert_eq!(sample(&scrape, "active_requests", &[]), 1.);
    call.respond.send(Json(json!({"choices":[{"text":"ok"}],"usage":{"prompt_tokens":5,"completion_tokens":1}})).into_response()).unwrap();
    let response = request.await.unwrap();
    assert!(response.status().is_success());
    response.bytes().await.unwrap();
    let scrape = metrics.scrape().await;
    assert_eq!(sample(&scrape, "active_requests", &[]), 0.);
    assert_eq!(sample(&scrape, "ttft_seconds_count", &[route]), 1.);
    assert_eq!(sample(&scrape, "input_tokens_count", &[route]), 2.);
    assert_eq!(sample(&scrape, "output_tokens_count", &[route]), 2.);

    let request = post(
        &router,
        COMPLETIONS,
        json!({"model":"test-model","prompt":"hi","stream":true}),
    );
    let _tx = stream_reply(worker.call().await);
    drop(request.await.unwrap());
    metrics
        .wait(|scrape| sample(scrape, "active_requests", &[]) == 0.)
        .await;
    let scrape = metrics.scrape().await;
    assert_eq!(
        sample(
            &scrape,
            "requests_cancelled_total",
            &[("stage", "streaming")]
        ),
        1.
    );
    assert_eq!(sample(&scrape, "requests_cancelled_total", &[]), 1.);
}

async fn expect_cancel(metrics: &Metrics, stage: &str, stage_total: f64, all: f64) {
    metrics
        .wait(|scrape| {
            sample(scrape, "active_requests", &[]) == 0.
                && sample(scrape, "requests_cancelled_total", &[("stage", stage)]) == stage_total
        })
        .await;
    let scrape = metrics.scrape().await;
    assert_eq!(sample(&scrape, "requests_cancelled_total", &[]), all);
    assert_eq!(sample(&scrape, "stage_requests", &[]), 0.);
}

async fn queue_and_routing_cancellation(metrics: &Metrics) {
    let mut worker = Worker::start().await;
    let mode = RoutingMode::Regular {
        worker_urls: vec![worker.server.url.clone()],
    };
    let (queue, mut pending) = mpsc::channel(1);
    let queued_router = router(
        RouterConfig {
            max_concurrent_requests: 0,
            ..config(mode.clone())
        },
        Some(queue),
    )
    .await;

    let request = post(
        &queued_router,
        COMPLETIONS,
        json!({"model":"test-model","prompt":"queued"}),
    );
    let _queued = pending.recv().await.unwrap();
    request.abort();
    let _ = request.await;
    expect_cancel(metrics, "queued", 1., 2.).await;

    // The real HTTP cache-aware prepare RPC holds routing open until its
    // response. It is not yet a generation request to a selected worker.
    let routing_router = router(
        RouterConfig {
            policy: PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 10,
                balance_rel_threshold: 2.,
                eviction_interval_secs: 120,
                max_tree_size: 1000,
            },
            ..config(mode)
        },
        None,
    )
    .await;
    let request = post(
        &routing_router,
        COMPLETIONS,
        json!({"model":"test-model","prompt":"routing"}),
    );
    let _render = worker.call().await;
    let scrape = metrics.scrape().await;
    assert_eq!(sample(&scrape, "stage_requests", &[("stage", "route")]), 1.);
    request.abort();
    let _ = request.await;
    expect_cancel(metrics, "routing", 1., 3.).await;
}

fn completion_reply() -> Response {
    Json(json!({"choices":[{"text":"ok"}],"usage":{"prompt_tokens":6,"completion_tokens":2}}))
        .into_response()
}

fn prefill_reply() -> Response {
    Json(json!({
        "choices":[{"text":"p"}],
        "kv_transfer_params":{
            "transfer_mode":"push","do_remote_prefill":true,
            "remote_engine_id":"mock-P","remote_host":"127.0.0.1","remote_port":1234,"tp_size":1
        }
    }))
    .into_response()
}

async fn pd(metrics: &Metrics) {
    let mut prefill = Worker::start().await;
    let mut decode = Worker::start().await;
    let router = router(
        config(RoutingMode::VllmPrefillDecode {
            prefill_urls: vec![(prefill.server.url.clone(), None)],
            decode_urls: vec![decode.server.url.clone()],
            prefill_policy: None,
            decode_policy: None,
            discovery_address: None,
        }),
        None,
    )
    .await;
    let request_body = || json!({"model":"test-model","prompt":"hi","stream":false});

    let request = post(&router, COMPLETIONS, request_body());
    let p = prefill.call().await;
    assert!(p.request.get("kv_transfer_params").is_some());
    let scrape = metrics.scrape().await;
    assert_eq!(
        sample(&scrape, "stage_requests", &[("stage", "prefill_dispatch")]),
        1.
    );
    assert_eq!(sample(&scrape, "stage_requests", &[("stage", "route")]), 0.);
    assert_eq!(
        sample(&scrape, "stage_requests", &[("stage", "decode_dispatch")]),
        0.
    );
    assert!(decode.calls.try_recv().is_err());
    p.respond.send(prefill_reply()).unwrap();
    let d = decode.call().await;
    assert!(d.request["kv_transfer_params"].is_object());
    d.respond.send(completion_reply()).unwrap();
    let response = request.await.unwrap().error_for_status().unwrap();
    response.bytes().await.unwrap();
    let scrape = metrics.scrape().await;
    assert_eq!(sample(&scrape, "pd_handoff_total", &[("result", "ok")]), 1.);
    assert_eq!(
        sample(
            &scrape,
            "stage_duration_seconds_count",
            &[("stage", "prefill_dispatch")]
        ),
        1.
    );
    assert_eq!(
        sample(
            &scrape,
            "stage_duration_seconds_bucket",
            &[("stage", "prefill_dispatch"), ("le", "240")]
        ),
        1.
    );
    assert_eq!(sample(&scrape, "stage_requests", &[]), 0.);

    // The preceding P response learned a NIXL push address. Both workers
    // now receive the same logical request before either replies.
    let mut streaming = request_body();
    streaming["stream"] = json!(true);
    let request = post(&router, COMPLETIONS, streaming);
    let (p, d) = tokio::join!(prefill.call(), decode.call());
    let scrape = metrics.scrape().await;
    assert_eq!(sample(&scrape, "active_requests", &[]), 1.);
    assert_eq!(
        sample(&scrape, "stage_requests", &[("stage", "prefill_dispatch")]),
        1.
    );
    assert_eq!(
        sample(&scrape, "stage_requests", &[("stage", "decode_dispatch")]),
        1.
    );
    assert_eq!(sample(&scrape, "stage_requests", &[("stage", "route")]), 0.);
    p.respond.send(prefill_reply()).unwrap();
    let tx = stream_reply(d);
    let mut response = request.await.unwrap().error_for_status().unwrap();
    // The P/D usage rewriter yields one frame per SSE event even if both
    // arrived together. That rewrite must not invent a token interval.
    send(
        &tx,
        concat!(
            "data: {\"choices\":[{\"index\":0,\"text\":\"co\",\"token_ids\":[8]}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"text\":\"alesced\",\"token_ids\":[9]}]}\n\n",
        ),
    )
    .await;
    read_until(&mut response, "[9]}]}\n\n").await;
    let route = ("route", COMPLETIONS);
    assert_eq!(
        sample(&metrics.scrape().await, "itl_seconds_count", &[route]),
        1.
    );
    for (token, frame) in [
        (
            "[10]}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"text\":\" next\",\"token_ids\":[10]}]}\n\n",
        ),
        (
            "[11]}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"text\":\" single\",\"token_ids\":[11]}]}\n\n",
        ),
    ] {
        send(&tx, frame).await;
        read_until(&mut response, token).await;
    }
    send(&tx, "data: [DONE]\n\n").await;
    drop(tx);
    response.bytes().await.unwrap();
    let scrape = metrics.scrape().await;
    assert_eq!(sample(&scrape, "itl_seconds_count", &[route]), 2.);
    assert_eq!(sample(&scrape, "pd_handoff_total", &[("result", "ok")]), 2.);
    assert_eq!(
        sample(&scrape, "ttft_seconds_count", &[("route", COMPLETIONS)]),
        2.
    );
    assert_eq!(
        sample(
            &scrape,
            "stage_duration_seconds_count",
            &[("stage", "prefill_dispatch")]
        ),
        2.
    );
    assert_eq!(
        sample(
            &scrape,
            "stage_duration_seconds_count",
            &[("stage", "decode_dispatch")]
        ),
        2.
    );
    assert_eq!(sample(&scrape, "stage_requests", &[]), 0.);

    let request = post(&router, COMPLETIONS, request_body());
    let (_p, _d) = tokio::join!(prefill.call(), decode.call());
    request.abort();
    let _ = request.await;
    expect_cancel(metrics, "prefill", 1., 4.).await;

    let request = post(&router, COMPLETIONS, request_body());
    let (p, _d) = tokio::join!(prefill.call(), decode.call());
    p.respond.send(prefill_reply()).unwrap();
    metrics
        .wait(|scrape| {
            sample(scrape, "stage_requests", &[("stage", "prefill_dispatch")]) == 0.
                && sample(scrape, "stage_requests", &[("stage", "decode_dispatch")]) == 1.
        })
        .await;
    request.abort();
    let _ = request.await;
    expect_cancel(metrics, "decode", 1., 5.).await;

    for (status, result) in [
        (StatusCode::BAD_REQUEST, "rejected"),
        (StatusCode::GATEWAY_TIMEOUT, "timeout"),
        (StatusCode::INTERNAL_SERVER_ERROR, "failed"),
    ] {
        let request = post(&router, COMPLETIONS, request_body());
        let (p, d) = tokio::join!(prefill.call(), decode.call());
        p.respond.send(prefill_reply()).unwrap();
        d.respond.send(status.into_response()).unwrap();
        request.await.unwrap().bytes().await.unwrap();
        let scrape = metrics.scrape().await;
        assert_eq!(
            sample(&scrape, "pd_handoff_total", &[("result", result)]),
            1.
        );
        assert_eq!(sample(&scrape, "pd_handoff_total", &[("result", "ok")]), 2.);
        assert_eq!(sample(&scrape, "requests_cancelled_total", &[]), 5.);
        assert_eq!(sample(&scrape, "active_requests", &[]), 0.);
    }
}

async fn chat_and_responses(metrics: &Metrics) {
    let mut worker = Worker::start().await;
    let router = router(
        config(RoutingMode::Regular {
            worker_urls: vec![worker.server.url.clone()],
        }),
        None,
    )
    .await;

    let request = post(
        &router,
        CHAT,
        json!({
            "model":"test-model", "messages":[{"role":"user","content":"hello"}], "stream":true,
            "stream_options":{"include_usage":true},
        }),
    );
    let tx = stream_reply(worker.call().await);
    let mut response = request.await.unwrap().error_for_status().unwrap();
    send(&tx, concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"function\":{\"name\":\"weather\",\"arguments\":\"\"}}]}}]}\n\n",
    )).await;
    read_until(&mut response, "\"arguments\":\"\"}}]}}]}\n\n").await;
    let route = ("route", CHAT);
    assert_eq!(
        sample(&metrics.scrape().await, "ttft_seconds_count", &[route]),
        0.
    );
    send(&tx, "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"Think\"},\"token_ids\":[7]}]}\n\n").await;
    read_until(&mut response, "[7]}]}\n\n").await;
    send(&tx, "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\"}}]},\"token_ids\":[8]}]}\n\n").await;
    read_until(&mut response, "[8]}]}\n\n").await;
    send(&tx, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":17,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n").await;
    drop(tx);
    response.bytes().await.unwrap();
    let scrape = metrics.scrape().await;
    assert_eq!(
        sample(&scrape, "ttft_seconds_count", &[route, ("stream", "true")]),
        1.
    );
    assert_eq!(sample(&scrape, "itl_seconds_count", &[route]), 1.);
    assert_eq!(
        sample(&scrape, "input_tokens_bucket", &[route, ("le", "16")]),
        0.
    );
    assert_eq!(
        sample(&scrape, "input_tokens_bucket", &[route, ("le", "32")]),
        1.
    );
    assert_eq!(
        sample(&scrape, "output_tokens_bucket", &[route, ("le", "4")]),
        1.
    );

    let request = post(
        &router,
        CHAT,
        json!({
            "model":"test-model", "messages":[{"role":"user","content":"echo me"}],
            "stream":true, "echo":true,
        }),
    );
    let tx = stream_reply(worker.call().await);
    send(&tx, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"echo me\"},\"token_ids\":[9]}]}\n\ndata: [DONE]\n\n").await;
    drop(tx);
    request
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .bytes()
        .await
        .unwrap();
    let scrape = metrics.scrape().await;
    assert_eq!(sample(&scrape, "ttft_seconds_count", &[route]), 1.);
    assert_eq!(sample(&scrape, "input_tokens_count", &[route]), 1.);

    // A worker can send SSE with Content-Length instead of chunked
    // framing. Hyper must still let the Router observe the terminal poll.
    let request = post(
        &router,
        CHAT,
        json!({
            "model":"test-model", "messages":[{"role":"user","content":"hello"}], "stream":true,
            "stream_options":{"include_usage":true},
        }),
    );
    worker
        .call()
        .await
        .respond
        .send(
            (
                [(CONTENT_TYPE, "text/event-stream")],
                concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1}}\n\n",
            "data: [DONE]\n\n",
        ),
            )
                .into_response(),
        )
        .unwrap();
    request
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .bytes()
        .await
        .unwrap();
    let scrape = metrics.scrape().await;
    assert_eq!(sample(&scrape, "ttft_seconds_count", &[route]), 2.);
    assert_eq!(sample(&scrape, "input_tokens_count", &[route]), 2.);
    assert_eq!(sample(&scrape, "output_tokens_count", &[route]), 2.);

    let responses_request = || json!({"model":"test-model","input":"hello","stream":true});
    let request = post(&router, RESPONSES, responses_request());
    let tx = stream_reply(worker.call().await);
    let mut response = request.await.unwrap().error_for_status().unwrap();
    send(&tx, "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\"}}\n\n").await;
    read_until(&mut response, "\"message\"}}\n\n").await;
    let route = ("route", RESPONSES);
    assert_eq!(
        sample(&metrics.scrape().await, "ttft_seconds_count", &[route]),
        0.
    );
    send(&tx, concat!(
        "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n",
        "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":8,\"output_tokens\":1}}}\n\n",
    )).await;
    drop(tx);
    response.bytes().await.unwrap();
    let scrape = metrics.scrape().await;
    assert_eq!(
        sample(&scrape, "ttft_seconds_count", &[route, ("stream", "true")]),
        1.
    );
    assert_eq!(sample(&scrape, "itl_seconds_count", &[route]), 0.);
    assert_eq!(sample(&scrape, "input_tokens_count", &[route]), 1.);
    assert_eq!(
        sample(&scrape, "output_tokens_bucket", &[route, ("le", "1")]),
        1.
    );

    // An incomplete upstream HTTP body after a success-looking terminal
    // usage frame must not count as successful usage or client cancellation.
    let request = post(&router, RESPONSES, responses_request());
    let tx = stream_reply(worker.call().await);
    let mut response = request.await.unwrap().error_for_status().unwrap();
    send(&tx, "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":98,\"output_tokens\":99}}}\n\n").await;
    read_until(&mut response, "99}}}\n\n").await;
    tx.send(Err(std::io::Error::other(
        "worker dropped its response body",
    )))
    .await
    .unwrap();
    drop(tx);
    assert!(response.bytes().await.is_err());
    let scrape = metrics.scrape().await;
    assert_eq!(sample(&scrape, "input_tokens_count", &[route]), 1.);
    assert_eq!(sample(&scrape, "output_tokens_count", &[route]), 1.);
    assert_eq!(sample(&scrape, "requests_cancelled_total", &[]), 5.);
    assert_eq!(sample(&scrape, "active_requests", &[]), 0.);
}

#[tokio::test(flavor = "multi_thread")]
async fn scrape_http_inference_metrics() {
    tokio::time::timeout(Duration::from_secs(60), async {
        let metrics = Metrics::start();
        ordinary(&metrics).await;
        queue_and_routing_cancellation(&metrics).await;
        pd(&metrics).await;
        chat_and_responses(&metrics).await;
    })
    .await
    .unwrap();
}
