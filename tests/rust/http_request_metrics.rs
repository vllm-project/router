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

#[tokio::test(flavor = "multi_thread")]
async fn scrape_http_inference_metrics() {
    tokio::time::timeout(Duration::from_secs(60), async {
        let metrics = Metrics::start();
        ordinary(&metrics).await;
    })
    .await
    .unwrap();
}
