//! Real HTTP regression coverage for the vLLM P/D request error contract.

use axum::{
    extract::{Request, State},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde_json::{json, Value};
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::{net::TcpListener, sync::Notify, task::JoinHandle};
use vllm_router_rs::{
    config::{KvConnector, PolicyConfig, RouterConfig, RoutingMode},
    routers::{RouterFactory, RouterTrait},
    server::AppContext,
};

const SECRET: &str =
    "Traceback /private/secret.py worker-secret.internal Authorization: secret-marker";

// gzip encoding of `data: [DONE]\n\n`, with a deterministic zero timestamp.
pub const GZIP_SSE: &[u8] = &[
    31, 139, 8, 0, 0, 0, 0, 0, 2, 255, 75, 73, 44, 73, 180, 82, 136, 118, 241, 247, 115, 141, 229,
    226, 2, 0, 28, 237, 200, 175, 14, 0, 0, 0,
];

pub struct Server {
    pub url: String,
    pub task: JoinHandle<()>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn serve(app: Router) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    Server { url, task }
}

#[derive(Default)]
pub struct Controls {
    pub p_profile_stops: AtomicUsize,
    pub d_profile_stops: AtomicUsize,
    pub p_profile_stopped: Notify,
    pub p_body_started: Notify,
    pub p_body_release: Arc<Notify>,
    pub p_seen: Notify,
    pub d_seen: Notify,
    pub p_release: Notify,
    pub d_release: Notify,
    pub p_wait: std::sync::atomic::AtomicBool,
    pub d_wait: std::sync::atomic::AtomicBool,
    pub p_status: AtomicUsize,
    pub p_active: Arc<AtomicUsize>,
    pub d_active: Arc<AtomicUsize>,
    pub stream_release: Arc<Notify>,
    pub d_status: AtomicUsize,
}
struct Active(Arc<AtomicUsize>);
impl Active {
    fn new(count: Arc<AtomicUsize>) -> Self {
        count.fetch_add(1, Ordering::SeqCst);
        Self(count)
    }
}
impl Drop for Active {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}
#[derive(Clone)]
struct Stage {
    prefill: bool,
    pub control: Arc<Controls>,
    calls: Arc<AtomicUsize>,
}
async fn stage(State(stage): State<Stage>, request: Request) -> Response {
    if request.method() == http::Method::GET {
        return Json(json!({})).into_response();
    }
    if request.uri().path() == "/start_profile" {
        return "ok".into_response();
    }
    if request.uri().path() == "/stop_profile" {
        let stops = if stage.prefill {
            &stage.control.p_profile_stops
        } else {
            &stage.control.d_profile_stops
        };
        stops.fetch_add(1, Ordering::SeqCst);
        if stage.prefill {
            stage.control.p_profile_stopped.notify_one();
        }
        return "ok".into_response();
    }
    stage.calls.fetch_add(1, Ordering::SeqCst);
    let bytes = axum::body::to_bytes(request.into_body(), 1 << 20)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    let model = body["model"].as_str().unwrap_or("");
    let control = &stage.control;
    let _active = Active::new(if stage.prefill {
        control.p_active.clone()
    } else {
        control.d_active.clone()
    });
    let (seen, release, wait, status) = if stage.prefill {
        (
            &control.p_seen,
            &control.p_release,
            &control.p_wait,
            &control.p_status,
        )
    } else {
        (
            &control.d_seen,
            &control.d_release,
            &control.d_wait,
            &control.d_status,
        )
    };
    seen.notify_one();
    if wait.load(Ordering::SeqCst) {
        release.notified().await;
    }
    let forced_status = status.load(Ordering::SeqCst);
    let phase = if stage.prefill { "p" } else { "d" };
    if stage.prefill && model == "p-error-body-wait" {
        let first_control = control.clone();
        let first = futures_util::stream::once(async move {
            first_control.p_body_started.notify_one();
            Ok::<_, std::io::Error>(bytes::Bytes::from_static(b"upstream unavailable"))
        });
        let release = control.p_body_release.clone();
        let last = futures_util::stream::once(async move {
            release.notified().await;
            Ok::<_, std::io::Error>(bytes::Bytes::new())
        });
        return (
            http::StatusCode::SERVICE_UNAVAILABLE,
            axum::body::Body::from_stream(futures_util::StreamExt::chain(first, last)),
        )
            .into_response();
    }
    if stage.prefill && matches!(model, "p-body-wait" | "p-body-reset") {
        let first_control = control.clone();
        let first = futures_util::stream::once(async move {
            first_control.p_body_started.notify_one();
            Ok::<_, std::io::Error>(bytes::Bytes::from_static(
                b"{\"kv_transfer_params\":{\"remote_engine_id\":\"P\"}}",
            ))
        });
        let release = control.p_body_release.clone();
        let reset = model == "p-body-reset";
        let last = futures_util::stream::once(async move {
            release.notified().await;
            if reset {
                return Err(std::io::Error::other("controlled P body reset"));
            }
            Ok::<_, std::io::Error>(bytes::Bytes::new())
        });
        return axum::body::Body::from_stream(futures_util::StreamExt::chain(first, last))
            .into_response();
    }
    if let Some(kind) = model.strip_prefix("opaque-kv-") {
        let kv = match kind {
            "null" => Value::Null,
            "array" => json!([1, 2]),
            _ => json!("legacy-value"),
        };
        return Json(if stage.prefill {
            json!({"kv_transfer_params":kv})
        } else {
            json!({"choices":[], "forwarded_kv":body.get("kv_transfer_params"), "forwarded_kv_present":body.get("kv_transfer_params").is_some()})
        })
        .into_response();
    }
    if !stage.prefill && model == "compat-gzip" {
        return (
            [
                ("content-type", "text/event-stream"),
                ("content-encoding", "gzip"),
            ],
            GZIP_SSE,
        )
            .into_response();
    }
    if !stage.prefill && matches!(model, "compat-headers" | "compat-body-wait") {
        let first = futures_util::stream::once(async {
            Ok::<_, std::io::Error>(bytes::Bytes::from_static(b"{\"choices\":[]}"))
        });
        let release = control.stream_release.clone();
        let wait = model == "compat-body-wait";
        let end = futures_util::stream::once(async move {
            if wait {
                release.notified().await;
            }
            Ok::<_, std::io::Error>(bytes::Bytes::new())
        });
        let mut response = (
            [
                ("content-type", "application/json"),
                ("content-encoding", "identity"),
                ("connection", "close"),
                ("set-cookie", "a=1"),
                ("x-backend", "preserved"),
            ],
            axum::body::Body::from_stream(futures_util::StreamExt::chain(first, end)),
        )
            .into_response();
        response
            .headers_mut()
            .append("set-cookie", "b=2".parse().unwrap());
        return response;
    }
    if model == format!("{phase}-error-body-timeout") {
        let first = futures_util::stream::once(async {
            Ok::<_, std::io::Error>(bytes::Bytes::from_static(b"incomplete-body-marker"))
        });
        let stream = futures_util::StreamExt::chain(first, futures_util::stream::pending());
        return (
            http::StatusCode::from_u16(if stage.prefill { 503 } else { 429 }).unwrap(),
            [("retry-after", "5")],
            axum::body::Body::from_stream(stream),
        )
            .into_response();
    }
    if stage.prefill && model == "p-preview" {
        return (
            http::StatusCode::SERVICE_UNAVAILABLE,
            format!("{}preview-tail-marker", "界".repeat(256)),
        )
            .into_response();
    }
    if model == format!("{phase}-timeout") {
        std::future::pending::<()>().await;
    }
    if model == "p-body-timeout" && stage.prefill {
        let chunks = futures_util::stream::once(async {
            Ok::<_, std::io::Error>(bytes::Bytes::from_static(b"{"))
        });
        let stream = futures_util::StreamExt::chain(chunks, futures_util::stream::pending());
        return axum::body::Body::from_stream(stream).into_response();
    }
    if model == "success-logprobs" {
        return Json(if stage.prefill {
            json!({"kv_transfer_params": {"remote_engine_id":"P"}, "prompt_logprobs": [null, {"1": {"logprob": -0.5}}]})
        } else { json!({"choices":[{"message":{"content":"A"}}]}) }).into_response();
    }
    if model == "p-invalid-json" && stage.prefill {
        return SECRET.into_response();
    }
    if model == "p-missing-kv" && stage.prefill {
        return Json(json!({"choices":[]})).into_response();
    }
    if !stage.prefill && model == "stream-reset" {
        use bytes::Bytes;
        let release = control.stream_release.clone();
        let first = futures_util::stream::once(async {
            Ok::<_, std::io::Error>(Bytes::from_static(b"data: partial"))
        });
        let reset = futures_util::stream::once(async move {
            release.notified().await;
            Err::<Bytes, _>(std::io::Error::other("safe-test-reset"))
        });
        let stream = futures_util::StreamExt::chain(first, reset);
        return (
            [("content-type", "text/event-stream")],
            axum::body::Body::from_stream(stream),
        )
            .into_response();
    }
    if let Some(code) = (forced_status != 0)
        .then_some(forced_status as u16)
        .or_else(|| {
            model
                .strip_prefix(&format!("{phase}-"))
                .and_then(|v| v.parse::<u16>().ok())
        })
    {
        return (
            http::StatusCode::from_u16(code).unwrap(),
            [
                ("retry-after", "5"),
                ("content-type", "text/html"),
                ("set-cookie", "private-cookie"),
                ("x-request-id", "worker-secret.internal"),
            ],
            if let Some(size) = model
                .strip_prefix("body-")
                .and_then(|s| s.parse::<usize>().ok())
            {
                // Includes invalid UTF-8, multibyte boundaries and an opaque payload.
                let mut body = vec![0xff; size];
                let len = size.min(SECRET.len());
                body[..len].copy_from_slice(&SECRET.as_bytes()[..len]);
                body
            } else {
                SECRET.as_bytes().to_vec()
            },
        )
            .into_response();
    }
    Json(if stage.prefill {
        json!({"kv_transfer_params": {"remote_engine_id": "P", "remote_host": "127.0.0.1", "remote_port": 9100, "transfer_mode": "push"}})
    } else { json!({"choices":[]}) }).into_response()
}

pub struct Fixture {
    pub _p: Server,
    pub _d: Server,
    pub gateway: Server,
    pub context: Arc<AppContext>,
    pub router: Arc<dyn RouterTrait>,
    pub control: Arc<Controls>,
    pub d_calls: Arc<AtomicUsize>,
}
impl Fixture {
    pub async fn direct(timeout: Duration) -> Self {
        Self::new(timeout, false).await
    }
    pub async fn new(timeout: Duration, discovery: bool) -> Self {
        Self::build(timeout, discovery, false, KvConnector::Nixl).await
    }
    pub async fn profiled(timeout: Duration) -> Self {
        Self::build(timeout, false, true, KvConnector::Nixl).await
    }
    pub async fn profiled_discovery(timeout: Duration) -> Self {
        Self::build(timeout, true, true, KvConnector::Nixl).await
    }
    pub async fn moriio_without_mode(timeout: Duration) -> Self {
        Self::build(timeout, false, false, KvConnector::MoriIO).await
    }
    async fn build(
        timeout: Duration,
        discovery: bool,
        enable_profiling: bool,
        connector: KvConnector,
    ) -> Self {
        let control = Arc::new(Controls::default());
        let d_calls = Arc::new(AtomicUsize::new(0));
        let p = serve(
            Router::new()
                .route("/health", get(|| async { "ok" }))
                .fallback(stage)
                .with_state(Stage {
                    prefill: true,
                    control: control.clone(),
                    calls: Arc::default(),
                }),
        )
        .await;
        let d = serve(
            Router::new()
                .route("/health", get(|| async { "ok" }))
                .fallback(stage)
                .with_state(Stage {
                    prefill: false,
                    control: control.clone(),
                    calls: d_calls.clone(),
                }),
        )
        .await;
        let discovery_address = if discovery {
            let socket = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            Some(socket.local_addr().unwrap().to_string())
        } else {
            None
        };
        let config = RouterConfig {
            enable_profiling,
            mode: RoutingMode::VllmPrefillDecode {
                prefill_urls: if discovery {
                    vec![]
                } else {
                    vec![(p.url.clone(), None)]
                },
                decode_urls: if discovery {
                    vec![]
                } else {
                    vec![d.url.clone()]
                },
                prefill_policy: None,
                decode_policy: None,
                discovery_address: discovery_address.clone(),
            },
            kv_connector: connector,
            policy: PolicyConfig::Random,
            worker_startup_timeout_secs: 5,
            ..Default::default()
        };
        let client = reqwest::Client::builder().timeout(timeout).build().unwrap();
        let context =
            Arc::new(AppContext::new(config.clone(), client.clone(), 64, None, vec![]).unwrap());
        let router: Arc<dyn RouterTrait> =
            Arc::from(RouterFactory::create_router(&context).await.unwrap());
        if let Some(address) = discovery_address {
            let zmq_context = zmq::Context::new();
            let socket = zmq_context.socket(zmq::DEALER).unwrap();
            socket.connect(&format!("tcp://{address}")).unwrap();
            for (phase, url) in [("P", &p.url), ("D", &d.url)] {
                let registration = rmp_serde::to_vec_named(&json!({"type": phase, "http_address": url.trim_start_matches("http://"), "zmq_address": "127.0.0.1:9999"})).unwrap();
                socket.send(registration, 0).unwrap();
            }
            tokio::time::timeout(Duration::from_secs(2), async {
                loop {
                    let response = router
                        .health(http::Request::new(axum::body::Body::empty()))
                        .await;
                    if response.status().is_success() {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .expect("real ZMQ registration must make discovery ready");
        }
        let gateway = serve(crate::common::test_app::create_test_app(
            router.clone(),
            client,
            &config,
        ))
        .await;
        Self {
            _p: p,
            _d: d,
            gateway,
            context,
            router,
            control,
            d_calls,
        }
    }
    pub async fn request(&self, path: &str, model: &str, logprobs: bool) -> reqwest::Response {
        let mut body = json!({"model":model, "messages":[{"role":"user","content":"test"}], "prompt":"test", "input":"test", "max_tokens":16});
        if model.starts_with("stream-") {
            body["stream"] = json!(true);
        }
        if path == "/v1/chat/completions" {
            body["logprobs"] = json!(logprobs);
        } else if path == "/v1/completions" && logprobs {
            body["logprobs"] = json!(1);
            body["echo"] = json!(true);
        }
        reqwest::Client::new()
            .post(format!("{}{path}", self.gateway.url))
            .header("x-request-id", "public-test-id")
            .json(&body)
            .send()
            .await
            .unwrap()
    }
}

pub async fn assert_error(response: reqwest::Response, code: u16, phase: &str) {
    if response.status().as_u16() != code {
        panic!(
            "expected {code}, got {}: {}",
            response.status(),
            response.text().await.unwrap()
        );
    }
    assert_eq!(response.headers()["content-type"], "application/json");
    assert_eq!(response.headers()["x-request-id"], "public-test-id");
    if ![502, 504].contains(&code) {
        assert_eq!(response.headers()["retry-after"], "5");
    }
    assert!(!response.headers().contains_key("set-cookie"));
    let text = response.text().await.unwrap();
    assert!(!text.contains("secret"), "{text}");
    let json: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(json["error"]["code"], code.to_string());
    assert_eq!(
        json["error"]["type"],
        if code == 429 {
            "rate_limit_error"
        } else if (400..500).contains(&code) {
            "invalid_request_error"
        } else {
            "server_error"
        }
    );
    assert!(json["error"]["message"].as_str().unwrap().contains(phase));
}

impl Fixture {
    pub async fn warmup(&self) {
        let response = self.request("/v1/chat/completions", "warmup", false).await;
        assert_eq!(response.status(), 200);
        response.bytes().await.unwrap();
        // Consume warmup notifications, making subsequent barriers unambiguous.
        self.control.p_seen.notified().await;
        self.control.d_seen.notified().await;
    }
    pub fn assert_no_load(&self) {
        for worker in self.context.worker_registry.get_all() {
            assert_eq!(worker.load(), 0, "{}", worker.url());
        }
    }
}
