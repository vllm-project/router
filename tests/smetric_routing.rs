mod common;

use axum::{
    body::{to_bytes, Body},
    extract::{Json, State},
    http::{Request, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router as AxumRouter,
};
use common::{create_test_context, test_app::create_test_app};
use futures_util::Stream;
use serde_json::{json, Value};
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};
use tokio::net::TcpListener;
use tower::ServiceExt;
use vllm_router_rs::{
    config::{PolicyConfig, RouterConfig, RoutingMode},
    routers::http::router::Router,
    server::AppContext,
};

struct HeldStream(Option<bytes::Bytes>);

impl Stream for HeldStream {
    type Item = Result<bytes::Bytes, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.0.take() {
            Some(bytes) => Poll::Ready(Some(Ok(bytes))),
            None => Poll::Pending,
        }
    }
}

struct Backend {
    url: String,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for Backend {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn backend_reply(State(id): State<String>, Json(body): Json<Value>) -> Response {
    if body.get("hold").and_then(Value::as_bool) == Some(true) {
        let first = if body.get("silent").and_then(Value::as_bool) == Some(true) {
            None
        } else {
            Some(bytes::Bytes::from_static(
                b"data: {\"choices\":[{\"text\":\"token\"}]}\n\n",
            ))
        };
        return Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .body(Body::from_stream(HeldStream(first)))
            .unwrap();
    }
    Json(json!({
        "id": id,
        "object": "text_completion",
        "created": 1,
        "model": "test-model",
        "choices": [{"index": 0, "text": "done", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
    }))
    .into_response()
}

async fn backend(id: &str) -> Backend {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = AxumRouter::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/completions", post(backend_reply))
        .route("/v1/messages", post(backend_reply))
        .with_state(id.to_owned());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Backend {
        url: format!("http://{addr}"),
        task,
    }
}

fn smetric() -> PolicyConfig {
    PolicyConfig::SMetric {
        config: serde_yaml::from_str(include_str!("../examples/configs/smetric.yaml")).unwrap(),
    }
}

async fn setup(policy: PolicyConfig) -> (AxumRouter, Arc<AppContext>, Vec<Backend>) {
    let workers = vec![backend("worker-0").await, backend("worker-1").await];
    let urls = workers.iter().map(|w| w.url.clone()).collect::<Vec<_>>();
    let config = RouterConfig {
        policy,
        mode: RoutingMode::Regular {
            worker_urls: urls.clone(),
        },
        disable_retries: true,
        disable_circuit_breaker: true,
        worker_startup_timeout_secs: 5,
        worker_startup_check_interval_secs: 1,
        ..RouterConfig::default()
    };
    let ctx = create_test_context(config.clone());
    let router = Router::new(urls, &ctx).await.unwrap();
    let app = create_test_app(Arc::new(router), reqwest::Client::new(), &config);
    (app, ctx, workers)
}

fn request(route: &str, body: Value, turn: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(route)
        .header("content-type", "application/json")
        .header("x-session-turn", turn)
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn reply_json(response: Response) -> Value {
    assert_eq!(response.status(), StatusCode::OK);
    serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap()
}

async fn wait_until_idle(ctx: &AppContext) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while ctx.worker_registry.get_all().iter().any(|w| w.load() != 0) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn smetric_continuation_reuses_the_warmed_worker() {
    let (app, ctx, _workers) = setup(smetric()).await;
    let prefix = "shared conversation history ".repeat(20);
    let first = app
        .clone()
        .oneshot(request(
            "/v1/completions",
            json!({"model":"test-model", "prompt":prefix, "max_tokens":1}),
            "1",
        ))
        .await
        .unwrap();
    let first = reply_json(first).await;
    let second = app
        .oneshot(request(
            "/v1/completions",
            json!({"model":"test-model", "prompt":format!("{prefix}next turn"), "max_tokens":1}),
            "2",
        ))
        .await
        .unwrap();
    assert_eq!(reply_json(second).await["id"], first["id"]);
    wait_until_idle(&ctx).await;
}

#[tokio::test]
async fn smetric_unsupported_input_uses_the_idle_worker() {
    let (app, ctx, workers) = setup(smetric()).await;
    let held = app
        .clone()
        .oneshot(request(
            "/v1/completions",
            json!({"model":"test-model", "prompt":"hold this request", "stream":true, "hold":true}),
            "1",
        ))
        .await
        .unwrap();
    assert_eq!(held.status(), StatusCode::OK);
    let busy = workers
        .iter()
        .position(|w| ctx.worker_registry.get_by_url(&w.url).unwrap().load() == 1)
        .unwrap();
    let reply = app
        .oneshot(request(
            "/v1/completions",
            json!({"model":"test-model", "prompt":[1,2,3], "max_tokens":1}),
            "2",
        ))
        .await
        .unwrap();
    assert_eq!(
        reply_json(reply).await["id"],
        format!("worker-{}", 1 - busy)
    );
    drop(held);
    wait_until_idle(&ctx).await;
}

#[tokio::test]
async fn smetric_disconnect_before_output_releases_load() {
    let (app, ctx, _workers) = setup(smetric()).await;
    let response = tokio::time::timeout(
        Duration::from_secs(5),
        app.oneshot(request(
            "/v1/completions",
            json!({"model":"test-model", "prompt":"pending prefill", "stream":true, "hold":true, "silent":true}),
            "1",
        )),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        ctx.worker_registry
            .get_all()
            .iter()
            .map(|w| w.load())
            .sum::<usize>(),
        1
    );
    drop(response);
    wait_until_idle(&ctx).await;
}

#[tokio::test]
async fn transparent_stream_load_tracking_is_policy_opt_in() {
    for (policy, expected_load) in [
        (
            PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.5,
                eviction_interval_secs: 120,
                max_tree_size: 67108864,
            },
            1,
        ),
        (PolicyConfig::RoundRobin, 0),
        (smetric(), 1),
    ] {
        let (app, ctx, _workers) = setup(policy).await;
        let response = app
            .oneshot(request(
                "/v1/messages",
                json!({"model":"test-model", "messages":[], "stream":true, "hold":true}),
                "1",
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            ctx.worker_registry
                .get_all()
                .iter()
                .map(|w| w.load())
                .sum::<usize>(),
            expected_load
        );
        drop(response);
        wait_until_idle(&ctx).await;
    }
}
