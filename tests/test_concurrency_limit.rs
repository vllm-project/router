use std::{sync::Arc, time::Duration};

use axum::{
    body::{Body, HttpBody},
    http::{Request, StatusCode},
    middleware,
    response::Response,
    routing::get,
    Router,
};
use bytes::Bytes;
use http_body_util::BodyExt;
use tokio::sync::{mpsc, Notify, Semaphore};
use tokio_stream::wrappers::ReceiverStream;
use tower::ServiceExt;
use vllm_router_rs::{
    config::RouterConfig,
    middleware::{concurrency_limit_middleware, ConcurrencyLimiter},
    routers::RouterFactory,
    server::{AppContext, AppState},
};

type Chunk = Result<Bytes, std::io::Error>;

async fn app(
    queue_size: usize,
    rate: usize,
) -> (
    Router,
    mpsc::UnboundedReceiver<mpsc::Sender<Chunk>>,
    Arc<Notify>,
) {
    let (app, started, entered, _) =
        app_with_queue_processor(queue_size, rate, Duration::from_secs(3), true, None).await;
    (app, started, entered)
}

async fn app_with_queue_processor(
    queue_size: usize,
    rate: usize,
    queue_timeout: Duration,
    start_processor: bool,
    semaphore: Option<Arc<Semaphore>>,
) -> (
    Router,
    mpsc::UnboundedReceiver<mpsc::Sender<Chunk>>,
    Arc<Notify>,
    Option<vllm_router_rs::middleware::QueueProcessor>,
) {
    let config = RouterConfig {
        max_concurrent_requests: 1,
        rate_limit_tokens_per_second: Some(rate),
        ..Default::default()
    };
    let mut context =
        Arc::new(AppContext::new(config, reqwest::Client::new(), 1, Some(rate), vec![]).unwrap());
    if let Some(semaphore) = semaphore {
        Arc::get_mut(&mut context).unwrap().concurrency_semaphore = semaphore;
    }
    let router = Arc::from(
        RouterFactory::create_regular_router(&[], &context)
            .await
            .unwrap(),
    );
    let (limiter, processor) = ConcurrencyLimiter::new(
        context.rate_limiter.clone(),
        context.concurrency_semaphore.clone(),
        queue_size,
        queue_timeout,
    );
    let processor = if start_processor {
        if let Some(processor) = processor {
            tokio::spawn(processor.run());
        }
        None
    } else {
        processor
    };
    let state = Arc::new(AppState {
        router,
        context,
        concurrency_queue_tx: limiter.queue_tx,
        router_manager: None,
    });
    let (started_tx, started_rx) = mpsc::unbounded_channel();
    let entered = Arc::new(Notify::new());
    let pending_entered = entered.clone();
    let app = Router::new()
        .route(
            "/stream",
            get(move || {
                let started_tx = started_tx.clone();
                async move {
                    let (tx, rx) = mpsc::channel::<Chunk>(4);
                    started_tx.send(tx).unwrap();
                    Response::new(Body::from_stream(ReceiverStream::new(rx)))
                }
            }),
        )
        .route("/short", get(|| async { "ok" }))
        .route("/empty", get(|| async { Body::empty() }))
        .route(
            "/pending",
            get(move || {
                let entered = pending_entered.clone();
                async move {
                    entered.notify_one();
                    std::future::pending::<Response>().await
                }
            }),
        )
        .layer(middleware::from_fn_with_state(
            state,
            concurrency_limit_middleware,
        ));
    (app, started_rx, entered, processor)
}

async fn request(app: Router, path: &str) -> Response {
    app.oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
        .await
        .unwrap()
}

async fn wait_for_short_ok(app: Router) {
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if request(app.clone(), "/short").await.status() == StatusCode::OK {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("short request was not accepted after permit release");
}

async fn wait_for_short_ok_client(client: &reqwest::Client, addr: &std::net::SocketAddr) {
    let url = format!("http://{addr}/short");
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if client.get(&url).send().await.unwrap().status() == StatusCode::OK {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("short request was not accepted after permit release");
}

#[tokio::test]
async fn disabled_queue_rejects_after_rate_refill_until_stream_eof() {
    let (app, mut started, _) = app(0, 1).await;
    let first = request(app.clone(), "/stream").await;
    let tx = started.recv().await.unwrap();
    tokio::time::sleep(Duration::from_millis(1100)).await;
    let rejected = request(app.clone(), "/short").await;
    assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_retry_after(&rejected);
    drop(tx);
    first.into_body().collect().await.unwrap();
    assert_eq!(request(app, "/short").await.status(), StatusCode::OK);
}

#[tokio::test]
async fn enabled_queue_waits_after_rate_refill_until_stream_eof() {
    let (app, mut started, _) = app(2, 1).await;
    let first = request(app.clone(), "/stream").await;
    let tx = started.recv().await.unwrap();
    tokio::time::sleep(Duration::from_millis(1100)).await;
    let mut second = tokio::spawn(request(app, "/short"));
    assert!(
        tokio::time::timeout(Duration::from_millis(100), &mut second)
            .await
            .is_err()
    );
    drop(tx);
    first.into_body().collect().await.unwrap();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), second)
            .await
            .unwrap()
            .unwrap()
            .status(),
        StatusCode::OK
    );
}

#[tokio::test]
async fn body_drop_releases_slot_with_and_without_queue() {
    for queue_size in [0, 2] {
        let (app, mut started, _) = app(queue_size, 100).await;
        let first = request(app.clone(), "/stream").await;
        let _tx = started.recv().await.unwrap();
        drop(first);
        wait_for_short_ok(app).await;
    }
}

#[tokio::test]
async fn body_error_releases_slot_before_body_drop() {
    let (app, mut started, _) = app(0, 100).await;
    let first = request(app.clone(), "/stream").await;
    let tx = started.recv().await.unwrap();
    tx.send(Err(std::io::Error::other("upstream failed")))
        .await
        .unwrap();
    let mut body = first.into_body();
    assert!(body.frame().await.unwrap().is_err());
    wait_for_short_ok(app).await;
    drop(body);
}

#[tokio::test]
async fn cancelling_handler_before_headers_releases_slot() {
    let (app, _, entered) = app(0, 100).await;
    let first = tokio::spawn(request(app.clone(), "/pending"));
    entered.notified().await;
    first.abort();
    assert!(first.await.unwrap_err().is_cancelled());
    wait_for_short_ok(app).await;
}

#[tokio::test]
async fn cancelled_queued_request_does_not_leak_slot() {
    let (app, mut started, _) = app(2, 100).await;
    let first = request(app.clone(), "/stream").await;
    let _tx = started.recv().await.unwrap();
    let mut queued = tokio::spawn(request(app.clone(), "/short"));
    assert!(tokio::time::timeout(Duration::from_millis(50), &mut queued)
        .await
        .is_err());
    queued.abort();
    assert!(queued.await.unwrap_err().is_cancelled());
    drop(first);
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), request(app, "/short"))
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
}

#[tokio::test]
async fn completed_request_does_not_restore_rate_token() {
    let (app, _, _) = app(0, 1).await;
    request(app.clone(), "/short")
        .await
        .into_body()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        request(app.clone(), "/short").await.status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    tokio::time::sleep(Duration::from_millis(1100)).await;
    assert_eq!(request(app, "/short").await.status(), StatusCode::OK);
}

#[tokio::test]
async fn final_data_frame_releases_slot_before_body_drop() {
    let (app, _, _) = app(0, 100).await;
    let mut body = request(app.clone(), "/short").await.into_body();
    assert_eq!(body.size_hint().exact(), Some(2));
    assert!(body.frame().await.unwrap().unwrap().is_data());
    wait_for_short_ok(app).await;
    drop(body);
}

#[tokio::test]
async fn empty_body_keeps_end_stream_hint_and_releases_immediately() {
    let (app, _, _) = app(0, 100).await;
    let body = request(app.clone(), "/empty").await.into_body();
    assert!(body.is_end_stream());
    assert_eq!(body.size_hint().exact(), Some(0));
    wait_for_short_ok(app).await;
    drop(body);
}

#[tokio::test]
async fn http_client_disconnect_releases_stream_slot() {
    for queue_size in [0, 2] {
        let (app, mut started, _) = app(queue_size, 100).await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = reqwest::Client::new();
        let streaming = tokio::spawn(client.get(format!("http://{addr}/stream")).send());
        let tx = tokio::time::timeout(Duration::from_secs(1), started.recv())
            .await
            .unwrap()
            .unwrap();
        tx.send(Ok(Bytes::from_static(b"first chunk\n")))
            .await
            .unwrap();
        let mut response = tokio::time::timeout(Duration::from_secs(1), streaming)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(response.chunk().await.unwrap().is_some());
        drop(response);
        tokio::time::timeout(Duration::from_secs(1), tx.closed())
            .await
            .expect("disconnect must drop the server response body");
        wait_for_short_ok_client(&client, &addr).await;
        server.abort();
        assert!(server.await.unwrap_err().is_cancelled());
    }
}

#[tokio::test]
async fn queued_rate_wait_uses_queue_deadline() {
    let (app, _, _) = app(2, 1).await;
    request(app.clone(), "/short")
        .await
        .into_body()
        .collect()
        .await
        .unwrap();
    let response = tokio::time::timeout(Duration::from_secs(2), request(app, "/short"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn queued_request_times_out_while_stream_remains_active() {
    let (app, mut started, _) = app(2, 100).await;
    let first = request(app.clone(), "/stream").await;
    let _tx = started.recv().await.unwrap();
    let response = tokio::time::timeout(Duration::from_secs(4), request(app.clone(), "/short"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
    drop(first);
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), request(app, "/short"))
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
}

fn assert_retry_after(response: &Response) {
    let seconds = response.headers()["retry-after"]
        .to_str()
        .unwrap()
        .parse::<u64>()
        .unwrap();
    assert!(seconds >= 1);
}

#[tokio::test]
async fn full_queue_returns_retry_after_without_starting_request() {
    let (app, mut started, _, _processor) =
        app_with_queue_processor(1, 100, Duration::from_secs(3), false, None).await;
    let first = request(app.clone(), "/stream").await;
    let _tx = started.recv().await.unwrap();
    let mut queued = tokio::spawn(request(app.clone(), "/short"));
    assert!(tokio::time::timeout(Duration::from_millis(50), &mut queued)
        .await
        .is_err());
    let rejected = request(app, "/short").await;
    assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_retry_after(&rejected);
    queued.abort();
    assert!(queued.await.unwrap_err().is_cancelled());
    drop(first);
}

#[tokio::test]
async fn queued_request_times_out_at_short_timeout() {
    let (app, mut started, _, _) =
        app_with_queue_processor(1, 100, Duration::from_millis(75), true, None).await;
    let first = request(app.clone(), "/stream").await;
    let _tx = started.recv().await.unwrap();
    let response = tokio::time::timeout(Duration::from_secs(1), request(app.clone(), "/short"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
    drop(first);
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), request(app, "/short"))
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
}

#[tokio::test]
async fn closed_semaphore_returns_503_with_or_without_queue() {
    for queue_size in [0, 2] {
        let semaphore = Arc::new(Semaphore::new(0));
        semaphore.close();
        let (app, _, _, _) = app_with_queue_processor(
            queue_size,
            100,
            Duration::from_secs(3),
            true,
            Some(semaphore),
        )
        .await;
        let response = request(app, "/short").await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}

#[tokio::test]
async fn exhausted_rate_bucket_returns_retry_after() {
    let (app, _, _) = app(0, 1).await;
    request(app.clone(), "/short")
        .await
        .into_body()
        .collect()
        .await
        .unwrap();
    let rejected = request(app, "/short").await;
    assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_retry_after(&rejected);
}
