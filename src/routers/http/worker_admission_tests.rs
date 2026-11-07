//! Lifecycle, overload and fault-injection tests for per-worker admission
//! (RFC #435) on the regular HTTP router.
//!
//! Every test runs against in-process mock workers that count the requests
//! they are actually serving. Completion is driven by a semaphore "gate", so
//! the tests are deterministic: nothing depends on sleeps racing real work.

use super::*;
use axum::body::Bytes;
use axum::extract::State;
use futures_util::future::join_all;
use std::sync::atomic::AtomicUsize;
use tokio::sync::Semaphore;

const WAIT: Duration = Duration::from_secs(10);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Behavior {
    /// Send headers and a first SSE frame, then wait for the gate.
    Hold,
    /// Fail every request with a 500.
    Fail,
    /// Never answer (until the router gives up and drops the connection).
    Hang,
}

#[derive(Clone)]
struct MockState {
    behavior: Behavior,
    in_flight: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
    received: Arc<AtomicUsize>,
    gate: Arc<Semaphore>,
}

/// Counts one request as being served by the worker until it is dropped:
/// on response end, failure, or a dropped connection.
struct Serving(Arc<AtomicUsize>);

impl Serving {
    fn enter(state: &MockState) -> Self {
        state.received.fetch_add(1, Ordering::SeqCst);
        let now = state.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        state.peak.fetch_max(now, Ordering::SeqCst);
        Self(state.in_flight.clone())
    }
}

impl Drop for Serving {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

async fn mock_handler(State(state): State<MockState>, body: Bytes) -> Response {
    let serving = Serving::enter(&state);
    match state.behavior {
        Behavior::Fail => return (StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response(),
        Behavior::Hang => {
            let _serving = serving;
            std::future::pending::<()>().await;
            unreachable!()
        }
        Behavior::Hold => {}
    }
    let stream = serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .and_then(|value| value.get("stream").and_then(serde_json::Value::as_bool))
        .unwrap_or(false);
    if !stream {
        state.gate.acquire().await.unwrap().forget();
        drop(serving);
        return axum::Json(serde_json::json!({"choices": []})).into_response();
    }
    let gate = state.gate.clone();
    let frames = futures_util::stream::unfold(
        (0u8, Some(serving), gate),
        |(step, serving, gate)| async move {
            let frame: &'static [u8] = match step {
                0 => b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
                1 => {
                    gate.acquire().await.unwrap().forget();
                    b"data: [DONE]\n\n"
                }
                _ => return None,
            };
            Some((
                Ok::<_, std::convert::Infallible>(Bytes::from_static(frame)),
                (step + 1, serving, gate),
            ))
        },
    );
    Response::builder()
        .header(CONTENT_TYPE, "text/event-stream")
        .body(Body::from_stream(frames))
        .unwrap()
}

struct MockWorker {
    url: String,
    state: MockState,
    server: tokio::task::JoinHandle<()>,
}

impl MockWorker {
    async fn spawn(behavior: Behavior) -> Self {
        let state = MockState {
            behavior,
            in_flight: Arc::default(),
            peak: Arc::default(),
            received: Arc::default(),
            gate: Arc::new(Semaphore::new(0)),
        };
        let app = axum::Router::new()
            .fallback(mock_handler)
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self { url, state, server }
    }

    fn release(&self, requests: usize) {
        self.state.gate.add_permits(requests);
    }

    fn peak(&self) -> usize {
        self.state.peak.load(Ordering::SeqCst)
    }

    fn received(&self) -> usize {
        self.state.received.load(Ordering::SeqCst)
    }
}

impl Drop for MockWorker {
    fn drop(&mut self) {
        self.server.abort();
    }
}

fn admission_router(workers: &[&MockWorker], limit: Option<usize>) -> Router {
    let worker_registry = Arc::new(WorkerRegistry::new());
    for worker in workers {
        worker_registry.register(Arc::new(BasicWorker::new(
            worker.url.clone(),
            WorkerType::Regular,
        )));
    }
    let (_, rx) = tokio::sync::watch::channel(HashMap::new());
    Router {
        epd: None,
        worker_registry,
        policy_registry: Arc::new(PolicyRegistry::new(
            crate::config::types::PolicyConfig::RoundRobin,
        )),
        worker_startup_timeout_secs: 5,
        worker_startup_check_interval_secs: 1,
        intra_node_data_parallel_size: 1,
        api_key: None,
        client: Client::new(),
        retry_config: RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 1,
            max_backoff_ms: 5,
            backoff_multiplier: 2.0,
            jitter_factor: 0.0,
        },
        circuit_breaker_config: CircuitBreakerConfig::default(),
        health_config: HealthConfig::default(),
        frontend: crate::backend::EngineFrontend::new(),
        _worker_loads: Arc::new(rx),
        _load_monitor_handle: None,
        kv_index_supervisor: None,
        _kv_signal_handle: None,
        kv_aware_session_affinity_ttl_secs: None,
        program_scheduler: None,
        _program_observation_handle: None,
        program_targets_cache: Mutex::new(HashMap::new()),
        program_token_estimator: Arc::new(MomentumTokenEstimator::default()),
        admission: Arc::new(WorkerAdmission::new(limit)),
    }
}

fn chat(stream: bool) -> ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
    }))
    .unwrap()
}

/// Fails instead of hanging when a request that should be answered at once
/// (a rejection, or headers of an admitted stream) instead waits on the gate.
async fn send(router: &Router, stream: bool) -> Response {
    tokio::time::timeout(
        Duration::from_secs(3),
        router.route_typed_request(None, &chat(stream), "/v1/chat/completions", None),
    )
    .await
    .expect("request was neither admitted nor rejected promptly")
}

async fn drain(response: Response) {
    tokio::time::timeout(WAIT, to_bytes(response.into_body(), usize::MAX))
        .await
        .expect("body finished")
        .expect("body readable");
}

async fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + WAIT;
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
}

fn assert_capacity_rejection(response: &Response) {
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()[axum::http::header::RETRY_AFTER], "1");
}

/// Sustained burst: 24 concurrent streams against 2 workers x 2 slots. The
/// workers never see more than their capacity, the excess is answered without
/// ever reaching a worker, and the pool fully recovers once the overload drains.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn burst_is_bounded_per_worker_and_excess_never_reaches_workers() {
    let (a, b) = (
        MockWorker::spawn(Behavior::Hold).await,
        MockWorker::spawn(Behavior::Hold).await,
    );
    let router = admission_router(&[&a, &b], Some(2));

    let responses = join_all((0..24).map(|_| send(&router, true))).await;
    let (admitted, rejected): (Vec<_>, Vec<_>) = responses
        .into_iter()
        .partition(|response| response.status() == StatusCode::OK);
    assert_eq!(admitted.len(), 4, "exactly capacity x workers are admitted");
    assert_eq!(rejected.len(), 20);
    rejected.iter().for_each(assert_capacity_rejection);

    for worker in [&a, &b] {
        assert_eq!(worker.peak(), 2, "worker saw more than its capacity");
        assert_eq!(worker.received(), 2, "a rejected request reached a worker");
        assert_eq!(router.admission.in_flight(&worker.url), 2);
    }

    // Recovery: finish the admitted streams; every permit comes back.
    a.release(2);
    b.release(2);
    join_all(admitted.into_iter().map(drain)).await;
    for worker in [&a, &b] {
        assert_eq!(router.admission.in_flight(&worker.url), 0, "leaked permit");
    }
    let again = join_all((0..4).map(|_| send(&router, true))).await;
    assert!(again
        .iter()
        .all(|response| response.status() == StatusCode::OK));
    a.release(2);
    b.release(2);
    join_all(again.into_iter().map(drain)).await;
    assert_eq!(router.admission.in_flight(&a.url), 0);
    assert_eq!(router.admission.in_flight(&b.url), 0);
}

/// A streaming permit is held for the whole body, not only until headers,
/// and a client that disconnects mid-stream returns it immediately.
#[tokio::test]
async fn streaming_permit_spans_the_body_and_is_released_on_client_disconnect() {
    let worker = MockWorker::spawn(Behavior::Hold).await;
    let router = admission_router(&[&worker], Some(1));

    let first = send(&router, true).await;
    assert_eq!(first.status(), StatusCode::OK);
    // Headers have arrived, but the stream is still open: still occupied.
    assert_capacity_rejection(&send(&router, true).await);
    assert_eq!(router.admission.in_flight(&worker.url), 1);

    // Slow consumer: reading only the first frame keeps the permit held.
    let mut body = first.into_body().into_data_stream();
    let frame = tokio::time::timeout(WAIT, body.next()).await.unwrap();
    assert!(frame.unwrap().is_ok());
    assert_capacity_rejection(&send(&router, true).await);

    // Client disconnect: dropping the body releases the permit at once.
    drop(body);
    assert_eq!(router.admission.in_flight(&worker.url), 0);
    let next = send(&router, true).await;
    assert_eq!(next.status(), StatusCode::OK);
    worker.release(1);
    drain(next).await;
    assert_eq!(router.admission.in_flight(&worker.url), 0);
}

#[tokio::test]
async fn non_streaming_permit_is_held_until_the_response_is_complete() {
    let worker = MockWorker::spawn(Behavior::Hold).await;
    let router = Arc::new(admission_router(&[&worker], Some(1)));

    let pending = tokio::spawn({
        let router = router.clone();
        async move { send(&router, false).await }
    });
    wait_until("the worker to receive the request", || {
        worker.received() == 1
    })
    .await;
    assert_capacity_rejection(&send(&router, false).await);

    worker.release(1);
    let response = tokio::time::timeout(WAIT, pending).await.unwrap().unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    drain(response).await;
    assert_eq!(router.admission.in_flight(&worker.url), 0);
    assert_eq!(worker.received(), 1, "the rejected request was dispatched");
}

/// Timeout / abandonment before the worker answers: the caller drops the
/// in-progress request future. The permit must not leak.
#[tokio::test]
async fn abandoned_dispatch_before_headers_releases_the_permit() {
    let worker = MockWorker::spawn(Behavior::Hang).await;
    let router = admission_router(&[&worker], Some(1));

    let abandoned = tokio::time::timeout(Duration::from_millis(100), send(&router, true)).await;
    assert!(abandoned.is_err(), "the hanging worker must not answer");
    assert_eq!(router.admission.in_flight(&worker.url), 0, "leaked permit");
    // The worker is admissible again rather than permanently locked out.
    assert!(router.admission.try_acquire(&worker.url).unwrap().is_some());
}

/// Worker failure: failed attempts release their permits, the retry moves to
/// a healthy worker, and failing workers do not keep capacity.
#[tokio::test]
async fn failed_attempts_release_permits_and_retries_reach_healthy_workers() {
    let failing = MockWorker::spawn(Behavior::Fail).await;
    let healthy = MockWorker::spawn(Behavior::Hold).await;
    let router = admission_router(&[&failing, &healthy], Some(1));

    let first = send(&router, true).await;
    assert_eq!(first.status(), StatusCode::OK);
    assert_eq!(router.admission.in_flight(&failing.url), 0);
    assert_eq!(router.admission.in_flight(&healthy.url), 1);
    healthy.release(1);
    drain(first).await;

    // Retry abandonment: with only failing capacity left, every attempt
    // fails and the request gives up; nothing stays reserved.
    let only_failing = admission_router(&[&failing], Some(1));
    let response = send(&only_failing, false).await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(only_failing.admission.in_flight(&failing.url), 0);
    assert_eq!(router.admission.in_flight(&healthy.url), 0);
}

/// Router-side overload rejections are not worker failures: a flood of them
/// must not open the circuit breaker of a healthy, busy worker.
#[tokio::test]
async fn capacity_rejections_do_not_trip_the_circuit_breaker() {
    let worker = MockWorker::spawn(Behavior::Hold).await;
    let router = admission_router(&[&worker], Some(1));
    let held = send(&router, true).await;
    assert_eq!(held.status(), StatusCode::OK);

    for _ in 0..50 {
        assert_capacity_rejection(&send(&router, true).await);
    }
    let registered = router.worker_registry.get_by_url(&worker.url).unwrap();
    assert!(registered.is_available(), "busy worker was marked failed");
    assert_eq!(worker.received(), 1);

    worker.release(1);
    drain(held).await;
}

/// The full-pool rejection is answered at once instead of being retried
/// locally: no hidden waiting inside the router.
#[tokio::test]
async fn full_pool_rejection_is_not_retried() {
    let worker = MockWorker::spawn(Behavior::Hold).await;
    let mut router = admission_router(&[&worker], Some(1));
    router.retry_config = RetryConfig {
        max_retries: 5,
        initial_backoff_ms: 2_000,
        max_backoff_ms: 2_000,
        backoff_multiplier: 1.0,
        jitter_factor: 0.0,
    };
    let held = send(&router, true).await;
    let started = Instant::now();
    assert_capacity_rejection(&send(&router, true).await);
    assert!(
        started.elapsed() < Duration::from_millis(1_500),
        "a retry backoff ran: {:?}",
        started.elapsed()
    );
    worker.release(1);
    drain(held).await;
}

/// Disabled (the default): nothing is limited or counted.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn disabled_admission_does_not_change_dispatch() {
    let worker = MockWorker::spawn(Behavior::Hold).await;
    let router = admission_router(&[&worker], None);

    let responses = join_all((0..8).map(|_| send(&router, true))).await;
    assert!(responses
        .iter()
        .all(|response| response.status() == StatusCode::OK));
    assert_eq!(worker.peak(), 8);
    assert_eq!(
        router.admission.in_flight(&worker.url),
        0,
        "disabled path counted"
    );
    worker.release(8);
    join_all(responses.into_iter().map(drain)).await;
}

/// The transparent proxy path applies the same capacity to POST traffic.
#[tokio::test]
async fn transparent_post_is_capped_and_released() {
    let worker = MockWorker::spawn(Behavior::Hold).await;
    let router = admission_router(&[&worker], Some(1));
    let body = || serde_json::json!({"model": "m", "stream": true});

    let first = router
        .route_transparent(None, "/v1/audio/anything", &Method::POST, body())
        .await;
    assert_eq!(first.status(), StatusCode::OK);
    assert_capacity_rejection(
        &router
            .route_transparent(None, "/v1/audio/anything", &Method::POST, body())
            .await,
    );
    assert_eq!(worker.received(), 1);
    drop(first);
    assert_eq!(router.admission.in_flight(&worker.url), 0);
}

/// Removing a busy worker and re-adding it under the same URL never locks it
/// out, and in-flight requests on the removed instance still release cleanly.
#[tokio::test]
async fn removed_and_readded_worker_is_not_locked_out() {
    let worker = MockWorker::spawn(Behavior::Hold).await;
    let router = admission_router(&[&worker], Some(1));
    let held = send(&router, true).await;
    assert_eq!(held.status(), StatusCode::OK);

    router.remove_worker(&worker.url);
    router.worker_registry.register(Arc::new(BasicWorker::new(
        worker.url.clone(),
        WorkerType::Regular,
    )));
    let after = send(&router, true).await;
    assert_eq!(after.status(), StatusCode::OK, "re-added worker locked out");

    worker.release(2);
    drain(held).await;
    drain(after).await;
    assert_eq!(router.admission.in_flight(&worker.url), 0);
}
