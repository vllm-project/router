//! End-to-end tests for the observational one-token inference probe.
//!
//! The probe exists to make visible the failure described in issue #158: a
//! worker whose process is up — and which the health checker therefore keeps in
//! rotation — but which cannot actually complete a generation.
//!
//! The invariant these tests protect is that the probe *reports only*. It must
//! never mark a worker unhealthy, never mark one healthy, and never touch load
//! accounting, so that a defect in the probe cannot withdraw a good worker from
//! routing.

mod common;

use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig};
use std::sync::Arc;
use vllm_router_rs::config::InferenceProbeConfig;
use vllm_router_rs::core::inference_probe::probe_inference;
use vllm_router_rs::core::{BasicWorker, Worker, WorkerRegistry, WorkerType};

/// A worker whose `/health` answers 200 while every inference request fails.
///
/// This is exactly the shape of #158: the liveness signal says healthy, the
/// inference path says otherwise.
#[tokio::test]
async fn probe_reports_worker_that_is_healthy_but_cannot_serve() {
    let mut mock = MockWorker::new(MockWorkerConfig {
        health_status: HealthStatus::Healthy,
        fail_rate: 1.0,
        ..Default::default()
    });
    let url = mock.start().await.expect("mock worker starts");

    let worker = BasicWorker::new(url, WorkerType::Regular);

    // The health checker's own probe is satisfied: `/health` is fine. This is
    // why the node stays in rotation and keeps receiving traffic.
    assert!(worker.check_health_async().await.is_ok());
    assert!(
        worker.is_healthy(),
        "the liveness probe sees a healthy worker"
    );

    // The inference probe disagrees, which is the whole point.
    let outcome = probe_inference(&worker, 5, None).await;
    assert!(
        !outcome.ok,
        "a worker that cannot complete a generation must be reported as failing"
    );
    assert_eq!(outcome.reason, "http_error");

    // ...and reports only. Nothing about routing may change as a result.
    assert!(
        worker.is_healthy(),
        "the inference probe must not mark the worker unhealthy"
    );
    assert_eq!(worker.load(), 0, "the probe must not touch load accounting");

    mock.stop().await;
}

/// A serving worker must be reported as healthy, but a passing probe must not
/// flip a worker's health either.
#[tokio::test]
async fn probe_reports_a_serving_worker_without_changing_health() {
    let mut mock = MockWorker::new(MockWorkerConfig {
        health_status: HealthStatus::Healthy,
        fail_rate: 0.0,
        ..Default::default()
    });
    let url = mock.start().await.expect("mock worker starts");

    let worker = BasicWorker::new(url, WorkerType::Regular);
    worker.set_healthy(false);

    let outcome = probe_inference(&worker, 5, None).await;

    assert!(outcome.ok, "unexpected failure: {}", outcome.detail);
    assert_eq!(outcome.reason, "ok");
    assert!(
        !worker.is_healthy(),
        "a passing probe reports; it must not mark the worker healthy"
    );

    mock.stop().await;
}

/// Driving the periodic sweep over a registry must leave every worker's health
/// exactly as the health checker left it.
#[tokio::test]
async fn periodic_sweep_never_mutates_worker_health() {
    let mut mock = MockWorker::new(MockWorkerConfig {
        health_status: HealthStatus::Healthy,
        fail_rate: 1.0,
        ..Default::default()
    });
    let url = mock.start().await.expect("mock worker starts");

    let worker = Arc::new(BasicWorker::new(url, WorkerType::Regular));
    worker.set_healthy(true);

    let registry = WorkerRegistry::new();
    registry.register(worker.clone() as Arc<dyn Worker>);

    let handle = registry.start_inference_probe(
        InferenceProbeConfig {
            enabled: true,
            interval_secs: 1,
            timeout_secs: 5,
            concurrency: 32,
        },
        None,
    );

    // Let at least one sweep run, twice over, to catch a mutating probe.
    tokio::time::sleep(tokio::time::Duration::from_millis(2500)).await;
    handle.stop();

    assert!(
        worker.is_healthy(),
        "a full sweep of a failing worker must not change its health"
    );
    assert_eq!(worker.load(), 0, "probes must not touch load accounting");

    mock.stop().await;
}

/// A sweep must probe workers with bounded concurrency.
///
/// Awaiting each worker in turn makes one sweep last up to `workers * timeout`
/// — 32 minutes for 64 stalled workers at the default timeout — which is far
/// longer than the probe interval and leaves later workers' gauges stale exactly
/// when an outage makes every probe slow.
#[tokio::test]
async fn periodic_sweep_probes_with_bounded_concurrency() {
    use axum::{extract::State, routing::post, Json, Router};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc as StdArc;

    #[derive(Default)]
    struct InFlight {
        current: AtomicUsize,
        peak: AtomicUsize,
        hits: AtomicUsize,
    }

    const WORKERS: usize = 6;
    const CONCURRENCY: usize = 2;

    async fn completions_handler(State(state): State<StdArc<InFlight>>) -> Json<serde_json::Value> {
        let now = state.current.fetch_add(1, Ordering::SeqCst) + 1;
        state.peak.fetch_max(now, Ordering::SeqCst);
        state.hits.fetch_add(1, Ordering::SeqCst);
        // Hold the request open so overlapping probes are observable.
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        state.current.fetch_sub(1, Ordering::SeqCst);

        Json(serde_json::json!({
            "id": "cmpl-probe",
            "choices": [{"text": ".", "index": 0, "finish_reason": "length"}]
        }))
    }

    let state = StdArc::new(InFlight::default());
    let registry = WorkerRegistry::new();

    for _ in 0..WORKERS {
        let app = Router::new()
            .route("/v1/completions", post(completions_handler))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });
        registry.register(Arc::new(BasicWorker::new(
            format!("http://127.0.0.1:{port}"),
            WorkerType::Regular,
        )) as Arc<dyn Worker>);
    }

    let handle = registry.start_inference_probe(
        InferenceProbeConfig {
            enabled: true,
            interval_secs: 1,
            timeout_secs: 5,
            concurrency: CONCURRENCY,
        },
        None,
    );

    tokio::time::sleep(tokio::time::Duration::from_millis(2500)).await;
    handle.stop();

    let peak = state.peak.load(Ordering::SeqCst);
    assert!(
        peak <= CONCURRENCY,
        "at most {CONCURRENCY} probes may be in flight at once, observed {peak}"
    );
    assert!(
        peak >= 2,
        "probes must overlap rather than run one at a time, observed peak {peak}"
    );
    assert!(
        state.hits.load(Ordering::SeqCst) >= WORKERS,
        "every registered worker must still be probed"
    );
}
