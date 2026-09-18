use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use axum::{http::StatusCode, routing::get, Router};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use vllm_router_rs::core::{
    BasicWorker, DPAwareWorker, HealthConfig, Worker, WorkerRegistry, WorkerType,
};

fn assert_health_metric(handle: &PrometheusHandle, url: &str, expected: u8) {
    let metrics = handle.render();
    let expected = format!("vllm_router_worker_health{{worker=\"{url}\"}} {expected}");
    assert!(
        metrics.lines().any(|line| line == expected),
        "Missing metric: {expected}\n{metrics}"
    );
}

#[test]
fn registration_exports_current_health() {
    let healthy = Arc::new(BasicWorker::new(
        "http://healthy:8000".to_string(),
        WorkerType::Regular,
    ));
    let unhealthy = Arc::new(BasicWorker::new(
        "http://unhealthy:8000".to_string(),
        WorkerType::Regular,
    ));
    unhealthy.set_healthy(false);

    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        let registry = WorkerRegistry::new();
        registry.register(healthy.clone());
        registry.register(unhealthy.clone());

        assert_health_metric(&handle, healthy.url(), 1);
        assert_health_metric(&handle, unhealthy.url(), 0);
        assert!(healthy.is_healthy());
        assert!(!unhealthy.is_healthy());
    });
}

#[test]
fn registration_exports_each_dp_rank() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        let registry = WorkerRegistry::new();
        for rank in 0..2 {
            registry.register(Arc::new(DPAwareWorker::new(
                "http://worker:8000".to_string(),
                rank,
                2,
                WorkerType::Regular,
            )));
        }

        assert_health_metric(&handle, "http://worker:8000@0", 1);
        assert_health_metric(&handle, "http://worker:8000@1", 1);
    });
}

#[tokio::test(flavor = "current_thread")]
async fn health_probes_preserve_thresholds_and_metrics() {
    let healthy = Arc::new(AtomicBool::new(true));
    let response_health = healthy.clone();
    let app = Router::new().route(
        "/health",
        get(move || {
            let healthy = response_health.clone();
            async move {
                if healthy.load(Ordering::Relaxed) {
                    StatusCode::OK
                } else {
                    StatusCode::SERVICE_UNAVAILABLE
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let worker = Arc::new(
        BasicWorker::new(url.clone(), WorkerType::Regular).with_health_config(HealthConfig {
            failure_threshold: 3,
            success_threshold: 2,
            ..HealthConfig::default()
        }),
    );
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    // Keep the recorder on the same thread as the health probes.
    let _guard = metrics::set_default_local_recorder(&recorder);
    let registry = WorkerRegistry::new();
    registry.register(worker.clone());

    for _ in 0..3 {
        worker.check_health_async().await.unwrap();
        assert_health_metric(&handle, &url, 1);
    }

    healthy.store(false, Ordering::Relaxed);
    for _ in 0..2 {
        assert!(worker.check_health_async().await.is_err());
        assert!(worker.is_healthy());
        assert_health_metric(&handle, &url, 1);
    }
    assert!(worker.check_health_async().await.is_err());
    assert!(!worker.is_healthy());
    assert_health_metric(&handle, &url, 0);

    healthy.store(true, Ordering::Relaxed);
    worker.check_health_async().await.unwrap();
    assert!(!worker.is_healthy());
    assert_health_metric(&handle, &url, 0);
    worker.check_health_async().await.unwrap();
    assert!(worker.is_healthy());
    assert_health_metric(&handle, &url, 1);

    server.abort();
}
