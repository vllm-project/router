#![cfg(all(feature = "cpu", feature = "ready"))]

use vllm_router_rs::metrics::{start_prometheus, PrometheusConfig};

#[test]
fn start_prometheus_runs_outside_tokio() {
    start_prometheus(PrometheusConfig {
        port: 0,
        host: "127.0.0.1".to_string(),
    });
}
