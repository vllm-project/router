//! `grpc.health.v1` Check on the worker `--grpc-port`.
//!
//! Empty service name = overall server status (`vllm-rs` publishes that
//! from engine health). Not `Control.GetServerInfo` and not HTTP `/health`.

use std::time::Duration;

use tonic::transport::Channel;
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::pb::health_client::HealthClient;
use tonic_health::pb::HealthCheckRequest;

use super::detect::grpc_connect_uri;

/// `Ok(())` when `grpc.health.v1` reports `SERVING`.
pub async fn check_grpc_health(worker_url: &str, timeout: Duration) -> Result<(), String> {
    let uri = grpc_connect_uri(worker_url)?;
    let channel = Channel::from_shared(uri.clone())
        .map_err(|e| format!("invalid grpc uri {uri}: {e}"))?
        .connect_timeout(timeout)
        .timeout(timeout)
        .connect()
        .await
        .map_err(|e| format!("grpc health connect {uri}: {e}"))?;
    let mut client = HealthClient::new(channel);
    let resp = tokio::time::timeout(
        timeout,
        client.check(HealthCheckRequest {
            service: String::new(),
        }),
    )
    .await
    .map_err(|_| format!("grpc health check timeout {uri}"))?
    .map_err(|e| format!("grpc health check {uri}: {e}"))?;
    match ServingStatus::try_from(resp.into_inner().status) {
        Ok(ServingStatus::Serving) => Ok(()),
        Ok(other) => Err(format!("grpc health {uri}: {other:?}")),
        Err(_) => Err(format!("grpc health {uri}: unknown status")),
    }
}
