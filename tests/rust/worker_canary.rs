#![cfg(all(feature = "cpu", feature = "ready"))]

use std::{
    process::{Child, Command, Stdio},
    time::Duration,
};

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use tokio::{
    net::TcpListener,
    sync::{mpsc, oneshot},
};

struct RouterProcess(Child);

impl Drop for RouterProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

type Reply = oneshot::Sender<Response>;

fn one_token() -> Response {
    Json(json!({
        "choices": [{"text":"x","finish_reason":"length"}],
        "usage": {"completion_tokens":1},
    }))
    .into_response()
}

fn sample(text: &str, key: &str) -> f64 {
    text.lines()
        .find_map(|line| {
            let (metric, value) = line.split_once(' ')?;
            (metric == key).then(|| value.parse().unwrap())
        })
        .unwrap_or_default()
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_canary_cli_controls_readiness_after_threshold_and_exports_results() {
    tokio::time::timeout(Duration::from_secs(30), async {
        let (calls, mut pending) = mpsc::channel(4);
        let app = Router::new()
            .route("/health", get(|| async { StatusCode::OK }))
            .route(
                "/v1/completions",
                post(
                    |State(calls): State<mpsc::Sender<Reply>>, Json(body): Json<Value>| async move {
                        assert_eq!(body["max_tokens"], 1);
                        assert_eq!(body["stream"], false);
                        assert!(body.get("kv_transfer_params").is_none());
                        let (tx, rx) = oneshot::channel();
                        calls.send(tx).await.unwrap();
                        rx.await
                            .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
                    },
                ),
            )
            .with_state(calls);
        let worker = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let worker_url = format!("http://{}", worker.local_addr().unwrap());
        let _worker = tokio::spawn(async move { axum::serve(worker, app).await.unwrap() });
        let api_port = portpicker::pick_unused_port().unwrap();
        let metrics_port = loop {
            let port = portpicker::pick_unused_port().unwrap();
            if port != api_port {
                break port;
            }
        };
        let _router = RouterProcess(
            Command::new(env!("CARGO_BIN_EXE_vllm-router"))
                .args([
                    "--host",
                    "127.0.0.1",
                    "--port",
                    &api_port.to_string(),
                    "--worker-urls",
                    &worker_url,
                    "--policy",
                    "round_robin",
                    "--prometheus-host",
                    "127.0.0.1",
                    "--prometheus-port",
                    &metrics_port.to_string(),
                    "--worker-canary-interval-secs",
                    "1",
                    "--health-check-interval-secs",
                    "1",
                    "--health-failure-threshold",
                    "2",
                    "--health-success-threshold",
                    "2",
                    "--worker-startup-check-interval",
                    "1",
                ])
                .stdout(Stdio::null())
                .spawn()
                .unwrap(),
        );
        let client = reqwest::Client::new();
        let metrics_url = format!("http://127.0.0.1:{metrics_port}/metrics");
        let readiness_url = format!("http://127.0.0.1:{api_port}/readiness");
        for (result, count, healthy) in [
            ("ok", 1, true),
            ("failed", 1, true),
            ("failed", 2, false),
            ("ok", 2, false),
            ("ok", 3, true),
        ] {
            let response = if result == "ok" {
                one_token()
            } else {
                // A bare successful HTTP status is not generation.
                Json(json!({"choices":[]})).into_response()
            };
            pending.recv().await.unwrap().send(response).unwrap();
            let key = format!("vllm_router_worker_canary_total{{result=\"{result}\"}}");
            loop {
                let scrape = client
                    .get(&metrics_url)
                    .send()
                    .await
                    .unwrap()
                    .text()
                    .await
                    .unwrap();
                if sample(&scrape, &key) >= count as f64 {
                    assert_eq!(sample(&scrape, &key), count as f64);
                    assert!(sample(&scrape, "vllm_router_worker_idle_seconds") >= 1.);
                    assert!(!scrape.lines().any(|line| line
                        .starts_with("vllm_router_requests_total{route=\"/v1/completions\"")));
                    break;
                }
                tokio::task::yield_now().await;
            }
            let response = client.get(&readiness_url).send().await.unwrap();
            assert_eq!(response.status().is_success(), healthy);
            assert!(client
                .get(format!("{worker_url}/health"))
                .send()
                .await
                .unwrap()
                .status()
                .is_success());
        }
    })
    .await
    .unwrap();
}
