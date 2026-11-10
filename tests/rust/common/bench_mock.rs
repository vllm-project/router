//! Minimal worker used by the router overhead harness.
//!
//! Unlike [`super::mock_worker::MockWorker`], it keeps no per-request
//! capture store, parses no JSON and sleeps for nothing, so at tens of
//! thousands of requests per second its own cost stays flat. It answers
//! the endpoints the router touches at startup (`/health`), during health
//! checks and on the benchmarked routes, and counts the generation
//! requests it served so the harness can report how load spread across
//! workers.

use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub struct BenchMockWorker {
    url: String,
    served: Arc<AtomicU64>,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl BenchMockWorker {
    /// Bind an ephemeral port on 127.0.0.1 and serve until [`stop`](Self::stop).
    pub async fn start() -> BenchMockWorker {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind bench mock worker");
        let port = listener.local_addr().expect("local addr").port();
        let served = Arc::new(AtomicU64::new(0));
        let app = Router::new()
            .route("/health", get(health))
            .route("/health_generate", get(health))
            .route("/v1/models", get(models))
            .route("/v1/completions", post(completion))
            .route("/v1/chat/completions", post(chat_completion))
            .route("/generate", post(generate))
            .with_state(Arc::clone(&served));
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let handle = tokio::spawn(async move {
            let server = axum::serve(listener, app).with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            });
            if let Err(e) = server.await {
                eprintln!("bench mock worker error: {e}");
            }
        });
        BenchMockWorker {
            url: format!("http://127.0.0.1:{port}"),
            served,
            shutdown_tx: Some(shutdown_tx),
            handle: Some(handle),
        }
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    /// Generation requests served since the last [`reset_served`](Self::reset_served).
    pub fn served(&self) -> u64 {
        self.served.load(Ordering::Relaxed)
    }

    pub fn reset_served(&self) {
        self.served.store(0, Ordering::Relaxed);
    }

    pub async fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
    }
}

type Served = State<Arc<AtomicU64>>;

async fn health() -> Response {
    Json(json!({"status": "healthy"})).into_response()
}

async fn models() -> Response {
    Json(json!({
        "object": "list",
        "data": [{"id": "mock-model", "object": "model", "owned_by": "vllm"}]
    }))
    .into_response()
}

// The body is read to completion so the connection can be reused, and then
// dropped without parsing: a worker that does nothing is the floor we are
// measuring the router against.
async fn completion(State(served): Served, _body: Bytes) -> Response {
    served.fetch_add(1, Ordering::Relaxed);
    Json(json!({
        "id": "cmpl-bench",
        "object": "text_completion",
        "created": 0,
        "model": "mock-model",
        "choices": [{"text": "ok", "index": 0, "logprobs": null, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    }))
    .into_response()
}

async fn chat_completion(State(served): Served, _body: Bytes) -> Response {
    served.fetch_add(1, Ordering::Relaxed);
    Json(json!({
        "id": "chatcmpl-bench",
        "object": "chat.completion",
        "created": 0,
        "model": "mock-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    }))
    .into_response()
}

async fn generate(State(served): Served, _body: Bytes) -> Response {
    served.fetch_add(1, Ordering::Relaxed);
    Json(json!({
        "text": "ok",
        "meta_info": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": {"type": "stop"}}
    }))
    .into_response()
}
