#![cfg(all(feature = "cpu", feature = "ready"))]

//! Integration tests for the typed anthropic `/v1/messages` route.

mod common;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
    Router,
};
use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::sync::Arc;
use tower::ServiceExt;
use vllm_router_rs::config::{PolicyConfig, RetryConfig, RouterConfig, RoutingMode};
use vllm_router_rs::routers::{RouterFactory, RouterTrait};

fn worker(fail_rate: f32) -> MockWorkerConfig {
    MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate,
    }
}

struct Ctx {
    workers: Vec<MockWorker>,
    app: Router,
}

impl Ctx {
    async fn new(
        policy: PolicyConfig,
        workers: Vec<MockWorkerConfig>,
        tweak: impl FnOnce(&mut RouterConfig),
    ) -> Self {
        let mut started = Vec::new();
        let mut urls = Vec::new();
        for cfg in workers {
            let mut w = MockWorker::new(cfg);
            urls.push(w.start().await.unwrap());
            started.push(w);
        }
        Self::with_urls(policy, started, urls, tweak).await
    }

    async fn with_urls(
        policy: PolicyConfig,
        started: Vec<MockWorker>,
        urls: Vec<String>,
        tweak: impl FnOnce(&mut RouterConfig),
    ) -> Self {
        let mut config = RouterConfig {
            mode: RoutingMode::Regular { worker_urls: urls },
            policy,
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            ..Default::default()
        };
        tweak(&mut config);
        let app_context = common::create_test_context(config.clone());
        let router: Arc<dyn RouterTrait> =
            Arc::from(RouterFactory::create_router(&app_context).await.unwrap());
        let app = common::test_app::create_test_app(router, Client::new(), &config);
        Self {
            workers: started,
            app,
        }
    }

    async fn post(&self, body: &str, headers: &[(&str, &str)]) -> (StatusCode, Value) {
        let mut req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header(CONTENT_TYPE, "application/json");
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        let resp = self
            .app
            .clone()
            .oneshot(req.body(Body::from(body.to_string())).unwrap())
            .await
            .unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, value)
    }

    async fn shutdown(mut self) {
        for w in &mut self.workers {
            w.stop().await;
        }
    }
}

fn agent_body(user_id: Option<&str>, question: &str) -> Value {
    let mut body = json!({
        "model": "mock-model",
        "max_tokens": 32,
        "temperature": 0.2,
        "system": [{"type": "text", "text": "you are a coding agent", "cache_control": {"type": "ephemeral"}}],
        "tools": [{"name": "read_file", "description": "read a file", "input_schema": {"type": "object", "properties": {"user_id": {"type": "string"}}}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": question}]}]
    });
    if let Some(id) = user_id {
        body["metadata"] = json!({"user_id": id});
    }
    body
}

#[tokio::test]
async fn messages_passthrough_forwards_body_unchanged() {
    let ctx = Ctx::new(PolicyConfig::RoundRobin, vec![worker(0.0)], |_| {}).await;
    let mut body = agent_body(Some("u1"), "hello");
    body["thinking"] = json!({"type": "enabled", "budget_tokens": 1024});
    body["stop_sequences"] = json!(["\n\nhuman:"]);
    let (status, resp) = ctx.post(&body.to_string(), &[]).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["type"], "message");
    assert_eq!(resp["echo"], body);
    ctx.shutdown().await;
}

#[tokio::test]
async fn messages_invalid_json_uses_anthropic_error_shape() {
    let ctx = Ctx::new(PolicyConfig::RoundRobin, vec![worker(0.0)], |_| {}).await;
    let (status, resp) = ctx.post("{not json", &[]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(resp["type"], "error");
    assert_eq!(resp["error"]["type"], "invalid_request_error");
    assert!(!resp["error"]["message"].as_str().unwrap().is_empty());

    // valid json, wrong shape (messages missing)
    let (status, resp) = ctx.post(r#"{"model":"m"}"#, &[]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(resp["error"]["type"], "invalid_request_error");

    // required by the Messages API: model and max_tokens
    for body in [
        r#"{"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#,
        r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#,
    ] {
        let (status, resp) = ctx.post(body, &[]).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(resp["error"]["type"], "invalid_request_error");
    }
    ctx.shutdown().await;
}

#[tokio::test]
async fn messages_no_workers_uses_anthropic_error_shape() {
    let mut ctx = Ctx::new(PolicyConfig::RoundRobin, vec![worker(0.0)], |c| {
        c.disable_retries = true;
    })
    .await;
    for w in &mut ctx.workers {
        w.stop().await;
    }
    let (status, resp) = ctx.post(&agent_body(None, "hi").to_string(), &[]).await;
    assert!(!status.is_success());
    assert_eq!(resp["type"], "error", "body: {resp}");
    assert!(resp["error"]["type"].is_string());
    assert!(resp["error"]["message"].is_string());
}

#[tokio::test]
async fn messages_worker_error_passes_through_unchanged() {
    let ctx = Ctx::new(PolicyConfig::RoundRobin, vec![worker(1.0)], |c| {
        c.disable_retries = true;
        c.disable_circuit_breaker = true;
    })
    .await;
    let (status, resp) = ctx.post(&agent_body(None, "hi").to_string(), &[]).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        resp,
        json!({"type": "error", "error": {"type": "api_error", "message": "mock worker failure"}})
    );
    ctx.shutdown().await;
}

#[tokio::test]
async fn messages_retries_on_worker_5xx() {
    let ctx = Ctx::new(
        PolicyConfig::RoundRobin,
        vec![worker(1.0), worker(0.0)],
        |c| {
            c.retry = RetryConfig {
                max_retries: 3,
                initial_backoff_ms: 1,
                max_backoff_ms: 2,
                ..RetryConfig::default()
            };
            c.disable_circuit_breaker = true;
        },
    )
    .await;
    for i in 0..4 {
        let (status, resp) = ctx
            .post(&agent_body(None, &format!("q{i}")).to_string(), &[])
            .await;
        assert_eq!(status, StatusCode::OK, "request {i}: {resp}");
    }
    ctx.shutdown().await;
}

#[tokio::test]
async fn messages_metadata_user_id_drives_affinity_after_headers() {
    let ctx = Ctx::new(
        PolicyConfig::ConsistentHash { virtual_nodes: 160 },
        vec![worker(0.0), worker(0.0), worker(0.0)],
        |_| {},
    )
    .await;

    // same user id, different conversations: same worker every time
    let mut seen = HashSet::new();
    for i in 0..8 {
        let (status, resp) = ctx
            .post(
                &agent_body(Some("alice"), &format!("q{i}")).to_string(),
                &[],
            )
            .await;
        assert_eq!(status, StatusCode::OK);
        seen.insert(resp["worker_port"].as_u64().unwrap());
    }
    assert_eq!(seen.len(), 1, "metadata.user_id should pin one worker");

    // different user ids spread over more than one worker
    let mut spread = HashSet::new();
    for i in 0..24 {
        let (_, resp) = ctx
            .post(
                &agent_body(Some(&format!("user-{i}")), "q").to_string(),
                &[],
            )
            .await;
        spread.insert(resp["worker_port"].as_u64().unwrap());
    }
    assert!(
        spread.len() > 1,
        "user ids should not collapse onto one worker"
    );

    // a session header wins over metadata.user_id
    let (_, pinned) = ctx
        .post(
            &agent_body(Some("user-0"), "q").to_string(),
            &[("x-session-id", "sess-1")],
        )
        .await;
    for i in 1..8 {
        let (_, resp) = ctx
            .post(
                &agent_body(Some(&format!("user-{i}")), "q").to_string(),
                &[("x-session-id", "sess-1")],
            )
            .await;
        assert_eq!(resp["worker_port"], pinned["worker_port"]);
    }

    // the router does not invent an x-user-id header for the worker
    let alice_port = *seen.iter().next().unwrap() as u16;
    let captured = common::mock_worker::get_captured_requests(alice_port);
    assert!(captured.iter().any(|c| c.path == "/v1/messages"));
    assert!(captured
        .iter()
        .all(|c| !c.headers.contains_key("x-user-id")));
    ctx.shutdown().await;
}

#[tokio::test]
async fn messages_concurrency_reject_uses_anthropic_error_shape() {
    // no tokens and no queue: the concurrency layer rejects with 429 before
    // the handler runs, and the body must still be an anthropic error.
    let ctx = Ctx::new(PolicyConfig::RoundRobin, vec![worker(0.0)], |c| {
        c.max_concurrent_requests = 0;
        c.rate_limit_tokens_per_second = Some(0);
    })
    .await;
    let (status, resp) = ctx.post(&agent_body(None, "hi").to_string(), &[]).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(resp["type"], "error", "body: {resp}");
    assert_eq!(resp["error"]["type"], "rate_limit_error");
    ctx.shutdown().await;
}

#[tokio::test]
async fn messages_stream_worker_json_error_keeps_status_and_content_type() {
    let ctx = Ctx::new(PolicyConfig::RoundRobin, vec![worker(1.0)], |c| {
        c.disable_retries = true;
        c.disable_circuit_breaker = true;
    })
    .await;
    let mut body = agent_body(None, "hi");
    body["stream"] = json!(true);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = ctx.app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let ct = resp.headers()[CONTENT_TYPE].to_str().unwrap().to_string();
    assert!(ct.contains("application/json"), "content-type: {ct}");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        value,
        json!({"type": "error", "error": {"type": "api_error", "message": "mock worker failure"}})
    );
    ctx.shutdown().await;
}

#[tokio::test]
async fn messages_stream_forwards_sse_events() {
    let ctx = Ctx::new(PolicyConfig::RoundRobin, vec![worker(0.0)], |_| {}).await;
    let mut body = agent_body(None, "hi");
    body["stream"] = json!(true);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = ctx.app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp.headers()[CONTENT_TYPE].to_str().unwrap().to_string();
    assert!(ct.contains("text/event-stream"), "content-type: {ct}");
    // the stream ends without `data: [DONE]`, so reading it to the end also
    // checks the router doesn't wait for one.
    let bytes = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        axum::body::to_bytes(resp.into_body(), usize::MAX),
    )
    .await
    .expect("stream did not finish")
    .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let names: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("event:").map(str::trim))
        .collect();
    assert_eq!(
        names,
        [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop"
        ],
        "body: {text}"
    );
    assert!(!text.contains("[DONE]"), "body: {text}");
    let start: Value = serde_json::from_str(
        text.lines()
            .find_map(|l| l.strip_prefix("data:"))
            .unwrap()
            .trim(),
    )
    .unwrap();
    let port = start["message"]["worker_port"].as_u64().unwrap() as u16;
    assert!(captured_messages(port)
        .iter()
        .any(|c| c.path == "/v1/messages"));
    ctx.shutdown().await;
}

/// A worker (or a reverse proxy in front of it) that rejects
/// `/v1/messages` with a large non-json body. Every other path is healthy.
async fn start_plain_text_error_worker(body: String) -> String {
    use axum::response::IntoResponse;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = Router::new().fallback(move |req: Request| {
        let body = body.clone();
        async move {
            if req.uri().path() == "/v1/messages" {
                (StatusCode::BAD_GATEWAY, [(CONTENT_TYPE, "text/html")], body).into_response()
            } else {
                (StatusCode::OK, [(CONTENT_TYPE, "application/json")], "{}").into_response()
            }
        }
    });
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn messages_worker_non_json_error_passes_through_unchanged() {
    // Larger than the router error rewrite limit, so a rewrite would also
    // have dropped the body.
    let html = format!("<html><body>{}</body></html>", "x".repeat(100 * 1024));
    let url = start_plain_text_error_worker(html.clone()).await;
    let ctx = Ctx::with_urls(PolicyConfig::RoundRobin, Vec::new(), vec![url], |c| {
        c.disable_retries = true;
        c.disable_circuit_breaker = true;
    })
    .await;
    for stream in [false, true] {
        let mut body = agent_body(None, "hi");
        body["stream"] = json!(stream);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = ctx.app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY, "stream={stream}");
        let ct = resp.headers()[CONTENT_TYPE].to_str().unwrap().to_string();
        assert!(
            ct.contains("text/html"),
            "stream={stream} content-type: {ct}"
        );
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(bytes, html.as_bytes(), "stream={stream}");
    }
    ctx.shutdown().await;
}

#[tokio::test]
async fn messages_oversized_content_length_uses_anthropic_error_shape() {
    let ctx = Ctx::new(PolicyConfig::RoundRobin, vec![worker(0.0)], |c| {
        c.max_payload_size = 1024;
    })
    .await;
    let body = agent_body(None, &"x".repeat(4096)).to_string();
    let len = body.len().to_string();
    let (status, resp) = ctx.post(&body, &[("content-length", &len)]).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(resp["type"], "error", "body: {resp}");
    assert_eq!(resp["error"]["type"], "request_too_large");
    assert!(!resp["error"]["message"].as_str().unwrap().is_empty());
    ctx.shutdown().await;
}

fn captured_messages(port: u16) -> Vec<common::mock_worker::CapturedRequest> {
    common::mock_worker::get_captured_requests(port)
        .into_iter()
        .filter(|c| c.path == "/v1/messages")
        .collect()
}

#[tokio::test]
async fn messages_configured_worker_api_key_wins_over_client_key() {
    let ctx = Ctx::new(PolicyConfig::RoundRobin, vec![worker(0.0)], |c| {
        c.api_key = Some("worker-secret".to_string());
    })
    .await;
    let (status, resp) = ctx
        .post(
            &agent_body(None, "hi").to_string(),
            &[("authorization", "Bearer client-secret")],
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let port = resp["worker_port"].as_u64().unwrap() as u16;
    let captured = captured_messages(port);
    assert!(!captured.is_empty());
    for c in &captured {
        assert_eq!(
            c.headers.get("authorization"),
            Some(&vec!["Bearer worker-secret".to_string()])
        );
    }
    ctx.shutdown().await;
}

#[tokio::test]
async fn messages_strips_hop_by_hop_headers() {
    let ctx = Ctx::new(PolicyConfig::RoundRobin, vec![worker(0.0)], |_| {}).await;
    let (status, resp) = ctx
        .post(
            &agent_body(None, "hi").to_string(),
            &[
                ("connection", "x-private-hop"),
                ("x-private-hop", "1"),
                ("proxy-authorization", "Basic cHJveHk6c2VjcmV0"),
                ("proxy-authenticate", "Basic"),
                ("keep-alive", "timeout=5"),
                ("te", "trailers"),
                ("trailer", "x-checksum"),
                ("upgrade", "h2c"),
                ("x-end-to-end", "kept"),
            ],
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let port = resp["worker_port"].as_u64().unwrap() as u16;
    let captured = captured_messages(port);
    assert!(!captured.is_empty());
    for c in &captured {
        for hop in [
            "connection",
            "x-private-hop",
            "proxy-authorization",
            "proxy-authenticate",
            "keep-alive",
            "te",
            "trailer",
            "upgrade",
        ] {
            let leaked = c.headers.get(hop).is_some_and(|vals| {
                vals.iter()
                    .any(|v| !(hop == "connection" && v.eq_ignore_ascii_case("keep-alive")))
            });
            assert!(!leaked, "{hop} leaked to worker: {:?}", c.headers.get(hop));
        }
        assert_eq!(
            c.headers.get("x-end-to-end"),
            Some(&vec!["kept".to_string()])
        );
    }
    ctx.shutdown().await;
}
