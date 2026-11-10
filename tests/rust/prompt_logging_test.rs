#![cfg(all(feature = "cpu", feature = "ready"))]

mod common;

use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
};
use common::{
    mock_worker::{MockWorker, MockWorkerConfig},
    test_app::create_test_app_with_tracing,
};
use reqwest::Client;
use serde_json::{json, Value};
use std::{
    io::{self, Write},
    sync::{Arc, Mutex},
};
use tower::ServiceExt;
use tracing::instrument::WithSubscriber;
use tracing_subscriber::fmt::MakeWriter;
use vllm_router_rs::{
    config::{EpdConfig, RouterConfig, RoutingMode},
    routers::RouterFactory,
};

#[derive(Clone, Default)]
struct LogBuffer(Arc<Mutex<Vec<u8>>>);

impl Write for LogBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for LogBuffer {
    type Writer = Self;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

async fn chat_request(
    enabled: bool,
    stream: bool,
    authorized: bool,
    epd_handler: bool,
    filter: &str,
) -> Vec<Value> {
    let mut worker = MockWorker::new(MockWorkerConfig::default());
    let worker_url = worker.start().await.unwrap();
    let mut config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![worker_url.clone()],
        },
        log_user_prompts: enabled,
        request_id_headers: Some(vec!["x-trace-id".to_string()]),
        api_key_validation_urls: vec![format!("{worker_url}/health")],
        ..Default::default()
    };
    let context = common::create_test_context(config.clone());
    let router = RouterFactory::create_router(&context).await.unwrap();
    if epd_handler {
        // Exercise the raw-JSON ingress handler with a regular mock backend;
        // encoder/decoder orchestration is independent of prompt logging.
        config.epd = Some(EpdConfig {
            encoder_urls: vec![],
            consumer_zmq_addrs: Default::default(),
        });
    }
    let app = create_test_app_with_tracing(Arc::from(router), Client::new(), &config, false);
    let logs = LogBuffer::default();
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(logs.clone())
        .json()
        .flatten_event(true)
        .without_time()
        .finish();

    let mut request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("x-trace-id", "prompt-test-request");
    if authorized {
        request = request.header("authorization", "Bearer credential-secret");
    }
    let response = app
        .oneshot(
            request
                .body(Body::from(
                    json!({
                        "model": "mock-model",
                        "stream": stream,
                        "messages": [
                            {"role": "system", "content": "system-secret"},
                            {"role": "user", "content": "第一条\nnext line"},
                            {"role": "assistant", "content": "assistant-secret"},
                            {"role": "user", "content": [
                                {"type": "text", "text": "second prompt"},
                                {"type": "image_url", "image_url": {"url": "media-secret"}},
                                {"type": "text", "text": "third part"}
                            ]}
                        ]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .with_subscriber(subscriber)
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        if authorized {
            StatusCode::OK
        } else {
            StatusCode::UNAUTHORIZED
        }
    );
    assert_eq!(response.headers()["x-request-id"], "prompt-test-request");
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    if authorized && stream {
        assert!(String::from_utf8_lossy(&body).contains("[DONE]"));
    }
    worker.stop().await;

    let output = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
    for excluded in [
        "system-secret",
        "assistant-secret",
        "media-secret",
        "credential-secret",
    ] {
        assert!(
            !output.contains(excluded),
            "unexpected log content: {excluded}"
        );
    }
    output
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[tokio::test(flavor = "current_thread")]
async fn logs_all_user_text_once_for_streaming_and_non_streaming_requests() {
    for epd_handler in [false, true] {
        for stream in [false, true] {
            let events = chat_request(
                true,
                stream,
                true,
                epd_handler,
                "vllm_router_rs::prompt=info",
            )
            .await;
            assert_eq!(events.len(), 1);
            assert_eq!(events[0]["request_id"], "prompt-test-request");
            assert_eq!(events[0]["model"], "mock-model");
            assert_eq!(events[0]["stream"], stream);
            assert_eq!(events[0]["level"], "INFO");
            assert_eq!(events[0]["target"], "vllm_router_rs::prompt");
            assert_eq!(
                events[0]["user_prompts"],
                r#"[["第一条\nnext line"], ["second prompt", "third part"]]"#
            );
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn disabled_switch_never_logs_prompts_even_at_trace_level() {
    for epd_handler in [false, true] {
        assert!(chat_request(
            false,
            false,
            true,
            epd_handler,
            "vllm_router_rs::prompt=trace"
        )
        .await
        .is_empty());
    }
}

#[tokio::test(flavor = "current_thread")]
async fn unauthorized_requests_never_log_prompts() {
    for epd_handler in [false, true] {
        assert!(chat_request(
            true,
            false,
            false,
            epd_handler,
            "vllm_router_rs::prompt=info"
        )
        .await
        .is_empty());
    }
}

#[tokio::test(flavor = "current_thread")]
async fn prompt_events_respect_the_existing_log_filter() {
    assert!(
        chat_request(true, false, true, false, "vllm_router_rs::prompt=warn")
            .await
            .is_empty()
    );
}

fn messages_payload(stream: bool) -> Value {
    json!({
        "model": "mock-model",
        "max_tokens": 128,
        "stream": stream,
        "system": [{"type": "text", "text": "system-secret"}],
        "tools": [{
            "name": "read_file", "description": "tool-definition-secret",
            "input_schema": {"type": "object", "properties": {}}
        }],
        "messages": [
            {"role": "user", "content": "第一条\nnext line"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "assistant-secret"},
                {"type": "tool_use", "id": "tool-1", "name": "read_file", "input": {}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tool-1", "content": [
                    {"type": "text", "text": "tool-result-secret"}
                ]},
                {"type": "text", "text": "second prompt", "cache_control": {"type": "ephemeral"}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "media-secret"}},
                {"type": "text", "text": "third part"}
            ]}
        ]
    })
}

const MESSAGES_STREAM: &str = concat!(
    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-test\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"mock-model\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
    "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
    "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1}}\n\n",
    "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
);

async fn messages_request(
    enabled: bool,
    authorized: bool,
    method: &str,
    uri: &str,
    body: &str,
    filter: &str,
) -> Vec<Value> {
    use axum::{
        response::IntoResponse,
        routing::{any, get},
        Json, Router,
    };

    let captured = Arc::new(Mutex::new(Vec::<Value>::new()));
    let observed = captured.clone();
    let backend = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(|| async {
            Json(json!({"object": "list", "data": [{"id": "mock-model", "object": "model"}]}))
        }))
        .route("/get_server_info", get(|| async { Json(json!({"model_path": "mock-model", "dp_size": 1})) }))
        .route("/get_model_info", get(|| async { Json(json!({"model_path": "mock-model"})) }))
        .fallback(any(move |Json(payload): Json<Value>| {
            let observed = observed.clone();
            async move {
                observed.lock().unwrap().push(payload.clone());
                if payload.get("stream").and_then(Value::as_bool).unwrap_or(false) {
                    ([("content-type", "text/event-stream")], MESSAGES_STREAM).into_response()
                } else {
                    Json(json!({"type": "message", "role": "assistant", "content": [{"type": "text", "text": "ok"}]})).into_response()
                }
            }
        }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker_url = format!("http://{}", listener.local_addr().unwrap());
    let worker = tokio::spawn(async move {
        axum::serve(listener, backend).await.unwrap();
    });

    let config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![worker_url.clone()],
        },
        log_user_prompts: enabled,
        api_key_validation_urls: vec![format!("{worker_url}/health")],
        ..Default::default()
    };
    let context = common::create_test_context(config.clone());
    let router = RouterFactory::create_router(&context).await.unwrap();
    let app = create_test_app_with_tracing(Arc::from(router), Client::new(), &config, false);
    let logs = LogBuffer::default();
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(logs.clone())
        .json()
        .flatten_event(true)
        .without_time()
        .finish();
    let mut request = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .header("x-request-id", "messages-test-request")
        .header("x-api-key", "api-key-secret")
        .header("anthropic-version", "2023-06-01");
    if authorized {
        request = request.header("authorization", "Bearer credential-secret");
    }
    let response = app
        .oneshot(request.body(Body::from(body.to_owned())).unwrap())
        .with_subscriber(subscriber)
        .await
        .unwrap();
    let parsed = serde_json::from_str::<Value>(body);
    let status = if !authorized {
        StatusCode::UNAUTHORIZED
    } else if parsed.is_err() {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::OK
    };
    assert_eq!(response.status(), status);
    assert_eq!(response.headers()["x-request-id"], "messages-test-request");
    let response_body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    if status == StatusCode::OK {
        let payload = parsed.unwrap();
        assert_eq!(*captured.lock().unwrap(), vec![payload.clone()]);
        if payload["stream"] == true {
            assert_eq!(response_body.as_ref(), MESSAGES_STREAM.as_bytes());
        } else {
            assert_eq!(
                serde_json::from_slice::<Value>(&response_body).unwrap()["content"][0]["text"],
                "ok"
            );
        }
    } else {
        assert!(captured.lock().unwrap().is_empty());
    }
    worker.abort();

    let output = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
    for excluded in [
        "system-secret",
        "assistant-secret",
        "tool-result-secret",
        "tool-definition-secret",
        "media-secret",
        "api-key-secret",
        "credential-secret",
    ] {
        assert!(
            !output.contains(excluded),
            "unexpected log content: {excluded}"
        );
    }
    output
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[tokio::test(flavor = "current_thread")]
async fn messages_requests_log_user_text_once_and_preserve_body() {
    for stream in [false, true] {
        for uri in ["/v1/messages", "/v1/messages?beta=true"] {
            let events = messages_request(
                true,
                true,
                "POST",
                uri,
                &messages_payload(stream).to_string(),
                "vllm_router_rs::prompt=info",
            )
            .await;
            assert_eq!(events.len(), 1, "stream={stream}, uri={uri}");
            assert_eq!(events[0]["request_id"], "messages-test-request");
            assert_eq!(events[0]["model"], "mock-model");
            assert_eq!(events[0]["route"], "/v1/messages");
            assert_eq!(events[0]["stream"], stream);
            assert_eq!(events[0]["level"], "INFO");
            assert_eq!(
                events[0]["user_prompts"],
                r#"[["第一条\nnext line"], ["second prompt", "third part"]]"#
            );
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn messages_prompt_logging_requires_opt_in() {
    assert!(messages_request(
        false,
        true,
        "POST",
        "/v1/messages",
        &messages_payload(false).to_string(),
        "vllm_router_rs::prompt=trace"
    )
    .await
    .is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn messages_prompt_logging_requires_authorization() {
    assert!(messages_request(
        true,
        false,
        "POST",
        "/v1/messages",
        &messages_payload(false).to_string(),
        "vllm_router_rs::prompt=info"
    )
    .await
    .is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn messages_prompt_logging_respects_log_filter() {
    assert!(messages_request(
        true,
        true,
        "POST",
        "/v1/messages",
        &messages_payload(false).to_string(),
        "vllm_router_rs::prompt=warn"
    )
    .await
    .is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn messages_prompt_logging_excludes_other_paths_and_methods() {
    for (method, uri) in [
        ("POST", "/v1/messages/count_tokens"),
        ("GET", "/v1/messages"),
        ("POST", "/v1/other"),
    ] {
        assert!(messages_request(
            true,
            true,
            method,
            uri,
            &messages_payload(false).to_string(),
            "vllm_router_rs::prompt=info"
        )
        .await
        .is_empty());
    }
}

#[tokio::test(flavor = "current_thread")]
async fn malformed_messages_requests_do_not_log_prompts() {
    assert!(messages_request(
        true,
        true,
        "POST",
        "/v1/messages",
        "{invalid-json",
        "vllm_router_rs::prompt=info"
    )
    .await
    .is_empty());
}
