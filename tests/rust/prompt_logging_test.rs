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
