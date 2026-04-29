mod common;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use common::mock_worker::{
    HealthStatus, MockHttpResponseMode, MockWorker, MockWorkerConfig, WorkerType,
};
use common::test_app::create_test_app;
use metrics::set_default_local_recorder;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serde_json::json;
use std::sync::Arc;
use tokio::time::{sleep, Duration, Instant};
use tower::ServiceExt;
use vllm_router_rs::{
    config::{PolicyConfig, RouterConfig, RoutingMode},
    routers::RouterFactory,
};

fn build_test_recorder() -> metrics_exporter_prometheus::PrometheusRecorder {
    PrometheusBuilder::new().build_recorder()
}

#[derive(Clone, Copy, Debug)]
enum PdModeUnderTest {
    PrefillDecode,
    VllmPrefillDecode,
}

impl PdModeUnderTest {
    fn router_config(self, prefill_url: &str, decode_url: &str) -> RouterConfig {
        let mode = match self {
            Self::PrefillDecode => RoutingMode::PrefillDecode {
                prefill_urls: vec![(prefill_url.to_string(), None)],
                decode_urls: vec![decode_url.to_string()],
                prefill_policy: None,
                decode_policy: None,
            },
            Self::VllmPrefillDecode => RoutingMode::VllmPrefillDecode {
                prefill_urls: vec![(prefill_url.to_string(), None)],
                decode_urls: vec![decode_url.to_string()],
                prefill_policy: None,
                decode_policy: None,
                discovery_address: None,
            },
        };

        RouterConfig {
            mode,
            policy: PolicyConfig::RoundRobin,
            host: "127.0.0.1".to_string(),
            port: 0,
            request_timeout_secs: 10,
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            ..Default::default()
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::PrefillDecode => "PrefillDecode",
            Self::VllmPrefillDecode => "VllmPrefillDecode",
        }
    }

    fn prefill_success_count(self) -> usize {
        match self {
            Self::PrefillDecode => 6,
            Self::VllmPrefillDecode => 2,
        }
    }

    fn decode_error_count(self) -> usize {
        match self {
            Self::PrefillDecode => 5,
            Self::VllmPrefillDecode => 1,
        }
    }
}

fn has_metric_line(rendered: &str, prefix: &str, fragments: &[&str], suffix: &str) -> bool {
    rendered.lines().any(|line| {
        line.starts_with(prefix)
            && fragments.iter().all(|fragment| line.contains(fragment))
            && line.ends_with(suffix)
    })
}

fn assert_metric_line(rendered: &str, prefix: &str, fragments: &[&str], suffix: &str) {
    assert!(
        has_metric_line(rendered, prefix, fragments, suffix),
        "missing metric line: prefix={prefix}, fragments={fragments:?}, suffix={suffix}\n{rendered}"
    );
}

async fn assert_metric_line_eventually(
    handle: &PrometheusHandle,
    prefix: &str,
    fragments: &[&str],
    suffix: &str,
) {
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut rendered = handle.render();

    loop {
        if has_metric_line(&rendered, prefix, fragments, suffix) {
            return;
        }

        if Instant::now() >= deadline {
            assert_metric_line(&rendered, prefix, fragments, suffix);
        }

        sleep(Duration::from_millis(10)).await;
        rendered = handle.render();
    }
}

async fn send_chat_completion_request(
    app: axum::Router,
    prompt: &str,
) -> axum::http::Response<Body> {
    let payload = json!({
        "model": "mock-model",
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": false,
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    app.oneshot(request).await.unwrap()
}

fn prefill_decode_router_config_without_retries(
    prefill_url: &str,
    decode_url: &str,
) -> RouterConfig {
    let mut config = PdModeUnderTest::PrefillDecode.router_config(prefill_url, decode_url);
    config.disable_retries = true;
    config
}

async fn run_prefill_decode_http_status_metrics_single_attempt_test() {
    let mut prefill_worker = MockWorker::with_http_response_mode(
        MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Prefill,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        },
        MockHttpResponseMode::Ok,
    );
    let prefill_url = prefill_worker
        .start()
        .await
        .expect("failed to start prefill worker");

    let mut decode_worker = MockWorker::with_http_response_mode(
        MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Decode,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        },
        MockHttpResponseMode::Ok,
    );
    let decode_url = decode_worker
        .start()
        .await
        .expect("failed to start decode worker");

    let config = prefill_decode_router_config_without_retries(&prefill_url, &decode_url);
    let app_context = common::create_test_context(config.clone());
    let router = RouterFactory::create_router(&app_context)
        .await
        .unwrap_or_else(|error| panic!("failed to create PrefillDecode router: {error}"));
    let app = create_test_app(Arc::from(router), reqwest::Client::new(), &config);

    let recorder = build_test_recorder();
    let handle = recorder.handle();
    // Use a thread-local recorder so this binary never contends for the global recorder.
    let _recorder_guard = set_default_local_recorder(&recorder);

    let success_response = send_chat_completion_request(app.clone(), "successful decode").await;
    assert_eq!(success_response.status(), StatusCode::OK);
    let _ = axum::body::to_bytes(success_response.into_body(), usize::MAX)
        .await
        .unwrap();

    decode_worker
        .set_http_response_mode(MockHttpResponseMode::TooManyRequests)
        .await;

    let downstream_error_response =
        send_chat_completion_request(app.clone(), "decode should return 429").await;
    assert_eq!(
        downstream_error_response.status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    let _ = axum::body::to_bytes(downstream_error_response.into_body(), usize::MAX)
        .await
        .unwrap();

    let route = "route=\"/v1/chat/completions\"";
    let method = "http_request_method=\"POST\"";
    let prefill_worker_label = format!("worker=\"{prefill_url}\"");
    let decode_worker_label = format!("worker=\"{decode_url}\"");

    assert_metric_line_eventually(
        &handle,
        "vllm_router_backend_http_responses_total{",
        &[
            route,
            prefill_worker_label.as_str(),
            "vllm_request_phase=\"prefill\"",
            method,
            "http_response_status_code=\"200\"",
            "http_response_status_class=\"2xx\"",
        ],
        " 2",
    )
    .await;
    assert_metric_line_eventually(
        &handle,
        "vllm_router_backend_http_response_duration_seconds_count{",
        &[
            route,
            prefill_worker_label.as_str(),
            "vllm_request_phase=\"prefill\"",
            method,
            "http_response_status_code=\"200\"",
            "http_response_status_class=\"2xx\"",
        ],
        " 2",
    )
    .await;
    assert_metric_line_eventually(
        &handle,
        "vllm_router_backend_http_responses_total{",
        &[
            route,
            decode_worker_label.as_str(),
            "vllm_request_phase=\"decode\"",
            method,
            "http_response_status_code=\"200\"",
            "http_response_status_class=\"2xx\"",
        ],
        " 1",
    )
    .await;
    assert_metric_line_eventually(
        &handle,
        "vllm_router_backend_http_response_duration_seconds_count{",
        &[
            route,
            decode_worker_label.as_str(),
            "vllm_request_phase=\"decode\"",
            method,
            "http_response_status_code=\"200\"",
            "http_response_status_class=\"2xx\"",
        ],
        " 1",
    )
    .await;
    assert_metric_line_eventually(
        &handle,
        "vllm_router_backend_http_responses_total{",
        &[
            route,
            decode_worker_label.as_str(),
            "vllm_request_phase=\"decode\"",
            method,
            "http_response_status_code=\"429\"",
            "http_response_status_class=\"4xx\"",
        ],
        " 1",
    )
    .await;
    assert_metric_line_eventually(
        &handle,
        "vllm_router_backend_http_response_duration_seconds_count{",
        &[
            route,
            decode_worker_label.as_str(),
            "vllm_request_phase=\"decode\"",
            method,
            "http_response_status_code=\"429\"",
            "http_response_status_class=\"4xx\"",
        ],
        " 1",
    )
    .await;
    assert_metric_line_eventually(
        &handle,
        "vllm_router_http_responses_total{",
        &[
            route,
            method,
            "http_response_status_code=\"200\"",
            "http_response_status_class=\"2xx\"",
        ],
        " 1",
    )
    .await;
    assert_metric_line_eventually(
        &handle,
        "vllm_router_http_response_duration_seconds_count{",
        &[
            route,
            method,
            "http_response_status_code=\"200\"",
            "http_response_status_class=\"2xx\"",
        ],
        " 1",
    )
    .await;
    assert_metric_line_eventually(
        &handle,
        "vllm_router_http_responses_total{",
        &[
            route,
            method,
            "http_response_status_code=\"429\"",
            "http_response_status_class=\"4xx\"",
        ],
        " 1",
    )
    .await;
    assert_metric_line_eventually(
        &handle,
        "vllm_router_http_response_duration_seconds_count{",
        &[
            route,
            method,
            "http_response_status_code=\"429\"",
            "http_response_status_class=\"4xx\"",
        ],
        " 1",
    )
    .await;

    decode_worker.clear_http_response_mode().await;
    prefill_worker.clear_http_response_mode().await;
    prefill_worker.stop().await;
    decode_worker.stop().await;
}

async fn run_pd_http_status_metrics_test(mode: PdModeUnderTest) {
    let mut prefill_worker = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Prefill,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });
    let prefill_url = prefill_worker
        .start()
        .await
        .expect("failed to start prefill worker");

    let mut decode_worker = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Decode,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });
    let decode_url = decode_worker
        .start()
        .await
        .expect("failed to start decode worker");

    let config = mode.router_config(&prefill_url, &decode_url);
    let app_context = common::create_test_context(config.clone());
    let router = RouterFactory::create_router(&app_context)
        .await
        .unwrap_or_else(|error| panic!("failed to create {} router: {error}", mode.name()));
    let app = create_test_app(Arc::from(router), reqwest::Client::new(), &config);

    let recorder = build_test_recorder();
    let handle = recorder.handle();
    // Use a thread-local recorder so this binary never contends for the global recorder.
    let _recorder_guard = set_default_local_recorder(&recorder);

    let success_response = send_chat_completion_request(app.clone(), "successful decode").await;
    assert_eq!(success_response.status(), StatusCode::OK);
    let _ = axum::body::to_bytes(success_response.into_body(), usize::MAX)
        .await
        .unwrap();

    decode_worker
        .set_http_response_mode(MockHttpResponseMode::TooManyRequests)
        .await;

    let downstream_error_response =
        send_chat_completion_request(app.clone(), "decode should return 429").await;
    assert_eq!(
        downstream_error_response.status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    let _ = axum::body::to_bytes(downstream_error_response.into_body(), usize::MAX)
        .await
        .unwrap();

    let route = "route=\"/v1/chat/completions\"";
    let method = "http_request_method=\"POST\"";
    let prefill_worker_label = format!("worker=\"{prefill_url}\"");
    let decode_worker_label = format!("worker=\"{decode_url}\"");
    let prefill_success_count = format!(" {}", mode.prefill_success_count());
    let decode_error_count = format!(" {}", mode.decode_error_count());

    assert_metric_line_eventually(
        &handle,
        "vllm_router_backend_http_responses_total{",
        &[
            route,
            prefill_worker_label.as_str(),
            "vllm_request_phase=\"prefill\"",
            method,
            "http_response_status_code=\"200\"",
            "http_response_status_class=\"2xx\"",
        ],
        prefill_success_count.as_str(),
    )
    .await;
    assert_metric_line_eventually(
        &handle,
        "vllm_router_backend_http_response_duration_seconds_count{",
        &[
            route,
            prefill_worker_label.as_str(),
            "vllm_request_phase=\"prefill\"",
            method,
            "http_response_status_code=\"200\"",
            "http_response_status_class=\"2xx\"",
        ],
        prefill_success_count.as_str(),
    )
    .await;
    assert_metric_line_eventually(
        &handle,
        "vllm_router_backend_http_responses_total{",
        &[
            route,
            decode_worker_label.as_str(),
            "vllm_request_phase=\"decode\"",
            method,
            "http_response_status_code=\"200\"",
            "http_response_status_class=\"2xx\"",
        ],
        " 1",
    )
    .await;
    assert_metric_line_eventually(
        &handle,
        "vllm_router_backend_http_responses_total{",
        &[
            route,
            decode_worker_label.as_str(),
            "vllm_request_phase=\"decode\"",
            method,
            "http_response_status_code=\"429\"",
            "http_response_status_class=\"4xx\"",
        ],
        decode_error_count.as_str(),
    )
    .await;
    assert_metric_line_eventually(
        &handle,
        "vllm_router_backend_http_response_duration_seconds_count{",
        &[
            route,
            decode_worker_label.as_str(),
            "vllm_request_phase=\"decode\"",
            method,
            "http_response_status_code=\"429\"",
            "http_response_status_class=\"4xx\"",
        ],
        decode_error_count.as_str(),
    )
    .await;
    assert_metric_line_eventually(
        &handle,
        "vllm_router_http_responses_total{",
        &[
            route,
            method,
            "http_response_status_code=\"200\"",
            "http_response_status_class=\"2xx\"",
        ],
        " 1",
    )
    .await;
    assert_metric_line_eventually(
        &handle,
        "vllm_router_http_responses_total{",
        &[
            route,
            method,
            "http_response_status_code=\"429\"",
            "http_response_status_class=\"4xx\"",
        ],
        " 1",
    )
    .await;
    assert_metric_line_eventually(
        &handle,
        "vllm_router_http_response_duration_seconds_count{",
        &[
            route,
            method,
            "http_response_status_code=\"429\"",
            "http_response_status_class=\"4xx\"",
        ],
        " 1",
    )
    .await;

    decode_worker.clear_http_response_mode().await;
    prefill_worker.stop().await;
    decode_worker.stop().await;
}

#[tokio::test(flavor = "current_thread")]
async fn test_pd_http_status_metrics_cover_prefill_decode_and_final_response() {
    run_pd_http_status_metrics_test(PdModeUnderTest::PrefillDecode).await;
}

#[tokio::test(flavor = "current_thread")]
async fn test_pd_http_status_metrics_cover_single_attempt_prefill_decode_backend_and_final_response(
) {
    run_prefill_decode_http_status_metrics_single_attempt_test().await;
}

#[tokio::test(flavor = "current_thread")]
async fn test_vllm_pd_http_status_metrics_cover_prefill_decode_and_final_response() {
    run_pd_http_status_metrics_test(PdModeUnderTest::VllmPrefillDecode).await;
}
