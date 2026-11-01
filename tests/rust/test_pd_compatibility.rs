//! Preserve successful responses, upstream dispatch boundaries, and metric
//! contracts while changing how P/D errors are returned.
#![cfg(all(feature = "cpu", feature = "ready"))]

mod common;

use common::pd_mock::{Fixture, GZIP_SSE};
use http::Method;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serde_json::json;
use std::time::Duration;

#[tokio::test]
async fn failed_prefill_headers_release_decode_before_error_body_eof() {
    use std::sync::atomic::Ordering;
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _guard = metrics::set_default_local_recorder(&recorder);
    for discovery in [false, true] {
        let f = if discovery {
            Fixture::profiled_discovery(Duration::from_secs(5)).await
        } else {
            Fixture::profiled(Duration::from_secs(5)).await
        };
        f.warmup().await;
        f.control.p_wait.store(true, Ordering::SeqCst);
        f.control.d_wait.store(true, Ordering::SeqCst);
        let path = format!("/compat/error-headers/{discovery}");
        let request = f.router.route_transparent(
            None,
            &path,
            &Method::POST,
            json!({"model":"p-error-body-wait", "messages":[]}),
        );
        tokio::pin!(request);
        tokio::select! {
            _ = &mut request => panic!("P and D must remain blocked"),
            _ = async { f.control.p_seen.notified().await; f.control.d_seen.notified().await; } => {}
        }
        f.control.p_release.notify_one();
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let loads_released = f
                    .context
                    .worker_registry
                    .get_all()
                    .iter()
                    .all(|w| w.load() == 0);
                if loads_released
                    && f.control.d_profile_stops.load(Ordering::SeqCst) == 2
                    && counter(&handle, "vllm_router_pd_requests_total", "route", &path) == 1
                {
                    break;
                }
                tokio::select! {
                    _ = &mut request => panic!("error body must remain blocked"),
                    _ = tokio::task::yield_now() => {}
                }
            }
        })
        .await
        .expect("failed P headers must release local D and loads before body EOF");
        f.assert_no_load();
        f.control.p_body_release.notify_one();
        assert_eq!(request.await.status(), 503);
        assert_eq!(
            counter(&handle, "vllm_router_pd_requests_total", "route", &path),
            1
        );
        f.control.d_release.notify_one();
    }
}

#[tokio::test]
async fn discovery_error_metrics_precede_body_capture_and_survive_cancellation() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _guard = metrics::set_default_local_recorder(&recorder);
    for cancel in [false, true] {
        let f = Fixture::new(Duration::from_secs(5), true).await;
        let path = format!("/compat/error-capture/{cancel}");
        {
            let request = f.router.route_transparent(
                None,
                &path,
                &Method::POST,
                json!({"model":"p-error-body-wait", "messages":[]}),
            );
            tokio::pin!(request);
            tokio::time::timeout(Duration::from_secs(1), async {
                loop {
                    if counter(&handle, "vllm_router_pd_requests_total", "route", &path) == 1 {
                        break;
                    }
                    tokio::select! {
                        _ = &mut request => panic!("error body must remain blocked"),
                        _ = tokio::task::yield_now() => {}
                    }
                }
            })
            .await
            .expect("legacy metrics must be recorded at non-2xx headers");
            let before = handle.render();
            let duration = before
                .lines()
                .find(|line| {
                    line.starts_with("vllm_router_pd_request_duration_seconds_sum{")
                        && line.contains(&format!("route=\"{path}\""))
                })
                .expect("duration must be observed before body EOF")
                .to_owned();
            assert_eq!(
                counter(
                    &handle,
                    "vllm_router_pd_prefill_errors_total",
                    "worker",
                    &f._p.url
                ),
                1
            );
            if !cancel {
                f.control.p_body_release.notify_one();
                assert_eq!(request.await.status(), 503);
                assert!(handle.render().lines().any(|line| line == duration));
            }
        }
        assert_eq!(
            counter(&handle, "vllm_router_pd_requests_total", "route", &path),
            1
        );
        f.control.p_body_release.notify_one();
    }
}

#[tokio::test]
async fn prefill_json_failure_uses_shared_stage_counters() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _guard = metrics::set_default_local_recorder(&recorder);
    for discovery in [true, false] {
        let f = Fixture::new(Duration::from_secs(2), discovery).await;
        let path = format!("/compat/prefill-json/{discovery}");
        let response = f
            .router
            .route_transparent(
                None,
                &path,
                &Method::POST,
                json!({"model":"p-invalid-json", "messages":[]}),
            )
            .await;
        assert_eq!(response.status(), 500);
        assert_eq!(
            counter(&handle, "vllm_router_pd_requests_total", "route", &path),
            1
        );
        assert_eq!(
            counter(
                &handle,
                "vllm_router_pd_prefill_errors_total",
                "worker",
                &f._p.url
            ),
            1
        );
        f.assert_no_load();
    }
}

#[tokio::test]
async fn prefill_body_reset_still_cancels_decode_waiting_for_headers() {
    for discovery in [false, true] {
        let f = Fixture::new(Duration::from_secs(3), discovery).await;
        f.warmup().await;
        f.control
            .d_wait
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let router = f.router.clone();
        let task = tokio::spawn(async move {
            router
                .route_transparent(
                    None,
                    "/v1/chat/completions",
                    &Method::POST,
                    json!({"model":"p-body-reset", "messages":[]}),
                )
                .await
        });
        tokio::time::timeout(Duration::from_secs(2), async {
            f.control.p_body_started.notified().await;
            f.control.d_seen.notified().await;
        })
        .await
        .unwrap();
        f.control.p_body_release.notify_one();
        let response = tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("P reset must not wait for D headers/deadline")
            .unwrap();
        assert_eq!(response.status(), 502);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("prefill"));
        f.assert_no_load();
        f.control.d_release.notify_one();
    }
}

#[tokio::test]
async fn concurrent_profiling_finishes_prefill_at_body_boundary() {
    for discovery in [true, false] {
        let f = if discovery {
            Fixture::profiled_discovery(Duration::from_secs(3)).await
        } else {
            Fixture::profiled(Duration::from_secs(3)).await
        };
        f.warmup().await;
        // Consume the warmup profile notification before installing barriers.
        f.control.p_profile_stopped.notified().await;
        f.control
            .d_wait
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let router = f.router.clone();
        let task = tokio::spawn(async move {
            router
                .route_transparent(
                    None,
                    "/v1/chat/completions",
                    &Method::POST,
                    json!({"model":"p-body-wait", "messages":[]}),
                )
                .await
        });
        tokio::time::timeout(Duration::from_secs(2), async {
            f.control.p_body_started.notified().await;
            f.control.d_seen.notified().await;
        })
        .await
        .expect("both phases must reach their barriers");
        assert_eq!(
            f.control
                .p_profile_stops
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        f.control.d_release.notify_one();
        assert!(!task.is_finished(), "P EOF still gates the response");
        assert_eq!(
            f.control
                .p_profile_stops
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "P profiling must include successful body consumption"
        );
        f.control.p_body_release.notify_one();
        let response = tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(response.status(), 200);
        axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        assert_eq!(
            f.control
                .p_profile_stops
                .load(std::sync::atomic::Ordering::SeqCst),
            2
        );
        assert_eq!(
            f.control
                .d_profile_stops
                .load(std::sync::atomic::Ordering::SeqCst),
            2
        );
        f.assert_no_load();
    }
}

#[tokio::test]
async fn nixl_opaque_transfer_params_are_forwarded_as_before() {
    for discovery in [false, true] {
        for (kind, expected) in [
            ("null", json!(null)),
            ("array", json!([1, 2])),
            ("string", json!("legacy-value")),
        ] {
            let f = Fixture::new(Duration::from_secs(2), discovery).await;
            let response = f
                .request("/v1/chat/completions", &format!("opaque-kv-{kind}"), false)
                .await;
            assert_eq!(response.status(), 200);
            let body: serde_json::Value = response.json().await.unwrap();
            assert_eq!(body["forwarded_kv"], expected);
            assert_eq!(body["forwarded_kv_present"], true);
        }
    }
}

#[tokio::test]
async fn configuration_errors_do_not_change_legacy_request_counters() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _guard = metrics::set_default_local_recorder(&recorder);
    let f = Fixture::moriio_without_mode(Duration::from_secs(2)).await;
    let path = "/compat/missing-transfer-mode";
    let response = f
        .router
        .route_transparent(None, path, &Method::POST, json!({"model":"test"}))
        .await;
    assert_eq!(response.status(), 500);
    assert_eq!(
        counter(&handle, "vllm_router_pd_requests_total", "route", path),
        0
    );
    assert_eq!(
        counter(
            &handle,
            "vllm_router_pd_errors_total",
            "error_type",
            "internal_error"
        ),
        0
    );
    f.assert_no_load();
}

#[tokio::test]
async fn response_body_failures_do_not_extend_legacy_stream_error_counter() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _guard = metrics::set_default_local_recorder(&recorder);
    for streaming in [false, true] {
        let f = Fixture::direct(Duration::from_secs(2)).await;
        // The mock declares an oversized non-streaming body to exercise the
        // passthrough path. Smaller non-streaming bodies are buffered first.
        let path = if streaming {
            "/v1/chat/completions"
        } else {
            "/custom"
        };
        let response = f
            .router
            .route_transparent(
                None,
                path,
                &Method::POST,
                json!({"model":"stream-reset", "stream":streaming, "messages":[]}),
            )
            .await;
        assert_eq!(response.status(), 200);
        f.control.stream_release.notify_one();
        assert!(axum::body::to_bytes(response.into_body(), 4096)
            .await
            .is_err());
        assert_eq!(
            counter(
                &handle,
                "vllm_router_pd_stream_errors_total",
                "worker",
                &f._d.url
            ),
            0
        );
    }
    let rendered = handle.render();
    let body_errors = rendered
        .lines()
        .find(|line| line.starts_with("vllm_router_pd_response_body_errors_total{"))
        .unwrap();
    assert!(body_errors.contains("phase=\"decode\""));
    assert!(body_errors.contains("error_type=\"upstream_transport_error\""));
    assert!(body_errors.ends_with(" 2"));
}

#[tokio::test]
async fn missing_optional_kv_params_still_reaches_decode() {
    for discovery in [false, true] {
        let f = Fixture::new(Duration::from_secs(2), discovery).await;
        let response = f
            .request("/v1/chat/completions", "p-missing-kv", false)
            .await;
        assert_eq!(
            response.status(),
            200,
            "HEAD continued to Decode without optional KV params"
        );
        response.bytes().await.unwrap();
        assert_eq!(f.d_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        f.assert_no_load();
    }
}

#[tokio::test]
async fn successful_profiling_stops_each_worker_once() {
    let f = Fixture::profiled(Duration::from_secs(2)).await;
    for expected in 1..=2 {
        let response = f.request("/v1/chat/completions", "test", false).await;
        assert_eq!(response.status(), 200);
        response.bytes().await.unwrap();
        assert_eq!(
            f.control
                .p_profile_stops
                .load(std::sync::atomic::Ordering::SeqCst),
            expected
        );
        assert_eq!(
            f.control
                .d_profile_stops
                .load(std::sync::atomic::Ordering::SeqCst),
            expected
        );
    }
}

#[tokio::test]
async fn successful_headers_keep_upstream_filtering_rules() {
    for (discovery, concurrent, stream, logprobs) in [
        (false, false, false, false),
        (false, false, true, false),
        (false, false, false, true),
        (false, true, false, false),
        (false, true, true, false),
        (false, true, false, true),
        (true, false, false, false),
        (true, false, true, false),
        (true, false, false, true),
        (true, true, false, false),
        (true, true, true, false),
        (true, true, false, true),
    ] {
        let f = Fixture::new(Duration::from_secs(2), discovery).await;
        if concurrent {
            f.warmup().await;
        }
        let mut body = json!({"model":"compat-headers", "stream":stream, "messages":[]});
        if logprobs {
            body["logprobs"] = json!(true);
        }
        let response = f
            .router
            .route_transparent(None, "/v1/chat/completions", &Method::POST, body)
            .await;
        assert_eq!(response.status(), 200);
        let headers = response.headers();
        assert!(!headers.contains_key("transfer-encoding"));
        assert!(!headers.contains_key("content-length"));
        assert!(!headers.contains_key("connection"));
        // The shared finalizer keeps the encoding when no bytes are rewritten,
        // even if logprobs were requested but Prefill supplied none.
        assert_eq!(headers["content-encoding"], "identity");
        assert_eq!(headers["x-backend"], "preserved");
        assert_eq!(headers.get_all("set-cookie").iter().count(), 2);
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap(),
            json!({"choices":[]})
        );
    }
}

#[tokio::test]
async fn direct_sequential_gzip_sse_keeps_encoding_and_bytes() {
    let f = Fixture::direct(Duration::from_secs(2)).await;
    let response = f
        .router
        .route_transparent(
            None,
            "/v1/chat/completions",
            &Method::POST,
            json!({"model":"compat-gzip", "stream":true, "messages":[]}),
        )
        .await;
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers().get("content-encoding").unwrap(), "gzip");
    assert_eq!(
        axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap()
            .as_ref(),
        GZIP_SSE
    );
}

fn counter(handle: &PrometheusHandle, name: &str, label: &str, value: &str) -> u64 {
    let prefix = format!("{name}{{{label}=\"{value}\"}}");
    handle
        .render()
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .map(|value| value.trim().parse().unwrap())
        .unwrap_or(0)
}

#[tokio::test]
async fn existing_phase_counters_and_request_timing_are_preserved() {
    // Each Tokio test uses one thread; a local recorder isolates these counters
    // from other tests without changing process-global metrics state.
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _guard = metrics::set_default_local_recorder(&recorder);
    for discovery in [false, true] {
        for concurrent in [false, true] {
            for (model, status) in [
                ("d-timeout", 504),
                ("d-503", 503),
                ("p-503", 503),
                ("test", 200),
            ] {
                let f = Fixture::new(Duration::from_millis(300), discovery).await;
                if concurrent {
                    f.warmup().await;
                }
                let path = format!("/compat/{discovery}/{concurrent}/{model}");
                let p = &f._p.url;
                let d = &f._d.url;
                let p_success = if concurrent {
                    f._p.url.trim_start_matches("http://")
                } else {
                    p
                };
                let d_success = if concurrent {
                    f._d.url.trim_start_matches("http://")
                } else {
                    d
                };
                let before_p = counter(
                    &handle,
                    "vllm_router_pd_prefill_requests_total",
                    "worker",
                    p,
                );
                let before_d =
                    counter(&handle, "vllm_router_pd_decode_requests_total", "worker", d);
                let before_p_success = counter(
                    &handle,
                    "vllm_router_pd_prefill_requests_total",
                    "worker",
                    p_success,
                );
                let before_d_success = counter(
                    &handle,
                    "vllm_router_pd_decode_requests_total",
                    "worker",
                    d_success,
                );
                let response = f
                    .router
                    .route_transparent(
                        None,
                        &path,
                        &Method::POST,
                        json!({"model":model, "messages":[]}),
                    )
                    .await;
                assert_eq!(response.status(), status);
                assert_eq!(
                    counter(&handle, "vllm_router_pd_requests_total", "route", &path),
                    1
                );
                if matches!(model, "test" | "d-503") {
                    assert_eq!(
                        counter(
                            &handle,
                            "vllm_router_pd_prefill_requests_total",
                            "worker",
                            p_success
                        ),
                        before_p_success + 1
                    );
                    assert_eq!(
                        counter(
                            &handle,
                            "vllm_router_pd_decode_requests_total",
                            "worker",
                            d_success
                        ),
                        before_d_success + 1
                    );
                } else {
                    let p_increment = u64::from(model == "d-timeout");
                    assert_eq!(
                        counter(
                            &handle,
                            "vllm_router_pd_prefill_requests_total",
                            "worker",
                            p
                        ),
                        before_p + p_increment,
                        "discovery={discovery}, concurrent={concurrent}, model={model}"
                    );
                    assert_eq!(
                        counter(&handle, "vllm_router_pd_decode_requests_total", "worker", d),
                        before_d
                    );
                }
                axum::body::to_bytes(response.into_body(), 4096)
                    .await
                    .unwrap();
            }
        }
    }

    let f = Fixture::direct(Duration::from_secs(2)).await;
    let path = "/compat/body-wait";
    let request = f.router.route_transparent(
        None,
        path,
        &Method::POST,
        json!({"model":"compat-body-wait", "messages":[]}),
    );
    tokio::pin!(request);
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if counter(&handle, "vllm_router_pd_requests_total", "route", path) == 1 {
                break;
            }
            tokio::select! {
                _ = &mut request => panic!("non-streaming body must remain buffered until EOF"),
                _ = tokio::task::yield_now() => {}
            }
        }
    })
    .await
    .expect("request metrics must be recorded at Decode headers");
    assert!(
        tokio::time::timeout(Duration::from_millis(40), &mut request)
            .await
            .is_err()
    );
    assert_eq!(
        counter(&handle, "vllm_router_pd_requests_total", "route", path),
        1
    );
    f.control.stream_release.notify_one();
    let response = request.await;
    assert_eq!(response.status(), 200);
    axum::body::to_bytes(response.into_body(), 4096)
        .await
        .unwrap();
    f.assert_no_load();

    let invalid_path = "/compat/invalid-input";
    assert_eq!(
        f.router
            .route_transparent(None, invalid_path, &Method::POST, json!(null))
            .await
            .status(),
        400
    );
    assert_eq!(
        counter(
            &handle,
            "vllm_router_pd_requests_total",
            "route",
            invalid_path
        ),
        0
    );

    for concurrent in [false, true] {
        let f = Fixture::direct(Duration::from_secs(2)).await;
        if concurrent {
            f.warmup().await;
        }
        f.control
            .p_wait
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let path = format!("/compat/cancel/{concurrent}");
        {
            let request = f.router.route_transparent(
                None,
                &path,
                &Method::POST,
                json!({"model":"test", "messages":[]}),
            );
            tokio::pin!(request);
            tokio::select! {
                _ = f.control.p_seen.notified() => {},
                _ = &mut request => panic!("P must remain blocked until the request is dropped"),
            }
        }
        assert_eq!(
            counter(&handle, "vllm_router_pd_requests_total", "route", &path),
            0
        );
        f.assert_no_load();
        f.control.p_release.notify_one();
    }
}
