//! Real HTTP regression coverage for the vLLM P/D request error contract.
#![cfg(all(feature = "cpu", feature = "ready"))]

mod common;
use common::pd_mock::{assert_error, Fixture};
use serde_json::{json, Value};
use std::{sync::atomic::Ordering, time::Duration};

#[tokio::test]
async fn real_timeout_and_connection_refusal() {
    let f = Fixture::direct(Duration::from_millis(150)).await;
    assert_error(
        f.request("/v1/chat/completions", "p-timeout", false).await,
        504,
        "prefill",
    )
    .await;
    f._p.task.abort();
    while !f._p.task.is_finished() {
        tokio::task::yield_now().await;
    }
    assert_error(
        f.request("/v1/chat/completions", "test", false).await,
        502,
        "prefill",
    )
    .await;
}

#[tokio::test]
async fn all_topologies_endpoints_and_statuses() {
    for discovery in [false, true] {
        for concurrent in [false, true] {
            for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
                // A fresh identity cache is required to genuinely exercise sequential D errors.
                for phase in ["p", "d"] {
                    for code in [400, 422, 429, 500, 503] {
                        let f = Fixture::new(Duration::from_secs(2), discovery).await;
                        if concurrent {
                            f.warmup().await;
                        }
                        let response = f.request(path, &format!("{phase}-{code}"), true).await;
                        assert_error(
                            response,
                            code,
                            if phase == "p" { "prefill" } else { "decode" },
                        )
                        .await;
                        if phase == "p" && !concurrent {
                            assert_eq!(f.d_calls.load(Ordering::SeqCst), 0);
                        }
                        f.assert_no_load();
                    }
                }
            }
        }
    }
}

#[tokio::test]
async fn prefill_gates_real_http_headers_and_wins_over_decode() {
    for discovery in [false, true] {
        for (p_status, d_status) in [(422, 0), (503, 429), (0, 429)] {
            let f = Fixture::new(Duration::from_secs(2), discovery).await;
            f.warmup().await; // NIXL push identity activates the real concurrent branch.
            f.control.p_wait.store(true, Ordering::SeqCst);
            f.control.p_status.store(p_status, Ordering::SeqCst);
            f.control.d_status.store(d_status, Ordering::SeqCst);
            let model = if d_status == 0 {
                "stream-reset"
            } else {
                "stream-error"
            };
            let request = f.request("/v1/chat/completions", model, false);
            tokio::pin!(request);
            tokio::select! {
                response = &mut request => panic!("headers committed before P finished: {}", response.status()),
                _ = f.control.d_seen.notified() => {}
            }
            assert!(
                tokio::time::timeout(Duration::from_millis(40), &mut request)
                    .await
                    .is_err(),
                "D completed but P is still unresolved"
            );
            f.control.p_release.notify_one();
            let response = tokio::time::timeout(Duration::from_secs(1), request)
                .await
                .unwrap();
            assert_error(
                response,
                if p_status == 0 { d_status } else { p_status } as u16,
                if p_status == 0 { "decode" } else { "prefill" },
            )
            .await;
            f.assert_no_load();
        }
    }
}

#[tokio::test]
async fn prefill_failure_cancels_pending_decode_and_body_failure_is_not_success() {
    for discovery in [false, true] {
        let f = Fixture::new(Duration::from_secs(2), discovery).await;
        f.warmup().await;
        f.control.d_wait.store(true, Ordering::SeqCst);
        f.control.p_status.store(503, Ordering::SeqCst);
        let request = f.request("/v1/chat/completions", "test", false);
        let response = tokio::time::timeout(Duration::from_millis(500), request)
            .await
            .expect("must finish while D remains blocked");
        assert_error(response, 503, "prefill").await;
        f.assert_no_load();
        f.control.d_release.notify_one();
    }
}

#[tokio::test]
async fn error_body_sizes_do_not_change_safety_or_status() {
    let f = Fixture::direct(Duration::from_secs(2)).await;
    f.warmup().await;
    for phase in ["p", "d"] {
        for size in [0, 1, 65535, 65536, 65537, 1_000_000] {
            if phase == "p" {
                f.control.p_status.store(503, Ordering::SeqCst);
            } else {
                f.control.p_status.store(0, Ordering::SeqCst);
                f.control.d_status.store(429, Ordering::SeqCst);
            }
            assert_error(
                f.request("/v1/chat/completions", &format!("body-{size}"), false)
                    .await,
                if phase == "p" { 503 } else { 429 },
                if phase == "p" { "prefill" } else { "decode" },
            )
            .await;
            f.assert_no_load();
        }
    }
}

#[tokio::test]
async fn required_prefill_protocol_is_checked_and_optional_push_body_is_not_required() {
    for discovery in [false, true] {
        let f = Fixture::new(Duration::from_secs(2), discovery).await;
        let response = f
            .request("/v1/chat/completions", "p-invalid-json", false)
            .await;
        assert_eq!(response.status(), 500);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(f.d_calls.load(Ordering::SeqCst), 0);
        f.assert_no_load();
        f.warmup().await;
        let response = f
            .request("/v1/chat/completions", "p-invalid-json", false)
            .await;
        assert_eq!(
            response.status(),
            200,
            "cached push identity needs no new P JSON"
        );
        response.bytes().await.unwrap();
        f.assert_no_load();
    }
}

#[tokio::test]
async fn concurrent_prefill_success_body_timeout_is_a_504() {
    for discovery in [false, true] {
        let f = Fixture::new(Duration::from_millis(150), discovery).await;
        f.warmup().await;
        assert_error(
            f.request("/v1/chat/completions", "p-body-timeout", false)
                .await,
            504,
            "prefill",
        )
        .await;
        f.assert_no_load();
    }
}

#[tokio::test]
async fn decode_body_timeout_during_usage_reconciliation_is_a_504() {
    for discovery in [false, true] {
        for concurrent in [false, true] {
            for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
                let f = Fixture::new(Duration::from_millis(150), discovery).await;
                if concurrent {
                    f.warmup().await;
                }
                assert_error(
                    f.request(path, "d-body-timeout", false).await,
                    504,
                    "decode",
                )
                .await;
                f.assert_no_load();
            }
        }
    }
}

#[tokio::test]
async fn successful_logprobs_still_merge() {
    for discovery in [false, true] {
        for concurrent in [false, true] {
            let f = Fixture::new(Duration::from_secs(2), discovery).await;
            if concurrent {
                f.warmup().await;
            }
            let response = f
                .request("/v1/chat/completions", "success-logprobs", true)
                .await;
            assert_eq!(response.status(), 200);
            let body: Value = response.json().await.unwrap();
            assert_eq!(
                body["prompt_logprobs"],
                json!([null, {"1":{"logprob":-0.5}}])
            );
            f.assert_no_load();
        }
    }
}

#[tokio::test]
async fn committed_stream_reset_terminates_without_done() {
    for discovery in [false, true] {
        let f = Fixture::new(Duration::from_secs(2), discovery).await;
        f.warmup().await;
        let mut response = f
            .request("/v1/chat/completions", "stream-reset", false)
            .await;
        assert_eq!(response.status(), 200);
        let first = response.chunk().await.unwrap().unwrap();
        assert_eq!(first, "data: partial\n\n");
        f.control.stream_release.notify_one();
        assert!(
            response.chunk().await.is_err(),
            "reset must be an observable failed stream"
        );
        f.assert_no_load();
    }
}

#[tokio::test]
async fn dropping_request_future_releases_each_worker_once() {
    use http::Method;
    for concurrent in [false, true] {
        let f = Fixture::direct(Duration::from_secs(2)).await;
        if concurrent {
            f.warmup().await;
        }
        f.control.p_wait.store(true, Ordering::SeqCst);
        f.control.d_wait.store(true, Ordering::SeqCst);
        let router = f.router.clone();
        let task = tokio::spawn(async move {
            router.route_transparent(None, "/v1/chat/completions", &Method::POST,
                json!({"model":"test", "messages":[{"role":"user", "content":"test"}], "max_tokens":16})).await
        });
        f.control.p_seen.notified().await;
        if concurrent {
            f.control.d_seen.notified().await;
        }
        let workers = f.context.worker_registry.get_all();
        assert_eq!(
            workers.iter().map(|w| w.load()).sum::<usize>(),
            2,
            "Prefill and the selected Decode are both reserved while Prefill is pending"
        );
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        f.assert_no_load();
        f.control.p_release.notify_one();
        f.control.d_release.notify_one();
    }
}

#[tokio::test]
async fn client_disconnect_before_headers_releases_load() {
    use tokio::io::AsyncWriteExt;
    for concurrent in [false, true] {
        let f = Fixture::direct(Duration::from_secs(3)).await;
        if concurrent {
            f.warmup().await;
        }
        f.control.p_wait.store(true, Ordering::SeqCst);
        f.control.d_wait.store(true, Ordering::SeqCst);
        let address = f.gateway.url.trim_start_matches("http://");
        let mut socket = tokio::net::TcpStream::connect(address).await.unwrap();
        let body = json!({"model":"test", "messages":[{"role":"user", "content":"test"}], "max_tokens":16}).to_string();
        socket.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
        tokio::time::timeout(Duration::from_secs(1), f.control.p_seen.notified())
            .await
            .unwrap();
        if concurrent {
            tokio::time::timeout(Duration::from_secs(1), f.control.d_seen.notified())
                .await
                .unwrap();
        }
        drop(socket);
        tokio::time::timeout(Duration::from_secs(1), async {
            while f
                .context
                .worker_registry
                .get_all()
                .iter()
                .any(|w| w.load() != 0)
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("disconnect must drop the local P/D request future");
        f.assert_no_load();
    }
}

#[tokio::test]
async fn received_http_error_survives_body_timeout_at_router_boundary() {
    for phase in ["p", "d"] {
        let f = Fixture::direct(Duration::from_millis(150)).await;
        f.warmup().await;
        assert_error(
            f.request(
                "/v1/chat/completions",
                &format!("{phase}-error-body-timeout"),
                false,
            )
            .await,
            if phase == "p" { 503 } else { 429 },
            if phase == "p" { "prefill" } else { "decode" },
        )
        .await;
        f.assert_no_load();
    }
}
