//! Run the process-global OTel toggle and scoped subscribers in isolation.
mod common;
use common::pd_mock::{assert_error, Fixture};
use std::{sync::Arc, time::Duration};

#[tokio::test]
async fn logs_and_otel_use_safe_diagnostics_at_info_and_debug() {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_sdk::{testing::trace::InMemorySpanExporter, trace::TracerProvider};
    use tracing_subscriber::{layer::SubscriberExt, Layer};
    #[derive(Clone)]
    struct Writer(Arc<std::sync::Mutex<Vec<u8>>>);
    impl std::io::Write for Writer {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut outputs = Vec::new();
    for level in [tracing::Level::INFO, tracing::Level::DEBUG] {
        let output = Arc::new(std::sync::Mutex::new(Vec::new()));
        let writer_output = output.clone();
        let exporter = InMemorySpanExporter::default();
        let provider = TracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let subscriber = tracing_subscriber::registry()
            .with(
                tracing_subscriber::fmt::layer()
                    .without_time()
                    .with_ansi(false)
                    .with_writer(move || Writer(writer_output.clone()))
                    .with_filter(tracing_subscriber::filter::LevelFilter::from_level(level)),
            )
            .with(
                tracing_opentelemetry::layer()
                    .with_tracer(provider.tracer("pd-errors"))
                    .with_filter(tracing_subscriber::filter::filter_fn(|metadata| {
                        // Match the production CustomOtelFilter: ordinary logs
                        // must not become span events, including DEBUG previews.
                        metadata.target().starts_with("otel_trace")
                    })),
            );
        let _guard = tracing::subscriber::set_default(subscriber);
        vllm_router_rs::otel_trace::mark_otel_enabled();
        let f = Fixture::direct(Duration::from_millis(150)).await;
        let response = f.request("/v1/chat/completions", "p-422", false).await;
        outputs.push(response.text().await.unwrap());
        assert_eq!(
            f.request("/v1/chat/completions", "p-preview", false)
                .await
                .status(),
            503
        );
        assert_eq!(
            f.request("/v1/chat/completions", "p-error-body-timeout", false)
                .await
                .status(),
            503
        );
        f.control
            .p_status
            .store(503, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            f.request("/v1/chat/completions", "body-65537", false)
                .await
                .status(),
            503
        );
        f.control
            .p_status
            .store(0, std::sync::atomic::Ordering::SeqCst);
        assert_error(
            f.request("/v1/chat/completions", "p-timeout", false).await,
            504,
            "prefill",
        )
        .await;
        let success = f
            .request("/v1/chat/completions", "success-logprobs", true)
            .await;
        assert_eq!(success.status(), 200);
        success.bytes().await.unwrap();
        let configured = Fixture::moriio_without_mode(Duration::from_secs(2)).await;
        let response = configured
            .request("/v1/chat/completions", "test", false)
            .await;
        assert_eq!(response.status(), 500);
        let public = response.text().await.unwrap();
        assert!(!public.contains("MoriIoTransferModeMissing"));
        assert!(public.contains("The router could not complete the request."));
        let client = reqwest::Client::new();
        let describe: fn(&reqwest::Error) -> &'static str = |_| "explicit policy";
        let mut policy_expectations = Vec::new();
        for (phase, callback) in [("renamed-stage", Some(describe)), ("prefill", None)] {
            let mut prepared = vllm_router_rs::otel_http::prepare_client_request(
                client.get("invalid URL"),
                None,
                vllm_router_rs::otel_http::ClientRequestOptions {
                    method: "GET",
                    url: &f._p.url,
                    route: Some("/policy-check"),
                    request_phase: Some(phase),
                },
            );
            if let Some(callback) = callback {
                prepared = prepared.with_error_description(callback);
            }
            let error = prepared.send().await.unwrap_err();
            policy_expectations.push((
                phase,
                callback
                    .map(|f| f(&error).to_owned())
                    .unwrap_or_else(|| error.to_string()),
            ));
        }
        for result in provider.force_flush() {
            result.unwrap();
        }
        let spans = exporter.get_finished_spans().unwrap();
        for (phase, message) in policy_expectations {
            assert!(spans.iter().any(|span| {
                let has = |key: &str, value: &str| {
                    span.attributes
                        .iter()
                        .any(|kv| kv.key.as_str() == key && kv.value.to_string() == value)
                };
                has("http.route", "/policy-check")
                    && has("vllm.request_phase", phase)
                    && has("error", &message)
            }));
        }
        assert!(spans.iter().any(|s| s.attributes.iter().any(
            |kv| kv.key.as_str() == "error" && kv.value.to_string() == "P/D upstream timeout"
        )));
        let logs = String::from_utf8(output.lock().unwrap().clone()).unwrap();
        let span_diagnostics = format!("{spans:?}");
        let diagnostics = format!("{logs} {span_diagnostics}");
        assert!(
            diagnostics.contains("MoriIoTransferModeMissing"),
            "controlled internal reason must remain visible to operators"
        );
        assert!(diagnostics.contains("📥 Decode response status:"));
        for restored in [
            "Serialized chat request:",
            "📤 Prefill request payload:",
            "📥 Prefill response body content:",
            "📤 Decode request payload:",
            "Successfully merged logprobs",
        ] {
            assert_eq!(
                // OTel sampling/filtering is independent of the fmt layer's
                // log level; only the formatted log output is level-filtered.
                logs.contains(restored),
                level == tracing::Level::DEBUG,
                "restored log: {restored}"
            );
        }
        for marker in [
            "/private/secret.py",
            "worker-secret.internal",
            "secret-marker",
            "Traceback",
        ] {
            assert!(!span_diagnostics.contains(marker), "span leaked {marker}");
            assert_eq!(
                logs.contains(marker),
                level == tracing::Level::DEBUG,
                "preview level for {marker}"
            );
        }
        assert_eq!(
            logs.matches("Upstream error body preview").count(),
            if level == tracing::Level::DEBUG { 2 } else { 0 }
        );
        assert_eq!(
            logs.contains(&"界".repeat(256)),
            level == tracing::Level::DEBUG
        );
        for marker in ["preview-tail-marker", "incomplete-body-marker"] {
            assert!(
                !diagnostics.contains(marker),
                "unexpected preview: {marker}"
            );
        }
    }
    assert_eq!(
        outputs[0], outputs[1],
        "log level must not affect public error body"
    );
}
