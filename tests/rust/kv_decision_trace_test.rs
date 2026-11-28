#![cfg(all(feature = "cpu", feature = "ready"))]

//! CPU-only observations of real router dispatches, using local HTTP/gRPC workers.

mod common;
#[path = "common/grpc_frontend_fixture.rs"]
mod fixture;

use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use axum::{
    body::Body,
    extract::ConnectInfo,
    http::{header, HeaderMap, HeaderValue, Method, Request, StatusCode, Uri},
    response::{IntoResponse, Response},
    Json, Router,
};
use common::mock_vllm_rs::{MockVllmRs, MockVllmRsServer};
use opentelemetry::{
    trace::{SpanKind, TraceId, TracerProvider as _},
    Value as OtelValue,
};
use opentelemetry_sdk::{
    export::trace::SpanData, testing::trace::InMemorySpanExporter, trace::TracerProvider,
};
use serde_json::{json, Value};
use tower::ServiceExt;
use tower_http::trace::MakeSpan;
use tracing::Instrument;
use tracing_subscriber::layer::SubscriberExt;
use vllm_router_rs::{
    config::{KvConnector, PolicyConfig, RouterConfig, RoutingMode},
    middleware::RequestSpan,
    otel_http::{self, ClientRequestOptions},
    otel_trace::{self, PreparedOtel},
    routers::{RouterFactory, RouterTrait},
};

type HttpCalls = Arc<Mutex<Vec<(Value, String)>>>;

/// This external-crate test intentionally uses only public data and observation
/// APIs. It must not need an unrelated built-in policy to obtain a trace handle.
#[test]
fn external_selection_can_attach_its_explicit_decision_receipt() {
    use vllm_router_rs::kv_decision::{
        KvDecisionOutcome, KvDecisionPolicy, KvDecisionReceipt, KvDecisionSourceState,
        KvDecisionStage, KvDecisionTrace,
    };
    use vllm_router_rs::kv_index::RoutableTarget;
    use vllm_router_rs::policies::{CostComponents, CostDecision, KvSelectionReceipt};

    // Facts supplied by this custom selection fixture, not a backend-reuse claim.
    let input_tokens = 12_u64;
    let candidates = [(2_u64, 3_usize), (8_u64, 1_usize)];
    let (worker_index, &(credited, active)) = candidates
        .iter()
        .enumerate()
        .min_by_key(|(_, (credit, load))| (input_tokens - credit) * (*load as u64 + 1))
        .unwrap();
    let new_prefill_tokens = input_tokens - credited;
    let facts = KvDecisionReceipt {
        policy: KvDecisionPolicy::KvAware,
        stage: KvDecisionStage::Regular,
        outcome: KvDecisionOutcome::PrefixCost,
        credited_tokens: Some(credited),
        new_prefill_tokens: Some(new_prefill_tokens),
        limiting_group: None,
        target_index: worker_index,
        source_state: KvDecisionSourceState::Unavailable,
        overhead_micros: 17,
    };
    let trace = KvDecisionTrace::new(facts);
    let selection = KvSelectionReceipt {
        worker_index,
        new_prefill_tokens,
        decision: CostDecision {
            target: RoutableTarget {
                instance_id: "<TEST_ENGINE_EXTERNAL_SELECTION>".into(),
                dp_rank: 0,
            },
            cost: new_prefill_tokens as f64 * (active + 1) as f64,
            components: CostComponents {
                ptoken: new_prefill_tokens,
                batchsize: active + 1,
            },
        },
        slot: None,
        trace,
    };
    assert_eq!(selection.worker_index, 1);
    assert_eq!(selection.trace.receipt, facts);
    assert_eq!(selection.new_prefill_tokens, 4);
    assert!(selection.slot.is_none());
}

struct VerifiedEmpty(AtomicUsize);
impl vllm_router_rs::kv_index::TieredMatchProvider for VerifiedEmpty {
    fn observed_matches_with_cache<'a>(
        &self,
        _: &'a [u32],
        _: u32,
        _: Option<&str>,
        _: u32,
        _: &mut vllm_router_rs::kv_index::RequestHashCache<'a>,
    ) -> vllm_router_rs::kv_index::ObservedMatches {
        self.0.fetch_add(1, Ordering::Relaxed);
        vllm_router_rs::kv_index::ObservedMatches {
            matches: Vec::new(),
            available: true,
        }
    }
}

struct HttpWorker {
    url: String,
    calls: HttpCalls,
    connections: Arc<Mutex<Vec<(String, SocketAddr)>>>,
    task: tokio::task::JoinHandle<()>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HttpPreparation {
    Direct,
    Render,
    TokenizeFallback,
}

impl Drop for HttpWorker {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl HttpWorker {
    async fn spawn(prefill: bool, push: bool, fail: bool) -> Self {
        Self::spawn_with_preparation(prefill, push, fail, HttpPreparation::Direct).await
    }

    async fn spawn_with_preparation(
        prefill: bool,
        push: bool,
        fail: bool,
        preparation: HttpPreparation,
    ) -> Self {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let recorded = calls.clone();
        let connections = Arc::new(Mutex::new(Vec::new()));
        let observed = connections.clone();
        let app = Router::new().fallback(move |ConnectInfo(peer): ConnectInfo<SocketAddr>, uri: Uri, headers: HeaderMap, body: Body| {
            let calls = recorded.clone();
            let connections = observed.clone();
            async move {
                let generation = uri.path().ends_with("/v1/completions");
                let preprocessing = uri.path().ends_with("/render") || uri.path().ends_with("/tokenize");
                connections.lock().unwrap().push((uri.path().to_owned(), peer));
                let mut response = async move {
                if uri.path().ends_with("/health") { return "ok".into_response(); }
                if uri.path().ends_with("/get_model_info") {
                    return Json(json!({"model_path":"trace-model"})).into_response();
                }
                if uri.path().ends_with("/v1/models") {
                    return Json(json!({"data":[{"id":"trace-model"}]})).into_response();
                }
                if uri.path().ends_with("/render") || uri.path().ends_with("/tokenize") {
                    if uri.path().ends_with("/render") && preparation == HttpPreparation::TokenizeFallback {
                        return StatusCode::NOT_FOUND.into_response();
                    }
                    return Json(json!({"token_ids":[1,2,3,4],"tokens":[1,2,3,4],"count":4})).into_response();
                }
                if !uri.path().ends_with("/v1/completions") { return StatusCode::NOT_FOUND.into_response(); }
                let bytes = axum::body::to_bytes(body, 1024 * 1024).await.unwrap();
                let request: Value = serde_json::from_slice(&bytes).unwrap();
                calls.lock().unwrap().push((request, headers.get("traceparent")
                    .and_then(|h| h.to_str().ok()).unwrap_or_default().to_owned()));
                if fail { return StatusCode::INTERNAL_SERVER_ERROR.into_response(); }
                let mut response = json!({"choices":[{"text":"ok","index":0,"finish_reason":"stop"}],
                    "usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}});
                if prefill {
                    response["kv_transfer_params"] = json!({"remote_engine_id":"P",
                        "remote_block_ids":[[1]],"do_remote_prefill":true,"do_remote_decode":false});
                    if push {
                        response["kv_transfer_params"]["remote_host"] = json!("127.0.0.1");
                        response["kv_transfer_params"]["remote_port"] = json!(9100);
                        response["kv_transfer_params"]["transfer_mode"] = json!("push");
                    }
                }
                Json(response).into_response()
                }.await;
                // Discovery and preprocessing must not warm a pooled connection
                // for the first generation. Generation itself stays keepalive.
                if !generation {
                    let connection = if preprocessing && preparation != HttpPreparation::Direct {
                        "keep-alive"
                    } else {
                        "close"
                    };
                    response.headers_mut().insert(header::CONNECTION, HeaderValue::from_static(connection));
                }
                response
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .await
            .unwrap();
        });
        Self {
            url,
            calls,
            connections,
            task,
        }
    }

    fn assert_first_generation_uses_a_cold_connection(&self) {
        let connections = self.connections.lock().unwrap();
        let generation = connections
            .iter()
            .position(|(path, _)| path.ends_with("/v1/completions"))
            .expect("a generation request");
        assert!(generation > 0, "startup requests must be observed");
        let (_, peer) = &connections[generation];
        assert!(
            connections[..generation]
                .iter()
                .all(|(_, previous)| previous != peer),
            "generation must not reuse a startup connection: {connections:?}"
        );
        assert!(
            connections[..generation]
                .iter()
                .all(|(path, _)| !path.ends_with("/render") && !path.ends_with("/tokenize")),
            "the token-ID fixture must not warm a preprocessing connection"
        );
    }

    fn assert_preparation_uses_a_cold_kept_connection(&self, tokenize_fallback: bool) {
        let connections = self.connections.lock().unwrap();
        let render = connections
            .iter()
            .position(|(path, _)| path.ends_with("/render"))
            .expect("a real render request");
        let generation = connections
            .iter()
            .position(|(path, _)| path.ends_with("/v1/completions"))
            .expect("a generation request");
        assert!(render > 0 && render < generation);
        assert_eq!(
            connections[..generation]
                .iter()
                .filter(|(path, _)| path.ends_with("/render"))
                .count(),
            1
        );
        let peer = connections[render].1;
        assert!(connections[..render]
            .iter()
            .all(|(_, earlier)| *earlier != peer));
        // Both preprocessing responses explicitly stay keepalive. The client
        // may reuse that socket or choose another pooled connection for generation.
        let tokenize = connections[..generation]
            .iter()
            .position(|(path, _)| path.ends_with("/tokenize"));
        assert_eq!(
            connections[..generation]
                .iter()
                .filter(|(path, _)| path.ends_with("/tokenize"))
                .count(),
            if tokenize_fallback { 1 } else { 0 }
        );
        if tokenize_fallback {
            let tokenize = tokenize.expect("404 render falls back to tokenize");
            assert!(render < tokenize && tokenize < generation);
        } else {
            assert!(tokenize.is_none());
        }
    }
}

fn kv_policy() -> PolicyConfig {
    serde_json::from_value(json!({"type":"kv_aware"})).unwrap()
}

fn config(mode: RoutingMode) -> RouterConfig {
    RouterConfig {
        mode,
        policy: kv_policy(),
        kv_connector: KvConnector::Nixl,
        disable_retries: true,
        disable_circuit_breaker: true,
        worker_startup_timeout_secs: 3,
        ..Default::default()
    }
}

fn pd_mode(prefill: &str, decode: &str) -> RoutingMode {
    RoutingMode::VllmPrefillDecode {
        prefill_urls: vec![(prefill.into(), None)],
        decode_urls: vec![decode.into()],
        prefill_policy: None,
        decode_policy: None,
        discovery_address: None,
    }
}

fn payload(model: &str) -> Value {
    json!({"model":model,"prompt":[1,2,3,4],"add_special_tokens":false,
        "max_tokens":2,"stream":false})
}

fn traceparent(id: u64) -> String {
    format!("00-{id:032x}-1234567890abcdef-01")
}

async fn route(app: Router, body: Value, id: u64) -> Response {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .header("traceparent", traceparent(id))
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let (parts, body) = response.into_parts();
    let bytes = axum::body::to_bytes(body, 1024 * 1024).await.unwrap();
    tokio::task::yield_now().await;
    Response::from_parts(parts, Body::from(bytes))
}

fn attribute<'a>(span: &'a SpanData, key: &str) -> &'a OtelValue {
    &span
        .attributes
        .iter()
        .find(|kv| kv.key.as_str() == key)
        .unwrap_or_else(|| panic!("missing {key} on {}", span.name))
        .value
}

fn assert_numeric_export_boundaries(exporter: &InMemorySpanExporter) {
    use vllm_router_rs::kv_decision::{
        KvDecisionOutcome, KvDecisionPolicy, KvDecisionReceipt, KvDecisionSourceState,
        KvDecisionStage, KvDecisionTrace,
    };

    let export = |facts: KvDecisionReceipt| {
        let before = exporter.get_finished_spans().unwrap().len();
        let trace = KvDecisionTrace::new(facts);
        assert_eq!(trace.receipt, facts, "observation preserves unsigned facts");
        let json = serde_json::to_value(trace.receipt).unwrap();
        assert_eq!(json["credited_tokens"].as_u64(), facts.credited_tokens);
        assert_eq!(
            json["new_prefill_tokens"].as_u64(),
            facts.new_prefill_tokens
        );
        assert_eq!(
            json["overhead_micros"].as_u64(),
            Some(facts.overhead_micros)
        );
        drop(trace);
        let mut spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), before + 1);
        let span = spans.pop().unwrap();
        assert_eq!(span.name, "kv_route");
        span
    };
    let zero = KvDecisionReceipt {
        policy: KvDecisionPolicy::KvAware,
        stage: KvDecisionStage::Regular,
        outcome: KvDecisionOutcome::NoReusablePrefix,
        credited_tokens: Some(0),
        new_prefill_tokens: Some(0),
        limiting_group: Some(0),
        target_index: 0,
        source_state: KvDecisionSourceState::Unavailable,
        overhead_micros: 0,
    };
    let signed_max = u64::try_from(i64::MAX).unwrap();
    let (signed_target, signed_target_value) = match usize::try_from(signed_max) {
        Ok(index) => (index, OtelValue::I64(i64::MAX)),
        Err(_) => (0, OtelValue::I64(0)),
    };
    let first_unsigned = signed_max + 1;
    let (first_unsigned_target, first_unsigned_target_value) = match usize::try_from(first_unsigned)
    {
        Ok(index) => (index, OtelValue::String("9223372036854775808".into())),
        Err(_) => (0, OtelValue::I64(0)),
    };
    let max_target_value = match i64::try_from(usize::MAX) {
        Ok(index) => OtelValue::I64(index),
        Err(_) => OtelValue::String(usize::MAX.to_string().into()),
    };
    for (facts, numeric, target, group) in [
        (
            zero,
            OtelValue::I64(0),
            OtelValue::I64(0),
            OtelValue::I64(0),
        ),
        (
            KvDecisionReceipt {
                credited_tokens: Some(signed_max),
                new_prefill_tokens: Some(signed_max),
                limiting_group: Some(u32::MAX),
                target_index: signed_target,
                overhead_micros: signed_max,
                ..zero
            },
            OtelValue::I64(i64::MAX),
            signed_target_value,
            OtelValue::I64(i64::from(u32::MAX)),
        ),
        (
            KvDecisionReceipt {
                credited_tokens: Some(first_unsigned),
                new_prefill_tokens: Some(first_unsigned),
                limiting_group: Some(u32::MAX),
                target_index: first_unsigned_target,
                overhead_micros: first_unsigned,
                ..zero
            },
            OtelValue::String("9223372036854775808".into()),
            first_unsigned_target_value,
            OtelValue::I64(i64::from(u32::MAX)),
        ),
        (
            KvDecisionReceipt {
                credited_tokens: Some(u64::MAX),
                new_prefill_tokens: Some(u64::MAX),
                limiting_group: Some(u32::MAX),
                target_index: usize::MAX,
                overhead_micros: u64::MAX,
                ..zero
            },
            OtelValue::String("18446744073709551615".into()),
            max_target_value,
            OtelValue::I64(i64::from(u32::MAX)),
        ),
    ] {
        let span = export(facts);
        for field in ["credited_tokens", "new_prefill_tokens", "overhead_micros"] {
            assert_eq!(attribute(&span, field), &numeric, "{field}: {facts:?}");
        }
        assert_eq!(attribute(&span, "target_index"), &target);
        assert_eq!(attribute(&span, "limiting_group"), &group);
    }
    let unavailable = KvDecisionReceipt {
        outcome: KvDecisionOutcome::MissingTokens,
        credited_tokens: None,
        new_prefill_tokens: None,
        limiting_group: None,
        ..zero
    };
    let span = export(unavailable);
    for field in ["credited_tokens", "new_prefill_tokens", "limiting_group"] {
        assert_eq!(
            attribute(&span, field),
            &OtelValue::String("unavailable".into())
        );
    }
    assert_eq!(attribute(&span, "target_index"), &OtelValue::I64(0));
    assert_eq!(attribute(&span, "overhead_micros"), &OtelValue::I64(0));

    // An enabled global OTel switch can still have a filtered tracing span.
    let before = exporter.get_finished_spans().unwrap().len();
    tracing::subscriber::with_default(
        tracing_subscriber::registry().with(tracing_subscriber::filter::LevelFilter::OFF),
        || {
            let facts = KvDecisionReceipt {
                credited_tokens: Some(u64::MAX),
                new_prefill_tokens: Some(u64::MAX),
                target_index: usize::MAX,
                overhead_micros: u64::MAX,
                ..zero
            };
            assert_eq!(KvDecisionTrace::new(facts).receipt, facts);
        },
    );
    assert_eq!(exporter.get_finished_spans().unwrap().len(), before);
}

fn assert_decisions(exporter: &InMemorySpanExporter, id: u64, stages: &[&str]) -> Vec<SpanData> {
    let trace_id = TraceId::from_hex(&format!("{id:032x}")).unwrap();
    let finished = exporter.get_finished_spans().unwrap();
    // These are completed SpanData values, not live span handles. Keep one
    // snapshot for both assertions and failure diagnostics.
    let spans: Vec<_> = finished
        .iter()
        .filter(|span| span.span_context.trace_id() == trace_id)
        .cloned()
        .collect();
    let root = spans
        .iter()
        .find(|span| span.span_kind == SpanKind::Server)
        .unwrap_or_else(|| {
            let summarize = |spans: &[SpanData]| {
                spans
                    .iter()
                    .map(|span| {
                        let field = |key| {
                            span.attributes
                                .iter()
                                .find(|attribute| attribute.key.as_str() == key)
                                .map(|attribute| attribute.value.to_string())
                        };
                        (
                            span.name.to_string(),
                            format!("{:?}", span.span_kind),
                            span.span_context.trace_id().to_string(),
                            span.span_context.span_id().to_string(),
                            span.parent_span_id.to_string(),
                            field("stage"),
                            field("outcome"),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            panic!(
                "request span missing: id={id}, stages={stages:?}, otel_enabled={}, \
                 initial_finished_total={}, columns=(name,kind,trace_id,span_id,parent_id,stage,outcome), \
                 matching={:?}, all_finished={:?}",
                otel_trace::is_otel_enabled(),
                finished.len(),
                summarize(&spans),
                summarize(&finished),
            )
        });
    let routes: Vec<_> = spans
        .iter()
        .filter(|span| span.name == "kv_route")
        .collect();
    assert_eq!(
        routes.len(),
        stages.len(),
        "one receipt per actual selection: {spans:?}"
    );
    let fields: BTreeSet<_> = [
        "policy",
        "stage",
        "outcome",
        "credited_tokens",
        "new_prefill_tokens",
        "limiting_group",
        "target_index",
        "source_state",
        "overhead_micros",
    ]
    .into_iter()
    .collect();
    for stage in stages {
        let decision = routes
            .iter()
            .find(|span| attribute(span, "stage").to_string() == *stage)
            .expect("stage decision");
        assert_eq!(decision.parent_span_id, root.span_context.span_id());
        let application_attributes: Vec<_> = decision
            .attributes
            .iter()
            .map(|kv| kv.key.as_str())
            .filter(|key| {
                !key.starts_with("code.")
                    && !key.starts_with("thread.")
                    && !matches!(*key, "busy_ns" | "idle_ns")
            })
            .collect();
        for field in &fields {
            assert_eq!(
                application_attributes
                    .iter()
                    .filter(|key| *key == field)
                    .count(),
                1,
                "each application attribute occurs once: {field} on {decision:?}"
            );
        }
        let application_fields: BTreeSet<_> = application_attributes.into_iter().collect();
        assert_eq!(application_fields, fields);
        assert_eq!(attribute(decision, "policy").to_string(), "kv_aware");
        assert_eq!(attribute(decision, "target_index"), &OtelValue::I64(0));
        assert_eq!(
            attribute(decision, "source_state").to_string(),
            "unavailable"
        );
        assert_eq!(
            attribute(decision, "limiting_group").to_string(),
            "unavailable"
        );
        let dispatch_name = if *stage == "decode" {
            "decode_dispatch"
        } else {
            "prefill_dispatch"
        };
        let dispatches: Vec<_> = spans
            .iter()
            .filter(|span| {
                span.name == dispatch_name && span.parent_span_id == decision.span_context.span_id()
            })
            .collect();
        assert_eq!(
            dispatches.len(),
            1,
            "dispatch belongs to its selected stage"
        );
    }
    spans
}

fn assert_http_dispatch(spans: &[SpanData], worker: &HttpWorker, stage: &str) {
    let calls = worker.calls.lock().unwrap();
    let parent = &calls.last().unwrap().1;
    let backend_parent = parent.split('-').nth(2).expect("backend traceparent");
    let transport = spans
        .iter()
        .find(|span| span.span_context.span_id().to_string() == backend_parent)
        .expect("backend parent is the actual exported HTTP transport span");
    assert_eq!(transport.span_kind, SpanKind::Client);
    let dispatch = spans
        .iter()
        .find(|span| span.span_context.span_id() == transport.parent_span_id)
        .expect("transport dispatch parent");
    assert_eq!(
        dispatch.name,
        if stage == "decode" {
            "decode_dispatch"
        } else {
            "prefill_dispatch"
        }
    );
    let decision = spans
        .iter()
        .find(|span| span.span_context.span_id() == dispatch.parent_span_id)
        .expect("dispatch decision parent");
    assert_eq!(attribute(decision, "stage").to_string(), stage);
}

/// A real pending send must be cancelled when its owning caller is dropped.
async fn assert_http_cancellation_before_headers(exporter: &InMemorySpanExporter, id: u64) {
    use tokio::io::AsyncReadExt;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (received_tx, received_rx) = tokio::sync::oneshot::channel();
    let (closed_tx, closed_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut headers = Vec::new();
        let mut chunk = [0; 1024];
        while !headers.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
            let count = stream.read(&mut chunk).await.unwrap();
            assert!(count > 0, "the request headers must reach the worker");
            headers.extend_from_slice(&chunk[..count]);
            assert!(headers.len() < 16384);
        }
        received_tx
            .send(String::from_utf8(headers).unwrap())
            .unwrap();
        // Keep response headers blocked until the caller cancels. A detached
        // send would leave this read pending, rather than closing the socket.
        let closed = stream.read(&mut chunk).await;
        let _ = closed_tx.send(closed);
    });
    let incoming = Request::builder()
        .method("GET")
        .uri("/cancel-before-headers")
        .header("traceparent", traceparent(id))
        .body(Body::empty())
        .unwrap();
    let root = RequestSpan.make_span(&incoming);
    let client = reqwest::Client::new();
    let mut pending = Box::pin(
        otel_http::send_client_request(
            client.get(&url),
            None,
            ClientRequestOptions {
                method: "GET",
                url: &url,
                route: Some("/cancel-before-headers"),
                request_phase: None,
            },
        )
        .instrument(root),
    );
    let headers = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        tokio::select! {
            headers = received_rx => headers.unwrap(),
            response = &mut pending => panic!("worker deliberately withheld response headers: {response:?}"),
        }
    })
    .await
    .expect("the HTTP send must start");
    drop(pending);
    let bytes = tokio::time::timeout(std::time::Duration::from_secs(5), closed_rx)
        .await
        .expect("dropping the caller must cancel the HTTP send")
        .unwrap()
        .unwrap();
    assert_eq!(bytes, 0, "the worker must observe TCP EOF");
    server.await.unwrap();
    let spans = assert_decisions(exporter, id, &[]);
    let root = spans
        .iter()
        .find(|span| span.span_kind == SpanKind::Server)
        .unwrap();
    let client = spans
        .iter()
        .find(|span| span.span_kind == SpanKind::Client)
        .unwrap();
    assert_eq!(client.parent_span_id, root.span_context.span_id());
    let propagated = headers
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.eq_ignore_ascii_case("traceparent"))
        .unwrap()
        .1
        .trim();
    assert_eq!(
        propagated,
        format!(
            "00-{}-{}-01",
            client.span_context.trace_id(),
            client.span_context.span_id()
        )
    );
}

async fn assert_http_builder_error_is_preserved(exporter: &InMemorySpanExporter, id: u64) {
    let incoming = Request::builder()
        .uri("/invalid-upstream")
        .header("traceparent", traceparent(id))
        .body(Body::empty())
        .unwrap();
    let root = RequestSpan.make_span(&incoming);
    let error = otel_http::send_client_request(
        reqwest::Client::new().get("invalid-url"),
        None,
        ClientRequestOptions {
            method: "GET",
            url: "invalid-url",
            route: Some("/invalid-upstream"),
            request_phase: None,
        },
    )
    .instrument(root)
    .await
    .unwrap_err();
    assert!(error.is_builder());
    let spans = assert_decisions(exporter, id, &[]);
    let root = spans
        .iter()
        .find(|span| span.span_kind == SpanKind::Server)
        .unwrap();
    let client = spans
        .iter()
        .find(|span| span.span_kind == SpanKind::Client)
        .unwrap();
    assert_eq!(client.parent_span_id, root.span_context.span_id());
}

/// One current-thread test owns process-global OTel state for the whole matrix.
#[tokio::test(flavor = "current_thread")]
async fn kv_decisions_follow_real_http_and_grpc_dispatches() {
    let exporter = InMemorySpanExporter::default();
    let provider = TracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    let prepared = PreparedOtel::from_parts(provider.clone(), provider.tracer("kv-decision-test"));
    let subscriber = tracing_subscriber::registry().with(prepared.layer());
    let _subscriber = tracing::subscriber::set_default(subscriber);
    otel_trace::activate_otel(prepared);

    assert_numeric_export_boundaries(&exporter);

    let regular = HttpWorker::spawn(false, false, false).await;
    let cfg = config(RoutingMode::Regular {
        worker_urls: vec![regular.url.clone()],
    });
    let context = common::create_test_context(cfg.clone());
    let router: Arc<dyn RouterTrait> =
        Arc::from(RouterFactory::create_router(&context).await.unwrap());
    let app = common::test_app::create_test_app(router.clone(), reqwest::Client::new(), &cfg);
    assert_eq!(
        route(app.clone(), payload("trace-model"), 1).await.status(),
        StatusCode::OK
    );
    regular.assert_first_generation_uses_a_cold_connection();
    let spans = assert_decisions(&exporter, 1, &["regular"]);
    assert_http_dispatch(&spans, &regular, "regular");
    assert_eq!(regular.calls.lock().unwrap().len(), 1);
    assert!(regular.calls.lock().unwrap()[0]
        .1
        .starts_with(&format!("00-{:032x}-", 1)));

    // Exercise the separate transparent selector and dispatch path under real middleware context.
    let mut incoming = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("traceparent", traceparent(2))
        .body(Body::empty())
        .unwrap();
    let root = RequestSpan.make_span(&incoming);
    let headers = std::mem::take(incoming.headers_mut());
    let response = router
        .route_transparent(
            Some(&headers),
            "/v1/completions",
            &Method::POST,
            payload("trace-model"),
        )
        .instrument(root)
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let spans = assert_decisions(&exporter, 2, &["regular"]);
    assert_http_dispatch(&spans, &regular, "regular");

    // A verified registration must keep the explicit cache-read exclusion,
    // rather than blame identity after its qualification becomes false.
    let provider = Arc::new(VerifiedEmpty(AtomicUsize::new(0)));
    let policies = std::iter::once(context.policy_registry.get_default_policy())
        .chain(context.policy_registry.get_policy("trace-model"));
    for policy in policies {
        policy
            .as_any()
            .downcast_ref::<vllm_router_rs::policies::KvAwarePolicy>()
            .unwrap()
            .register_target_with_cache_identity(
                &regular.url,
                vllm_router_rs::kv_index::RoutableTarget {
                    instance_id: "<TEST_ENGINE_VERIFIED_REGISTRATION>".into(),
                    dp_rank: 0,
                },
                vllm_router_rs::kv_index::CacheKey {
                    model: "trace-model".into(),
                    block_size: 1,
                },
                provider.clone(),
            );
    }
    for (case, extra, expected) in [
        (
            0,
            json!({"skip_reading_prefix_cache":true}),
            "skip_prefix_cache",
        ),
        (
            1,
            json!({"extra_body":{"skip_reading_prefix_cache":true}}),
            "skip_prefix_cache",
        ),
        (
            2,
            json!({"sampling_params":{"skip_reading_prefix_cache":true}}),
            "skip_prefix_cache",
        ),
        (
            3,
            json!({"skip_reading_external_kv":true}),
            "skip_external_kv",
        ),
        (
            4,
            json!({"skip_reading_prefix_cache":"invalid"}),
            "unsupported_cache_read",
        ),
        (
            5,
            json!({"prompt_logprobs":1}),
            "unsupported_prompt_scoring",
        ),
    ] {
        let mut body = payload("trace-model");
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let id = 100 + case * 2;
        let app = common::test_app::create_test_app(router.clone(), reqwest::Client::new(), &cfg);
        assert_eq!(
            route(app.clone(), body.clone(), id).await.status(),
            StatusCode::OK
        );
        let spans = assert_decisions(&exporter, id, &["regular"]);
        let receipt = spans.iter().find(|span| span.name == "kv_route").unwrap();
        assert_eq!(attribute(receipt, "outcome").to_string(), expected);
        assert_eq!(
            attribute(receipt, "credited_tokens").to_string(),
            "unavailable"
        );
        assert_http_dispatch(&spans, &regular, "regular");

        let incoming = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("traceparent", traceparent(id + 1))
            .body(Body::empty())
            .unwrap();
        let root = RequestSpan.make_span(&incoming);
        let response = router
            .route_transparent(
                Some(incoming.headers()),
                "/v1/completions",
                &Method::POST,
                body,
            )
            .instrument(root)
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let spans = assert_decisions(&exporter, id + 1, &["regular"]);
        let receipt = spans.iter().find(|span| span.name == "kv_route").unwrap();
        assert_eq!(attribute(receipt, "outcome").to_string(), expected);
        assert_http_dispatch(&spans, &regular, "regular");
    }
    assert_eq!(
        provider.0.load(Ordering::Relaxed),
        0,
        "excluded reads must not query placement"
    );

    for (preparation, id) in [
        (HttpPreparation::Render, 200),
        (HttpPreparation::TokenizeFallback, 210),
    ] {
        let worker = HttpWorker::spawn_with_preparation(false, false, false, preparation).await;
        let cfg = config(RoutingMode::Regular {
            worker_urls: vec![worker.url.clone()],
        });
        let context = common::create_test_context(cfg.clone());
        let router = Arc::from(RouterFactory::create_router(&context).await.unwrap());
        let app = common::test_app::create_test_app(router, reqwest::Client::new(), &cfg);
        let mut body = payload("trace-model");
        body["prompt"] = json!("cold text");
        assert_eq!(route(app.clone(), body, id).await.status(), StatusCode::OK);
        worker.assert_preparation_uses_a_cold_kept_connection(
            preparation == HttpPreparation::TokenizeFallback,
        );
        let spans = assert_decisions(&exporter, id, &["regular"]);
        assert_http_dispatch(&spans, &worker, "regular");
        assert_eq!(worker.calls.lock().unwrap().len(), 1);
    }
    assert_http_cancellation_before_headers(&exporter, 300).await;
    assert_http_builder_error_is_preserved(&exporter, 302).await;

    for (push, first_id) in [(false, 10), (true, 20)] {
        let prefill = HttpWorker::spawn(true, push, false).await;
        let decode = HttpWorker::spawn(false, false, false).await;
        let cfg = config(pd_mode(&prefill.url, &decode.url));
        let context = common::create_test_context(cfg.clone());
        let router = Arc::from(RouterFactory::create_router(&context).await.unwrap());
        let app = common::test_app::create_test_app(router, reqwest::Client::new(), &cfg);
        // First request makes a deferred Decode commitment; push mode's second uses concurrent legs.
        for id in first_id..first_id + 2 {
            assert_eq!(
                route(app.clone(), payload("trace-model"), id)
                    .await
                    .status(),
                StatusCode::OK
            );
            let spans = assert_decisions(&exporter, id, &["prefill", "decode"]);
            assert_http_dispatch(&spans, &prefill, "prefill");
            assert_http_dispatch(&spans, &decode, "decode");
        }
        assert_eq!(prefill.calls.lock().unwrap().len(), 2);
        assert_eq!(decode.calls.lock().unwrap().len(), 2);
    }

    let model = fixture::model_fixture();
    let model_name = model.path().to_str().unwrap();
    let grpc = MockVllmRsServer::spawn().await;
    let cfg = config(RoutingMode::Regular {
        worker_urls: vec![grpc.grpc_url.clone()],
    });
    let context = common::create_test_context(cfg.clone());
    let router = Arc::from(RouterFactory::create_router(&context).await.unwrap());
    let app = common::test_app::create_test_app(router, reqwest::Client::new(), &cfg);
    assert_eq!(
        route(app.clone(), payload(model_name), 30).await.status(),
        StatusCode::OK
    );
    assert_decisions(&exporter, 30, &["regular"]);
    assert_eq!(grpc.captured().len(), 1);
    assert_eq!(grpc.captured()[0].token_ids, [1, 2, 3, 4]);
    assert_eq!(
        route(app.clone(), payload(model_name), 31).await.status(),
        StatusCode::OK
    );
    assert_decisions(&exporter, 31, &["regular"]);
    assert_eq!(grpc.captured().len(), 2);
    assert_eq!(grpc.captured()[1].token_ids, [1, 2, 3, 4]);

    for (push, first_id) in [(false, 40), (true, 50)] {
        let mut kvt = json!({"remote_engine_id":"P","remote_block_ids":[[1]],
            "do_remote_prefill":true,"do_remote_decode":false});
        if push {
            kvt["remote_host"] = json!("127.0.0.1");
            kvt["remote_port"] = json!(9100);
            kvt["transfer_mode"] = json!("push");
        }
        let p = MockVllmRsServer::spawn_with_state(MockVllmRs {
            finish_kv_transfer_params: Some(kvt),
            ..Default::default()
        })
        .await;
        let d = MockVllmRsServer::spawn().await;
        let cfg = config(pd_mode(&p.grpc_url, &d.grpc_url));
        let context = common::create_test_context(cfg.clone());
        let router = Arc::from(RouterFactory::create_router(&context).await.unwrap());
        let app = common::test_app::create_test_app(router, reqwest::Client::new(), &cfg);
        for id in first_id..first_id + 2 {
            assert_eq!(
                route(app.clone(), payload(model_name), id).await.status(),
                StatusCode::OK
            );
            assert_decisions(&exporter, id, &["prefill", "decode"]);
        }
        assert_eq!((p.captured().len(), d.captured().len()), (2, 2));
    }

    // A failed Prefill never dispatches Decode and therefore has no Decode decision.
    let p = HttpWorker::spawn(true, false, true).await;
    let d = HttpWorker::spawn(false, false, false).await;
    let cfg = config(pd_mode(&p.url, &d.url));
    let context = common::create_test_context(cfg.clone());
    let router = Arc::from(RouterFactory::create_router(&context).await.unwrap());
    let app = common::test_app::create_test_app(router, reqwest::Client::new(), &cfg);
    assert!(route(app.clone(), payload("trace-model"), 60)
        .await
        .status()
        .is_server_error());
    assert_decisions(&exporter, 60, &["prefill"]);
    assert!(d.calls.lock().unwrap().is_empty());

    otel_trace::shutdown_otel();

    let before = exporter.get_finished_spans().unwrap().len();
    let cfg = config(RoutingMode::Regular {
        worker_urls: vec![regular.url.clone()],
    });
    let context = common::create_test_context(cfg.clone());
    let router = Arc::from(RouterFactory::create_router(&context).await.unwrap());
    let app = common::test_app::create_test_app(router, reqwest::Client::new(), &cfg);
    assert_eq!(
        route(app.clone(), payload("trace-model"), 70)
            .await
            .status(),
        StatusCode::OK
    );
    assert_eq!(
        exporter.get_finished_spans().unwrap().len(),
        before,
        "disabled KV routing must create no observation spans"
    );
    assert_eq!(
        regular.calls.lock().unwrap().last().unwrap().1,
        traceparent(70),
        "disabled observation preserves passive context forwarding"
    );
}
