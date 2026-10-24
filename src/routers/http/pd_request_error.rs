//! Request-level P/D failures. Generic formatting omits upstream body data.
use axum::{
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::{Bytes, BytesMut};
use futures_util::{Stream, StreamExt};
use serde_json::{json, Value};
use std::fmt;

pub(super) const MAX_UPSTREAM_ERROR_BODY_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug)]
pub(super) enum PdPhase {
    Prefill,
    Decode,
}
impl fmt::Display for PdPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TransportErrorKind {
    Timeout,
    Connection,
    Other,
}
pub(super) fn classify_transport(error: &reqwest::Error) -> TransportErrorKind {
    if error.is_timeout() {
        TransportErrorKind::Timeout
    } else if error.is_connect() {
        TransportErrorKind::Connection
    } else {
        TransportErrorKind::Other
    }
}

/// Explicit policy supplied by the P/D caller to the generic HTTP tracer.
pub(super) fn pd_error_description(error: &reqwest::Error) -> &'static str {
    if error.is_builder() {
        return "P/D request construction failed";
    }
    match classify_transport(error) {
        TransportErrorKind::Timeout => "P/D upstream timeout",
        TransportErrorKind::Connection => "P/D upstream connection failed",
        TransportErrorKind::Other => "P/D upstream transport failed",
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ErrorBodyState {
    Complete,
    LimitExceeded,
    ReadFailed { kind: TransportErrorKind },
}
impl ErrorBodyState {
    fn label(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::LimitExceeded => "limit_exceeded",
            Self::ReadFailed { .. } => "read_failed",
        }
    }
}
pub(super) struct UpstreamHttpError {
    phase: PdPhase,
    status: StatusCode,
    headers: HeaderMap,
    body: Bytes,
    body_state: ErrorBodyState,
}
#[derive(Debug)]
pub(super) struct TransportError {
    phase: PdPhase,
    kind: TransportErrorKind,
}
#[derive(Debug)]
pub(super) enum InternalErrorReason {
    MoriIoTransferModeMissing,
    SerializePrefillRequest,
    SerializeDecodeRequest,
    SerializeRequest,
    SerializeResponse,
    BuildResponse,
    RequestBuilder { phase: PdPhase },
}
#[derive(Debug)]
pub(super) enum PdRequestError {
    UpstreamHttp(Box<UpstreamHttpError>),
    Transport(TransportError),
    Protocol { phase: PdPhase },
    Internal(InternalErrorReason),
}
impl PdRequestError {
    pub(super) fn from_reqwest(phase: PdPhase, error: reqwest::Error) -> Self {
        if error.is_builder() {
            Self::Internal(InternalErrorReason::RequestBuilder { phase })
        } else {
            Self::Transport(TransportError {
                phase,
                kind: classify_transport(&error),
            })
        }
    }
    pub(super) fn protocol(phase: PdPhase) -> Self {
        Self::Protocol { phase }
    }
    pub(super) fn phase(&self) -> Option<PdPhase> {
        match self {
            Self::UpstreamHttp(e) => Some(e.phase),
            Self::Transport(e) => Some(e.phase),
            Self::Protocol { phase }
            | Self::Internal(InternalErrorReason::RequestBuilder { phase }) => Some(*phase),
            Self::Internal(_) => None,
        }
    }
    fn status(&self) -> StatusCode {
        match self {
            Self::UpstreamHttp(error) => error.status,
            Self::Transport(TransportError {
                kind: TransportErrorKind::Timeout,
                ..
            }) => StatusCode::GATEWAY_TIMEOUT,
            Self::Transport(_) => StatusCode::BAD_GATEWAY,
            Self::Protocol { .. } | Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
    /// Internal classification and controlled fallback message. Public error
    /// types are mapped independently when rendering the response.
    fn public_fields(&self) -> (&'static str, String) {
        match self {
            Self::UpstreamHttp(e) => (
                "upstream_http_error",
                format!(
                    "Upstream {} request failed (HTTP {}).",
                    e.phase,
                    e.status.as_u16()
                ),
            ),
            Self::Transport(TransportError {
                phase,
                kind: TransportErrorKind::Timeout,
            }) => (
                "upstream_timeout",
                format!("Upstream {phase} request timed out."),
            ),
            Self::Transport(TransportError { phase, .. }) => (
                "upstream_transport_error",
                format!("Unable to communicate with the upstream {phase} service."),
            ),
            Self::Protocol { phase } => (
                "upstream_protocol_error",
                format!("Upstream {phase} response did not satisfy the required protocol."),
            ),
            Self::Internal(_) => (
                "internal_error",
                "The router could not complete the request.".into(),
            ),
        }
    }
}
impl fmt::Display for PdRequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.public_fields().1)
    }
}
// No raw source chain: generic error reporters must remain safe as well.
impl std::error::Error for PdRequestError {}
impl fmt::Debug for UpstreamHttpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UpstreamHttpError")
            .field("phase", &self.phase)
            .field("status", &self.status.as_u16())
            .field("saved_bytes", &self.body.len())
            .field("body_state", &self.body_state)
            .finish()
    }
}
/// Rendering records a warning and one returned-failure observation. Internal
/// metric classifications are independent of the public OpenAI error types.
impl IntoResponse for PdRequestError {
    fn into_response(self) -> Response {
        tracing::warn!(error = %self, details = ?self, "P/D request failed");
        tracing::debug!(details = ?self, "P/D error classification");
        let (kind, message) = self.public_fields();
        let phase = match &self {
            Self::UpstreamHttp(e) => e.phase.to_string(),
            Self::Transport(e) => e.phase.to_string(),
            Self::Protocol { phase } => phase.to_string(),
            Self::Internal(_) => "router".to_owned(),
        };
        metrics::counter!("vllm_router_pd_phase_errors_total", "phase" => phase, "error_type" => kind).increment(1);
        let status = self.status();
        let public_type = if status == StatusCode::TOO_MANY_REQUESTS {
            "rate_limit_error"
        } else if status.is_client_error() {
            "invalid_request_error"
        } else {
            "server_error"
        };
        let mut detail = json!({
            "message": message,
            "type": public_type,
            "code": status.as_u16().to_string(),
        });
        if let Self::UpstreamHttp(error) = &self {
            if let Some(upstream) = error.client_detail() {
                detail["message"] = json!(format!(
                    "{} {}",
                    message,
                    upstream["message"].as_str().unwrap()
                ));
                if let Some(param) = upstream.get("param").and_then(Value::as_str) {
                    detail["param"] = json!(param);
                }
                match upstream.get("code") {
                    Some(Value::String(code)) => detail["code"] = json!(code),
                    Some(Value::Number(code)) => detail["code"] = json!(code.to_string()),
                    _ => {}
                }
            }
        }
        let mut response = (status, Json(json!({"error": detail}))).into_response();
        if let Self::UpstreamHttp(error) = self {
            if let Some(value) = error.headers.get(header::RETRY_AFTER) {
                response
                    .headers_mut()
                    .insert(header::RETRY_AFTER, value.clone());
            }
        }
        response
    }
}

impl UpstreamHttpError {
    /// Retain actionable client diagnostics only for a complete JSON 4xx.
    /// Unknown fields never enter the public response or generic formatting.
    fn client_detail(&self) -> Option<Value> {
        if !self.status.is_client_error() || self.body_state != ErrorBodyState::Complete {
            return None;
        }
        let content_type = self.headers.get(header::CONTENT_TYPE)?.to_str().ok()?;
        if !content_type
            .split(';')
            .next()?
            .trim()
            .eq_ignore_ascii_case("application/json")
        {
            return None;
        }
        let body: Value = serde_json::from_slice(&self.body).ok()?;
        let error = body.get("error")?.as_object()?;
        error.get("message")?.as_str()?;
        Some(Value::Object(error.clone()))
    }
}

// Accept only delay-seconds or a round-tripped HTTP-date. chrono is already a
// direct dependency; no dependency/lockfile change is needed for this header.
fn valid_retry_after(value: &HeaderValue) -> bool {
    let Ok(text) = value.to_str() else {
        return false;
    };
    if !text.is_empty() && text.bytes().all(|c| c.is_ascii_digit()) {
        return true;
    }
    [
        "%a, %d %b %Y %H:%M:%S GMT",
        "%A, %d-%b-%y %H:%M:%S GMT",
        "%a %b %e %H:%M:%S %Y",
    ]
    .iter()
    .any(|format| {
        chrono::NaiveDateTime::parse_from_str(text, format)
            .is_ok_and(|date| date.format(format).to_string() == text)
    })
}

async fn capture_body<S>(stream: S) -> (Bytes, ErrorBodyState)
where
    S: Stream<Item = Result<Bytes, TransportErrorKind>>,
{
    futures_util::pin_mut!(stream);
    let mut body = BytesMut::new();
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(kind) => return (body.freeze(), ErrorBodyState::ReadFailed { kind }),
        };
        let remaining = MAX_UPSTREAM_ERROR_BODY_BYTES - body.len();
        // Copy only the saved prefix; never retain a giant chunk's allocation.
        body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
        if chunk.len() > remaining {
            return (body.freeze(), ErrorBodyState::LimitExceeded);
        }
    }
    (body.freeze(), ErrorBodyState::Complete)
}

pub(super) async fn capture_upstream_error(
    phase: PdPhase,
    response: reqwest::Response,
) -> PdRequestError {
    if response.status().is_success() {
        return PdRequestError::protocol(phase);
    }
    let status = response.status();
    let mut headers = HeaderMap::new();
    if let Some(value) = response.headers().get(header::CONTENT_TYPE) {
        headers.insert(header::CONTENT_TYPE, value.clone());
    }
    if let Some(value) = response
        .headers()
        .get(header::RETRY_AFTER)
        .filter(|v| valid_retry_after(v))
    {
        headers.insert(header::RETRY_AFTER, value.clone());
    }
    // reqwest's configured total timeout covers the body, including the wait
    // for EOF after exactly 64 KiB; do not reset that deadline per chunk.
    let (body, body_state) = capture_body(
        response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| classify_transport(&e))),
    )
    .await;
    metrics::counter!("vllm_router_pd_error_body_total", "phase" => phase.to_string(), "state" => body_state.label()).increment(1);
    let error = UpstreamHttpError {
        phase,
        status,
        headers,
        body,
        body_state,
    };
    if body_state == ErrorBodyState::Complete && tracing::enabled!(tracing::Level::DEBUG) {
        let preview: String = String::from_utf8_lossy(&error.body)
            .chars()
            .take(256)
            .collect();
        tracing::debug!(phase = %phase, status = status.as_u16(), preview = ?preview, "Upstream error body preview");
    }
    PdRequestError::UpstreamHttp(Box::new(error))
}

/// Stream failures occur after commitment: terminate without synthesizing SSE.
pub(super) fn decode_stream(
    response: reqwest::Response,
) -> impl Stream<Item = Result<Bytes, PdRequestError>> {
    response.bytes_stream().map(|chunk| {
        chunk.map_err(|e| {
            let error = PdRequestError::from_reqwest(PdPhase::Decode, e);
            let (kind, _) = error.public_fields();
            // Byte streaming is used for both stream=true and ordinary direct
            // responses. New diagnostics must not change legacy SSE counters.
            metrics::counter!("vllm_router_pd_response_body_errors_total", "phase" => "decode", "error_type" => kind).increment(1);
            tracing::warn!(error = %error, "P/D stream terminated");
            error
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        pin::Pin,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        task::{Context, Poll},
    };

    #[test]
    fn internal_diagnostics_are_specific_but_public_messages_are_stable() {
        for (reason, name, records_request) in [
            (
                InternalErrorReason::MoriIoTransferModeMissing,
                "MoriIoTransferModeMissing",
                false,
            ),
            (
                InternalErrorReason::SerializePrefillRequest,
                "SerializePrefillRequest",
                false,
            ),
            (
                InternalErrorReason::SerializeDecodeRequest,
                "SerializeDecodeRequest",
                false,
            ),
            (
                InternalErrorReason::SerializeRequest,
                "SerializeRequest",
                false,
            ),
            (
                InternalErrorReason::SerializeResponse,
                "SerializeResponse",
                true,
            ),
            (InternalErrorReason::BuildResponse, "BuildResponse", true),
            (
                InternalErrorReason::RequestBuilder {
                    phase: PdPhase::Decode,
                },
                "RequestBuilder",
                true,
            ),
        ] {
            let error = PdRequestError::Internal(reason);
            assert!(format!("{error:?}").contains(name));
            assert_eq!(
                super::super::pd_dispatch::records_legacy_request(&error),
                records_request
            );
            assert_eq!(
                error.public_fields(),
                (
                    "internal_error",
                    "The router could not complete the request.".to_owned()
                )
            );
            assert_eq!(error.status(), StatusCode::INTERNAL_SERVER_ERROR);
            assert!(std::error::Error::source(&error).is_none());
        }
    }

    struct Chunks {
        chunks: std::collections::VecDeque<Result<Bytes, TransportErrorKind>>,
        polls: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
    }
    impl Stream for Chunks {
        type Item = Result<Bytes, TransportErrorKind>;
        fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            self.polls.fetch_add(1, Ordering::SeqCst);
            Poll::Ready(self.chunks.pop_front())
        }
    }
    impl Drop for Chunks {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn bounded_bytes_and_exact_eof() {
        for size in [0, 1, 65535, 65536, 65537, 1_000_000] {
            for chunk_size in [1, 8192, 2_000_000] {
                let data = vec![0xff; size];
                let polls = Arc::new(AtomicUsize::new(0));
                let drops = Arc::new(AtomicUsize::new(0));
                let stream = Chunks {
                    chunks: data
                        .chunks(chunk_size)
                        .map(|c| Ok(Bytes::copy_from_slice(c)))
                        .collect(),
                    polls: polls.clone(),
                    drops: drops.clone(),
                };
                let (body, state) = capture_body(stream).await;
                assert_eq!(body.len(), size.min(MAX_UPSTREAM_ERROR_BODY_BYTES));
                assert_eq!(
                    state,
                    if size > 65536 {
                        ErrorBodyState::LimitExceeded
                    } else {
                        ErrorBodyState::Complete
                    }
                );
                let expected_polls = if size > 65536 {
                    65536 / chunk_size + 1
                } else {
                    size.div_ceil(chunk_size) + 1
                };
                assert_eq!(polls.load(Ordering::SeqCst), expected_polls);
                assert_eq!(drops.load(Ordering::SeqCst), 1);
            }
        }
        let utf8 = "中".repeat(21846).into_bytes();
        let (body, state) =
            capture_body(futures_util::stream::iter([Ok(Bytes::from(utf8.clone()))])).await;
        assert_eq!(&body[..], &utf8[..65536]);
        assert_eq!(state, ErrorBodyState::LimitExceeded);
        assert!(std::str::from_utf8(&body).is_err());
    }
    #[tokio::test]
    async fn read_failure_keeps_prefix() {
        for kind in [
            TransportErrorKind::Timeout,
            TransportErrorKind::Connection,
            TransportErrorKind::Other,
        ] {
            let (bytes, state) = capture_body(futures_util::stream::iter([
                Ok(Bytes::from_static(b"prefix")),
                Err(kind),
                Ok(Bytes::from_static(b"never")),
            ]))
            .await;
            assert_eq!(&bytes[..], b"prefix");
            assert_eq!(state, ErrorBodyState::ReadFailed { kind });
        }
    }
    #[test]
    fn status_envelope_and_formatting_never_include_upstream_data() {
        for phase in [PdPhase::Prefill, PdPhase::Decode] {
            for code in 100..=599 {
                let status = StatusCode::from_u16(code).unwrap();
                if status.is_success() {
                    continue;
                }
                let mut headers = HeaderMap::new();
                headers.insert(
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("secret-host"),
                );
                let error = PdRequestError::UpstreamHttp(Box::new(UpstreamHttpError {
                    phase,
                    status,
                    headers,
                    body: Bytes::from_static(b"Traceback /private/secret secret-host"),
                    body_state: ErrorBodyState::Complete,
                }));
                assert_eq!(error.status(), status);
                for formatted in [
                    format!("{error}"),
                    format!("{error:?}"),
                    crate::routers::http::pd_types::error_chain(&error),
                ] {
                    assert!(!formatted.contains("secret"));
                    assert!(formatted.len() < 1024);
                }
                assert_eq!(std::error::Error::source(&error).map(|_| ()), None);
            }
            for (kind, code) in [
                (TransportErrorKind::Timeout, 504),
                (TransportErrorKind::Connection, 502),
                (TransportErrorKind::Other, 502),
            ] {
                assert_eq!(
                    PdRequestError::Transport(TransportError { phase, kind })
                        .status()
                        .as_u16(),
                    code
                );
            }
            assert_eq!(
                PdRequestError::protocol(phase).status(),
                StatusCode::INTERNAL_SERVER_ERROR
            );
        }
        assert_eq!(
            PdRequestError::Internal(InternalErrorReason::SerializeRequest).status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }
    #[tokio::test]
    async fn capture_and_renderer_enforce_header_allowlist() {
        for (retry, keep) in [
            ("5", true),
            ("Sun, 06 Nov 1994 08:49:37 GMT", true),
            ("secret-host", false),
        ] {
            for size in [0, 65537] {
                let upstream = http::Response::builder()
                    .status(429)
                    .header("retry-after", retry)
                    .header("content-type", "text/html")
                    .header("content-length", "999999")
                    .header("content-encoding", "gzip")
                    .header("set-cookie", "secret")
                    .header("www-authenticate", "secret")
                    .header("connection", "keep-alive")
                    .header("x-request-id", "secret-host")
                    .body(Bytes::from(vec![0xff; size]))
                    .unwrap();
                let error =
                    capture_upstream_error(PdPhase::Decode, reqwest::Response::from(upstream))
                        .await;
                let PdRequestError::UpstreamHttp(ref e) = error else {
                    panic!("expected HTTP error");
                };
                assert_eq!(e.headers.len(), if keep { 2 } else { 1 });
                let response = error.into_response();
                assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
                assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
                assert_eq!(response.headers().contains_key(header::RETRY_AFTER), keep);
                if keep {
                    assert_eq!(response.headers()[header::RETRY_AFTER], retry);
                }
                for header in [
                    "content-length",
                    "content-encoding",
                    "set-cookie",
                    "www-authenticate",
                    "connection",
                    "x-request-id",
                ] {
                    assert!(!response.headers().contains_key(header));
                }
            }
        }
        // Extend this renderer test rather than duplicate the routing matrix.
        for (status, content_type, body, code, retained) in [
            (400, "application/json; charset=utf-8", json!({"error":{"message":"bad max_tokens", "param":"max_tokens", "code":"context_length", "type":"private", "traceback":"private-detail"}}).to_string(), "context_length", true),
            (429, "Application/JSON", json!({"error":{"message":"retry later", "code":429}}).to_string(), "429", true),
            (422, "application/json", json!({"error":{"message":"invalid input", "param":7, "code":null}}).to_string(), "422", true),
            (400, "application/json", "invalid JSON".to_owned(), "400", false),
            (400, "application/json", json!({"error":{"message":7}}).to_string(), "400", false),
            (400, "text/html", json!({"error":{"message":"not a JSON media type"}}).to_string(), "400", false),
            (503, "application/json", json!({"error":{"message":"CUDA OOM"}}).to_string(), "503", false),
        ] {
            let upstream = http::Response::builder().status(status)
                .header("content-type", content_type).header("retry-after", "5")
                .body(Bytes::from(body.clone())).unwrap();
            let response = capture_upstream_error(PdPhase::Prefill, reqwest::Response::from(upstream))
                .await.into_response();
            assert_eq!(response.status().as_u16(), status);
            assert_eq!(response.headers()["retry-after"], "5");
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
            let parsed: crate::protocols::spec::ErrorResponse = serde_json::from_slice(&bytes).unwrap();
            let expected_type = if status == 429 { "rate_limit_error" } else if status < 500 { "invalid_request_error" } else { "server_error" };
            assert_eq!(parsed.error.error_type, expected_type);
            assert_eq!(parsed.error.code.as_deref(), Some(code));
            let prefix = format!("Upstream prefill request failed (HTTP {status}).");
            if retained {
                let source: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(parsed.error.message, format!("{prefix} {}", source["error"]["message"].as_str().unwrap()));
                assert_eq!(parsed.error.param.as_deref(), source["error"]["param"].as_str());
            } else {
                assert_eq!(parsed.error.message, prefix);
            }
            assert!(!String::from_utf8_lossy(&bytes).contains("private-detail"));
        }
        for body_state in [
            ErrorBodyState::LimitExceeded,
            ErrorBodyState::ReadFailed {
                kind: TransportErrorKind::Timeout,
            },
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            );
            let error = UpstreamHttpError {
                phase: PdPhase::Prefill,
                status: StatusCode::BAD_REQUEST,
                headers,
                body: Bytes::from_static(b"{\"error\":{\"message\":\"incomplete\"}}"),
                body_state,
            };
            assert!(error.client_detail().is_none());
        }
        let success = reqwest::Response::from(http::Response::new(Bytes::new()));
        assert!(matches!(
            capture_upstream_error(PdPhase::Prefill, success).await,
            PdRequestError::Protocol {
                phase: PdPhase::Prefill
            }
        ));
    }

    #[test]
    fn retry_after_is_strict() {
        for value in [
            "0",
            "5",
            "99999999999999999999",
            "Sun, 06 Nov 1994 08:49:37 GMT",
            "Sunday, 06-Nov-94 08:49:37 GMT",
            "Sun Nov  6 08:49:37 1994",
        ] {
            assert!(
                valid_retry_after(&HeaderValue::from_str(value).unwrap()),
                "{value}"
            );
        }
        for value in [
            "",
            "-1",
            "+5",
            " 5",
            "5 ",
            "1.2",
            "retry-secret-host",
            "Sun, 99 Nov 1994 08:49:37 GMT",
        ] {
            assert!(
                !valid_retry_after(&HeaderValue::from_str(value).unwrap()),
                "{value}"
            );
        }
    }
}

#[cfg(test)]
mod network_tests {
    use super::*;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn raw_response(
        status: u16,
        body: Vec<u8>,
        declared: usize,
        stall: bool,
    ) -> (reqwest::Response, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            while !request.ends_with(b"\r\n\r\n") {
                let mut byte = [0];
                socket.read_exact(&mut byte).await.unwrap();
                request.extend_from_slice(&byte);
                assert!(request.len() < 8192);
            }
            socket.write_all(format!("HTTP/1.1 {status} Error\r\nContent-Length: {declared}\r\nRetry-After: 5\r\nContent-Type: text/html\r\nContent-Encoding: gzip\r\nSet-Cookie: secret\r\n\r\n").as_bytes()).await.unwrap();
            socket.write_all(&body).await.unwrap();
            if stall {
                std::future::pending::<()>().await;
            }
        });
        let response = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_millis(150))
            .build()
            .unwrap()
            .get(url)
            .send()
            .await
            .unwrap();
        (response, task)
    }

    #[tokio::test]
    async fn real_http_read_reset_timeout_and_exact_limit_deadline() {
        for status in [429, 503] {
            for (size, declared, stall, expected) in [
                (
                    6,
                    7,
                    false,
                    ErrorBodyState::ReadFailed {
                        kind: TransportErrorKind::Other,
                    },
                ),
                (
                    6,
                    7,
                    true,
                    ErrorBodyState::ReadFailed {
                        kind: TransportErrorKind::Timeout,
                    },
                ),
                (
                    65536,
                    65537,
                    true,
                    ErrorBodyState::ReadFailed {
                        kind: TransportErrorKind::Timeout,
                    },
                ),
                (65536, 65536, false, ErrorBodyState::Complete),
                (65537, 65537, false, ErrorBodyState::LimitExceeded),
            ] {
                let (response, task) =
                    raw_response(status, vec![0xff; size], declared, stall).await;
                let error = capture_upstream_error(PdPhase::Prefill, response).await;
                task.abort();
                let PdRequestError::UpstreamHttp(ref e) = error else {
                    panic!("HTTP status lost");
                };
                assert_eq!(e.body.len(), size.min(65536));
                assert_eq!(e.body_state, expected);
                assert_eq!(e.status.as_u16(), status);
                let response = error.into_response();
                assert_eq!(response.status().as_u16(), status);
                assert_eq!(response.headers()[header::RETRY_AFTER], "5");
                assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
                assert!(!response.headers().contains_key(header::CONTENT_ENCODING));
                assert!(!response.headers().contains_key(header::CONTENT_LENGTH));
                assert!(!response.headers().contains_key(header::SET_COOKIE));
            }
        }
    }

    #[tokio::test]
    async fn real_transport_classification_and_builder_errors() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let error = client.get(&url).send().await.unwrap_err();
        assert_eq!(classify_transport(&error), TransportErrorKind::Connection);
        assert_eq!(
            PdRequestError::from_reqwest(PdPhase::Decode, error).status(),
            StatusCode::BAD_GATEWAY
        );
        let error = client.get("invalid secret path").send().await.unwrap_err();
        assert!(error.is_builder());
        assert!(matches!(
            PdRequestError::from_reqwest(PdPhase::Prefill, error),
            PdRequestError::Internal(InternalErrorReason::RequestBuilder { .. })
        ));
    }
}
