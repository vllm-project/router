use axum::{
    body::{Body, HttpBody},
    extract::Request,
    extract::State,
    http::StatusCode,
    http::{header, HeaderValue},
    middleware::Next,
    response::IntoResponse,
    response::Response,
};
use http_body::{Frame, SizeHint};
use rand::Rng;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, OwnedSemaphorePermit, Semaphore};
use tower::{Layer, Service};
use tower_http::classify::ServerErrorsFailureClass;
use tower_http::trace::{MakeSpan, OnFailure, OnRequest, OnResponse, TraceLayer};
use tracing::{debug, error, field::Empty, info, info_span, warn, Span};
use tracing_opentelemetry::OpenTelemetrySpanExt;

pub use crate::core::token_bucket::TokenBucket;

use crate::metrics::RouterMetrics;
use crate::server::AppState;

static EMBEDDINGS_QUEUE_SIZE: AtomicU64 = AtomicU64::new(0);

/// Generate OpenAI-compatible request ID based on endpoint
fn generate_request_id(path: &str) -> String {
    let prefix = if path.contains("/chat/completions") {
        "chatcmpl-"
    } else if path.contains("/completions") {
        "cmpl-"
    } else if path.contains("/generate") {
        "gnt-"
    } else {
        "req-"
    };

    // Generate a random string similar to OpenAI's format
    let chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut rng = rand::rng();
    let random_part: String = (0..24)
        .map(|_| {
            let idx = rng.random_range(0..chars.len());
            chars.chars().nth(idx).unwrap()
        })
        .collect();

    format!("{}{}", prefix, random_part)
}

/// Extension type for storing request ID
#[derive(Clone, Debug)]
pub struct RequestId(pub String);

/// Tower Layer for request ID middleware
#[derive(Clone)]
pub struct RequestIdLayer {
    headers: Arc<Vec<String>>,
}

impl RequestIdLayer {
    pub fn new(headers: Vec<String>) -> Self {
        Self {
            headers: Arc::new(headers),
        }
    }
}

impl<S> Layer<S> for RequestIdLayer {
    type Service = RequestIdMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RequestIdMiddleware {
            inner,
            headers: self.headers.clone(),
        }
    }
}

/// Tower Service for request ID middleware
#[derive(Clone)]
pub struct RequestIdMiddleware<S> {
    inner: S,
    headers: Arc<Vec<String>>,
}

impl<S> Service<Request> for RequestIdMiddleware<S>
where
    S: Service<Request, Response = Response> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
    >;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut req: Request) -> Self::Future {
        let headers = self.headers.clone();

        // Extract request ID from headers or generate new one
        let mut request_id = None;

        for header_name in headers.iter() {
            if let Some(header_value) = req.headers().get(header_name) {
                if let Ok(value) = header_value.to_str() {
                    request_id = Some(value.to_string());
                    break;
                }
            }
        }

        let request_id = request_id.unwrap_or_else(|| generate_request_id(req.uri().path()));

        // Insert request ID into request extensions for other middleware/handlers to use
        req.extensions_mut().insert(RequestId(request_id.clone()));

        // Call the inner service
        let future = self.inner.call(req);

        Box::pin(async move {
            let mut response = future.await?;

            // Add request ID to response headers
            response.headers_mut().insert(
                "x-request-id",
                HeaderValue::from_str(&request_id)
                    .unwrap_or_else(|_| HeaderValue::from_static("invalid-request-id")),
            );

            Ok(response)
        })
    }
}

// ============= Logging Middleware =============

/// Custom span maker that includes request ID
#[derive(Clone, Debug)]
pub struct RequestSpan;

impl<B> MakeSpan<B> for RequestSpan {
    fn make_span(&mut self, request: &Request<B>) -> Span {
        let request_path = request.uri().path();
        if !crate::otel_trace::is_otel_enabled()
            || crate::otel_trace::is_excluded_http_path(request_path)
        {
            return Span::none();
        }

        // Don't try to extract request ID here - it won't be available yet
        // The RequestIdLayer runs after TraceLayer creates the span

        // Use MatchedPath (route template) when available for low-cardinality
        // span names, falling back to the raw URI path otherwise.
        let route = request
            .extensions()
            .get::<axum::extract::MatchedPath>()
            .map(axum::extract::MatchedPath::as_str);
        let path_for_name = route.unwrap_or(request_path);

        let span = info_span!(
            target: "otel_trace",
            "http_request",
            otel.kind = "server",
            otel.name = %format_args!("{} {}", request.method(), path_for_name),
            http.request.method = %request.method(),
            http.route = Empty,
            url.path = %request_path,
            http.response.status_code = Empty,
            request_id = Empty,  // Will be set later
            latency_ms = Empty,
            error = Empty,
        );

        if let Some(route) = route {
            span.record("http.route", route);
        }

        // Extract W3C TraceContext (traceparent/tracestate) from incoming request
        // headers and set it as the parent context on this span. This links the
        // router's span as a child of the upstream caller's trace.
        if !span.is_disabled() {
            let parent_cx = opentelemetry::global::get_text_map_propagator(|propagator| {
                propagator.extract(&opentelemetry_http::HeaderExtractor(request.headers()))
            });
            span.set_parent(parent_cx);
        }

        span
    }
}

/// Custom on_request handler
#[derive(Clone, Debug)]
pub struct RequestLogger;

impl<B> OnRequest<B> for RequestLogger {
    fn on_request(&mut self, request: &Request<B>, span: &Span) {
        let _enter = span.enter();

        // Try to get the request ID from extensions
        // This will work if RequestIdLayer has already run
        if let Some(request_id) = request.extensions().get::<RequestId>() {
            span.record("request_id", request_id.0.as_str());
        }

        // Log the request start
        debug!(
            target: "vllm_router_rs::request",
            "started processing request"
        );
    }
}

/// Custom on_response handler
#[derive(Clone, Debug)]
pub struct ResponseLogger {
    _start_time: Instant,
}

impl Default for ResponseLogger {
    fn default() -> Self {
        Self {
            _start_time: Instant::now(),
        }
    }
}

impl<B> OnResponse<B> for ResponseLogger {
    fn on_response(self, response: &Response<B>, latency: std::time::Duration, span: &Span) {
        let status = response.status();

        if !span.is_disabled() {
            // Record OTel semantic convention attributes only when a real span exists.
            span.record("http.response.status_code", status.as_u16());
            span.record("latency_ms", latency.as_secs_f64() * 1000.0);

            if status.is_server_error() {
                span.set_status(opentelemetry::trace::Status::error(format!(
                    "HTTP {}",
                    status.as_u16()
                )));
            }
        }

        // Log the response completion
        let _enter = span.enter();
        if status.is_server_error() {
            error!(
                target: "vllm_router_rs::response",
                "request failed with server error"
            );
        } else if status.is_client_error() {
            warn!(
                target: "vllm_router_rs::response",
                "request failed with client error"
            );
        } else {
            debug!(
                target: "vllm_router_rs::response",
                "finished processing request"
            );
        }
    }
}

/// Custom on_failure handler for pre-response errors (e.g. connection drops).
#[derive(Clone, Debug, Default)]
pub struct FailureLogger;

impl OnFailure<ServerErrorsFailureClass> for FailureLogger {
    fn on_failure(
        &mut self,
        failure: ServerErrorsFailureClass,
        latency: std::time::Duration,
        span: &Span,
    ) {
        let message = match &failure {
            ServerErrorsFailureClass::StatusCode(code) => {
                format!("response failed with status {code}")
            }
            ServerErrorsFailureClass::Error(err) => {
                format!("request failed: {err}")
            }
        };

        let _enter = span.enter();
        error!(
            target: "vllm_router_rs::response",
            "{message}"
        );

        if !span.is_disabled() {
            span.record("latency_ms", latency.as_secs_f64() * 1000.0);
            span.record("error", message.as_str());
            span.set_status(opentelemetry::trace::Status::error(message));
        }
    }
}

/// Create a configured TraceLayer for HTTP logging
/// Note: Actual request/response logging with request IDs is done in RequestIdService
pub fn create_logging_layer() -> TraceLayer<
    tower_http::classify::SharedClassifier<tower_http::classify::ServerErrorsAsFailures>,
    RequestSpan,
    RequestLogger,
    ResponseLogger,
    (),
    (),
    FailureLogger,
> {
    TraceLayer::new_for_http()
        .make_span_with(RequestSpan)
        .on_request(RequestLogger)
        .on_response(ResponseLogger::default())
        .on_body_chunk(())
        .on_eos(())
        .on_failure(FailureLogger)
}

/// Structured logging data for requests
#[derive(Debug, serde::Serialize)]
pub struct RequestLogEntry {
    pub timestamp: String,
    pub request_id: String,
    pub method: String,
    pub uri: String,
    pub status: u16,
    pub latency_ms: u64,
    pub user_agent: Option<String>,
    pub remote_addr: Option<String>,
    pub error: Option<String>,
}

/// Log a request with structured data
pub fn log_request(entry: RequestLogEntry) {
    if entry.status >= 500 {
        tracing::error!(
            target: "vllm_router_rs::http",
            request_id = %entry.request_id,
            method = %entry.method,
            uri = %entry.uri,
            status = entry.status,
            latency_ms = entry.latency_ms,
            user_agent = ?entry.user_agent,
            remote_addr = ?entry.remote_addr,
            error = ?entry.error,
            "HTTP request failed"
        );
    } else if entry.status >= 400 {
        tracing::warn!(
            target: "vllm_router_rs::http",
            request_id = %entry.request_id,
            method = %entry.method,
            uri = %entry.uri,
            status = entry.status,
            latency_ms = entry.latency_ms,
            user_agent = ?entry.user_agent,
            remote_addr = ?entry.remote_addr,
            "HTTP request client error"
        );
    } else {
        tracing::info!(
            target: "vllm_router_rs::http",
            request_id = %entry.request_id,
            method = %entry.method,
            uri = %entry.uri,
            status = entry.status,
            latency_ms = entry.latency_ms,
            user_agent = ?entry.user_agent,
            remote_addr = ?entry.remote_addr,
            "HTTP request completed"
        );
    }
}

// ============ Concurrency Limiting with Queue Support ============

/// Request queue entry
pub struct QueuedRequest {
    /// Time when the request was queued
    queued_at: tokio::time::Instant,
    /// Channel to send the permit back when acquired
    permit_tx: oneshot::Sender<Result<OwnedSemaphorePermit, StatusCode>>,
}

/// Queue metrics for monitoring
#[derive(Debug, Default)]
pub struct QueueMetrics {
    pub total_queued: std::sync::atomic::AtomicU64,
    pub current_queued: std::sync::atomic::AtomicU64,
    pub total_timeout: std::sync::atomic::AtomicU64,
    pub total_rejected: std::sync::atomic::AtomicU64,
}

/// Queue processor that handles queued requests
pub struct QueueProcessor {
    token_bucket: Arc<TokenBucket>,
    concurrency_semaphore: Arc<Semaphore>,
    queue_rx: mpsc::Receiver<QueuedRequest>,
    queue_timeout: Duration,
}

impl QueueProcessor {
    pub fn new(
        token_bucket: Arc<TokenBucket>,
        concurrency_semaphore: Arc<Semaphore>,
        queue_rx: mpsc::Receiver<QueuedRequest>,
        queue_timeout: Duration,
    ) -> Self {
        Self {
            token_bucket,
            concurrency_semaphore,
            queue_rx,
            queue_timeout,
        }
    }

    pub async fn run(mut self) {
        info!("Starting concurrency queue processor");

        // Let each queued request wait independently so one wait does not stall the queue.
        while let Some(mut queued) = self.queue_rx.recv().await {
            let deadline = queued.queued_at + self.queue_timeout;
            let token_bucket = self.token_bucket.clone();
            let concurrency_semaphore = self.concurrency_semaphore.clone();

            tokio::spawn(async move {
                let acquisition = tokio::time::timeout_at(deadline, async {
                    if tokio::time::Instant::now() >= deadline {
                        return Some(Err(StatusCode::REQUEST_TIMEOUT));
                    }

                    let permit = tokio::select! {
                        biased;
                        _ = queued.permit_tx.closed() => return None,
                        result = concurrency_semaphore.acquire_owned() => match result {
                            Ok(permit) => permit,
                            Err(_) => return Some(Err(StatusCode::SERVICE_UNAVAILABLE)),
                        },
                    };

                    loop {
                        if tokio::time::Instant::now() >= deadline {
                            return Some(Err(StatusCode::REQUEST_TIMEOUT));
                        }

                        if token_bucket.try_acquire(1.0).await.is_ok() {
                            return Some(Ok(permit));
                        }

                        tokio::select! {
                            biased;
                            _ = queued.permit_tx.closed() => return None,
                            _ = tokio::time::sleep(Duration::from_millis(10)) => {}
                        }
                    }
                })
                .await;

                match acquisition {
                    Ok(Some(Ok(permit))) => {
                        if queued.permit_tx.send(Ok(permit)).is_err() {
                            debug!("Queued request was cancelled before permit delivery");
                        }
                    }
                    Ok(Some(Err(status))) => {
                        let _ = queued.permit_tx.send(Err(status));
                    }
                    Ok(None) => {
                        debug!("Queued request was cancelled while waiting for a permit");
                    }
                    Err(_) => {
                        warn!("Queue: request timed out waiting for a permit or rate token");
                        let _ = queued.permit_tx.send(Err(StatusCode::REQUEST_TIMEOUT));
                    }
                }
            });
        }

        warn!("Concurrency queue processor shutting down");
    }
}

/// State for the concurrency limiter
pub struct ConcurrencyLimiter {
    pub queue_tx: Option<mpsc::Sender<QueuedRequest>>,
}

impl ConcurrencyLimiter {
    /// Create new concurrency limiter with optional queue
    pub fn new(
        token_bucket: Arc<TokenBucket>,
        concurrency_semaphore: Arc<Semaphore>,
        queue_size: usize,
        queue_timeout: Duration,
    ) -> (Self, Option<QueueProcessor>) {
        if queue_size > 0 {
            let (queue_tx, queue_rx) = mpsc::channel(queue_size);
            let processor =
                QueueProcessor::new(token_bucket, concurrency_semaphore, queue_rx, queue_timeout);

            (
                Self {
                    queue_tx: Some(queue_tx),
                },
                Some(processor),
            )
        } else {
            (Self { queue_tx: None }, None)
        }
    }
}

/// Hold the concurrency slot until the response body has been fully sent.
///
/// `next.run()` returns as soon as the upstream response head is ready. On a streaming
/// endpoint the body is still being forwarded at that point, so releasing the permit
/// there frees the slot at the first token instead of at the end of generation, and
/// `--max-concurrent-requests` ends up bounding time-to-first-token rather than
/// concurrent generations. Moving the permit into the body ties the slot to the whole
/// response, including a client disconnect, which drops the body.
fn hold_permit_for_body(response: Response, permit: OwnedSemaphorePermit) -> Response {
    let (parts, body) = response.into_parts();
    let permit = if body.is_end_stream() {
        drop(permit);
        None
    } else {
        Some(permit)
    };
    Response::from_parts(parts, Body::new(ConcurrencyBody { body, permit }))
}

struct ConcurrencyBody {
    body: Body,
    permit: Option<OwnedSemaphorePermit>,
}

impl HttpBody for ConcurrencyBody {
    type Data = bytes::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let frame = std::pin::Pin::new(&mut self.body).poll_frame(cx);
        if matches!(frame, std::task::Poll::Ready(None | Some(Err(_)))) || self.body.is_end_stream()
        {
            self.permit.take();
        }
        frame
    }

    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.body.size_hint()
    }
}

struct EmbeddingsQueueMetricGuard(bool);

impl EmbeddingsQueueMetricGuard {
    fn new(is_embeddings: bool) -> Self {
        if is_embeddings {
            let new_val = EMBEDDINGS_QUEUE_SIZE.fetch_add(1, Ordering::Relaxed) + 1;
            RouterMetrics::set_embeddings_queue_size(new_val as usize);
        }
        Self(is_embeddings)
    }
}

impl Drop for EmbeddingsQueueMetricGuard {
    fn drop(&mut self) {
        if self.0 {
            let new_val = EMBEDDINGS_QUEUE_SIZE.fetch_sub(1, Ordering::Relaxed) - 1;
            RouterMetrics::set_embeddings_queue_size(new_val as usize);
        }
    }
}

/// A 429 carrying a `Retry-After` hint.
///
/// Without the header OpenAI-compatible SDKs treat the 429 as non-retryable and
/// surface it to the caller instead of backing off, so a bare 429 turns a transient
/// queue-full into a hard failure. The hint comes from the token bucket rather than a
/// new knob; it is a lower bound, since a finishing request can free a slot sooner.
async fn too_many_requests(token_bucket: &TokenBucket) -> Response {
    let seconds = token_bucket.seconds_until_available(1.0).await.ceil() as u64;
    let mut response = StatusCode::TOO_MANY_REQUESTS.into_response();
    response
        .headers_mut()
        .insert(header::RETRY_AFTER, HeaderValue::from(seconds.max(1)));
    response
}

/// Middleware function for concurrency limiting with optional queuing
pub async fn concurrency_limit_middleware(
    State(app_state): State<Arc<AppState>>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    // Identify if this is an embeddings request based on path
    let is_embeddings = request.uri().path().contains("/v1/embeddings");
    let token_bucket = app_state.context.rate_limiter.clone();
    let concurrency_semaphore = app_state.context.concurrency_semaphore.clone();

    // Acquire concurrency before consuming a rate token.
    if let Ok(permit) = concurrency_semaphore.clone().try_acquire_owned() {
        if token_bucket.try_acquire(1.0).await.is_ok() {
            debug!("Acquired concurrency permit and rate token immediately");
            let response = next.run(request).await;
            return hold_permit_for_body(response, permit);
        }

        drop(permit);
        warn!("Rate limit exhausted, attempting to queue request");
    }

    // No concurrency permit or rate token is available. Queue if enabled.
    if let Some(queue_tx) = &app_state.concurrency_queue_tx {
        debug!("No immediate admission available, attempting to queue request");

        // Receive the semaphore permit from the queue processor.
        let (permit_tx, permit_rx) = oneshot::channel();

        let queued = QueuedRequest {
            queued_at: tokio::time::Instant::now(),
            permit_tx,
        };

        // Try to send to queue
        match queue_tx.try_send(queued) {
            Ok(_) => {
                let queue_metric = EmbeddingsQueueMetricGuard::new(is_embeddings);
                let permit_result = permit_rx.await;
                drop(queue_metric);

                match permit_result {
                    Ok(Ok(permit)) => {
                        debug!("Acquired concurrency permit from queue");
                        let response = next.run(request).await;
                        hold_permit_for_body(response, permit)
                    }
                    Ok(Err(status)) => {
                        warn!("Queue returned error status: {}", status);
                        status.into_response()
                    }
                    Err(_) => {
                        error!("Queue response channel closed");
                        StatusCode::INTERNAL_SERVER_ERROR.into_response()
                    }
                }
            }
            Err(_) => {
                warn!("Request queue is full, returning 429");
                too_many_requests(&token_bucket).await
            }
        }
    } else {
        warn!("No admission capacity available and queuing is disabled, returning 429");
        too_many_requests(&token_bucket).await
    }
}
