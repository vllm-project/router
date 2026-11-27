//! Request metrics scoped to one HTTP inference request, outside retry and P/D attempts.

use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::Instant,
};

use axum::{body::Body, extract::Request, middleware::Next, response::Response};
use http_body::{Body as HttpBody, Frame, SizeHint};
use metrics::{counter, gauge, histogram, Gauge};
use parking_lot::Mutex;

use crate::{config::types::RoutingMode, core::Worker, server::AppState};

use super::response_metrics::{Observer, SseBatch, Usage};

#[derive(Clone, Copy)]
pub(crate) enum Phase {
    Queued,
    Routing,
    Prefill,
    Decode,
    Streaming,
}

impl Phase {
    fn label(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Routing => "routing",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Streaming => "streaming",
        }
    }
}

struct RequestState {
    phase: Phase,
    prefill_pending: bool,
    model: Option<(String, Gauge)>,
    cancelled: bool,
    stream: bool,
    echo: bool,
    preprocess: Option<StageTimer>,
    handoff_pending: bool,
}

tokio::task_local! {
    static REQUEST: Arc<Mutex<RequestState>>;
}

#[derive(Clone, Copy)]
pub(super) enum Stage {
    Preprocess,
    Route,
    PrefillDispatch,
    DecodeDispatch,
}

impl Stage {
    fn label(self) -> &'static str {
        match self {
            Self::Preprocess => "preprocess",
            Self::Route => "route",
            Self::PrefillDispatch => "prefill_dispatch",
            Self::DecodeDispatch => "decode_dispatch",
        }
    }
}

pub(super) struct StageTimer(Stage, Instant, Gauge);

impl StageTimer {
    fn new(stage: Stage) -> Self {
        let gauge = gauge!("vllm_router_stage_requests", "stage" => stage.label());
        gauge.increment(1);
        Self(stage, Instant::now(), gauge)
    }
}

impl Drop for StageTimer {
    fn drop(&mut self) {
        self.2.decrement(1);
        histogram!("vllm_router_stage_duration_seconds", "stage" => self.0.label())
            .record(self.1.elapsed().as_secs_f64());
    }
}

pub(super) fn active() -> bool {
    REQUEST
        .try_with(|request| request.lock().model.is_some())
        .unwrap_or(false)
}

pub(super) fn stage(stage: Stage) -> Option<StageTimer> {
    active().then(|| StageTimer::new(stage))
}

pub(crate) fn begin_preprocess() {
    let _ = REQUEST.try_with(|request| {
        let mut state = request.lock();
        if state.model.is_some() {
            state.preprocess = Some(StageTimer::new(Stage::Preprocess));
        }
    });
}

pub(super) fn end_preprocess() {
    let _ = REQUEST.try_with(|request| request.lock().preprocess.take());
}

#[derive(Clone, Copy)]
pub(super) enum Handoff {
    Ok,
    Rejected,
    Timeout,
    Failed,
}

impl Handoff {
    fn record(self) {
        let result = match self {
            Self::Ok => "ok",
            Self::Rejected => "rejected",
            Self::Timeout => "timeout",
            Self::Failed => "failed",
        };
        counter!("vllm_router_pd_handoff_total", "result" => result).increment(1);
    }

    pub(super) fn observed(self) {
        if active() {
            self.record();
        }
    }

    pub(super) fn http_failure(status: http::StatusCode) {
        if !status.is_success() {
            if matches!(
                status,
                http::StatusCode::REQUEST_TIMEOUT | http::StatusCode::GATEWAY_TIMEOUT
            ) {
                Self::Timeout
            } else if status.is_client_error() {
                Self::Rejected
            } else {
                Self::Failed
            }
            .observed();
        }
    }

    pub(super) fn from_error(error: &(dyn std::error::Error + 'static)) -> Self {
        let mut source = Some(error);
        while let Some(error) = source {
            if error
                .downcast_ref::<reqwest::Error>()
                .is_some_and(reqwest::Error::is_timeout)
            {
                return Self::Timeout;
            }
            source = error.source();
        }
        Self::Failed
    }

    pub(super) fn pending(verified_envelope: bool) {
        let _ = REQUEST.try_with(|request| request.lock().handoff_pending = verified_envelope);
    }

    pub(super) fn body_failure(error: &(dyn std::error::Error + 'static)) {
        let _ = REQUEST.try_with(|request| request.lock().handoff_pending = false);
        Self::from_error(error).observed();
    }
}

pub(crate) fn phase(phase: Phase) {
    let _ = REQUEST.try_with(|request| request.lock().phase = phase);
}

pub(super) fn upstream_terminal() {
    let _ = REQUEST.try_with(|request| request.lock().cancelled = false);
}

pub(super) fn retrying() {
    let _ = REQUEST.try_with(|request| {
        let mut state = request.lock();
        state.phase = Phase::Routing;
        state.cancelled = true;
    });
}

pub(super) fn prefill_pending(pending: bool) {
    let _ = REQUEST.try_with(|request| request.lock().prefill_pending = pending);
}

pub(super) fn stream(stream: bool) {
    let _ = REQUEST.try_with(|request| request.lock().stream = stream);
}

pub(super) fn echo(echo: bool) {
    let _ = REQUEST.try_with(|request| request.lock().echo = echo);
}

/// Per-response status for HTTP forwarding loops that run outside the body
/// poll. If a retry discards a response, this evidence is discarded with it.
#[derive(Clone, Default)]
pub(super) struct ProducerEvidence(Arc<ProducerState>);

#[derive(Default)]
struct ProducerState {
    failed: AtomicBool,
    usage: Mutex<Option<Usage>>,
}

impl ProducerEvidence {
    pub(super) fn current() -> Option<Self> {
        active().then(Self::default)
    }

    pub(super) fn fail(evidence: Option<&Self>) {
        if let Some(evidence) = evidence {
            evidence.0.failed.store(true, Ordering::Release);
        }
    }

    pub(super) fn hidden_usage(&self, usage: Usage) {
        *self.0.usage.lock() = Some(usage);
    }

    fn failed(&self) -> bool {
        self.0.failed.load(Ordering::Acquire)
    }
}

pub(super) fn worker(worker: &dyn Worker) {
    let _ = REQUEST.try_with(|request| {
        let mut state = request.lock();
        let Some((model, old)) = state.model.as_ref() else {
            return;
        };
        let name = worker.model_id();
        if model != name {
            let next = gauge!("vllm_router_active_requests", "model" => name.to_owned());
            old.decrement(1);
            next.increment(1);
            state.model = Some((name.to_owned(), next));
        }
    });
}

// Dynamic pools can start empty and later become gRPC. Keep the HTTP
// middleware independent of changes to gRPC dispatch or response semantics.
pub(super) fn disable() {
    let _ = REQUEST.try_with(|request| {
        if let Some((_, gauge)) = request.lock().model.take() {
            gauge.decrement(1);
        }
    });
}

pub(crate) fn enabled(app: &AppState, request: &Request) -> bool {
    if request.method() != http::Method::POST || route_label(request.uri().path()).is_none() {
        return false;
    }

    let config = &app.context.router_config;
    if config.epd.is_some() {
        return false;
    }
    let grpc = crate::backend::is_grpc_url;
    // A discovery-only gRPC pool has no startup URLs. Detect it here as
    // well, before entering the HTTP admission queue.
    if config.mode.worker_count() == 0
        && !app
            .context
            .worker_registry
            .get_by_connection(&crate::core::ConnectionMode::Grpc)
            .is_empty()
    {
        return false;
    }
    match &config.mode {
        RoutingMode::Regular { worker_urls } => !worker_urls.iter().any(|url| grpc(url)),
        RoutingMode::VllmPrefillDecode {
            prefill_urls,
            decode_urls,
            ..
        } => {
            !prefill_urls.iter().any(|(url, _)| grpc(url))
                && !decode_urls.iter().any(|url| grpc(url))
        }
        RoutingMode::OpenAI { .. } => false,
    }
}

fn route_label(path: &str) -> Option<&'static str> {
    match path {
        "/v1/chat/completions" => Some("/v1/chat/completions"),
        "/v1/completions" => Some("/v1/completions"),
        "/v1/responses" => Some("/v1/responses"),
        "/generate" => Some("/generate"),
        "/inference/v1/generate" => Some("/inference/v1/generate"),
        _ => None,
    }
}

pub(crate) async fn measure(request: Request, next: Next) -> Response {
    let started = Instant::now();
    let route = route_label(request.uri().path()).expect("measured route has a static label");
    const UNRESOLVED: &str = "unresolved";
    let gauge = gauge!("vllm_router_active_requests", "model" => UNRESOLVED);
    gauge.increment(1);
    let state = Arc::new(Mutex::new(RequestState {
        phase: Phase::Routing,
        prefill_pending: false,
        model: Some((UNRESOLVED.into(), gauge)),
        cancelled: true,
        stream: false,
        echo: false,
        preprocess: None,
        handoff_pending: false,
    }));
    let guard = RequestGuard(state.clone());
    let response = REQUEST.scope(state.clone(), next.run(request)).await;
    let (mut parts, body) = response.into_parts();
    let (is_stream, echo, enabled, handoff_pending) = {
        let mut state = state.lock();
        state.preprocess.take();
        (
            state.stream,
            state.echo,
            state.model.is_some(),
            state.handoff_pending,
        )
    };
    if !enabled || body.is_end_stream() || (!parts.status.is_success() && !is_stream) {
        guard.complete();
        return Response::from_parts(parts, body);
    }

    // A buffered (non-streaming) response already completed upstream. A
    // passthrough body can still be waiting on decode even after headers.
    if !parts.status.is_success() {
        guard.terminal();
    } else if is_stream {
        state.lock().phase = Phase::Streaming;
    } else if body.size_hint().exact().is_some() {
        state.lock().cancelled = false;
    }
    let content_type = parts
        .headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<mime::Mime>().ok());
    let identity = parts
        .headers
        .get_all(http::header::CONTENT_ENCODING)
        .iter()
        .all(|value| value.as_bytes().eq_ignore_ascii_case(b"identity"));
    let sse = is_stream
        && content_type
            .as_ref()
            .is_some_and(|ty| ty.essence_str() == "text/event-stream");
    let json = !is_stream
        && content_type.is_some_and(|ty| {
            ty.subtype() == "json" || ty.suffix().is_some_and(|suffix| suffix == "json")
        });
    let observer = (parts.status.is_success() && identity && (sse || json))
        .then(|| Observer::new(route, sse, echo))
        .flatten();
    // Hyper may stop polling once an explicit Content-Length is written.
    // A wrapped stream without an exact size hint needs its terminal poll
    // for usage/handoff settlement and for any inner Program body observer.
    // Leave exact-size, already-buffered bodies and their headers unchanged.
    if body.size_hint().exact().is_none() {
        parts.headers.remove(http::header::CONTENT_LENGTH);
    }
    let evidence = parts.extensions.remove::<ProducerEvidence>();
    let sse_batch = parts.extensions.remove::<SseBatch>();
    Response::from_parts(
        parts,
        Body::new(MeasuredBody {
            body,
            guard: Some(guard),
            observer,
            started,
            route,
            evidence,
            handoff_pending,
            sse_batch,
        }),
    )
}

struct RequestGuard(Arc<Mutex<RequestState>>);

impl RequestGuard {
    fn complete(self) {
        self.terminal();
    }

    fn terminal(&self) {
        self.0.lock().cancelled = false;
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        let mut state = self.0.lock();
        if let Some((_, gauge)) = state.model.take() {
            gauge.decrement(1);
            if state.cancelled {
                // When both P/D legs are outstanding, P gates the response
                // and wins over D. Once P finishes, D owns the wait.
                let stage = if state.prefill_pending {
                    Phase::Prefill
                } else {
                    state.phase
                };
                counter!("vllm_router_requests_cancelled_total", "stage" => stage.label())
                    .increment(1);
            }
        }
    }
}

struct MeasuredBody {
    body: Body,
    guard: Option<RequestGuard>,
    observer: Option<Observer>,
    started: Instant,
    route: &'static str,
    evidence: Option<ProducerEvidence>,
    handoff_pending: bool,
    sse_batch: Option<SseBatch>,
}

impl HttpBody for MeasuredBody {
    type Data = bytes::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        let result = Pin::new(&mut this.body).poll_frame(cx);
        if let Poll::Ready(frame) = &result {
            if let (Some(Ok(frame)), Some(observer)) = (frame, &mut this.observer) {
                if let Some(bytes) = frame.data_ref() {
                    observer.observe(
                        bytes,
                        this.started,
                        Instant::now(),
                        this.sse_batch.as_ref().is_some_and(SseBatch::is_coalesced),
                    );
                }
                if observer.terminal() {
                    if let Some(guard) = &this.guard {
                        guard.terminal();
                    }
                }
            }
            let failed = frame.as_ref().is_some_and(Result::is_err)
                || this.evidence.as_ref().is_some_and(ProducerEvidence::failed);
            if failed {
                if let Some(guard) = &this.guard {
                    guard.terminal();
                }
            }
            if frame.as_ref().is_none_or(Result::is_err) || this.body.is_end_stream() {
                if let Some(Err(error)) = frame {
                    if std::mem::take(&mut this.handoff_pending) {
                        Handoff::from_error(error).record();
                    }
                }
                if let Some(observer) = this.observer.take().filter(|_| !failed) {
                    let usage = this
                        .evidence
                        .as_ref()
                        .and_then(|evidence| *evidence.0.usage.lock());
                    let observed = observer.finish(this.route, usage);
                    if std::mem::take(&mut this.handoff_pending) {
                        if let Some(success) = observed {
                            if success {
                                Handoff::Ok
                            } else {
                                Handoff::Failed
                            }
                            .record();
                        }
                    }
                }
                if let Some(guard) = this.guard.take() {
                    guard.complete();
                }
            }
        }
        result
    }

    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.body.size_hint()
    }
}

impl Drop for MeasuredBody {
    fn drop(&mut self) {
        if self.body.is_end_stream() || self.evidence.as_ref().is_some_and(ProducerEvidence::failed)
        {
            if let Some(guard) = self.guard.take() {
                guard.complete();
            }
        }
    }
}
