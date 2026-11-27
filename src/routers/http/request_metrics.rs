//! Request metrics scoped to one HTTP inference request, outside retry and P/D attempts.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use axum::{body::Body, extract::Request, middleware::Next, response::Response};
use http_body::{Body as HttpBody, Frame, SizeHint};
use metrics::{counter, gauge, Gauge};
use parking_lot::Mutex;

use crate::{config::types::RoutingMode, core::Worker, server::AppState};

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
}

tokio::task_local! {
    static REQUEST: Arc<Mutex<RequestState>>;
}

pub(crate) fn phase(phase: Phase) {
    let _ = REQUEST.try_with(|request| request.lock().phase = phase);
}

pub(super) fn prefill_pending(pending: bool) {
    let _ = REQUEST.try_with(|request| request.lock().prefill_pending = pending);
}

pub(super) fn stream(stream: bool) {
    let _ = REQUEST.try_with(|request| request.lock().stream = stream);
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
    if request.method() != http::Method::POST
        || !matches!(
            request.uri().path(),
            "/v1/chat/completions"
                | "/v1/completions"
                | "/v1/responses"
                | "/generate"
                | "/inference/v1/generate"
        )
    {
        return false;
    }

    let config = &app.context.router_config;
    if config.epd.is_some() {
        return false;
    }
    let grpc = |url: &str| crate::backend::is_grpc_url(url);
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

pub(crate) async fn measure(request: Request, next: Next) -> Response {
    const UNRESOLVED: &str = "unresolved";
    let gauge = gauge!("vllm_router_active_requests", "model" => UNRESOLVED);
    gauge.increment(1);
    let state = Arc::new(Mutex::new(RequestState {
        phase: Phase::Routing,
        prefill_pending: false,
        model: Some((UNRESOLVED.into(), gauge)),
        cancelled: true,
        stream: false,
    }));
    let guard = RequestGuard(state.clone());
    let response = REQUEST.scope(state.clone(), next.run(request)).await;
    let (parts, body) = response.into_parts();
    let is_stream = state.lock().stream;
    if !parts.status.is_success() || body.is_end_stream() {
        guard.complete();
        return Response::from_parts(parts, body);
    }

    // A buffered (non-streaming) response already completed upstream. A
    // passthrough body can still be waiting on decode even after headers.
    if is_stream {
        state.lock().phase = Phase::Streaming;
    } else if body.size_hint().exact().is_some() {
        state.lock().cancelled = false;
    }
    Response::from_parts(
        parts,
        Body::new(MeasuredBody {
            body,
            guard: Some(guard),
        }),
    )
}

struct RequestGuard(Arc<Mutex<RequestState>>);

impl RequestGuard {
    fn complete(self) {
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
            if frame.as_ref().is_none_or(Result::is_err) || this.body.is_end_stream() {
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
