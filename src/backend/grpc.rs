//! tonic client for vLLM rust `Inference.GenerateStream`.
//!
//! Tokenize on the router, send `token_ids`, detok the returned ids.
//! Channel cache is per connect URI.

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    body::Body,
    http::{HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use dashmap::DashMap;
use futures_util::StreamExt;
use tokio::sync::{mpsc, oneshot, watch, OnceCell};
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::Channel;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use super::convert::{
    chat_to_generate_request_with_progress, limit_to_prefill, proto_struct_to_json,
    requires_worker_output_text, set_kv_transfer_params,
};
use super::detect::{grpc_connect_uri, parse_dp_rank};
use super::frontend::{PreparedRequest, ResponseShape};
use super::openai::{
    completion_final_response, final_response, finish_reason_name, now_secs, sse_chunk,
    sse_usage_chunk, SSE_DONE,
};
use super::pb::control_client::ControlClient;
use super::pb::finish_info::FinishReason;
use super::pb::inference_client::InferenceClient;
use super::pb::AbortRequest;
use super::stream_output::{output_frame_tracking, OutputFrameConsumer, OutputFrameProducer};
use super::vllm_frontend::{decode_stream, emit_detok, DynTokenizer, IncrementalDecoderTrait};
use crate::protocols::spec::Usage;

const DP_RANK_METADATA: &str = "x-data-parallel-rank";
const STREAM_BUFFER_CAPACITY: usize = 32;
const ENGINE_ABORT_RETRY_DELAY: Duration = Duration::from_millis(500);

pub(crate) type FirstOutputCallback = Arc<dyn Fn() + Send + Sync + 'static>;
pub(crate) type PrefillProgressCallback = Arc<dyn Fn(u64, u64) + Send + Sync + 'static>;
type PrefillProgressCapabilityCache = Arc<DashMap<String, Arc<OnceCell<Result<bool, String>>>>>;

/// P/D fields layered on a prepared chat. Default is a colocated request.
#[derive(Clone, Debug, Default)]
pub(crate) struct KvHandoff {
    /// Engine `GenerateRequest.request_id`; the OpenAI response id is separate.
    pub engine_request_id: Option<String>,
    pub kv_transfer_params: Option<serde_json::Value>,
}

/// Sends `Control.Abort` for a P/D engine request if dropped before
/// [`disarm`](Self::disarm). A vLLM worker notices a cancelled RPC only at its
/// next send, so one waiting on KV would otherwise keep the request running.
///
/// vLLM registers a request before it answers the RPC, and an Abort for an
/// unregistered id is a no-op. A leg dropped before its answer may race its
/// own registration, so it gets a second Abort after a short delay.
struct EngineAbortGuard {
    target: Option<(Channel, String)>,
    timeout: Duration,
    answered: bool,
}

impl EngineAbortGuard {
    fn new(channel: &Channel, engine_request_id: Option<&str>, timeout: Duration) -> Self {
        Self {
            target: engine_request_id.map(|id| (channel.clone(), id.to_string())),
            timeout,
            answered: false,
        }
    }

    fn answered(&mut self) {
        self.answered = true;
    }

    fn disarm(mut self) {
        self.target = None;
    }
}

impl Drop for EngineAbortGuard {
    fn drop(&mut self) {
        let Some((channel, request_id)) = self.target.take() else {
            return;
        };
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let timeout = self.timeout;
        let retry = !self.answered;
        runtime.spawn(async move {
            send_engine_abort(&channel, &request_id, timeout).await;
            if retry {
                tokio::time::sleep(ENGINE_ABORT_RETRY_DELAY).await;
                send_engine_abort(&channel, &request_id, timeout).await;
            }
        });
    }
}

async fn send_engine_abort(channel: &Channel, request_id: &str, timeout: Duration) {
    let request = AbortRequest {
        request_ids: vec![request_id.to_string()],
    };
    let result =
        tokio::time::timeout(timeout, ControlClient::new(channel.clone()).abort(request)).await;
    match result {
        Ok(Ok(_)) => debug!(%request_id, "grpc Control.Abort sent"),
        Ok(Err(status)) => {
            debug!(%request_id, code = ?status.code(), msg = %status.message(), "grpc Control.Abort failed")
        }
        Err(_) => debug!(%request_id, "grpc Control.Abort timed out"),
    }
}

#[derive(Clone)]
pub struct GrpcStreamTask {
    handle: Arc<std::sync::Mutex<Option<tokio::task::JoinHandle<()>>>>,
    output_frame: Option<OutputFrameConsumer>,
    failed: Option<watch::Receiver<bool>>,
}

impl GrpcStreamTask {
    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    pub(crate) fn new(handle: tokio::task::JoinHandle<()>) -> Self {
        Self {
            handle: Arc::new(std::sync::Mutex::new(Some(handle))),
            output_frame: None,
            failed: None,
        }
    }

    pub(crate) fn with_output_frame(
        handle: tokio::task::JoinHandle<()>,
        output_frame: OutputFrameConsumer,
    ) -> Self {
        Self {
            handle: Arc::new(std::sync::Mutex::new(Some(handle))),
            output_frame: Some(output_frame),
            failed: None,
        }
    }

    fn with_failure(mut self, failed: watch::Receiver<bool>) -> Self {
        self.failed = Some(failed);
        self
    }

    /// Resolves once the producer has queued an error frame and stopped.
    /// Stays pending for a stream that finishes cleanly.
    pub(crate) fn failure(&self) -> impl std::future::Future<Output = ()> + Send + 'static {
        let outcome = self.outcome();
        async move {
            if !outcome.await {
                std::future::pending::<()>().await
            }
        }
    }

    /// Resolves once the producer stops: `true` if it queued an error frame,
    /// `false` if it finished cleanly or was dropped.
    pub(crate) fn outcome(&self) -> impl std::future::Future<Output = bool> + Send + 'static {
        let failed = self.failed.clone();
        async move {
            match failed {
                Some(mut failed) => failed.wait_for(|failed| *failed).await.is_ok(),
                None => false,
            }
        }
    }

    pub fn take(&self) -> Option<tokio::task::JoinHandle<()>> {
        self.handle.lock().ok()?.take()
    }

    pub(crate) fn output_frame(&self) -> Option<OutputFrameConsumer> {
        self.output_frame.clone()
    }
}

async fn send_stream_chunk(
    tx: &mpsc::Sender<Result<String, std::io::Error>>,
    output: &mut OutputFrameProducer,
    chunk: String,
    has_generated_output: bool,
) -> bool {
    output.send(tx, chunk, has_generated_output).await
}

fn grpc_status_code(status: &tonic::Status) -> StatusCode {
    match status.code() {
        tonic::Code::InvalidArgument
        | tonic::Code::FailedPrecondition
        | tonic::Code::OutOfRange => StatusCode::BAD_REQUEST,
        tonic::Code::NotFound => StatusCode::NOT_FOUND,
        tonic::Code::AlreadyExists | tonic::Code::Aborted => StatusCode::CONFLICT,
        tonic::Code::ResourceExhausted => StatusCode::TOO_MANY_REQUESTS,
        tonic::Code::DeadlineExceeded => StatusCode::GATEWAY_TIMEOUT,
        tonic::Code::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        _ => StatusCode::BAD_GATEWAY,
    }
}

#[derive(Clone)]
pub struct GrpcEngineBackend {
    clients: Arc<DashMap<String, Channel>>,
    prefill_progress_capabilities: PrefillProgressCapabilityCache,
    connect_timeout: Duration,
    request_timeout: Duration,
    pub(super) prefill_progress_stale_after: Option<Duration>,
}

impl std::fmt::Debug for GrpcEngineBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrpcEngineBackend")
            .field("cached_channels", &self.clients.len())
            .finish()
    }
}

impl Default for GrpcEngineBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GrpcEngineBackend {
    pub fn new() -> Self {
        Self::with_timeouts(Duration::from_secs(10), Duration::from_secs(1800))
    }

    pub fn with_connect_timeout(connect_timeout: Duration) -> Self {
        Self::with_timeouts(connect_timeout, Duration::from_secs(1800))
    }

    pub fn with_timeouts(connect_timeout: Duration, request_timeout: Duration) -> Self {
        Self {
            clients: Arc::new(DashMap::new()),
            prefill_progress_capabilities: Arc::new(DashMap::new()),
            connect_timeout,
            request_timeout,
            prefill_progress_stale_after: None,
        }
    }

    async fn channel_for(&self, worker_url: &str) -> Result<Channel, String> {
        let uri = grpc_connect_uri(worker_url)?;
        if let Some(hit) = self.clients.get(&uri) {
            return Ok(hit.clone());
        }
        let channel = Channel::from_shared(uri.clone())
            .map_err(|e| format!("invalid grpc uri {uri}: {e}"))?
            .connect_timeout(self.connect_timeout)
            .connect()
            .await
            .map_err(|e| format!("grpc connect {uri}: {e}"))?;
        self.clients.insert(uri, channel.clone());
        Ok(channel)
    }

    async fn supports_prefill_progress(&self, worker_url: &str) -> Result<bool, String> {
        let uri = grpc_connect_uri(worker_url)?;
        let connect_timeout = self.connect_timeout;
        cached_prefill_progress_capability(
            &self.prefill_progress_capabilities,
            uri.clone(),
            || async move {
                let channel = Channel::from_shared(uri.clone())
                    .map_err(|e| format!("invalid grpc uri {uri}: {e}"))?
                    .connect_timeout(connect_timeout)
                    .connect()
                    .await
                    .map_err(|e| format!("grpc control connect {uri}: {e}"))?;
                let mut client = ControlClient::new(channel);
                let mut request = tonic::Request::new(super::pb::GetServerInfoRequest {});
                request.set_timeout(Duration::from_secs(5));
                let info = client
                    .get_server_info(request)
                    .await
                    .map_err(|e| format!("grpc GetServerInfo {uri}: {e}"))?
                    .into_inner();
                Ok(info.supports_prefill_progress)
            },
        )
        .await
    }

    pub fn remove_client(&self, worker_url: &str) {
        if let Ok(uri) = grpc_connect_uri(worker_url) {
            self.clients.remove(&uri);
            self.prefill_progress_capabilities.remove(&uri);
        }
    }

    /// P/D prefill leg: one sampled token. Returns the engine's
    /// `FinishInfo.kv_transfer_params`, which the decode leg consumes.
    ///
    /// With `on_prefill_progress` and a progress-capable worker it streams,
    /// so remaining-work snapshots replace the admission estimate while P
    /// runs. Otherwise it stays unary `Generate`, as progress opt-in is
    /// rejected there.
    pub(crate) async fn prefill_prepared(
        &self,
        worker_url: &str,
        prepared: &PreparedRequest,
        engine_request_id: String,
        kv_transfer_params: &serde_json::Value,
        on_prefill_progress: Option<PrefillProgressCallback>,
    ) -> Result<Option<serde_json::Value>, Response> {
        let deadline = tokio::time::Instant::now()
            + self
                .request_timeout
                .saturating_sub(prepared.t_req.elapsed());
        let include_prefill_progress =
            self.prefill_progress_stale_after.is_some() && on_prefill_progress.is_some();
        let mut proto = chat_to_generate_request_with_progress(
            prepared.request.as_ref(),
            prepared.tokenized.token_ids.as_ref().to_vec(),
            engine_request_id.clone(),
            include_prefill_progress,
        )
        .map_err(|error| (StatusCode::BAD_REQUEST, error).into_response())?;
        limit_to_prefill(&mut proto);
        set_kv_transfer_params(&mut proto, kv_transfer_params)
            .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error).into_response())?;
        if include_prefill_progress {
            let supported =
                tokio::time::timeout_at(deadline, self.supports_prefill_progress(worker_url))
                    .await
                    .unwrap_or_else(
                        |_| Err("prefill progress capability deadline exceeded".into()),
                    );
            apply_prefill_progress_capability(&mut proto, supported, worker_url);
        }
        let mut progress = PrefillProgressState::new(
            engine_request_id.clone(),
            on_prefill_progress.filter(|_| proto.include_prefill_progress),
            self.prefill_progress_stale_after,
        );
        let streaming = proto.include_prefill_progress;
        let rpc = if streaming {
            "GenerateStream"
        } else {
            "Generate"
        };

        let channel = match tokio::time::timeout_at(deadline, self.channel_for(worker_url)).await {
            Ok(Ok(channel)) => channel,
            Ok(Err(error)) => {
                error!(error = %error, "grpc prefill connect failed");
                return Err((StatusCode::BAD_GATEWAY, error).into_response());
            }
            Err(_) => {
                return Err((
                    StatusCode::GATEWAY_TIMEOUT,
                    "gRPC request deadline exceeded",
                )
                    .into_response());
            }
        };
        let mut abort =
            EngineAbortGuard::new(&channel, Some(&engine_request_id), self.connect_timeout);
        let mut client = InferenceClient::new(channel);
        let mut tonic_req = tonic::Request::new(proto);
        tonic_req.set_timeout(deadline.saturating_duration_since(tokio::time::Instant::now()));
        if let Some(rank) = parse_dp_rank(worker_url) {
            tonic_req
                .metadata_mut()
                .insert(DP_RANK_METADATA, rank.to_string().parse().unwrap());
        }
        let failed = |status: tonic::Status| {
            warn!(code = ?status.code(), msg = %status.message(), "prefill {rpc} failed");
            (
                grpc_status_code(&status),
                format!("grpc {rpc}: {}", status.message()),
            )
                .into_response()
        };
        let timed_out = || {
            (
                StatusCode::GATEWAY_TIMEOUT,
                format!("grpc {rpc} exceeded request timeout"),
            )
                .into_response()
        };
        let finish = if streaming {
            let mut stream =
                match tokio::time::timeout_at(deadline, client.generate_stream(tonic_req)).await {
                    Ok(Ok(stream)) => {
                        abort.answered();
                        stream.into_inner()
                    }
                    Ok(Err(status)) => {
                        abort.disarm();
                        return Err(failed(status));
                    }
                    Err(_) => return Err(timed_out()),
                };
            let mut finish = None;
            // A dropped stream (timeout, cancelled leg) leaves `abort` armed.
            loop {
                match next_response(&mut stream, deadline, &mut progress).await {
                    Ok(Some(Ok(message))) => {
                        if let Some(info) = message.outputs.and_then(|outputs| outputs.finish_info)
                        {
                            finish = Some(info);
                        }
                    }
                    Ok(Some(Err(status))) => return Err(failed(status)),
                    Ok(None) => break,
                    Err(_) => return Err(timed_out()),
                }
            }
            if finish.is_none() {
                return Err((
                    StatusCode::BAD_GATEWAY,
                    "grpc GenerateStream: prefill closed before finish_info",
                )
                    .into_response());
            }
            abort.disarm();
            finish
        } else {
            match tokio::time::timeout_at(deadline, client.generate(tonic_req)).await {
                Ok(Ok(response)) => {
                    abort.disarm();
                    response
                        .into_inner()
                        .outputs
                        .and_then(|outputs| outputs.finish_info)
                }
                Ok(Err(status)) => {
                    abort.disarm();
                    return Err(failed(status));
                }
                Err(_) => return Err(timed_out()),
            }
        };
        if finish
            .as_ref()
            .is_some_and(|finish| finish.finish_reason == FinishReason::Aborted as i32)
        {
            warn!(%engine_request_id, "prefill {rpc} finished aborted");
            return Err((
                StatusCode::BAD_GATEWAY,
                format!("grpc {rpc}: prefill aborted"),
            )
                .into_response());
        }
        Ok(finish
            .and_then(|finish| finish.kv_transfer_params)
            .map(|params| proto_struct_to_json(&params)))
    }

    pub(crate) async fn dispatch_prepared(
        &self,
        worker_url: &str,
        prepared: PreparedRequest,
        on_first_output: Option<FirstOutputCallback>,
        on_prefill_progress: Option<PrefillProgressCallback>,
        handoff: KvHandoff,
    ) -> Response {
        let PreparedRequest {
            request,
            tokenized,
            handle,
            resolve_ms,
            t_req,
            shape,
            emit_token_ids,
        } = prepared;
        let request = request.as_ref();
        let deadline =
            tokio::time::Instant::now() + self.request_timeout.saturating_sub(t_req.elapsed());
        let include_prefill_progress =
            self.prefill_progress_stale_after.is_some() && on_prefill_progress.is_some();
        let n_prompt_tokens = tokenized.token_ids.len();
        let adapt_ms = tokenized.adapt_ms;
        let template_ms = tokenized.template_ms;
        let encode_ms = tokenized.encode_ms;
        let frontend_ms = tokenized.frontend_ms();
        let encode_backend = tokenized.encode_backend;
        let cached_tokens = tokenized.cached_tokens;

        let request_id = match shape {
            ResponseShape::Chat => format!("chatcmpl-{}", Uuid::new_v4()),
            ResponseShape::Completion => format!("cmpl-{}", Uuid::new_v4()),
        };
        let model = request.model.clone().unwrap_or_else(|| "unknown".into());
        let created = now_secs();
        let skip_special = request.skip_special_tokens;
        let detok_tokenizer = handle.tokenizer();
        let detok_prompt_ids = tokenized.token_ids.clone();
        let abort_request_id = handoff.engine_request_id.clone();
        // Progress snapshots echo the engine request id, which a P/D leg sets
        // apart from the OpenAI response id.
        let engine_request_id = handoff
            .engine_request_id
            .unwrap_or_else(|| request_id.clone());
        let mut proto = match chat_to_generate_request_with_progress(
            request,
            tokenized.token_ids.as_ref().to_vec(),
            engine_request_id.clone(),
            include_prefill_progress,
        ) {
            Ok(proto) => proto,
            Err(error) => return (StatusCode::BAD_REQUEST, error).into_response(),
        };
        if let Some(params) = &handoff.kv_transfer_params {
            if let Err(error) = set_kv_transfer_params(&mut proto, params) {
                return (StatusCode::INTERNAL_SERVER_ERROR, error).into_response();
            }
        }

        if include_prefill_progress {
            let supported =
                tokio::time::timeout_at(deadline, self.supports_prefill_progress(worker_url))
                    .await
                    .unwrap_or_else(
                        |_| Err("prefill progress capability deadline exceeded".into()),
                    );
            apply_prefill_progress_capability(&mut proto, supported, worker_url);
        }

        let prefill_progress = PrefillProgressState::new(
            engine_request_id,
            on_prefill_progress.filter(|_| proto.include_prefill_progress),
            self.prefill_progress_stale_after,
        );

        let t_connect = Instant::now();
        let channel = match tokio::time::timeout_at(deadline, self.channel_for(worker_url)).await {
            Ok(Ok(c)) => c,
            Ok(Err(e)) => {
                error!(error = %e, "grpc backend connect failed");
                return (StatusCode::BAD_GATEWAY, e).into_response();
            }
            Err(_) => {
                return (
                    StatusCode::GATEWAY_TIMEOUT,
                    "gRPC request deadline exceeded",
                )
                    .into_response();
            }
        };
        let mut abort =
            EngineAbortGuard::new(&channel, abort_request_id.as_deref(), self.connect_timeout);
        let mut client = InferenceClient::new(channel);
        let grpc_connect_ms = t_connect.elapsed().as_secs_f64() * 1000.0;

        let mut tonic_req = tonic::Request::new(proto);
        tonic_req.set_timeout(deadline.saturating_duration_since(tokio::time::Instant::now()));
        if let Some(rank) = parse_dp_rank(worker_url) {
            tonic_req
                .metadata_mut()
                .insert(DP_RANK_METADATA, rank.to_string().parse().unwrap());
        }

        let t_invoke = Instant::now();
        let stream =
            match tokio::time::timeout_at(deadline, client.generate_stream(tonic_req)).await {
                Ok(Ok(s)) => {
                    abort.answered();
                    s.into_inner()
                }
                Ok(Err(status)) => {
                    abort.disarm();
                    warn!(code = ?status.code(), msg = %status.message(), "GenerateStream failed");
                    return (
                        grpc_status_code(&status),
                        format!("grpc GenerateStream: {}", status.message()),
                    )
                        .into_response();
                }
                Err(_) => {
                    return (
                        StatusCode::GATEWAY_TIMEOUT,
                        "grpc GenerateStream exceeded request timeout",
                    )
                        .into_response();
                }
            };
        let grpc_invoke_ms = t_invoke.elapsed().as_secs_f64() * 1000.0;

        let xfer_ms = grpc_connect_ms + grpc_invoke_ms;
        let stages = serde_json::json!({
            "path": "grpc",
            "encode_backend": encode_backend,
            "n_prompt_tokens": n_prompt_tokens,
            "resolve_ms": resolve_ms,
            "adapt_ms": adapt_ms,
            "template_ms": template_ms,
            "encode_ms": encode_ms,
            "encode_cached_tokens": cached_tokens,
            "frontend_ms": frontend_ms,
            "xfer_ms": xfer_ms,
            "grpc_connect_ms": grpc_connect_ms,
            "grpc_invoke_ms": grpc_invoke_ms,
        });

        let reply = ChatReply {
            request_id,
            model,
            created,
            tok: detok_tokenizer,
            prompt_ids: detok_prompt_ids,
            skip_special,
            prefer_worker_text: requires_worker_output_text(request),
            include_usage: request
                .stream_options
                .as_ref()
                .and_then(|options| options.include_usage)
                .unwrap_or(false),
            shape,
            emit_token_ids,
            deadline,
            stages,
            t_req,
            on_first_output,
            prefill_progress,
            fail_on_abort: abort_request_id.is_some(),
            abort,
        };
        if request.stream {
            stream_openai(stream, reply, n_prompt_tokens).await
        } else {
            collect_openai(stream, reply).await
        }
    }
}

fn stages_header(stages: &serde_json::Value) -> HeaderValue {
    HeaderValue::from_str(&stages.to_string()).unwrap_or_else(|_| HeaderValue::from_static("{}"))
}

struct ChatReply {
    request_id: String,
    model: String,
    created: u64,
    tok: Option<DynTokenizer>,
    prompt_ids: Arc<[u32]>,
    skip_special: bool,
    prefer_worker_text: bool,
    include_usage: bool,
    shape: ResponseShape,
    emit_token_ids: bool,
    deadline: tokio::time::Instant,
    stages: serde_json::Value,
    t_req: Instant,
    on_first_output: Option<FirstOutputCallback>,
    prefill_progress: PrefillProgressState,
    /// A P/D decode leg reports an `abort` finish as an error, so the push
    /// path can cancel P instead of returning an aborted reply as success.
    fail_on_abort: bool,
    abort: EngineAbortGuard,
}

struct PrefillProgressState {
    request_id: String,
    callback: Option<PrefillProgressCallback>,
    stale_after: Option<Duration>,
    last_received: tokio::time::Instant,
    sequence: u64,
    pending: bool,
    stale: bool,
}

impl PrefillProgressState {
    fn new(
        request_id: String,
        callback: Option<PrefillProgressCallback>,
        stale_after: Option<Duration>,
    ) -> Self {
        Self {
            request_id,
            pending: callback.is_some(),
            callback,
            stale_after,
            last_received: tokio::time::Instant::now(),
            sequence: 0,
            stale: false,
        }
    }

    fn observe(&mut self, progress: &super::pb::PrefillProgress) {
        let Some(callback) = &self.callback else {
            return;
        };
        let outcome = if progress.request_id != self.request_id {
            "ignored_identity"
        } else if progress.sequence <= self.sequence {
            "ignored_sequence"
        } else {
            "accepted"
        };
        metrics::counter!("vllm_router_prefill_progress_updates_total", "outcome" => outcome)
            .increment(1);
        if outcome != "accepted" {
            return;
        }
        if self.sequence != 0 {
            metrics::histogram!("vllm_router_prefill_progress_update_gap_seconds")
                .record(self.last_received.elapsed().as_secs_f64());
        }
        self.sequence = progress.sequence;
        self.last_received = tokio::time::Instant::now();
        self.pending = progress.remaining_compute_tokens != 0;
        self.stale = false;
        callback(
            progress.sequence,
            u64::from(progress.remaining_compute_tokens),
        );
    }

    fn next_deadline(&self, deadline: tokio::time::Instant) -> tokio::time::Instant {
        if self.pending && !self.stale {
            if let Some(stale_after) = self.stale_after {
                return deadline.min(self.last_received + stale_after);
            }
        }
        deadline
    }

    fn mark_stale(&mut self) {
        self.stale = true;
        metrics::counter!("vllm_router_prefill_progress_stale_total").increment(1);
        warn!("prefill progress is stale; retaining last-known remaining work");
    }
}

async fn next_response<S>(
    stream: &mut S,
    deadline: tokio::time::Instant,
    progress: &mut PrefillProgressState,
) -> Result<Option<Result<super::pb::GenerateResponse, tonic::Status>>, tokio::time::error::Elapsed>
where
    S: futures_util::Stream<Item = Result<super::pb::GenerateResponse, tonic::Status>> + Unpin,
{
    loop {
        let next_deadline = progress.next_deadline(deadline);
        match tokio::time::timeout_at(next_deadline, stream.next()).await {
            Err(_) if next_deadline < deadline => progress.mark_stale(),
            Ok(Some(Ok(message))) => {
                if let Some(snapshot) = &message.prefill_progress {
                    progress.observe(snapshot);
                }
                return Ok(Some(Ok(message)));
            }
            result => return result,
        }
    }
}

fn fill_engine_ms(stages: &mut serde_json::Value) {
    // Diagnostic residual only. This subtracts router frontend and gRPC
    // invoke spans from first output; it is not direct EngineCore telemetry.
    let first = stages.get("first_token_ms").and_then(|v| v.as_f64());
    let pre = stages
        .get("resolve_ms")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        + stages
            .get("frontend_ms")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
        + stages
            .get("xfer_ms")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
    if let Some(first) = first {
        stages["engine_ms"] = serde_json::json!(first - pre);
    }
}

async fn stream_openai(
    mut stream: tonic::Streaming<super::pb::GenerateResponse>,
    reply: ChatReply,
    n_prompt_tokens: usize,
) -> Response {
    let ChatReply {
        request_id,
        model,
        created,
        tok,
        prompt_ids,
        skip_special,
        prefer_worker_text,
        include_usage,
        shape,
        emit_token_ids,
        deadline,
        mut stages,
        t_req,
        on_first_output,
        mut prefill_progress,
        fail_on_abort: _,
        abort,
    } = reply;
    let header_stages = stages.clone();
    let stages_on = crate::backend::stages_enabled();
    let (tx, rx) = mpsc::channel::<Result<String, std::io::Error>>(STREAM_BUFFER_CAPACITY);
    let (failure_tx, failure_rx) = oneshot::channel();
    let (mut producer_output_frame, output_frame) = output_frame_tracking();
    let (failed_tx, failed_rx) = watch::channel(false);
    let emit = async move {
        let mut detok = tok
            .as_ref()
            .map(|t| decode_stream(&**t, prompt_ids.as_ref(), skip_special));
        let mut first = true;
        let mut first_output_seen = false;
        let mut first_token_logged = false;
        let mut prompt_tokens = n_prompt_tokens as u32;
        let mut completion_tokens = 0u32;
        loop {
            let item = match next_response(&mut stream, deadline, &mut prefill_progress).await {
                Ok(Some(item)) => item,
                Ok(None) => break,
                Err(_) => {
                    let _ = send_stream_chunk(
                        &tx,
                        &mut producer_output_frame,
                        "data: {\"error\":\"gRPC stream exceeded request timeout\"}\n\n"
                            .to_string(),
                        false,
                    )
                    .await;
                    return false;
                }
            };
            match item {
                Ok(msg) => {
                    if stages.get("first_grpc_msg_ms").is_none() {
                        stages["first_grpc_msg_ms"] =
                            serde_json::json!(t_req.elapsed().as_secs_f64() * 1000.0);
                    }
                    if let Some(info) = &msg.prompt_info {
                        prompt_tokens = info.num_prompt_tokens;
                    }
                    if let Some(outputs) = &msg.outputs {
                        completion_tokens = completion_tokens.saturating_add(outputs.num_tokens);
                    }
                    let (text, finish) =
                        match output_text_and_finish(&msg, &mut detok, prefer_worker_text) {
                            Ok(output) => output,
                            Err(error) => {
                                let _ = send_stream_chunk(
                                    &tx,
                                    &mut producer_output_frame,
                                    format!(
                                        "data: {{\"error\":{}}}\n\n",
                                        serde_json::to_string(&error).unwrap_or_default()
                                    ),
                                    false,
                                )
                                .await;
                                return false;
                            }
                        };
                    let generated_output = has_generated_output(&msg, &text);
                    if prefill_progress.callback.is_some() {
                        notify_first_output(
                            &msg,
                            &text,
                            &mut first_output_seen,
                            on_first_output.as_ref(),
                        );
                    }
                    if !first_token_logged && !text.is_empty() {
                        stages["first_token_ms"] =
                            serde_json::json!(t_req.elapsed().as_secs_f64() * 1000.0);
                        stages["first_detok_ms"] = stages["first_token_ms"].clone();
                        fill_engine_ms(&mut stages);
                        if stages_on {
                            info!(%stages, "grpc generate stages");
                            let _ = send_stream_chunk(
                                &tx,
                                &mut producer_output_frame,
                                format!(": router-stages {stages}\n\n"),
                                false,
                            )
                            .await;
                        }
                        first_token_logged = true;
                    }
                    // Ids alone qualify: the decoder buffers byte-fallback
                    // fragments, so text can be empty while the frame holds
                    // tokens a token-exact client still needs.
                    let output_token_ids = if emit_token_ids {
                        msg.outputs
                            .as_ref()
                            .map(|outputs| outputs.token_ids.clone())
                            .filter(|ids| !ids.is_empty())
                    } else {
                        None
                    };
                    if !text.is_empty()
                        || output_token_ids.is_some()
                        || finish.is_some()
                        || (first && shape == ResponseShape::Chat && msg.prompt_info.is_some())
                    {
                        let line = sse_chunk(
                            shape,
                            &request_id,
                            &model,
                            created,
                            &text,
                            first,
                            finish.clone(),
                            output_token_ids,
                        );
                        if !send_stream_chunk(
                            &tx,
                            &mut producer_output_frame,
                            line,
                            generated_output,
                        )
                        .await
                        {
                            break;
                        }
                        first = false;
                    }
                    if finish.is_some() {
                        if include_usage {
                            let usage = Usage {
                                prompt_tokens,
                                completion_tokens,
                                total_tokens: prompt_tokens + completion_tokens,
                                completion_tokens_details: None,
                            };
                            let line = sse_usage_chunk(shape, &request_id, &model, created, usage);
                            let _ = send_stream_chunk(&tx, &mut producer_output_frame, line, false)
                                .await;
                        }
                        let _ = send_stream_chunk(
                            &tx,
                            &mut producer_output_frame,
                            SSE_DONE.to_string(),
                            false,
                        )
                        .await;
                        return finish.as_deref() != Some("abort");
                    }
                }
                Err(status) => {
                    let _ = send_stream_chunk(
                        &tx,
                        &mut producer_output_frame,
                        format!(
                            "data: {{\"error\":{}}}\n\n",
                            serde_json::to_string(status.message()).unwrap_or_default()
                        ),
                        false,
                    )
                    .await;
                    return false;
                }
            }
        }
        let _ = send_stream_chunk(
            &tx,
            &mut producer_output_frame,
            "data: {\"error\":\"gRPC stream closed before finish_info\"}\n\n".to_string(),
            false,
        )
        .await;
        false
    };
    let producer = tokio::spawn(async move {
        let finished = match tokio::time::timeout_at(deadline, emit).await {
            Ok(finished) => finished,
            Err(_) => {
                let _ = failure_tx.send(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "gRPC response delivery exceeded request deadline",
                ));
                false
            }
        };
        if finished {
            abort.disarm();
        } else {
            failed_tx.send_replace(true);
        }
    });

    // Drain already accepted tokens first, then surface a blocked producer's
    // timeout without needing another slot in the full SSE queue.
    let tail = futures_util::stream::once(async move { failure_rx.await.ok().map(Err) })
        .filter_map(std::future::ready);
    let body = Body::from_stream(ReceiverStream::new(rx).chain(tail));
    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache");
    if stages_on {
        builder = builder.header("x-router-stages", stages_header(&header_stages));
    }
    let mut response = builder
        .body(body)
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response());
    response
        .extensions_mut()
        .insert(GrpcStreamTask::with_output_frame(producer, output_frame).with_failure(failed_rx));
    response
}

async fn collect_openai(
    mut stream: tonic::Streaming<super::pb::GenerateResponse>,
    reply: ChatReply,
) -> Response {
    let ChatReply {
        request_id,
        model,
        created,
        tok,
        prompt_ids,
        skip_special,
        prefer_worker_text,
        include_usage: _,
        shape,
        emit_token_ids,
        deadline,
        mut stages,
        t_req,
        on_first_output,
        mut prefill_progress,
        fail_on_abort,
        abort,
    } = reply;
    let mut detok = tok
        .as_ref()
        .map(|t| decode_stream(&**t, prompt_ids.as_ref(), skip_special));
    let mut text = String::new();
    let mut token_ids: Vec<u32> = Vec::new();
    let mut finish = None;
    let mut prompt_tokens = prompt_ids.len() as u32;
    let mut completion_tokens = 0u32;
    let mut first_output_seen = false;
    loop {
        let item = match next_response(&mut stream, deadline, &mut prefill_progress).await {
            Ok(Some(item)) => item,
            Ok(None) => break,
            Err(_) => {
                return (
                    StatusCode::GATEWAY_TIMEOUT,
                    "gRPC stream exceeded request timeout",
                )
                    .into_response();
            }
        };
        match item {
            Ok(msg) => {
                if stages.get("first_grpc_msg_ms").is_none() {
                    stages["first_grpc_msg_ms"] =
                        serde_json::json!(t_req.elapsed().as_secs_f64() * 1000.0);
                }
                if let Some(info) = &msg.prompt_info {
                    prompt_tokens = info.num_prompt_tokens;
                }
                let (delta, reason) =
                    match output_text_and_finish(&msg, &mut detok, prefer_worker_text) {
                        Ok(output) => output,
                        Err(error) => {
                            return (StatusCode::BAD_GATEWAY, error).into_response();
                        }
                    };
                notify_first_output(
                    &msg,
                    &delta,
                    &mut first_output_seen,
                    on_first_output.as_ref(),
                );
                if stages.get("first_token_ms").is_none() && !delta.is_empty() {
                    stages["first_token_ms"] =
                        serde_json::json!(t_req.elapsed().as_secs_f64() * 1000.0);
                    stages["first_detok_ms"] = stages["first_token_ms"].clone();
                    fill_engine_ms(&mut stages);
                }
                text.push_str(&delta);
                if let Some(outputs) = &msg.outputs {
                    completion_tokens = completion_tokens.saturating_add(outputs.num_tokens);
                    if emit_token_ids {
                        token_ids.extend_from_slice(&outputs.token_ids);
                    }
                }
                if reason.is_some() {
                    finish = reason;
                }
            }
            Err(status) => {
                return (
                    StatusCode::BAD_GATEWAY,
                    format!("grpc stream: {}", status.message()),
                )
                    .into_response();
            }
        }
    }
    if finish.is_none() {
        return (
            StatusCode::BAD_GATEWAY,
            "gRPC stream closed before finish_info",
        )
            .into_response();
    }
    abort.disarm();
    if fail_on_abort && finish.as_deref() == Some("abort") {
        warn!(%request_id, "gRPC decode finished aborted");
        return (StatusCode::BAD_GATEWAY, "gRPC decode finished aborted").into_response();
    }
    let mut resp = match shape {
        ResponseShape::Chat => axum::Json(final_response(
            &request_id,
            &model,
            created,
            text,
            finish,
            prompt_tokens,
            completion_tokens,
        ))
        .into_response(),
        ResponseShape::Completion => axum::Json(completion_final_response(
            &request_id,
            &model,
            created,
            text,
            finish,
            prompt_tokens,
            completion_tokens,
            emit_token_ids.then_some(token_ids),
            emit_token_ids.then(|| prompt_ids.to_vec()),
        ))
        .into_response(),
    };
    if crate::backend::stages_enabled() {
        resp.headers_mut()
            .insert("x-router-stages", stages_header(&stages));
        info!(%stages, "grpc generate stages");
    }
    resp
}

fn notify_first_output(
    msg: &super::pb::GenerateResponse,
    text: &str,
    first_output_seen: &mut bool,
    callback: Option<&FirstOutputCallback>,
) {
    if !*first_output_seen && has_generated_output(msg, text) {
        *first_output_seen = true;
        if let Some(callback) = callback {
            callback();
        }
    }
}

fn output_text_and_finish(
    msg: &super::pb::GenerateResponse,
    detok: &mut Option<Box<dyn IncrementalDecoderTrait + '_>>,
    prefer_worker_text: bool,
) -> Result<(String, Option<String>), String> {
    let Some(outputs) = &msg.outputs else {
        return Ok((String::new(), None));
    };
    let finish = outputs
        .finish_info
        .as_ref()
        .and_then(|info| finish_reason_name(info.finish_reason));
    if prefer_worker_text {
        return Ok((outputs.text.clone(), finish));
    }
    let has_decoder = detok.is_some();
    let mut decoded = String::new();
    if !outputs.token_ids.is_empty() {
        if let Some(decoder) = detok.as_deref_mut() {
            decoded = emit_detok(decoder, &outputs.token_ids)
                .map_err(|error| format!("detokenization failed: {error}"))?;
        }
    }
    // vLLM may send finish_info in a separate response with no token_ids.
    // Always flush on terminal output so buffered UTF-8/special-token state
    // cannot be silently dropped.
    if finish.is_some() {
        if let Some(decoder) = detok.as_deref_mut() {
            let (last, _) = decoder
                .flush(None)
                .map_err(|error| format!("detokenization flush failed: {error}"))?;
            if let Some(last) = last {
                decoded.push_str(&last);
            }
        }
    }
    let text = if outputs.token_ids.is_empty() || !has_decoder {
        outputs.text.clone()
    } else {
        decoded
    };
    Ok((text, finish))
}

fn apply_prefill_progress_capability(
    request: &mut super::pb::GenerateRequest,
    supported: Result<bool, String>,
    worker_url: &str,
) {
    let supported = match supported {
        Ok(supported) => supported,
        Err(error) => {
            warn!(worker_url, %error, "worker prefill progress probe failed; continuing without progress");
            false
        }
    };
    if !supported {
        request.include_prefill_progress = false;
    }
}

async fn cached_prefill_progress_capability<F, Fut>(
    cache: &DashMap<String, Arc<OnceCell<Result<bool, String>>>>,
    uri: String,
    probe: F,
) -> Result<bool, String>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<bool, String>>,
{
    let cell = cache
        .entry(uri.clone())
        .or_insert_with(|| Arc::new(OnceCell::new()))
        .clone();
    let result = cell.get_or_init(probe).await.clone();

    if result.is_err() {
        if let dashmap::mapref::entry::Entry::Occupied(entry) = cache.entry(uri) {
            if Arc::ptr_eq(entry.get(), &cell) {
                entry.remove();
            }
        }
    }

    result
}

fn has_generated_output(msg: &super::pb::GenerateResponse, text: &str) -> bool {
    msg.outputs.as_ref().is_some_and(|outputs| {
        outputs.num_tokens > 0 || !outputs.token_ids.is_empty() || !text.is_empty()
    })
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::backend::pb::{GenerateResponse, PromptInfo, SequenceOutput};
    use crate::policies::{InMemorySlotManager, KvRequestSlot, LoadProvider, SlotManager};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn progress(sequence: u64, remaining: u32) -> GenerateResponse {
        GenerateResponse {
            prefill_progress: Some(super::super::pb::PrefillProgress {
                request_id: "request".into(),
                sequence,
                prompt_tokens: 100,
                remaining_compute_tokens: remaining,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn request_slot() -> (
        Arc<InMemorySlotManager>,
        KvRequestSlot,
        crate::kv_index::RoutableTarget,
    ) {
        let manager = Arc::new(InMemorySlotManager::new());
        let target = crate::kv_index::RoutableTarget {
            instance_id: Arc::from("worker"),
            dp_rank: 0,
        };
        manager.begin_prefill(&target, 100);
        let slot = KvRequestSlot::new(manager.clone(), target.clone(), 100, None);
        (manager, slot, target)
    }

    #[tokio::test(start_paused = true)]
    async fn stale_progress_retains_work_until_the_absolute_deadline() {
        let (manager, slot, target) = request_slot();
        let callback_slot = slot.clone();
        let callback: PrefillProgressCallback = Arc::new(move |sequence, remaining| {
            callback_slot.update_prefill_progress(sequence, remaining);
        });
        let mut state = PrefillProgressState::new(
            "request".into(),
            Some(callback),
            Some(Duration::from_secs(2)),
        );
        let start = tokio::time::Instant::now();
        let deadline = start + Duration::from_secs(5);
        let (tx, rx) = mpsc::channel(2);
        let mut stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        tx.send(Ok(progress(2, 40))).await.unwrap();
        next_response(&mut stream, deadline, &mut state)
            .await
            .unwrap();
        tokio::time::advance(Duration::from_secs(1)).await;
        let mut wrong_attempt = progress(3, 90);
        wrong_attempt.prefill_progress.as_mut().unwrap().request_id = "other-request".into();
        tx.send(Ok(wrong_attempt)).await.unwrap();
        next_response(&mut stream, deadline, &mut state)
            .await
            .unwrap();
        assert_eq!(state.sequence, 2);
        assert_eq!(manager.sample(&target).active_prefill_tokens, 40);
        assert_eq!(
            state.next_deadline(deadline),
            start + Duration::from_secs(2)
        );
        assert!(next_response(&mut stream, deadline, &mut state)
            .await
            .is_err());
        assert!(state.stale);
        assert_eq!(tokio::time::Instant::now(), deadline);
        assert_eq!(manager.sample(&target).active_prefill_tokens, 40);
        slot.release();
        assert_eq!(manager.sample(&target).active_requests, 0);
        assert_eq!(manager.sample(&target).active_prefill_tokens, 0);
    }

    #[derive(Clone)]
    struct ScriptedWorker {
        frames: Vec<GenerateResponse>,
    }

    #[tonic::async_trait]
    impl super::super::pb::inference_server::Inference for ScriptedWorker {
        type GenerateStreamStream =
            tokio_stream::Iter<std::vec::IntoIter<Result<GenerateResponse, tonic::Status>>>;

        async fn generate(
            &self,
            _: tonic::Request<super::super::pb::GenerateRequest>,
        ) -> Result<tonic::Response<GenerateResponse>, tonic::Status> {
            Err(tonic::Status::unimplemented("streaming test worker"))
        }

        async fn generate_stream(
            &self,
            _: tonic::Request<super::super::pb::GenerateRequest>,
        ) -> Result<tonic::Response<Self::GenerateStreamStream>, tonic::Status> {
            Ok(tonic::Response::new(tokio_stream::iter(
                self.frames.iter().cloned().map(Ok).collect::<Vec<_>>(),
            )))
        }
    }

    #[tokio::test]
    async fn both_response_consumers_apply_reprefill_and_filter_progress_frames() {
        for mode in 0..3 {
            let streaming = mode != 0;
            let queue_full = Arc::new(tokio::sync::Notify::new());
            let mut frames = vec![
                progress(1, 60),
                GenerateResponse {
                    outputs: Some(SequenceOutput {
                        text: "A".into(),
                        num_tokens: 1,
                        token_ids: vec![1],
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                progress(3, 40),
                GenerateResponse {
                    outputs: Some(SequenceOutput {
                        text: "B".into(),
                        num_tokens: 1,
                        token_ids: vec![2],
                        finish_info: Some(super::super::pb::FinishInfo {
                            finish_reason: super::super::pb::finish_info::FinishReason::Stop as i32,
                            ..Default::default()
                        }),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ];
            for _ in 0..STREAM_BUFFER_CAPACITY * 2 {
                frames.insert(
                    2,
                    GenerateResponse {
                        outputs: Some(SequenceOutput {
                            text: "A".into(),
                            num_tokens: 1,
                            token_ids: vec![1],
                            ..Default::default()
                        }),
                        ..Default::default()
                    },
                );
            }
            frames.insert(STREAM_BUFFER_CAPACITY + 1, progress(2, 70));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(
                tonic::transport::Server::builder()
                    .add_service(super::super::pb::inference_server::InferenceServer::new(
                        ScriptedWorker { frames },
                    ))
                    .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
            );
            let mut client = InferenceClient::connect(format!("http://{address}"))
                .await
                .unwrap();
            let stream = client
                .generate_stream(super::super::pb::GenerateRequest::default())
                .await
                .unwrap()
                .into_inner();
            let (manager, slot, target) = request_slot();
            let output_slot = slot.clone();
            let progress_slot = slot.clone();
            let progress_queue_full = queue_full.clone();
            let reply = ChatReply {
                request_id: "request".into(),
                model: "model".into(),
                created: 0,
                tok: None,
                prompt_ids: Arc::from([1_u32; 100]),
                skip_special: true,
                prefer_worker_text: true,
                include_usage: false,
                shape: ResponseShape::Chat,
                emit_token_ids: false,
                deadline: tokio::time::Instant::now() + Duration::from_secs(5),
                stages: serde_json::json!({}),
                t_req: Instant::now(),
                on_first_output: Some(Arc::new(move || output_slot.begin_decode())),
                prefill_progress: PrefillProgressState::new(
                    "request".into(),
                    Some(Arc::new(move |sequence, remaining| {
                        progress_slot.update_prefill_progress(sequence, remaining);
                        if sequence == 2 {
                            progress_queue_full.notify_one();
                        }
                    })),
                    Some(Duration::from_secs(2)),
                ),
                fail_on_abort: false,
                abort: EngineAbortGuard {
                    target: None,
                    timeout: Duration::ZERO,
                    answered: false,
                },
            };
            let response = if streaming {
                stream_openai(stream, reply, 100).await
            } else {
                collect_openai(stream, reply).await
            };
            if mode == 2 {
                queue_full.notified().await;
                tokio::time::pause();
                tokio::time::advance(Duration::from_secs(6)).await;
                let task = response
                    .extensions()
                    .get::<GrpcStreamTask>()
                    .unwrap()
                    .handle
                    .lock()
                    .unwrap()
                    .take()
                    .unwrap();
                task.await.unwrap();
                let mut chunks = response.into_body().into_data_stream();
                let mut accepted = 0;
                while chunks.next().await.unwrap().is_ok() {
                    accepted += 1;
                }
                assert_eq!(accepted, STREAM_BUFFER_CAPACITY);
                assert!(chunks.next().await.is_none());
                assert_eq!(manager.sample(&target).active_prefill_tokens, 70);
                slot.release();
                assert_eq!(manager.sample(&target).active_requests, 0);
                assert_eq!(manager.sample(&target).active_prefill_tokens, 0);
                server.abort();
                tokio::time::resume();
                continue;
            }
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            if streaming {
                let text = std::str::from_utf8(&body).unwrap();
                assert_eq!(
                    text.matches("chat.completion.chunk").count(),
                    STREAM_BUFFER_CAPACITY * 2 + 2
                );
                assert!(text.contains("[DONE]"));
            } else {
                let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(
                    value["choices"][0]["message"]["content"],
                    format!("{}B", "A".repeat(STREAM_BUFFER_CAPACITY * 2 + 1))
                );
            }
            assert_eq!(manager.sample(&target).active_prefill_tokens, 40);
            assert_eq!(manager.sample(&target).active_requests, 1);
            slot.release();
            assert_eq!(manager.sample(&target).active_prefill_tokens, 0);
            assert_eq!(manager.sample(&target).active_requests, 0);
            server.abort();
        }
    }

    #[test]
    fn unsupported_worker_keeps_the_admission_estimate_path() {
        let mut request = crate::backend::pb::GenerateRequest {
            include_prefill_progress: true,
            ..Default::default()
        };

        apply_prefill_progress_capability(&mut request, Ok(false), "worker");

        assert!(!request.include_prefill_progress);
    }

    #[test]
    fn capability_probe_error_keeps_the_admission_estimate_path() {
        let mut request = crate::backend::pb::GenerateRequest {
            include_prefill_progress: true,
            ..Default::default()
        };

        apply_prefill_progress_capability(
            &mut request,
            Err("control service unavailable".to_owned()),
            "worker",
        );

        assert!(!request.include_prefill_progress);
    }

    #[tokio::test]
    async fn concurrent_capability_probes_are_coalesced_and_failures_can_retry() {
        let cache = Arc::new(DashMap::new());
        let probe_count = Arc::new(AtomicUsize::new(0));
        let mut requests = Vec::new();

        for _ in 0..16 {
            let cache = cache.clone();
            let probe_count = probe_count.clone();
            requests.push(tokio::spawn(async move {
                cached_prefill_progress_capability(&cache, "worker".to_owned(), || async move {
                    probe_count.fetch_add(1, Ordering::Relaxed);
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    Err("control service unavailable".to_owned())
                })
                .await
            }));
        }

        for request in requests {
            assert_eq!(
                request.await.unwrap(),
                Err("control service unavailable".to_owned())
            );
        }
        assert_eq!(probe_count.load(Ordering::Relaxed), 1);
        assert_eq!(
            cached_prefill_progress_capability(&cache, "worker".to_owned(), || async {
                probe_count.fetch_add(1, Ordering::Relaxed);
                Ok(true)
            })
            .await,
            Ok(true)
        );
        assert_eq!(probe_count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn prompt_info_does_not_count_as_generated_output() {
        let prompt_info = GenerateResponse {
            prompt_info: Some(PromptInfo {
                num_prompt_tokens: 32,
                ..Default::default()
            }),
            outputs: None,
            prefill_progress: None,
        };
        assert!(!has_generated_output(&prompt_info, ""));

        let decode_started = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let decode_started_callback = decode_started.clone();
        let callback: FirstOutputCallback = Arc::new(move || {
            decode_started_callback.fetch_add(1, Ordering::Relaxed);
        });
        let mut first_output_seen = false;
        notify_first_output(&prompt_info, "", &mut first_output_seen, Some(&callback));
        assert_eq!(decode_started.load(Ordering::Relaxed), 0);

        let output = GenerateResponse {
            prompt_info: None,
            outputs: Some(SequenceOutput {
                num_tokens: 1,
                token_ids: vec![42],
                ..Default::default()
            }),
            prefill_progress: None,
        };
        assert!(has_generated_output(&output, ""));
        notify_first_output(&output, "", &mut first_output_seen, Some(&callback));
        notify_first_output(&output, "", &mut first_output_seen, Some(&callback));
        assert_eq!(decode_started.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn output_frame_latches_to_the_first_generated_response() {
        let (tx, mut rx) = mpsc::channel(1);
        let (mut producer, mut consumer) = output_frame_tracking();

        assert!(send_stream_chunk(&tx, &mut producer, "prompt info".into(), false,).await);
        let mut blocked = Box::pin(send_stream_chunk(
            &tx,
            &mut producer,
            "first token".into(),
            true,
        ));
        assert!(futures_util::poll!(blocked.as_mut()).is_pending());
        assert_eq!(rx.recv().await.unwrap().unwrap(), "prompt info");
        assert!(!consumer.on_frame());
        assert!(blocked.await);
        assert_eq!(rx.recv().await.unwrap().unwrap(), "first token");
        assert!(consumer.on_frame());

        assert!(send_stream_chunk(&tx, &mut producer, "second token".into(), true,).await);
        assert_eq!(rx.recv().await.unwrap().unwrap(), "second token");
        assert!(consumer.on_frame());
        drop(rx);
        assert!(!send_stream_chunk(&tx, &mut producer, "cancelled".into(), true).await);
    }
}
