//! tonic client for vLLM rust `Inference.GenerateStream`.
//!
//! Tokenize on the router, send `token_ids`, detok the returned ids.
//! Channel cache is per connect URI.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    body::Body,
    http::{HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use dashmap::DashMap;
use futures_util::StreamExt;
use tokio::sync::{mpsc, watch};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::transport::Channel;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use super::convert::{
    chat_to_generate_request, chat_to_generate_request_with_options, completion_prompt_logprobs,
    limit_to_prefill, proto_struct_to_json, requires_worker_output_text, set_kv_transfer_params,
    CompletionOptions,
};
use super::detect::{grpc_connect_uri, parse_dp_rank};
use super::frontend::{PreparedRequest, ResponseShape};
use super::openai::{
    completion_response, final_response, finish_reason_name, now_secs, sse_chunk, sse_usage_chunk,
    SSE_DONE,
};
use super::outcome::{EngineOutcomeObserver, EngineUsage};
use super::pb;
use super::pb::control_client::ControlClient;
use super::pb::finish_info::FinishReason;
use super::pb::inference_client::InferenceClient;
use super::pb::AbortRequest;
use super::stream_output::{output_frame_tracking, OutputFrameConsumer, OutputFrameProducer};
use super::vllm_frontend::{decode_stream, emit_detok, DynTokenizer, IncrementalDecoderTrait};
use crate::protocols::spec::{CompletionChoice, LogProbs, PromptLogprob, PromptLogprobs, Usage};

const DP_RANK_METADATA: &str = "x-data-parallel-rank";
const ENGINE_ABORT_RETRY_DELAY: Duration = Duration::from_millis(500);

pub(crate) type FirstOutputCallback = Arc<dyn Fn() + Send + Sync + 'static>;

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
    #[cfg_attr(coverage_nightly, coverage(off))]
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

fn send_stream_chunk(
    tx: &mpsc::UnboundedSender<Result<String, std::io::Error>>,
    output: &mut OutputFrameProducer,
    chunk: String,
    has_generated_output: bool,
) -> bool {
    output.send(tx, chunk, has_generated_output)
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
    connect_timeout: Duration,
    request_timeout: Duration,
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
            connect_timeout,
            request_timeout,
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

    pub fn remove_client(&self, worker_url: &str) {
        if let Ok(uri) = grpc_connect_uri(worker_url) {
            self.clients.remove(&uri);
        }
    }

    /// P/D prefill leg as unary `Generate`. Returns the engine's
    /// `FinishInfo.kv_transfer_params`, which the decode leg consumes.
    pub(crate) async fn prefill_prepared(
        &self,
        worker_url: &str,
        prepared: &PreparedRequest,
        engine_request_id: String,
        kv_transfer_params: &serde_json::Value,
    ) -> Result<Option<serde_json::Value>, Response> {
        let mut proto = chat_to_generate_request(
            prepared.request.as_ref(),
            prepared.tokenized.token_ids.as_ref().to_vec(),
            engine_request_id.clone(),
        )
        .map_err(|error| (StatusCode::BAD_REQUEST, error).into_response())?;
        limit_to_prefill(&mut proto);
        set_kv_transfer_params(&mut proto, kv_transfer_params)
            .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error).into_response())?;

        let channel = self.channel_for(worker_url).await.map_err(|error| {
            error!(error = %error, "grpc prefill connect failed");
            (StatusCode::BAD_GATEWAY, error).into_response()
        })?;
        let abort = EngineAbortGuard::new(&channel, Some(&engine_request_id), self.connect_timeout);
        let mut client = InferenceClient::new(channel);
        let mut tonic_req = tonic::Request::new(proto);
        tonic_req.set_timeout(self.request_timeout);
        if let Some(rank) = parse_dp_rank(worker_url) {
            tonic_req
                .metadata_mut()
                .insert(DP_RANK_METADATA, rank.to_string().parse().unwrap());
        }
        let response = match tokio::time::timeout(self.request_timeout, client.generate(tonic_req))
            .await
        {
            Ok(Ok(response)) => {
                abort.disarm();
                response.into_inner()
            }
            Ok(Err(status)) => {
                abort.disarm();
                warn!(code = ?status.code(), msg = %status.message(), "prefill Generate failed");
                return Err((
                    grpc_status_code(&status),
                    format!("grpc Generate: {}", status.message()),
                )
                    .into_response());
            }
            Err(_) => {
                return Err((
                    StatusCode::GATEWAY_TIMEOUT,
                    "grpc Generate exceeded request timeout",
                )
                    .into_response());
            }
        };
        let finish = response.outputs.and_then(|outputs| outputs.finish_info);
        if finish
            .as_ref()
            .is_some_and(|finish| finish.finish_reason == FinishReason::Aborted as i32)
        {
            warn!(%engine_request_id, "prefill Generate finished aborted");
            return Err((StatusCode::BAD_GATEWAY, "grpc Generate: prefill aborted").into_response());
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
        handoff: KvHandoff,
        observer: Option<std::sync::Arc<EngineOutcomeObserver>>,
    ) -> Response {
        let PreparedRequest {
            request,
            tokenized,
            handle,
            resolve_ms,
            t_req,
            shape,
            emit_token_ids,
            completion,
        } = prepared;
        let request = request.as_ref();
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
        let mut proto = match chat_to_generate_request_with_options(
            request,
            tokenized.token_ids.as_ref().to_vec(),
            handoff
                .engine_request_id
                .unwrap_or_else(|| request_id.clone()),
            completion.as_ref(),
        ) {
            Ok(proto) => proto,
            Err(error) => return (StatusCode::BAD_REQUEST, error).into_response(),
        };
        if let Some(params) = &handoff.kv_transfer_params {
            if let Err(error) = set_kv_transfer_params(&mut proto, params) {
                return (StatusCode::INTERNAL_SERVER_ERROR, error).into_response();
            }
        }

        let t_connect = Instant::now();
        let channel = match self.channel_for(worker_url).await {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "grpc backend connect failed");
                return (StatusCode::BAD_GATEWAY, e).into_response();
            }
        };
        let mut abort =
            EngineAbortGuard::new(&channel, abort_request_id.as_deref(), self.connect_timeout);
        let mut client = InferenceClient::new(channel);
        let grpc_connect_ms = t_connect.elapsed().as_secs_f64() * 1000.0;

        let mut tonic_req = tonic::Request::new(proto);
        tonic_req.set_timeout(self.request_timeout);
        if let Some(rank) = parse_dp_rank(worker_url) {
            tonic_req
                .metadata_mut()
                .insert(DP_RANK_METADATA, rank.to_string().parse().unwrap());
        }

        let t_invoke = Instant::now();
        let stream =
            match tokio::time::timeout(self.request_timeout, client.generate_stream(tonic_req))
                .await
            {
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
            completion,
            request_timeout: self.request_timeout,
            stages,
            t_req,
            on_first_output,
            fail_on_abort: abort_request_id.is_some(),
            abort,
            observer,
        };
        // Stage 2 keeps `n = 1` (the completions lower rejects `n > 1` and
        // the engine rejects `num_sequences > 1`); the accumulator is
        // index-aware so Stage 3 fan-out reuses it unchanged.
        let n_choices = request.n.unwrap_or(1).max(1) as usize;
        if request.stream {
            stream_openai(stream, reply, n_choices).await
        } else {
            collect_openai(stream, reply, n_choices).await
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
    /// `/v1/completions` options (`echo`, `logprobs`, …); `None` for chat.
    completion: Option<CompletionOptions>,
    request_timeout: Duration,
    stages: serde_json::Value,
    t_req: Instant,
    on_first_output: Option<FirstOutputCallback>,
    /// A P/D decode leg reports an `abort` finish as an error, so the push
    /// path can cancel P instead of returning an aborted reply as success.
    fail_on_abort: bool,
    abort: EngineAbortGuard,
    observer: Option<std::sync::Arc<EngineOutcomeObserver>>,
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

/// Settle the producer's terminal `finish_reason` for a streaming attempt.
///
/// The body ends normally after `finish_reason="abort"`, so the attempt must
/// record an error (and therefore settle as failed) rather than succeeding on
/// EOF.
///
/// Test-only: the streaming and collect paths inline this settlement now,
/// so the helper survives only as the unit-test oracle for the contract.
#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
fn record_finish_outcome(observer: Option<&EngineOutcomeObserver>, finish: Option<&str>) -> bool {
    let aborted = finish == Some("abort");
    if aborted {
        if let Some(observer) = observer {
            observer.record_error();
        }
    }
    !aborted
}

/// One generated sequence inside one GenerateStream (`n > 1` splits by `index`).
struct SequenceAccum {
    text: String,
    token_ids: Vec<u32>,
    logprobs: LogProbs,
    finish: Option<String>,
    num_tokens: u32,
    echoed: bool,
    /// Character offset of the next output token in the client-visible text.
    next_offset: u32,
}

impl SequenceAccum {
    fn new(next_offset: u32) -> Self {
        Self {
            text: String::new(),
            token_ids: Vec::new(),
            logprobs: empty_logprobs(),
            finish: None,
            num_tokens: 0,
            echoed: false,
            next_offset,
        }
    }
}

fn empty_logprobs() -> LogProbs {
    LogProbs {
        tokens: Vec::new(),
        token_logprobs: Vec::new(),
        top_logprobs: Vec::new(),
        text_offset: Vec::new(),
    }
}

fn optional_values(values: &[f32]) -> Vec<Option<f32>> {
    values.iter().map(|value| Some(*value)).collect()
}

/// OpenAI serializes `-inf` logprobs as `-9999.0`.
fn logprob_value(value: f32) -> f32 {
    if value.is_nan() {
        -9999.0
    } else {
        value.max(-9999.0)
    }
}

fn token_text(
    tok: Option<&DynTokenizer>,
    id: u32,
    as_token_ids: bool,
    skip_special: bool,
) -> Result<String, String> {
    if as_token_ids {
        return Ok(format!("token_id:{id}"));
    }
    match tok {
        Some(tokenizer) => tokenizer
            .decode(&[id], skip_special)
            .map_err(|error| format!("logprob detokenization failed: {error}")),
        None => Ok(String::new()),
    }
}

/// Text and character width of one token in the client-visible text.
///
/// The token is fed through an incremental decoder that holds incomplete
/// UTF-8 sequences across tokens, so a token carrying only part of a
/// multi-byte character contributes "" here and the token completing the
/// character contributes the whole character. Decoding the token in
/// isolation instead would put one replacement character per fragment
/// into `logprobs.tokens` while `choices[].text` shows the real
/// character; vLLM corrects this the same way, with context, in
/// `LogprobsProcessor._correct_decoded_token`.
fn decode_token_contextual(
    decoder: &mut dyn IncrementalDecoderTrait,
    id: u32,
) -> Result<(String, u32), String> {
    decoder
        .push_token(id)
        .map_err(|error| format!("logprob detokenization failed: {error}"))?;
    let text = decoder
        .next_chunk()
        .map(|chunk| chunk.text)
        .unwrap_or_default();
    let width = text.chars().count() as u32;
    Ok((text, width))
}

/// Flush a finished sequence's offset decoder into its logprobs.
///
/// When generation ends mid-character, the text decoder's flush appends
/// the replacement character to `choices[].text` (see
/// `sequence_text_and_finish`), but the offset decoder still holds the
/// buffered bytes, so the last sampled token would report "" while the
/// text shows `�`. Give the flushed text to the last token instead, so
/// the tokens still concatenate to the text. Returns the flushed
/// character count, for the offset of any following entry.
///
/// `sampled_selected` says whether the sampled token passes the TokenIds
/// selection (true when there is no selection). The sampled token's
/// top_logprobs entry moves to the flushed text only when it is in the
/// map at all: an entry under the old text may belong to a candidate
/// that happens to share it (e.g. a skipped special token), and equal
/// scores do not establish ownership — so the move is gated on the
/// selection, not on the score. The sampled entry takes precedence when
/// the flushed key collides with a candidate's.
///
/// `displaced` is the candidate the sampled entry displaced at the last
/// position (see append_logprobs): after the flush the two entries have
/// distinct keys, so it is restored under the old text here.
fn flush_logprobs_tail(
    logprobs: &mut LogProbs,
    decoder: &mut Option<Box<dyn IncrementalDecoderTrait + '_>>,
    sampled_selected: bool,
    displaced: Option<(String, f32)>,
) -> Result<u32, String> {
    let Some(decoder) = decoder.as_deref_mut() else {
        return Ok(0);
    };
    let (last, _) = decoder
        .flush(None)
        .map_err(|error| format!("logprob detokenization flush failed: {error}"))?;
    let Some(last) = last else {
        return Ok(0);
    };
    let Some(token) = logprobs.tokens.last_mut() else {
        return Ok(0);
    };
    let old_text = token.clone();
    token.push_str(&last.text);
    // The sampled token's top_logprobs key was its pre-flush text; move
    // that entry to the flushed text so the sampled score stays findable
    // by the token's reported text. append_logprobs inserts the sampled
    // entry with precedence, so the entry under the old text is the
    // sampled entry exactly when the sampled token is in the map.
    let sampled_in_top =
        sampled_selected && logprobs.token_logprobs.last().copied().flatten().is_some();
    if sampled_in_top {
        if let Some(top) = logprobs.top_logprobs.last_mut().and_then(Option::as_mut) {
            if let Some(value) = top.remove(&old_text) {
                top.insert(token.clone(), value);
            }
        }
    }
    // Restore a candidate the sampled entry displaced at the last
    // position: after the flush the two entries have distinct keys, so
    // both can be reported.
    if let Some((text, value)) = displaced {
        if let Some(top) = logprobs.top_logprobs.last_mut().and_then(Option::as_mut) {
            top.entry(text).or_insert(value);
        }
    }
    Ok(last.text.chars().count() as u32)
}

/// Append one chunk's token logprobs; returns the next character offset
/// and, when the sampled entry at the last position displaced a different
/// token's candidate that decoded to the same text, that displaced
/// (text, score) — the terminal flush can give the two entries distinct
/// keys, so the caller can restore it then. Earlier positions keep
/// sampled-wins: their keys are final, so a displaced candidate there
/// cannot be reported either way.
///
/// `decoder` is the incremental decoder for this sequence, primed with
/// the prompt ids and fed the same output ids in the same order as the
/// text decoder; it must be the same decoder across chunks so a
/// multi-byte character split across SSE frames still counts once.
/// It is ignored in `token_id:<id>` label mode (labels have no UTF-8
/// state) and unused when there is no tokenizer. It is taken as a
/// mutable reference to the `Option` (like `sequence_text_and_finish`
/// takes the text decoder) so the borrow stays local to this call.
#[allow(clippy::too_many_arguments)]
fn append_logprobs(
    logprobs: &mut LogProbs,
    tok: Option<&DynTokenizer>,
    decoder: &mut Option<Box<dyn IncrementalDecoderTrait + '_>>,
    ids: &[u32],
    values: &[Option<f32>],
    candidates: &[pb::CandidateTokenInfo],
    as_token_ids: bool,
    skip_special: bool,
    selected_ids: Option<&[u32]>,
    start_offset: u32,
) -> Result<(u32, Option<(String, f32)>), String> {
    let mut offset = start_offset;
    let mut displaced: Option<(String, f32)> = None;
    for (position, id) in ids.iter().enumerate() {
        // The sampled token's text comes from the same incremental
        // decoder that builds `choices[].text`, so the logprob tokens
        // concatenate to the completion text. In `token_id:<id>` label
        // mode there is no UTF-8 state, and without a tokenizer the
        // token is empty.
        let (token, width) = if as_token_ids {
            let token = format!("token_id:{id}");
            let width = token.chars().count() as u32;
            (token, width)
        } else if let Some(decoder) = decoder.as_deref_mut() {
            decode_token_contextual(decoder, *id)?
        } else {
            let token = token_text(tok, *id, as_token_ids, skip_special)?;
            debug_assert!(token.is_empty());
            (token, 0)
        };
        let value = values.get(position).copied().flatten().map(logprob_value);
        let mut top = std::collections::HashMap::new();
        // The last candidate with a different id that decoded to the
        // sampled token's text, if any — the sampled insert below
        // displaces that candidate's entry, and the terminal flush can
        // only restore it when it is a different token. Tracked
        // independently of a sampled-id copy, which shares the text
        // but is the same token reported twice: the copy can come
        // after the distinct candidate and overwrite its score in
        // the map, so the displaced score is captured here, not read
        // back from the map at the sampled insert.
        let mut colliding_candidate: Option<(u32, f32)> = None;
        if let Some(info) = candidates.get(position) {
            for candidate in &info.tokens {
                // A candidate with the sampled token's id is the same
                // token: give it the contextual text, or it would appear
                // twice — once under its isolated decode and once under
                // the sampled text (e.g. the byte completing an emoji
                // as both `�` and `😀`).
                let text = if candidate.id == *id {
                    token.clone()
                } else {
                    token_text(tok, candidate.id, as_token_ids, skip_special)?
                };
                let score = logprob_value(candidate.logprob);
                if text == token && candidate.id != *id {
                    colliding_candidate = Some((candidate.id, score));
                }
                top.insert(text, score);
            }
        }
        if let Some(value) = value {
            // The sampled token joins OpenAI's top_logprobs map — except
            // under a TokenIds selection, which reports logprobs only for
            // the requested ids. It takes precedence over a candidate
            // that decodes to the same text: the map is keyed by text,
            // and the reported token's score must be findable by that
            // text. (The flush in flush_logprobs_tail relies on this:
            // the entry under the sampled text is the sampled entry.)
            if selected_ids.is_none_or(|ids| ids.contains(id)) {
                top.insert(token.clone(), value);
                // The sampled entry displaced the colliding candidate,
                // if any. Only the last position can get it back: the
                // terminal flush gives the two entries distinct keys.
                if position + 1 == ids.len() {
                    if let Some((_, score)) = colliding_candidate {
                        displaced = Some((token.clone(), score));
                    }
                }
            }
        }
        logprobs.tokens.push(token.clone());
        logprobs.token_logprobs.push(value);
        logprobs.top_logprobs.push((!top.is_empty()).then_some(top));
        logprobs.text_offset.push(offset);
        offset = offset.saturating_add(width);
    }
    Ok((offset, displaced))
}

/// First generated logprob offset: the prompt length in the coordinate
/// space the logprob entries use. Normally that is the decoded prompt
/// text, but in token-ID display mode the prompt entries count
/// `token_id:<id>` label lengths, so generated entries must continue
/// from the end of the labels — restarting at the decoded length would
/// make the merged offsets go backward at the merge boundary. The base
/// applies only when prompt entries merge in front of the output
/// entries, which happens only under `echo`; prompt entries that live
/// in the separate `prompt_logprobs` field do not shift generated
/// offsets, so without echo the base stays 0 and generated offsets are
/// relative to the generated text.
fn prompt_offset_base(
    prompt_ids: &[u32],
    echo_text: Option<&str>,
    merge_prompt_logprobs: bool,
    as_token_ids: bool,
) -> u32 {
    if merge_prompt_logprobs && as_token_ids {
        prompt_ids
            .iter()
            .map(|id| format!("token_id:{id}").chars().count() as u32)
            .sum()
    } else if merge_prompt_logprobs {
        echo_text
            .map(|text| text.chars().count() as u32)
            .unwrap_or(0)
    } else {
        0
    }
}

/// Align per-position prompt scalars to the prompt ids: the engine sends
/// either one value per prompt token, or N-1 values with the unscorable
/// first token omitted. Returns `None` when the length matches neither
/// shape. The first position is forced to `None` — nothing predicts the
/// first prompt token.
fn aligned_prompt_scalars<T: Copy>(values: &[T], n: usize) -> Option<Vec<Option<T>>> {
    let mut aligned: Vec<Option<T>> = if values.len() == n {
        values.iter().map(|value| Some(*value)).collect()
    } else if values.len() + 1 == n {
        std::iter::once(None)
            .chain(values.iter().map(|value| Some(*value)))
            .collect()
    } else {
        return None;
    };
    if let Some(first) = aligned.first_mut() {
        *first = None;
    }
    Some(aligned)
}

/// Align per-position prompt candidate groups to the prompt ids, the same
/// way `aligned_prompt_scalars` aligns the scalars: the engine sends
/// either one group per prompt token, or N-1 groups with the unscorable
/// first token omitted. Returns `None` when the length matches neither
/// shape, in which case callers report no candidates rather than
/// misaligning them. The first position is forced to `None` — nothing
/// predicts the first prompt token. (The proto documents no length for
/// `candidate_tokens`, so the length is the only signal.)
fn aligned_prompt_candidates<'a>(
    candidates: &'a [pb::CandidateTokenInfo],
    n: usize,
) -> Option<Vec<Option<&'a pb::CandidateTokenInfo>>> {
    let mut aligned: Vec<Option<&'a pb::CandidateTokenInfo>> = if candidates.len() == n {
        candidates.iter().map(Some).collect()
    } else if candidates.len() + 1 == n {
        std::iter::once(None)
            .chain(candidates.iter().map(Some))
            .collect()
    } else {
        return None;
    };
    if let Some(first) = aligned.first_mut() {
        *first = None;
    }
    Some(aligned)
}

/// A proto3 `uint32` rank defaults to 0 when the engine does not send
/// one; vLLM ranks are 1-based, so 0 means "no rank".
fn nonzero_rank(rank: u32) -> Option<u32> {
    (rank != 0).then_some(rank)
}

/// Decoded text for one candidate token in the vLLM `prompt_logprobs`
/// shape: the real token text, not the `token_id:<id>` display label.
/// `None` without a tokenizer.
fn decoded_candidate_token(
    tok: Option<&DynTokenizer>,
    id: u32,
    skip_special: bool,
) -> Result<Option<String>, String> {
    match tok {
        Some(_) => Ok(Some(token_text(tok, id, false, skip_special)?)),
        None => Ok(None),
    }
}

/// Build the vLLM `prompt_logprobs` array from the engine's `PromptInfo`:
/// one entry per prompt position — `None` for the first position
/// (nothing predicts it), then a map from token id to candidate logprob.
/// Candidates come from `candidate_tokens` (id, logprob and rank are all
/// on the wire); the sampled token joins the map, as it does in the
/// OpenAI `top_logprobs`, with its rank from `PromptInfo.ranks`. The
/// OpenAI `LogProbs` shape cannot supply this: it drops ranks and keys
/// candidates by text, so the ids would not survive the conversion.
fn prompt_logprobs_array_from_info(
    info: &pb::PromptInfo,
    prompt_ids: &[u32],
    tok: Option<&DynTokenizer>,
    skip_special: bool,
) -> Result<PromptLogprobs, String> {
    let Some(values) = aligned_prompt_scalars(&info.logprobs, prompt_ids.len()) else {
        return Ok(Vec::new());
    };
    // Ranks are auxiliary: when they do not line up with the values,
    // report no ranks rather than dropping the logprobs.
    let ranks = aligned_prompt_scalars(&info.ranks, prompt_ids.len())
        .unwrap_or_else(|| vec![None; prompt_ids.len()]);
    // Candidates align like the scalars (see `aligned_prompt_candidates`):
    // indexing them by raw position would put C's candidates at B under
    // the N-1 encoding. A length matching neither shape reports no
    // candidates rather than misaligning them.
    let aligned_candidates = aligned_prompt_candidates(&info.candidate_tokens, prompt_ids.len())
        .unwrap_or_else(|| vec![None; prompt_ids.len()]);
    let mut out: PromptLogprobs = Vec::with_capacity(prompt_ids.len());
    for (position, id) in prompt_ids.iter().enumerate() {
        // The first prompt token has no preceding token to score.
        if position == 0 {
            out.push(None);
            continue;
        }
        let mut entries = std::collections::HashMap::new();
        if let Some(candidates) = aligned_candidates[position] {
            for candidate in &candidates.tokens {
                entries.insert(
                    candidate.id,
                    PromptLogprob {
                        logprob: logprob_value(candidate.logprob),
                        rank: nonzero_rank(candidate.rank),
                        decoded_token: decoded_candidate_token(tok, candidate.id, skip_special)?,
                    },
                );
            }
        }
        if let Some(value) = values[position] {
            if !entries.contains_key(id) {
                entries.insert(
                    *id,
                    PromptLogprob {
                        logprob: logprob_value(value),
                        rank: ranks[position].and_then(nonzero_rank),
                        decoded_token: decoded_candidate_token(tok, *id, skip_special)?,
                    },
                );
            }
        }
        out.push((!entries.is_empty()).then_some(entries));
    }
    Ok(out)
}

fn prompt_logprobs_from_info(
    info: &pb::PromptInfo,
    prompt_ids: &[u32],
    tok: Option<&DynTokenizer>,
    as_token_ids: bool,
    skip_special: bool,
    // Always None: prompt candidates are TopN/All on the wire, never
    // TokenIds, so the output-only ID selection must not filter the
    // sampled prompt token out of the prompt top_logprobs.
    selected_ids: Option<&[u32]>,
) -> Result<LogProbs, String> {
    let mut logprobs = empty_logprobs();
    // A one-token prompt legitimately has no scorable logprob (the N-1
    // encoding omits the unscorable first token); the alignment below
    // still reports the prompt token at offset 0 with a null scalar. A
    // length that matches neither shape means no data, and stays dropped.
    let Some(values) = aligned_prompt_scalars(&info.logprobs, prompt_ids.len()) else {
        return Ok(logprobs);
    };
    // Candidates align like the scalars (see `aligned_prompt_candidates`):
    // passing them raw would put B's candidates at A under the N-1
    // encoding. A length matching neither shape reports no candidates
    // rather than misaligning them.
    let aligned_candidates: Vec<pb::CandidateTokenInfo> =
        match aligned_prompt_candidates(&info.candidate_tokens, prompt_ids.len()) {
            Some(aligned) => aligned
                .into_iter()
                .map(|group| group.cloned().unwrap_or_default())
                .collect(),
            None => Vec::new(),
        };
    // Prompt entries decode the prompt ids from scratch, so the offset
    // decoder starts with no prompt prefix and is fed the prompt ids in
    // order — a multi-byte character split across prompt tokens still
    // counts once.
    let mut decoder = tok.map(|tokenizer| decode_stream(&**tokenizer, &[], skip_special));
    append_logprobs(
        &mut logprobs,
        tok,
        &mut decoder,
        prompt_ids,
        &values,
        &aligned_candidates,
        as_token_ids,
        skip_special,
        selected_ids,
        0,
    )?;
    Ok(logprobs)
}

fn error_frame(message: &str) -> String {
    format!(
        "data: {{\"error\":{}}}\n\n",
        serde_json::to_string(message).unwrap_or_default()
    )
}

async fn stream_openai(
    mut stream: tonic::Streaming<super::pb::GenerateResponse>,
    reply: ChatReply,
    n_choices: usize,
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
        completion,
        request_timeout,
        mut stages,
        t_req,
        on_first_output: _,
        fail_on_abort: _,
        abort,
        observer,
    } = reply;
    let echo = completion.as_ref().is_some_and(|options| options.echo);
    let echo_text = completion
        .as_ref()
        .and_then(|options| options.echo_text.clone());
    let want_logprobs = completion
        .as_ref()
        .is_some_and(|options| options.logprobs.is_some());
    let as_token_ids = completion
        .as_ref()
        .is_some_and(|options| options.return_tokens_as_token_ids);
    let echo_only = completion.as_ref().is_some_and(|options| options.echo_only);
    // Prompt logprobs are merged only when the request asked for them
    // (explicit `prompt_logprobs`, or `echo` + `logprobs`); an engine that
    // sends them unrequested must not leak them into the reply.
    let want_prompt_logprobs = completion
        .as_ref()
        .and_then(completion_prompt_logprobs)
        .is_some();
    let logprob_token_ids = completion
        .as_ref()
        .and_then(|options| options.logprob_token_ids.clone());
    // First generated logprob offset (see prompt_offset_base): only
    // under `echo`, when prompt entries merge in front.
    let logprob_offset_base = prompt_offset_base(
        &prompt_ids,
        echo_text.as_deref(),
        want_prompt_logprobs && echo,
        as_token_ids,
    );
    let header_stages = stages.clone();
    let stages_on = crate::backend::stages_enabled();
    let (tx, rx) = mpsc::unbounded_channel::<Result<String, std::io::Error>>();
    let (mut producer_output_frame, output_frame) = output_frame_tracking();
    let (failed_tx, failed_rx) = watch::channel(false);
    let producer = tokio::spawn(async move {
        let finished = async {
            let mut sequences: Vec<SequenceAccum> = (0..n_choices.max(1))
                .map(|_| SequenceAccum::new(logprob_offset_base))
                .collect();
            let mut detoks: Vec<Option<Box<dyn IncrementalDecoderTrait + '_>>> = Vec::new();
            // Offset decoders mirror the text decoders: one per sequence,
            // primed with the prompt ids and fed the same output ids in
            // the same order, so a multi-byte character split across
            // tokens — or across SSE frames — counts once in text_offset.
            let mut offset_detoks: Vec<Option<Box<dyn IncrementalDecoderTrait + '_>>> =
                Vec::new();
            let mut prompt_logprobs: Option<LogProbs> = None;
            // Prompt logprobs merge into the first chunk that carries output
            // logprobs, once they have arrived.
            let mut prompt_logprobs_sent = false;
            // Once a chunk carrying output logprobs has gone out, prompt
            // logprobs arriving later can no longer be merged in front:
            // they would land after the output entries with non-monotonic
            // offsets. They are dropped (with a warning) instead. The
            // proto documents `prompt_info` as present only in the first
            // response, so this only fires for a non-conforming engine.
            let mut sent_output_logprobs = false;
            let mut first = true;
            let mut first_token_logged = false;
            let mut prompt_tokens = prompt_ids.len() as u32;
            let mut engine_prompt_tokens: Option<u32> = None;
            let mut completion_tokens = 0u32;
            let mut finished_choices = 0usize;
            let mut all_done = false;
            while !all_done {
                let item = match tokio::time::timeout(request_timeout, stream.next()).await {
                    Ok(Some(item)) => item,
                    Ok(None) => break,
                    Err(_) => {
                        if let Some(observer) = &observer {
                            observer.record_error();
                        }
                        let _ = send_stream_chunk(
                            &tx,
                            &mut producer_output_frame,
                            error_frame("gRPC stream exceeded request timeout"),
                            false,
                        );
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
                            engine_prompt_tokens = Some(info.num_prompt_tokens);
                            if want_prompt_logprobs && prompt_logprobs.is_none() {
                                if sent_output_logprobs {
                                    warn!(%request_id, "dropping late streaming prompt logprobs: a chunk with output logprobs was already sent");
                                } else {
                                    match prompt_logprobs_from_info(
                                        info,
                                        &prompt_ids,
                                        tok.as_ref(),
                                        as_token_ids,
                                        skip_special,
                                        // Prompt entries carry no ID filter
                                        // (see prompt_logprobs_from_info).
                                        None,
                                    ) {
                                        Ok(value) => prompt_logprobs = Some(value),
                                        Err(error) => {
                                            let _ = send_stream_chunk(
                                                &tx,
                                                &mut producer_output_frame,
                                                error_frame(&error),
                                                false,
                                            );
                                            return false;
                                        }
                                    }
                                }
                            }
                        }
                        let Some(outputs) = &msg.outputs else {
                            continue;
                        };
                        let index = outputs.index as usize;
                        if index >= sequences.len() {
                            sequences.resize_with(index + 1, || SequenceAccum::new(logprob_offset_base));
                        }
                        while detoks.len() <= index {
                            detoks.push(tok.as_ref().map(|tokenizer| {
                                decode_stream(&**tokenizer, prompt_ids.as_ref(), skip_special)
                            }));
                        }
                        // Offset decoders are only read by append_logprobs
                        // (inside the want_logprobs branch below), so skip
                        // building them — and copying the prompt ids — for
                        // chat, P/D and any path with logprobs off.
                        if want_logprobs {
                            while offset_detoks.len() <= index {
                                offset_detoks.push(tok.as_ref().map(|tokenizer| {
                                    decode_stream(&**tokenizer, prompt_ids.as_ref(), skip_special)
                                }));
                            }
                        }
                        let (delta, finish) = match sequence_text_and_finish(
                            outputs,
                            &mut detoks[index],
                            prefer_worker_text,
                        ) {
                            Ok(output) => output,
                            Err(error) => {
                                if let Some(observer) = &observer {
                                    observer.record_error();
                                }
                                let _ = send_stream_chunk(
                                    &tx,
                                    &mut producer_output_frame,
                                    error_frame(&error),
                                    false,
                                );
                                return false;
                            }
                        };
                        let generated_output = has_generated_output(&msg, &delta);
                        if !first_token_logged && !delta.is_empty() {
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
                                );
                            }
                            first_token_logged = true;
                        }
                        let sequence = &mut sequences[index];
                        completion_tokens = completion_tokens.saturating_add(outputs.num_tokens);
                        sequence.num_tokens =
                            sequence.num_tokens.saturating_add(outputs.num_tokens);
                        let mut chunk_text = delta;
                        let mut chunk_logprobs: Option<LogProbs> = None;
                        let mut chunk_has_output_logprobs = false;
                        let echo_now = echo && !sequence.echoed;
                        if echo_now {
                            sequence.echoed = true;
                            if echo_only {
                                // `max_tokens: 0`: the prompt (when `echo` is
                                // set) is the whole answer; the generated
                                // token is dropped.
                                chunk_text = echo_text.clone().unwrap_or_default();
                            } else if let Some(text) = echo_text.as_deref() {
                                chunk_text = format!("{text}{chunk_text}");
                            }
                        } else if echo_only {
                            chunk_text.clear();
                        }
                        if echo_only {
                            if echo_now {
                                chunk_logprobs = prompt_logprobs.clone();
                            }
                        } else if want_logprobs {
                            // Prompt logprobs merge into the first chunk
                            // that carries output logprobs, if they arrived
                            // before any output chunk went out; a later
                            // PromptInfo is dropped at the prompt-info
                            // handler instead of landing after the output
                            // entries.
                            let mut logprobs = if !prompt_logprobs_sent {
                                match &prompt_logprobs {
                                    Some(prompt) => {
                                        prompt_logprobs_sent = true;
                                        prompt.clone()
                                    }
                                    None => empty_logprobs(),
                                }
                            } else {
                                empty_logprobs()
                            };
                            let values = optional_values(&outputs.logprobs);
                            let start = sequence.next_offset;
                            let tokens_before = logprobs.tokens.len();
                            let displaced = match append_logprobs(
                                &mut logprobs,
                                tok.as_ref(),
                                &mut offset_detoks[index],
                                &outputs.token_ids,
                                &values,
                                &outputs.candidate_tokens,
                                as_token_ids,
                                skip_special,
                                logprob_token_ids.as_deref(),
                                start,
                            ) {
                                Ok((next, d)) => {
                                    sequence.next_offset = next;
                                    chunk_has_output_logprobs =
                                        logprobs.tokens.len() > tokens_before;
                                    d
                                }
                                Err(error) => {
                                    let _ = send_stream_chunk(
                                        &tx,
                                        &mut producer_output_frame,
                                        error_frame(&error),
                                        false,
                                    );
                                    return false;
                                }
                            };
                            if finish.is_some() {
                                // The text decoder's terminal flush (see
                                // sequence_text_and_finish) puts a
                                // replacement character into the chunk
                                // text when generation ends mid-character;
                                // give the same text to the last sampled
                                // token so the chunk's tokens still
                                // concatenate to its text. When finish
                                // arrives in a separate message with no
                                // token ids, the last token already went
                                // out in an earlier chunk, so there is
                                // nothing to attach the flush to.
                                let sampled_selected = logprob_token_ids
                                    .as_deref()
                                    .is_none_or(|ids| {
                                        outputs
                                            .token_ids
                                            .last()
                                            .is_some_and(|id| ids.contains(id))
                                    });
                                match flush_logprobs_tail(
                                    &mut logprobs,
                                    &mut offset_detoks[index],
                                    sampled_selected,
                                    displaced,
                                ) {
                                    Ok(width) => {
                                        sequence.next_offset =
                                            sequence.next_offset.saturating_add(width);
                                    }
                                    Err(error) => {
                                        let _ = send_stream_chunk(
                                            &tx,
                                            &mut producer_output_frame,
                                            error_frame(&error),
                                            false,
                                        );
                                        return false;
                                    }
                                }
                            }
                            if !logprobs.tokens.is_empty() {
                                chunk_logprobs = Some(logprobs);
                            }
                        }
                        if emit_token_ids {
                            sequence.token_ids.extend_from_slice(&outputs.token_ids);
                        }
                        let chunk_ids = if echo_only {
                            // Echo-only carries no generated ids (the one
                            // token the engine had to generate is dropped),
                            // so `token_ids` is empty; the prompt ids go
                            // in `prompt_token_ids` below.
                            if echo_now && echo && emit_token_ids {
                                Some(Vec::new())
                            } else {
                                None
                            }
                        } else if emit_token_ids && (echo_now || !outputs.token_ids.is_empty()) {
                            // `token_ids` holds generated ids only, echo
                            // chunk included; the prompt ids ride in the
                            // separate `prompt_token_ids` field, as in
                            // vLLM HTTP.
                            Some(outputs.token_ids.clone())
                        } else {
                            None
                        };
                        // The echo chunk carries the prompt text, so the
                        // prompt ids go in the separate field, on the
                        // first echo chunk only.
                        let chunk_prompt_token_ids =
                            if echo_now && echo && emit_token_ids {
                                Some(prompt_ids.to_vec())
                            } else {
                                None
                            };
                        if !chunk_text.is_empty()
                            || chunk_logprobs.is_some()
                            || chunk_ids.is_some()
                            || chunk_prompt_token_ids.is_some()
                            || finish.is_some()
                            || (first && shape == ResponseShape::Chat)
                        {
                            let line = sse_chunk(
                                shape,
                                index as u32,
                                &request_id,
                                &model,
                                created,
                                &chunk_text,
                                first,
                                finish.clone(),
                                chunk_logprobs,
                                chunk_ids,
                                chunk_prompt_token_ids,
                            );
                            if !send_stream_chunk(
                                &tx,
                                &mut producer_output_frame,
                                line,
                                generated_output,
                            ) {
                                break;
                            }
                            if chunk_has_output_logprobs {
                                sent_output_logprobs = true;
                            }
                            first = false;
                        }
                        if finish.is_some() {
                            sequence.finish = finish;
                            finished_choices += 1;
                            if finished_choices >= n_choices.max(1) {
                                if let Some(observer) = &observer {
                                    observer.record_usage(EngineUsage {
                                        prompt_tokens: engine_prompt_tokens
                                            .map(|tokens| tokens as usize),
                                        completion_tokens: completion_tokens as usize,
                                    });
                                }
                                if include_usage {
                                    // Echo-only answers with zero generated
                                    // tokens: the engine's one token was
                                    // dropped, so the client usage reports
                                    // zero while the engine observer above
                                    // keeps the physical count.
                                    let client_completion_tokens =
                                        if echo_only { 0 } else { completion_tokens };
                                    let usage = Usage {
                                        prompt_tokens,
                                        completion_tokens: client_completion_tokens,
                                        total_tokens: prompt_tokens + client_completion_tokens,
                                        completion_tokens_details: None,
                                    };
                                    let line =
                                        sse_usage_chunk(shape, &request_id, &model, created, usage);
                                    let _ = send_stream_chunk(
                                        &tx,
                                        &mut producer_output_frame,
                                        line,
                                        false,
                                    );
                                }
                                let _ = send_stream_chunk(
                                    &tx,
                                    &mut producer_output_frame,
                                    SSE_DONE.to_string(),
                                    false,
                                );
                                all_done = true;
                            }
                        }
                    }
                    Err(status) => {
                        if let Some(observer) = &observer {
                            observer.record_error();
                        }
                        let _ = send_stream_chunk(
                            &tx,
                            &mut producer_output_frame,
                            error_frame(status.message()),
                            false,
                        );
                        return false;
                    }
                }
            }
            if !all_done {
                if let Some(observer) = &observer {
                    observer.record_error();
                }
                let _ = send_stream_chunk(
                    &tx,
                    &mut producer_output_frame,
                    error_frame("gRPC stream closed before finish_info"),
                    false,
                );
                return false;
            }
            // An `abort` finish is a failed decode leg for the P/D push path:
            // report it through the failure flag so P gets cancelled.
            // It is also an error for the outcome observer (#460).
            let aborted = sequences
                .iter()
                .any(|sequence| sequence.finish.as_deref() == Some("abort"));
            if aborted {
                if let Some(observer) = &observer {
                    observer.record_error();
                }
            }
            !aborted
        }
        .await;
        if finished {
            abort.disarm();
        } else {
            failed_tx.send_replace(true);
        }
    });

    let body = Body::from_stream(UnboundedReceiverStream::new(rx));
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
    n_choices: usize,
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
        completion,
        request_timeout,
        mut stages,
        t_req,
        on_first_output,
        fail_on_abort,
        abort,
        observer,
    } = reply;
    let echo = completion.as_ref().is_some_and(|options| options.echo);
    let echo_only = completion.as_ref().is_some_and(|options| options.echo_only);
    let echo_text = completion
        .as_ref()
        .and_then(|options| options.echo_text.clone());
    let want_logprobs = completion
        .as_ref()
        .is_some_and(|options| options.logprobs.is_some());
    let as_token_ids = completion
        .as_ref()
        .is_some_and(|options| options.return_tokens_as_token_ids);
    // Prompt logprobs are merged only when the request asked for them
    // (explicit `prompt_logprobs`, or `echo` + `logprobs`); an engine that
    // sends them unrequested must not leak them into the reply.
    let want_prompt_logprobs = completion
        .as_ref()
        .and_then(completion_prompt_logprobs)
        .is_some();
    // First generated logprob offset (see prompt_offset_base): only
    // under `echo`, when prompt entries merge in front.
    let logprob_offset_base = prompt_offset_base(
        &prompt_ids,
        echo_text.as_deref(),
        want_prompt_logprobs && echo,
        as_token_ids,
    );
    let mut sequences: Vec<SequenceAccum> = (0..n_choices.max(1))
        .map(|_| SequenceAccum::new(logprob_offset_base))
        .collect();
    let mut detoks: Vec<Option<Box<dyn IncrementalDecoderTrait + '_>>> = Vec::new();
    // Offset decoders mirror the text decoders (see stream_openai): one
    // per sequence, primed with the prompt ids, fed the same output ids
    // in the same order, so a split multi-byte character counts once.
    let mut offset_detoks: Vec<Option<Box<dyn IncrementalDecoderTrait + '_>>> = Vec::new();
    let mut prompt_logprobs: Option<LogProbs> = None;
    let mut prompt_logprobs_array: Option<PromptLogprobs> = None;
    let mut prompt_tokens = prompt_ids.len() as u32;
    let mut engine_prompt_tokens: Option<u32> = None;
    let mut completion_tokens = 0u32;
    let mut first_output_seen = false;
    loop {
        let item = match tokio::time::timeout(request_timeout, stream.next()).await {
            Ok(Some(item)) => item,
            Ok(None) => break,
            Err(_) => {
                if let Some(observer) = &observer {
                    observer.record_error();
                }
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
                    engine_prompt_tokens = Some(info.num_prompt_tokens);
                    if want_prompt_logprobs && prompt_logprobs_array.is_none() {
                        // The text-keyed scores are only consumed when
                        // they merge into `logprobs` under `echo` (see
                        // completion_choice_parts); without echo they
                        // are discarded, so skip building them — with
                        // `prompt_logprobs: -1` that repeats
                        // detokenization for every prompt position and
                        // vocabulary candidate. The array is always
                        // needed: it is the separate `prompt_logprobs`
                        // field. The guard is on the array so a later
                        // PromptInfo still does not rebuild it.
                        if echo && want_logprobs {
                            prompt_logprobs = Some(
                                match prompt_logprobs_from_info(
                                    info,
                                    &prompt_ids,
                                    tok.as_ref(),
                                    as_token_ids,
                                    skip_special,
                                    // Prompt entries carry no ID filter (see
                                    // prompt_logprobs_from_info).
                                    None,
                                ) {
                                    Ok(value) => value,
                                    Err(error) => {
                                        return (StatusCode::BAD_GATEWAY, error).into_response();
                                    }
                                },
                            );
                        }
                        prompt_logprobs_array = Some(
                            match prompt_logprobs_array_from_info(
                                info,
                                &prompt_ids,
                                tok.as_ref(),
                                skip_special,
                            ) {
                                Ok(value) => value,
                                Err(error) => {
                                    return (StatusCode::BAD_GATEWAY, error).into_response();
                                }
                            },
                        );
                    }
                }
                let Some(outputs) = &msg.outputs else {
                    continue;
                };
                let index = outputs.index as usize;
                if index >= sequences.len() {
                    sequences.resize_with(index + 1, || SequenceAccum::new(logprob_offset_base));
                }
                while detoks.len() <= index {
                    detoks.push(tok.as_ref().map(|tokenizer| {
                        decode_stream(&**tokenizer, prompt_ids.as_ref(), skip_special)
                    }));
                }
                // Offset decoders are only read by append_logprobs (inside
                // the want_logprobs branch below), so skip building them —
                // and copying the prompt ids — when logprobs are off.
                if want_logprobs {
                    while offset_detoks.len() <= index {
                        offset_detoks.push(tok.as_ref().map(|tokenizer| {
                            decode_stream(&**tokenizer, prompt_ids.as_ref(), skip_special)
                        }));
                    }
                }
                let (delta, reason) =
                    match sequence_text_and_finish(outputs, &mut detoks[index], prefer_worker_text)
                    {
                        Ok(output) => output,
                        Err(error) => {
                            if let Some(observer) = &observer {
                                observer.record_error();
                            }
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
                let sequence = &mut sequences[index];
                sequence.text.push_str(&delta);
                sequence.num_tokens = sequence.num_tokens.saturating_add(outputs.num_tokens);
                completion_tokens = completion_tokens.saturating_add(outputs.num_tokens);
                if emit_token_ids {
                    sequence.token_ids.extend_from_slice(&outputs.token_ids);
                }
                // A candidate the sampled entry displaced at the last
                // position, for the terminal flush to restore (see
                // append_logprobs); fresh per message.
                let displaced = if want_logprobs {
                    let values = optional_values(&outputs.logprobs);
                    let start = sequence.next_offset;
                    match append_logprobs(
                        &mut sequence.logprobs,
                        tok.as_ref(),
                        &mut offset_detoks[index],
                        &outputs.token_ids,
                        &values,
                        &outputs.candidate_tokens,
                        as_token_ids,
                        skip_special,
                        completion
                            .as_ref()
                            .and_then(|options| options.logprob_token_ids.as_deref()),
                        start,
                    ) {
                        Ok((next, d)) => {
                            sequence.next_offset = next;
                            d
                        }
                        Err(error) => {
                            return (StatusCode::BAD_GATEWAY, error).into_response();
                        }
                    }
                } else {
                    None
                };
                if reason.is_some() {
                    sequence.finish = reason;
                    // Same terminal flush as the stream path: the text
                    // decoder's flush puts a replacement character into
                    // the text when generation ends mid-character; give
                    // it to the last sampled token. Only when this
                    // message carried token ids — with a finish-only
                    // message sequence_text_and_finish falls back to
                    // outputs.text and discards the flush, so flushing
                    // the logprobs would add a character the text does
                    // not have.
                    if want_logprobs && !outputs.token_ids.is_empty() {
                        let sampled_selected = completion
                            .as_ref()
                            .and_then(|options| options.logprob_token_ids.as_deref())
                            .is_none_or(|ids| {
                                outputs.token_ids.last().is_some_and(|id| ids.contains(id))
                            });
                        match flush_logprobs_tail(
                            &mut sequence.logprobs,
                            &mut offset_detoks[index],
                            sampled_selected,
                            displaced,
                        ) {
                            Ok(width) => {
                                sequence.next_offset = sequence.next_offset.saturating_add(width);
                            }
                            Err(error) => {
                                return (StatusCode::BAD_GATEWAY, error).into_response();
                            }
                        }
                    }
                }
            }
            Err(status) => {
                if let Some(observer) = &observer {
                    observer.record_error();
                }
                return (
                    StatusCode::BAD_GATEWAY,
                    format!("grpc stream: {}", status.message()),
                )
                    .into_response();
            }
        }
    }
    if sequences.iter().any(|sequence| sequence.finish.is_none()) {
        if let Some(observer) = &observer {
            observer.record_error();
        }
        return (
            StatusCode::BAD_GATEWAY,
            "gRPC stream closed before finish_info",
        )
            .into_response();
    }
    if let Some(observer) = &observer {
        observer.record_usage(EngineUsage {
            prompt_tokens: engine_prompt_tokens.map(|tokens| tokens as usize),
            completion_tokens: completion_tokens as usize,
        });
    }
    abort.disarm();
    // Program settlement must see aborts even when this adapter preserves the
    // regular (non-PD) HTTP response status for compatibility.
    let aborted = sequences
        .iter()
        .any(|sequence| sequence.finish.as_deref() == Some("abort"));
    if aborted {
        if let Some(observer) = &observer {
            observer.record_error();
        }
    }
    if fail_on_abort && aborted {
        warn!(%request_id, "gRPC decode finished aborted");
        return (StatusCode::BAD_GATEWAY, "gRPC decode finished aborted").into_response();
    }
    // Echo-only answers with zero generated tokens: the engine's one token
    // was dropped, so the client usage reports zero while the engine
    // observer above keeps the physical count.
    let client_completion_tokens = if echo_only { 0 } else { completion_tokens };
    let usage = Usage {
        prompt_tokens,
        completion_tokens: client_completion_tokens,
        total_tokens: prompt_tokens + client_completion_tokens,
        completion_tokens_details: None,
    };
    let mut resp = if shape == ResponseShape::Chat {
        let sequence = &sequences[0];
        axum::Json(final_response(
            &request_id,
            &model,
            created,
            sequence.text.clone(),
            sequence.finish.clone(),
            prompt_tokens,
            completion_tokens,
        ))
        .into_response()
    } else {
        let mut choices: Vec<CompletionChoice> = Vec::with_capacity(sequences.len());
        for (index, sequence) in sequences.iter().enumerate() {
            let (text, logprobs, prompt_logprobs, token_ids) = completion_choice_parts(
                sequence,
                echo_text.as_deref(),
                echo,
                echo_only,
                want_logprobs,
                prompt_logprobs.as_ref(),
                prompt_logprobs_array.as_ref(),
                emit_token_ids,
            );
            choices.push(CompletionChoice {
                text,
                index: index as u32,
                logprobs,
                prompt_logprobs,
                finish_reason: sequence.finish.clone(),
                matched_stop: None,
                hidden_states: None,
                prompt_token_ids: emit_token_ids.then(|| prompt_ids.to_vec()),
                token_ids,
            });
        }
        axum::Json(completion_response(
            &request_id,
            &model,
            created,
            choices,
            usage,
        ))
        .into_response()
    };
    if crate::backend::stages_enabled() {
        resp.headers_mut()
            .insert("x-router-stages", stages_header(&stages));
        info!(%stages, "grpc generate stages");
    }
    resp
}

/// Merge prompt logprob entries in front of the output entries.
/// Callers pass the prompt entries only under `echo` *and* when output
/// logprobs were requested; without echo they live in the separate
/// `prompt_logprobs` field, and with echo but no `logprobs` request
/// `logprobs` stays null (vLLM builds it only when `request.logprobs`
/// is not None), so `logprobs` holds output entries only. Without
/// prompt entries this is just the output logprobs (or none when empty).
fn merge_prompt_logprobs(
    prompt_logprobs: Option<&LogProbs>,
    output: &LogProbs,
) -> Option<LogProbs> {
    match (prompt_logprobs, output.tokens.is_empty()) {
        (Some(prompt), false) => {
            let mut merged = prompt.clone();
            merged.tokens.extend(output.tokens.iter().cloned());
            merged
                .token_logprobs
                .extend(output.token_logprobs.iter().cloned());
            merged
                .top_logprobs
                .extend(output.top_logprobs.iter().cloned());
            merged
                .text_offset
                .extend(output.text_offset.iter().cloned());
            Some(merged)
        }
        (Some(prompt), true) => Some(prompt.clone()),
        (None, false) => Some(output.clone()),
        (None, true) => None,
    }
}

/// Build one choice's text, logprobs, prompt logprobs and ids,
/// applying `echo`. Prompt entries merge into `logprobs` only under
/// `echo` when output logprobs were requested (`want_logprobs`); with
/// echo but no `logprobs` request, `logprobs` stays null and the prompt
/// entries live only in the separate `prompt_logprobs` field. Without
/// echo they are returned in the separate `prompt_logprobs` field (vLLM
/// array shape: one entry per prompt position, `None` first, then
/// token-id candidate maps) and `logprobs` holds output entries only,
/// with offsets relative to the generated text.
#[allow(clippy::too_many_arguments)]
fn completion_choice_parts(
    sequence: &SequenceAccum,
    echo_text: Option<&str>,
    echo: bool,
    echo_only: bool,
    want_logprobs: bool,
    prompt_logprobs: Option<&LogProbs>,
    prompt_logprobs_array: Option<&PromptLogprobs>,
    emit_token_ids: bool,
) -> (
    String,
    Option<LogProbs>,
    Option<PromptLogprobs>,
    Option<Vec<u32>>,
) {
    // `max_tokens: 0` answers with the prompt only (or nothing without echo)
    // and drops the single token the engine had to generate.
    if echo_only {
        return (
            if echo {
                echo_text.unwrap_or_default().to_string()
            } else {
                String::new()
            },
            if want_logprobs {
                prompt_logprobs.cloned()
            } else {
                None
            },
            prompt_logprobs_array.cloned(),
            // No generated ids survive (the one token the engine had to
            // generate is dropped), so `token_ids` is empty; the prompt
            // ids are returned separately in `prompt_token_ids`.
            (echo && emit_token_ids).then(Vec::new),
        );
    }
    let (text, logprobs) = if echo {
        let mut text = echo_text.unwrap_or_default().to_string();
        text.push_str(&sequence.text);
        (
            text,
            merge_prompt_logprobs(
                if want_logprobs { prompt_logprobs } else { None },
                &sequence.logprobs,
            ),
        )
    } else {
        (
            sequence.text.clone(),
            merge_prompt_logprobs(None, &sequence.logprobs),
        )
    };
    let token_ids = emit_token_ids.then(|| sequence.token_ids.clone());
    (text, logprobs, prompt_logprobs_array.cloned(), token_ids)
}

fn notify_first_output(
    msg: &super::pb::GenerateResponse,
    text: &str,
    first_output_seen: &mut bool,
    callback: Option<&FirstOutputCallback>,
) {
    let aborted = msg
        .outputs
        .as_ref()
        .and_then(|output| output.finish_info.as_ref())
        .is_some_and(|finish| finish.finish_reason == FinishReason::Aborted as i32);
    if !*first_output_seen && !aborted && has_generated_output(msg, text) {
        *first_output_seen = true;
        if let Some(callback) = callback {
            callback();
        }
    }
}

fn sequence_text_and_finish(
    outputs: &pb::SequenceOutput,
    detok: &mut Option<Box<dyn IncrementalDecoderTrait + '_>>,
    prefer_worker_text: bool,
) -> Result<(String, Option<String>), String> {
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
                decoded.push_str(&last.text);
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

fn has_generated_output(msg: &super::pb::GenerateResponse, text: &str) -> bool {
    msg.outputs.as_ref().is_some_and(|outputs| {
        outputs.num_tokens > 0 || !outputs.token_ids.is_empty() || !text.is_empty()
    })
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::backend::pb::{GenerateResponse, PromptInfo, SequenceOutput};
    use std::sync::atomic::Ordering;

    #[test]
    fn prompt_info_does_not_count_as_generated_output() {
        let prompt_info = GenerateResponse {
            prompt_info: Some(PromptInfo {
                num_prompt_tokens: 32,
                ..Default::default()
            }),
            outputs: None,
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
        };
        assert!(has_generated_output(&output, ""));
        notify_first_output(&output, "", &mut first_output_seen, Some(&callback));
        notify_first_output(&output, "", &mut first_output_seen, Some(&callback));
        assert_eq!(decode_started.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn aborted_response_does_not_notify_first_output() {
        let aborted = GenerateResponse {
            outputs: Some(SequenceOutput {
                num_tokens: 1,
                token_ids: vec![42],
                finish_info: Some(super::super::pb::FinishInfo {
                    finish_reason: FinishReason::Aborted as i32,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            prompt_info: None,
        };
        let decode_started = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let decode_started_callback = decode_started.clone();
        let callback: FirstOutputCallback = Arc::new(move || {
            decode_started_callback.fetch_add(1, Ordering::Relaxed);
        });
        let mut first_output_seen = false;
        notify_first_output(&aborted, "", &mut first_output_seen, Some(&callback));
        assert!(!first_output_seen);
        assert_eq!(decode_started.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn output_frame_latches_to_the_first_generated_response() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let (mut producer, mut consumer) = output_frame_tracking();

        assert!(send_stream_chunk(
            &tx,
            &mut producer,
            "prompt info".into(),
            false,
        ));
        assert!(!consumer.on_frame());

        assert!(send_stream_chunk(
            &tx,
            &mut producer,
            "first token".into(),
            true,
        ));
        assert!(consumer.on_frame());

        assert!(send_stream_chunk(
            &tx,
            &mut producer,
            "second token".into(),
            true,
        ));
        assert!(consumer.on_frame());
    }

    #[test]
    fn abort_finish_records_an_error_and_fails_the_round() {
        let observer = EngineOutcomeObserver::default();
        assert!(!record_finish_outcome(Some(&observer), Some("abort")));
        assert!(observer.failed());

        let observer = EngineOutcomeObserver::default();
        assert!(record_finish_outcome(Some(&observer), Some("stop")));
        assert!(!observer.failed());

        // No observer still reports the aborted round as failed.
        assert!(!record_finish_outcome(None, Some("abort")));
    }

    #[test]
    fn logprob_value_clamps_minus_inf_and_nan() {
        assert_eq!(logprob_value(f32::NEG_INFINITY), -9999.0);
        assert_eq!(logprob_value(f32::NAN), -9999.0);
        assert_eq!(logprob_value(-0.5), -0.5);
        assert_eq!(logprob_value(-20_000.0), -9999.0);
    }

    #[test]
    fn append_logprobs_tracks_offsets_top_k_and_clamping() {
        use crate::backend::pb::candidate_token_info::TokenInfo;
        // `token_id:<id>` rendering needs no tokenizer, so the offsets are
        // the character lengths of the rendered tokens.
        let mut logprobs = empty_logprobs();
        let candidates = vec![pb::CandidateTokenInfo {
            tokens: vec![TokenInfo {
                id: 9,
                logprob: -1.5,
                rank: 2,
            }],
        }];
        let (next, _) = append_logprobs(
            &mut logprobs,
            None,
            &mut None,
            &[7, 8],
            &[Some(-0.5), Some(f32::NEG_INFINITY)],
            &candidates,
            true,
            false,
            None,
            0,
        )
        .unwrap();
        assert_eq!(logprobs.tokens, vec!["token_id:7", "token_id:8"]);
        assert_eq!(logprobs.token_logprobs, vec![Some(-0.5), Some(-9999.0)]);
        assert_eq!(logprobs.text_offset, vec![0, 10]);
        assert_eq!(next, 20);
        let top0 = logprobs.top_logprobs[0].as_ref().unwrap();
        // The sampled token joins the candidate map.
        assert_eq!(top0.get("token_id:9"), Some(&-1.5));
        assert_eq!(top0.get("token_id:7"), Some(&-0.5));
        let top1 = logprobs.top_logprobs[1].as_ref().unwrap();
        assert_eq!(top1.get("token_id:8"), Some(&-9999.0));
    }

    #[test]
    fn append_logprobs_restricts_sampled_token_to_selected_ids() {
        use crate::backend::pb::candidate_token_info::TokenInfo;
        // TokenIds selection: the sampled token joins the top map only
        // when it is one of the requested ids; its scalar logprob is
        // still reported.
        let candidates = vec![pb::CandidateTokenInfo {
            tokens: vec![TokenInfo {
                id: 9,
                logprob: -1.5,
                rank: 2,
            }],
        }];
        let mut logprobs = empty_logprobs();
        append_logprobs(
            &mut logprobs,
            None,
            &mut None,
            &[7],
            &[Some(-0.5)],
            &candidates,
            true,
            false,
            Some(&[9]),
            0,
        )
        .unwrap();
        assert_eq!(logprobs.token_logprobs, vec![Some(-0.5)]);
        let top0 = logprobs.top_logprobs[0].as_ref().unwrap();
        assert_eq!(top0.get("token_id:9"), Some(&-1.5));
        assert_eq!(top0.get("token_id:7"), None);

        let mut logprobs = empty_logprobs();
        append_logprobs(
            &mut logprobs,
            None,
            &mut None,
            &[7],
            &[Some(-0.5)],
            &candidates,
            true,
            false,
            Some(&[7, 9]),
            0,
        )
        .unwrap();
        let top0 = logprobs.top_logprobs[0].as_ref().unwrap();
        assert_eq!(top0.get("token_id:7"), Some(&-0.5));
        assert_eq!(top0.get("token_id:9"), Some(&-1.5));
    }

    #[test]
    fn prompt_logprobs_first_position_is_null() {
        // Engine returns one logprob per prompt token: the first has no
        // preceding token, so OpenAI reports it as null.
        let info = PromptInfo {
            num_prompt_tokens: 3,
            logprobs: vec![-0.1, -0.2, -0.3],
            ..Default::default()
        };
        let logprobs =
            prompt_logprobs_from_info(&info, &[1, 2, 3], None, true, false, None).unwrap();
        assert_eq!(logprobs.token_logprobs, vec![None, Some(-0.2), Some(-0.3)]);
        assert_eq!(logprobs.text_offset, vec![0, 10, 20]);

        // Engine returns one fewer (no entry for the first token): same
        // result, the null is prepended.
        let info = PromptInfo {
            num_prompt_tokens: 3,
            logprobs: vec![-0.2, -0.3],
            ..Default::default()
        };
        let logprobs =
            prompt_logprobs_from_info(&info, &[1, 2, 3], None, true, false, None).unwrap();
        assert_eq!(logprobs.token_logprobs, vec![None, Some(-0.2), Some(-0.3)]);

        // A length that matches neither shape is dropped, not misaligned.
        let info = PromptInfo {
            num_prompt_tokens: 3,
            logprobs: vec![-0.1],
            ..Default::default()
        };
        let logprobs =
            prompt_logprobs_from_info(&info, &[1, 2, 3], None, true, false, None).unwrap();
        assert!(logprobs.tokens.is_empty());
    }

    #[test]
    fn prompt_logprobs_n_minus_one_candidates_align_in_echo_merge() {
        use crate::backend::pb::candidate_token_info::TokenInfo;
        // N-1 candidate groups: the first token's group is omitted, so
        // the groups shift by one like the scalars — B's candidates
        // belong to B, not A (token-ID labels keep the entries distinct
        // without a tokenizer).
        let info = PromptInfo {
            num_prompt_tokens: 3,
            logprobs: vec![-0.2, -0.3],
            ranks: vec![2, 3],
            candidate_tokens: vec![
                pb::CandidateTokenInfo {
                    tokens: vec![TokenInfo {
                        id: 20,
                        logprob: -1.5,
                        rank: 2,
                    }],
                },
                pb::CandidateTokenInfo {
                    tokens: vec![TokenInfo {
                        id: 30,
                        logprob: -2.5,
                        rank: 3,
                    }],
                },
            ],
            ..Default::default()
        };
        let logprobs =
            prompt_logprobs_from_info(&info, &[1, 2, 3], None, true, false, None).unwrap();
        assert_eq!(logprobs.token_logprobs, vec![None, Some(-0.2), Some(-0.3)]);
        assert!(logprobs.top_logprobs[0].is_none());
        let pos1 = logprobs.top_logprobs[1].as_ref().unwrap();
        assert_eq!(pos1.len(), 2);
        assert_eq!(pos1["token_id:20"], -1.5);
        assert_eq!(pos1["token_id:2"], -0.2);
        let pos2 = logprobs.top_logprobs[2].as_ref().unwrap();
        assert_eq!(pos2.len(), 2);
        assert_eq!(pos2["token_id:30"], -2.5);
        assert_eq!(pos2["token_id:3"], -0.3);
    }

    #[test]
    fn prompt_logprobs_one_token_prompt_keeps_null_entry() {
        // A one-token prompt has no scorable first token: the N-1
        // encoding is an empty vector, and the prompt token is still
        // reported at offset 0 with a null scalar instead of being
        // dropped entirely.
        let info = PromptInfo {
            num_prompt_tokens: 1,
            logprobs: vec![],
            ..Default::default()
        };
        let logprobs = prompt_logprobs_from_info(&info, &[42], None, true, false, None).unwrap();
        assert_eq!(logprobs.tokens, vec!["token_id:42"]);
        assert_eq!(logprobs.token_logprobs, vec![None]);
        assert_eq!(logprobs.text_offset, vec![0]);
    }

    #[test]
    fn prompt_offset_base_uses_one_coordinate_space() {
        // Decoded space: the first generated offset is the echoed
        // prompt text length.
        assert_eq!(prompt_offset_base(&[1, 2], Some("hi"), true, false), 2);
        // Token-ID display mode: prompt entries count `token_id:<id>`
        // label lengths, so generated entries continue from the end of
        // the labels (10 + 10), not from the decoded length (2).
        assert_eq!(prompt_offset_base(&[1, 2], Some("hi"), true, true), 20);
        // Token-ID prompts have no echo text; the labels still set the
        // base.
        assert_eq!(prompt_offset_base(&[1, 2], None, true, true), 20);
        // Without merged prompt entries the base stays 0, even when the
        // prompt text is known.
        assert_eq!(prompt_offset_base(&[1, 2], Some("hi"), false, false), 0);
        assert_eq!(prompt_offset_base(&[1, 2], None, false, true), 0);
    }

    #[test]
    fn merged_logprobs_offsets_do_not_regress_at_the_echo_boundary() {
        // Token-ID display mode, echo: prompt entries at label offsets
        // [0, 10], then one generated token. Before the fix the
        // generated entry restarted at the decoded prompt length (2)
        // and the merged offsets went backward.
        let info = PromptInfo {
            num_prompt_tokens: 2,
            logprobs: vec![-0.2],
            ..Default::default()
        };
        let mut merged =
            prompt_logprobs_from_info(&info, &[1, 2], None, true, false, None).unwrap();
        assert_eq!(merged.text_offset, vec![0, 10]);
        let start = prompt_offset_base(&[1, 2], Some("hi"), true, true);
        let (next, _) = append_logprobs(
            &mut merged,
            None,
            &mut None,
            &[8],
            &[Some(-0.5)],
            &[],
            true,
            false,
            None,
            start,
        )
        .unwrap();
        assert_eq!(merged.text_offset, vec![0, 10, 20]);
        assert_eq!(next, 30);
    }

    fn fixture_tokenizer() -> DynTokenizer {
        use vllm_tokenizer::HuggingFaceTokenizer;
        Arc::new(
            HuggingFaceTokenizer::new_hf(std::path::Path::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/rust/fixtures/tokenizer/byte_level_bpe.json"
            )))
            .unwrap(),
        )
    }

    #[test]
    fn append_logprobs_counts_a_split_emoji_once() {
        // The fixture tokenizer encodes U+1F600 as four byte tokens, and
        // each byte token decodes alone to one replacement character — so
        // counting per-token decode output would put the token after the
        // emoji 3 chars too far. The offset decoder counts the emoji
        // once, at the token that completes it.
        let tok = fixture_tokenizer();
        let mut decoder = Some(decode_stream(&*tok, &[97], false));
        let mut logprobs = empty_logprobs();
        let (next, _) = append_logprobs(
            &mut logprobs,
            Some(&tok),
            &mut decoder,
            &[240, 159, 152, 128, 98],
            &[Some(-0.5); 5],
            &[],
            false,
            false,
            None,
            1,
        )
        .unwrap();
        assert_eq!(logprobs.text_offset, vec![1, 1, 1, 1, 2]);
        // The fragments carry no text of their own; the token that
        // completes the emoji carries it, so the tokens concatenate
        // to the completion text instead of four replacement chars.
        assert_eq!(logprobs.tokens, vec!["", "", "", "😀", "b"]);
        assert_eq!(next, 3);
    }

    #[test]
    fn append_logprobs_split_emoji_across_calls_counts_once() {
        // Same emoji, but the first two byte tokens arrive in one chunk
        // and the rest in the next: the decoder state survives the call
        // boundary, as it does across SSE frames in the streaming path.
        let tok = fixture_tokenizer();
        let mut decoder = Some(decode_stream(&*tok, &[97], false));
        let mut logprobs = empty_logprobs();
        let (next, _) = append_logprobs(
            &mut logprobs,
            Some(&tok),
            &mut decoder,
            &[240, 159],
            &[Some(-0.5), Some(-0.6)],
            &[],
            false,
            false,
            None,
            1,
        )
        .unwrap();
        assert_eq!(logprobs.text_offset, vec![1, 1]);
        assert_eq!(logprobs.tokens, vec!["", ""]);
        assert_eq!(next, 1);
        let (next, _) = append_logprobs(
            &mut logprobs,
            Some(&tok),
            &mut decoder,
            &[152, 128, 98],
            &[Some(-0.7), Some(-0.8), Some(-0.9)],
            &[],
            false,
            false,
            None,
            next,
        )
        .unwrap();
        assert_eq!(logprobs.text_offset, vec![1, 1, 1, 1, 2]);
        assert_eq!(logprobs.tokens, vec!["", "", "", "😀", "b"]);
        assert_eq!(next, 3);
    }

    #[test]
    fn append_logprobs_sampled_token_appears_once_in_top_logprobs() {
        use crate::backend::pb::candidate_token_info::TokenInfo;
        // When the engine includes the sampled id among its candidates,
        // the candidate must use the sampled token's contextual text:
        // the byte completing the emoji would otherwise appear twice
        // in top_logprobs, once as `�` (isolated) and once as `😀`.
        let tok = fixture_tokenizer();
        let mut decoder = Some(decode_stream(&*tok, &[97], false));
        let mut logprobs = empty_logprobs();
        let candidates = vec![
            pb::CandidateTokenInfo { tokens: vec![] },
            pb::CandidateTokenInfo { tokens: vec![] },
            pb::CandidateTokenInfo { tokens: vec![] },
            pb::CandidateTokenInfo {
                tokens: vec![
                    TokenInfo {
                        id: 128,
                        logprob: -0.5,
                        rank: 1,
                    },
                    TokenInfo {
                        id: 99,
                        logprob: -1.0,
                        rank: 2,
                    },
                ],
            },
        ];
        append_logprobs(
            &mut logprobs,
            Some(&tok),
            &mut decoder,
            &[240, 159, 152, 128, 98],
            &[Some(-0.5); 5],
            &candidates,
            false,
            false,
            None,
            1,
        )
        .unwrap();
        let top3 = logprobs.top_logprobs[3].as_ref().unwrap();
        assert_eq!(top3.len(), 2);
        assert_eq!(top3.get("😀"), Some(&-0.5));
        assert_eq!(top3.get("c"), Some(&-1.0));
        assert_eq!(top3.get("�"), None);
    }

    #[test]
    fn flush_logprobs_tail_gives_flushed_text_to_last_token() {
        // Generation ends mid-emoji: the two byte tokens report ""
        // each, and the flush gives the replacement character to the
        // last one — the same text the text decoder's flush puts into
        // choices[].text.
        let tok = fixture_tokenizer();
        let mut decoder = Some(decode_stream(&*tok, &[97], false));
        let mut logprobs = empty_logprobs();
        append_logprobs(
            &mut logprobs,
            Some(&tok),
            &mut decoder,
            &[240, 159],
            &[Some(-0.5), Some(-0.6)],
            &[],
            false,
            false,
            None,
            1,
        )
        .unwrap();
        assert_eq!(logprobs.tokens, vec!["", ""]);
        let width = flush_logprobs_tail(&mut logprobs, &mut decoder, true, None).unwrap();
        assert_eq!(width, 1);
        assert_eq!(logprobs.tokens, vec!["", "�"]);
    }

    #[test]
    fn flush_logprobs_tail_moves_sampled_top_entry_to_flushed_text() {
        use crate::backend::pb::candidate_token_info::TokenInfo;
        // The last fragment token's top entry is keyed by its pre-flush
        // text (""); after the flush its text is "�", so the sampled
        // score must move to that key or clients cannot find it.
        let tok = fixture_tokenizer();
        let mut decoder = Some(decode_stream(&*tok, &[97], false));
        let mut logprobs = empty_logprobs();
        let candidates = vec![
            pb::CandidateTokenInfo { tokens: vec![] },
            pb::CandidateTokenInfo {
                tokens: vec![
                    TokenInfo {
                        id: 159,
                        logprob: -0.6,
                        rank: 1,
                    },
                    TokenInfo {
                        id: 99,
                        logprob: -1.0,
                        rank: 2,
                    },
                ],
            },
        ];
        append_logprobs(
            &mut logprobs,
            Some(&tok),
            &mut decoder,
            &[240, 159],
            &[Some(-0.5), Some(-0.6)],
            &candidates,
            false,
            false,
            None,
            1,
        )
        .unwrap();
        let top1 = logprobs.top_logprobs[1].as_ref().unwrap();
        assert_eq!(top1.get(""), Some(&-0.6));
        flush_logprobs_tail(&mut logprobs, &mut decoder, true, None).unwrap();
        assert_eq!(logprobs.tokens, vec!["", "�"]);
        let top1 = logprobs.top_logprobs[1].as_ref().unwrap();
        assert_eq!(top1.len(), 2);
        assert_eq!(top1.get("�"), Some(&-0.6));
        assert_eq!(top1.get("c"), Some(&-1.0));
        assert_eq!(top1.get(""), None);
    }

    #[test]
    fn flush_logprobs_tail_sampled_score_wins_flushed_key_collision() {
        use crate::backend::pb::candidate_token_info::TokenInfo;
        // Another candidate already decodes to `�` with a different
        // score; after the flush the sampled token's text is also `�`,
        // and looking it up must give the sampled score, not the other
        // candidate's.
        let tok = fixture_tokenizer();
        let mut decoder = Some(decode_stream(&*tok, &[97], false));
        let mut logprobs = empty_logprobs();
        let candidates = vec![
            pb::CandidateTokenInfo { tokens: vec![] },
            pb::CandidateTokenInfo {
                tokens: vec![TokenInfo {
                    id: 240,
                    logprob: -1.0,
                    rank: 2,
                }],
            },
        ];
        append_logprobs(
            &mut logprobs,
            Some(&tok),
            &mut decoder,
            &[240, 159],
            &[Some(-0.5), Some(-0.6)],
            &candidates,
            false,
            false,
            None,
            1,
        )
        .unwrap();
        let top1 = logprobs.top_logprobs[1].as_ref().unwrap();
        assert_eq!(top1.get("�"), Some(&-1.0));
        assert_eq!(top1.get(""), Some(&-0.6));
        flush_logprobs_tail(&mut logprobs, &mut decoder, true, None).unwrap();
        assert_eq!(logprobs.tokens, vec!["", "�"]);
        let top1 = logprobs.top_logprobs[1].as_ref().unwrap();
        assert_eq!(top1.len(), 1);
        assert_eq!(top1.get("�"), Some(&-0.6));
        assert_eq!(top1.get(""), None);
    }

    #[test]
    fn flush_logprobs_tail_keeps_unselected_candidates_key() {
        use crate::backend::pb::candidate_token_info::TokenInfo;
        // A TokenIds selection excludes the sampled token, while a
        // selected skipped special token holds the "" key with a score
        // equal to the sampled score. The flush must not mistake that
        // candidate's entry for the sampled entry and relabel it.
        let tok = fixture_tokenizer();
        let mut decoder = Some(decode_stream(&*tok, &[97], true));
        let mut logprobs = empty_logprobs();
        let candidates = vec![
            pb::CandidateTokenInfo { tokens: vec![] },
            pb::CandidateTokenInfo {
                tokens: vec![TokenInfo {
                    id: 290,
                    logprob: -0.6,
                    rank: 1,
                }],
            },
        ];
        append_logprobs(
            &mut logprobs,
            Some(&tok),
            &mut decoder,
            &[240, 159],
            &[Some(-0.5), Some(-0.6)],
            &candidates,
            false,
            true,
            Some(&[290]),
            1,
        )
        .unwrap();
        let top1 = logprobs.top_logprobs[1].as_ref().unwrap();
        assert_eq!(top1.get(""), Some(&-0.6));
        assert_eq!(top1.get("�"), None);
        flush_logprobs_tail(&mut logprobs, &mut decoder, false, None).unwrap();
        assert_eq!(logprobs.tokens, vec!["", "�"]);
        let top1 = logprobs.top_logprobs[1].as_ref().unwrap();
        assert_eq!(top1.len(), 1);
        assert_eq!(top1.get(""), Some(&-0.6));
        assert_eq!(top1.get("�"), None);
    }

    #[test]
    fn flush_logprobs_tail_restores_displaced_candidate() {
        use crate::backend::pb::candidate_token_info::TokenInfo;
        // The sampled token ends generation mid-character, and a
        // selected skipped special token decodes to the same pre-flush
        // text with a different score. The sampled entry displaces the
        // candidate, but after the flush the two keys are distinct, so
        // both scores must be reported.
        let tok = fixture_tokenizer();
        let mut decoder = Some(decode_stream(&*tok, &[97], true));
        let mut logprobs = empty_logprobs();
        let candidates = vec![
            pb::CandidateTokenInfo { tokens: vec![] },
            pb::CandidateTokenInfo {
                tokens: vec![TokenInfo {
                    id: 290,
                    logprob: -1.0,
                    rank: 1,
                }],
            },
        ];
        let (_next, displaced) = append_logprobs(
            &mut logprobs,
            Some(&tok),
            &mut decoder,
            &[240, 159],
            &[Some(-0.5), Some(-0.6)],
            &candidates,
            false,
            true,
            None,
            1,
        )
        .unwrap();
        assert_eq!(displaced, Some(("".to_string(), -1.0)));
        let top1 = logprobs.top_logprobs[1].as_ref().unwrap();
        assert_eq!(top1.len(), 1);
        assert_eq!(top1.get(""), Some(&-0.6));
        flush_logprobs_tail(&mut logprobs, &mut decoder, true, displaced).unwrap();
        assert_eq!(logprobs.tokens, vec!["", "�"]);
        let top1 = logprobs.top_logprobs[1].as_ref().unwrap();
        assert_eq!(top1.len(), 2);
        assert_eq!(top1.get(""), Some(&-1.0));
        assert_eq!(top1.get("�"), Some(&-0.6));
    }

    #[test]
    fn flush_logprobs_tail_restores_displaced_candidate_with_equal_score() {
        use crate::backend::pb::candidate_token_info::TokenInfo;
        // Same setup as the displaced-candidate test, but the distinct
        // candidate's score ties the sampled score. Score equality
        // cannot tell the two tokens apart, so the candidate must be
        // tracked by id and restored regardless of score.
        let tok = fixture_tokenizer();
        let mut decoder = Some(decode_stream(&*tok, &[97], true));
        let mut logprobs = empty_logprobs();
        let candidates = vec![
            pb::CandidateTokenInfo { tokens: vec![] },
            pb::CandidateTokenInfo {
                tokens: vec![TokenInfo {
                    id: 290,
                    logprob: -0.6,
                    rank: 1,
                }],
            },
        ];
        let (_next, displaced) = append_logprobs(
            &mut logprobs,
            Some(&tok),
            &mut decoder,
            &[240, 159],
            &[Some(-0.5), Some(-0.6)],
            &candidates,
            false,
            true,
            None,
            1,
        )
        .unwrap();
        assert_eq!(displaced, Some(("".to_string(), -0.6)));
        flush_logprobs_tail(&mut logprobs, &mut decoder, true, displaced).unwrap();
        assert_eq!(logprobs.tokens, vec!["", "�"]);
        let top1 = logprobs.top_logprobs[1].as_ref().unwrap();
        assert_eq!(top1.len(), 2);
        assert_eq!(top1.get(""), Some(&-0.6));
        assert_eq!(top1.get("�"), Some(&-0.6));
    }

    #[test]
    fn flush_logprobs_tail_does_not_restore_same_token_candidate() {
        use crate::backend::pb::candidate_token_info::TokenInfo;
        // A candidate carrying the sampled token's own id is the same
        // token reported twice, even when the backend gives it a
        // different score in the candidate field. The sampled entry
        // overwrites it, and the flush must not restore the duplicate
        // under the old text.
        let tok = fixture_tokenizer();
        let mut decoder = Some(decode_stream(&*tok, &[97], true));
        let mut logprobs = empty_logprobs();
        let candidates = vec![
            pb::CandidateTokenInfo { tokens: vec![] },
            pb::CandidateTokenInfo {
                tokens: vec![TokenInfo {
                    id: 159,
                    logprob: -1.0,
                    rank: 1,
                }],
            },
        ];
        let (_next, displaced) = append_logprobs(
            &mut logprobs,
            Some(&tok),
            &mut decoder,
            &[240, 159],
            &[Some(-0.5), Some(-0.6)],
            &candidates,
            false,
            true,
            None,
            1,
        )
        .unwrap();
        assert_eq!(displaced, None);
        let top1 = logprobs.top_logprobs[1].as_ref().unwrap();
        assert_eq!(top1.len(), 1);
        assert_eq!(top1.get(""), Some(&-0.6));
        flush_logprobs_tail(&mut logprobs, &mut decoder, true, displaced).unwrap();
        assert_eq!(logprobs.tokens, vec!["", "�"]);
        let top1 = logprobs.top_logprobs[1].as_ref().unwrap();
        assert_eq!(top1.len(), 1);
        assert_eq!(top1.get("�"), Some(&-0.6));
        assert_eq!(top1.get(""), None);
    }

    #[test]
    fn flush_logprobs_tail_restores_candidate_before_sampled_id_copy() {
        use crate::backend::pb::candidate_token_info::TokenInfo;
        // A distinct candidate decodes to the sampled token's pre-flush
        // text, followed by a copy of the sampled id with its own
        // score. The copy overwrites the distinct candidate's score in
        // the map, but the flush must still restore the distinct
        // candidate's score — captured when the candidate was seen,
        // not read back from the map.
        let tok = fixture_tokenizer();
        let mut decoder = Some(decode_stream(&*tok, &[97], true));
        let mut logprobs = empty_logprobs();
        let candidates = vec![
            pb::CandidateTokenInfo { tokens: vec![] },
            pb::CandidateTokenInfo {
                tokens: vec![
                    TokenInfo {
                        id: 290,
                        logprob: -1.0,
                        rank: 1,
                    },
                    TokenInfo {
                        id: 159,
                        logprob: -2.0,
                        rank: 2,
                    },
                ],
            },
        ];
        let (_next, displaced) = append_logprobs(
            &mut logprobs,
            Some(&tok),
            &mut decoder,
            &[240, 159],
            &[Some(-0.5), Some(-0.6)],
            &candidates,
            false,
            true,
            None,
            1,
        )
        .unwrap();
        assert_eq!(displaced, Some(("".to_string(), -1.0)));
        let top1 = logprobs.top_logprobs[1].as_ref().unwrap();
        assert_eq!(top1.len(), 1);
        assert_eq!(top1.get(""), Some(&-0.6));
        flush_logprobs_tail(&mut logprobs, &mut decoder, true, displaced).unwrap();
        assert_eq!(logprobs.tokens, vec!["", "�"]);
        let top1 = logprobs.top_logprobs[1].as_ref().unwrap();
        assert_eq!(top1.len(), 2);
        assert_eq!(top1.get(""), Some(&-1.0));
        assert_eq!(top1.get("�"), Some(&-0.6));
    }

    #[test]
    fn flush_logprobs_tail_is_noop_when_nothing_buffered() {
        let tok = fixture_tokenizer();
        let mut decoder = Some(decode_stream(&*tok, &[97], false));
        let mut logprobs = empty_logprobs();
        append_logprobs(
            &mut logprobs,
            Some(&tok),
            &mut decoder,
            &[98],
            &[Some(-0.5)],
            &[],
            false,
            false,
            None,
            1,
        )
        .unwrap();
        assert_eq!(logprobs.tokens, vec!["b"]);
        let width = flush_logprobs_tail(&mut logprobs, &mut decoder, true, None).unwrap();
        assert_eq!(width, 0);
        assert_eq!(logprobs.tokens, vec!["b"]);
    }

    #[test]
    fn prompt_logprobs_split_emoji_counts_once() {
        // Prompt logprobs decode the prompt ids from scratch with their
        // own decoder, so an emoji split across prompt tokens counts
        // once there too.
        let tok = fixture_tokenizer();
        let info = PromptInfo {
            num_prompt_tokens: 6,
            logprobs: vec![-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
            ..Default::default()
        };
        let logprobs = prompt_logprobs_from_info(
            &info,
            &[97, 240, 159, 152, 128, 98],
            Some(&tok),
            false,
            false,
            None,
        )
        .unwrap();
        assert_eq!(logprobs.text_offset, vec![0, 1, 1, 1, 1, 2]);
        assert_eq!(logprobs.tokens, vec!["a", "", "", "", "😀", "b"]);
    }

    #[test]
    fn prompt_logprobs_array_matches_vllm_shape() {
        use crate::backend::pb::candidate_token_info::TokenInfo;
        // Candidates and the sampled token are keyed by token id; the
        // first position is null. Ranks ride along; a zero rank (proto3
        // default) means "no rank".
        let info = PromptInfo {
            num_prompt_tokens: 3,
            logprobs: vec![-0.1, -0.2, -0.3],
            ranks: vec![1, 2, 3],
            candidate_tokens: vec![
                pb::CandidateTokenInfo { tokens: vec![] },
                pb::CandidateTokenInfo {
                    tokens: vec![
                        TokenInfo {
                            id: 9,
                            logprob: -1.5,
                            rank: 2,
                        },
                        TokenInfo {
                            id: 7,
                            logprob: -2.5,
                            rank: 0,
                        },
                    ],
                },
                pb::CandidateTokenInfo { tokens: vec![] },
            ],
            ..Default::default()
        };
        let array = prompt_logprobs_array_from_info(&info, &[1, 2, 3], None, false).unwrap();
        assert_eq!(array.len(), 3);
        assert!(array[0].is_none());
        let pos1 = array[1].as_ref().unwrap();
        assert_eq!(pos1.len(), 3);
        assert_eq!(pos1[&9].logprob, -1.5);
        assert_eq!(pos1[&9].rank, Some(2));
        assert_eq!(pos1[&9].decoded_token, None);
        assert_eq!(pos1[&7].logprob, -2.5);
        assert_eq!(pos1[&7].rank, None);
        assert_eq!(pos1[&2].logprob, -0.2);
        assert_eq!(pos1[&2].rank, Some(2));
        assert_eq!(pos1[&2].decoded_token, None);
        let pos2 = array[2].as_ref().unwrap();
        assert_eq!(pos2.len(), 1);
        assert_eq!(pos2[&3].logprob, -0.3);
        assert_eq!(pos2[&3].rank, Some(3));
    }

    #[test]
    fn prompt_logprobs_array_handles_short_and_mismatched_shapes() {
        use crate::backend::pb::candidate_token_info::TokenInfo;
        // N-1 encoding: same array, null prepended — including the
        // candidate groups, which shift by one like the scalars: B's
        // candidates belong to B, C's to C.
        let info = PromptInfo {
            num_prompt_tokens: 3,
            logprobs: vec![-0.2, -0.3],
            ranks: vec![2, 3],
            candidate_tokens: vec![
                pb::CandidateTokenInfo {
                    tokens: vec![TokenInfo {
                        id: 20,
                        logprob: -1.5,
                        rank: 2,
                    }],
                },
                pb::CandidateTokenInfo {
                    tokens: vec![TokenInfo {
                        id: 30,
                        logprob: -2.5,
                        rank: 3,
                    }],
                },
            ],
            ..Default::default()
        };
        let array = prompt_logprobs_array_from_info(&info, &[1, 2, 3], None, false).unwrap();
        assert_eq!(array.len(), 3);
        assert!(array[0].is_none());
        let pos1 = array[1].as_ref().unwrap();
        assert_eq!(pos1.len(), 2);
        assert_eq!(pos1[&20].logprob, -1.5);
        assert_eq!(pos1[&20].rank, Some(2));
        assert_eq!(pos1[&2].logprob, -0.2);
        assert_eq!(pos1[&2].rank, Some(2));
        let pos2 = array[2].as_ref().unwrap();
        assert_eq!(pos2.len(), 2);
        assert_eq!(pos2[&30].logprob, -2.5);
        assert_eq!(pos2[&30].rank, Some(3));
        assert_eq!(pos2[&3].logprob, -0.3);
        assert_eq!(pos2[&3].rank, Some(3));
        // A length that matches neither shape is dropped, not misaligned.
        let info = PromptInfo {
            num_prompt_tokens: 3,
            logprobs: vec![-0.1],
            ..Default::default()
        };
        let array = prompt_logprobs_array_from_info(&info, &[1, 2, 3], None, false).unwrap();
        assert!(array.is_empty());
        // One-token prompt: a single null entry.
        let info = PromptInfo {
            num_prompt_tokens: 1,
            logprobs: vec![],
            ..Default::default()
        };
        let array = prompt_logprobs_array_from_info(&info, &[42], None, false).unwrap();
        assert_eq!(array, vec![None]);
    }

    #[test]
    fn prompt_logprobs_array_decodes_candidate_text() {
        use crate::backend::pb::candidate_token_info::TokenInfo;
        // The fixture tokenizer encodes "a" as 97 and "b" as 98; the
        // decoded text rides on the candidate entries, keyed by id.
        let tok = fixture_tokenizer();
        let info = PromptInfo {
            num_prompt_tokens: 2,
            logprobs: vec![-0.1, -0.2],
            ranks: vec![1, 1],
            candidate_tokens: vec![
                pb::CandidateTokenInfo { tokens: vec![] },
                pb::CandidateTokenInfo {
                    tokens: vec![TokenInfo {
                        id: 97,
                        logprob: -1.5,
                        rank: 3,
                    }],
                },
            ],
            ..Default::default()
        };
        let array = prompt_logprobs_array_from_info(&info, &[97, 98], Some(&tok), false).unwrap();
        assert_eq!(array.len(), 2);
        assert!(array[0].is_none());
        let pos1 = array[1].as_ref().unwrap();
        assert_eq!(pos1.len(), 2);
        assert_eq!(pos1[&97].decoded_token.as_deref(), Some("a"));
        assert_eq!(pos1[&97].rank, Some(3));
        assert_eq!(pos1[&98].logprob, -0.2);
        assert_eq!(pos1[&98].decoded_token.as_deref(), Some("b"));
    }

    #[test]
    fn completion_choice_parts_merges_echo_and_drops_the_echo_only_token() {
        let prompt_logprobs = LogProbs {
            tokens: vec!["token_id:1".to_string()],
            token_logprobs: vec![None],
            top_logprobs: vec![None],
            text_offset: vec![0],
        };
        let prompt_logprobs_array: PromptLogprobs = vec![
            None,
            Some(std::collections::HashMap::from([(
                2,
                PromptLogprob {
                    logprob: -0.2,
                    rank: Some(1),
                    decoded_token: None,
                },
            )])),
        ];
        let mut sequence = SequenceAccum::new(2);
        sequence.text = " world".to_string();
        sequence.token_ids = vec![8];
        sequence.logprobs.tokens = vec!["token_id:8".to_string()];
        sequence.logprobs.token_logprobs = vec![Some(-0.5)];
        sequence.logprobs.top_logprobs = vec![None];
        sequence.logprobs.text_offset = vec![2];

        // Echo: prompt text and prompt logprobs come first.
        let (text, logprobs, field_prompt_logprobs, token_ids) = completion_choice_parts(
            &sequence,
            Some("hi"),
            true,
            false,
            true,
            Some(&prompt_logprobs),
            Some(&prompt_logprobs_array),
            true,
        );
        assert_eq!(text, "hi world");
        let logprobs = logprobs.unwrap();
        assert_eq!(logprobs.tokens, vec!["token_id:1", "token_id:8"]);
        assert_eq!(logprobs.token_logprobs, vec![None, Some(-0.5)]);
        assert_eq!(logprobs.text_offset, vec![0, 2]);
        assert_eq!(field_prompt_logprobs.unwrap(), prompt_logprobs_array);
        assert_eq!(token_ids, Some(vec![8]));

        // Echo-only (`max_tokens: 0`): the prompt is the whole answer.
        let (text, logprobs, field_prompt_logprobs, token_ids) = completion_choice_parts(
            &sequence,
            Some("hi"),
            true,
            true,
            true,
            Some(&prompt_logprobs),
            Some(&prompt_logprobs_array),
            true,
        );
        assert_eq!(text, "hi");
        let logprobs = logprobs.unwrap();
        assert_eq!(logprobs.tokens, prompt_logprobs.tokens);
        assert_eq!(logprobs.token_logprobs, prompt_logprobs.token_logprobs);
        assert_eq!(logprobs.text_offset, prompt_logprobs.text_offset);
        assert_eq!(field_prompt_logprobs.unwrap(), prompt_logprobs_array);
        assert_eq!(token_ids, Some(vec![]));

        // No echo: output only in `logprobs`, prompt entries in the
        // separate field. (The output offset base is the caller's job —
        // see prompt_offset_base — so this sequence keeps its base 2.)
        let (text, logprobs, field_prompt_logprobs, token_ids) = completion_choice_parts(
            &sequence,
            None,
            false,
            false,
            true,
            Some(&prompt_logprobs),
            Some(&prompt_logprobs_array),
            false,
        );
        assert_eq!(text, " world");
        let logprobs = logprobs.unwrap();
        assert_eq!(logprobs.tokens, vec!["token_id:8"]);
        assert_eq!(logprobs.text_offset, vec![2]);
        assert_eq!(field_prompt_logprobs.unwrap(), prompt_logprobs_array);
        assert_eq!(token_ids, None);

        // No echo, no prompt logprobs requested: both fields empty.
        let (text, logprobs, field_prompt_logprobs, token_ids) =
            completion_choice_parts(&sequence, None, false, false, true, None, None, false);
        assert_eq!(text, " world");
        assert_eq!(logprobs.unwrap().tokens, vec!["token_id:8"]);
        assert!(field_prompt_logprobs.is_none());
        assert_eq!(token_ids, None);

        // Echo with standalone `prompt_logprobs` but no `logprobs`
        // request: `logprobs` stays null — the prompt entries live only
        // in the separate field (vLLM builds `logprobs` only when
        // `request.logprobs` is not None). The sequence carries no
        // output logprobs here, as in the real path where they are only
        // collected when `logprobs` was requested.
        let mut no_output_logprobs = SequenceAccum::new(2);
        no_output_logprobs.text = " world".to_string();
        no_output_logprobs.token_ids = vec![8];
        let (text, logprobs, field_prompt_logprobs, _token_ids) = completion_choice_parts(
            &no_output_logprobs,
            Some("hi"),
            true,
            false,
            false,
            Some(&prompt_logprobs),
            Some(&prompt_logprobs_array),
            true,
        );
        assert_eq!(text, "hi world");
        assert!(logprobs.is_none());
        assert_eq!(field_prompt_logprobs.unwrap(), prompt_logprobs_array);

        // Same, echo-only (`max_tokens: 0`).
        let (text, logprobs, field_prompt_logprobs, _token_ids) = completion_choice_parts(
            &no_output_logprobs,
            Some("hi"),
            true,
            true,
            false,
            Some(&prompt_logprobs),
            Some(&prompt_logprobs_array),
            true,
        );
        assert_eq!(text, "hi");
        assert!(logprobs.is_none());
        assert_eq!(field_prompt_logprobs.unwrap(), prompt_logprobs_array);
    }
}
