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
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::transport::Channel;
use tracing::{error, info, warn};
use uuid::Uuid;

use super::convert::{chat_to_generate_request, requires_worker_output_text};
use super::detect::{grpc_connect_uri, parse_dp_rank};
use super::frontend::{PreparedRequest, ResponseShape};
use super::openai::{
    completion_final_response, final_response, finish_reason_name, now_secs, sse_chunk,
    sse_usage_chunk, SSE_DONE,
};
use super::pb::inference_client::InferenceClient;
use super::stream_output::{output_frame_tracking, OutputFrameConsumer, OutputFrameProducer};
use super::vllm_frontend::{decode_stream, emit_detok, DynTokenizer, IncrementalDecoderTrait};
use crate::protocols::spec::Usage;

const DP_RANK_METADATA: &str = "x-data-parallel-rank";

pub(crate) type FirstOutputCallback = Arc<dyn Fn() + Send + Sync + 'static>;

#[derive(Clone)]
pub struct GrpcStreamTask {
    handle: Arc<std::sync::Mutex<Option<tokio::task::JoinHandle<()>>>>,
    output_frame: Option<OutputFrameConsumer>,
}

impl GrpcStreamTask {
    #[cfg(test)]
    pub(crate) fn new(handle: tokio::task::JoinHandle<()>) -> Self {
        Self {
            handle: Arc::new(std::sync::Mutex::new(Some(handle))),
            output_frame: None,
        }
    }

    pub(crate) fn with_output_frame(
        handle: tokio::task::JoinHandle<()>,
        output_frame: OutputFrameConsumer,
    ) -> Self {
        Self {
            handle: Arc::new(std::sync::Mutex::new(Some(handle))),
            output_frame: Some(output_frame),
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
    clients: Arc<DashMap<String, InferenceClient<Channel>>>,
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

    async fn client_for(&self, worker_url: &str) -> Result<InferenceClient<Channel>, String> {
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
        let client = InferenceClient::new(channel);
        self.clients.insert(uri, client.clone());
        Ok(client)
    }

    pub fn remove_client(&self, worker_url: &str) {
        if let Ok(uri) = grpc_connect_uri(worker_url) {
            self.clients.remove(&uri);
        }
    }

    pub(crate) async fn dispatch_prepared(
        &self,
        worker_url: &str,
        prepared: PreparedRequest,
        on_first_output: Option<FirstOutputCallback>,
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
        let proto = match chat_to_generate_request(
            request,
            tokenized.token_ids.as_ref().to_vec(),
            request_id.clone(),
        ) {
            Ok(proto) => proto,
            Err(error) => return (StatusCode::BAD_REQUEST, error).into_response(),
        };

        let t_connect = Instant::now();
        let mut client = match self.client_for(worker_url).await {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "grpc backend connect failed");
                return (StatusCode::BAD_GATEWAY, e).into_response();
            }
        };
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
                Ok(Ok(s)) => s.into_inner(),
                Ok(Err(status)) => {
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
            request_timeout: self.request_timeout,
            stages,
            t_req,
            on_first_output,
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
    request_timeout: Duration,
    stages: serde_json::Value,
    t_req: Instant,
    on_first_output: Option<FirstOutputCallback>,
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
        request_timeout,
        mut stages,
        t_req,
        on_first_output: _,
    } = reply;
    let header_stages = stages.clone();
    let stages_on = crate::backend::stages_enabled();
    let (tx, rx) = mpsc::unbounded_channel::<Result<String, std::io::Error>>();
    let (mut producer_output_frame, output_frame) = output_frame_tracking();
    let producer = tokio::spawn(async move {
        let mut detok = tok
            .as_ref()
            .map(|t| decode_stream(&**t, prompt_ids.as_ref(), skip_special));
        let mut first = true;
        let mut first_token_logged = false;
        let mut prompt_tokens = n_prompt_tokens as u32;
        let mut completion_tokens = 0u32;
        loop {
            let item = match tokio::time::timeout(request_timeout, stream.next()).await {
                Ok(Some(item)) => item,
                Ok(None) => break,
                Err(_) => {
                    let _ = send_stream_chunk(
                        &tx,
                        &mut producer_output_frame,
                        "data: {\"error\":\"gRPC stream exceeded request timeout\"}\n\n"
                            .to_string(),
                        false,
                    );
                    return;
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
                                );
                                return;
                            }
                        };
                    let generated_output = has_generated_output(&msg, &text);
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
                            );
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
                        || (first && shape == ResponseShape::Chat)
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
                        ) {
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
                            let _ = send_stream_chunk(&tx, &mut producer_output_frame, line, false);
                        }
                        let _ = send_stream_chunk(
                            &tx,
                            &mut producer_output_frame,
                            SSE_DONE.to_string(),
                            false,
                        );
                        return;
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
                    );
                    return;
                }
            }
        }
        let _ = send_stream_chunk(
            &tx,
            &mut producer_output_frame,
            "data: {\"error\":\"gRPC stream closed before finish_info\"}\n\n".to_string(),
            false,
        );
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
        .insert(GrpcStreamTask::with_output_frame(producer, output_frame));
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
        request_timeout,
        mut stages,
        t_req,
        on_first_output,
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
        let item = match tokio::time::timeout(request_timeout, stream.next()).await {
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

fn has_generated_output(msg: &super::pb::GenerateResponse, text: &str) -> bool {
    msg.outputs.as_ref().is_some_and(|outputs| {
        outputs.num_tokens > 0 || !outputs.token_ids.is_empty() || !text.is_empty()
    })
}

#[cfg(test)]
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
}
