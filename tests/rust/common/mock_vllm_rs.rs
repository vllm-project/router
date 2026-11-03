//! In-process vLLM `Inference` gRPC worker for router tests.
//!
//! Also serves `grpc.health.v1` as `SERVING` so `Router::new` startup
//! probes pass, and records `Control.Abort` calls. Used by
//! `tests/rust/grpc_vs_http_e2e.rs`. Records `token_ids`
//! vs text prompt on each `GenerateStream`. Not Python EngineCore.

#![allow(dead_code)]

use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::Stream;
use tonic::{transport::Server, Request, Response, Status};
use vllm_router_rs::backend::convert::{json_to_proto_struct, proto_struct_to_json};
use vllm_router_rs::backend::pb::{
    self,
    control_server::{Control, ControlServer},
    finish_info, generate_request,
    inference_server::{Inference, InferenceServer},
    FinishInfo, GenerateRequest, GenerateResponse, PromptInfo, SequenceOutput, TokenIds,
};

#[derive(Clone, Default)]
pub struct CapturedGrpcGenerate {
    pub token_ids: Vec<u32>,
    pub had_text_prompt: bool,
    pub model: String,
    pub cache_salt: String,
    pub request_id: String,
    pub max_new_tokens: u32,
    pub min_new_tokens: u32,
    pub kv_transfer_params: Option<serde_json::Value>,
    /// Unary `Generate` rather than `GenerateStream`.
    pub unary: bool,
    /// `x-data-parallel-rank` request metadata.
    pub dp_rank: Option<String>,
    pub received_at: Option<Instant>,
}

#[derive(Clone)]
pub struct MockVllmRs {
    pub captured: Arc<Mutex<Vec<CapturedGrpcGenerate>>>,
    pub reply_text: String,
    /// `(text, token_ids)` per streamed frame; the last one carries
    /// `finish_info`. `None` sends a single frame: `reply_text` + `[7, 8, 9]`.
    /// Empty text with non-empty ids reproduces a buffered decoder fragment.
    pub frames: Option<Vec<(String, Vec<u32>)>>,
    pub capture_limit: usize,
    pub failures_remaining: Arc<AtomicUsize>,
    /// Returned in `FinishInfo.kv_transfer_params`, as a P/D prefill engine does.
    pub finish_kv_transfer_params: Option<serde_json::Value>,
    /// Wait before sending any output, as a long prefill would.
    pub reply_delay: Duration,
    /// Streams the client cancelled before their first output.
    pub cancelled: Arc<AtomicUsize>,
    /// Streams to fail with a gRPC error after their prompt info, i.e.
    /// after the stream has opened.
    pub stream_failures_remaining: Arc<AtomicUsize>,
    /// Requests to finish with `FinishReason::Aborted` instead of `Stop`.
    pub aborted_finishes_remaining: Arc<AtomicUsize>,
    /// Request ids received by `Control.Abort`.
    pub aborts: Arc<Mutex<Vec<String>>>,
}

impl Default for MockVllmRs {
    fn default() -> Self {
        Self {
            captured: Arc::new(Mutex::new(Vec::new())),
            reply_text: "hello from worker".to_string(),
            frames: None,
            capture_limit: usize::MAX,
            failures_remaining: Arc::new(AtomicUsize::new(0)),
            finish_kv_transfer_params: None,
            reply_delay: Duration::ZERO,
            cancelled: Arc::new(AtomicUsize::new(0)),
            stream_failures_remaining: Arc::new(AtomicUsize::new(0)),
            aborted_finishes_remaining: Arc::new(AtomicUsize::new(0)),
            aborts: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

pub struct MockVllmRsServer {
    pub addr: String,
    pub grpc_url: String,
    pub state: MockVllmRs,
    _handle: tokio::task::JoinHandle<()>,
}

impl MockVllmRsServer {
    pub async fn spawn() -> Self {
        Self::spawn_with_state(MockVllmRs::default()).await
    }

    pub async fn spawn_with_state(state: MockVllmRs) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
        let (reporter, health_service) = tonic_health::server::health_reporter();
        reporter.set_serving::<InferenceServer<MockVllmRs>>().await;
        reporter
            .set_service_status("", tonic_health::ServingStatus::Serving)
            .await;
        let svc = InferenceServer::new(state.clone());
        let control = ControlServer::new(state.clone());
        let handle = tokio::spawn(async move {
            Server::builder()
                .add_service(health_service)
                .add_service(svc)
                .add_service(control)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        // Let the acceptor start.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        Self {
            addr: addr.to_string(),
            grpc_url: format!("grpc://{addr}"),
            state,
            _handle: handle,
        }
    }

    pub fn captured(&self) -> Vec<CapturedGrpcGenerate> {
        self.state.captured.lock().unwrap().clone()
    }

    pub fn aborts(&self) -> Vec<String> {
        self.state.aborts.lock().unwrap().clone()
    }
}

type ResponseStream = Pin<Box<dyn Stream<Item = Result<GenerateResponse, Status>> + Send>>;

#[tonic::async_trait]
impl Inference for MockVllmRs {
    type GenerateStreamStream = ResponseStream;

    async fn generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<GenerateResponse>, Status> {
        let mut stream = self.generate_stream(request).await?.into_inner();
        if let Some(last) = self.captured.lock().unwrap().last_mut() {
            last.unary = true;
        }
        let mut text = String::new();
        let mut prompt_info = None;
        let mut finish = None;
        use futures_util::StreamExt;
        while let Some(item) = stream.next().await {
            let msg = item?;
            if msg.prompt_info.is_some() {
                prompt_info = msg.prompt_info;
            }
            if let Some(out) = msg.outputs {
                text.push_str(&out.text);
                if out.finish_info.is_some() {
                    finish = out.finish_info;
                }
            }
        }
        Ok(Response::new(GenerateResponse {
            prompt_info,
            outputs: Some(SequenceOutput {
                index: 0,
                text,
                num_tokens: 3,
                finish_info: finish,
                ..Default::default()
            }),
        }))
    }

    async fn generate_stream(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        let dp_rank = request
            .metadata()
            .get("x-data-parallel-rank")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let req = request.into_inner();
        let (token_ids, had_text_prompt) = match req.prompt {
            Some(generate_request::Prompt::TokenIds(TokenIds { ids })) => (ids, false),
            Some(generate_request::Prompt::Text(_)) => (Vec::new(), true),
            None => (Vec::new(), false),
        };
        {
            let mut captured = self.captured.lock().unwrap();
            if captured.len() < self.capture_limit {
                let stopping = req.stopping.unwrap_or_default();
                captured.push(CapturedGrpcGenerate {
                    token_ids: token_ids.clone(),
                    had_text_prompt,
                    model: req.model,
                    cache_salt: req
                        .kv
                        .as_ref()
                        .map(|kv| kv.cache_salt.clone())
                        .unwrap_or_default(),
                    request_id: req.request_id,
                    max_new_tokens: stopping.max_new_tokens,
                    min_new_tokens: stopping.min_new_tokens,
                    kv_transfer_params: req
                        .kv
                        .and_then(|kv| kv.kv_transfer_params)
                        .map(|params| proto_struct_to_json(&params)),
                    unary: false,
                    dp_rank,
                    received_at: Some(Instant::now()),
                });
            }
        }

        if self
            .failures_remaining
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .is_ok()
        {
            return Err(Status::unavailable("retry test"));
        }

        let (tx, rx) = mpsc::channel(4);
        let frames = self
            .frames
            .clone()
            .unwrap_or_else(|| vec![(self.reply_text.clone(), vec![7, 8, 9])]);
        let n_prompt = token_ids.len() as u32;
        let finish_kv = self
            .finish_kv_transfer_params
            .as_ref()
            .map(|params| json_to_proto_struct(params).expect("JSON object"));
        let reply_delay = self.reply_delay;
        let cancelled = self.cancelled.clone();
        let fail_stream = self
            .stream_failures_remaining
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .is_ok();
        let finish_reason = if self
            .aborted_finishes_remaining
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .is_ok()
        {
            finish_info::FinishReason::Aborted
        } else {
            finish_info::FinishReason::Stop
        };
        tokio::spawn(async move {
            tokio::select! {
                _ = tokio::time::sleep(reply_delay) => {}
                _ = tx.closed() => {
                    cancelled.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            }
            let _ = tx
                .send(Ok(GenerateResponse {
                    prompt_info: Some(PromptInfo {
                        num_prompt_tokens: n_prompt,
                        ..Default::default()
                    }),
                    outputs: None,
                }))
                .await;
            if fail_stream {
                let _ = tx.send(Err(Status::internal("decode failed"))).await;
                return;
            }
            let last = frames.len().saturating_sub(1);
            let total: u32 = frames.iter().map(|(_, ids)| ids.len() as u32).sum();
            let mut finish_kv = finish_kv;
            for (index, (text, token_ids)) in frames.into_iter().enumerate() {
                let finish_info = (index == last).then(|| FinishInfo {
                    num_output_tokens: total,
                    finish_reason: finish_reason as i32,
                    kv_transfer_params: finish_kv.take(),
                    ..Default::default()
                });
                let sent = tx
                    .send(Ok(GenerateResponse {
                        prompt_info: None,
                        outputs: Some(SequenceOutput {
                            index: 0,
                            text,
                            num_tokens: token_ids.len() as u32,
                            token_ids,
                            finish_info,
                            ..Default::default()
                        }),
                    }))
                    .await;
                if sent.is_err() {
                    break;
                }
            }
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

macro_rules! mock_control {
    ($($method:ident($request:ident) -> $response:ident;)*) => {
        /// Records `Abort` and serves model and server info; every other
        /// Control RPC is unimplemented.
        #[tonic::async_trait]
        impl Control for MockVllmRs {
            async fn abort(
                &self,
                request: Request<pb::AbortRequest>,
            ) -> Result<Response<pb::AbortResponse>, Status> {
                self.aborts
                    .lock()
                    .unwrap()
                    .extend(request.into_inner().request_ids);
                Ok(Response::new(pb::AbortResponse {}))
            }

            async fn get_model_info(
                &self,
                _request: Request<pb::GetModelInfoRequest>,
            ) -> Result<Response<pb::ModelInfo>, Status> {
                Ok(Response::new(pb::ModelInfo {
                    model_id: "mock-model".into(),
                    supports_token_ids_input: true,
                    ..Default::default()
                }))
            }

            async fn get_server_info(
                &self,
                _request: Request<pb::GetServerInfoRequest>,
            ) -> Result<Response<pb::ServerInfo>, Status> {
                Ok(Response::new(pb::ServerInfo {
                    engine_version: "mock".into(),
                    ..Default::default()
                }))
            }

            $(
                async fn $method(
                    &self,
                    _request: Request<pb::$request>,
                ) -> Result<Response<pb::$response>, Status> {
                    Err(Status::unimplemented(stringify!($method)))
                }
            )*
        }
    };
}

mock_control! {
    load_lora(LoadLoraRequest) -> LoadLoraResponse;
    unload_lora(UnloadLoraRequest) -> UnloadLoraResponse;
    list_loras(ListLorasRequest) -> ListLorasResponse;
    get_kv_event_sources(GetKvEventSourcesRequest) -> GetKvEventSourcesResponse;
    pause_generation(PauseGenerationRequest) -> PauseGenerationResponse;
    resume_generation(ResumeGenerationRequest) -> ResumeGenerationResponse;
    is_paused(IsPausedRequest) -> IsPausedResponse;
    sleep(SleepRequest) -> SleepResponse;
    wake_up(WakeUpRequest) -> WakeUpResponse;
    is_sleeping(IsSleepingRequest) -> IsSleepingResponse;
    init_weight_transfer_engine(InitWeightTransferEngineRequest) -> InitWeightTransferEngineResponse;
    start_weight_update(StartWeightUpdateRequest) -> StartWeightUpdateResponse;
    start_draft_weight_update(StartDraftWeightUpdateRequest) -> StartDraftWeightUpdateResponse;
    update_weights(UpdateWeightsRequest) -> UpdateWeightsResponse;
    finish_weight_update(FinishWeightUpdateRequest) -> FinishWeightUpdateResponse;
    update_weight_version(UpdateWeightVersionRequest) -> UpdateWeightVersionResponse;
    get_weight_version(GetWeightVersionRequest) -> GetWeightVersionResponse;
}
