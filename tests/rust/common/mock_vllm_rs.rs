//! In-process vLLM 0.29 `Inference` gRPC worker for router tests.
//!
//! Also serves `grpc.health.v1` as `SERVING` so `Router::new` startup
//! probes pass. Used by `tests/rust/grpc_vs_http_e2e.rs`. Records `token_ids`
//! vs text prompt on each `GenerateStream`. Not Python EngineCore.

#![allow(dead_code)]

use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::Stream;
use tonic::{transport::Server, Request, Response, Status};
use vllm_router_rs::backend::pb::{
    self,
    control_server::{Control, ControlServer},
    finish_info, generate_request,
    inference_server::{Inference, InferenceServer},
    FinishInfo, GenerateRequest, GenerateResponse, PromptInfo, SequenceOutput, TokenIds,
};

#[derive(Clone, Default)]
pub struct CapturedGrpcGenerate {
    pub request_id: String,
    pub include_prefill_progress: bool,
    pub token_ids: Vec<u32>,
    pub had_text_prompt: bool,
    pub model: String,
    pub cache_salt: String,
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
    /// `None` models a worker without capability discovery.
    pub prefill_progress_supported: Option<bool>,
    /// Optional externally driven stream for lifecycle and progress tests.
    pub scripted_stream: Arc<Mutex<Option<ResponseStream>>>,
}

impl Default for MockVllmRs {
    fn default() -> Self {
        Self {
            captured: Arc::new(Mutex::new(Vec::new())),
            reply_text: "hello from worker".to_string(),
            frames: None,
            capture_limit: usize::MAX,
            failures_remaining: Arc::new(AtomicUsize::new(0)),
            prefill_progress_supported: None,
            scripted_stream: Arc::new(Mutex::new(None)),
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
}

pub type ResponseStream = Pin<Box<dyn Stream<Item = Result<GenerateResponse, Status>> + Send>>;

#[tonic::async_trait]
impl Inference for MockVllmRs {
    type GenerateStreamStream = ResponseStream;

    async fn generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<GenerateResponse>, Status> {
        let mut stream = self.generate_stream(request).await?.into_inner();
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
            prefill_progress: None,
        }))
    }

    async fn generate_stream(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        let req = request.into_inner();
        let (token_ids, had_text_prompt) = match req.prompt {
            Some(generate_request::Prompt::TokenIds(TokenIds { ids })) => (ids, false),
            Some(generate_request::Prompt::Text(_)) => (Vec::new(), true),
            None => (Vec::new(), false),
        };
        {
            let mut captured = self.captured.lock().unwrap();
            if captured.len() < self.capture_limit {
                captured.push(CapturedGrpcGenerate {
                    request_id: req.request_id,
                    include_prefill_progress: req.include_prefill_progress,
                    token_ids: token_ids.clone(),
                    had_text_prompt,
                    model: req.model,
                    cache_salt: req.kv.map(|kv| kv.cache_salt).unwrap_or_default(),
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

        if let Some(stream) = self.scripted_stream.lock().unwrap().take() {
            return Ok(Response::new(stream));
        }

        let (tx, rx) = mpsc::channel(4);
        let frames = self
            .frames
            .clone()
            .unwrap_or_else(|| vec![(self.reply_text.clone(), vec![7, 8, 9])]);
        let n_prompt = token_ids.len() as u32;
        tokio::spawn(async move {
            let _ = tx
                .send(Ok(GenerateResponse {
                    prompt_info: Some(PromptInfo {
                        num_prompt_tokens: n_prompt,
                        ..Default::default()
                    }),
                    outputs: None,
                    prefill_progress: None,
                }))
                .await;
            let last = frames.len().saturating_sub(1);
            let total: u32 = frames.iter().map(|(_, ids)| ids.len() as u32).sum();
            for (index, (text, token_ids)) in frames.into_iter().enumerate() {
                let finish_info = (index == last).then(|| FinishInfo {
                    num_output_tokens: total,
                    finish_reason: finish_info::FinishReason::Stop as i32,
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
                        prefill_progress: None,
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
        #[tonic::async_trait]
        impl Control for MockVllmRs {
            async fn get_server_info(
                &self,
                _request: Request<pb::GetServerInfoRequest>,
            ) -> Result<Response<pb::ServerInfo>, Status> {
                let supported = self.prefill_progress_supported
                    .ok_or_else(|| Status::unimplemented("GetServerInfo"))?;
                Ok(Response::new(pb::ServerInfo {
                    supports_prefill_progress: supported,
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
    get_model_info(GetModelInfoRequest) -> ModelInfo;
    shutdown(ShutdownRequest) -> ShutdownResponse;
    abort(AbortRequest) -> AbortResponse;
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
