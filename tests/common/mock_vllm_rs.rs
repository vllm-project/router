//! In-process vLLM 0.29 `Inference` gRPC worker for router tests.
//!
//! Also serves `grpc.health.v1` as `SERVING` so `Router::new` startup
//! probes pass. Used by `tests/grpc_vs_http_e2e.rs`. Records `token_ids`
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
}

impl Default for MockVllmRs {
    fn default() -> Self {
        Self {
            captured: Arc::new(Mutex::new(Vec::new())),
            reply_text: "hello from worker".to_string(),
            capture_limit: usize::MAX,
            failures_remaining: Arc::new(AtomicUsize::new(0)),
            finish_kv_transfer_params: None,
            reply_delay: Duration::ZERO,
            cancelled: Arc::new(AtomicUsize::new(0)),
            stream_failures_remaining: Arc::new(AtomicUsize::new(0)),
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
        let handle = tokio::spawn(async move {
            Server::builder()
                .add_service(health_service)
                .add_service(svc)
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
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .is_ok()
        {
            return Err(Status::unavailable("retry test"));
        }

        let (tx, rx) = mpsc::channel(4);
        let reply = self.reply_text.clone();
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
            let _ = tx
                .send(Ok(GenerateResponse {
                    prompt_info: None,
                    outputs: Some(SequenceOutput {
                        index: 0,
                        text: reply,
                        num_tokens: 3,
                        token_ids: vec![7, 8, 9],
                        finish_info: Some(FinishInfo {
                            num_output_tokens: 3,
                            finish_reason: finish_info::FinishReason::Stop as i32,
                            kv_transfer_params: finish_kv,
                            ..Default::default()
                        }),
                        ..Default::default()
                    }),
                }))
                .await;
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}
