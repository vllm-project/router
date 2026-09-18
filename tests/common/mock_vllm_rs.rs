//! In-process vLLM 0.29 `Inference` gRPC worker for router tests.
//!
//! Also serves `grpc.health.v1` as `SERVING` so `Router::new` startup
//! probes pass. Used by `tests/grpc_vs_http_e2e.rs`. Records `token_ids`
//! vs text prompt on each `GenerateStream`. Not Python EngineCore.

#![allow(dead_code)]

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::Stream;
use tonic::{transport::Server, Request, Response, Status};
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
}

#[derive(Clone)]
pub struct MockVllmRs {
    pub captured: Arc<Mutex<Vec<CapturedGrpcGenerate>>>,
    pub reply_text: String,
}

impl Default for MockVllmRs {
    fn default() -> Self {
        Self {
            captured: Arc::new(Mutex::new(Vec::new())),
            reply_text: "hello from worker".to_string(),
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
        let state = MockVllmRs::default();
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
        let req = request.into_inner();
        let (token_ids, had_text_prompt) = match req.prompt {
            Some(generate_request::Prompt::TokenIds(TokenIds { ids })) => (ids, false),
            Some(generate_request::Prompt::Text(_)) => (Vec::new(), true),
            None => (Vec::new(), false),
        };
        self.captured.lock().unwrap().push(CapturedGrpcGenerate {
            token_ids: token_ids.clone(),
            had_text_prompt,
            model: req.model,
        });

        let (tx, rx) = mpsc::channel(4);
        let reply = self.reply_text.clone();
        let n_prompt = token_ids.len() as u32;
        tokio::spawn(async move {
            let _ = tx
                .send(Ok(GenerateResponse {
                    prompt_info: Some(PromptInfo {
                        num_prompt_tokens: n_prompt,
                        ..Default::default()
                    }),
                    outputs: None,
                }))
                .await;
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
