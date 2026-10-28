//! Mock vLLM Inference + Control worker for router tests.
//!
//! Speaks the same `vllm-proto` 0.2 services the router dials on `grpc://`:
//! `grpc.health.v1` (empty service name is SERVING by default), Control, and
//! Inference `Generate` / `GenerateStream`. It does not load a model. Each
//! response is a fixed string in `outputs.text` plus `finish_reason=STOP`, so
//! the router can forward that text without detokenizing worker tokens.
//!
//! Prefix-cache stickiness is not simulated: no KV-cache events are published.
//! `--kv-events-endpoint` only answers `GetKvEventSources`.
//!
//! ```text
//! cargo run --manifest-path tests/python/fixtures/mock_grpc_worker/Cargo.toml -- --grpc-port 50051 --http-port 8000
//! ```

use std::env;
use std::io::{self, Write};
use std::net::SocketAddr;
use std::pin::Pin;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_stream::Stream;
use tonic::transport::Server;
use tonic::{Request, Response, Status};
use tonic_health::server::health_reporter;
use tonic_health::ServingStatus;
use vllm_proto::control_server::{Control, ControlServer};
use vllm_proto::inference_server::{Inference, InferenceServer};
use vllm_proto::{
    finish_info, AbortRequest, AbortResponse, FinishInfo, FinishWeightUpdateRequest,
    FinishWeightUpdateResponse, GenerateRequest, GenerateResponse, GetKvEventSourcesRequest,
    GetKvEventSourcesResponse, GetModelInfoRequest, GetServerInfoRequest, GetWeightVersionRequest,
    GetWeightVersionResponse, InitWeightTransferEngineRequest, InitWeightTransferEngineResponse,
    IsPausedRequest, IsPausedResponse, IsSleepingRequest, IsSleepingResponse, KvEventSource,
    ListLorasRequest, ListLorasResponse, LoadLoraRequest, LoadLoraResponse, ModelInfo,
    PauseGenerationRequest, PauseGenerationResponse, PromptInfo, ResumeGenerationRequest,
    ResumeGenerationResponse, SequenceOutput, ServerInfo, SleepRequest, SleepResponse,
    StartDraftWeightUpdateRequest, StartDraftWeightUpdateResponse, StartWeightUpdateRequest,
    StartWeightUpdateResponse, UnloadLoraRequest, UnloadLoraResponse, UpdateWeightsRequest,
    UpdateWeightsResponse, UpdateWeightVersionRequest, UpdateWeightVersionResponse,
    WakeUpRequest, WakeUpResponse,
};

type GenerateStream = Pin<Box<dyn Stream<Item = Result<GenerateResponse, Status>> + Send>>;

#[derive(Clone)]
struct MockWorker {
    model_id: String,
    text: String,
    kv_events_endpoint: Option<String>,
    kv_replay_endpoint: Option<String>,
}

impl MockWorker {
    fn finished_output(&self, req: &GenerateRequest) -> GenerateResponse {
        log_line(&format!("request_id={}", req.request_id));
        GenerateResponse {
            prompt_info: Some(PromptInfo {
                num_prompt_tokens: prompt_len(req),
                ..Default::default()
            }),
            outputs: Some(SequenceOutput {
                text: self.text.clone(),
                num_tokens: 1,
                finish_info: Some(FinishInfo {
                    num_output_tokens: 1,
                    finish_reason: finish_info::FinishReason::Stop as i32,
                    ..Default::default()
                }),
                ..Default::default()
            }),
        }
    }
}

#[tonic::async_trait]
impl Inference for MockWorker {
    async fn generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<GenerateResponse>, Status> {
        Ok(Response::new(self.finished_output(request.get_ref())))
    }

    type GenerateStreamStream = GenerateStream;

    async fn generate_stream(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        let req = request.into_inner();
        log_line(&format!("request_id={}", req.request_id));
        let prompt = GenerateResponse {
            prompt_info: Some(PromptInfo {
                num_prompt_tokens: prompt_len(&req),
                ..Default::default()
            }),
            outputs: None,
        };
        let output = GenerateResponse {
            prompt_info: None,
            outputs: Some(SequenceOutput {
                text: self.text.clone(),
                num_tokens: 1,
                finish_info: Some(FinishInfo {
                    num_output_tokens: 1,
                    finish_reason: finish_info::FinishReason::Stop as i32,
                    ..Default::default()
                }),
                ..Default::default()
            }),
        };
        let stream = tokio_stream::iter(vec![Ok(prompt), Ok(output)]);
        Ok(Response::new(Box::pin(stream)))
    }
}

fn ok<T: Default>(_request: Request<impl Sized>) -> Result<Response<T>, Status> {
    Ok(Response::new(T::default()))
}

#[tonic::async_trait]
impl Control for MockWorker {
    async fn get_server_info(
        &self,
        _request: Request<GetServerInfoRequest>,
    ) -> Result<Response<ServerInfo>, Status> {
        Ok(Response::new(ServerInfo {
            engine_version: "mock".to_string(),
            api_version: "0.2".to_string(),
            max_model_len: 8192,
            ..Default::default()
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<GetModelInfoRequest>,
    ) -> Result<Response<ModelInfo>, Status> {
        Ok(Response::new(ModelInfo {
            model_id: self.model_id.clone(),
            served_model_name: self.model_id.clone(),
            supports_token_ids_input: true,
            supports_text_input: true,
            ..Default::default()
        }))
    }

    async fn abort(
        &self,
        request: Request<AbortRequest>,
    ) -> Result<Response<AbortResponse>, Status> {
        log_line(&format!(
            "abort request_ids={}",
            request.get_ref().request_ids.join(",")
        ));
        Ok(Response::new(AbortResponse::default()))
    }

    async fn load_lora(
        &self,
        request: Request<LoadLoraRequest>,
    ) -> Result<Response<LoadLoraResponse>, Status> {
        ok(request)
    }

    async fn unload_lora(
        &self,
        request: Request<UnloadLoraRequest>,
    ) -> Result<Response<UnloadLoraResponse>, Status> {
        ok(request)
    }

    async fn list_loras(
        &self,
        request: Request<ListLorasRequest>,
    ) -> Result<Response<ListLorasResponse>, Status> {
        ok(request)
    }

    async fn get_kv_event_sources(
        &self,
        _request: Request<GetKvEventSourcesRequest>,
    ) -> Result<Response<GetKvEventSourcesResponse>, Status> {
        let sources = self
            .kv_events_endpoint
            .as_ref()
            .map(|endpoint| {
                vec![KvEventSource {
                    transport: "zmq".to_string(),
                    endpoint: endpoint.clone(),
                    replay_endpoint: self.kv_replay_endpoint.clone().unwrap_or_default(),
                    data_parallel_rank: Some(0),
                    encoding: "msgpack".to_string(),
                    schema_version: 1,
                    buffer_steps: 1,
                    hwm: 1000,
                    ..Default::default()
                }]
            })
            .unwrap_or_default();
        Ok(Response::new(GetKvEventSourcesResponse { sources }))
    }

    async fn pause_generation(
        &self,
        request: Request<PauseGenerationRequest>,
    ) -> Result<Response<PauseGenerationResponse>, Status> {
        ok(request)
    }

    async fn resume_generation(
        &self,
        request: Request<ResumeGenerationRequest>,
    ) -> Result<Response<ResumeGenerationResponse>, Status> {
        ok(request)
    }

    async fn is_paused(
        &self,
        request: Request<IsPausedRequest>,
    ) -> Result<Response<IsPausedResponse>, Status> {
        ok(request)
    }

    async fn sleep(
        &self,
        request: Request<SleepRequest>,
    ) -> Result<Response<SleepResponse>, Status> {
        ok(request)
    }

    async fn wake_up(
        &self,
        request: Request<WakeUpRequest>,
    ) -> Result<Response<WakeUpResponse>, Status> {
        ok(request)
    }

    async fn is_sleeping(
        &self,
        request: Request<IsSleepingRequest>,
    ) -> Result<Response<IsSleepingResponse>, Status> {
        ok(request)
    }

    async fn init_weight_transfer_engine(
        &self,
        request: Request<InitWeightTransferEngineRequest>,
    ) -> Result<Response<InitWeightTransferEngineResponse>, Status> {
        ok(request)
    }

    async fn start_weight_update(
        &self,
        request: Request<StartWeightUpdateRequest>,
    ) -> Result<Response<StartWeightUpdateResponse>, Status> {
        ok(request)
    }

    async fn start_draft_weight_update(
        &self,
        request: Request<StartDraftWeightUpdateRequest>,
    ) -> Result<Response<StartDraftWeightUpdateResponse>, Status> {
        ok(request)
    }

    async fn update_weights(
        &self,
        request: Request<UpdateWeightsRequest>,
    ) -> Result<Response<UpdateWeightsResponse>, Status> {
        ok(request)
    }

    async fn finish_weight_update(
        &self,
        request: Request<FinishWeightUpdateRequest>,
    ) -> Result<Response<FinishWeightUpdateResponse>, Status> {
        ok(request)
    }

    async fn update_weight_version(
        &self,
        request: Request<UpdateWeightVersionRequest>,
    ) -> Result<Response<UpdateWeightVersionResponse>, Status> {
        ok(request)
    }

    async fn get_weight_version(
        &self,
        request: Request<GetWeightVersionRequest>,
    ) -> Result<Response<GetWeightVersionResponse>, Status> {
        ok(request)
    }
}

fn log_line(line: &str) {
    println!("{line}");
    let _ = io::stdout().flush();
}

fn prompt_len(req: &GenerateRequest) -> u32 {
    match &req.prompt {
        Some(vllm_proto::generate_request::Prompt::TokenIds(ids)) => ids.ids.len() as u32,
        _ => 0,
    }
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

fn require_port(args: &[String], name: &str) -> u16 {
    flag(args, name)
        .unwrap_or_else(|| {
            eprintln!("missing {name}");
            std::process::exit(2);
        })
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("{name} is not a port");
            std::process::exit(2);
        })
}

async fn serve_http_health(port: u16) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], port))).await?;
    loop {
        let (mut socket, _) = listener.accept().await?;
        tokio::spawn(async move {
            let mut buf = [0_u8; 1024];
            let _ = socket.read(&mut buf).await;
            let body = b"ok";
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = socket.write_all(header.as_bytes()).await;
            let _ = socket.write_all(body).await;
        });
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        eprintln!(
            "mock-grpc-worker --grpc-port PORT [--http-port PORT] [--model ID] [--text TEXT] [--kv-events-endpoint tcp://127.0.0.1:5557] [--kv-replay-endpoint tcp://127.0.0.1:5558]"
        );
        return Ok(());
    }

    let grpc_port = require_port(&args, "--grpc-port");
    let http_port = flag(&args, "--http-port").and_then(|value| value.parse().ok());
    let worker = MockWorker {
        model_id: flag(&args, "--model").unwrap_or_else(|| "mock-model".to_string()),
        text: flag(&args, "--text").unwrap_or_else(|| "mock".to_string()),
        kv_events_endpoint: flag(&args, "--kv-events-endpoint"),
        kv_replay_endpoint: flag(&args, "--kv-replay-endpoint"),
    };

    let (health_reporter, health_service) = health_reporter();
    health_reporter
        .set_service_status("", ServingStatus::Serving)
        .await;
    health_reporter
        .set_serving::<InferenceServer<MockWorker>>()
        .await;
    health_reporter
        .set_serving::<ControlServer<MockWorker>>()
        .await;

    if let Some(port) = http_port {
        tokio::spawn(async move {
            if let Err(error) = serve_http_health(port).await {
                eprintln!("http health failed: {error}");
            }
        });
    }

    let addr = SocketAddr::from(([127, 0, 0, 1], grpc_port));
    log_line(&format!("grpc://{addr}"));
    Server::builder()
        .add_service(health_service)
        .add_service(InferenceServer::new(worker.clone()))
        .add_service(ControlServer::new(worker))
        .serve(addr)
        .await?;
    Ok(())
}
