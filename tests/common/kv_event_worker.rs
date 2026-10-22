use crate::common::mock_vllm_rs::MockVllmRs;
use std::time::Duration;
use tonic::{Request, Response, Status};
use vllm_router_rs::backend::pb;

#[derive(Clone)]
pub struct KvControl {
    instance: String,
    endpoint: String,
}

pub struct EventWorker {
    pub url: String,
    pub state: MockVllmRs,
    publisher: zmq::Socket,
    task: tokio::task::JoinHandle<()>,
}

impl EventWorker {
    pub async fn spawn(instance: &str) -> Self {
        let context = zmq::Context::new();
        let publisher = context.socket(zmq::XPUB).unwrap();
        publisher.set_linger(0).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (reporter, health) = tonic_health::server::health_reporter();
        reporter
            .set_service_status("", tonic_health::ServingStatus::Serving)
            .await;
        let state = MockVllmRs {
            reply_text: instance.to_string(),
            ..Default::default()
        };
        let inference = pb::inference_server::InferenceServer::new(state.clone());
        let control = pb::control_server::ControlServer::new(KvControl {
            instance: instance.to_string(),
            endpoint,
        });
        let task = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(health)
                .add_service(inference)
                .add_service(control)
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
                .unwrap();
        });
        Self {
            url: format!("grpc://{address}"),
            state,
            publisher,
            task,
        }
    }

    pub async fn wait_for_subscription(&self) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        loop {
            match self.publisher.recv_bytes(zmq::DONTWAIT) {
                Ok(message) if message.first() == Some(&1) => return,
                Ok(_) | Err(zmq::Error::EAGAIN) => {}
                Err(error) => panic!("publisher failed: {error}"),
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "SMetric did not subscribe to the worker's KV events"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    pub fn publish(&self, seq: i64, event: serde_json::Value) {
        let event: vllm_router_rs::kv_index::KVEvent = serde_json::from_value(event).unwrap();
        let payload = rmp_serde::to_vec_named(&(0.0_f64, vec![event], 0_u32)).unwrap();
        self.publisher
            .send_multipart(
                [b"kv-events".to_vec(), seq.to_be_bytes().to_vec(), payload],
                0,
            )
            .unwrap();
    }
}

impl Drop for EventWorker {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[tonic::async_trait]
impl pb::control_server::Control for KvControl {
    async fn get_server_info(
        &self,
        _: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::ServerInfo>, Status> {
        Ok(Response::new(pb::ServerInfo {
            instance_id: self.instance.clone(),
            kv_block_size: 4,
            parallelism: Some(pb::ParallelismInfo {
                data_parallel_size: 1,
                tensor_parallel_size: 1,
                ..Default::default()
            }),
            ..Default::default()
        }))
    }
    async fn get_model_info(
        &self,
        _: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        Ok(Response::new(pb::ModelInfo {
            model_id: "test-model".into(),
            supports_token_ids_input: true,
            ..Default::default()
        }))
    }
    async fn get_kv_event_sources(
        &self,
        _: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        Ok(Response::new(pb::GetKvEventSourcesResponse {
            sources: vec![pb::KvEventSource {
                transport: "zmq".into(),
                endpoint: self.endpoint.clone(),
                topic: "kv-events".into(),
                data_parallel_rank: Some(0),
                encoding: "msgpack".into(),
                schema_version: 1,
                hwm: 1000,
                ..Default::default()
            }],
        }))
    }
    async fn abort(
        &self,
        _: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        Err(Status::unimplemented("unused control operation"))
    }
    async fn load_lora(
        &self,
        _: Request<pb::LoadLoraRequest>,
    ) -> Result<Response<pb::LoadLoraResponse>, Status> {
        Err(Status::unimplemented("unused control operation"))
    }
    async fn unload_lora(
        &self,
        _: Request<pb::UnloadLoraRequest>,
    ) -> Result<Response<pb::UnloadLoraResponse>, Status> {
        Err(Status::unimplemented("unused control operation"))
    }
    async fn list_loras(
        &self,
        _: Request<pb::ListLorasRequest>,
    ) -> Result<Response<pb::ListLorasResponse>, Status> {
        Err(Status::unimplemented("unused control operation"))
    }
    async fn pause_generation(
        &self,
        _: Request<pb::PauseGenerationRequest>,
    ) -> Result<Response<pb::PauseGenerationResponse>, Status> {
        Err(Status::unimplemented("unused control operation"))
    }
    async fn resume_generation(
        &self,
        _: Request<pb::ResumeGenerationRequest>,
    ) -> Result<Response<pb::ResumeGenerationResponse>, Status> {
        Err(Status::unimplemented("unused control operation"))
    }
    async fn is_paused(
        &self,
        _: Request<pb::IsPausedRequest>,
    ) -> Result<Response<pb::IsPausedResponse>, Status> {
        Err(Status::unimplemented("unused control operation"))
    }
    async fn sleep(
        &self,
        _: Request<pb::SleepRequest>,
    ) -> Result<Response<pb::SleepResponse>, Status> {
        Err(Status::unimplemented("unused control operation"))
    }
    async fn wake_up(
        &self,
        _: Request<pb::WakeUpRequest>,
    ) -> Result<Response<pb::WakeUpResponse>, Status> {
        Err(Status::unimplemented("unused control operation"))
    }
    async fn is_sleeping(
        &self,
        _: Request<pb::IsSleepingRequest>,
    ) -> Result<Response<pb::IsSleepingResponse>, Status> {
        Err(Status::unimplemented("unused control operation"))
    }
    async fn init_weight_transfer_engine(
        &self,
        _: Request<pb::InitWeightTransferEngineRequest>,
    ) -> Result<Response<pb::InitWeightTransferEngineResponse>, Status> {
        Err(Status::unimplemented("unused control operation"))
    }
    async fn start_weight_update(
        &self,
        _: Request<pb::StartWeightUpdateRequest>,
    ) -> Result<Response<pb::StartWeightUpdateResponse>, Status> {
        Err(Status::unimplemented("unused control operation"))
    }
    async fn start_draft_weight_update(
        &self,
        _: Request<pb::StartDraftWeightUpdateRequest>,
    ) -> Result<Response<pb::StartDraftWeightUpdateResponse>, Status> {
        Err(Status::unimplemented("unused control operation"))
    }
    async fn update_weights(
        &self,
        _: Request<pb::UpdateWeightsRequest>,
    ) -> Result<Response<pb::UpdateWeightsResponse>, Status> {
        Err(Status::unimplemented("unused control operation"))
    }
    async fn finish_weight_update(
        &self,
        _: Request<pb::FinishWeightUpdateRequest>,
    ) -> Result<Response<pb::FinishWeightUpdateResponse>, Status> {
        Err(Status::unimplemented("unused control operation"))
    }
    async fn update_weight_version(
        &self,
        _: Request<pb::UpdateWeightVersionRequest>,
    ) -> Result<Response<pb::UpdateWeightVersionResponse>, Status> {
        Err(Status::unimplemented("unused control operation"))
    }
    async fn get_weight_version(
        &self,
        _: Request<pb::GetWeightVersionRequest>,
    ) -> Result<Response<pb::GetWeightVersionResponse>, Status> {
        Err(Status::unimplemented("unused control operation"))
    }
}
