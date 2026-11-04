//! gRPC Control helpers for all-`grpc://` worker pools.
//!
//! vLLM Rust exposes model/server discovery through the Control service, while
//! OpenAI-compatible HTTP routes still need their own response shapes.

use std::time::Duration;

use serde_json::{json, Value};
use tonic::transport::Channel;

use super::detect::grpc_connect_uri;
use super::pb::{
    control_client::ControlClient, GetKvEventSourcesRequest, GetKvEventSourcesResponse,
    GetModelInfoRequest, GetServerInfoRequest, ModelInfo, ServerInfo,
};

async fn control_client(
    worker_url: &str,
    timeout: Duration,
) -> Result<ControlClient<Channel>, String> {
    let uri = grpc_connect_uri(worker_url)?;
    let channel = Channel::from_shared(uri.clone())
        .map_err(|e| format!("invalid grpc uri {uri}: {e}"))?
        .connect_timeout(timeout)
        .timeout(timeout)
        .connect()
        .await
        .map_err(|e| format!("grpc control connect {uri}: {e}"))?;
    Ok(ControlClient::new(channel))
}

pub async fn get_grpc_model_info(worker_url: &str, timeout: Duration) -> Result<ModelInfo, String> {
    let uri = grpc_connect_uri(worker_url)?;
    let mut client = control_client(worker_url, timeout).await?;
    tokio::time::timeout(timeout, client.get_model_info(GetModelInfoRequest {}))
        .await
        .map_err(|_| format!("grpc GetModelInfo timeout {uri}"))?
        .map_err(|e| format!("grpc GetModelInfo {uri}: {e}"))
        .map(|response| response.into_inner())
}

pub async fn get_grpc_server_info(
    worker_url: &str,
    timeout: Duration,
) -> Result<ServerInfo, String> {
    let uri = grpc_connect_uri(worker_url)?;
    let mut client = control_client(worker_url, timeout).await?;
    tokio::time::timeout(timeout, client.get_server_info(GetServerInfoRequest {}))
        .await
        .map_err(|_| format!("grpc GetServerInfo timeout {uri}"))?
        .map_err(|e| format!("grpc GetServerInfo {uri}: {e}"))
        .map(|response| response.into_inner())
}

pub async fn get_grpc_kv_event_sources(
    worker_url: &str,
    timeout: Duration,
) -> Result<GetKvEventSourcesResponse, String> {
    let uri = grpc_connect_uri(worker_url)?;
    let mut client = control_client(worker_url, timeout).await?;
    tokio::time::timeout(
        timeout,
        client.get_kv_event_sources(GetKvEventSourcesRequest {}),
    )
    .await
    .map_err(|_| format!("grpc GetKvEventSources timeout {uri}"))?
    .map_err(|e| format!("grpc GetKvEventSources {uri}: {e}"))
    .map(|response| response.into_inner())
}

pub fn model_info_json(info: &ModelInfo) -> Value {
    json!({
        "model_id": info.model_id,
        "served_model_name": info.served_model_name,
        "served_model_aliases": info.served_model_aliases,
        "supports_text_input": info.supports_text_input,
        "supports_token_ids_input": info.supports_token_ids_input,
        "supports_lora": info.supports_lora,
        "supports_multimodal": info.supports_multimodal,
        "reasoning_parser": info.reasoning_parser,
        "tool_call_parser": info.tool_call_parser,
    })
}

/// True when `version` is a dotted release older than 0.30.0.
/// Values that are not a release, such as the test mock's `"mock"`, are false.
pub fn grpc_engine_version_older_than_0_30(version: &str) -> bool {
    let raw = version.trim().trim_start_matches(['v', 'V']);
    if !raw.starts_with(|c: char| c.is_ascii_digit()) {
        return false;
    }
    let mut numbers = raw
        .split(|c: char| !c.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .map(|part| part.parse::<u64>().unwrap_or(0));
    let major = numbers.next().unwrap_or(0);
    let minor = numbers.next().unwrap_or(0);
    let patch = numbers.next().unwrap_or(0);
    (major, minor, patch) < (0, 30, 0)
}

/// Log when a gRPC P/D worker will abort every decode. Health checks pass on
/// vLLM 0.29.x; the float-port failure only shows up on the first request.
pub async fn warn_old_grpc_pd_workers<'a>(urls: impl IntoIterator<Item = &'a str>) {
    for url in urls {
        if !crate::backend::is_grpc_url(url) {
            continue;
        }
        match get_grpc_server_info(url, Duration::from_secs(2)).await {
            Ok(info) if grpc_engine_version_older_than_0_30(&info.engine_version) => {
                tracing::warn!(
                    worker = %url,
                    engine_version = %info.engine_version,
                    "gRPC P/D worker is older than vLLM 0.30.0; every decode aborts because connector ports and block ids arrive as floats. Upgrade the worker. Upgrading the router does not fix this."
                );
            }
            Ok(_) => {}
            Err(error) => {
                tracing::warn!(
                    worker = %url,
                    %error,
                    "could not read the gRPC P/D worker engine version"
                );
            }
        }
    }
}

pub fn server_info_json(info: &ServerInfo) -> Value {
    json!({
        "engine_version": info.engine_version,
        "api_version": info.api_version,
        "instance_id": info.instance_id,
        "parallelism": info.parallelism.as_ref().map(|parallelism| json!({
            "tensor_parallel_size": parallelism.tensor_parallel_size,
            "pipeline_parallel_size": parallelism.pipeline_parallel_size,
            "data_parallel_size": parallelism.data_parallel_size,
            "data_parallel_rank": parallelism.data_parallel_rank,
            "decode_context_parallel_size": parallelism.decode_context_parallel_size,
            "world_size": parallelism.world_size,
        })),
        "max_model_len": info.max_model_len,
        "kv_block_size": info.kv_block_size,
        "total_kv_blocks": info.total_kv_blocks,
        "max_running_requests": info.max_running_requests,
        "max_batched_tokens": info.max_batched_tokens,
        "max_loras": info.max_loras,
        "effective_attention_block_size": info.effective_attention_block_size,
        "rl_capabilities": info.rl_capabilities.as_ref().map(|capabilities| json!({
            "weight_transfer_enabled": capabilities.weight_transfer_enabled,
            "weight_transfer_backend": capabilities.weight_transfer_backend,
            "sleep_mode_enabled": capabilities.sleep_mode_enabled,
            "draft_weight_updates_enabled": capabilities.draft_weight_updates_enabled,
        })),
    })
}

pub fn openai_models_json(info: &ModelInfo) -> Value {
    let primary = if info.served_model_name.is_empty() {
        &info.model_id
    } else {
        &info.served_model_name
    };
    let ids = std::iter::once(primary)
        .chain(info.served_model_aliases.iter())
        .filter(|id| !id.is_empty());
    let data: Vec<_> = ids
        .map(|id| {
            json!({
                "id": id,
                "object": "model",
                "created": 0,
                "owned_by": "vllm",
            })
        })
        .collect();
    json!({
        "object": "list",
        "data": data,
    })
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::backend::pb::{ParallelismInfo, RlCapabilities};

    #[test]
    fn openai_models_uses_served_name_and_aliases() {
        let info = ModelInfo {
            model_id: "/models/qwen".to_string(),
            served_model_name: "qwen".to_string(),
            served_model_aliases: vec!["alias-a".to_string(), "alias-b".to_string()],
            ..Default::default()
        };

        let value = openai_models_json(&info);
        assert_eq!(value["object"], "list");
        assert_eq!(value["data"][0]["id"], "qwen");
        assert_eq!(value["data"][1]["id"], "alias-a");
        assert_eq!(value["data"][2]["id"], "alias-b");
        assert_eq!(value["data"][0]["object"], "model");
    }

    #[test]
    fn openai_models_falls_back_to_model_id_when_served_name_empty() {
        let info = ModelInfo {
            model_id: "/models/qwen".to_string(),
            served_model_name: String::new(),
            ..Default::default()
        };

        let value = openai_models_json(&info);
        assert_eq!(value["data"][0]["id"], "/models/qwen");
    }

    #[test]
    fn raw_control_json_preserves_discovery_fields() {
        let model = ModelInfo {
            model_id: "model-id".to_string(),
            supports_text_input: true,
            supports_token_ids_input: true,
            supports_lora: true,
            supports_multimodal: false,
            reasoning_parser: "r".to_string(),
            tool_call_parser: "t".to_string(),
            ..Default::default()
        };
        let model_json = model_info_json(&model);
        assert_eq!(model_json["model_id"], "model-id");
        assert_eq!(model_json["supports_token_ids_input"], true);
        assert_eq!(model_json["tool_call_parser"], "t");

        let server = ServerInfo {
            engine_version: "0.29.0".to_string(),
            api_version: "0.2".to_string(),
            parallelism: Some(ParallelismInfo {
                tensor_parallel_size: 2,
                world_size: 2,
                ..Default::default()
            }),
            rl_capabilities: Some(RlCapabilities {
                sleep_mode_enabled: true,
                ..Default::default()
            }),
            max_model_len: 4096,
            effective_attention_block_size: Some(128),
            ..Default::default()
        };
        let server_json = server_info_json(&server);
        assert_eq!(server_json["engine_version"], "0.29.0");
        assert_eq!(server_json["parallelism"]["tensor_parallel_size"], 2);
        assert_eq!(server_json["rl_capabilities"]["sleep_mode_enabled"], true);
        assert_eq!(server_json["effective_attention_block_size"], 128);
    }

    #[test]
    fn old_grpc_engine_versions_are_the_ones_before_0_30() {
        assert!(grpc_engine_version_older_than_0_30("0.29.0"));
        assert!(grpc_engine_version_older_than_0_30("v0.29.1"));
        assert!(grpc_engine_version_older_than_0_30("0.29.0+cu129"));
        assert!(!grpc_engine_version_older_than_0_30("0.30.0"));
        assert!(!grpc_engine_version_older_than_0_30("0.32.0"));
        assert!(!grpc_engine_version_older_than_0_30("mock"));
        assert!(!grpc_engine_version_older_than_0_30(""));
    }
}
