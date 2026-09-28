//! gRPC Control helpers for all-`grpc://` worker pools.
//!
//! vLLM Rust exposes model/server discovery through the Control service, while
//! OpenAI-compatible HTTP routes still need their own response shapes.

use std::time::Duration;

use serde_json::{json, Value};
use tonic::transport::Channel;

use super::detect::grpc_connect_uri;
use super::pb::{
    control_client::ControlClient, GetModelInfoRequest, GetServerInfoRequest, ModelInfo, ServerInfo,
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

#[cfg(test)]
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
}
