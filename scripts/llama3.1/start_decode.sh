#!/bin/bash

# Start vLLM decode server with NIXL Connector
# This script starts a vLLM server configured for decode operations with NIXL transfer

# Activate virtual environment
source ~/uv_env/vllm/bin/activate

# Change to vllm directory
cd ~/gitrepos/vllm

# Set NIXL environment variables
export VLLM_NIXL_SIDE_CHANNEL_HOST=0.0.0.0
export VLLM_NIXL_SIDE_CHANNEL_PORT=8084
export UCX_TLS=all
export UCX_NET_DEVICES=all
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_RPC_TIMEOUT=300
export VLLM_WORKER_RPC_TIMEOUT=300
export HF_HUB_DISABLE_XET="1"

echo "NIXL configuration:"
echo "  Side channel host: $VLLM_NIXL_SIDE_CHANNEL_HOST"
echo "  Side channel port: $VLLM_NIXL_SIDE_CHANNEL_PORT"

CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8082 \
    --tensor-parallel-size 1 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"backends":["UCX","GDS"],"http_port":8086}}' \
    --disable-log-stats \
    2>&1 | tee decode.log
