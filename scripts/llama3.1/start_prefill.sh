#!/bin/bash

# Start vLLM prefill server with NIXL Connector and DP=4
# This script starts a vLLM server configured for prefill operations with NIXL transfer

# Set NIXL environment variables
# Note: With DP=4, the base port 8083 will expand to use ports 8083-8086 (one per replica)
export VLLM_NIXL_SIDE_CHANNEL_HOST=0.0.0.0
export VLLM_NIXL_SIDE_CHANNEL_PORT=8083
export UCX_TLS=all
export UCX_NET_DEVICES=all
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_RPC_TIMEOUT=300
export VLLM_WORKER_RPC_TIMEOUT=300
export FLASHINFER_DISABLE_VERSION_CHECK=1
# export HF_HUB_DISABLE_XET="1"

echo "NIXL configuration:"
echo "  Side channel host: $VLLM_NIXL_SIDE_CHANNEL_HOST"
echo "  Side channel base port: $VLLM_NIXL_SIDE_CHANNEL_PORT (will use 8083-8086 for DP=4)"
echo "  Data parallel size: 4"
echo "  NIXL HTTP port: 8097"

CUDA_VISIBLE_DEVICES=0,1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8081 \
    --tensor-parallel-size 1 \
    --data-parallel-size 2 \
    --enforce-eager \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.9 \
    --disable-hybrid-kv-cache-manager \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"backends":["UCX","GDS"],"http_port":8097}}' \
    --disable-log-stats \
    2>&1 | tee prefill.log
