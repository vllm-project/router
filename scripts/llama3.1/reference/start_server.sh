#!/bin/bash

# Start standard vLLM server for baseline benchmarking (without router)
# This runs a unified vLLM server where prefill and decode happen together
# No prefill/decode disaggregation - standard vLLM behavior

# Activate virtual environment
source ~/uv_env/vllm/bin/activate

# Change to vllm directory
cd /data/users/nlalit/gitrepos/vllm

# Environment variables for optimal performance
export VLLM_USE_V1=1
export HF_HUB_DISABLE_XET="1"

echo "=========================================="
echo "Starting BASELINE vLLM Server"
echo "=========================================="
echo "Configuration:"
echo "  Model: meta-llama/Llama-3.1-8B-Instruct"
echo "  Mode: UNIFIED (prefill + decode together)"
echo "  Data Parallel Size: 8"
echo "  Tensor Parallel Size: 1"
echo "  GPUs: 0-7 (all 8 GPUs)"
echo "  Port: 30000"
echo "  No Router - Direct vLLM serving"
echo "=========================================="
echo ""

# Start vLLM server with standard configuration
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_USE_PRECOMPILED=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8
