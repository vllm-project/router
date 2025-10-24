#!/bin/bash

# Baseline Benchmark Script for vLLM (without router)

# Activate virtual environment
source ~/uv_env/vllm/bin/activate

cd /data/users/nlalit/gitrepos/vllm

# Configuration
MODEL="meta-llama/Llama-3.1-8B-Instruct"
# For normal benchmarking
# PORT="30000"
# HOST="0.0.0.0"

# For vllm router benchmarking
HOST="127.0.0.1" 
PORT="8090"
ENDPOINT="/v1/completions"
RESULTS_DIR="/tmp/benchmark_results"

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "vLLM Baseline Benchmark Suite"
echo "=========================================="
echo "Model: $MODEL"
echo "PORT: $PORT"
echo "=========================================="
echo ""

# Note: Removed --ignore-eos and letting stream default to True
# This avoids the "stream_options requires stream=True" error
vllm bench serve \
    --dataset-name random \
    --random-input-len 2000 \
    --random-output-len 2000 \
    --num-prompts 1000 \
    --model "$MODEL" \
    --tokenizer "$MODEL" \
    --endpoint "$ENDPOINT" \
    --max-concurrency 32 \
    --save-result \
    --result-filename "$RESULTS_DIR/serving_test.json" \
    --port "$PORT" \
    --host "$HOST" \

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "Results are saved to: $RESULTS_DIR/serving_test.json"
echo "=========================================="


# vllm bench serve --dataset-name random --num-prompts 100 --model meta-llama/Llama-3.1-8B-Instruct --random-input-len 2000 --random-output-len 150 --endpoint /v1/completions --max-concurrency 32 --save-result --ignore-eos --port 8090 --host 127.0.0.1