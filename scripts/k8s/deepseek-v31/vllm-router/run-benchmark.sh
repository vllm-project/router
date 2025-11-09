#!/bin/bash

# Run benchmark for DeepSeek V3.1 with vllm-router (P-D disaggregation)
# This script runs the benchmark from inside a pod against the vllm-router
# Usage: ./run-benchmark.sh [num_prompts] [concurrency]
#
# Examples:
#   ./run-benchmark.sh           # 100 prompts, 16 concurrency
#   ./run-benchmark.sh 200 32    # 200 prompts, 32 concurrency

set -e

NAMESPACE="vllm-router-deepseek-v31"
NUM_PROMPTS="${1:-100}"
MAX_CONCURRENCY="${2:-16}"
MODEL="deepseek-ai/DeepSeek-V3.1"
INPUT_LEN="${INPUT_LEN:-1000}"
OUTPUT_LEN="${OUTPUT_LEN:-1000}"
ROUTER_SERVICE="vllm-router-deepseek-v31"
ROUTER_PORT="10001"

echo "=========================================="
echo "Running Benchmark - vllm-router (P-D)"
echo "=========================================="
echo "Namespace: $NAMESPACE"
echo "Model: $MODEL"
echo "Num Prompts: $NUM_PROMPTS"
echo "Concurrency: $MAX_CONCURRENCY"
echo "Input Length: $INPUT_LEN"
echo "Output Length: $OUTPUT_LEN"
echo "Router Service: $ROUTER_SERVICE:$ROUTER_PORT"
echo ""

# Get any vllm pod to run benchmark from (needs vllm bench command)
VLLM_POD=$(kubectl get pod -n "$NAMESPACE" -l llm-d.ai/role=prefill -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -z "$VLLM_POD" ]; then
    # Try decode pods if prefill not found
    VLLM_POD=$(kubectl get pod -n "$NAMESPACE" -l llm-d.ai/role=decode -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
fi
if [ -z "$VLLM_POD" ]; then
    echo "Error: Cannot find vllm pod to run benchmark from"
    exit 1
fi

echo "Running benchmark from pod: $VLLM_POD"
echo ""
echo "Starting benchmark..."
echo "This may take several minutes..."
echo ""

# Build benchmark command - connect to router service from inside the cluster
BENCH_CMD="vllm bench serve \
    --dataset-name random \
    --num-prompts $NUM_PROMPTS \
    --model $MODEL \
    --random-input-len $INPUT_LEN \
    --random-output-len $OUTPUT_LEN \
    --endpoint /v1/completions \
    --max-concurrency $MAX_CONCURRENCY \
    --save-result \
    --ignore-eos \
    --served-model-name $MODEL \
    --host $ROUTER_SERVICE \
    --port $ROUTER_PORT"

# Run benchmark from inside the vllm pod
kubectl exec -n "$NAMESPACE" "$VLLM_POD" -c vllm -- bash -c "$BENCH_CMD"

echo ""
echo "=========================================="
echo "Benchmark completed!"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Routing: vllm-router with consistent_hash policy"
echo "  Architecture: 1 prefill pod + 1 decode pod (8 GPUs each, TP=8)"
echo "  Concurrency: $MAX_CONCURRENCY"
echo "  Prompts: $NUM_PROMPTS"
echo ""
echo "Compare with llm-d by running:"
echo "  cd ../llm-d"
echo "  ./run-benchmark.sh $NUM_PROMPTS $MAX_CONCURRENCY"
echo ""
