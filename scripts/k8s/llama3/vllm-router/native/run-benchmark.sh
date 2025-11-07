#!/bin/bash

# Run benchmark for Llama 3.1 with vllm-router (native load balancing)
# This script runs the benchmark from inside a backend pod against the vllm-router
# Usage: ./run-benchmark.sh [num_prompts] [concurrency]
#
# Examples:
#   ./run-benchmark.sh           # 100 prompts, 16 concurrency
#   ./run-benchmark.sh 200 32    # 200 prompts, 32 concurrency

set -e

NAMESPACE="vllm-router-llama31"
NUM_PROMPTS="${1:-100}"
MAX_CONCURRENCY="${2:-16}"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
INPUT_LEN="${INPUT_LEN:-1000}"
OUTPUT_LEN="${OUTPUT_LEN:-1000}"
ROUTER_SERVICE="vllm-router-llama31"
ROUTER_PORT="10001"

echo "=========================================="
echo "Running Benchmark - vllm-router (native)"
echo "=========================================="
echo "Namespace: $NAMESPACE"
echo "Model: $MODEL"
echo "Num Prompts: $NUM_PROMPTS"
echo "Concurrency: $MAX_CONCURRENCY"
echo "Input Length: $INPUT_LEN"
echo "Output Length: $OUTPUT_LEN"
echo "Router Service: $ROUTER_SERVICE:$ROUTER_PORT"
echo ""

# Get any backend pod to run benchmark from (needs vllm bench command)
BACKEND_POD=$(kubectl get pod -n "$NAMESPACE" -l llm-d.ai/role=decode -o jsonpath='{.items[0].metadata.name}')
if [ -z "$BACKEND_POD" ]; then
    echo "Error: Cannot find backend pod to run benchmark from"
    exit 1
fi

echo "Running benchmark from pod: $BACKEND_POD"
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

# Run benchmark from inside the backend pod
kubectl exec -n "$NAMESPACE" "$BACKEND_POD" -c vllm -- bash -c "$BENCH_CMD"

echo ""
echo "=========================================="
echo "Benchmark completed!"
echo "=========================================="
echo ""
echo "Note: Benchmark ran through vllm-router"
echo "      Requests load balanced across 2 vLLM backends"
