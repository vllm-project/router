#!/bin/bash

# Run benchmark against the vLLM service (tests K8s load balancing across both pods)
# This script runs the benchmark from inside a pod against the service
# Usage: ./run-benchmark.sh [num_prompts] [concurrency]

set -e

NAMESPACE="llm-d-deepseek-v31-native"
NUM_PROMPTS="${1:-100}"
MAX_CONCURRENCY="${2:-16}"
MODEL="deepseek-ai/DeepSeek-V3.1"
INPUT_LEN="${INPUT_LEN:-2000}"
OUTPUT_LEN="${OUTPUT_LEN:-2000}"
SERVICE_NAME="ms-deepseek-v31-native-llm-d-modelservice-decode"
SERVICE_PORT="8000"

echo "=========================================="
echo "Running Benchmark - DeepSeek V3.1 (K8s LB)"
echo "=========================================="
echo "Namespace: $NAMESPACE"
echo "Model: $MODEL"
echo "Num Prompts: $NUM_PROMPTS"
echo "Concurrency: $MAX_CONCURRENCY"
echo "Input Length: $INPUT_LEN"
echo "Output Length: $OUTPUT_LEN"
echo "Service: $SERVICE_NAME:$SERVICE_PORT"
echo ""

# Check if pods are running
POD_COUNT=$(kubectl get pod -n "$NAMESPACE" -l llm-d.ai/role=decode --no-headers 2>/dev/null | wc -l)
if [ "$POD_COUNT" -eq 0 ]; then
    echo "Error: No vLLM pods found. Deploy first with ./deploy.sh"
    exit 1
fi

echo "Found $POD_COUNT vLLM pods"
echo ""

# Check if service exists
if ! kubectl get svc -n "$NAMESPACE" "$SERVICE_NAME" &> /dev/null; then
    echo "Error: Service $SERVICE_NAME not found in namespace $NAMESPACE"
    exit 1
fi

echo "Service: $SERVICE_NAME"
echo ""

# Get a vLLM pod to run benchmark from (needs vllm bench command)
VLLM_POD=$(kubectl get pod -n "$NAMESPACE" -l llm-d.ai/role=decode -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -z "$VLLM_POD" ]; then
    echo "Error: Cannot find vLLM pod to run benchmark from"
    exit 1
fi

echo "Running benchmark from pod: $VLLM_POD"
echo ""

# Check if service is accessible from inside the cluster
echo "Checking service readiness..."
if kubectl exec -n "$NAMESPACE" "$VLLM_POD" -c vllm -- curl -s -f "http://$SERVICE_NAME:$SERVICE_PORT/v1/models" > /dev/null 2>&1; then
    echo "✓ Service is ready and accessible"
else
    echo "✗ Service not accessible"
    echo "Error: Cannot reach the service endpoint from inside the cluster"
    exit 1
fi

echo ""
echo "Starting benchmark..."
echo "This will test K8s load balancing across all $POD_COUNT pods"
echo "This may take several minutes..."
echo ""

# Build benchmark command - connect to service from inside the cluster
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
    --host $SERVICE_NAME \
    --port $SERVICE_PORT"

# Run benchmark from inside the vLLM pod
kubectl exec -n "$NAMESPACE" "$VLLM_POD" -c vllm -- bash -c "$BENCH_CMD"

echo ""
echo "=========================================="
echo "Benchmark completed!"
echo "=========================================="
echo "Requests were distributed across $POD_COUNT pods via K8s service"
echo ""
