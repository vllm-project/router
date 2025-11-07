#!/bin/bash

# Run benchmark from prefill pod against the router
# Usage: ./run-benchmark.sh [num_prompts] [concurrency]

set -e

NAMESPACE="llm-d-pd"
PREFILL_POD="${PREFILL_POD:-ms-pd-llm-d-modelservice-prefill-95875d469-nwtsk}"
ROUTER_SERVICE="vllm-router-pd"
NUM_PROMPTS="${1:-1000}"
MAX_CONCURRENCY="${2:-32}"
MODEL="deepseek-ai/DeepSeek-V3"
INPUT_LEN=2000
OUTPUT_LEN=2000

echo "==================================="
echo "Running Benchmark"
echo "==================================="
echo "Namespace: $NAMESPACE"
echo "Prefill Pod: $PREFILL_POD"
echo "Router Service: $ROUTER_SERVICE"
echo "Num Prompts: $NUM_PROMPTS"
echo "Concurrency: $MAX_CONCURRENCY"
echo ""

# Check if router is running
if ! kubectl get pod -n "$NAMESPACE" -l app=vllm-router &> /dev/null; then
    echo "Error: Router pod not found. Deploy it first with ./deploy.sh"
    exit 1
fi

# Get router pod IP
ROUTER_POD=$(kubectl get pod -n "$NAMESPACE" -l app=vllm-router -o jsonpath='{.items[0].metadata.name}')
ROUTER_IP=$(kubectl get pod -n "$NAMESPACE" "$ROUTER_POD" -o jsonpath='{.status.podIP}')

echo "Router Pod: $ROUTER_POD"
echo "Router IP: $ROUTER_IP"
echo ""

# Check router health from prefill pod
echo "Checking router connectivity..."
if kubectl exec -n "$NAMESPACE" "$PREFILL_POD" -- curl -s -f "http://${ROUTER_IP}:10001/health" > /dev/null 2>&1; then
    echo "✓ Router is accessible from prefill pod"
else
    echo "✗ Cannot reach router from prefill pod"
    echo "Trying with service name..."
    if kubectl exec -n "$NAMESPACE" "$PREFILL_POD" -- curl -s -f "http://${ROUTER_SERVICE}:10001/health" > /dev/null 2>&1; then
        echo "✓ Router is accessible via service"
        ROUTER_IP="$ROUTER_SERVICE"
    else
        echo "Error: Router is not accessible"
        exit 1
    fi
fi

echo ""
echo "Starting benchmark..."
echo "This may take several minutes..."
echo ""

# Run benchmark
kubectl exec -n "$NAMESPACE" "$PREFILL_POD" -- \
    vllm bench serve \
        --dataset-name random \
        --num-prompts "$NUM_PROMPTS" \
        --model "$MODEL" \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --endpoint /v1/completions \
        --max-concurrency "$MAX_CONCURRENCY" \
        --save-result \
        --ignore-eos \
        --served-model-name "$MODEL" \
        --host "$ROUTER_IP" \
        --port 10001

echo ""
echo "==================================="
echo "Benchmark completed!"
echo "==================================="
