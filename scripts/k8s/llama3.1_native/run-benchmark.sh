#!/bin/bash

# Run benchmark from decode pod against the gateway
# Usage: ./run-benchmark.sh [num_prompts] [concurrency]

set -e

NAMESPACE="llm-d-llama31-native"
GATEWAY_SERVICE="infra-llama31-inference-gateway-istio"
GATEWAY_PORT="80"
NUM_PROMPTS="${1:-100}"
MAX_CONCURRENCY="${2:-16}"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
INPUT_LEN="${INPUT_LEN:-2000}"
OUTPUT_LEN="${OUTPUT_LEN:-2000}"

echo "==================================="
echo "Running Benchmark - Llama 3.1 Native"
echo "==================================="
echo "Namespace: $NAMESPACE"
echo "Gateway Service: $GATEWAY_SERVICE:$GATEWAY_PORT"
echo "Model: $MODEL"
echo "Num Prompts: $NUM_PROMPTS"
echo "Concurrency: $MAX_CONCURRENCY"
echo "Input Length: $INPUT_LEN"
echo "Output Length: $OUTPUT_LEN"
echo ""

# Check if decode pods are running
if ! kubectl get pod -n "$NAMESPACE" -l llm-d.ai/inferenceServing=true &> /dev/null; then
    echo "Error: Decode pods not found. Deploy first with ./deploy.sh"
    exit 1
fi

# Get first decode pod
DECODE_POD=$(kubectl get pod -n "$NAMESPACE" -l llm-d.ai/inferenceServing=true -o jsonpath='{.items[0].metadata.name}')

if [ -z "$DECODE_POD" ]; then
    echo "Error: No decode pods found in namespace $NAMESPACE"
    exit 1
fi

echo "Using decode pod: $DECODE_POD"
echo ""

# Check gateway health
echo "Checking gateway connectivity..."
if kubectl exec -n "$NAMESPACE" "$DECODE_POD" -c vllm -- curl -s -f "http://${GATEWAY_SERVICE}:${GATEWAY_PORT}/v1/models" > /dev/null 2>&1; then
    echo "✓ Gateway is accessible from decode pod"
else
    echo "✗ Cannot reach gateway from decode pod"
    echo "Error: Gateway is not accessible"
    exit 1
fi

echo ""
echo "Starting benchmark..."
echo "This may take several minutes..."
echo ""

# Run benchmark from inside the decode pod
kubectl exec -n "$NAMESPACE" "$DECODE_POD" -c vllm -- \
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
        --host "$GATEWAY_SERVICE" \
        --port "$GATEWAY_PORT"

echo ""
echo "==================================="
echo "Benchmark completed!"
echo "==================================="
