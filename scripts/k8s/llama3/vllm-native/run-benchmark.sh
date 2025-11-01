#!/bin/bash

# Run benchmark from rank0 pod against itself
# Usage: ./run-benchmark.sh [num_prompts] [concurrency]

set -e

NAMESPACE="llm-d-llama31-multinode"
NUM_PROMPTS="${1:-100}"
MAX_CONCURRENCY="${2:-16}"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
INPUT_LEN="${INPUT_LEN:-2000}"
OUTPUT_LEN="${OUTPUT_LEN:-2000}"

echo "=========================================="
echo "Running Benchmark - Llama 3.1 Multi-Node"
echo "=========================================="
echo "Namespace: $NAMESPACE"
echo "Model: $MODEL"
echo "Num Prompts: $NUM_PROMPTS"
echo "Concurrency: $MAX_CONCURRENCY"
echo "Input Length: $INPUT_LEN"
echo "Output Length: $OUTPUT_LEN"
echo ""

# Check if rank0 pod is running
if ! kubectl get pod -n "$NAMESPACE" -l llm-d.ai/role=decode &> /dev/null; then
    echo "Error: Rank 0 (decode) pod not found. Deploy first with ./deploy.sh"
    exit 1
fi

# Get rank0 pod (decode pod)
RANK0_POD=$(kubectl get pod -n "$NAMESPACE" -l llm-d.ai/role=decode -o jsonpath='{.items[0].metadata.name}')

if [ -z "$RANK0_POD" ]; then
    echo "Error: No rank 0 pod found in namespace $NAMESPACE"
    exit 1
fi

echo "Using rank 0 pod (master): $RANK0_POD"
echo ""

# Check if vLLM is ready
echo "Checking vLLM readiness..."
if kubectl exec -n "$NAMESPACE" "$RANK0_POD" -c vllm -- curl -s -f "http://localhost:8000/v1/models" > /dev/null 2>&1; then
    echo "✓ vLLM is ready on rank 0"
else
    echo "✗ vLLM not ready on rank 0"
    echo "Error: vLLM is not accessible"
    exit 1
fi

echo ""
echo "Starting benchmark..."
echo "This may take several minutes..."
echo ""

# Run benchmark from inside the rank0 pod
kubectl exec -n "$NAMESPACE" "$RANK0_POD" -c vllm -- \
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
        --host "localhost" \
        --port "8000"

echo ""
echo "=========================================="
echo "Benchmark completed!"
echo "=========================================="
