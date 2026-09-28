#!/bin/bash

# Run benchmark for DeepSeek V3 with llm-d (GAIE P-D disaggregation)
# This script waits for deployment, verifies setup, and runs benchmarks
# Usage: ./run-benchmark.sh [num_prompts] [concurrency]
#
# Examples:
#   ./run-benchmark.sh           # 200 prompts, 32 concurrency
#   ./run-benchmark.sh 200 64    # 200 prompts, 64 concurrency
#   ./run-benchmark.sh 200 128   # 200 prompts, 128 concurrency

set -e

NAMESPACE="llm-d-deepseek-v3"
NUM_PROMPTS="${1:-200}"
MAX_CONCURRENCY="${2:-32}"
MODEL="deepseek-ai/DeepSeek-V3.1"
INPUT_LEN="${INPUT_LEN:-1000}"
OUTPUT_LEN="${OUTPUT_LEN:-1000}"
GATEWAY_SERVICE="infra-deepseek-v3-inference-gateway-istio"
GATEWAY_PORT="80"

echo "=========================================="
echo "Benchmark Setup - llm-d (GAIE)"
echo "=========================================="
echo "Namespace: $NAMESPACE"
echo "Model: $MODEL"
echo "Num Prompts: $NUM_PROMPTS"
echo "Concurrency: $MAX_CONCURRENCY"
echo "Input Length: $INPUT_LEN"
echo "Output Length: $OUTPUT_LEN"
echo "Gateway: $GATEWAY_SERVICE:$GATEWAY_PORT"
echo ""

# Step 1: Wait for all pods to be ready
echo "Step 1/5: Waiting for all pods to be ready..."
echo "This may take 10-15 minutes on first deployment (large model download ~700GB)"
echo ""

EXPECTED_PREFILL=1
EXPECTED_DECODE=1

while true; do
    PREFILL_READY=$(kubectl get pods -n "$NAMESPACE" -l llm-d.ai/role=prefill -o json 2>/dev/null | jq -r '[.items[] | select(.status.phase=="Running" and (.status.containerStatuses[]? | select(.ready==true)))] | length' || echo "0")
    DECODE_READY=$(kubectl get pods -n "$NAMESPACE" -l llm-d.ai/role=decode -o json 2>/dev/null | jq -r '[.items[] | select(.status.phase=="Running" and (.status.containerStatuses[]? | select(.name=="vllm" and .ready==true)))] | length' || echo "0")
    GAIE_READY=$(kubectl get pods -n "$NAMESPACE" -l inferencepool=gaie-deepseek-v3-epp -o json 2>/dev/null | jq -r '[.items[] | select(.status.phase=="Running" and (.status.containerStatuses[]? | select(.ready==true)))] | length' || echo "0")

    echo "  Prefill pods: $PREFILL_READY/$EXPECTED_PREFILL ready (TP=8)"
    echo "  Decode pods: $DECODE_READY/$EXPECTED_DECODE ready (TP=8)"
    echo "  GAIE EPP: $GAIE_READY/1 ready"

    if [ "$PREFILL_READY" -eq "$EXPECTED_PREFILL" ] && [ "$DECODE_READY" -eq "$EXPECTED_DECODE" ] && [ "$GAIE_READY" -eq "1" ]; then
        echo ""
        echo "✓ All pods are ready!"
        break
    fi

    echo "  Waiting... (checking again in 10s)"
    echo ""
    sleep 10
done

# Step 2: Verify GAIE components
echo ""
echo "Step 2/5: Verifying GAIE components..."

# Check InferencePool (use explicit API version)
if ! kubectl get inferencepool.inference.networking.x-k8s.io gaie-deepseek-v3 -n "$NAMESPACE" &>/dev/null; then
    echo "❌ Error: InferencePool 'gaie-deepseek-v3' not found"
    exit 1
fi
echo "  ✓ InferencePool exists"

# Check HTTPRoute
if ! kubectl get httproute -n "$NAMESPACE" &>/dev/null 2>&1; then
    echo "  ⚠ Warning: HTTPRoute not found (may be optional)"
else
    echo "  ✓ HTTPRoute exists"
fi

# Check Gateway
if ! kubectl get gateway -n "$NAMESPACE" &>/dev/null 2>&1; then
    echo "  ⚠ Warning: Gateway not found (may be optional)"
else
    echo "  ✓ Gateway exists"
fi

# Check Gateway service
if ! kubectl get svc "$GATEWAY_SERVICE" -n "$NAMESPACE" &>/dev/null; then
    echo "❌ Error: Gateway service '$GATEWAY_SERVICE' not found"
    exit 1
fi
echo "  ✓ Gateway service exists"

# Step 3: Get a pod to run benchmark from
echo ""
echo "Step 3/5: Finding pod to run benchmark from..."
DECODE_POD=$(kubectl get pod -n "$NAMESPACE" -l llm-d.ai/role=decode -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -z "$DECODE_POD" ]; then
    echo "❌ Error: Cannot find decode pod"
    exit 1
fi
echo "  ✓ Using pod: $DECODE_POD"

# Step 4: Test inference endpoint
echo ""
echo "Step 4/5: Testing inference endpoint..."
TEST_RESULT=$(kubectl exec -n "$NAMESPACE" "$DECODE_POD" -c vllm -- \
    curl -s -f -X POST "http://$GATEWAY_SERVICE/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"Hello\", \"max_tokens\": 5}" 2>/dev/null || echo "FAILED")

if [[ "$TEST_RESULT" == "FAILED" ]] || [[ ! "$TEST_RESULT" =~ "choices" ]]; then
    echo "❌ Error: Inference test failed"
    echo "Gateway may not be routing correctly"
    echo ""
    echo "Troubleshooting steps:"
    echo "  kubectl logs -n $NAMESPACE -l inferencepool=gaie-deepseek-v3-epp --tail=50"
    echo "  kubectl describe inferencepool gaie-deepseek-v3 -n $NAMESPACE"
    exit 1
fi
echo "  ✓ Inference endpoint working!"

# Step 5: Run benchmark
echo ""
echo "=========================================="
echo "Step 5/5: Running Benchmark"
echo "=========================================="
echo ""
echo "This may take several minutes depending on concurrency..."
echo ""

# Build benchmark command
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
    --host $GATEWAY_SERVICE \
    --port $GATEWAY_PORT"

# Run benchmark from inside the decode pod
kubectl exec -n "$NAMESPACE" "$DECODE_POD" -c vllm -- bash -c "$BENCH_CMD"

echo ""
echo "=========================================="
echo "Benchmark Completed!"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Routing: GAIE EPP (queue scoring + prefix matching)"
echo "  Architecture: 1 prefill pod + 1 decode pod (8 GPUs each, TP=8)"
echo "  Concurrency: $MAX_CONCURRENCY"
echo "  Prompts: $NUM_PROMPTS"
echo ""
echo "Compare with vllm-router by running:"
echo "  cd ../vllm-router/pd-disagg"
echo "  ./run-benchmark.sh $NUM_PROMPTS $MAX_CONCURRENCY"
echo ""
