#!/bin/bash

# Quick test script for the router
# Usage: ./test-router.sh

set -e

NAMESPACE="llm-d-pd"

echo "==================================="
echo "Testing vLLM Router"
echo "==================================="

# Get router pod
ROUTER_POD=$(kubectl get pod -n "$NAMESPACE" -l app=vllm-router -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$ROUTER_POD" ]; then
    echo "Error: Router pod not found"
    echo "Deploy it first with: ./deploy.sh pod"
    exit 1
fi

echo "Router Pod: $ROUTER_POD"

# Get router IP
ROUTER_IP=$(kubectl get pod -n "$NAMESPACE" "$ROUTER_POD" -o jsonpath='{.status.podIP}')
echo "Router IP: $ROUTER_IP"
echo ""

# Test health endpoint
echo "Testing health endpoint..."
HEALTH_RESPONSE=$(kubectl exec -n "$NAMESPACE" "$ROUTER_POD" -- curl -s http://localhost:10001/health)
echo "Health response: $HEALTH_RESPONSE"
echo ""

# Get prefill pod for testing from
PREFILL_POD=$(kubectl get pod -n "$NAMESPACE" -l app=ms-pd-llm-d-modelservice-prefill -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$PREFILL_POD" ]; then
    echo "Warning: Prefill pod not found, skipping connectivity test from prefill pod"
else
    echo "Testing connectivity from prefill pod..."
    if kubectl exec -n "$NAMESPACE" "$PREFILL_POD" -- curl -s -f "http://${ROUTER_IP}:10001/health" > /dev/null 2>&1; then
        echo "✓ Router is accessible from prefill pod"
    else
        echo "✗ Cannot reach router from prefill pod"
    fi
    echo ""
fi

# Send a simple completion request
echo "Sending test completion request..."
COMPLETION_RESPONSE=$(kubectl exec -n "$NAMESPACE" "$ROUTER_POD" -- curl -s http://localhost:10001/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-ai/DeepSeek-V3",
        "prompt": "Hello, how are you?",
        "max_tokens": 50,
        "temperature": 0.7
    }')

if echo "$COMPLETION_RESPONSE" | jq . > /dev/null 2>&1; then
    echo "✓ Completion request successful"
    echo "$COMPLETION_RESPONSE" | jq -r '.choices[0].text // .error // "Response received"' | head -c 200
    echo ""
else
    echo "Response (first 200 chars): ${COMPLETION_RESPONSE:0:200}"
fi

echo ""
echo "==================================="
echo "Test completed!"
echo "==================================="
echo ""
echo "View router logs:"
echo "  kubectl logs -n $NAMESPACE $ROUTER_POD -f"
