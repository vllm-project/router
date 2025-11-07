#!/bin/bash

# Deploy vLLM with vllm-router (P-D disaggregation)
# Usage: ./deploy.sh

set -e

NAMESPACE="vllm-router-pd-llama31"

echo "=========================================="
echo "Deploying vLLM with vllm-router (P-D)"
echo "=========================================="
echo "Namespace: $NAMESPACE"
echo ""

# Check if namespace exists, create if not
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo "Creating namespace: $NAMESPACE"
    kubectl create namespace "$NAMESPACE"
fi

# Check if HF token secret exists
if ! kubectl get secret llm-d-hf-token -n "$NAMESPACE" &> /dev/null; then
    echo ""
    echo "HuggingFace token secret 'llm-d-hf-token' not found in namespace $NAMESPACE"

    # Check if HF_TOKEN environment variable is set
    if [ -z "$HF_TOKEN" ]; then
        echo "ERROR: HF_TOKEN environment variable is not set"
        echo ""
        echo "Please set your HuggingFace token as an environment variable:"
        echo "  export HF_TOKEN=hf_your_token_here"
        echo ""
        echo "Example:"
        echo "  export HF_TOKEN=your_actual_token"
        echo ""
        exit 1
    fi

    echo "Creating HF token secret from environment variable..."
    kubectl create secret generic llm-d-hf-token \
        --from-literal=HF_TOKEN="$HF_TOKEN" \
        -n "$NAMESPACE"
    echo "✓ HF token secret created successfully"
fi

# Deploy using helmfile
echo ""
echo "Deploying vLLM prefill and decode with helmfile..."
cd "$(dirname "$0")"
helmfile apply -n "$NAMESPACE"

# Deploy backend services
echo ""
echo "Deploying backend services..."
kubectl apply -f backend-services.yaml

echo ""
echo "Waiting for vLLM prefill and decode to be ready..."
kubectl wait --for=condition=ready --timeout=600s \
    pod -l llm-d.ai/role=prefill -n "$NAMESPACE" 2>/dev/null || \
    echo "Prefill pod not ready yet, continuing..."

kubectl wait --for=condition=ready --timeout=600s \
    pod -l llm-d.ai/role=decode -n "$NAMESPACE" 2>/dev/null || \
    echo "Decode pod not ready yet, continuing..."

# Deploy vllm-router
echo ""
echo "Deploying vllm-router..."
kubectl apply -f router-deployment.yaml
kubectl apply -f router-service.yaml

echo ""
echo "=========================================="
echo "Deployment Status"
echo "=========================================="
kubectl get pods -n "$NAMESPACE"

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Check deployment status:"
echo "   kubectl get all -n $NAMESPACE"
echo ""
echo "2. View logs:"
echo "   kubectl logs -n $NAMESPACE -l llm-d.ai/role=prefill -f"
echo "   kubectl logs -n $NAMESPACE -l llm-d.ai/role=decode -f"
echo "   kubectl logs -n $NAMESPACE -l app=vllm-router -f"
echo ""
echo "3. Port forward to router:"
echo "   kubectl port-forward -n $NAMESPACE svc/vllm-router-llama31 10001:10001"
echo ""
echo "4. Run benchmarks:"
echo "   ./run-benchmark.sh"
echo ""
