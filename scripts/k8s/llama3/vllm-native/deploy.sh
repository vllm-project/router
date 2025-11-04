#!/bin/bash

# Deploy Llama 3.1 8B Multi-Node DP using helmfile
# Usage: ./deploy.sh [namespace] [gateway-provider]
#
# Examples:
#   ./deploy.sh                                    # Deploy to llm-d-llama31-multinode with default (istioBench)
#   ./deploy.sh llm-d-llama31-multinode istio      # Deploy to llm-d-llama31-multinode with istio
#   ./deploy.sh my-namespace kgateway              # Deploy to my-namespace with kgateway

set -e

NAMESPACE="${1:-llm-d-llama31-multinode}"
GATEWAY_PROVIDER="${2:-default}"

TOTAL_RANKS=16
WORKER_RANKS=$((TOTAL_RANKS - 1))

echo "==========================================="
echo "Deploying Llama 3.1 8B Multi-Node DP"
echo "==========================================="
echo "Namespace: $NAMESPACE"
echo "Gateway Provider: $GATEWAY_PROVIDER"
echo ""
echo "Architecture:"
echo "  - Rank 0 (Master): 1 GPU (DP coordinator + router)"
echo "  - Ranks 1-$WORKER_RANKS (Workers): 1 GPU each"
echo "  - Total GPUs: $TOTAL_RANKS"
echo "  - All requests → Rank 0 → DP Coordinator → Workers"
echo ""

# Check if namespace exists, create if not
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo "Creating namespace: $NAMESPACE"
    kubectl create namespace "$NAMESPACE"
fi

# Check if HF token secret exists
if ! kubectl get secret llm-d-hf-token -n "$NAMESPACE" &> /dev/null; then
    echo ""
    echo "ERROR: HuggingFace token secret 'llm-d-hf-token' not found in namespace $NAMESPACE"
    echo "Please create it with:"
    echo "  kubectl create secret generic llm-d-hf-token --from-literal=HF_TOKEN=your_token_here -n $NAMESPACE"
    echo ""
    exit 1
fi

# Deploy using helmfile
echo ""
echo "Deploying with helmfile..."
cd "$(dirname "$0")"

if [ "$GATEWAY_PROVIDER" = "default" ]; then
    helmfile apply -n "$NAMESPACE"
else
    helmfile apply -e "$GATEWAY_PROVIDER" -n "$NAMESPACE"
fi

echo ""
echo "==========================================="
echo "Creating Decode Service (for RPC)"
echo "==========================================="
echo ""
echo "Applying decode service to expose port 13345 for worker registration..."
kubectl apply -f decode-service.yaml

echo ""
echo "Verifying service was created..."
kubectl get svc -n "$NAMESPACE" ms-llama31-multinode-llm-d-modelservice-decode

echo ""
echo "==========================================="
echo "Deployment initiated!"
echo "==========================================="
echo ""
echo "Monitoring deployment progress..."
echo ""

# Wait for infrastructure gateway
echo "Waiting for infrastructure gateway..."
kubectl wait --for=condition=available --timeout=300s \
    deployment/infra-llama31-multinode-inference-gateway-istio -n "$NAMESPACE" 2>/dev/null || \
    echo "Gateway deployment not ready yet, continuing..."

echo ""
echo "Waiting for Rank 0 (Master) pod..."
kubectl wait --for=condition=ready --timeout=900s \
    pod -l app=ms-llama31-multinode-llm-d-modelservice-decode -n "$NAMESPACE" 2>/dev/null || \
    echo "⚠️  Rank 0 not ready yet. This may take 5-10 minutes for first-time model download."

echo ""
echo "Waiting for all worker pods ($WORKER_RANKS total)..."
kubectl wait --for=condition=ready --timeout=1200s \
    pod -l llm-d.ai/role=prefill -n "$NAMESPACE" 2>/dev/null || \
    echo "⚠️  Some worker pods are not ready yet. They may still be downloading the model."

echo ""
echo "==========================================="
echo "Deployment Status"
echo "==========================================="
kubectl get pods -n "$NAMESPACE" -o wide

echo ""
echo "==========================================="
echo "Next Steps"
echo "==========================================="
echo ""
echo "1. Check full logs:"
echo "   kubectl logs -n $NAMESPACE -l app=ms-llama31-multinode-llm-d-modelservice-decode -c vllm -f        # Rank 0"
echo "   kubectl logs -n $NAMESPACE -l llm-d.ai/role=prefill -c vllm -f                                    # All workers"
echo ""
echo "2. Verify DP coordination:"
echo "   kubectl logs -n $NAMESPACE -l app=ms-llama31-multinode-llm-d-modelservice-decode -c vllm | grep 'data-parallel'"
echo ""
echo "3. Test the deployment:"
echo "   kubectl port-forward -n $NAMESPACE svc/ms-llama31-multinode-llm-d-modelservice-decode 8000:8000"
echo "   curl http://localhost:8000/v1/models"
echo ""
echo "4. Get gateway external IP:"
echo "   kubectl get svc -n $NAMESPACE infra-llama31-multinode-inference-gateway-istio"
echo ""
echo "5. Run the benchmarking harness:"
echo "   ./run-benchmark.sh $NAMESPACE"
echo ""
