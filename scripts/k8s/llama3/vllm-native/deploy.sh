#!/bin/bash

# Deploy Llama 3.1 8B with Kubernetes Native Load Balancing
# Usage: ./deploy.sh [namespace] [gateway-provider]
#
# Examples:
#   ./deploy.sh                                    # Deploy to llm-d-llama31-multinode with default (istioBench)
#   ./deploy.sh llm-d-llama31-multinode istio      # Deploy to llm-d-llama31-multinode with istio
#   ./deploy.sh my-namespace kgateway              # Deploy to my-namespace with kgateway

set -e

NAMESPACE="${1:-llm-d-llama31-multinode}"
GATEWAY_PROVIDER="${2:-default}"

echo "==========================================="
echo "Deploying Llama 3.1 8B (K8s Native LB)"
echo "==========================================="
echo "Namespace: $NAMESPACE"
echo "Gateway Provider: $GATEWAY_PROVIDER"
echo ""
echo "Architecture:"
echo "  - 16 independent vLLM pods (1 GPU each)"
echo "  - No DP coordinator - pure K8s load balancing"
echo "  - Total: 16 pods, 16 GPUs"
echo "  - Kubernetes service distributes requests across all pods"
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
echo "Creating vLLM service for load balancing..."
kubectl apply -f vllm-service.yaml

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
echo "Waiting for vLLM pods..."
kubectl wait --for=condition=ready --timeout=900s \
    pod -l llm-d.ai/role=decode -n "$NAMESPACE" 2>/dev/null || \
    echo "⚠️  Pods not ready yet. This may take 10-15 minutes for first-time model download."

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
echo "1. Check pod logs:"
echo "   kubectl logs -n $NAMESPACE -l llm-d.ai/role=decode -c vllm -f"
echo ""
echo "2. Test the deployment:"
echo "   kubectl port-forward -n $NAMESPACE svc/ms-llama31-multinode-llm-d-modelservice-decode 8000:8000"
echo "   curl http://localhost:8000/v1/models"
echo ""
echo "3. Send a test request:"
echo "   curl -X POST http://localhost:8000/v1/completions \\"
echo "     -H \"Content-Type: application/json\" \\"
echo "     -d '{\"model\": \"meta-llama/Llama-3.1-8B-Instruct\", \"prompt\": \"Hello\", \"max_tokens\": 50}'"
echo ""
echo "4. Get gateway external IP:"
echo "   kubectl get svc -n $NAMESPACE infra-llama31-multinode-inference-gateway-istio"
echo ""
echo "5. Check all pods:"
echo "   kubectl get pods -n $NAMESPACE -l llm-d.ai/role=decode"
echo ""
echo "6. Run benchmarks:"
echo "   ./run-benchmark.sh"
echo ""
