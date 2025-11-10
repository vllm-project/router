#!/bin/bash

# Deploy DeepSeek V3.1 with Kubernetes Native Load Balancing
# Usage: ./deploy.sh [namespace] [gateway-provider]
#
# Examples:
#   ./deploy.sh                                           # Deploy to llm-d-deepseek-v31-native with default (istioBench)
#   ./deploy.sh llm-d-deepseek-v31-native istio          # Deploy to llm-d-deepseek-v31-native with istio
#   ./deploy.sh my-namespace default                      # Deploy to my-namespace with default

set -e

NAMESPACE="${1:-llm-d-deepseek-v31-native}"
GATEWAY_PROVIDER="${2:-default}"

echo "==========================================="
echo "Deploying DeepSeek V3.1 (K8s Native LB)"
echo "==========================================="
echo "Namespace: $NAMESPACE"
echo "Gateway Provider: $GATEWAY_PROVIDER"
echo ""
echo "Architecture:"
echo "  - 2 independent vLLM pods (8 GPUs each, TP=8)"
echo "  - No DP coordinator - pure K8s load balancing"
echo "  - Total: 2 pods, 16 GPUs"
echo "  - Kubernetes service distributes requests across both pods"
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
kubectl get svc -n "$NAMESPACE" ms-deepseek-v31-native-llm-d-modelservice-decode

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
    deployment/infra-deepseek-v31-native-inference-gateway-istio -n "$NAMESPACE" 2>/dev/null || \
    echo "Gateway deployment not ready yet, continuing..."

echo ""
echo "Waiting for vLLM pods..."
kubectl wait --for=condition=ready --timeout=1800s \
    pod -l llm-d.ai/role=decode -n "$NAMESPACE" 2>/dev/null || \
    echo "⚠️  Pods not ready yet. This may take 15-20 minutes for first-time model download (~700GB)."

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
echo "   kubectl port-forward -n $NAMESPACE svc/ms-deepseek-v31-native-llm-d-modelservice-decode 8000:8000"
echo "   curl http://localhost:8000/v1/models"
echo ""
echo "3. Send a test request:"
echo "   curl -X POST http://localhost:8000/v1/completions \\"
echo "     -H \"Content-Type: application/json\" \\"
echo "     -d '{\"model\": \"deepseek-ai/DeepSeek-V3.1\", \"prompt\": \"Hello\", \"max_tokens\": 50}'"
echo ""
echo "4. Get gateway external IP:"
echo "   kubectl get svc -n $NAMESPACE infra-deepseek-v31-native-inference-gateway-istio"
echo ""
echo "5. Check all pods:"
echo "   kubectl get pods -n $NAMESPACE -l llm-d.ai/role=decode"
echo ""
echo "6. Run benchmarks:"
echo "   ./run-benchmark.sh"
echo ""
