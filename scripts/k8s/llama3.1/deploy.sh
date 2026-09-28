#!/bin/bash

# Deploy Llama 3.1 8B with P/D disaggregation using helmfile
# Usage: ./deploy.sh [namespace] [gateway-provider]
#
# Examples:
#   ./deploy.sh                          # Deploy to llm-d-llama31 with default (istioBench)
#   ./deploy.sh llm-d-llama31 istio      # Deploy to llm-d-llama31 with istio
#   ./deploy.sh my-namespace kgateway    # Deploy to my-namespace with kgateway

set -e

NAMESPACE="${1:-llm-d-llama31}"
GATEWAY_PROVIDER="${2:-default}"

echo "===================================="
echo "Deploying Llama 3.1 8B P/D Stack"
echo "===================================="
echo "Namespace: $NAMESPACE"
echo "Gateway Provider: $GATEWAY_PROVIDER"
echo ""

# Check if namespace exists, create if not
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo "Creating namespace: $NAMESPACE"
    kubectl create namespace "$NAMESPACE"
fi

# Check if HF token secret exists
if ! kubectl get secret llm-d-hf-token -n "$NAMESPACE" &> /dev/null; then
    echo ""
    echo "WARNING: HuggingFace token secret 'llm-d-hf-token' not found in namespace $NAMESPACE"
    echo "Please create it with:"
    echo "  kubectl create secret generic llm-d-hf-token --from-literal=HF_TOKEN=your_token_here -n $NAMESPACE"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
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
echo "===================================="
echo "Deployment initiated!"
echo "===================================="
echo ""
echo "Waiting for pods to be ready..."
echo ""

# Wait for infrastructure gateway
echo "Waiting for infrastructure gateway..."
kubectl wait --for=condition=available --timeout=300s \
    deployment/infra-llama31-inference-gateway-istio -n "$NAMESPACE" 2>/dev/null || \
    echo "Gateway deployment not ready yet, continuing..."

# Wait for GAIE EPP
echo "Waiting for GAIE EPP..."
kubectl wait --for=condition=available --timeout=300s \
    deployment/gaie-llama31-epp -n "$NAMESPACE" 2>/dev/null || \
    echo "GAIE EPP not ready yet, continuing..."

# Wait for prefill pods
echo "Waiting for prefill pods..."
kubectl wait --for=condition=ready --timeout=600s \
    pod -l app=ms-llama31-llm-d-modelservice-prefill -n "$NAMESPACE" 2>/dev/null || \
    echo "Prefill pods not ready yet, continuing..."

# Wait for decode pods
echo "Waiting for decode pods..."
kubectl wait --for=condition=ready --timeout=600s \
    pod -l app=ms-llama31-llm-d-modelservice-decode -n "$NAMESPACE" 2>/dev/null || \
    echo "Decode pods not ready yet, continuing..."

echo ""
echo "===================================="
echo "Deployment Status"
echo "===================================="
kubectl get pods -n "$NAMESPACE"

echo ""
echo "===================================="
echo "Next Steps"
echo "===================================="
echo ""
echo "1. Install HTTPRoute:"
if [ "$GATEWAY_PROVIDER" = "gke" ]; then
    echo "   kubectl apply -f httproute.gke.yaml -n $NAMESPACE"
else
    echo "   kubectl apply -f httproute.yaml -n $NAMESPACE"
fi
echo ""
echo "2. Check deployment status:"
echo "   kubectl get all -n $NAMESPACE"
echo ""
echo "3. View logs:"
echo "   kubectl logs -n $NAMESPACE -l app=ms-llama31-llm-d-modelservice-prefill -f"
echo "   kubectl logs -n $NAMESPACE -l app=ms-llama31-llm-d-modelservice-decode -f"
echo ""
echo "4. Get gateway external IP:"
echo "   kubectl get svc -n $NAMESPACE infra-llama31-inference-gateway-istio"
echo ""
