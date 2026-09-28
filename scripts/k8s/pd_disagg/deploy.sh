#!/bin/bash

# Deploy vLLM Router with PD Disaggregation to Kubernetes
# Usage: ./deploy.sh [pod|deployment]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="llm-d-pd"
MODE="${1:-pod}"  # Default to pod for testing

echo "==================================="
echo "Deploying vLLM Router (PD Mode)"
echo "==================================="
echo "Namespace: $NAMESPACE"
echo "Mode: $MODE"
echo ""

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo "Creating namespace $NAMESPACE..."
    kubectl create namespace "$NAMESPACE"
fi

case "$MODE" in
    pod)
        echo "Deploying router as Pod..."
        kubectl apply -f "$SCRIPT_DIR/router-pod.yaml"
        echo ""
        echo "Waiting for pod to be ready..."
        kubectl wait --for=condition=Ready pod/vllm-router-pd -n "$NAMESPACE" --timeout=120s || true
        echo ""
        echo "Pod status:"
        kubectl get pod vllm-router-pd -n "$NAMESPACE"
        echo ""
        echo "View logs with:"
        echo "  kubectl logs -n $NAMESPACE vllm-router-pd -f"
        ;;

    deployment)
        echo "Deploying router as Deployment..."
        kubectl apply -f "$SCRIPT_DIR/router-deployment.yaml"
        kubectl apply -f "$SCRIPT_DIR/router-service.yaml"
        echo ""
        echo "Waiting for deployment to be ready..."
        kubectl wait --for=condition=Available deployment/vllm-router-pd -n "$NAMESPACE" --timeout=120s || true
        echo ""
        echo "Deployment status:"
        kubectl get deployment vllm-router-pd -n "$NAMESPACE"
        kubectl get pods -n "$NAMESPACE" -l app=vllm-router
        kubectl get svc -n "$NAMESPACE" -l app=vllm-router
        echo ""
        echo "View logs with:"
        echo "  kubectl logs -n $NAMESPACE -l app=vllm-router -f"
        ;;

    *)
        echo "Error: Invalid mode. Use 'pod' or 'deployment'"
        exit 1
        ;;
esac

echo ""
echo "==================================="
echo "Router deployed successfully!"
echo "==================================="
echo ""
echo "Test the router:"
echo "  # Port forward to access from localhost"
echo "  kubectl port-forward -n $NAMESPACE pod/vllm-router-pd 10001:10001"
echo ""
echo "  # Send test request"
echo "  curl http://localhost:10001/health"
