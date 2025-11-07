#!/bin/bash

# Cleanup vllm-router native load balancing deployment
# Usage: ./cleanup.sh

set -e

NAMESPACE="vllm-router-llama31"

echo "===================================="
echo "Cleaning up vllm-router Native Stack"
echo "===================================="
echo "Namespace: $NAMESPACE"
echo ""

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo "Namespace $NAMESPACE does not exist. Nothing to clean up."
    exit 0
fi

# Remove vllm-router deployment
echo "Removing vllm-router..."
kubectl delete -f router-deployment.yaml -n "$NAMESPACE" --ignore-not-found=true
kubectl delete -f router-service.yaml -n "$NAMESPACE" --ignore-not-found=true

# Remove backend services
echo "Removing backend services..."
kubectl delete -f backend-services.yaml -n "$NAMESPACE" --ignore-not-found=true

# Use helmfile to destroy vLLM backends
echo ""
echo "Destroying vLLM backends..."
cd "$(dirname "$0")"
helmfile destroy -n "$NAMESPACE"

# Additional cleanup for any stuck resources
echo ""
echo "Cleaning up any remaining resources..."

# Remove helm releases directly if helmfile didn't work
helm uninstall ms-llama31 -n "$NAMESPACE" 2>/dev/null || echo "ms-llama31 not found"

echo ""
echo "===================================="
echo "Cleanup completed!"
echo "===================================="
echo ""
echo "To delete the namespace entirely:"
echo "  kubectl delete namespace $NAMESPACE"
echo ""
