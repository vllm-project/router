#!/bin/bash

# Cleanup DeepSeek V3.1 vllm-router deployment
# This script removes all deployed resources

set -e

NAMESPACE="vllm-router-deepseek-v31"

echo "=========================================="
echo "Cleaning up vllm-router deployment"
echo "=========================================="
echo "Namespace: $NAMESPACE"
echo ""

# Delete router
echo "Deleting vllm-router..."
kubectl delete -f router-service.yaml --ignore-not-found=true
kubectl delete -f router-deployment.yaml --ignore-not-found=true

# Delete backend services
echo "Deleting backend services..."
kubectl delete -f backend-services.yaml --ignore-not-found=true

# Delete helmfile releases
echo "Deleting vLLM prefill and decode pods..."
cd "$(dirname "$0")"
helmfile destroy -n "$NAMESPACE" || echo "Helmfile destroy completed with warnings"

echo ""
echo "=========================================="
echo "Cleanup completed!"
echo "=========================================="
echo ""
echo "Note: Namespace $NAMESPACE still exists with:"
echo "  - Secret (llm-d-hf-token)"
echo "  - PVCs (cached model artifacts)"
echo ""
echo "To completely remove everything including cached model:"
echo "  kubectl delete namespace $NAMESPACE"
echo ""
