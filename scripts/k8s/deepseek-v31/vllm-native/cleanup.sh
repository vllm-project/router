#!/bin/bash

# Cleanup DeepSeek V3.1 deployment
# Usage: ./cleanup.sh [namespace] [gateway-provider]
#
# Examples:
#   ./cleanup.sh                                           # Cleanup from llm-d-deepseek-v31-native with default
#   ./cleanup.sh llm-d-deepseek-v31-native istio          # Cleanup from llm-d-deepseek-v31-native with istio
#   ./cleanup.sh my-namespace default                      # Cleanup from my-namespace with default

set -e

NAMESPACE="${1:-llm-d-deepseek-v31-native}"
GATEWAY_PROVIDER="${2:-default}"

echo "==========================================="
echo "Cleaning up DeepSeek V3.1 Deployment"
echo "==========================================="
echo "Namespace: $NAMESPACE"
echo "Gateway Provider: $GATEWAY_PROVIDER"
echo ""

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo "Namespace $NAMESPACE does not exist. Nothing to clean up."
    exit 0
fi

# Remove vLLM service
echo ""
echo "Removing vLLM service..."
cd "$(dirname "$0")"
kubectl delete -f vllm-service.yaml --ignore-not-found=true

# Use helmfile to destroy
echo ""
echo "Destroying helmfile releases..."
cd "$(dirname "$0")"

if [ "$GATEWAY_PROVIDER" = "default" ]; then
    helmfile destroy -n "$NAMESPACE"
else
    helmfile destroy -e "$GATEWAY_PROVIDER" -n "$NAMESPACE"
fi

# Additional cleanup for any stuck resources
echo ""
echo "Cleaning up any remaining resources..."

# Remove helm releases directly if helmfile didn't work
helm uninstall ms-deepseek-v31-native -n "$NAMESPACE" 2>/dev/null || echo "ms-deepseek-v31-native not found"
helm uninstall infra-deepseek-v31-native -n "$NAMESPACE" 2>/dev/null || echo "infra-deepseek-v31-native not found"

echo ""
echo "==========================================="
echo "Cleanup completed!"
echo "==========================================="
echo ""
echo "To delete the namespace entirely:"
echo "  kubectl delete namespace $NAMESPACE"
echo ""
