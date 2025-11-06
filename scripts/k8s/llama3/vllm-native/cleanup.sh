#!/bin/bash

# Cleanup Llama 3.1 8B Multi-Node DP deployment
# Usage: ./cleanup.sh [namespace] [gateway-provider]
#
# Examples:
#   ./cleanup.sh                                    # Cleanup from llm-d-llama31-multinode with default
#   ./cleanup.sh llm-d-llama31-multinode istio      # Cleanup from llm-d-llama31-multinode with istio
#   ./cleanup.sh my-namespace kgateway              # Cleanup from my-namespace with kgateway

set -e

NAMESPACE="${1:-llm-d-llama31-multinode}"
GATEWAY_PROVIDER="${2:-default}"

echo "==========================================="
echo "Cleaning up Llama 3.1 8B Multi-Node DP"
echo "==========================================="
echo "Namespace: $NAMESPACE"
echo "Gateway Provider: $GATEWAY_PROVIDER"
echo ""

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo "Namespace $NAMESPACE does not exist. Nothing to clean up."
    exit 0
fi

# Remove worker StatefulSet first
echo ""
echo "Removing worker StatefulSet..."
cd "$(dirname "$0")"
kubectl delete -f workers-statefulset.yaml --ignore-not-found=true

# Remove decode service
echo ""
echo "Removing decode service..."
kubectl delete -f decode-service.yaml --ignore-not-found=true

# Use helmfile to destroy
echo ""
echo "Destroying helmfile releases..."

if [ "$GATEWAY_PROVIDER" = "default" ]; then
    helmfile destroy -n "$NAMESPACE"
else
    helmfile destroy -e "$GATEWAY_PROVIDER" -n "$NAMESPACE"
fi

# Additional cleanup for any stuck resources
echo ""
echo "Cleaning up any remaining resources..."

# Remove helm releases directly if helmfile didn't work
helm uninstall ms-llama31-multinode -n "$NAMESPACE" 2>/dev/null || echo "ms-llama31-multinode not found"
helm uninstall infra-llama31-multinode -n "$NAMESPACE" 2>/dev/null || echo "infra-llama31-multinode not found"

echo ""
echo "==========================================="
echo "Cleanup completed!"
echo "==========================================="
echo ""
echo "To delete the namespace entirely (including PVC with cached model):"
echo "  kubectl delete namespace $NAMESPACE"
echo ""
echo "Note: Keeping the namespace preserves the PVC, so next deployment"
echo "      won't need to re-download the model!"
echo ""
