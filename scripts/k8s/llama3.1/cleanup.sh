#!/bin/bash

# Cleanup Llama 3.1 8B P/D deployment
# Usage: ./cleanup.sh [namespace] [gateway-provider]
#
# Examples:
#   ./cleanup.sh                          # Cleanup from llm-d-llama31 with default
#   ./cleanup.sh llm-d-llama31 istio      # Cleanup from llm-d-llama31 with istio
#   ./cleanup.sh my-namespace kgateway    # Cleanup from my-namespace with kgateway

set -e

NAMESPACE="${1:-llm-d-llama31}"
GATEWAY_PROVIDER="${2:-default}"

echo "===================================="
echo "Cleaning up Llama 3.1 8B P/D Stack"
echo "===================================="
echo "Namespace: $NAMESPACE"
echo "Gateway Provider: $GATEWAY_PROVIDER"
echo ""

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo "Namespace $NAMESPACE does not exist. Nothing to clean up."
    exit 0
fi

# Remove HTTPRoute
echo "Removing HTTPRoute..."
if [ "$GATEWAY_PROVIDER" = "gke" ]; then
    kubectl delete -f httproute.gke.yaml -n "$NAMESPACE" --ignore-not-found=true
else
    kubectl delete -f httproute.yaml -n "$NAMESPACE" --ignore-not-found=true
fi

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
helm uninstall ms-llama31 -n "$NAMESPACE" 2>/dev/null || echo "ms-llama31 not found"
helm uninstall gaie-llama31 -n "$NAMESPACE" 2>/dev/null || echo "gaie-llama31 not found"
helm uninstall infra-llama31 -n "$NAMESPACE" 2>/dev/null || echo "infra-llama31 not found"

echo ""
echo "===================================="
echo "Cleanup completed!"
echo "===================================="
echo ""
echo "To delete the namespace entirely:"
echo "  kubectl delete namespace $NAMESPACE"
echo ""
