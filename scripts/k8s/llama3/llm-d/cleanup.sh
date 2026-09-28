#!/bin/bash

# Cleanup Llama 3.1 8B P/D deployment
# This script removes all deployed resources

set -e

NAMESPACE="llm-d-llama31"

echo "=========================================="
echo "Cleaning up P/D Disaggregation Deployment"
echo "=========================================="
echo "Namespace: $NAMESPACE"
echo ""

# Remove helmfile releases
echo "Removing helmfile releases..."
helmfile destroy

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
