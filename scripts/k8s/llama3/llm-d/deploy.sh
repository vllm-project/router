#!/bin/bash

# Deploy Llama 3.1 8B with P/D disaggregation using llm-d
# This script deploys the model service, GAIE components, and gateway infrastructure

set -e

NAMESPACE="llm-d-llama31"

echo "=========================================="
echo "Deploying Llama 3.1 8B P/D Disaggregation"
echo "=========================================="
echo "Namespace: $NAMESPACE"
echo ""

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo "Error: Namespace $NAMESPACE does not exist"
    echo "Please create namespace and secret first:"
    echo ""
    echo "  kubectl create namespace $NAMESPACE"
    echo "  kubectl create secret generic llm-d-hf-token \\"
    echo "    --from-literal=HF_TOKEN=hf_xxxxx \\"
    echo "    -n $NAMESPACE"
    echo ""
    exit 1
fi

# Check if secret exists
if ! kubectl get secret llm-d-hf-token -n "$NAMESPACE" &> /dev/null; then
    echo "Error: Secret llm-d-hf-token does not exist in $NAMESPACE"
    echo "Please create secret first:"
    echo ""
    echo "  kubectl create secret generic llm-d-hf-token \\"
    echo "    --from-literal=HF_TOKEN=hf_xxxxx \\"
    echo "    -n $NAMESPACE"
    echo ""
    exit 1
fi

echo "Prerequisites check passed"
echo ""
echo "Deploying with helmfile..."
helmfile apply

echo ""
echo "=========================================="
echo "Deployment initiated!"
echo "=========================================="
echo ""
echo "Monitor deployment progress:"
echo "  kubectl get pods -n $NAMESPACE -w"
echo ""
echo "Expected pods:"
echo "  - 8 prefill pods (1/1 ready)"
echo "  - 8 decode pods (2/2 ready)"
echo "  - 1 GAIE EPP pod (1/1 ready)"
echo "  - 1 gateway pod (1/1 ready)"
echo ""
echo "This may take 5-10 minutes for first deployment (model download)"
echo "Subsequent deployments take ~2 minutes (model cached)"
