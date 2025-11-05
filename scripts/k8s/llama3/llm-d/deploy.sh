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
echo "Verifying routing resources..."

# Check backend services
if kubectl get svc ms-llama31-llm-d-modelservice-prefill -n "$NAMESPACE" &>/dev/null; then
    echo "✓ Prefill service already exists"
else
    echo "Creating backend services (prefill & decode)..."
    kubectl apply -f backend-services.yaml
    if [ $? -eq 0 ]; then
        echo "✓ Backend services created"
    else
        echo "⚠ Warning: Could not create backend services"
    fi
fi

# Check InferencePool (use correct API version)
if kubectl get inferencepool.inference.networking.x-k8s.io gaie-llama31 -n "$NAMESPACE" &>/dev/null; then
    echo "✓ InferencePool already exists"
else
    echo "Creating InferencePool..."
    kubectl apply -f inferencepool.yaml
    if [ $? -eq 0 ]; then
        echo "✓ InferencePool created"
    else
        echo "⚠ Warning: Could not create InferencePool, may already exist with different format"
    fi
fi

# Check HTTPRoute
if kubectl get httproute -n "$NAMESPACE" &>/dev/null 2>&1; then
    echo "✓ HTTPRoute already exists"
else
    echo "Creating HTTPRoute..."
    kubectl apply -f httproute.yaml
    if [ $? -eq 0 ]; then
        echo "✓ HTTPRoute created"
    else
        echo "⚠ Warning: Could not create HTTPRoute"
    fi
fi

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Verify deployment:"
echo "  kubectl get pods -n $NAMESPACE"
echo "  kubectl get inferencepool.inference.networking.x-k8s.io -n $NAMESPACE"
echo "  kubectl get httproute -n $NAMESPACE"
echo ""
echo "Expected resources:"
echo "  - 8 prefill pods (1/1 ready)"
echo "  - 8 decode pods (2/2 ready)"
echo "  - 1 GAIE EPP pod (1/1 ready)"
echo "  - 1 gateway pod (1/1 ready)"
echo "  - 2 backend services (prefill, decode)"
echo "  - 1 InferencePool (gaie-llama31)"
echo "  - 1 HTTPRoute"
echo ""
echo "Deployment may take 5-10 minutes for first time (model download)"
echo "Subsequent deployments take ~2 minutes (model cached)"
echo ""
echo "Once all pods are ready, run benchmark:"
echo "  ./run-benchmark.sh 200 32"
echo ""
