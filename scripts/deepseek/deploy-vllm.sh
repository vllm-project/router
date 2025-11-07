#!/bin/bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    error "HF_TOKEN environment variable is not set"
    echo ""
    echo "Please set your Hugging Face token:"
    echo "  export HF_TOKEN=hf_your_token_here"
    echo ""
    echo "Get your token from: https://huggingface.co/settings/tokens"
    exit 1
fi

log "Deploying vLLM DeepSeek V3 to Kubernetes..."

# Create a temporary file with the HF_TOKEN substituted
TEMP_MANIFEST=$(mktemp)
sed "s/REPLACE_WITH_YOUR_HF_TOKEN/$HF_TOKEN/g" /Users/congc/vllm-deepseek-deployment.yaml > "$TEMP_MANIFEST"

log "Applying Kubernetes manifests..."

# Apply the manifests
kubectl apply -f "$TEMP_MANIFEST"

# Clean up
rm -f "$TEMP_MANIFEST"

success "Manifests applied successfully!"
echo ""
log "Checking deployment status..."
echo ""

# Wait a moment for resources to be created
sleep 2

# Show status
kubectl get ns vllm
echo ""
kubectl get all -n vllm
echo ""

log "To check pod status:"
echo "  kubectl get pods -n vllm -w"
echo ""

log "To check decode logs:"
echo "  kubectl logs -n vllm -l app=vllm-decode -f"
echo ""

log "To check prefill logs:"
echo "  kubectl logs -n vllm -l app=vllm-prefill -f"
echo ""

log "To test the services (after pods are Running):"
echo "  # Port forward decode service"
echo "  kubectl port-forward -n vllm svc/vllm-decode 20005:20005"
echo ""
echo "  # Test decode endpoint"
echo "  curl http://localhost:20005/health"
echo ""

success "Deployment initiated! Pods may take 5-10 minutes to start (model download + initialization)"
