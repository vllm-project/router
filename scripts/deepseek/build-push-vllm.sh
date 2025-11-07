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

# Configuration
GITHUB_USERNAME="Prowindy"
IMAGE_NAME="ghcr.io/${GITHUB_USERNAME}/vllm-custom"
IMAGE_TAG="latest"
VLLM_DIR="$HOME/gitrepos/vllm"

log "Building and pushing vLLM custom image..."
echo ""
echo "Configuration:"
echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  vLLM Dir: ${VLLM_DIR}"
echo ""

# Check if GITHUB_TOKEN is set
if [ -z "$GITHUB_TOKEN" ]; then
    error "GITHUB_TOKEN environment variable is not set"
    echo ""
    echo "Please set your GitHub Personal Access Token:"
    echo "  export GITHUB_TOKEN=ghp_your_token_here"
    echo ""
    echo "Create token at: https://github.com/settings/tokens"
    echo "Required scopes: write:packages, read:packages"
    exit 1
fi

# Check if vLLM directory exists
if [ ! -d "$VLLM_DIR" ]; then
    error "vLLM directory not found at: $VLLM_DIR"
    echo ""
    echo "Please clone vLLM first:"
    echo "  cd ~/gitrepos && git clone https://github.com/vllm-project/vllm.git"
    exit 1
fi

# Login to GitHub Container Registry
log "Logging into GitHub Container Registry..."
echo $GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin

if [ $? -ne 0 ]; then
    error "Failed to login to GitHub Container Registry"
    exit 1
fi

success "Logged in successfully"
echo ""

# Build Docker image
log "Building Docker image (this will take 10-20 minutes)..."
cd "$VLLM_DIR"

docker build \
    --target vllm-openai \
    --tag ${IMAGE_NAME}:${IMAGE_TAG} \
    --build-arg torch_cuda_arch_list="" \
    --build-arg INSTALL_KV_CONNECTORS=true \
    --build-arg VLLM_USE_PRECOMPILED=1 \
    --file docker/Dockerfile \
    .

if [ $? -ne 0 ]; then
    error "Docker build failed"
    exit 1
fi

success "Docker image built successfully"
echo ""

# Push to GitHub Container Registry
log "Pushing image to GitHub Container Registry..."
docker push ${IMAGE_NAME}:${IMAGE_TAG}

if [ $? -ne 0 ]; then
    error "Failed to push image to registry"
    exit 1
fi

success "Image pushed successfully!"
echo ""
echo "Image URL: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
log "You can now deploy to Kubernetes using:"
echo "  HF_TOKEN=hf_your_token ./deploy-vllm.sh"
