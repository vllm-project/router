#!/bin/bash

# vLLM Deployment Script
# This script can clone vLLM, build Docker image, upload to ECR, and deploy to remote hosts

set -e

# Configuration
VLLM_REPO="https://github.com/vllm-project/vllm.git"
VLLM_COMMIT="9fac6aa30b669de75d8718164cd99676d3530e7d"
ECR_REPO="584868043064.dkr.ecr.us-west-2.amazonaws.com/dev-vllm-repo"
ECR_REGION="us-west-2"
DOCKER_TAG="vllm/vllm-openai"

# SSH Configuration (REQUIRED - must be set via environment variables)
SSH_KEY_FILE="${SSH_KEY_FILE}"
SSH_USER="${SSH_USER:-congc}"
# Function to get or generate ECR tag
get_ecr_tag() {
    local tag_file="/tmp/vllm_ecr_tag"
    if [ -f "$tag_file" ] && [ -s "$tag_file" ]; then
        cat "$tag_file"
    else
        echo "$(whoami)_$(date +%Y%m%d_%H%M%S)"
    fi
}

# Function to save ECR tag
save_ecr_tag() {
    local tag_file="/tmp/vllm_ecr_tag"
    echo "$1" > "$tag_file"
}

ECR_TAG=${ECR_TAG:-$(get_ecr_tag)}

# vLLM Configuration (REQUIRED - must be set via environment variables)
HF_TOKEN="${HF_TOKEN}"
GPU_ID="${GPU_ID:-0,1,2,3,4,5,6,7}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V3-0324}"
DECODE_PORT="20005"
KV_PORT="22001"
PREFILL_PORT="20003"
PREFILL_KV_PORT="21001"
TP_SIZE="${TP_SIZE:-8}"
ENABLE_PROFILING="${ENABLE_PROFILING:-false}"
PROFILING_DIR="${PROFILING_DIR:-/opt/dlami/nvme/vllm_profiles}"

# Remote hosts configuration (space-separated)
# AWS EC2 instances for deployment (US-WEST-1)
# GPU instance 1 - for prefill
PREFILL_REMOTE_HOSTS="congc@ec2-13-57-84-212.us-west-1.compute.amazonaws.com"
# GPU instance 2 - for decode
DECODE_REMOTE_HOSTS="congc@ec2-54-183-209-205.us-west-1.compute.amazonaws.com"
# Backward compatibility - all hosts combined
REMOTE_HOSTS="$PREFILL_REMOTE_HOSTS $DECODE_REMOTE_HOSTS"

# SSH host aliases (for reference):
# aws-gpu-node1: ec2-13-57-84-212.us-west-1.compute.amazonaws.com (prefill)
# aws-gpu-node2: ec2-54-183-209-205.us-west-1.compute.amazonaws.com (decode)
# User: congc
# SSH Key: ~/.ssh/ec2_instance_private_key.pem

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Function to validate required environment variables
validate_env() {
    local missing_vars=()
    local need_ssh=false
    local need_hf=false

    # Check if any command requires SSH
    for cmd in "${COMMANDS[@]}"; do
        case $cmd in
            deploy_remote_decode|deploy_remote_prefill|deploy_router|all)
                need_ssh=true
                need_hf=true
                ;;
            deploy_local|deploy_prefill_local|deploy_decode_local)
                need_hf=true
                ;;
        esac
    done

    # Validate SSH_KEY_FILE if needed
    if [ "$need_ssh" = true ]; then
        if [ -z "$SSH_KEY_FILE" ]; then
            missing_vars+=("SSH_KEY_FILE")
        elif [ ! -f "$SSH_KEY_FILE" ]; then
            error "SSH key file not found: $SSH_KEY_FILE"
            exit 1
        fi
    fi

    # Validate HF_TOKEN if needed
    if [ "$need_hf" = true ]; then
        if [ -z "$HF_TOKEN" ]; then
            missing_vars+=("HF_TOKEN")
        fi
    fi

    # Report missing variables
    if [ ${#missing_vars[@]} -gt 0 ]; then
        error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        echo ""
        error "Please set the required environment variables and try again."
        echo ""
        echo "Example usage:"
        echo "  HF_TOKEN=hf_xxxxx SSH_KEY_FILE=~/.ssh/mykey.pem $0 all"
        echo ""
        exit 1
    fi
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
    clone               Clone vLLM repository
    build               Build Docker image
    upload              Upload image to ECR
    deploy_local        Deploy and run vLLM locally (requires HF_TOKEN)
    deploy_remote_decode Deploy decode instances to decode hosts (requires HF_TOKEN, SSH_KEY_FILE)
    deploy_remote_prefill Deploy prefill instances to prefill hosts (requires HF_TOKEN, SSH_KEY_FILE)
    deploy_router       Deploy vLLM router to remote hosts (requires SSH_KEY_FILE)
    deploy_router_local Deploy vLLM router locally (ports 10001, 30001)
    deploy_prefill_local Deploy vLLM prefill server locally (requires HF_TOKEN)
    deploy_decode_local Deploy vLLM decode server locally (requires HF_TOKEN)
    benchmark           Run vLLM benchmark against the router
    setup_docker_storage Setup Docker to use largest available storage (run directly on target host)
    install_gdrcopy     Install gdrcopy for optimal NIXL performance (run on target host)
    all                 Run all steps: verify env, clone, build, upload, deploy (requires HF_TOKEN, SSH_KEY_FILE)

Options:
    -h, --help              Show this help message
    -c, --commit COMMIT     Use specific commit (default: $VLLM_COMMIT)
    -t, --tag TAG           ECR tag (default: auto-generated)
    -r, --hosts "host1 host2"  Remote hosts (default: from script)
    --no-precompiled        Build from source instead of using precompiled
    --dry-run               Show what would be done without executing

REQUIRED Environment Variables (for deployment commands):
    HF_TOKEN               Hugging Face API token (required for model access)
                          Get your token from: https://huggingface.co/settings/tokens

    SSH_KEY_FILE          Path to SSH private key for remote hosts
                          Example: ~/.ssh/ec2_instance_private_key.pem

OPTIONAL Environment Variables:
    SSH_USER              SSH username (default: congc)
    ECR_TAG               Custom ECR tag (default: auto-generated)
    ENABLE_PROFILING      Enable PyTorch profiling (default: false)
                          Set to "true" to enable profiling with VLLM_TORCH_PROFILER_DIR
    PROFILING_DIR         Directory for profiling traces (default: /opt/dlami/nvme/vllm_profiles)
    BENCHMARK_NUM_PROMPTS      Number of prompts (default: 10000)
    BENCHMARK_INPUT_LEN        Input token length (default: 2000)
    BENCHMARK_OUTPUT_LEN       Output token length (default: 2000)
    BENCHMARK_MAX_CONCURRENCY  Max concurrent requests (default: 32)
    BENCHMARK_ROUTER_HOST      Router host (default: host.docker.internal)
    BENCHMARK_ROUTER_PORT      Router port (default: 10001)

Examples:
    # Full deployment (RECOMMENDED - sets up everything)
    HF_TOKEN=hf_xxxxx SSH_KEY_FILE=~/.ssh/mykey.pem $0 all

    # Local deployment only
    HF_TOKEN=hf_xxxxx $0 deploy_prefill_local deploy_decode_local

    # Build and upload only (no env vars needed)
    $0 build upload

    # Deploy to specific remote hosts
    HF_TOKEN=hf_xxxxx SSH_KEY_FILE=~/.ssh/mykey.pem \\
        $0 -r "user@host1.com user@host2.com" deploy_remote_decode

    # Run benchmark
    BENCHMARK_NUM_PROMPTS=1000 $0 benchmark

    # Preview all steps without executing
    HF_TOKEN=hf_xxxxx SSH_KEY_FILE=~/.ssh/mykey.pem $0 --dry-run all

IMPORTANT NOTES:
    - HF_TOKEN is REQUIRED for all deployment commands (local and remote)
    - SSH_KEY_FILE is REQUIRED for remote deployment commands
    - The script will validate these variables before proceeding
    - For 'all' command, both HF_TOKEN and SSH_KEY_FILE are required
EOF
}

# Function to clone vLLM
clone_vllm() {
    log "Cloning vLLM repository..."

    if [ -d "vllm/.git" ]; then
        log "vLLM directory exists, skipping clone to preserve local changes..."
    else
        if [ -d "vllm" ]; then
            log "vLLM directory exists but is not a git repo, removing..."
            rm -rf vllm
        fi
        git clone $VLLM_REPO
    fi

    success "vLLM repository ready"
}

# Function to build Docker image
build_docker() {
    log "Building Docker image..."

    if [ ! -d "vllm" ]; then
        error "vLLM directory not found. Run 'clone' first."
        exit 1
    fi

    # Check if image already exists
    if docker images | grep -q "$DOCKER_TAG"; then
        log "Docker image $DOCKER_TAG already exists, skipping build..."
        success "Docker image already built: $DOCKER_TAG"
        return 0
    fi

    cd vllm

    BUILD_ARGS="--target vllm-openai --tag $DOCKER_TAG --build-arg torch_cuda_arch_list=\"\" --build-arg INSTALL_KV_CONNECTORS=true --file docker/Dockerfile"

    if [ "$USE_PRECOMPILED" = "true" ]; then
        BUILD_ARGS="$BUILD_ARGS --build-arg VLLM_USE_PRECOMPILED=1"
        log "Building with precompiled binaries..."
    else
        BUILD_ARGS="$BUILD_ARGS --build-arg max_jobs=$(($(nproc) * 2)) --build-arg nvcc_threads=3 --build-arg VLLM_MAX_SIZE_MB=2000"
        log "Building from source (this will take a long time)..."
    fi

    if [ "$DRY_RUN" = "true" ]; then
        echo "Would run: DOCKER_BUILDKIT=1 docker build . $BUILD_ARGS"
    else
        DOCKER_BUILDKIT=1 docker build . $BUILD_ARGS
        success "Docker image built successfully: $DOCKER_TAG"
    fi

    cd ..
}

# Function to get largest available mount point for Docker storage
get_largest_mount() {
    # Find the largest mount point by available space, excluding tmpfs, devtmpfs, and NFS
    # Works across AWS, GCP, Azure, and other cloud providers
    local largest_mount=$(df -h | grep -vE 'tmpfs|devtmpfs|:/|loop|udev' | awk 'NR>1 {print $4, $6}' | sort -hr | head -1 | awk '{print $2}')

    # Fallback to root if nothing found
    if [[ -z "$largest_mount" ]]; then
        largest_mount="/"
    fi

    echo "$largest_mount"
}

# Function to setup Docker to use largest storage mount
setup_docker_storage() {
    log "Configuring Docker to use largest available storage..." >&2

    # Detect largest mount point
    local largest_mount=$(get_largest_mount)
    local docker_data_root="${largest_mount}/docker"

    log "Largest mount detected: $largest_mount" >&2
    log "Docker data-root will be: $docker_data_root" >&2

    # Create Docker daemon.json configuration script
    cat << 'DOCKER_SETUP_EOF'
#!/bin/bash

# Get the largest mount point
LARGEST_MOUNT=$(df -h | grep -vE 'tmpfs|devtmpfs|:/|loop|udev' | awk 'NR>1 {print $4, $6}' | sort -hr | head -1 | awk '{print $2}')
if [[ -z "$LARGEST_MOUNT" ]]; then
    LARGEST_MOUNT="/"
fi

DOCKER_DATA_ROOT="${LARGEST_MOUNT}/docker"

echo "Configuring Docker to use: $DOCKER_DATA_ROOT"

# Stop Docker service
sudo systemctl stop docker 2>/dev/null || sudo service docker stop 2>/dev/null || true

# Backup existing Docker data if it exists
if [ -d /var/lib/docker ] && [ "$(ls -A /var/lib/docker)" ]; then
    echo "Backing up existing Docker data..."
    sudo mkdir -p "${DOCKER_DATA_ROOT}_backup"
    sudo rsync -av /var/lib/docker/ "${DOCKER_DATA_ROOT}_backup/" || true
fi

# Create new Docker data directory
sudo mkdir -p "$DOCKER_DATA_ROOT"

# Read existing Docker daemon.json if it exists
sudo mkdir -p /etc/docker
if [ -f /etc/docker/daemon.json ] && [ -s /etc/docker/daemon.json ]; then
    # Backup existing config
    sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak

    # Check if data-root already exists
    if sudo cat /etc/docker/daemon.json | grep -q '"data-root"'; then
        echo "data-root already configured, skipping..."
    else
        # Merge with existing config using jq if available
        if command -v jq &> /dev/null; then
            sudo jq --arg dataroot "$DOCKER_DATA_ROOT" '. + {"data-root": $dataroot}' /etc/docker/daemon.json | sudo tee /etc/docker/daemon.json.tmp > /dev/null
            sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json
        else
            # No jq available, use python to merge JSON properly
            sudo python3 -c "
import json
with open('/etc/docker/daemon.json', 'r') as f:
    config = json.load(f)
config['data-root'] = '$DOCKER_DATA_ROOT'
with open('/etc/docker/daemon.json', 'w') as f:
    json.dump(config, f, indent=2)
"
        fi
    fi
else
    # Create new daemon.json with NVIDIA runtime and data-root
    sudo tee /etc/docker/daemon.json > /dev/null << DAEMON_JSON
{
  "runtimes": {
    "nvidia": {
      "args": [],
      "path": "nvidia-container-runtime"
    }
  },
  "data-root": "$DOCKER_DATA_ROOT"
}
DAEMON_JSON
fi

# If backup exists, move it to new location
if [ -d "${DOCKER_DATA_ROOT}_backup" ] && [ "$(ls -A ${DOCKER_DATA_ROOT}_backup)" ]; then
    echo "Moving Docker data to new location..."
    sudo rsync -av "${DOCKER_DATA_ROOT}_backup/" "$DOCKER_DATA_ROOT/" || true
    sudo rm -rf "${DOCKER_DATA_ROOT}_backup"
fi

# Start Docker service
sudo systemctl start docker 2>/dev/null || sudo service docker start 2>/dev/null || true

# Wait for Docker to be ready
sleep 5

# Verify Docker is running
if docker info &> /dev/null; then
    echo "Docker successfully reconfigured!"
    echo "Docker Root Dir: $(docker info 2>/dev/null | grep 'Docker Root Dir' | awk '{print $4}')"
else
    echo "WARNING: Docker may not be running properly. Check with: sudo systemctl status docker"
fi
DOCKER_SETUP_EOF
}

# Function to show gdrcopy installation instructions
install_gdrcopy() {
    cat << 'GDRCOPY_HELP_EOF'
================================================================================
gdrcopy Installation Instructions (Optional but Recommended)
================================================================================

For optimal NIXL performance, install gdrcopy on both prefill and decode hosts.
If gdrcopy is not installed, NIXL will still work but with lower performance.

Installation Steps:
-------------------

1. Clone vLLM repository (if not already cloned):
   git clone https://github.com/vllm-project/vllm.git
   cd vllm

2. Run the install_gdrcopy.sh script:

   For Ubuntu 20.04:
   sudo tools/install_gdrcopy.sh "ubuntu2004" "12.8" "x64"

   For Ubuntu 22.04:
   sudo tools/install_gdrcopy.sh "ubuntu2204" "12.8" "x64"

   For Ubuntu 24.04:
   sudo tools/install_gdrcopy.sh "ubuntu2404" "12.8" "x64"

   For ARM64 (aarch64) systems, replace "x64" with "aarch64"

3. Verify installation:
   dpkg -l | grep libgdrapi

Available OS versions can be found here:
https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2012.8/

Note: This command requires sudo privileges and will download and install
the appropriate gdrcopy package for your system.

================================================================================
GDRCOPY_HELP_EOF
}

# Function to get local IP address
get_local_ip() {
    # Try to get the primary interface IP (not localhost)
    local ip=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'src \K\S+' | head -1)
    if [[ -z "$ip" ]]; then
        # Fallback to hostname -I
        ip=$(hostname -I | awk '{print $1}')
    fi
    if [[ -z "$ip" ]]; then
        # Last resort fallback
        ip="127.0.0.1"
    fi
    echo "$ip"
}

# Function to get router container IP address for local deployments
get_router_container_ip() {
    # Try to get the router container IP from docker inspect
    local router_ip=$(docker inspect vllm-router-local 2>/dev/null | grep '"IPAddress":' | head -1 | grep -o '[0-9.]*[0-9]')
    if [[ -n "$router_ip" && "$router_ip" != "" ]]; then
        echo "$router_ip"
    else
        # Fallback to host IP for remote deployments
        get_local_ip
    fi
}

# Shared function to generate vLLM run script
generate_vllm_run_script() {
    local ROUTER_IP=${1:-$(get_local_ip)}
    local INTERNAL_IP=${2:-$(get_local_ip)}
    cat << EOF
#!/bin/bash

# Stop any existing containers
docker stop vllm-deepseek 2>/dev/null || true
sleep 2  # Give container time to stop properly
docker rm vllm-deepseek 2>/dev/null || true

# Login to ECR
aws ecr get-login-password --region $ECR_REGION | docker login --username AWS --password-stdin $ECR_REPO

# Pull latest image
docker pull $ECR_REPO:$ECR_TAG

# Create huggingface cache directory if it doesn't exist
sudo mkdir -p /opt/dlami/nvme/huggingface_cache
sudo chown -R \$(id -u):\$(id -g) /opt/dlami/nvme/huggingface_cache

# Create profiling directory if profiling is enabled
if [ "$ENABLE_PROFILING" = "true" ]; then
    sudo mkdir -p $PROFILING_DIR
    sudo chown -R \$(id -u):\$(id -g) $PROFILING_DIR
fi

# Run vLLM container with EFA support (EFA libraries built into image)
docker run -d --runtime nvidia --gpus all \
    --device /dev/infiniband/uverbs0 \
    --device /dev/infiniband/uverbs1 \
    --device /dev/infiniband/uverbs2 \
    --device /dev/infiniband/uverbs3 \
    --device /dev/infiniband/uverbs4 \
    --device /dev/infiniband/uverbs5 \
    --device /dev/infiniband/uverbs6 \
    --device /dev/infiniband/uverbs7 \
    --device /dev/infiniband/uverbs8 \
    --device /dev/infiniband/uverbs9 \
    --device /dev/infiniband/uverbs10 \
    --device /dev/infiniband/uverbs11 \
    --device /dev/infiniband/uverbs12 \
    --device /dev/infiniband/uverbs13 \
    --device /dev/infiniband/uverbs14 \
    --device /dev/infiniband/uverbs15 \
    -v /opt/dlami/nvme/huggingface_cache:/root/.cache/huggingface \$(if [ "$ENABLE_PROFILING" = "true" ]; then echo " -v $PROFILING_DIR:/vllm_profiles"; fi) \
    --shm-size=1000g \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    --env "VLLM_MOE_DP_CHUNK_SIZE=512" \
    --env "TRITON_LIBCUDA_PATH=/usr/lib64" \
    --env "HF_HUB_DISABLE_XET=1" \
    --env "VLLM_SKIP_P2P_CHECK=1" \
    --env "VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1" \
    --env "VLLM_USE_DEEP_GEMM=1" \
    --env "VLLM_ALL2ALL_BACKEND=deepep_low_latency" \
    --env "NVIDIA_GDRCOPY=enabled" \
    --env "UCX_TLS=all" \
    --env "UCX_NET_DEVICES=all" \
    --env "NVSHMEM_DEBUG=INFO" \
    --env "NVSHMEM_REMOTE_TRANSPORT=ibgda" \
    --env "NVSHMEM_IB_ENABLE_IBGDA=true" \
    --env "GLOO_SOCKET_IFNAME=" \
    --env "NCCL_SOCKET_IFNAME=" \
    --env "NCCL_IB_HCA=ibp" \
    --env "FI_EFA_USE_DEVICE_RDMA=1" \
    --env "NCCL_DEBUG=INFO" \
    --env "NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,P2P" \
    --env "NCCL_DEBUG_FILE=/tmp/nccl_debug_%h_%p.log" \
    --env "VLLM_LOGGING_LEVEL=DEBUG" \
    --env "VLLM_LOG_REQUESTS=1" \
    --env "VLLM_RPC_TIMEOUT=\$(if [ "$ENABLE_PROFILING" = "true" ]; then echo "1800"; else echo "300"; fi)" \
    --env "VLLM_WORKER_RPC_TIMEOUT=300" \
    --env "HF_HUB_CACHE=/root/.cache/huggingface/hub" \
    --env "CUDA_VISIBLE_DEVICES=$GPU_ID" \
    --env "VLLM_USE_V1=1" \
    --env "VLLM_NIXL_SIDE_CHANNEL_HOST=$INTERNAL_IP" \$(if [ "$ENABLE_PROFILING" = "true" ]; then echo " --env VLLM_TORCH_PROFILER_DIR=/vllm_profiles --env VLLM_TORCH_PROFILER_WITH_STACK=1"; fi) \
    --network host \
    --name vllm-deepseek \
    $ECR_REPO:$ECR_TAG \
    --model $MODEL \
    --host 0.0.0.0 \
    --port $DECODE_PORT \
    --disable-log-requests \
    --disable-uvicorn-access-log \
    --enable-expert-parallel \
    --tensor-parallel-size $TP_SIZE \
    --trust-remote-code \
    --async-scheduling \
    --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}" \
    --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"backends\":[\"UCX\",\"GDS\"]}}"

echo "vLLM started in background. Check logs with: docker logs -f vllm-deepseek"
echo "API available at: http://localhost:$DECODE_PORT"
EOF
}

# Shared function to generate vLLM prefill run script
generate_vllm_prefill_run_script() {
    local ROUTER_IP=${1:-$(get_local_ip)}
    local INTERNAL_IP=${2:-$(get_local_ip)}
    cat << EOF
#!/bin/bash

# Stop any existing prefill containers
docker stop vllm-deepseek-prefill 2>/dev/null || true
sleep 2  # Give container time to stop properly
docker rm vllm-deepseek-prefill 2>/dev/null || true

# Login to ECR
aws ecr get-login-password --region $ECR_REGION | docker login --username AWS --password-stdin $ECR_REPO

# Pull latest image
docker pull $ECR_REPO:$ECR_TAG

# Create huggingface cache directory if it doesn't exist
sudo mkdir -p /opt/dlami/nvme/huggingface_cache
sudo chown -R \$(id -u):\$(id -g) /opt/dlami/nvme/huggingface_cache

# Create profiling directory if profiling is enabled
if [ "$ENABLE_PROFILING" = "true" ]; then
    sudo mkdir -p $PROFILING_DIR
    sudo chown -R \$(id -u):\$(id -g) $PROFILING_DIR
fi

# Run vLLM prefill container with EFA support (EFA libraries built into image)
docker run -d --runtime nvidia --gpus all \
    --device /dev/infiniband/uverbs0 \
    --device /dev/infiniband/uverbs1 \
    --device /dev/infiniband/uverbs2 \
    --device /dev/infiniband/uverbs3 \
    --device /dev/infiniband/uverbs4 \
    --device /dev/infiniband/uverbs5 \
    --device /dev/infiniband/uverbs6 \
    --device /dev/infiniband/uverbs7 \
    --device /dev/infiniband/uverbs8 \
    --device /dev/infiniband/uverbs9 \
    --device /dev/infiniband/uverbs10 \
    --device /dev/infiniband/uverbs11 \
    --device /dev/infiniband/uverbs12 \
    --device /dev/infiniband/uverbs13 \
    --device /dev/infiniband/uverbs14 \
    --device /dev/infiniband/uverbs15 \
    -v /opt/dlami/nvme/huggingface_cache:/root/.cache/huggingface \$(if [ "$ENABLE_PROFILING" = "true" ]; then echo " -v $PROFILING_DIR:/vllm_profiles"; fi) \
    --shm-size=1000g \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    --env "VLLM_MOE_DP_CHUNK_SIZE=512" \
    --env "TRITON_LIBCUDA_PATH=/usr/lib64" \
    --env "HF_HUB_DISABLE_XET=1" \
    --env "VLLM_SKIP_P2P_CHECK=1" \
    --env "VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1" \
    --env "VLLM_USE_DEEP_GEMM=1" \
    --env "VLLM_ALL2ALL_BACKEND=deepep_high_throughput" \
    --env "NVIDIA_GDRCOPY=enabled" \
    --env "UCX_TLS=all" \
    --env "UCX_NET_DEVICES=all" \
    --env "NVSHMEM_DEBUG=INFO" \
    --env "NVSHMEM_REMOTE_TRANSPORT=ibgda" \
    --env "NVSHMEM_IB_ENABLE_IBGDA=true" \
    --env "GLOO_SOCKET_IFNAME=" \
    --env "NCCL_SOCKET_IFNAME=" \
    --env "NCCL_IB_HCA=ibp" \
    --env "FI_EFA_USE_DEVICE_RDMA=1" \
    --env "NCCL_DEBUG=INFO" \
    --env "NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,P2P" \
    --env "NCCL_DEBUG_FILE=/tmp/nccl_debug_%h_%p.log" \
    --env "VLLM_LOGGING_LEVEL=DEBUG" \
    --env "VLLM_LOG_REQUESTS=1" \
    --env "VLLM_RPC_TIMEOUT=\$(if [ "$ENABLE_PROFILING" = "true" ]; then echo "1800"; else echo "300"; fi)" \
    --env "VLLM_WORKER_RPC_TIMEOUT=300" \
    --env "HF_HUB_CACHE=/root/.cache/huggingface/hub" \
    --env "CUDA_VISIBLE_DEVICES=$GPU_ID" \
    --env "VLLM_USE_V1=1" \
    --env "VLLM_NIXL_SIDE_CHANNEL_HOST=$INTERNAL_IP" \$(if [ "$ENABLE_PROFILING" = "true" ]; then echo " --env VLLM_TORCH_PROFILER_DIR=/vllm_profiles --env VLLM_TORCH_PROFILER_WITH_STACK=1"; fi) \
    --network host \
    --name vllm-deepseek-prefill \
    $ECR_REPO:$ECR_TAG \
    --model $MODEL \
    --enforce-eager \
    --host 0.0.0.0 \
    --port $PREFILL_PORT \
    --enable-expert-parallel \
    --tensor-parallel-size $TP_SIZE \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --disable-log-stats \
    --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"backends\":[\"UCX\",\"GDS\"]}}"

echo "vLLM prefill started in background. Check logs with: docker logs -f vllm-deepseek-prefill"
echo "Prefill API available at: http://localhost:$PREFILL_PORT"
EOF
}

# Function to upload to ECR
upload_ecr() {
    log "Uploading image to ECR..."

    # Check if image already exists in ECR (skip if no permissions)
    ECR_REPO_NAME=$(echo $ECR_REPO | cut -d'/' -f2)
    if aws ecr describe-images --repository-name $ECR_REPO_NAME --image-ids imageTag=$ECR_TAG --region $ECR_REGION >/dev/null 2>&1; then
        log "Image $ECR_REPO:$ECR_TAG already exists in ECR, skipping upload..."
        success "Image already uploaded to ECR: $ECR_REPO:$ECR_TAG"
        return 0
    else
        # If describe-images fails (likely due to permissions), continue with upload
        # The push will be fast if layers already exist
        log "Cannot check if image exists in ECR (may lack describe permissions), proceeding with upload..."
    fi

    # Login to ECR
    if [ "$DRY_RUN" = "true" ]; then
        echo "Would run: aws ecr get-login-password --region $ECR_REGION | docker login --username AWS --password-stdin $ECR_REPO"
        echo "Would run: docker tag $DOCKER_TAG $ECR_REPO:$ECR_TAG"
        echo "Would run: docker push $ECR_REPO:$ECR_TAG"
    else
        log "Logging into ECR..."
        aws ecr get-login-password --region $ECR_REGION | docker login --username AWS --password-stdin $ECR_REPO

        log "Tagging image for ECR..."
        docker tag $DOCKER_TAG $ECR_REPO:$ECR_TAG

        log "Pushing to ECR..."
        docker push $ECR_REPO:$ECR_TAG

        # Save the ECR tag for future use
        save_ecr_tag "$ECR_TAG"
        success "Image uploaded to ECR: $ECR_REPO:$ECR_TAG"
    fi
}

# Function to deploy locally
deploy_local() {
    log "Deploying vLLM locally..."

    if [ "$DRY_RUN" = "true" ]; then
        echo "Would run vLLM container with:"
        echo "  Image: $ECR_REPO:$ECR_TAG"
        echo "  Model: $MODEL"
        echo "  Port: $DECODE_PORT"
        echo "  Tensor Parallel Size: $TP_SIZE"
    else
        # Generate and execute the vLLM run script locally
        generate_vllm_run_script $(get_local_ip) | bash

        sleep 3
        if docker ps | grep -q vllm-deepseek; then
            success "vLLM started successfully"
            log "Check logs with: docker logs vllm-deepseek"
            log "API available at: http://localhost:$DECODE_PORT"
        else
            error "Failed to start vLLM container"
            log "Check logs with: docker logs vllm-deepseek"
            exit 1
        fi
    fi
}

# Function to deploy decode instances to remote hosts
deploy_remote_decode() {
    log "Deploying decode instances to remote hosts..."

    # Validation is already done by validate_env()

    # Deploy to each host with its specific external IP
    ROUTER_IP=$(get_local_ip)

    for host in $DECODE_REMOTE_HOSTS; do
        log "Deploying decode to $host..."

        if [ "$DRY_RUN" = "true" ]; then
            echo "Would deploy decode script to $host with external IP detection"
        else
            # First, setup Docker storage on the remote host (skip if already configured)
            DOCKER_ROOT=$(ssh -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no $host "docker info 2>/dev/null | grep 'Docker Root Dir' | grep -v '/var/lib/docker' || true")
            if [ -z "$DOCKER_ROOT" ]; then
                log "Setting up Docker storage on $host..." >&2
                setup_docker_storage | ssh -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no $host "bash"
            else
                log "Docker storage already configured on $host" >&2
            fi

            # Get the internal IP of the remote host for AWS VPC communication
            INTERNAL_IP=$(ssh -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no $host "ip route get 8.8.8.8 | grep -oP 'src \K\S+'")
            log "Internal IP for $host: $INTERNAL_IP"

            # Generate script with host-specific internal IP
            generate_vllm_run_script $ROUTER_IP $INTERNAL_IP > /tmp/vllm_run_remote_${INTERNAL_IP}.sh
            chmod +x /tmp/vllm_run_remote_${INTERNAL_IP}.sh

            # Copy script to remote host
            scp -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no /tmp/vllm_run_remote_${INTERNAL_IP}.sh $host:/tmp/vllm_run_remote.sh

            # Execute on remote host
            ssh -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no $host "bash /tmp/vllm_run_remote.sh"

            # Clean up host-specific temp file
            rm -f /tmp/vllm_run_remote_${INTERNAL_IP}.sh

            success "Decode deployed to $host (Internal IP: $INTERNAL_IP)"
        fi
    done
}

# Function to deploy prefill to remote hosts
deploy_remote_prefill() {
    log "Deploying vLLM prefill to remote hosts..."

    # Validation is already done by validate_env()

    # Deploy to each host with its specific external IP
    ROUTER_IP=$(get_local_ip)

    for host in $PREFILL_REMOTE_HOSTS; do
        log "Deploying prefill to $host..."

        if [ "$DRY_RUN" = "true" ]; then
            echo "Would deploy prefill script to $host with external IP detection"
        else
            # First, setup Docker storage on the remote host (skip if already configured)
            DOCKER_ROOT=$(ssh -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no $host "docker info 2>/dev/null | grep 'Docker Root Dir' | grep -v '/var/lib/docker' || true")
            if [ -z "$DOCKER_ROOT" ]; then
                log "Setting up Docker storage on $host..." >&2
                setup_docker_storage | ssh -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no $host "bash"
            else
                log "Docker storage already configured on $host" >&2
            fi

            # Get the internal IP of the remote host for AWS VPC communication
            INTERNAL_IP=$(ssh -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no $host "ip route get 8.8.8.8 | grep -oP 'src \K\S+'")
            log "Internal IP for $host: $INTERNAL_IP"

            # Generate script with host-specific internal IP
            generate_vllm_prefill_run_script $ROUTER_IP $INTERNAL_IP > /tmp/vllm_run_remote_prefill_${INTERNAL_IP}.sh
            chmod +x /tmp/vllm_run_remote_prefill_${INTERNAL_IP}.sh

            # Copy script to remote host
            scp -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no /tmp/vllm_run_remote_prefill_${INTERNAL_IP}.sh $host:/tmp/vllm_run_remote_prefill.sh

            # Execute on remote host
            ssh -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no $host "bash /tmp/vllm_run_remote_prefill.sh"

            # Clean up host-specific temp file
            rm -f /tmp/vllm_run_remote_prefill_${INTERNAL_IP}.sh

            success "Prefill deployed to $host (Internal IP: $INTERNAL_IP)"
        fi
    done
}

# Function to deploy router (clone, build, upload, deploy)
deploy_router() {
    log "Deploying vLLM router..."

    # Check if SSH key exists for remote deployment
    if [ ! -f "$HOME/.ssh/ec2_instance_private_key.pem" ]; then
        error "SSH private key not found at $HOME/.ssh/ec2_instance_private_key.pem"
        log "Please run: $0 setup_ssh"
        exit 1
    fi

    # Step 1: Use the existing router directory (this script should be run from ~/router)
    log "Using current directory as router repository..."
    ROUTER_DIR="$HOME/router"

    if [ ! -d "$ROUTER_DIR" ]; then
        error "Router directory not found at $ROUTER_DIR"
        exit 1
    fi

    # Step 2: Build router Docker image
    log "Building router Docker image..."
    cd "$ROUTER_DIR"

    if [ ! -f "Dockerfile" ] && [ ! -f "Dockerfile.router" ]; then
        error "Dockerfile or Dockerfile.router not found in vllm-router directory"
        exit 1
    fi

    # Use the correct Dockerfile name
    DOCKERFILE_NAME="Dockerfile"
    if [ -f "Dockerfile.router" ]; then
        DOCKERFILE_NAME="Dockerfile.router"
    fi

    ROUTER_DOCKER_TAG="vllm-router-service"
    ROUTER_ECR_TAG="router_$(whoami)_$(date +%Y%m%d_%H%M%S)"

    if [ "$DRY_RUN" = "true" ]; then
        echo "Would run: docker build -f $DOCKERFILE_NAME -t $ROUTER_DOCKER_TAG ."
    else
        docker build -f $DOCKERFILE_NAME -t $ROUTER_DOCKER_TAG .
        success "Router Docker image built: $ROUTER_DOCKER_TAG"
    fi

    cd ..

    # Step 3: Upload router image to ECR
    log "Uploading router image to ECR..."
    if [ "$DRY_RUN" = "true" ]; then
        echo "Would tag and push router image to ECR"
    else
        # Login to ECR
        aws ecr get-login-password --region $ECR_REGION | docker login --username AWS --password-stdin $ECR_REPO

        # Tag for ECR
        docker tag $ROUTER_DOCKER_TAG $ECR_REPO:$ROUTER_ECR_TAG

        # Push to ECR
        docker push $ECR_REPO:$ROUTER_ECR_TAG
        success "Router image uploaded to ECR: $ECR_REPO:$ROUTER_ECR_TAG"
    fi

    # Step 4: Deploy router to remote hosts
    log "Deploying router to remote hosts..."

    # Create router deployment script
    cat > /tmp/router_deploy.sh << EOF
#!/bin/bash

# Stop any existing router containers
docker stop vllm-router 2>/dev/null || true
docker rm vllm-router 2>/dev/null || true

# Login to ECR
aws ecr get-login-password --region $ECR_REGION | docker login --username AWS --password-stdin $ECR_REPO

# Pull router image
docker pull $ECR_REPO:$ROUTER_ECR_TAG

# Run router container with correct ports for P/D disaggregation
docker run -d \\
    --name vllm-router \\
    -p 10001:10001 \\
    -p 30001:30001 \\
    --restart unless-stopped \\
    $ECR_REPO:$ROUTER_ECR_TAG \\
    vllm-router --vllm-pd-disaggregation --vllm-discovery-address 0.0.0.0:30001 --host 0.0.0.0 --port 10001 \\\$(if [ "\$ENABLE_PROFILING" = "true" ]; then echo " --profile"; fi)

echo "Router started. Check logs with: docker logs vllm-router"
echo "Router HTTP available at: http://localhost:10001"
echo "Router Discovery available at: localhost:30001"
EOF

    chmod +x /tmp/router_deploy.sh

    warn "NOTE: deploy_router deploys to remote hosts. Use deploy_router_local for localhost deployment."

    for host in $REMOTE_HOSTS; do
        log "Deploying router to $host..."

        if [ "$DRY_RUN" = "true" ]; then
            echo "Would copy router deployment script to $host and execute"
        else
            # Copy script to remote host
            scp -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no /tmp/router_deploy.sh $host:/tmp/

            # Execute on remote host
            ssh -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no $host "bash /tmp/router_deploy.sh"

            success "Router deployed to $host"
        fi
    done

    # Clean up temp file
    rm -f /tmp/router_deploy.sh

    success "Router deployment completed successfully!"
}

# Function to deploy router locally with correct ports
deploy_router_local() {
    log "Deploying vLLM router locally..."

    # Step 1: Use the existing router directory (this script should be run from ~/router)
    log "Using current directory as router repository..."
    ROUTER_DIR="$HOME/router"

    if [ ! -d "$ROUTER_DIR" ]; then
        error "Router directory not found at $ROUTER_DIR"
        exit 1
    fi

    # Step 2: Build router Docker image
    log "Building router Docker image..."
    cd "$ROUTER_DIR"

    if [ ! -f "Dockerfile" ] && [ ! -f "Dockerfile.router" ]; then
        error "Dockerfile or Dockerfile.router not found in vllm-router directory"
        exit 1
    fi

    # Use the correct Dockerfile name
    DOCKERFILE_NAME="Dockerfile"
    if [ -f "Dockerfile.router" ]; then
        DOCKERFILE_NAME="Dockerfile.router"
    fi

    ROUTER_DOCKER_TAG="vllm-router-service"

    if [ "$DRY_RUN" = "true" ]; then
        echo "Would run: docker build -f $DOCKERFILE_NAME -t $ROUTER_DOCKER_TAG ."
    else
        docker build -f $DOCKERFILE_NAME -t $ROUTER_DOCKER_TAG .
        success "Router Docker image built: $ROUTER_DOCKER_TAG"
    fi

    cd ..

    # Step 3: Stop any existing router container
    log "Stopping any existing router container..."
    docker stop vllm-router-local 2>/dev/null || true
    docker rm vllm-router-local 2>/dev/null || true

    # Step 4: Run router container locally with correct ports
    log "Starting router container locally..."

    # Get internal IPs from all prefill and decode hosts
    log "Getting internal IP addresses from remote hosts..."
    PREFILL_ARGS=""
    DECODE_ARGS=""

    # Build --prefill arguments for each prefill host
    for host in $PREFILL_REMOTE_HOSTS; do
        INTERNAL_IP=$(ssh -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no $host "hostname -I | awk '{print \$1}'" 2>/dev/null || echo "")
        if [ -z "$INTERNAL_IP" ]; then
            error "Failed to get internal IP from prefill host: $host"
            exit 1
        fi
        log "Prefill host $host -> Internal IP: $INTERNAL_IP"
        PREFILL_ARGS="$PREFILL_ARGS --prefill http://$INTERNAL_IP:$PREFILL_PORT"
    done

    # Build --decode arguments for each decode host
    for host in $DECODE_REMOTE_HOSTS; do
        INTERNAL_IP=$(ssh -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no $host "hostname -I | awk '{print \$1}'" 2>/dev/null || echo "")
        if [ -z "$INTERNAL_IP" ]; then
            error "Failed to get internal IP from decode host: $host"
            exit 1
        fi
        log "Decode host $host -> Internal IP: $INTERNAL_IP"
        DECODE_ARGS="$DECODE_ARGS --decode http://$INTERNAL_IP:$DECODE_PORT"
    done

    if [ "$DRY_RUN" = "true" ]; then
        echo "Would run router container with:"
        echo "  Prefill args: $PREFILL_ARGS"
        echo "  Decode args: $DECODE_ARGS"
    else
        docker run -d \
            --name vllm-router-local \
            -p 10001:10001 \
            --restart unless-stopped \
            $ROUTER_DOCKER_TAG \
            vllm-router --vllm-pd-disaggregation \
            $PREFILL_ARGS \
            $DECODE_ARGS \
            --host 0.0.0.0 --port 10001 \
            $(if [ "$ENABLE_PROFILING" = "true" ]; then echo "--profile"; fi)

        # Wait a moment and check if container started successfully
        sleep 2
        if docker ps | grep -q vllm-router-local; then
            success "Router started locally successfully!"
            log "Router HTTP available at: http://localhost:10001"
            log "Router Discovery available at: localhost:30001"
            log "Check logs with: docker logs vllm-router-local"
        else
            error "Failed to start router container"
            log "Check logs with: docker logs vllm-router-local"
            exit 1
        fi
    fi
}

# Function to deploy prefill server locally
deploy_prefill_local() {
    log "Deploying vLLM prefill server locally..."

    # Get router container IP for local deployment
    local ROUTER_IP=$(get_router_container_ip)
    log "Using router IP: $ROUTER_IP"

    # Generate and execute the prefill run script locally
    if [ "$DRY_RUN" = "true" ]; then
        echo "Would deploy prefill server with router IP: $ROUTER_IP"
        generate_vllm_prefill_run_script $ROUTER_IP
    else
        generate_vllm_prefill_run_script $ROUTER_IP | bash

        sleep 3
        if docker ps | grep -q vllm-deepseek-prefill; then
            success "vLLM prefill server started successfully"
            log "Check logs with: docker logs vllm-deepseek-prefill"
            log "Prefill API available at: http://localhost:$PREFILL_PORT"
        else
            error "Failed to start vLLM prefill container"
            log "Check logs with: docker logs vllm-deepseek-prefill"
            exit 1
        fi
    fi
}

# Function to deploy decode server locally
deploy_decode_local() {
    log "Deploying vLLM decode server locally..."

    # Get router container IP for local deployment
    local ROUTER_IP=$(get_router_container_ip)
    log "Using router IP: $ROUTER_IP"

    # Generate and execute the decode run script locally
    if [ "$DRY_RUN" = "true" ]; then
        echo "Would deploy decode server with router IP: $ROUTER_IP"
        generate_vllm_run_script $ROUTER_IP
    else
        generate_vllm_run_script $ROUTER_IP | bash

        sleep 3
        if docker ps | grep -q vllm-deepseek; then
            success "vLLM decode server started successfully"
            log "Check logs with: docker logs vllm-deepseek"
            log "Decode API available at: http://localhost:$DECODE_PORT"
        else
            error "Failed to start vLLM decode container"
            log "Check logs with: docker logs vllm-deepseek"
            exit 1
        fi
    fi
}

# Function to run benchmark against the router
benchmark() {
    log "Running vLLM benchmark against the router..."

    # Check if there's a vLLM container to run the benchmark from
    local BENCHMARK_CONTAINER=""
    if docker ps | grep -q vllm-deepseek-prefill; then
        BENCHMARK_CONTAINER="vllm-deepseek-prefill"
    elif docker ps | grep -q vllm-deepseek; then
        BENCHMARK_CONTAINER="vllm-deepseek"
    else
        error "No vLLM container found to run benchmark from"
        log "Please deploy a vLLM service first with: $0 deploy_prefill_local or $0 deploy_decode_local"
        exit 1
    fi

    log "Using container: $BENCHMARK_CONTAINER"

    # Default benchmark parameters
    local NUM_PROMPTS="${BENCHMARK_NUM_PROMPTS:-10000}"
    local INPUT_LEN="${BENCHMARK_INPUT_LEN:-2000}"
    local OUTPUT_LEN="${BENCHMARK_OUTPUT_LEN:-2000}"
    local MAX_CONCURRENCY="${BENCHMARK_MAX_CONCURRENCY:-32}"
    local ROUTER_HOST="${BENCHMARK_ROUTER_HOST:-host.docker.internal}"
    local ROUTER_PORT="${BENCHMARK_ROUTER_PORT:-10001}"

    log "Benchmark configuration:"
    echo "  Container: $BENCHMARK_CONTAINER"
    echo "  Model: $MODEL"
    echo "  Router: $ROUTER_HOST:$ROUTER_PORT"
    echo "  Prompts: $NUM_PROMPTS"
    echo "  Input Length: $INPUT_LEN tokens"
    echo "  Output Length: $OUTPUT_LEN tokens"
    echo "  Max Concurrency: $MAX_CONCURRENCY"
    echo ""

    if [ "$DRY_RUN" = "true" ]; then
        echo "Would run benchmark command in container $BENCHMARK_CONTAINER"
    else
        # Run the benchmark
        log "Starting benchmark (this may take a while)..."
        docker exec $BENCHMARK_CONTAINER vllm bench serve \
            --dataset-name random \
            --num-prompts $NUM_PROMPTS \
            --model "$MODEL" \
            --random-input-len $INPUT_LEN \
            --random-output-len $OUTPUT_LEN \
            --endpoint /v1/completions \
            --max-concurrency $MAX_CONCURRENCY \
            --save-result \
            --ignore-eos \
            --served-model-name "$MODEL" \
            --host $ROUTER_HOST \
            --port $ROUTER_PORT

        if [ $? -eq 0 ]; then
            success "Benchmark completed successfully!"
            log "Results saved to benchmark output file"
        else
            error "Benchmark failed or was interrupted"
            exit 1
        fi
    fi
}

# Function to verify environment setup
verify_environment() {
    log "Verifying environment setup..."

    # Verify HF_TOKEN is set
    if [ -z "$HF_TOKEN" ]; then
        error "HF_TOKEN is not set. This should not happen - validation failed!"
        exit 1
    fi
    success "HF_TOKEN: Configured"

    # Verify SSH key if needed for remote operations
    local need_ssh=false
    for cmd in "${COMMANDS[@]}"; do
        case $cmd in
            deploy_remote_decode|deploy_remote_prefill|deploy_router|all)
                need_ssh=true
                ;;
        esac
    done

    if [ "$need_ssh" = true ]; then
        if [ -z "$SSH_KEY_FILE" ]; then
            error "SSH_KEY_FILE is not set. This should not happen - validation failed!"
            exit 1
        fi

        if [ ! -f "$SSH_KEY_FILE" ]; then
            error "SSH key file not found: $SSH_KEY_FILE"
            exit 1
        fi

        # Test SSH connection to the first remote host
        local first_host=$(echo $PREFILL_REMOTE_HOSTS | awk '{print $1}')
        if [ -n "$first_host" ]; then
            log "Testing SSH connection to $first_host..."
            if ssh -i "$SSH_KEY_FILE" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$first_host" "echo 'SSH connection successful'" 2>/dev/null; then
                success "SSH connection: Verified"
            else
                warn "SSH connection test failed to $first_host"
                warn "Proceeding anyway - deployment may fail if SSH is not properly configured"
            fi
        fi

        # Add all EC2 hosts to known_hosts
        log "Adding EC2 hosts to known_hosts..."
        for host_spec in $PREFILL_REMOTE_HOSTS $DECODE_REMOTE_HOSTS; do
            # Extract just the hostname part (remove username@)
            local hostname=$(echo "$host_spec" | sed 's/.*@//')
            ssh-keyscan -H "$hostname" >> ~/.ssh/known_hosts 2>/dev/null
        done
    fi

    success "Environment verification completed"
}

# Parse command line arguments
USE_PRECOMPILED="true"
DRY_RUN="false"
COMMANDS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -c|--commit)
            VLLM_COMMIT="$2"
            shift 2
            ;;
        -t|--tag)
            ECR_TAG="$2"
            shift 2
            ;;
        -r|--hosts)
            REMOTE_HOSTS="$2"
            shift 2
            ;;
        --no-precompiled)
            USE_PRECOMPILED="false"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        clone|build|upload|deploy_local|deploy_remote_decode|deploy_remote_prefill|deploy_router|deploy_router_local|deploy_prefill_local|deploy_decode_local|benchmark|setup_docker_storage|install_gdrcopy|all)
            COMMANDS+=("$1")
            shift
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# If no commands specified, show usage
if [ ${#COMMANDS[@]} -eq 0 ]; then
    usage
    exit 1
fi

# Validate required environment variables
validate_env

# Show configuration
log "Configuration:"
echo "  vLLM Commit: $VLLM_COMMIT"
echo "  Docker Tag: $DOCKER_TAG"
echo "  ECR Tag: $ECR_TAG"
echo "  Use Precompiled: $USE_PRECOMPILED"
echo "  Prefill Hosts: $PREFILL_REMOTE_HOSTS"
echo "  Decode Hosts: $DECODE_REMOTE_HOSTS"
echo "  All Remote Hosts: $REMOTE_HOSTS"
echo "  Model: $MODEL"
echo "  Dry Run: $DRY_RUN"
echo ""

# Execute commands
for cmd in "${COMMANDS[@]}"; do
    case $cmd in
        clone)
            clone_vllm
            ;;
        build)
            build_docker
            ;;
        upload)
            upload_ecr
            ;;
        deploy_local)
            deploy_local
            ;;
        deploy_remote_decode)
            deploy_remote_decode
            ;;
        deploy_remote_prefill)
            deploy_remote_prefill
            ;;
        deploy_router)
            deploy_router
            ;;
        deploy_router_local)
            deploy_router_local
            ;;
        deploy_prefill_local)
            deploy_prefill_local
            ;;
        deploy_decode_local)
            deploy_decode_local
            ;;
        benchmark)
            benchmark
            ;;
        setup_ssh)
            setup_ssh
            ;;
        setup_docker_storage)
            setup_docker_storage
            ;;
        install_gdrcopy)
            install_gdrcopy
            ;;
        all)
            log "Running full deployment pipeline..."
            verify_environment
            clone_vllm
            build_docker
            upload_ecr
            deploy_router_local
            deploy_remote_prefill
            deploy_remote_decode
            ;;
    esac
done

success "All operations completed successfully!"
