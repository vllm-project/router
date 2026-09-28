#!/bin/bash

# VLLM Router Installation Script
# This script provides automated installation and setup for vllm-router
# with support for Prefill/Decode disaggregation setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration variables
INSTALL_RUST=${INSTALL_RUST:-true}
BUILD_RELEASE=${BUILD_RELEASE:-true}
INSTALL_PYTHON_DEPS=${INSTALL_PYTHON_DEPS:-true}
SETUP_VENV=${SETUP_VENV:-false}
VENV_NAME=${VENV_NAME:-"vllm-router-env"}

get_local_ip() {
    # Try to get the primary interface IP (not localhost)
    local ip=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'src \K\S+' | head -1)
    if [[ -z "$ip" ]]; then
        # Fallback to hostname -I
        ip=$(hostname -I | awk '{print $1}')
    fi
    if [[ -z "$ip" ]]; then
        # Last resort fallback
        ip="0.0.0.0"
    fi
    echo "$ip"
}

# Router configuration for P/D disaggregation
ROUTER_HOST=${ROUTER_HOST:-$(get_local_ip)}
ROUTER_HTTP_PORT=${ROUTER_HTTP_PORT:-"10001"}
ROUTER_DISCOVERY_PORT=${ROUTER_DISCOVERY_PORT:-"30001"}

print_header() {
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}    VLLM Router Installation${NC}"
    echo -e "${BLUE}=====================================${NC}"
    echo
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

check_requirements() {
    print_step "Checking system requirements..."

    # Check if we're in the right directory
    if [[ ! -f "Cargo.toml" ]]; then
        print_error "This script must be run from the vllm-router root directory"
        exit 1
    fi

    print_success "System requirements check passed"
}

install_system_deps() {
    print_step "Installing system dependencies..."
    print_success "No additional system dependencies required"
}

install_rust() {
    if [[ "$INSTALL_RUST" == "true" ]]; then
        print_step "Installing Rust..."

        if command -v rustc &> /dev/null; then
            print_info "Rust is already installed ($(rustc --version))"
        else
            print_info "Installing Rust via rustup..."
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source "$HOME/.cargo/env"
            print_success "Rust installed successfully"
        fi

        # Verify installation
        if command -v cargo &> /dev/null; then
            print_info "Cargo version: $(cargo --version)"
        else
            print_error "Rust installation failed"
            exit 1
        fi
    else
        print_info "Skipping Rust installation (INSTALL_RUST=false)"
    fi
}

setup_python_env() {
    if [[ "$SETUP_VENV" == "true" ]]; then
        print_step "Setting up Python virtual environment..."

        if [[ -d "$VENV_NAME" ]]; then
            print_info "Virtual environment '$VENV_NAME' already exists"
        else
            python3 -m venv "$VENV_NAME"
            print_success "Created virtual environment '$VENV_NAME'"
        fi

        print_info "Activating virtual environment..."
        source "$VENV_NAME/bin/activate"

        # Upgrade pip
        python -m pip install --upgrade pip
    fi
}

install_python_dependencies() {
    if [[ "$INSTALL_PYTHON_DEPS" == "true" ]]; then
        print_step "Installing Python dependencies..."

        # Install the package in development mode
        if [[ "$SETUP_VENV" == "true" ]]; then
            source "$VENV_NAME/bin/activate"
        fi

        python3 -m pip install -e ".[dev]"
        print_success "Python dependencies installed"
    else
        print_info "Skipping Python dependencies installation (INSTALL_PYTHON_DEPS=false)"
    fi
}

build_rust_components() {
    if [[ "$BUILD_RELEASE" == "true" ]]; then
        print_step "Building Rust components..."

        # Ensure cargo is in PATH
        if [[ -f "$HOME/.cargo/env" ]]; then
            source "$HOME/.cargo/env"
        fi

        print_info "Building release version..."
        cargo build --release

        print_success "Rust components built successfully"
        print_info "Binary location: ./target/release/vllm-router"
    else
        print_info "Skipping Rust build (BUILD_RELEASE=false)"
    fi
}

generate_router_command() {
    echo "cargo run --release -- \\"
    echo "    --vllm-pd-disaggregation \\"
    echo "    --vllm-discovery-address ${ROUTER_HOST}:${ROUTER_DISCOVERY_PORT} \\"
    echo "    --host ${ROUTER_HOST} \\"
    echo "    --port ${ROUTER_HTTP_PORT} \\"
    echo "    --prefill-policy consistent_hash \\"
    echo "    --decode-policy consistent_hash"
}

show_usage_examples() {
    print_step "Installation complete! Here are some usage examples:"
    echo

    print_info "1. Standard Data Parallelism Routing:"
    echo "   ./target/release/vllm-router \\"
    echo "       --worker-urls http://0.0.0.0:8000 \\"
    echo "       --policy consistent_hash \\"
    echo "       --intra-node-data-parallel-size 8"
    echo

    print_info "2. Using cargo run (development):"
    echo "   cargo run --release -- \\"
    echo "       --worker-urls http://0.0.0.0:8000 \\"
    echo "       --policy consistent_hash \\"
    echo "       --intra-node-data-parallel-size 8"
    echo

    print_info "3. Python module usage:"
    echo "   python -m vllm_router.launch_router \\"
    echo "       --worker-urls http://localhost:8080 http://localhost:8081 \\"
    echo "       --policy consistent_hash"
    echo

    if [[ "$SETUP_VENV" == "true" ]]; then
        print_warning "Remember to activate your virtual environment:"
        echo "   source $VENV_NAME/bin/activate"
        echo
    fi

    print_info "For more configuration options, see README.md or run:"
    echo "   ./target/release/vllm-router --help"
}

show_prefill_decode_setup() {
    print_step "Prefill/Decode Disaggregation Setup:"
    echo

    print_info "Router Configuration:"
    echo "   Router IP: ${ROUTER_HOST}"
    echo "   Router HTTP Port: ${ROUTER_HTTP_PORT}"
    echo "   ZMQ Discovery Port: ${ROUTER_DISCOVERY_PORT}"
    echo

    print_info "Step 1: Start vLLM Router"
    echo "========================================"
    generate_router_command
    echo

    print_info "Step 2: Copy and run Prefill Server command (customize as needed):"
    echo "========================================"
    echo "# Set default values (modify as needed for your environment)"
    echo "VENV_PATH=~/uv_env/vllm && \\"
    echo "VLLM_PATH=/home/ubuntu/gitrepos/vllm && \\"
    echo "GPU_ID=0 && \\"
    echo "MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct && \\"
    echo "PREFILL_PORT=20003 && \\"
    echo "KV_PORT=21001 && \\"
    echo "source \${VENV_PATH}/bin/activate && \\"
    echo "cd \${VLLM_PATH} && \\"
    echo "CUDA_VISIBLE_DEVICES=\${GPU_ID} VLLM_USE_V1=1 vllm serve \${MODEL} \\"
    echo "    --enforce-eager \\"
    echo "    --host 0.0.0.0 \\"
    echo "    --port \${PREFILL_PORT} \\"
    echo "    --tensor-parallel-size 1 \\"
    echo "    --seed 1024 \\"
    echo "    --dtype float16 \\"
    echo "    --max-model-len 10000 \\"
    echo "    --max-num-batched-tokens 10000 \\"
    echo "    --max-num-seqs 256 \\"
    echo "    --trust-remote-code \\"
    echo "    --gpu-memory-utilization 0.9 \\"
    echo "    --kv-transfer-config '{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e1\",\"kv_port\":\"'\${KV_PORT}'\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${ROUTER_HOST}\",\"proxy_port\":\"${ROUTER_DISCOVERY_PORT}\",\"http_port\":\"'\${PREFILL_PORT}'\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}' \\"
    echo "    > prefill.log 2>&1 &"
    echo

    print_info "Step 3: Copy and run Decode Server command (customize as needed):"
    echo "========================================"
    echo "# Set default values (modify as needed for your environment)"
    echo "VENV_PATH=~/uv_env/vllm && \\"
    echo "VLLM_PATH=/home/ubuntu/gitrepos/vllm && \\"
    echo "GPU_ID=1 && \\"
    echo "MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct && \\"
    echo "DECODE_PORT=20005 && \\"
    echo "KV_PORT=22001 && \\"
    echo "source \${VENV_PATH}/bin/activate && \\"
    echo "cd \${VLLM_PATH} && \\"
    echo "CUDA_VISIBLE_DEVICES=\${GPU_ID} VLLM_USE_V1=1 vllm serve \${MODEL} \\"
    echo "    --enforce-eager \\"
    echo "    --host 0.0.0.0 \\"
    echo "    --port \${DECODE_PORT} \\"
    echo "    --tensor-parallel-size 1 \\"
    echo "    --seed 1024 \\"
    echo "    --dtype float16 \\"
    echo "    --max-model-len 10000 \\"
    echo "    --max-num-batched-tokens 10000 \\"
    echo "    --max-num-seqs 256 \\"
    echo "    --trust-remote-code \\"
    echo "    --gpu-memory-utilization 0.7 \\"
    echo "    --kv-transfer-config '{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"'\${KV_PORT}'\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${ROUTER_HOST}\",\"proxy_port\":\"${ROUTER_DISCOVERY_PORT}\",\"http_port\":\"'\${DECODE_PORT}'\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}' \\"
    echo "    > decode.log 2>&1 &"
    echo

    print_warning "IMPORTANT NOTES:"
    echo "• P/D servers register via ZMQ at ${ROUTER_HOST}:${ROUTER_DISCOVERY_PORT}"
    echo "• Commands above use default values - modify as needed for your environment"
    echo "• Service discovery is automatic - start order doesn't matter"
    echo "• Ensure network connectivity from P/D servers to router"
    echo

    print_info "Router address customization:"
    echo "Set ROUTER_HOST environment variable to override:"
    echo "   ROUTER_HOST=192.168.1.100 ./scripts/install.sh"
}

generate_pd_scripts() {
    print_step "Generating router startup script..."

    # Create scripts directory if it doesn't exist
    mkdir -p scripts/pd-setup

    # Generate router script
    cat > scripts/pd-setup/start_router.sh << EOF
#!/bin/bash
# VLLM Router startup script for Prefill/Decode disaggregation

echo "Starting VLLM Router"
echo "Router IP: ${ROUTER_HOST}"
echo "HTTP Port: ${ROUTER_HTTP_PORT}"
echo "Discovery Port: ${ROUTER_DISCOVERY_PORT}"
echo "P/D servers should use proxy_ip=${ROUTER_HOST} and proxy_port=${ROUTER_DISCOVERY_PORT}"
echo

$(generate_router_command)
EOF
    chmod +x scripts/pd-setup/start_router.sh

    # Generate README with template commands
    cat > scripts/pd-setup/README.md << EOF
# Prefill/Decode Disaggregation Setup

Router Configuration:
- Router IP: ${ROUTER_HOST}
- Router HTTP Port: ${ROUTER_HTTP_PORT}
- ZMQ Discovery Port: ${ROUTER_DISCOVERY_PORT}

## Quick Start

1. **Start Router:**
   \`\`\`bash
   ./start_router.sh
   \`\`\`

2. **Configure P/D Servers:**
   Copy the template commands below and customize for your environment.

## Prefill Server Template

\`\`\`bash
# Customize these variables for your environment
VENV_PATH="~/uv_env/vllm"
VLLM_PATH="/home/ubuntu/gitrepos/vllm"
GPU_ID="0"
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
PREFILL_PORT="20003"
KV_PREFILL_PORT="21001"

source \${VENV_PATH}/bin/activate && \\
cd \${VLLM_PATH} && \\
CUDA_VISIBLE_DEVICES=\${GPU_ID} VLLM_USE_V1=1 vllm serve \${MODEL} \\
    --enforce-eager \\
    --host 0.0.0.0 \\
    --port \${PREFILL_PORT} \\
    --tensor-parallel-size 1 \\
    --seed 1024 \\
    --dtype float16 \\
    --max-model-len 10000 \\
    --max-num-batched-tokens 10000 \\
    --max-num-seqs 256 \\
    --trust-remote-code \\
    --gpu-memory-utilization 0.9 \\
    --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"'\${KV_PREFILL_PORT}'","kv_connector_extra_config":{"proxy_ip":"${ROUTER_HOST}","proxy_port":"${ROUTER_DISCOVERY_PORT}","http_port":"'\${PREFILL_PORT}'","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' \\
    > prefill.log 2>&1 &
\`\`\`

## Decode Server Template

\`\`\`bash
# Customize these variables for your environment
VENV_PATH="~/uv_env/vllm"
VLLM_PATH="/home/ubuntu/gitrepos/vllm"
GPU_ID="1"
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
DECODE_PORT="20005"
KV_DECODE_PORT="22001"

source \${VENV_PATH}/bin/activate && \\
cd \${VLLM_PATH} && \\
CUDA_VISIBLE_DEVICES=\${GPU_ID} VLLM_USE_V1=1 vllm serve \${MODEL} \\
    --enforce-eager \\
    --host 0.0.0.0 \\
    --port \${DECODE_PORT} \\
    --tensor-parallel-size 1 \\
    --seed 1024 \\
    --dtype float16 \\
    --max-model-len 10000 \\
    --max-num-batched-tokens 10000 \\
    --max-num-seqs 256 \\
    --trust-remote-code \\
    --gpu-memory-utilization 0.7 \\
    --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"'\${KV_DECODE_PORT}'","kv_connector_extra_config":{"proxy_ip":"${ROUTER_HOST}","proxy_port":"${ROUTER_DISCOVERY_PORT}","http_port":"'\${DECODE_PORT}'","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' \\
    > decode.log 2>&1 &
\`\`\`

## Important Notes

- **Start router first** before P/D servers
- **proxy_ip and proxy_port must match router address** (${ROUTER_HOST}:${ROUTER_DISCOVERY_PORT})
- Customize all variables marked with YOUR_* or \${VARIABLE} for your environment
- Ensure network connectivity from P/D servers to router
- Use different KV ports for each server instance

Generated with router at ${ROUTER_HOST}:${ROUTER_DISCOVERY_PORT}
EOF

    print_success "Scripts generated in scripts/pd-setup/"
    print_info "Router script: scripts/pd-setup/start_router.sh"
    print_info "Setup guide: scripts/pd-setup/README.md"
}

print_help() {
    echo "VLLM Router Installation Script"
    echo
    echo "Environment Variables:"
    echo "  INSTALL_RUST=true/false       Install Rust toolchain (default: true)"
    echo "  BUILD_RELEASE=true/false      Build release binary (default: true)"
    echo "  INSTALL_PYTHON_DEPS=true/false Install Python dependencies (default: true)"
    echo "  SETUP_VENV=true/false         Create Python virtual environment (default: false)"
    echo "  VENV_NAME=name                Virtual environment name (default: vllm-router-env)"
    echo
    echo "Router Configuration:"
    echo "  ROUTER_HOST=ip                Router IP address (default: auto-detected local IP)"
    echo "  ROUTER_HTTP_PORT=port         Router HTTP port (default: 10001)"
    echo "  ROUTER_DISCOVERY_PORT=port    ZMQ discovery port (default: 30001)"
    echo
    echo "Usage:"
    echo "  ./scripts/install.sh          Generate P/D scripts (default)"
    echo "  ./scripts/install.sh --help   Show this help"
    echo "  ./scripts/install.sh --full-install Run full installation"
    echo
    echo "Examples:"
    echo "  # Generate P/D scripts with auto-detected IP (default)"
    echo "  ./scripts/install.sh"
    echo
    echo "  # Custom router IP"
    echo "  ROUTER_HOST=192.168.1.100 ./scripts/install.sh"
    echo
    echo "  # Full installation with virtual environment"
    echo "  SETUP_VENV=true ./scripts/install.sh --full-install"
}

main() {
    # Parse command line arguments
    case "${1:-}" in
        --help|-h)
            print_help
            exit 0
            ;;
        --full-install)
            print_header
            print_info "Router IP detected/configured: ${ROUTER_HOST}"

            # Run installation steps
            check_requirements
            install_rust
            setup_python_env
            install_python_dependencies
            build_rust_components

            # Generate P/D scripts
            generate_pd_scripts

            echo
            print_success "VLLM Router installation completed successfully!"
            echo

            show_usage_examples
            echo
            show_prefill_decode_setup
            exit 0
            ;;
    esac

    # Default behavior: Install, build, run router, and show P/D setup
    print_header
    print_info "Router IP detected/configured: ${ROUTER_HOST}"

    # Run installation steps
    check_requirements
    install_system_deps
    install_rust
    build_rust_components

    # Generate P/D scripts
    generate_pd_scripts

    echo
    print_success "VLLM Router installation completed!"
    echo "Router binary available at: ./target/release/vllm-router"
    echo

    show_prefill_decode_setup
}

# Run main function
main "$@"
