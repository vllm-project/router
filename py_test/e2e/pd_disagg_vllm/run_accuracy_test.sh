#!/bin/bash
# P/D Disaggregation Accuracy Test using vLLM NixlConnector
# This script launches vLLM prefill and decode instances natively (no Docker)
# and validates routing accuracy through the router

set -xe

# =============================================================================
# Configuration Variables
# =============================================================================

# KV buffer device
KV_BUFFER_DEVICE=${KV_BUFFER_DEVICE:-"cuda"}

# KV layout configuration
DECODER_KV_LAYOUT=${DECODER_KV_LAYOUT:-"HND"} # Default to HND, optional NHD
if [[ "$DECODER_KV_LAYOUT" == "NHD" ]]; then
  KV_CONFIG_HETERO_LAYOUT=',"enable_permute_local_kv":"True"'
else
  KV_CONFIG_HETERO_LAYOUT=''
fi

# Build the kv-transfer-config
if [[ "$KV_BUFFER_DEVICE" == "cuda" ]]; then
  KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both"'${KV_CONFIG_HETERO_LAYOUT}'}'
else
  KV_CONFIG="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"$KV_BUFFER_DEVICE\""${KV_CONFIG_HETERO_LAYOUT}"}"
fi

# Models to test
MODEL_NAMES=${MODEL_NAMES:-"meta-llama/Llama-3.2-1B-Instruct"}

# Instance configuration
NUM_PREFILL_INSTANCES=${NUM_PREFILL_INSTANCES:-1}
NUM_DECODE_INSTANCES=${NUM_DECODE_INSTANCES:-1}
PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:-1}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
PREFILL_BLOCK_SIZE=${PREFILL_BLOCK_SIZE:-128}
DECODE_BLOCK_SIZE=${DECODE_BLOCK_SIZE:-128}

# Intra-node data parallel size (exercises DP-aware routing)
INTRA_NODE_DP_SIZE=${INTRA_NODE_DP_SIZE:-2}

# Port configuration
PREFILL_BASE_PORT=${PREFILL_BASE_PORT:-8100}
DECODE_BASE_PORT=${DECODE_BASE_PORT:-8200}
ROUTER_PORT=${ROUTER_PORT:-8300}

# NIXL side channel port configuration
PREFILL_NIXL_BASE_PORT=${PREFILL_NIXL_BASE_PORT:-9100}
DECODE_NIXL_BASE_PORT=${DECODE_NIXL_BASE_PORT:-9200}

# NIXL HTTP port configuration for kv_connector_extra_config
PREFILL_NIXL_HTTP_BASE_PORT=${PREFILL_NIXL_HTTP_BASE_PORT:-8097}
DECODE_NIXL_HTTP_BASE_PORT=${DECODE_NIXL_HTTP_BASE_PORT:-8098}

# Find script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect number of GPUs
SMI_BIN=$(which nvidia-smi || which rocm-smi || echo "")
get_num_gpus() {
  if [[ "$SMI_BIN" == *"nvidia"* ]]; then
    echo "$($SMI_BIN --query-gpu=name --format=csv,noheader | wc -l)"
  elif [[ "$SMI_BIN" == *"rocm"* ]]; then
    echo "$($SMI_BIN -l | grep GPU | wc -l)"
  else
    echo "1"
  fi
}

# =============================================================================
# Cleanup Functions
# =============================================================================

cleanup_instances() {
  echo "=== Cleaning up vLLM instances ==="
  pkill -f "vllm serve" || true
  if [[ -n "${ROUTER_PID:-}" ]]; then
    echo "=== Cleaning up router ==="
    kill $ROUTER_PID 2>/dev/null || true
  fi
  sleep 2
}

# Trap signals for cleanup
trap cleanup_instances EXIT SIGINT SIGTERM

# =============================================================================
# Helper Functions
# =============================================================================

wait_for_server() {
  local port=$1
  local max_timeout=${2:-300}   # 5 minutes maximum
  echo "Waiting for server on port ${port} (max: ${max_timeout}s)..."

  local start_time=$(date +%s)
  while true; do
    # Check if server is ready using the /health endpoint
    if curl -s -f "http://localhost:${port}/health" > /dev/null 2>&1; then
      echo "Server on port ${port} is ready!"
      return 0
    fi

    local elapsed=$(($(date +%s) - start_time))
    if [[ $elapsed -ge $max_timeout ]]; then
      echo "ERROR: Server on port ${port} failed to start within ${max_timeout}s"
      return 1
    fi

    sleep 5
  done
}

wait_for_router() {
  local port=$1
  local max_timeout=${2:-60}
  echo "Waiting for router on port ${port} (max: ${max_timeout}s)..."

  local start_time=$(date +%s)
  while true; do
    if curl -s "http://localhost:${port}/health" | grep -q "ok"; then
      echo "Router on port ${port} is ready!"
      return 0
    fi

    local elapsed=$(($(date +%s) - start_time))
    if [[ $elapsed -ge $max_timeout ]]; then
      echo "ERROR: Router on port ${port} failed to start within ${max_timeout}s"
      return 1
    fi

    sleep 1
  done
}

# =============================================================================
# Launch vLLM Instances
# =============================================================================

# Cleanup any existing instances first
echo "=== Cleaning up any existing vLLM instances ==="
cleanup_instances

echo "=== Launching ${NUM_PREFILL_INSTANCES} Prefill Instance(s) ==="

PREFILL_URLS=()
PREFILL_PORTS=()

for i in $(seq 0 $((NUM_PREFILL_INSTANCES - 1))); do
  PORT=$((PREFILL_BASE_PORT + i))
  NIXL_PORT=$((PREFILL_NIXL_BASE_PORT + i))
  NIXL_HTTP_PORT=$((PREFILL_NIXL_HTTP_BASE_PORT + i))

  # Calculate GPU IDs for this instance (TP * DP GPUs per instance)
  PREFILL_GPUS_PER_INSTANCE=$((PREFILLER_TP_SIZE * INTRA_NODE_DP_SIZE))
  GPU_START=$((i * PREFILL_GPUS_PER_INSTANCE))
  GPU_END=$((GPU_START + PREFILL_GPUS_PER_INSTANCE - 1))
  GPU_IDS=$(seq -s, $GPU_START $GPU_END)

  # Build instance-specific KV config with HTTP port
  if [[ "$KV_BUFFER_DEVICE" == "cuda" ]]; then
    INSTANCE_KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"backends":["UCX","GDS"],"http_port":'${NIXL_HTTP_PORT}'}'${KV_CONFIG_HETERO_LAYOUT}'}'
  else
    INSTANCE_KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"'${KV_BUFFER_DEVICE}'","kv_connector_extra_config":{"backends":["UCX","GDS"],"http_port":'${NIXL_HTTP_PORT}'}'${KV_CONFIG_HETERO_LAYOUT}'}'
  fi

  echo "Launching Prefill Instance ${i} on port ${PORT} (GPUs: ${GPU_IDS}, NIXL port: ${NIXL_PORT}, HTTP: ${NIXL_HTTP_PORT})"

  CUDA_VISIBLE_DEVICES=$GPU_IDS \
  VLLM_USE_V1=1 \
  VLLM_LOGGING_LEVEL=DEBUG \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  VLLM_NIXL_SIDE_CHANNEL_HOST=0.0.0.0 \
  VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_PORT \
  UCX_TLS=all \
  UCX_NET_DEVICES=all \
  vllm serve "$MODEL_NAMES" \
    --port $PORT \
    --block-size ${PREFILL_BLOCK_SIZE} \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --data-parallel-size $INTRA_NODE_DP_SIZE \
    --enable-prefix-caching \
    --enforce-eager \
    --disable-hybrid-kv-cache-manager \
    --disable-log-stats \
    --kv-transfer-config "$INSTANCE_KV_CONFIG" \
    > /tmp/prefill_${i}.log 2>&1 &

  PREFILL_URLS+=("http://localhost:${PORT}")
  PREFILL_PORTS+=($PORT)
done

echo "=== Launching ${NUM_DECODE_INSTANCES} Decode Instance(s) ==="

DECODE_URLS=()
DECODE_PORTS=()

for i in $(seq 0 $((NUM_DECODE_INSTANCES - 1))); do
  PORT=$((DECODE_BASE_PORT + i))
  NIXL_PORT=$((DECODE_NIXL_BASE_PORT + i))
  NIXL_HTTP_PORT=$((DECODE_NIXL_HTTP_BASE_PORT + i))

  # Calculate GPU IDs for decode instances (after prefill GPUs)
  PREFILL_TOTAL_GPUS=$((NUM_PREFILL_INSTANCES * PREFILLER_TP_SIZE * INTRA_NODE_DP_SIZE))
  DECODE_GPUS_PER_INSTANCE=$((DECODER_TP_SIZE * INTRA_NODE_DP_SIZE))
  GPU_START=$(( PREFILL_TOTAL_GPUS + (i * DECODE_GPUS_PER_INSTANCE) ))
  GPU_END=$((GPU_START + DECODE_GPUS_PER_INSTANCE - 1))
  GPU_IDS=$(seq -s, $GPU_START $GPU_END)

  # Build instance-specific KV config with HTTP port
  if [[ "$KV_BUFFER_DEVICE" == "cuda" ]]; then
    INSTANCE_KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"backends":["UCX","GDS"],"http_port":'${NIXL_HTTP_PORT}'}'${KV_CONFIG_HETERO_LAYOUT}'}'
  else
    INSTANCE_KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"'${KV_BUFFER_DEVICE}'","kv_connector_extra_config":{"backends":["UCX","GDS"],"http_port":'${NIXL_HTTP_PORT}'}'${KV_CONFIG_HETERO_LAYOUT}'}'
  fi

  echo "Launching Decode Instance ${i} on port ${PORT} (GPUs: ${GPU_IDS}, NIXL port: ${NIXL_PORT}, HTTP: ${NIXL_HTTP_PORT})"

  CUDA_VISIBLE_DEVICES=$GPU_IDS \
  VLLM_USE_V1=1 \
  VLLM_LOGGING_LEVEL=DEBUG \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  VLLM_NIXL_SIDE_CHANNEL_HOST=0.0.0.0 \
  VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_PORT \
  UCX_TLS=all \
  UCX_NET_DEVICES=all \
  vllm serve "$MODEL_NAMES" \
    --port $PORT \
    --block-size ${DECODE_BLOCK_SIZE} \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --data-parallel-size $INTRA_NODE_DP_SIZE \
    --disable-hybrid-kv-cache-manager \
    --disable-log-stats \
    --kv-transfer-config "$INSTANCE_KV_CONFIG" \
    > /tmp/decode_${i}.log 2>&1 &

  DECODE_URLS+=("http://localhost:${PORT}")
  DECODE_PORTS+=($PORT)
done

# =============================================================================
# Build Router (while vLLM is starting up)
# =============================================================================

echo "=== Building Router (parallel with vLLM startup) ==="

# Check if vllm-router is already available
if ! command -v vllm-router &> /dev/null; then
  echo "vllm-router not found, building from source..."

  # Install build dependencies
  if ! command -v pkg-config &> /dev/null; then
    echo "Installing build dependencies (pkg-config, libssl-dev)..."
    apt-get update && apt-get install -y pkg-config libssl-dev
  fi

  # Install Rust if not available
  if ! command -v cargo &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source /root/.cargo/env
  fi

  # Build router from repository root
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
  pushd "$REPO_ROOT" > /dev/null
  cargo build --release
  export PATH="${REPO_ROOT}/target/release:$PATH"
  popd > /dev/null
fi

# Verify router is available
vllm-router --version

# =============================================================================
# Wait for vLLM Instances
# =============================================================================

echo "=== Waiting for Prefill Instances ==="
for PORT in "${PREFILL_PORTS[@]}"; do
  if ! wait_for_server "$PORT"; then
    echo "ERROR: Prefill instance on port ${PORT} failed to start"
    echo "=== Prefill logs ==="
    cat /tmp/prefill_*.log 2>&1 || true
    exit 1
  fi
  echo "✓ Prefill instance on port ${PORT} started successfully"
done

echo "=== Waiting for Decode Instances ==="
for PORT in "${DECODE_PORTS[@]}"; do
  if ! wait_for_server "$PORT"; then
    echo "ERROR: Decode instance on port ${PORT} failed to start"
    echo "=== Decode logs ==="
    cat /tmp/decode_*.log 2>&1 || true
    exit 1
  fi
  echo "✓ Decode instance on port ${PORT} started successfully"
done

# =============================================================================
# Launch Router
# =============================================================================

echo "=== Launching Router on port ${ROUTER_PORT} ==="

# Build prefill URLs argument
PREFILL_ARGS=""
for url in "${PREFILL_URLS[@]}"; do
  PREFILL_ARGS="${PREFILL_ARGS} --prefill ${url}"
done

# Build decode URLs argument
DECODE_ARGS=""
for url in "${DECODE_URLS[@]}"; do
  DECODE_ARGS="${DECODE_ARGS} --decode ${url}"
done

# Launch router in background
vllm-router \
  --port "$ROUTER_PORT" \
  --policy power_of_two \
  --vllm-pd-disaggregation \
  --intra-node-data-parallel-size "$INTRA_NODE_DP_SIZE" \
  $PREFILL_ARGS \
  $DECODE_ARGS \
  --worker-startup-check-interval 1 \
  > /tmp/router.log 2>&1 &

ROUTER_PID=$!

echo "Waiting 10 seconds for router to initialize..."
sleep 10

echo "✓ Router started"

# =============================================================================
# Run Accuracy Tests
# =============================================================================

echo "=== Running Router P/D Disaggregation Sanity Test ==="

# Test basic completion through router
python3 "${SCRIPT_DIR}/test_pd_accuracy.py" \
  --router-url "http://localhost:${ROUTER_PORT}" \
  --model "$MODEL_NAMES" \
  --num-requests 20 \
  --skip-streaming

SANITY_EXIT_CODE=$?

if [ $SANITY_EXIT_CODE -ne 0 ]; then
  echo "❌ Router P/D disaggregation sanity test FAILED"
  TEST_EXIT_CODE=$SANITY_EXIT_CODE
else
  echo "✅ Router P/D disaggregation sanity test PASSED"
  echo ""
  echo "=== Running LM-Eval Accuracy Test ==="

  # Run LM-Eval benchmark to validate accuracy through P/D disaggregation
  python3 "${SCRIPT_DIR}/test_lm_eval_accuracy.py" \
    --router-url "http://localhost:${ROUTER_PORT}" \
    --model "$MODEL_NAMES" \
    --num-concurrent 10

  LMEVAL_EXIT_CODE=$?

  if [ $LMEVAL_EXIT_CODE -ne 0 ]; then
    echo "❌ LM-Eval accuracy test FAILED"
    TEST_EXIT_CODE=$LMEVAL_EXIT_CODE
  else
    echo "✅ LM-Eval accuracy test PASSED"
    TEST_EXIT_CODE=0
  fi
fi

# =============================================================================
# Cleanup
# =============================================================================

echo "=== Test Results ==="
if [ $TEST_EXIT_CODE -eq 0 ]; then
  echo "✅ All P/D disaggregation accuracy tests PASSED"
else
  echo "❌ P/D disaggregation accuracy tests FAILED"
  echo "=== Prefill Logs ==="
  for i in $(seq 0 $((NUM_PREFILL_INSTANCES - 1))); do
    echo "--- Prefill Instance ${i} ---"
    cat /tmp/prefill_${i}.log 2>&1 | tail -100
  done

  echo "=== Decode Logs ==="
  for i in $(seq 0 $((NUM_DECODE_INSTANCES - 1))); do
    echo "--- Decode Instance ${i} ---"
    cat /tmp/decode_${i}.log 2>&1 | tail -100
  done

  echo "=== Router Logs ==="
  cat /tmp/router.log
fi

# Kill router
kill $ROUTER_PID 2>/dev/null || true

exit $TEST_EXIT_CODE
