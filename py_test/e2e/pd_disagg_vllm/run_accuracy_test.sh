#!/bin/bash
# P/D Disaggregation Accuracy Test
# Supports KV_CONNECTOR=nixl (default, NVIDIA/CUDA) and KV_CONNECTOR=moriio (AMD/ROCm).
#
# Usage:
#   bash run_accuracy_test.sh                          # NIXL (default)
#   KV_CONNECTOR=moriio bash run_accuracy_test.sh      # MoRI-IO

set -xe

# =============================================================================
# Connector selection
# =============================================================================

KV_CONNECTOR=${KV_CONNECTOR:-"nixl"}

if [[ "$KV_CONNECTOR" != "nixl" && "$KV_CONNECTOR" != "moriio" ]]; then
  echo "ERROR: KV_CONNECTOR must be 'nixl' or 'moriio', got '${KV_CONNECTOR}'"
  exit 1
fi

# =============================================================================
# Configuration Variables
# =============================================================================

MODEL_NAMES=${MODEL_NAMES:-"meta-llama/Llama-3.2-1B-Instruct"}

NUM_PREFILL_INSTANCES=${NUM_PREFILL_INSTANCES:-1}
NUM_DECODE_INSTANCES=${NUM_DECODE_INSTANCES:-1}
PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:-1}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}

PREFILL_BASE_PORT=${PREFILL_BASE_PORT:-8100}
DECODE_BASE_PORT=${DECODE_BASE_PORT:-8200}
ROUTER_PORT=${ROUTER_PORT:-8300}

# =============================================================================
# Connector-specific defaults
# =============================================================================

if [[ "$KV_CONNECTOR" == "moriio" ]]; then
  # MoRI-IO: block-size 1 is needed for some models (e.g. DeepSeek) but not supported
  # by all ROCm attention backends — override with PREFILL_BLOCK_SIZE/DECODE_BLOCK_SIZE if needed.
  PREFILL_BLOCK_SIZE=${PREFILL_BLOCK_SIZE:-16}
  DECODE_BLOCK_SIZE=${DECODE_BLOCK_SIZE:-16}
  INTRA_NODE_DP_SIZE=${INTRA_NODE_DP_SIZE:-1}
  GPU_DEVICE_VAR="HIP_VISIBLE_DEVICES"

  # MoRI ZMQ / side-channel ports
  PROXY_IP=${PROXY_IP:-"127.0.0.1"}
  PROXY_PING_PORT=${PROXY_PING_PORT:-36367}
  PREFILL_HANDSHAKE_BASE_PORT=${PREFILL_HANDSHAKE_BASE_PORT:-6301}
  DECODE_HANDSHAKE_BASE_PORT=${DECODE_HANDSHAKE_BASE_PORT:-6401}
  PREFILL_NOTIFY_BASE_PORT=${PREFILL_NOTIFY_BASE_PORT:-61005}
  DECODE_NOTIFY_BASE_PORT=${DECODE_NOTIFY_BASE_PORT:-61105}
else
  # NIXL: larger block sizes; intra-node DP exercises DP-aware routing
  PREFILL_BLOCK_SIZE=${PREFILL_BLOCK_SIZE:-128}
  DECODE_BLOCK_SIZE=${DECODE_BLOCK_SIZE:-128}
  INTRA_NODE_DP_SIZE=${INTRA_NODE_DP_SIZE:-2}
  GPU_DEVICE_VAR="CUDA_VISIBLE_DEVICES"

  # NIXL side-channel ports
  PREFILL_NIXL_BASE_PORT=${PREFILL_NIXL_BASE_PORT:-9100}
  DECODE_NIXL_BASE_PORT=${DECODE_NIXL_BASE_PORT:-9200}
  PREFILL_NIXL_HTTP_BASE_PORT=${PREFILL_NIXL_HTTP_BASE_PORT:-8097}
  DECODE_NIXL_HTTP_BASE_PORT=${DECODE_NIXL_HTTP_BASE_PORT:-8098}

  # KV buffer / layout (NIXL only)
  KV_BUFFER_DEVICE=${KV_BUFFER_DEVICE:-"cuda"}
  DECODER_KV_LAYOUT=${DECODER_KV_LAYOUT:-"HND"}
  if [[ "$DECODER_KV_LAYOUT" == "NHD" ]]; then
    KV_CONFIG_HETERO_LAYOUT=',"enable_permute_local_kv":"True"'
  else
    KV_CONFIG_HETERO_LAYOUT=''
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
# Connector-specific KV config builders
# =============================================================================

build_prefill_kv_config() {
  local port=$1
  if [[ "$KV_CONNECTOR" == "moriio" ]]; then
    local handshake_port=$2
    local notify_port=$3
    echo "{\"kv_connector\":\"MoRIIOConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${PROXY_IP}\",\"proxy_ping_port\":\"${PROXY_PING_PORT}\",\"http_port\":\"${port}\",\"handshake_port\":\"${handshake_port}\",\"notify_port\":\"${notify_port}\"}}"
  else
    local nixl_http_port=$2
    if [[ "$KV_BUFFER_DEVICE" == "cuda" ]]; then
      echo "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"backends\":[\"UCX\",\"GDS\"],\"http_port\":${nixl_http_port}}${KV_CONFIG_HETERO_LAYOUT}}"
    else
      echo "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"${KV_BUFFER_DEVICE}\",\"kv_connector_extra_config\":{\"backends\":[\"UCX\",\"GDS\"],\"http_port\":${nixl_http_port}}${KV_CONFIG_HETERO_LAYOUT}}"
    fi
  fi
}

build_decode_kv_config() {
  local port=$1
  if [[ "$KV_CONNECTOR" == "moriio" ]]; then
    local handshake_port=$2
    local notify_port=$3
    echo "{\"kv_connector\":\"MoRIIOConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${PROXY_IP}\",\"proxy_ping_port\":\"${PROXY_PING_PORT}\",\"http_port\":\"${port}\",\"handshake_port\":\"${handshake_port}\",\"notify_port\":\"${notify_port}\"}}"
  else
    local nixl_http_port=$2
    if [[ "$KV_BUFFER_DEVICE" == "cuda" ]]; then
      echo "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"backends\":[\"UCX\",\"GDS\"],\"http_port\":${nixl_http_port}}${KV_CONFIG_HETERO_LAYOUT}}"
    else
      echo "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"${KV_BUFFER_DEVICE}\",\"kv_connector_extra_config\":{\"backends\":[\"UCX\",\"GDS\"],\"http_port\":${nixl_http_port}}${KV_CONFIG_HETERO_LAYOUT}}"
    fi
  fi
}

# =============================================================================
# Cleanup
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

trap cleanup_instances EXIT SIGINT SIGTERM

# =============================================================================
# Health check helpers
# =============================================================================

wait_for_server() {
  local port=$1
  local max_timeout=${2:-300}
  echo "Waiting for server on port ${port} (max: ${max_timeout}s)..."

  local start_time=$(date +%s)
  while true; do
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

# For ZMQ discovery mode (MoRI-IO): the health endpoint doesn't return "ok" until
# after the first periodic worker health check fires (up to 60s after startup).
# We only need to confirm the HTTP server is bound before proceeding.
wait_for_router_up() {
  local port=$1
  local max_timeout=${2:-30}
  echo "Waiting for router HTTP server on port ${port} (max: ${max_timeout}s)..."

  local start_time=$(date +%s)
  while true; do
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/health" 2>/dev/null || echo "000")
    if [[ "$http_code" != "000" ]]; then
      echo "Router on port ${port} is accepting connections (HTTP ${http_code})"
      return 0
    fi

    local elapsed=$(($(date +%s) - start_time))
    if [[ $elapsed -ge $max_timeout ]]; then
      echo "ERROR: Router on port ${port} not accessible within ${max_timeout}s"
      return 1
    fi

    sleep 1
  done
}

# =============================================================================
# Clean up existing instances
# =============================================================================

echo "=== Cleaning up any existing vLLM instances ==="
cleanup_instances

# =============================================================================
# Launch Prefill Instances
# =============================================================================

echo "=== Launching ${NUM_PREFILL_INSTANCES} Prefill Instance(s) [connector: ${KV_CONNECTOR}] ==="

PREFILL_URLS=()
PREFILL_PORTS=()

for i in $(seq 0 $((NUM_PREFILL_INSTANCES - 1))); do
  PORT=$((PREFILL_BASE_PORT + i))

  PREFILL_GPUS_PER_INSTANCE=$((PREFILLER_TP_SIZE * INTRA_NODE_DP_SIZE))
  GPU_START=$((i * PREFILL_GPUS_PER_INSTANCE))
  GPU_END=$((GPU_START + PREFILL_GPUS_PER_INSTANCE - 1))
  GPU_IDS=$(seq -s, $GPU_START $GPU_END)

  if [[ "$KV_CONNECTOR" == "moriio" ]]; then
    HANDSHAKE_PORT=$((PREFILL_HANDSHAKE_BASE_PORT + i))
    NOTIFY_PORT=$((PREFILL_NOTIFY_BASE_PORT + i))
    INSTANCE_KV_CONFIG=$(build_prefill_kv_config $PORT $HANDSHAKE_PORT $NOTIFY_PORT)
    CONNECTOR_EXTRA_ARGS=()
    CONNECTOR_ENV="${GPU_DEVICE_VAR}=${GPU_IDS} VLLM_MORIIO_CONNECTOR_READ_MODE=1 MORI_IO_ENABLE_NOTIFICATION=0 VLLM_ENGINE_READY_TIMEOUT_S=3600"
  else
    NIXL_PORT=$((PREFILL_NIXL_BASE_PORT + i))
    NIXL_HTTP_PORT=$((PREFILL_NIXL_HTTP_BASE_PORT + i))
    INSTANCE_KV_CONFIG=$(build_prefill_kv_config $PORT $NIXL_HTTP_PORT)
    CONNECTOR_EXTRA_ARGS=(--data-parallel-size ${INTRA_NODE_DP_SIZE} --enable-prefix-caching --disable-hybrid-kv-cache-manager --disable-log-stats)
    CONNECTOR_ENV="${GPU_DEVICE_VAR}=${GPU_IDS} VLLM_NIXL_SIDE_CHANNEL_HOST=0.0.0.0 VLLM_NIXL_SIDE_CHANNEL_PORT=${NIXL_PORT} UCX_TLS=all UCX_NET_DEVICES=all FLASHINFER_DISABLE_VERSION_CHECK=1"
  fi

  echo "Launching Prefill Instance ${i} on port ${PORT} (GPUs: ${GPU_IDS})"

  env \
    VLLM_USE_V1=1 \
    VLLM_LOGGING_LEVEL=DEBUG \
    $CONNECTOR_ENV \
  vllm serve "$MODEL_NAMES" \
    --port $PORT \
    --block-size ${PREFILL_BLOCK_SIZE} \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --tensor-parallel-size $PREFILLER_TP_SIZE \
    --enforce-eager \
    --trust-remote-code \
    "${CONNECTOR_EXTRA_ARGS[@]}" \
    --kv-transfer-config "$INSTANCE_KV_CONFIG" \
    > /tmp/prefill_${i}.log 2>&1 &

  PREFILL_URLS+=("http://localhost:${PORT}")
  PREFILL_PORTS+=($PORT)
done

# =============================================================================
# Launch Decode Instances
# =============================================================================

echo "=== Launching ${NUM_DECODE_INSTANCES} Decode Instance(s) [connector: ${KV_CONNECTOR}] ==="

DECODE_URLS=()
DECODE_PORTS=()

for i in $(seq 0 $((NUM_DECODE_INSTANCES - 1))); do
  PORT=$((DECODE_BASE_PORT + i))

  PREFILL_TOTAL_GPUS=$((NUM_PREFILL_INSTANCES * PREFILLER_TP_SIZE * INTRA_NODE_DP_SIZE))
  DECODE_GPUS_PER_INSTANCE=$((DECODER_TP_SIZE * INTRA_NODE_DP_SIZE))
  GPU_START=$(( PREFILL_TOTAL_GPUS + (i * DECODE_GPUS_PER_INSTANCE) ))
  GPU_END=$((GPU_START + DECODE_GPUS_PER_INSTANCE - 1))
  GPU_IDS=$(seq -s, $GPU_START $GPU_END)

  if [[ "$KV_CONNECTOR" == "moriio" ]]; then
    HANDSHAKE_PORT=$((DECODE_HANDSHAKE_BASE_PORT + i))
    NOTIFY_PORT=$((DECODE_NOTIFY_BASE_PORT + i))
    INSTANCE_KV_CONFIG=$(build_decode_kv_config $PORT $HANDSHAKE_PORT $NOTIFY_PORT)
    CONNECTOR_EXTRA_ARGS=(--all2all-backend mori --compilation-config '{"cudagraph_mode":"PIECEWISE"}')
    CONNECTOR_ENV="${GPU_DEVICE_VAR}=${GPU_IDS} VLLM_MORIIO_CONNECTOR_READ_MODE=1 MORI_IO_ENABLE_NOTIFICATION=0 VLLM_ENGINE_READY_TIMEOUT_S=3600"
  else
    NIXL_PORT=$((DECODE_NIXL_BASE_PORT + i))
    NIXL_HTTP_PORT=$((DECODE_NIXL_HTTP_BASE_PORT + i))
    INSTANCE_KV_CONFIG=$(build_decode_kv_config $PORT $NIXL_HTTP_PORT)
    CONNECTOR_EXTRA_ARGS=(--data-parallel-size ${INTRA_NODE_DP_SIZE} --disable-hybrid-kv-cache-manager --disable-log-stats)
    CONNECTOR_ENV="${GPU_DEVICE_VAR}=${GPU_IDS} VLLM_NIXL_SIDE_CHANNEL_HOST=0.0.0.0 VLLM_NIXL_SIDE_CHANNEL_PORT=${NIXL_PORT} UCX_TLS=all UCX_NET_DEVICES=all FLASHINFER_DISABLE_VERSION_CHECK=1"
  fi

  echo "Launching Decode Instance ${i} on port ${PORT} (GPUs: ${GPU_IDS})"

  env \
    VLLM_USE_V1=1 \
    VLLM_LOGGING_LEVEL=DEBUG \
    $CONNECTOR_ENV \
  vllm serve "$MODEL_NAMES" \
    --port $PORT \
    --block-size ${DECODE_BLOCK_SIZE} \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --tensor-parallel-size $DECODER_TP_SIZE \
    --enforce-eager \
    --trust-remote-code \
    "${CONNECTOR_EXTRA_ARGS[@]}" \
    --kv-transfer-config "$INSTANCE_KV_CONFIG" \
    > /tmp/decode_${i}.log 2>&1 &

  DECODE_URLS+=("http://localhost:${PORT}")
  DECODE_PORTS+=($PORT)
done

# =============================================================================
# Build Router (parallel with vLLM startup)
# =============================================================================

echo "=== Building Router ==="

if ! command -v vllm-router &> /dev/null; then
  if ! command -v pkg-config &> /dev/null; then
    apt-get update && apt-get install -y pkg-config libssl-dev protobuf-compiler
  fi

  if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source /root/.cargo/env
  fi

  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
  pushd "$REPO_ROOT" > /dev/null
  cargo build --release
  export PATH="${REPO_ROOT}/target/release:$PATH"
  popd > /dev/null
fi

vllm-router --version

# =============================================================================
# Wait for vLLM Instances
# =============================================================================

echo "=== Waiting for Prefill Instances ==="
for PORT in "${PREFILL_PORTS[@]}"; do
  if ! wait_for_server "$PORT"; then
    echo "ERROR: Prefill instance on port ${PORT} failed to start"
    cat /tmp/prefill_*.log 2>&1 || true
    exit 1
  fi
  echo "✓ Prefill instance on port ${PORT} started successfully"
done

echo "=== Waiting for Decode Instances ==="
for PORT in "${DECODE_PORTS[@]}"; do
  if ! wait_for_server "$PORT"; then
    echo "ERROR: Decode instance on port ${PORT} failed to start"
    cat /tmp/decode_*.log 2>&1 || true
    exit 1
  fi
  echo "✓ Decode instance on port ${PORT} started successfully"
done

# =============================================================================
# Launch Router
# MoRI-IO uses ZMQ service discovery (instances self-register).
# NIXL uses static --prefill/--decode URL flags.
# =============================================================================

echo "=== Launching Router on port ${ROUTER_PORT} ==="

if [[ "$KV_CONNECTOR" == "moriio" ]]; then
  vllm-router \
    --port "$ROUTER_PORT" \
    --policy consistent_hash \
    --prefill-policy consistent_hash \
    --decode-policy consistent_hash \
    --vllm-pd-disaggregation \
    --kv-connector moriio \
    --vllm-discovery-address "0.0.0.0:${PROXY_PING_PORT}" \
    --worker-startup-check-interval 1 \
    > /tmp/router.log 2>&1 &
else
  PREFILL_ARGS=""
  for url in "${PREFILL_URLS[@]}"; do
    PREFILL_ARGS="${PREFILL_ARGS} --prefill ${url}"
  done
  DECODE_ARGS=""
  for url in "${DECODE_URLS[@]}"; do
    DECODE_ARGS="${DECODE_ARGS} --decode ${url}"
  done

  vllm-router \
    --port "$ROUTER_PORT" \
    --policy power_of_two \
    --vllm-pd-disaggregation \
    --intra-node-data-parallel-size "$INTRA_NODE_DP_SIZE" \
    $PREFILL_ARGS \
    $DECODE_ARGS \
    --worker-startup-check-interval 1 \
    > /tmp/router.log 2>&1 &
fi

ROUTER_PID=$!

if [[ "$KV_CONNECTOR" == "moriio" ]]; then
  # With ZMQ discovery the health endpoint won't return "ok" until after the first
  # periodic worker health check (60s interval). Just confirm the HTTP server is up.
  wait_for_router_up "$ROUTER_PORT"
else
  wait_for_router "$ROUTER_PORT"
fi

# =============================================================================
# Run Accuracy Tests
# =============================================================================

echo "=== Running P/D Disaggregation Sanity Test [${KV_CONNECTOR}] ==="

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
# Results
# =============================================================================

echo "=== Test Results ==="
if [ $TEST_EXIT_CODE -eq 0 ]; then
  echo "✅ All P/D disaggregation accuracy tests PASSED [${KV_CONNECTOR}]"
else
  echo "❌ P/D disaggregation accuracy tests FAILED [${KV_CONNECTOR}]"
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

kill $ROUTER_PID 2>/dev/null || true

exit $TEST_EXIT_CODE
