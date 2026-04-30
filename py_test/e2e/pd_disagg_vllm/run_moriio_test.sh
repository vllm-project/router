#!/bin/bash
# P/D Disaggregation Accuracy Test using vLLM MoRIIOConnector (AMD ROCm)
# Launches vLLM prefill (kv_producer) and decode (kv_consumer) instances natively,
# connects them via the router using ZMQ service discovery, and validates routing.

set -xe

# =============================================================================
# Configuration Variables
# =============================================================================

# Models to test — dummy weights so no HF_TOKEN is needed for CI
MODEL_NAMES=${MODEL_NAMES:-"facebook/opt-125m"}

# Instance configuration
NUM_PREFILL_INSTANCES=${NUM_PREFILL_INSTANCES:-1}
NUM_DECODE_INSTANCES=${NUM_DECODE_INSTANCES:-1}
PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:-1}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
BLOCK_SIZE=${BLOCK_SIZE:-1}

# Port configuration
PREFILL_BASE_PORT=${PREFILL_BASE_PORT:-8100}
DECODE_BASE_PORT=${DECODE_BASE_PORT:-8200}
ROUTER_PORT=${ROUTER_PORT:-8300}

# MoRI-IO proxy / side-channel configuration
PROXY_IP=${PROXY_IP:-"127.0.0.1"}
PROXY_PING_PORT=${PROXY_PING_PORT:-36367}
PREFILL_HANDSHAKE_BASE_PORT=${PREFILL_HANDSHAKE_BASE_PORT:-6301}
DECODE_HANDSHAKE_BASE_PORT=${DECODE_HANDSHAKE_BASE_PORT:-6401}
PREFILL_NOTIFY_BASE_PORT=${PREFILL_NOTIFY_BASE_PORT:-61005}
DECODE_NOTIFY_BASE_PORT=${DECODE_NOTIFY_BASE_PORT:-61105}

# Find script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect number of GPUs (supports both NVIDIA and AMD)
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

trap cleanup_instances EXIT SIGINT SIGTERM

# =============================================================================
# Helper Functions
# =============================================================================

wait_for_server() {
  local port=$1
  local max_timeout=${2:-600}
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

# =============================================================================
# Cleanup any existing instances
# =============================================================================

echo "=== Cleaning up any existing vLLM instances ==="
cleanup_instances

# =============================================================================
# Launch Router (before vLLM instances so ZMQ listener is ready)
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

echo "=== Launching Router on port ${ROUTER_PORT} (ZMQ discovery on ${PROXY_PING_PORT}) ==="

vllm-router \
  --port "$ROUTER_PORT" \
  --policy consistent_hash \
  --prefill-policy consistent_hash \
  --decode-policy consistent_hash \
  --vllm-pd-disaggregation \
  --kv-connector moriio \
  --vllm-discovery-address "0.0.0.0:${PROXY_PING_PORT}" \
  --log-level info \
  > /tmp/router.log 2>&1 &

ROUTER_PID=$!

wait_for_router "$ROUTER_PORT"

# =============================================================================
# Launch Prefill Instances (kv_producer)
# =============================================================================

echo "=== Launching ${NUM_PREFILL_INSTANCES} Prefill Instance(s) ==="

PREFILL_PORTS=()

for i in $(seq 0 $((NUM_PREFILL_INSTANCES - 1))); do
  PORT=$((PREFILL_BASE_PORT + i))
  HANDSHAKE_PORT=$((PREFILL_HANDSHAKE_BASE_PORT + i))
  NOTIFY_PORT=$((PREFILL_NOTIFY_BASE_PORT + i))

  PREFILL_GPUS_PER_INSTANCE=$PREFILLER_TP_SIZE
  GPU_START=$((i * PREFILL_GPUS_PER_INSTANCE))
  GPU_END=$((GPU_START + PREFILL_GPUS_PER_INSTANCE - 1))
  GPU_IDS=$(seq -s, $GPU_START $GPU_END)

  PREFILL_KV_CONFIG=$(cat <<EOF
{
  "kv_connector": "MoRIIOConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "proxy_ip": "${PROXY_IP}",
    "proxy_ping_port": "${PROXY_PING_PORT}",
    "http_port": "${PORT}",
    "handshake_port": "${HANDSHAKE_PORT}",
    "notify_port": "${NOTIFY_PORT}"
  }
}
EOF
)

  echo "Launching Prefill Instance ${i} on port ${PORT} (GPUs: ${GPU_IDS})"

  HIP_VISIBLE_DEVICES=$GPU_IDS \
  VLLM_USE_V1=1 \
  VLLM_LOGGING_LEVEL=DEBUG \
  VLLM_MORIIO_CONNECTOR_READ_MODE=0 \
  MORI_IO_ENABLE_NOTIFICATION=0 \
  VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  vllm serve "$MODEL_NAMES" \
    --port $PORT \
    --load-format dummy \
    --tensor-parallel-size $PREFILLER_TP_SIZE \
    --block-size ${BLOCK_SIZE} \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --no-enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code \
    --kv-transfer-config "$(echo $PREFILL_KV_CONFIG | tr -d '\n')" \
    > /tmp/prefill_${i}.log 2>&1 &

  PREFILL_PORTS+=($PORT)
done

# =============================================================================
# Launch Decode Instances (kv_consumer)
# =============================================================================

echo "=== Launching ${NUM_DECODE_INSTANCES} Decode Instance(s) ==="

DECODE_PORTS=()

for i in $(seq 0 $((NUM_DECODE_INSTANCES - 1))); do
  PORT=$((DECODE_BASE_PORT + i))
  HANDSHAKE_PORT=$((DECODE_HANDSHAKE_BASE_PORT + i))
  NOTIFY_PORT=$((DECODE_NOTIFY_BASE_PORT + i))

  PREFILL_TOTAL_GPUS=$((NUM_PREFILL_INSTANCES * PREFILLER_TP_SIZE))
  DECODE_GPUS_PER_INSTANCE=$DECODER_TP_SIZE
  GPU_START=$(( PREFILL_TOTAL_GPUS + (i * DECODE_GPUS_PER_INSTANCE) ))
  GPU_END=$((GPU_START + DECODE_GPUS_PER_INSTANCE - 1))
  GPU_IDS=$(seq -s, $GPU_START $GPU_END)

  DECODE_KV_CONFIG=$(cat <<EOF
{
  "kv_connector": "MoRIIOConnector",
  "kv_role": "kv_consumer",
  "kv_connector_extra_config": {
    "proxy_ip": "${PROXY_IP}",
    "proxy_ping_port": "${PROXY_PING_PORT}",
    "http_port": "${PORT}",
    "handshake_port": "${HANDSHAKE_PORT}",
    "notify_port": "${NOTIFY_PORT}"
  }
}
EOF
)

  echo "Launching Decode Instance ${i} on port ${PORT} (GPUs: ${GPU_IDS})"

  HIP_VISIBLE_DEVICES=$GPU_IDS \
  VLLM_USE_V1=1 \
  VLLM_LOGGING_LEVEL=DEBUG \
  VLLM_MORIIO_CONNECTOR_READ_MODE=0 \
  MORI_IO_ENABLE_NOTIFICATION=0 \
  VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  vllm serve "$MODEL_NAMES" \
    --port $PORT \
    --load-format dummy \
    --tensor-parallel-size $DECODER_TP_SIZE \
    --block-size ${BLOCK_SIZE} \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --no-enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code \
    --all2all-backend mori \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
    --kv-transfer-config "$(echo $DECODE_KV_CONFIG | tr -d '\n')" \
    > /tmp/decode_${i}.log 2>&1 &

  DECODE_PORTS+=($PORT)
done

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
  echo "Prefill instance on port ${PORT} started successfully"
done

echo "=== Waiting for Decode Instances ==="
for PORT in "${DECODE_PORTS[@]}"; do
  if ! wait_for_server "$PORT"; then
    echo "ERROR: Decode instance on port ${PORT} failed to start"
    echo "=== Decode logs ==="
    cat /tmp/decode_*.log 2>&1 || true
    exit 1
  fi
  echo "Decode instance on port ${PORT} started successfully"
done

# Give ZMQ registrations time to propagate to the router
sleep 10

# =============================================================================
# Run Accuracy Tests
# =============================================================================

echo "=== Running MoRI-IO P/D Disaggregation Sanity Test ==="

python3 "${SCRIPT_DIR}/test_pd_accuracy.py" \
  --router-url "http://localhost:${ROUTER_PORT}" \
  --model "$MODEL_NAMES" \
  --num-requests 20 \
  --skip-streaming

SANITY_EXIT_CODE=$?

if [ $SANITY_EXIT_CODE -ne 0 ]; then
  echo "MoRI-IO P/D disaggregation sanity test FAILED"
  TEST_EXIT_CODE=$SANITY_EXIT_CODE
else
  echo "MoRI-IO P/D disaggregation sanity test PASSED"
  TEST_EXIT_CODE=0
fi

# =============================================================================
# Results
# =============================================================================

echo "=== Test Results ==="
if [ $TEST_EXIT_CODE -eq 0 ]; then
  echo "All MoRI-IO P/D disaggregation tests PASSED"
else
  echo "MoRI-IO P/D disaggregation tests FAILED"
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
