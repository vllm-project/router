#!/usr/bin/env bash
# gRPC prefill/decode accuracy test.
#
# Same layout as run_accuracy_test.sh: one prefill and one decode, each with
# data-parallel size 2, so four GPUs. Workers are `vllm-rs serve --grpc-port`
# and the router reaches them with grpc:// URLs. Workers must be
# vLLM >= 0.30.0: older engines turn connector integers into floats and abort
# every decode.
#
#   PREFILL_GPUS=0,1 DECODE_GPUS=2,3 MODEL_NAMES=/path/or/hf-id \
#     bash tests/python/e2e/pd_disagg_vllm/run_grpc_pd_accuracy_test.sh
set -euo pipefail

MODEL_NAMES=${MODEL_NAMES:-meta-llama/Llama-3.2-1B-Instruct}
# Name the client sends. Defaults to MODEL_NAMES. Set this when the weights
# live in a local directory but the accuracy baseline is keyed by a hub id.
SERVED_MODEL=${SERVED_MODEL:-$MODEL_NAMES}
INTRA_NODE_DP_SIZE=${INTRA_NODE_DP_SIZE:-2}
PREFILL_GPUS=${PREFILL_GPUS:-0,1}
DECODE_GPUS=${DECODE_GPUS:-2,3}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
BLOCK_SIZE=${BLOCK_SIZE:-128}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
ENGINE_READY_TIMEOUT=${ENGINE_READY_TIMEOUT:-1800}
PREFILL_HTTP_PORT=${PREFILL_HTTP_PORT:-18420}
PREFILL_GRPC_PORT=${PREFILL_GRPC_PORT:-55420}
DECODE_HTTP_PORT=${DECODE_HTTP_PORT:-18421}
DECODE_GRPC_PORT=${DECODE_GRPC_PORT:-55421}
# vLLM adds the data-parallel rank to the side-channel port, so keep the
# prefill and decode ranges apart.
PREFILL_SIDE_PORT=${PREFILL_SIDE_PORT:-58620}
DECODE_SIDE_PORT=${DECODE_SIDE_PORT:-58640}
ROUTER_PORT=${ROUTER_PORT:-30420}
UCX_TLS=${UCX_TLS:-all}
NUM_REQUESTS=${NUM_REQUESTS:-20}
NUM_STREAMING_REQUESTS=${NUM_STREAMING_REQUESTS:-4}
MAX_TOKENS=${MAX_TOKENS:-32}
RUN_LM_EVAL=${RUN_LM_EVAL:-1}
LM_EVAL_LIMIT=${LM_EVAL_LIMIT:-500}
LM_EVAL_CONCURRENCY=${LM_EVAL_CONCURRENCY:-10}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-900}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
LOG_DIR=${LOG_DIR:-/tmp/vllm-router-grpc-pd}
mkdir -p "$LOG_DIR"

PREFILL_PID=""
DECODE_PID=""
ROUTER_PID=""

cleanup() {
  local pid
  for pid in "$ROUTER_PID" "$PREFILL_PID" "$DECODE_PID"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

resolve_vllm_rs() {
  if [[ -n "${VLLM_RS:-}" ]]; then
    echo "$VLLM_RS"
    return
  fi
  if command -v vllm-rs >/dev/null 2>&1; then
    command -v vllm-rs
    return
  fi
  python3 -c "import os, vllm; print(os.path.join(os.path.dirname(vllm.__file__), 'vllm-rs'))"
}

version_at_least() {
  printf '%s\n%s\n' "$2" "$1" | sort -C -V
}

port_busy() {
  timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/$1" 2>/dev/null
}

wait_for_http() {
  local port=$1
  local label=$2
  local pid=$3
  local start
  start=$(date +%s)
  while true; do
    if curl -sf -m 3 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "${label} HTTP ${port} is ready"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: ${label} pid ${pid} exited before becoming healthy" >&2
      return 1
    fi
    if (( $(date +%s) - start >= STARTUP_TIMEOUT )); then
      echo "ERROR: ${label} on port ${port} failed to become healthy" >&2
      return 1
    fi
    sleep 5
  done
}

wait_for_tcp() {
  local port=$1
  local label=$2
  local start
  start=$(date +%s)
  until port_busy "$port"; do
    if (( $(date +%s) - start >= STARTUP_TIMEOUT )); then
      echo "ERROR: ${label} port ${port} never accepted connections" >&2
      return 1
    fi
    sleep 2
  done
  echo "${label} TCP ${port} is ready"
}

VLLM_RS="$(resolve_vllm_rs)"
if [[ ! -x "$VLLM_RS" ]]; then
  echo "vllm-rs is not executable at ${VLLM_RS}" >&2
  exit 1
fi
RS_VERSION="$("$VLLM_RS" --version | awk '{print $NF}')"
if ! version_at_least "$RS_VERSION" "0.30.0"; then
  echo "gRPC P/D needs vllm-rs >= 0.30.0, found ${RS_VERSION}" >&2
  exit 1
fi
echo "vllm-rs ${RS_VERSION} at ${VLLM_RS}"

if ! python3 -c "import nixl._api" >/dev/null 2>&1; then
  echo "NIXL is not importable from $(command -v python3). The CI step installs it with:" >&2
  echo "  uv pip install nixl-cu13 && uv pip install nixl --no-deps" >&2
  exit 1
fi

IFS=, read -r -a prefill_gpu_list <<<"$PREFILL_GPUS"
IFS=, read -r -a decode_gpu_list <<<"$DECODE_GPUS"
if (( ${#prefill_gpu_list[@]} != INTRA_NODE_DP_SIZE || ${#decode_gpu_list[@]} != INTRA_NODE_DP_SIZE )); then
  echo "PREFILL_GPUS and DECODE_GPUS must each list ${INTRA_NODE_DP_SIZE} GPUs" >&2
  exit 1
fi
for gpu in "${prefill_gpu_list[@]}" "${decode_gpu_list[@]}"; do
  if ! nvidia-smi -i "$gpu" >/dev/null 2>&1; then
    echo "GPU ${gpu} is not visible to this process" >&2
    exit 1
  fi
done

for port in \
  "$PREFILL_HTTP_PORT" "$PREFILL_GRPC_PORT" "$PREFILL_SIDE_PORT" \
  "$DECODE_HTTP_PORT" "$DECODE_GRPC_PORT" "$DECODE_SIDE_PORT" \
  "$ROUTER_PORT"
do
  if port_busy "$port"; then
    echo "port ${port} is already in use" >&2
    exit 1
  fi
done

if [[ -n "${ROUTER_BIN:-}" ]]; then
  ROUTER="$ROUTER_BIN"
elif command -v vllm-router >/dev/null 2>&1; then
  ROUTER="$(command -v vllm-router)"
elif [[ -x "${REPO_ROOT}/target/release/vllm-router" ]]; then
  ROUTER="${REPO_ROOT}/target/release/vllm-router"
else
  echo "building vllm-router"
  (cd "$REPO_ROOT" && cargo build --release)
  ROUTER="${REPO_ROOT}/target/release/vllm-router"
fi

launch_worker() {
  local name=$1
  local gpus=$2
  local http=$3
  local grpc=$4
  local side=$5
  local role=$6
  local kv
  kv=$(printf '{"kv_connector":"NixlConnector","kv_role":"%s"}' "$role")
  echo "launch ${name} gpus=${gpus} dp=${INTRA_NODE_DP_SIZE} http=${http} grpc=${grpc} side=${side} role=${role}"
  CUDA_VISIBLE_DEVICES="$gpus" \
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
    VLLM_NIXL_SIDE_CHANNEL_PORT="$side" \
    UCX_TLS="$UCX_TLS" \
    "$VLLM_RS" serve "$MODEL_NAMES" \
      --host 127.0.0.1 \
      --port "$http" \
      --grpc-port "$grpc" \
      --data-parallel-size "$INTRA_NODE_DP_SIZE" \
      --max-model-len "$MAX_MODEL_LEN" \
      --engine-ready-timeout-secs "$ENGINE_READY_TIMEOUT" \
      -- \
      --tensor-parallel-size 1 \
      --block-size "$BLOCK_SIZE" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --enforce-eager \
      --enable-prefix-caching \
      --disable-hybrid-kv-cache-manager \
      --served-model-name "$SERVED_MODEL" \
      --kv-transfer-config "$kv" \
      >"${LOG_DIR}/${name}.log" 2>&1 &
}

launch_worker prefill "$PREFILL_GPUS" "$PREFILL_HTTP_PORT" "$PREFILL_GRPC_PORT" "$PREFILL_SIDE_PORT" kv_producer
PREFILL_PID=$!
launch_worker decode "$DECODE_GPUS" "$DECODE_HTTP_PORT" "$DECODE_GRPC_PORT" "$DECODE_SIDE_PORT" kv_consumer
DECODE_PID=$!
echo "prefill pid=${PREFILL_PID} decode pid=${DECODE_PID}"

if ! wait_for_http "$PREFILL_HTTP_PORT" prefill "$PREFILL_PID"; then
  tail -n 80 "${LOG_DIR}/prefill.log" || true
  exit 1
fi
if ! wait_for_http "$DECODE_HTTP_PORT" decode "$DECODE_PID"; then
  tail -n 80 "${LOG_DIR}/decode.log" || true
  exit 1
fi
wait_for_tcp "$PREFILL_GRPC_PORT" "prefill grpc"
wait_for_tcp "$DECODE_GRPC_PORT" "decode grpc"

export VLLM_ROUTER_MODEL="${VLLM_ROUTER_MODEL:-$MODEL_NAMES}"
if [[ -d "$MODEL_NAMES" && -f "$MODEL_NAMES/tokenizer.json" ]]; then
  export VLLM_ROUTER_TOKENIZER="${VLLM_ROUTER_TOKENIZER:-$MODEL_NAMES/tokenizer.json}"
fi

echo "launch router on ${ROUTER_PORT}"
"$ROUTER" \
  --host 127.0.0.1 \
  --port "$ROUTER_PORT" \
  --vllm-pd-disaggregation \
  --intra-node-data-parallel-size "$INTRA_NODE_DP_SIZE" \
  --prefill "grpc://127.0.0.1:${PREFILL_GRPC_PORT}" \
  --decode "grpc://127.0.0.1:${DECODE_GRPC_PORT}" \
  --kv-connector nixl \
  --policy power_of_two \
  --worker-startup-check-interval 1 \
  >"${LOG_DIR}/router.log" 2>&1 &
ROUTER_PID=$!

if ! wait_for_http "$ROUTER_PORT" router "$ROUTER_PID"; then
  tail -n 80 "${LOG_DIR}/router.log" || true
  exit 1
fi

if ! python3 "${SCRIPT_DIR}/test_grpc_pd_chat.py" \
  --router-url "http://127.0.0.1:${ROUTER_PORT}" \
  --model "$SERVED_MODEL" \
  --num-requests "$NUM_REQUESTS" \
  --num-streaming-requests "$NUM_STREAMING_REQUESTS" \
  --max-tokens "$MAX_TOKENS"
then
  echo "chat accuracy failed" >&2
  tail -n 80 "${LOG_DIR}/router.log" || true
  exit 1
fi

if [[ "$RUN_LM_EVAL" == "1" ]]; then
  LM_ARGS=(
    --router-url "http://127.0.0.1:${ROUTER_PORT}"
    --model "$SERVED_MODEL"
    --num-concurrent "$LM_EVAL_CONCURRENCY"
    --limit "$LM_EVAL_LIMIT"
    --skip-connectivity
  )
  if [[ "$SERVED_MODEL$MODEL_NAMES" == *[Qq]wen3* ]]; then
    LM_ARGS+=(--disable-thinking --filter "exact_match,flexible-extract")
  fi
  python3 "${SCRIPT_DIR}/test_lm_eval_accuracy.py" "${LM_ARGS[@]}"
fi

echo "gRPC P/D accuracy test passed"
