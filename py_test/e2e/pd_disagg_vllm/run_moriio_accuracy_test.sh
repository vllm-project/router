#!/usr/bin/env bash
# Same-host ROCm P/D disaggregation coverage using vLLM's MoRIIOConnector.
# The CI matrix runs this script once in READ mode and once in WRITE mode.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODEL_NAMES=${MODEL_NAMES:-"Qwen/Qwen3-0.6B"}
MORIIO_READ_MODE=${MORIIO_READ_MODE:-"true"}
MORIIO_BACKEND=${MORIIO_BACKEND:-"xgmi"}
PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:-1}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
BLOCK_SIZE=${BLOCK_SIZE:-16}
ENGINE_STARTUP_TIMEOUT=${ENGINE_STARTUP_TIMEOUT:-900}
ROUTER_STARTUP_TIMEOUT=${ROUTER_STARTUP_TIMEOUT:-180}
NUM_SANITY_REQUESTS=${NUM_SANITY_REQUESTS:-20}
NUM_LM_EVAL_CONCURRENT=${NUM_LM_EVAL_CONCURRENT:-10}
LM_EVAL_LIMIT=${LM_EVAL_LIMIT:-500}
LM_EVAL_LOG_SAMPLES=${LM_EVAL_LOG_SAMPLES:-false}

PREFILL_PORT=${PREFILL_PORT:-8100}
DECODE_PORT=${DECODE_PORT:-8200}
ROUTER_PORT=${ROUTER_PORT:-8300}
PROXY_PING_PORT=${PROXY_PING_PORT:-36367}
PREFILL_HANDSHAKE_PORT=${PREFILL_HANDSHAKE_PORT:-6301}
DECODE_HANDSHAKE_PORT=${DECODE_HANDSHAKE_PORT:-7301}
PREFILL_NOTIFY_PORT=${PREFILL_NOTIFY_PORT:-61005}
DECODE_NOTIFY_PORT=${DECODE_NOTIFY_PORT:-62005}
LOG_DIR=${LOG_DIR:-/tmp/vllm-router-moriio}

case "${MORIIO_READ_MODE}" in
  true|false) ;;
  *)
    echo "ERROR: MORIIO_READ_MODE must be 'true' or 'false', got '${MORIIO_READ_MODE}'" >&2
    exit 2
    ;;
esac

if [[ "${MORIIO_BACKEND}" != "xgmi" ]]; then
  echo "ERROR: this single-host CI lane requires MORIIO_BACKEND=xgmi" >&2
  exit 2
fi

for command_name in curl python3 vllm; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "ERROR: required command '${command_name}' is unavailable" >&2
    exit 2
  fi
done

mkdir -p "${LOG_DIR}"

PREFILL_PID=""
DECODE_PID=""
ROUTER_PID=""

terminate_process_tree() {
  local pid=$1
  local child

  [[ -n "${pid}" ]] || return 0
  kill -0 "${pid}" 2>/dev/null || return 0

  while read -r child; do
    [[ -n "${child}" ]] && terminate_process_tree "${child}"
  done < <(ps -o pid= --ppid "${pid}" 2>/dev/null || true)

  kill -TERM "${pid}" 2>/dev/null || true
}

show_logs() {
  local log_file
  for log_file in "${LOG_DIR}"/*.log; do
    [[ -e "${log_file}" ]] || continue
    echo "===== ${log_file} (last 200 lines) ====="
    tail -n 200 "${log_file}" || true
  done
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  set +e

  if (( exit_code != 0 )); then
    show_logs
  fi

  terminate_process_tree "${PREFILL_PID}"
  terminate_process_tree "${DECODE_PID}"
  terminate_process_tree "${ROUTER_PID}"

  # Bound shutdown time. The Docker plugin removes the container after this
  # command exits, but a stuck engine must not hold the CI job open forever.
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if ! kill -0 "${PREFILL_PID}" 2>/dev/null \
      && ! kill -0 "${DECODE_PID}" 2>/dev/null \
      && ! kill -0 "${ROUTER_PID}" 2>/dev/null; then
      break
    fi
    sleep 1
  done

  [[ -z "${PREFILL_PID}" ]] || kill -KILL "${PREFILL_PID}" 2>/dev/null
  [[ -z "${DECODE_PID}" ]] || kill -KILL "${DECODE_PID}" 2>/dev/null
  [[ -z "${ROUTER_PID}" ]] || kill -KILL "${ROUTER_PID}" 2>/dev/null

  [[ -z "${PREFILL_PID}" ]] || wait "${PREFILL_PID}" 2>/dev/null
  [[ -z "${DECODE_PID}" ]] || wait "${DECODE_PID}" 2>/dev/null
  [[ -z "${ROUTER_PID}" ]] || wait "${ROUTER_PID}" 2>/dev/null
  exit "${exit_code}"
}

trap cleanup EXIT
trap 'exit 130' INT TERM

wait_for_health() {
  local name=$1
  local port=$2
  local timeout=$3
  local pid=$4
  local start_time
  local elapsed

  start_time=$(date +%s)
  until curl --fail --silent --show-error "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ERROR: ${name} exited before becoming healthy" >&2
      return 1
    fi

    elapsed=$(( $(date +%s) - start_time ))
    if (( elapsed >= timeout )); then
      echo "ERROR: ${name} did not become healthy within ${timeout}s" >&2
      return 1
    fi
    sleep 5
  done

  echo "${name} is healthy"
}

if [[ -n "${ROUTER_BIN:-}" ]]; then
  if [[ ! -x "${ROUTER_BIN}" ]]; then
    echo "ERROR: ROUTER_BIN is not executable: ${ROUTER_BIN}" >&2
    exit 2
  fi
elif command -v vllm-router >/dev/null 2>&1; then
  ROUTER_BIN=$(command -v vllm-router)
elif [[ -x "${REPO_ROOT}/target/release/vllm-router" ]]; then
  ROUTER_BIN="${REPO_ROOT}/target/release/vllm-router"
else
  if ! command -v cargo >/dev/null 2>&1; then
    echo "ERROR: vllm-router is unavailable and cargo is not installed" >&2
    exit 2
  fi
  cargo build --release --manifest-path "${REPO_ROOT}/Cargo.toml"
  ROUTER_BIN="${REPO_ROOT}/target/release/vllm-router"
fi

read -r AVAILABLE_GPUS ROCM_VERSION < <(
  python3 - <<'PY'
import torch

if not torch.version.hip:
    raise SystemExit("ERROR: PyTorch does not report a ROCm runtime")
print(torch.cuda.device_count(), torch.version.hip)
PY
)

REQUIRED_GPUS=$((PREFILLER_TP_SIZE + DECODER_TP_SIZE))
if (( AVAILABLE_GPUS < REQUIRED_GPUS )); then
  echo "ERROR: ${REQUIRED_GPUS} GPUs are required, but PyTorch sees ${AVAILABLE_GPUS}" >&2
  exit 2
fi

PREFILL_GPU_END=$((PREFILLER_TP_SIZE - 1))
DECODE_GPU_START=${PREFILLER_TP_SIZE}
DECODE_GPU_END=$((REQUIRED_GPUS - 1))
PREFILL_GPUS=$(seq -s, 0 "${PREFILL_GPU_END}")
DECODE_GPUS=$(seq -s, "${DECODE_GPU_START}" "${DECODE_GPU_END}")

echo "ROCm ${ROCM_VERSION}; MoRI ${MORIIO_BACKEND}; read_mode=${MORIIO_READ_MODE}"
echo "Model: ${MODEL_NAMES}"
echo "Prefill GPUs: ${PREFILL_GPUS}; decode GPUs: ${DECODE_GPUS}"
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showtopo || true
fi

make_kv_config() {
  local role=$1
  local http_port=$2
  local handshake_port=$3
  local notify_port=$4

  printf '{"kv_connector":"MoRIIOConnector","kv_role":"%s","kv_connector_extra_config":{"proxy_ip":"127.0.0.1","proxy_ping_port":%s,"http_port":%s,"handshake_port":%s,"notify_port":%s,"read_mode":%s,"backend":"%s"}}' \
    "${role}" \
    "${PROXY_PING_PORT}" \
    "${http_port}" \
    "${handshake_port}" \
    "${notify_port}" \
    "${MORIIO_READ_MODE}" \
    "${MORIIO_BACKEND}"
}

PREFILL_KV_CONFIG=$(make_kv_config \
  kv_producer "${PREFILL_PORT}" "${PREFILL_HANDSHAKE_PORT}" "${PREFILL_NOTIFY_PORT}")
DECODE_KV_CONFIG=$(make_kv_config \
  kv_consumer "${DECODE_PORT}" "${DECODE_HANDSHAKE_PORT}" "${DECODE_NOTIFY_PORT}")

# Start the discovery proxy first. vLLM workers retry registration while their
# HTTP engines initialize, and Router health stays unavailable until both roles
# have registered.
"${ROUTER_BIN}" \
  --port "${ROUTER_PORT}" \
  --policy consistent_hash \
  --prefill-policy consistent_hash \
  --decode-policy consistent_hash \
  --vllm-pd-disaggregation \
  --kv-connector moriio \
  --vllm-discovery-address "0.0.0.0:${PROXY_PING_PORT}" \
  --worker-startup-check-interval 1 \
  >"${LOG_DIR}/router.log" 2>&1 &
ROUTER_PID=$!

HIP_VISIBLE_DEVICES="${PREFILL_GPUS}" \
VLLM_ROCM_USE_AITER=1 \
VLLM_LOGGING_LEVEL=INFO \
vllm serve "${MODEL_NAMES}" \
  --port "${PREFILL_PORT}" \
  --tensor-parallel-size "${PREFILLER_TP_SIZE}" \
  --block-size "${BLOCK_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enforce-eager \
  --kv-transfer-config "${PREFILL_KV_CONFIG}" \
  >"${LOG_DIR}/prefill.log" 2>&1 &
PREFILL_PID=$!

HIP_VISIBLE_DEVICES="${DECODE_GPUS}" \
VLLM_ROCM_USE_AITER=1 \
VLLM_LOGGING_LEVEL=INFO \
vllm serve "${MODEL_NAMES}" \
  --port "${DECODE_PORT}" \
  --tensor-parallel-size "${DECODER_TP_SIZE}" \
  --block-size "${BLOCK_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enforce-eager \
  --kv-transfer-config "${DECODE_KV_CONFIG}" \
  >"${LOG_DIR}/decode.log" 2>&1 &
DECODE_PID=$!

wait_for_health "prefill" "${PREFILL_PORT}" "${ENGINE_STARTUP_TIMEOUT}" "${PREFILL_PID}"
wait_for_health "decode" "${DECODE_PORT}" "${ENGINE_STARTUP_TIMEOUT}" "${DECODE_PID}"
wait_for_health "router discovery" "${ROUTER_PORT}" "${ROUTER_STARTUP_TIMEOUT}" "${ROUTER_PID}"

python3 "${SCRIPT_DIR}/test_pd_accuracy.py" \
  --router-url "http://127.0.0.1:${ROUTER_PORT}" \
  --model "${MODEL_NAMES}" \
  --num-requests "${NUM_SANITY_REQUESTS}" \
  --skip-streaming

LM_EVAL_ARGS=(
  --router-url "http://127.0.0.1:${ROUTER_PORT}"
  --model "${MODEL_NAMES}"
  --num-concurrent "${NUM_LM_EVAL_CONCURRENT}"
  --max-gen-toks 512
  --limit "${LM_EVAL_LIMIT}"
  --filter "exact_match,flexible-extract"
  --disable-thinking
)
if [[ "${LM_EVAL_LOG_SAMPLES}" == "true" ]]; then
  LM_EVAL_ARGS+=(--log-samples)
fi
python3 "${SCRIPT_DIR}/test_lm_eval_accuracy.py" "${LM_EVAL_ARGS[@]}"

echo "MoRI XGMI P/D accuracy passed (read_mode=${MORIIO_READ_MODE})"
