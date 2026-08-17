#!/usr/bin/env bash
# Same-host ROCm P/D disaggregation coverage using vLLM's MoRIIOConnector.
# The CI matrix runs this script once in READ mode and once in WRITE mode.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Shared ROCm P/D scaffolding: defaults, preflight, cleanup trap, health waits,
# router-binary resolution, GPU detection, and the accuracy harness call.
# shellcheck source=py_test/e2e/pd_disagg_vllm/rocm/_pd_rocm_common.sh
source "${SCRIPT_DIR}/_pd_rocm_common.sh"

# Connector-specific configuration.
MORIIO_READ_MODE=${MORIIO_READ_MODE:-"true"}
MORIIO_BACKEND=${MORIIO_BACKEND:-"xgmi"}
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

pd_rocm_require_commands curl python3 vllm

mkdir -p "${LOG_DIR}"

trap pd_rocm_cleanup EXIT
trap 'exit 130' INT TERM

pd_rocm_resolve_router_bin
pd_rocm_detect_gpus

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

pd_rocm_wait_for_health "prefill" "${PREFILL_PORT}" "${ENGINE_STARTUP_TIMEOUT}" "${PREFILL_PID}"
pd_rocm_wait_for_health "decode" "${DECODE_PORT}" "${ENGINE_STARTUP_TIMEOUT}" "${DECODE_PID}"
pd_rocm_wait_for_health "router discovery" "${ROUTER_PORT}" "${ROUTER_STARTUP_TIMEOUT}" "${ROUTER_PID}"

pd_rocm_run_accuracy "${ROUTER_PORT}"

echo "MoRI XGMI P/D accuracy passed (read_mode=${MORIIO_READ_MODE})"
