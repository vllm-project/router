#!/usr/bin/env bash
# Same-host ROCm P/D disaggregation coverage using vLLM's NixlConnector.
# This is the ROCm/MI300 analogue of the NVIDIA `run_accuracy_test.sh` NIXL
# lane. It launches a TP1 prefill and a TP1 decode engine on two GPUs of one
# ROCm host, wires them through the router, and validates accuracy.
#
# Unlike MoRI, NIXL has no read/write transfer mode, so there is no matrix
# dimension here: the connector negotiates a single pull-based transfer path
# over its UCX side channel.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Shared ROCm P/D scaffolding: defaults, preflight, cleanup trap, health waits,
# router-binary resolution, GPU detection, and the accuracy harness call.
# shellcheck source=py_test/e2e/pd_disagg_vllm/rocm/_pd_rocm_common.sh
source "${SCRIPT_DIR}/_pd_rocm_common.sh"

# Connector-specific configuration.
# On ROCm, PyTorch exposes AMD GPUs through the "cuda" device namespace, so the
# NIXL buffer device stays "cuda" here. GDS is NVIDIA GPUDirect Storage and is
# unavailable on ROCm, so only the UCX backend is requested.
KV_BUFFER_DEVICE=${KV_BUFFER_DEVICE:-"cuda"}
NIXL_BACKENDS=${NIXL_BACKENDS:-'["UCX"]'}
PREFILL_NIXL_PORT=${PREFILL_NIXL_PORT:-9100}
DECODE_NIXL_PORT=${DECODE_NIXL_PORT:-9200}
PREFILL_NIXL_HTTP_PORT=${PREFILL_NIXL_HTTP_PORT:-8097}
DECODE_NIXL_HTTP_PORT=${DECODE_NIXL_HTTP_PORT:-8098}
LOG_DIR=${LOG_DIR:-/tmp/vllm-router-nixl-rocm}

pd_rocm_require_commands curl python3 vllm

mkdir -p "${LOG_DIR}"

trap pd_rocm_cleanup EXIT
trap 'exit 130' INT TERM

pd_rocm_resolve_router_bin

# The pinned ROCm vLLM image is expected to ship a ROCm-enabled NIXL runtime
# (UCX built with ROCm support). If the module is missing, install the generic
# `nixl` package, which builds against the system UCX/ROCm at import time.
# NOTE: the exact ROCm NIXL wheel name is not yet stable upstream; adjust
# NIXL_PACKAGE if the image ever drops the bundled runtime.
NIXL_PACKAGE=${NIXL_PACKAGE:-"nixl"}
if ! python3 -c "import nixl" >/dev/null 2>&1; then
  echo "nixl module not found in image; installing ${NIXL_PACKAGE}"
  python3 -m pip install --no-cache-dir "${NIXL_PACKAGE}"
fi

pd_rocm_detect_gpus

echo "ROCm ${ROCM_VERSION}; connector=NixlConnector (kv_both); buffer=${KV_BUFFER_DEVICE}"
echo "Model: ${MODEL_NAMES}"
echo "Prefill GPUs: ${PREFILL_GPUS}; decode GPUs: ${DECODE_GPUS}"
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showtopo || true
fi

make_kv_config() {
  local http_port=$1

  printf '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"%s","kv_connector_extra_config":{"backends":%s,"http_port":%s}}' \
    "${KV_BUFFER_DEVICE}" \
    "${NIXL_BACKENDS}" \
    "${http_port}"
}

PREFILL_KV_CONFIG=$(make_kv_config "${PREFILL_NIXL_HTTP_PORT}")
DECODE_KV_CONFIG=$(make_kv_config "${DECODE_NIXL_HTTP_PORT}")

HIP_VISIBLE_DEVICES="${PREFILL_GPUS}" \
VLLM_USE_V1=1 \
VLLM_ROCM_USE_AITER=1 \
VLLM_LOGGING_LEVEL=INFO \
VLLM_NIXL_SIDE_CHANNEL_HOST=0.0.0.0 \
VLLM_NIXL_SIDE_CHANNEL_PORT="${PREFILL_NIXL_PORT}" \
UCX_TLS=all \
UCX_NET_DEVICES=all \
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
VLLM_USE_V1=1 \
VLLM_ROCM_USE_AITER=1 \
VLLM_LOGGING_LEVEL=INFO \
VLLM_NIXL_SIDE_CHANNEL_HOST=0.0.0.0 \
VLLM_NIXL_SIDE_CHANNEL_PORT="${DECODE_NIXL_PORT}" \
UCX_TLS=all \
UCX_NET_DEVICES=all \
vllm serve "${MODEL_NAMES}" \
  --port "${DECODE_PORT}" \
  --tensor-parallel-size "${DECODER_TP_SIZE}" \
  --block-size "${BLOCK_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enforce-eager \
  --kv-transfer-config "${DECODE_KV_CONFIG}" \
  >"${LOG_DIR}/decode.log" 2>&1 &
DECODE_PID=$!

# NIXL negotiates its KV handshake directly between the engines over the side
# channel, so the router uses static prefill/decode URLs (no ZMQ discovery),
# matching the NVIDIA NIXL lane.
"${ROUTER_BIN}" \
  --port "${ROUTER_PORT}" \
  --policy power_of_two \
  --vllm-pd-disaggregation \
  --kv-connector nixl \
  --prefill "http://127.0.0.1:${PREFILL_PORT}" \
  --decode "http://127.0.0.1:${DECODE_PORT}" \
  --worker-startup-check-interval 1 \
  >"${LOG_DIR}/router.log" 2>&1 &
ROUTER_PID=$!

pd_rocm_wait_for_health "prefill" "${PREFILL_PORT}" "${ENGINE_STARTUP_TIMEOUT}" "${PREFILL_PID}"
pd_rocm_wait_for_health "decode" "${DECODE_PORT}" "${ENGINE_STARTUP_TIMEOUT}" "${DECODE_PID}"
pd_rocm_wait_for_health "router" "${ROUTER_PORT}" "${ROUTER_STARTUP_TIMEOUT}" "${ROUTER_PID}"

pd_rocm_run_accuracy "${ROUTER_PORT}"

echo "NIXL ROCm XGMI P/D accuracy passed"
