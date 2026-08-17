#!/usr/bin/env bash
# Two-host ROCm P/D disaggregation coverage using MoRI GPU-memory RDMA.
# Run this script on the coordinator; it controls the decode host over SSH.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODEL_NAMES=${MODEL_NAMES:-"Qwen/Qwen3-0.6B"}
ROCM_VLLM_IMAGE=${ROCM_VLLM_IMAGE:-"vllm/vllm-openai-rocm:nightly@sha256:d22922d540810d90c5a3eafe91d3b4a62c2b881f3e990d0b7180b2875a5d176d"}
WORKER_SSH_HOST=${WORKER_SSH_HOST:-"vllm-router-rocm-worker"}
COORDINATOR_FABRIC_IP=${COORDINATOR_FABRIC_IP:-"192.168.200.1"}
WORKER_FABRIC_IP=${WORKER_FABRIC_IP:-"192.168.200.2"}
HF_CACHE_DIR=${HF_CACHE_DIR:-"/var/lib/buildkite-agent/.cache/huggingface"}
WORKER_HF_CACHE_DIR=${WORKER_HF_CACHE_DIR:-"/var/lib/buildkite-agent/.cache/huggingface"}
MORIIO_READ_MODE=${MORIIO_READ_MODE:-"true"}
MORIIO_QP_PER_TRANSFER=${MORIIO_QP_PER_TRANSFER:-1}
MORIIO_POST_BATCH_SIZE=${MORIIO_POST_BATCH_SIZE:-1}
MORIIO_NUM_WORKERS=${MORIIO_NUM_WORKERS:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-ROCM_AITER_FA}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-false}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-false}
HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-0}
ENGINE_STARTUP_TIMEOUT=${ENGINE_STARTUP_TIMEOUT:-1800}
ROUTER_STARTUP_TIMEOUT=${ROUTER_STARTUP_TIMEOUT:-180}
NUM_SANITY_REQUESTS=${NUM_SANITY_REQUESTS:-20}
NUM_LM_EVAL_CONCURRENT=${NUM_LM_EVAL_CONCURRENT:-1}
LM_EVAL_LIMIT=${LM_EVAL_LIMIT:-500}
LM_EVAL_LOG_SAMPLES=${LM_EVAL_LOG_SAMPLES:-false}
SMOKE_ONLY=${SMOKE_ONLY:-false}
SKIP_LM_EVAL=${SKIP_LM_EVAL:-false}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-300}
EVAL_PIP_CACHE_VOLUME=${EVAL_PIP_CACHE_VOLUME:-"vllm-router-rocm-pip-cache"}

PREFILL_PORT=${PREFILL_PORT:-8100}
DECODE_PORT=${DECODE_PORT:-8200}
ROUTER_PORT=${ROUTER_PORT:-8300}
PROXY_PING_PORT=${PROXY_PING_PORT:-36367}
PREFILL_HANDSHAKE_PORT=${PREFILL_HANDSHAKE_PORT:-6301}
DECODE_HANDSHAKE_PORT=${DECODE_HANDSHAKE_PORT:-7301}
PREFILL_NOTIFY_PORT=${PREFILL_NOTIFY_PORT:-61005}
DECODE_NOTIFY_PORT=${DECODE_NOTIFY_PORT:-62005}
LOG_DIR=${LOG_DIR:-"${REPO_ROOT}/target/ci-logs/moriio-rdma-${MORIIO_READ_MODE}"}

PREFILL_CONTAINER="vllm-router-mori-rdma-prefill"
DECODE_CONTAINER="vllm-router-mori-rdma-decode"
EVAL_CONTAINER="vllm-router-mori-rdma-eval"

case "${MORIIO_READ_MODE}" in
  true|false) ;;
  *)
    echo "ERROR: MORIIO_READ_MODE must be 'true' or 'false', got '${MORIIO_READ_MODE}'" >&2
    exit 2
    ;;
esac

for integer_setting in \
  MORIIO_QP_PER_TRANSFER \
  MORIIO_POST_BATCH_SIZE \
  MORIIO_NUM_WORKERS; do
  if ! [[ "${!integer_setting}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${integer_setting} must be a positive integer, got '${!integer_setting}'" >&2
    exit 2
  fi
done

case "${HSA_NO_SCRATCH_RECLAIM}" in
  0|1) ;;
  *)
    echo "ERROR: HSA_NO_SCRATCH_RECLAIM must be '0' or '1', got '${HSA_NO_SCRATCH_RECLAIM}'" >&2
    exit 2
    ;;
esac

for command_name in curl docker ssh; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "ERROR: required command '${command_name}' is unavailable" >&2
    exit 2
  fi
done

if [[ -n "${ROUTER_BIN:-}" ]]; then
  if [[ ! -x "${ROUTER_BIN}" ]]; then
    echo "ERROR: ROUTER_BIN is not executable: ${ROUTER_BIN}" >&2
    exit 2
  fi
elif [[ -x "${REPO_ROOT}/target/release/vllm-router" ]]; then
  ROUTER_BIN="${REPO_ROOT}/target/release/vllm-router"
else
  echo "ERROR: set ROUTER_BIN or build ${REPO_ROOT}/target/release/vllm-router" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
docker volume create "${EVAL_PIP_CACHE_VOLUME}" >/dev/null

# OpenSSH joins command arguments without preserving their boundaries. Render a
# Bash-escaped command so JSON connector configuration remains one argument.
worker_run() {
  local command_string
  printf -v command_string '%q ' "$@"
  ssh -o BatchMode=yes "${WORKER_SSH_HOST}" "exec ${command_string}"
}

ROUTER_PID=""

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

  docker logs "${PREFILL_CONTAINER}" >"${LOG_DIR}/prefill.log" 2>&1
  worker_run docker logs "${DECODE_CONTAINER}" >"${LOG_DIR}/decode.log" 2>&1

  docker rm -f "${EVAL_CONTAINER}" "${PREFILL_CONTAINER}" >/dev/null 2>&1
  worker_run docker rm -f "${DECODE_CONTAINER}" >/dev/null 2>&1

  if [[ -n "${ROUTER_PID}" ]]; then
    kill -TERM "${ROUTER_PID}" 2>/dev/null
    wait "${ROUTER_PID}" 2>/dev/null
  fi

  if (( exit_code != 0 )); then
    show_logs
  fi
  exit "${exit_code}"
}

trap cleanup EXIT
trap 'exit 130' INT TERM

wait_for_health() {
  local name=$1
  local url=$2
  local timeout=$3
  local start_time
  local elapsed

  start_time=$(date +%s)
  until curl --fail --silent --show-error "${url}" >/dev/null 2>&1; do
    case "${name}" in
      prefill)
        if [[ "$(docker inspect --format '{{.State.Running}}' "${PREFILL_CONTAINER}" 2>/dev/null || true)" != "true" ]]; then
          echo "ERROR: prefill container exited during startup" >&2
          return 1
        fi
        ;;
      decode)
        if [[ "$(worker_run docker inspect --format '{{.State.Running}}' "${DECODE_CONTAINER}" 2>/dev/null || true)" != "true" ]]; then
          echo "ERROR: decode container exited during startup" >&2
          return 1
        fi
        ;;
      "router discovery")
        if ! kill -0 "${ROUTER_PID}" 2>/dev/null; then
          echo "ERROR: router exited during startup" >&2
          return 1
        fi
        ;;
    esac

    elapsed=$(( $(date +%s) - start_time ))
    if (( elapsed >= timeout )); then
      echo "ERROR: ${name} did not become healthy within ${timeout}s" >&2
      return 1
    fi
    sleep 5
  done
  echo "${name} is healthy at ${url}"
}

docker info >/dev/null
worker_run docker info >/dev/null
test -d "${HF_CACHE_DIR}"
worker_run test -d "${WORKER_HF_CACHE_DIR}"

# Refuse to reuse stale processes or containers from an interrupted CI job.
docker rm -f "${EVAL_CONTAINER}" "${PREFILL_CONTAINER}" >/dev/null 2>&1 || true
worker_run docker rm -f "${DECODE_CONTAINER}" >/dev/null 2>&1 || true

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

PREFILL_KV_CONFIG=$(printf '{"kv_connector":"MoRIIOConnector","kv_role":"kv_producer","kv_connector_extra_config":{"proxy_ip":"%s","proxy_ping_port":%s,"http_port":%s,"handshake_port":%s,"notify_port":%s,"host_ip":"%s","read_mode":%s,"backend":"rdma","qp_per_transfer":%s,"post_batch_size":%s,"num_workers":%s}}' \
  "${COORDINATOR_FABRIC_IP}" "${PROXY_PING_PORT}" "${PREFILL_PORT}" \
  "${PREFILL_HANDSHAKE_PORT}" "${PREFILL_NOTIFY_PORT}" \
  "${COORDINATOR_FABRIC_IP}" "${MORIIO_READ_MODE}" \
  "${MORIIO_QP_PER_TRANSFER}" "${MORIIO_POST_BATCH_SIZE}" \
  "${MORIIO_NUM_WORKERS}")

DECODE_KV_CONFIG=$(printf '{"kv_connector":"MoRIIOConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"proxy_ip":"%s","proxy_ping_port":%s,"http_port":%s,"handshake_port":%s,"notify_port":%s,"host_ip":"%s","read_mode":%s,"backend":"rdma","qp_per_transfer":%s,"post_batch_size":%s,"num_workers":%s}}' \
  "${COORDINATOR_FABRIC_IP}" "${PROXY_PING_PORT}" "${DECODE_PORT}" \
  "${DECODE_HANDSHAKE_PORT}" "${DECODE_NOTIFY_PORT}" \
  "${WORKER_FABRIC_IP}" "${MORIIO_READ_MODE}" \
  "${MORIIO_QP_PER_TRANSFER}" "${MORIIO_POST_BATCH_SIZE}" \
  "${MORIIO_NUM_WORKERS}")

COMMON_DOCKER_ARGS=(
  run -d
  --network host
  --ipc host
  --privileged
  --init
  --shm-size 256g
  --ulimit memlock=-1
  --ulimit stack=67108864
  -e HF_HOME=/root/.cache/huggingface
  -e HIP_VISIBLE_DEVICES=0
  -e MORI_DEVICE_NIC=mlx5
  # MoRI otherwise auto-creates an XGMI backend and may prefer it when remote
  # node identity is ambiguous. This lane must prove RDMA, so make the backend
  # selection exclusive and deterministic.
  -e MORI_DISABLE_AUTO_XGMI=1
  -e GLOO_SOCKET_IFNAME=eth2
  # The upstream image defaults this to 1 for older firmware. Retaining scratch
  # allocations indefinitely can exhaust runtime resources under concurrent decode.
  -e HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM}"
  -e VLLM_ROCM_USE_AITER=1
  -e VLLM_LOGGING_LEVEL=INFO
  --entrypoint vllm
)

PREFIX_CACHING_ARGS=(--no-enable-prefix-caching)
if [[ "${ENABLE_PREFIX_CACHING}" == "true" ]]; then
  PREFIX_CACHING_ARGS=(--enable-prefix-caching)
fi

CHUNKED_PREFILL_ARGS=(--no-enable-chunked-prefill)
if [[ "${ENABLE_CHUNKED_PREFILL}" == "true" ]]; then
  CHUNKED_PREFILL_ARGS=(--enable-chunked-prefill)
fi

docker "${COMMON_DOCKER_ARGS[@]}" \
  --name "${PREFILL_CONTAINER}" \
  --hostname vllm-router-rocm-node-a \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  "${ROCM_VLLM_IMAGE}" \
  serve "${MODEL_NAMES}" \
  --host 0.0.0.0 \
  --port "${PREFILL_PORT}" \
  --tensor-parallel-size 1 \
  --block-size 16 \
  --attention-backend "${ATTENTION_BACKEND}" \
  "${PREFIX_CACHING_ARGS[@]}" \
  "${CHUNKED_PREFILL_ARGS[@]}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enforce-eager \
  --kv-transfer-config "${PREFILL_KV_CONFIG}" >/dev/null

worker_run docker "${COMMON_DOCKER_ARGS[@]}" \
  --name "${DECODE_CONTAINER}" \
  --hostname vllm-router-rocm-node-b \
  -v "${WORKER_HF_CACHE_DIR}:/root/.cache/huggingface" \
  "${ROCM_VLLM_IMAGE}" \
  serve "${MODEL_NAMES}" \
  --host 0.0.0.0 \
  --port "${DECODE_PORT}" \
  --tensor-parallel-size 1 \
  --block-size 16 \
  --attention-backend "${ATTENTION_BACKEND}" \
  "${PREFIX_CACHING_ARGS[@]}" \
  "${CHUNKED_PREFILL_ARGS[@]}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enforce-eager \
  --kv-transfer-config "${DECODE_KV_CONFIG}" >/dev/null

wait_for_health "prefill" "http://${COORDINATOR_FABRIC_IP}:${PREFILL_PORT}/health" "${ENGINE_STARTUP_TIMEOUT}"
wait_for_health "decode" "http://${WORKER_FABRIC_IP}:${DECODE_PORT}/health" "${ENGINE_STARTUP_TIMEOUT}"
wait_for_health "router discovery" "http://127.0.0.1:${ROUTER_PORT}/health" "${ROUTER_STARTUP_TIMEOUT}"

# This request must traverse prefill on node A and decode on node B. A correct
# response therefore proves that MoRI registered and transferred GPU KV memory
# across the RDMA fabric, not merely that host-memory verbs work.
curl --fail --silent --show-error \
  --max-time "${REQUEST_TIMEOUT}" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL_NAMES}\",\"prompt\":\"The capital of France is\",\"max_tokens\":8,\"temperature\":0}" \
  "http://127.0.0.1:${ROUTER_PORT}/v1/completions" \
  | tee "${LOG_DIR}/smoke-response.json"
echo

if [[ "${SMOKE_ONLY}" == "true" ]]; then
  echo "MoRI RDMA cross-node smoke passed (read_mode=${MORIIO_READ_MODE})"
  exit 0
fi

EVAL_COMMAND=$(cat <<EOF
set -Eeuo pipefail
export PIP_ROOT_USER_ACTION=ignore
export PIP_CACHE_DIR=/root/.cache/pip
python3 -m pip install --disable-pip-version-check --quiet requests
if [[ ${SKIP_LM_EVAL} != true ]]; then
  python3 -m pip install --disable-pip-version-check --quiet openai 'lm-eval[api]==0.4.12'
fi
python3 py_test/e2e/pd_disagg_vllm/test_pd_accuracy.py \\
  --router-url http://127.0.0.1:${ROUTER_PORT} \\
  --model ${MODEL_NAMES} \\
  --num-requests ${NUM_SANITY_REQUESTS} \\
  --skip-streaming
if [[ ${SKIP_LM_EVAL} == true ]]; then exit 0; fi
args=(
  --router-url http://127.0.0.1:${ROUTER_PORT}
  --model ${MODEL_NAMES}
  --num-concurrent ${NUM_LM_EVAL_CONCURRENT}
  --max-gen-toks 512
  --limit ${LM_EVAL_LIMIT}
  --filter "exact_match,flexible-extract"
  --disable-thinking
)
if [[ ${LM_EVAL_LOG_SAMPLES} == true ]]; then args+=(--log-samples); fi
python3 py_test/e2e/pd_disagg_vllm/test_lm_eval_accuracy.py "\${args[@]}"
EOF
)

docker run --rm \
  --name "${EVAL_CONTAINER}" \
  --network host \
  -v "${REPO_ROOT}:/workdir" \
  -v "${EVAL_PIP_CACHE_VOLUME}:/root/.cache/pip" \
  -w /workdir \
  --entrypoint bash \
  "${ROCM_VLLM_IMAGE}" \
  -lc "${EVAL_COMMAND}"

echo "MoRI RDMA cross-node accuracy passed (read_mode=${MORIIO_READ_MODE})"
