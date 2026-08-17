#!/usr/bin/env bash
# Shared scaffolding for the same-host ROCm P/D disaggregation runners
# (run_moriio_accuracy_test.sh and run_nixl_rocm_accuracy_test.sh).
#
# This file is meant to be *sourced*, not executed. It defines functions and
# applies defaults for configuration that is genuinely identical across the
# connector-specific runners. Connector-specific logic (kv-transfer JSON,
# engine env vars, router launch flags/ordering, package installs) stays in the
# individual scripts.
#
# Contract with the sourcing script:
#   - The sourcing script owns `set -Eeuo pipefail` and installs the traps
#     (`trap pd_rocm_cleanup EXIT`). This library only defines the functions
#     those traps call; it installs no traps of its own.
#   - All default assignments use `:=`, which is safe under `set -u`.
#   - The sourcing script assigns PREFILL_PID / DECODE_PID / ROUTER_PID as it
#     launches each component; they are pre-initialized here so the cleanup
#     trap is safe even if it fires before a launch.
#   - The sourcing script must set LOG_DIR (its default value differs per
#     connector) before installing the trap.

# Guard against double-sourcing.
if [[ -n "${_PD_ROCM_COMMON_SOURCED:-}" ]]; then
  return 0
fi
_PD_ROCM_COMMON_SOURCED=1

# Directory holding this helper (the rocm/ subfolder). The shared test_*.py
# accuracy harnesses live one level up, in the parent pd_disagg_vllm/ dir.
_PD_ROCM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Shared configuration defaults (identical values across both runners).
# ---------------------------------------------------------------------------
: "${MODEL_NAMES:=Qwen/Qwen3-0.6B}"
: "${PREFILLER_TP_SIZE:=1}"
: "${DECODER_TP_SIZE:=1}"
: "${GPU_MEMORY_UTILIZATION:=0.6}"
: "${BLOCK_SIZE:=16}"
: "${ENGINE_STARTUP_TIMEOUT:=900}"
: "${ROUTER_STARTUP_TIMEOUT:=180}"
: "${NUM_SANITY_REQUESTS:=20}"
: "${NUM_LM_EVAL_CONCURRENT:=10}"
: "${LM_EVAL_LIMIT:=500}"
: "${LM_EVAL_LOG_SAMPLES:=false}"
: "${PREFILL_PORT:=8100}"
: "${DECODE_PORT:=8200}"
: "${ROUTER_PORT:=8300}"

# Process IDs the cleanup trap tears down. The sourcing script overwrites these
# as it launches each component.
PREFILL_PID=${PREFILL_PID:-""}
DECODE_PID=${DECODE_PID:-""}
ROUTER_PID=${ROUTER_PID:-""}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
pd_rocm_require_commands() {
  local command_name
  for command_name in "$@"; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
      echo "ERROR: required command '${command_name}' is unavailable" >&2
      exit 2
    fi
  done
}

# ---------------------------------------------------------------------------
# Process teardown + cleanup trap
# ---------------------------------------------------------------------------
pd_rocm_terminate_process_tree() {
  local pid=$1
  local child

  [[ -n "${pid}" ]] || return 0
  kill -0 "${pid}" 2>/dev/null || return 0

  while read -r child; do
    [[ -n "${child}" ]] && pd_rocm_terminate_process_tree "${child}"
  done < <(ps -o pid= --ppid "${pid}" 2>/dev/null || true)

  kill -TERM "${pid}" 2>/dev/null || true
}

pd_rocm_show_logs() {
  local log_file
  for log_file in "${LOG_DIR}"/*.log; do
    [[ -e "${log_file}" ]] || continue
    echo "===== ${log_file} (last 200 lines) ====="
    tail -n 200 "${log_file}" || true
  done
}

pd_rocm_cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  set +e

  if (( exit_code != 0 )); then
    pd_rocm_show_logs
  fi

  pd_rocm_terminate_process_tree "${PREFILL_PID}"
  pd_rocm_terminate_process_tree "${DECODE_PID}"
  pd_rocm_terminate_process_tree "${ROUTER_PID}"

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

# ---------------------------------------------------------------------------
# Health polling
# ---------------------------------------------------------------------------
pd_rocm_wait_for_health() {
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

# ---------------------------------------------------------------------------
# Router binary resolution (sets the ROUTER_BIN global)
# ---------------------------------------------------------------------------
pd_rocm_resolve_router_bin() {
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
}

# ---------------------------------------------------------------------------
# GPU detection + prefill/decode GPU split
# Sets: AVAILABLE_GPUS, ROCM_VERSION, PREFILL_GPUS, DECODE_GPUS
# ---------------------------------------------------------------------------
pd_rocm_detect_gpus() {
  read -r AVAILABLE_GPUS ROCM_VERSION < <(
    python3 - <<'PY'
import torch

if not torch.version.hip:
    raise SystemExit("ERROR: PyTorch does not report a ROCm runtime")
print(torch.cuda.device_count(), torch.version.hip)
PY
  )

  local required_gpus=$((PREFILLER_TP_SIZE + DECODER_TP_SIZE))
  if (( AVAILABLE_GPUS < required_gpus )); then
    echo "ERROR: ${required_gpus} GPUs are required, but PyTorch sees ${AVAILABLE_GPUS}" >&2
    exit 2
  fi

  local prefill_gpu_end=$((PREFILLER_TP_SIZE - 1))
  local decode_gpu_start=${PREFILLER_TP_SIZE}
  local decode_gpu_end=$((required_gpus - 1))
  PREFILL_GPUS=$(seq -s, 0 "${prefill_gpu_end}")
  DECODE_GPUS=$(seq -s, "${decode_gpu_start}" "${decode_gpu_end}")
}

# ---------------------------------------------------------------------------
# Accuracy harness: sanity completions + bounded GSM8K evaluation
# ---------------------------------------------------------------------------
pd_rocm_run_accuracy() {
  local router_port=$1

  python3 "${_PD_ROCM_DIR}/../test_pd_accuracy.py" \
    --router-url "http://127.0.0.1:${router_port}" \
    --model "${MODEL_NAMES}" \
    --num-requests "${NUM_SANITY_REQUESTS}" \
    --skip-streaming

  local lm_eval_args=(
    --router-url "http://127.0.0.1:${router_port}"
    --model "${MODEL_NAMES}"
    --num-concurrent "${NUM_LM_EVAL_CONCURRENT}"
    --max-gen-toks 512
    --limit "${LM_EVAL_LIMIT}"
    --filter "exact_match,flexible-extract"
    --disable-thinking
  )
  if [[ "${LM_EVAL_LOG_SAMPLES}" == "true" ]]; then
    lm_eval_args+=(--log-samples)
  fi
  python3 "${_PD_ROCM_DIR}/../test_lm_eval_accuracy.py" "${lm_eval_args[@]}"
}
