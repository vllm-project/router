#!/usr/bin/env bash
# Official rust HTTP frontend: Python CLI finds wheel `vllm-rs` and runs
# `vllm-rs frontend`. Not the same as `vllm-rs serve --grpc-port`.
# Extra flags go to `vllm serve`.
#
#   export MODEL=/path/or/hf-id
#   ./serve_rust_http.sh --max-model-len 8192
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"
need_model
command -v vllm >/dev/null || {
  echo "vllm not on PATH (activate the vLLM 0.29 env)" >&2
  exit 1
}

H="$(host)"
WP="$(worker_port)"
echo "rust-http: VLLM_USE_RUST_FRONTEND=1 vllm serve $MODEL --host $H --port $WP $*" >&2
VLLM_USE_RUST_FRONTEND=1 vllm serve "$MODEL" --host "$H" --port "$WP" "$@" &
WORKER=$!
trap 'kill "$WORKER" 2>/dev/null || true' EXIT
sleep 2
start_router "http://${H}:${WP}"
