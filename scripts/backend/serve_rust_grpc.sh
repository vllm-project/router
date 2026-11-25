#!/usr/bin/env bash
# Standalone wheel binary: `vllm-rs serve --grpc-port` (GenerateStream).
# `VLLM_USE_RUST_FRONTEND=1 vllm serve` cannot set this port.
# Extra flags go to `vllm-rs serve` (use `--` before engine-only flags).
#
#   export MODEL=/path/or/hf-id
#   ./serve_rust_grpc.sh -- --tensor-parallel-size 1
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"
need_model

RS="$(vllm_rs_bin)"
if [[ ! -x "$RS" ]]; then
  echo "vllm-rs not executable at $RS (need vllm 0.29 with the rust binary)" >&2
  exit 1
fi

H="$(host)"
WP="$(worker_port)"
GP="$(grpc_port)"
echo "rust-grpc: $RS serve $MODEL --host $H --port $WP --grpc-port $GP $*" >&2
"$RS" serve "$MODEL" --host "$H" --port "$WP" --grpc-port "$GP" "$@" &
WORKER=$!
trap 'kill "$WORKER" 2>/dev/null || true' EXIT
sleep 2
start_router "grpc://${H}:${GP}"
