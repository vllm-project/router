# Shared env for scripts/backend/*.sh — sourced, not executed.
# No Slurm, Docker, GPU reclaim, or hardcoded model paths.

need_model() {
  if [[ -z "${MODEL:-}" ]]; then
    echo "set MODEL to a HuggingFace id or local model directory" >&2
    exit 1
  fi
}

host() { echo "${HOST:-127.0.0.1}"; }
worker_port() { echo "${WORKER_PORT:-8000}"; }
router_port() { echo "${ROUTER_PORT:-30000}"; }
grpc_port() { echo "${GRPC_PORT:-50051}"; }

router_bin() {
  if [[ -n "${ROUTER_BIN:-}" ]]; then
    echo "$ROUTER_BIN"
    return
  fi
  if command -v vllm-router >/dev/null 2>&1; then
    command -v vllm-router
    return
  fi
  local hint
  hint="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/target/release/vllm-router"
  if [[ -x "$hint" ]]; then
    echo "$hint"
    return
  fi
  echo "vllm-router not on PATH and $hint missing; set ROUTER_BIN" >&2
  exit 1
}

# Same file VLLM_USE_RUST_FRONTEND=1 discovers (vllm==0.29 wheel).
# Order: VLLM_RS, PATH `vllm-rs`, then site-packages/vllm/vllm-rs.
vllm_rs_bin() {
  if [[ -n "${VLLM_RS:-}" ]]; then
    echo "$VLLM_RS"
    return
  fi
  if command -v vllm-rs >/dev/null 2>&1; then
    command -v vllm-rs
    return
  fi
  python -c "import os, vllm; print(os.path.join(os.path.dirname(vllm.__file__), 'vllm-rs'))"
}

start_router() {
  local worker_url=$1
  local rp host_
  rp="$(router_port)"
  host_="$(host)"
  # VLLM_ROUTER_MODEL is the load key (wins). VLLM_ROUTER_TOKENIZER is
  # a last-resort unused fallback in preprocess if MODEL is unset.
  export VLLM_ROUTER_MODEL="${VLLM_ROUTER_MODEL:-$MODEL}"
  if [[ -d "$MODEL" && -f "$MODEL/tokenizer.json" ]]; then
    export VLLM_ROUTER_TOKENIZER="${VLLM_ROUTER_TOKENIZER:-$MODEL/tokenizer.json}"
  fi
  local extra=()
  if [[ -n "${PROMETHEUS_PORT:-}" ]]; then
    extra+=(--prometheus-port "$PROMETHEUS_PORT")
  fi
  echo "router: $(router_bin) --host $host_ --port $rp --worker-urls $worker_url ${extra[*]}" >&2
  "$(router_bin)" --host "$host_" --port "$rp" --worker-urls "$worker_url" "${extra[@]}"
}
