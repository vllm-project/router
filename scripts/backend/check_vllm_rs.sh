#!/usr/bin/env bash
# Resolve vllm-rs (VLLM_RS → PATH → wheel). Print path, --version, and a
# copy-paste export line. Does not change the caller's shell.
#
#   ./check_vllm_rs.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

if [[ $# -gt 0 ]]; then
  echo "usage: check_vllm_rs.sh  (no flags; prints path, version, export hint)" >&2
  exit 1
fi

BIN="$(vllm_rs_bin)"
if [[ ! -x "$BIN" ]]; then
  echo "vllm-rs not found (tried VLLM_RS, PATH, wheel)." >&2
  exit 1
fi

echo "vllm-rs: $BIN"
"$BIN" --version || echo "(no --version; still usable if EngineCore matches)" >&2
echo "to pin for serve_*.sh:  export VLLM_RS=$BIN"
