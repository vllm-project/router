#!/usr/bin/env bash
# TP Configuration Sweep for P/D Disaggregation Accuracy Tests
# Runs integration tests sequentially with varying TP configurations

set -euo pipefail

# Script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/run_accuracy_test.sh"

# Define test configurations
# Format: "ENV_VAR=value ENV_VAR2=value2 ..."
configs=(
  # TP=2 for both prefill and decode
  "GPU_MEMORY_UTILIZATION=0.6 PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=2"

  # TP=1 for prefill, TP=2 for decode (asymmetric)
  "GPU_MEMORY_UTILIZATION=0.6 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2"

  # Single TP with smaller model (baseline)
  "GPU_MEMORY_UTILIZATION=0.8 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=1"

  # Test with larger block size
  "GPU_MEMORY_UTILIZATION=0.6 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=1 PREFILL_BLOCK_SIZE=256 DECODE_BLOCK_SIZE=256"
)

run_tests() {
  local label=$1
  local extra_env=$2

  echo "=== Running P/D Disaggregation Tests (${label}) ==="
  echo ""

  for cfg in "${configs[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "→ Running with: ${cfg} ${extra_env:+and ${extra_env}}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Use 'env' to safely set variables without eval
    if ! env ${extra_env} ${cfg} bash "${TEST_SCRIPT}"; then
      echo ""
      echo "❌ Test FAILED for config: ${cfg} ${extra_env:+(${extra_env})}"
      echo ""
      exit 1
    fi

    echo ""
    echo "✅ Test PASSED for config: ${cfg}"
    echo ""
  done

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "✅ All ${label} tests PASSED!"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
}

# Main execution
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   P/D Disaggregation TP Configuration Sweep Test Suite    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Run tests with default configuration
run_tests "default" ""

# Check if VLLM_ATTENTION_BACKEND should be tested
if [[ -n "${TEST_FLASHINFER:-}" ]]; then
  echo ""
  echo "TEST_FLASHINFER is set, rerunning with VLLM_ATTENTION_BACKEND=FLASHINFER"
  run_tests "FLASHINFER backend" "VLLM_ATTENTION_BACKEND=FLASHINFER"
else
  echo "TEST_FLASHINFER not set, skipping FLASHINFER runs."
  echo "(Set TEST_FLASHINFER=1 to enable)"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║        🎉 All P/D Disaggregation Tests PASSED! 🎉         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

exit 0
