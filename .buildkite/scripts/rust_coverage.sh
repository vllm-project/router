#!/usr/bin/env bash
# Run one existing Rust CI test group and generate an LCOV artifact.
set -euo pipefail

case "${1:-}" in
  unit) targets=(--lib --bins) ;;
  integration) targets=(--test '*') ;;
  *) echo "usage: rust_coverage.sh <unit|integration>" >&2; exit 2 ;;
esac
if [ "$#" -ne 1 ]; then
  echo "usage: rust_coverage.sh <unit|integration>" >&2
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Keep LLVM tools matched to the active Rust toolchain.
rustup component add llvm-tools-preview
cargo install cargo-llvm-cov --version 0.6.21 --locked

report_dir="target/coverage/rust-$1"
mkdir -p "${report_dir}"

# llvm-cov cleans old profiles by default. Never reuse another run's coverage.
# Preserve test failure while still attempting to report completed tests.
# Target-specific flags keep instrumentation out of nested WASM builds.
host_target="$(rustc -vV | sed -n 's/^host: //p')"
test_status=0
cargo llvm-cov --locked --features cpu,ready --no-report \
  --target "${host_target}" --coverage-target-only \
  --no-fail-fast "${targets[@]}" || test_status=$?

# Measure production Rust code rather than test and benchmark harnesses.
cargo llvm-cov report --target "${host_target}" \
  --ignore-filename-regex '(^|/)(tests|benches)/' --lcov \
  --output-path "${report_dir}/lcov.info"

exit "${test_status}"
