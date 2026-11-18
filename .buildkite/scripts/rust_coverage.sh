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

# Never leave a previous run's report at the artifact path if setup fails.
report_dir="target/coverage/rust-$1"
rm -f -- "${report_dir}/lcov.info"

# coverage(off) is unstable. Pin nightly only for coverage; ordinary CI stays stable.
coverage_toolchain="nightly-2026-04-15"
rustup toolchain install "${coverage_toolchain}" --profile minimal --no-self-update \
  --component llvm-tools-preview
export RUSTUP_TOOLCHAIN="${coverage_toolchain}"
host_target="$(rustc -vV | sed -n 's/^host: //p')"
coverage_version="0.6.21"
cargo_bin="${CARGO_HOME:-$HOME/.cargo}/bin"
mkdir -p "${cargo_bin}"
curl --fail --silent --show-error --location --connect-timeout 15 --retry 3 \
  "https://github.com/taiki-e/cargo-llvm-cov/releases/download/v${coverage_version}/cargo-llvm-cov-${host_target}.tar.gz" \
  | tar -xz -C "${cargo_bin}" cargo-llvm-cov
export PATH="${cargo_bin}:${PATH}"
cargo llvm-cov --version

mkdir -p "${report_dir}"

# Remove stale workspace binaries as well as profiles, including other toolchains
# and test groups. Dependencies remain cached.
cargo llvm-cov clean --workspace

# Preserve test failure while still attempting to report completed tests.
# Target-specific flags keep instrumentation out of nested WASM builds.
test_status=0
cargo llvm-cov --locked --features cpu,ready --no-report \
  --target "${host_target}" --coverage-target-only \
  --no-fail-fast "${targets[@]}" || test_status=$?

# Measure production Rust code rather than test and benchmark harnesses.
report_tmp="$(mktemp "${report_dir}/.lcov.info.XXXXXX")"
trap 'rm -f -- "${report_tmp}"' EXIT
cargo llvm-cov report --target "${host_target}" \
  --ignore-filename-regex '(^|/)(tests|benches)/' --lcov \
  --output-path "${report_tmp}"
# Allow the host agent to read the report before publishing the final artifact.
chmod 0644 -- "${report_tmp}"
mv -- "${report_tmp}" "${report_dir}/lcov.info"

exit "${test_status}"
