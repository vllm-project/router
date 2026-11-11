#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

rustup target add wasm32-wasip2 >/dev/null
cargo build --release --target wasm32-wasip2

# Honor CARGO_TARGET_DIR (CI and shared build caches set it) instead of
# assuming the crate-local target directory.
target_dir="${CARGO_TARGET_DIR:-target}"

# wit-bindgen + wasm32-wasip2 already emits a component.
cp -f "$target_dir/wasm32-wasip2/release/wasm_middleware_example.wasm" \
  ./wasm_middleware_example.component.wasm

echo "Built $(pwd)/wasm_middleware_example.component.wasm"
