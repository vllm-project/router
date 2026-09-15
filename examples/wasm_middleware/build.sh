#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

rustup target add wasm32-wasip2 >/dev/null
cargo build --release --target wasm32-wasip2

# wit-bindgen + wasm32-wasip2 already emits a component.
cp -f target/wasm32-wasip2/release/wasm_middleware_example.wasm \
  ./wasm_middleware_example.component.wasm

echo "Built $(pwd)/wasm_middleware_example.component.wasm"
