#!/bin/bash
set -euo pipefail

# Install system dependencies (manylinux_2_28 uses dnf)
dnf install -y openssl-devel protobuf-compiler protobuf-devel

# Install Rust toolchain (explicitly use nightly to match previous pipeline)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
source $HOME/.cargo/env

# Build Rust binary for release
cargo build --release

# Build Python wheels using Python 3.9 (available in manylinux image)
/opt/python/cp39-cp39/bin/pip install -U pip setuptools wheel setuptools-rust build auditwheel
/opt/python/cp39-cp39/bin/python -m build

# Repair wheel to have proper manylinux tags
/opt/python/cp39-cp39/bin/auditwheel repair dist/*.whl -w dist/
# Remove the original non-manylinux wheel
rm dist/*-linux_*.whl || true

