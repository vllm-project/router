#!/bin/bash

# Router configuration for Llama 3.1 Prefill-Decode Disaggregation with NIXL Connector
# This script starts the vLLM router with static prefill and decode URLs
#
# Testing DP=4 configuration:
# - Prefill and Decode servers are running with --data-parallel-size 4
# - Router uses --data-parallel-size 4 to automatically create DP-aware workers

cd /home/congc/gitrepos/router

echo "Starting router with data parallelism (DP=4)"
echo "Configuration:"
echo "  Policy: consistent_hash"
echo "  Data Parallel Size: 4"
echo "  Prefill: http://127.0.0.1:8081 (DP=4)"
echo "  Decode: http://127.0.0.1:8082 (DP=4)"
echo "  Router port: 8090"
echo ""

# Start the router with static prefill/decode URLs
# When data-parallel-size > 1, router automatically creates DP-aware workers
# Using pre-built binary directly instead of "cargo run --release"
#  ./target/release/vllm-router \
cargo run --release -- \
    --policy round_robin \
    --vllm-pd-disaggregation \
    --prefill http://127.0.0.1:8081 \
    --decode http://127.0.0.1:8082 \
    --host 127.0.0.1 \
    --port 8090 \
    --data-parallel-size 2 \
    --log-level debug
