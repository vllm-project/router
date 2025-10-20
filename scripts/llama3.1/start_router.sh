#!/bin/bash

# Router configuration for Llama 3.1 Prefill-Decode Disaggregation with NIXL Connector
# This script starts the vLLM router with static prefill and decode URLs
#
# Testing DP=4 configuration:
# - Prefill and Decode servers are running with --data-parallel-size 4
# - Baseline test completed successfully WITHOUT --dp-aware flag
# - Now testing WITH --dp-aware flag enabled

cd /home/congc/gitrepos/router

echo "Starting router with DP-aware scheduling enabled"
echo "Configuration:"
echo "  Policy: consistent_hash"
echo "  DP-aware: ENABLED"
echo "  Prefill: http://127.0.0.1:8081 (DP=4)"
echo "  Decode: http://127.0.0.1:8082 (DP=4)"
echo "  Router port: 8090"
echo ""

# Start the router with static prefill/decode URLs and DP-aware flag
# Using pre-built binary directly instead of "cargo run --release"
# cargo run --release -- \
 ./target/release/vllm-router \
    --policy round_robin \
    --vllm-pd-disaggregation \
    --prefill http://127.0.0.1:8081 \
    --decode http://127.0.0.1:8082 \
    --host 127.0.0.1 \
    --port 8090 \
    --dp-aware \
    --log-level debug
