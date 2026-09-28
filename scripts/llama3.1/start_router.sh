#!/bin/bash

# Router configuration for Llama 3.1 Prefill-Decode Disaggregation with NIXL Connector
# This script starts the vLLM router with static prefill and decode URLs
#
# Basic configuration (DP-aware mode disabled):
# - Prefill and Decode servers running with --intra-node-data-parallel-size 2
# - Router uses --intra-node-data-parallel-size 1 (DP-aware mode disabled)
# - Router treats each server as a single endpoint

echo "Starting router for Prefill-Decode disaggregation"
echo "Configuration:"
echo "  Policy: round_robin"
echo "  Data Parallel Size: 1 (DP-aware mode disabled)"
echo "  Prefill: http://127.0.0.1:8081"
echo "  Decode: http://127.0.0.1:8082"
echo "  Router port: 8090"
echo ""

# Start the router with static prefill/decode URLs
# Using data-parallel-size 1 to disable DP-aware URL expansion
# This treats each server as a single endpoint regardless of their internal DP configuration
cargo run --release -- \
    --policy round_robin \
    --vllm-pd-disaggregation \
    --prefill http://127.0.0.1:8081 \
    --decode http://127.0.0.1:8082 \
    --host 127.0.0.1 \
    --port 8090 \
    --intra-node-data-parallel-size 2 \
    --log-level debug
