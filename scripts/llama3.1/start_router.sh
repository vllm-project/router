#!/bin/bash

# Router configuration for Llama 3.1 Prefill-Decode Disaggregation with NIXL Connector
# This script starts the vLLM router with static prefill and decode URLs
#
# Testing DP=4 configuration:
# - Prefill and Decode servers are running with --data-parallel-size 4
# - First test: WITHOUT --dp-aware flag (to establish baseline)
# - Next test: WITH --dp-aware flag

cd /home/congc/gitrepos/router

# Start the router with static prefill/decode URLs
# NOTE: --dp-aware flag is currently disabled for baseline testing
cargo run --release -- \
    --policy consistent_hash \
    --vllm-pd-disaggregation \
    --prefill http://127.0.0.1:8081 \
    --decode http://127.0.0.1:8082 \
    --host 127.0.0.1 \
    --port 8090 \
    --log-level debug
