#!/bin/bash
# VLLM Router startup script for Prefill/Decode disaggregation

echo "Starting VLLM Router"
echo "Router IP: 10.0.1.77"
echo "HTTP Port: 10001"
echo "Discovery Port: 30001"
echo "P/D servers should use proxy_ip=10.0.1.77 and proxy_port=30001"
echo

cargo run --release -- \
    --vllm-pd-disaggregation \
    --vllm-discovery-address 10.0.1.77:30001 \
    --host 10.0.1.77 \
    --port 10001 \
    --prefill-policy consistent_hash \
    --decode-policy consistent_hash
