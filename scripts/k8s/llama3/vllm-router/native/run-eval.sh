#!/bin/bash

# Run LM evaluation using lm_eval harness for Llama 3.1 with vllm-router (native load balancing)
# Usage: ./run-eval.sh [task] [num_concurrent] [limit]
#
# Examples:
#   ./run-eval.sh                    # Run gsm8k with defaults
#   ./run-eval.sh mmlu 4 50          # Run mmlu with 4 concurrent, limit 50 samples
#   ./run-eval.sh hellaswag 1        # Run hellaswag with 1 concurrent, no limit

set -e

TASK="${1:-gsm8k}"
NUM_CONCURRENT="${2:-1}"
LIMIT="${3:-}"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
NAMESPACE="vllm-router-llama31"
ROUTER_PORT="${ROUTER_PORT:-10001}"
BASE_URL="http://localhost:$ROUTER_PORT/v1/completions"

echo "============================================"
echo "Running LM Eval - vllm-router (Native)"
echo "============================================"
echo "Task: $TASK"
echo "Model: $MODEL"
echo "Concurrent Requests: $NUM_CONCURRENT"
if [ -n "$LIMIT" ]; then
    echo "Sample Limit: $LIMIT"
fi
echo "Router Port: $ROUTER_PORT"
echo ""

# Check if lm_eval is installed
if ! command -v lm_eval &> /dev/null; then
    echo "Error: lm_eval is not installed"
    echo "Install it with: pip install lm-eval"
    exit 1
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable not set"
    echo "You may need to: export HF_TOKEN=your_token_here"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Set up port forward to vllm-router
echo "Setting up port forward to vllm-router on local port $ROUTER_PORT..."
kubectl port-forward -n "$NAMESPACE" \
    svc/vllm-router-llama31 "$ROUTER_PORT":10001 &
PF_PID=$!
sleep 3
echo "✓ Port forward established (PID: $PF_PID)"

# Check connection
echo ""
echo "Checking connection to vllm-router..."
HEALTH_URL="http://localhost:$ROUTER_PORT/health"
if ! curl -s -f "$HEALTH_URL" > /dev/null 2>&1; then
    echo "Error: Cannot connect to vllm-router at $HEALTH_URL"
    kill $PF_PID 2>/dev/null || true
    exit 1
fi

echo "✓ Connection successful"
echo ""
echo "Starting evaluation..."
echo ""

# Build the lm_eval command
EVAL_CMD="lm_eval --model local-completions \
    --tasks $TASK \
    --model_args model=$MODEL,base_url=$BASE_URL,num_concurrent=$NUM_CONCURRENT,max_retries=3,tokenized_requests=False"

# Add limit if specified
if [ -n "$LIMIT" ]; then
    EVAL_CMD="$EVAL_CMD --limit $LIMIT"
fi

# Run the evaluation
eval "$EVAL_CMD"

# Cleanup port forward
echo ""
echo "Cleaning up port forward..."
kill $PF_PID 2>/dev/null || true

echo ""
echo "============================================"
echo "Evaluation completed!"
echo "============================================"
echo ""
echo "Note: Evaluation ran through vllm-router"
echo "      Requests load balanced across 2 decode pods (full inference)"
