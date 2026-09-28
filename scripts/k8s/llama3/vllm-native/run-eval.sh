#!/bin/bash

# Run LM evaluation using lm_eval harness against multi-node DP deployment
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
BASE_URL="http://localhost:8000/v1/completions"

echo "============================================"
echo "Running LM Eval - Llama 3.1 Multi-Node DP"
echo "============================================"
echo "Task: $TASK"
echo "Model: $MODEL"
echo "Base URL: $BASE_URL"
echo "Concurrent Requests: $NUM_CONCURRENT"
if [ -n "$LIMIT" ]; then
    echo "Sample Limit: $LIMIT"
fi
echo ""
echo "Note: Requests go to Rank 0 (Master)"
echo "      DP Coordinator distributes work to all ranks"
echo ""

# Check if lm_eval is installed
if ! command -v lm_eval &> /dev/null; then
    echo "Error: lm_eval is not installed"
    echo "Install it with: pip install lm-eval"
    exit 1
fi

# Check if HF_TOKEN is set (needed for Llama 3.1 tokenizer)
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable not set"
    echo "Llama 3.1 is a gated model. You may need to:"
    echo "  export HF_TOKEN=your_token_here"
    echo "  OR run: huggingface-cli login"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if port-forward is running
echo "Checking connection to $BASE_URL..."
if ! curl -s -f "$BASE_URL/../models" > /dev/null 2>&1; then
    echo ""
    echo "Error: Cannot connect to $BASE_URL"
    echo ""
    echo "Make sure port-forward is running to Rank 0 (decode pod):"
    echo "  kubectl port-forward -n llm-d-llama31-multinode \\"
    echo "    \$(kubectl get pod -n llm-d-llama31-multinode -l llm-d.ai/role=decode -o jsonpath='{.items[0].metadata.name}') \\"
    echo "    8000:8000"
    echo ""
    exit 1
fi

echo "✓ Connection successful to Rank 0 (Master)"
echo ""
echo "Starting evaluation..."
echo "This may take a while depending on the task size..."
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

echo ""
echo "============================================"
echo "Evaluation completed!"
echo "============================================"
echo ""
echo "Multi-Node DP Info:"
echo "  - All requests sent to Rank 0 (Master)"
echo "  - DP Coordinator distributed work across all ranks"
echo "  - Both Rank 0 and Rank 1 GPUs were utilized"
echo ""
