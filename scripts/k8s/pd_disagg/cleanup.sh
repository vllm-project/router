#!/bin/bash

# Cleanup vLLM Router deployment
# Usage: ./cleanup.sh [pod|deployment|all]

set -e

NAMESPACE="llm-d-pd"
MODE="${1:-all}"

echo "==================================="
echo "Cleaning up vLLM Router (PD Mode)"
echo "==================================="
echo "Namespace: $NAMESPACE"
echo "Mode: $MODE"
echo ""

case "$MODE" in
    pod)
        echo "Deleting router pod..."
        kubectl delete pod vllm-router-pd -n "$NAMESPACE" --ignore-not-found=true
        ;;

    deployment)
        echo "Deleting router deployment and services..."
        kubectl delete deployment vllm-router-pd -n "$NAMESPACE" --ignore-not-found=true
        kubectl delete svc vllm-router-pd -n "$NAMESPACE" --ignore-not-found=true
        kubectl delete svc vllm-router-pd-nodeport -n "$NAMESPACE" --ignore-not-found=true
        ;;

    all)
        echo "Deleting all router resources..."
        kubectl delete pod vllm-router-pd -n "$NAMESPACE" --ignore-not-found=true
        kubectl delete deployment vllm-router-pd -n "$NAMESPACE" --ignore-not-found=true
        kubectl delete svc vllm-router-pd -n "$NAMESPACE" --ignore-not-found=true
        kubectl delete svc vllm-router-pd-nodeport -n "$NAMESPACE" --ignore-not-found=true
        ;;

    *)
        echo "Error: Invalid mode. Use 'pod', 'deployment', or 'all'"
        exit 1
        ;;
esac

echo ""
echo "Cleanup completed!"
