# Llama 3.1 8B Multi-Node Data Parallelism

Production-ready vLLM multi-node data parallelism (DP) deployment with explicit master-worker coordination.

## Overview

This setup implements **vLLM's multi-node DP pattern** where:
- **Rank 0 (Master)**: Receives ALL client requests and runs DP Coordinator
- **Rank 1 (Worker)**: Connects to master via RPC, provides additional compute
- **Total capacity**: 2 ranks with 16 GPUs total (8 per node)
- **Request flow**: Client → Rank 0 → DP Coordinator → All workers across both ranks

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Client                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Rank 0 Service (Port 8000 + RPC 13345)            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Rank 0 Pod (Master)                       │
│  ┌─────────────────────────────────────────────────────────┐│
│  │             vLLM DP Coordinator                          ││
│  │  • Receives all HTTP requests                            ││
│  │  • Distributes work across all 16 workers               ││
│  │  • 8 local GPU workers                                   ││
│  └─────────────────────────────────┬───────────────────────┘│
│                                     │ RPC Port 13345          │
└─────────────────────────────────────┼───────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Rank 1 Pod (Worker)                       │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              vLLM Worker Process                         ││
│  │  • Connects to rank 0 via RPC                            ││
│  │  • No HTTP API (work received via RPC only)              ││
│  │  • 8 remote GPU workers                                  ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## How It Works

1. **Client sends request** → HTTP request to rank 0 on port 8000
2. **Rank 0 receives request** → Logged in rank 0's HTTP API layer
3. **DP Coordinator distributes work** → Splits work across all 16 GPU workers (8 local + 8 remote)
4. **Rank 1 processes work** → Receives work via RPC (no HTTP logs), GPUs at 80%+ utilization
5. **Response aggregated** → Rank 0 collects results and returns to client

## Quick Start

### Prerequisites

- Kubernetes cluster with 2 nodes (8 GPUs each)
- `kubectl` and `helmfile` installed
- HuggingFace token with Llama 3.1 access

### 1. Create Secret

```bash
kubectl create namespace llm-d-llama31-multinode
kubectl create secret generic llm-d-hf-token \
  --from-literal=HF_TOKEN=hf_xxxxxxxxxxxxx \
  -n llm-d-llama31-multinode
```

### 2. Deploy

```bash
cd /data/users/nlalit/gitrepos/router/scripts/k8s/llama3.1_multinode
./deploy.sh
```

Wait for pods to be ready (5-10 minutes first time for model download, ~2 minutes after).

### 3. Port Forward (Required for Testing)

In a separate terminal, set up port forwarding to rank 0:

```bash
kubectl port-forward -n llm-d-llama31-multinode \
  $(kubectl get pod -n llm-d-llama31-multinode -l llm-d.ai/role=decode -o jsonpath='{.items[0].metadata.name}') \
  8000:8000
```

Keep this running while you test, benchmark, or evaluate.

### 4. Verify Deployment

```bash
# Check pods are on different nodes
kubectl get pods -n llm-d-llama31-multinode -o wide

# Test connection
curl http://localhost:8000/v1/models
```

### 5. Run Benchmark

```bash
# Basic benchmark (100 prompts, concurrency 16)
./run-benchmark.sh

# Custom configuration
./run-benchmark.sh 1000 32
```

### 6. Run Evaluation

```bash
# Run gsm8k (default)
./run-eval.sh

# Run specific task with concurrency
./run-eval.sh mmlu 4

# Quick test with limited samples
./run-eval.sh hellaswag 2 50
```

## Deployment Flow Summary

```bash
# 1. Deploy
./deploy.sh

# 2. Wait for ready
kubectl get pods -n llm-d-llama31-multinode -w

# 3. Port forward (in separate terminal)
kubectl port-forward -n llm-d-llama31-multinode \
  $(kubectl get pod -n llm-d-llama31-multinode -l llm-d.ai/role=decode -o jsonpath='{.items[0].metadata.name}') \
  8000:8000

# 4. Benchmark
./run-benchmark.sh

# 5. Evaluate
./run-eval.sh gsm8k 1
```

## Debugging

### Check Pods are on Different Nodes

```bash
kubectl get pods -n llm-d-llama31-multinode -o wide
```

**Expected**: 2 pods (decode and prefill) on different nodes

### Check Service Was Created

```bash
kubectl get svc -n llm-d-llama31-multinode ms-llama31-multinode-llm-d-modelservice-decode
```

**Expected**: Service with ports `8000/TCP,13345/TCP`

### Check Service Has Endpoint

```bash
kubectl get endpoints -n llm-d-llama31-multinode ms-llama31-multinode-llm-d-modelservice-decode
```

**Expected**: One endpoint pointing to decode pod IP

### Check Rank 0 (Master) Logs

```bash
kubectl logs -n llm-d-llama31-multinode \
  $(kubectl get pod -n llm-d-llama31-multinode -l llm-d.ai/role=decode -o jsonpath='{.items[0].metadata.name}') \
  -c vllm | tail -50
```

**Expected**: Should see "Uvicorn running" or "Application startup complete" (NOT stuck waiting)

### Check Rank 1 (Worker) Logs

```bash
kubectl logs -n llm-d-llama31-multinode \
  $(kubectl get pod -n llm-d-llama31-multinode -l llm-d.ai/role=prefill -o jsonpath='{.items[0].metadata.name}') \
  -c vllm | tail -50
```

**Expected**: Should see successful connection to rank 0

### Verify Multi-Node DP is Working

Check GPU utilization on both pods during benchmark/eval:

```bash
# Get pod names
DECODE_POD=$(kubectl get pod -n llm-d-llama31-multinode -l llm-d.ai/role=decode -o jsonpath='{.items[0].metadata.name}')
PREFILL_POD=$(kubectl get pod -n llm-d-llama31-multinode -l llm-d.ai/role=prefill -o jsonpath='{.items[0].metadata.name}')

# Check GPU usage on rank 0
kubectl exec -n llm-d-llama31-multinode $DECODE_POD -c vllm -- \
  nvidia-smi --query-gpu=utilization.gpu --format=csv

# Check GPU usage on rank 1
kubectl exec -n llm-d-llama31-multinode $PREFILL_POD -c vllm -- \
  nvidia-smi --query-gpu=utilization.gpu --format=csv
```

**Expected**: Both pods should show GPU utilization > 0% (typically 60-90%) during workload

**Important Note**: Rank 1 will NOT show HTTP request logs (normal behavior). It only processes work via RPC. GPU utilization is the proof that multi-node DP is working.

## Cleanup

```bash
./cleanup.sh

# To delete everything including cached model
kubectl delete namespace llm-d-llama31-multinode
```

## Troubleshooting

### Pods Not Scheduling

```bash
kubectl describe pod -n llm-d-llama31-multinode <pod-name>
```

Check for: GPU availability, node resources, anti-affinity conflicts

### Rank 1 Can't Connect to Rank 0

```bash
# Verify service exists
kubectl get svc -n llm-d-llama31-multinode ms-llama31-multinode-llm-d-modelservice-decode

# Check rank 1 logs for connection errors
kubectl logs -n llm-d-llama31-multinode \
  $(kubectl get pod -n llm-d-llama31-multinode -l llm-d.ai/role=prefill -o jsonpath='{.items[0].metadata.name}') \
  -c vllm | grep -i "error\|connection"
```

### vLLM Not Ready

```bash
kubectl logs -n llm-d-llama31-multinode \
  $(kubectl get pod -n llm-d-llama31-multinode -l llm-d.ai/role=decode -o jsonpath='{.items[0].metadata.name}') \
  -c vllm | grep -i error
```

## Key Configuration

- **2 ranks** (pods): Each pod is a separate vLLM instance
- **8 GPUs per rank**: Configured via `parallelism.data: 8`
- **Total**: 16 GPUs coordinated across 2 nodes
- **Load balancing**: vLLM DP coordinator distributes work automatically
- **Model caching**: PVC shared between pods for fast redeployment

## References

- [vLLM Multi-Node DP Documentation](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html)
- Helm chart: `llm-d-modelservice` v0.2.11

---

**Built for production-grade multi-node vLLM deployments with true master-worker coordination.**
