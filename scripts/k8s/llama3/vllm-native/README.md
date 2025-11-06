# Llama 3.1 8B Multi-Node Data Parallelism (16 Pods)

Production-ready vLLM multi-node data parallelism (DP) deployment with 16 independent GPU workers.

## Overview

This setup implements **vLLM's native DP coordinator pattern** where:
- **Rank 0 (Master)**: Receives ALL client requests and runs DP Coordinator (1 GPU)
- **Ranks 1-15 (Workers)**: Connect to master via RPC, provide distributed compute (15 pods × 1 GPU each)
- **Total capacity**: 16 ranks with 16 GPUs total (1 GPU per pod)
- **Request flow**: Client → Rank 0 → DP Coordinator → All 16 workers via RPC

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Client                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         Rank 0 Service (HTTP 8000 + RPC 13345)               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Rank 0 Pod (Master) - 1 GPU                     │
│  ┌─────────────────────────────────────────────────────────┐│
│  │             vLLM DP Coordinator                          ││
│  │  • Receives all HTTP requests                            ││
│  │  • Distributes work across 16 ranks                      ││
│  │  • Processes on 1 local GPU (TP=1)                       ││
│  └─────────────────────────────────┬───────────────────────┘│
│                                     │ RPC Port 13345          │
└─────────────────────────────────────┼───────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────┐
│           Worker StatefulSet (Ranks 1-15)                    │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  workers-0 (Rank 1) → 1 GPU                              ││
│  │  workers-1 (Rank 2) → 1 GPU                              ││
│  │  workers-2 (Rank 3) → 1 GPU                              ││
│  │  ...                                                      ││
│  │  workers-14 (Rank 15) → 1 GPU                            ││
│  │                                                           ││
│  │  • Each connects to Rank 0 via RPC                       ││
│  │  • No HTTP API (work received via RPC only)              ││
│  │  • Dynamic rank calculation from pod ordinal             ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Configuration Summary

**Pods:** 16 pods total
- 1 master pod (Rank 0) - deployed via Helm chart
- 15 worker pods (Ranks 1-15) - deployed via StatefulSet

**Resources:**
- 16 GPUs total (1 GPU per pod)
- TP=1 (no tensor parallelism, 1 GPU per pod)
- DP=16 (16 data parallel ranks)

**Deployment:**
```
Rank 0: --tensor-parallel-size 1 --data-parallel-size 16 --data-parallel-rank 0
        --data-parallel-address $(POD_IP) --data-parallel-rpc-port 13345

Ranks 1-15: --tensor-parallel-size 1 --data-parallel-size 16
            --data-parallel-rank <1-15>
            --data-parallel-address ms-llama31-multinode-llm-d-modelservice-decode.llm-d-llama31-multinode.svc.cluster.local
            --data-parallel-rpc-port 13345
```

## How It Works

1. **Client sends request** → HTTP request to Rank 0 service on port 8000
2. **Rank 0 receives request** → Logged in Rank 0's HTTP API layer
3. **DP Coordinator distributes work** → Splits work across all 16 GPU workers
4. **Workers process work** → Ranks 1-15 receive work via RPC (no HTTP logs), GPUs process independently
5. **Response aggregated** → Rank 0 collects results and returns to client

## Quick Start

### Prerequisites

- Kubernetes cluster with GPU nodes (16 GPUs total)
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
cd scripts/k8s/llama3/vllm-native
./deploy.sh
```

This will:
- Deploy Rank 0 (master) via Helmfile
- Create service for Rank 0 (ports 8000 + 13345)
- Deploy StatefulSet for workers (Ranks 1-15)

Wait for pods to be ready (10-15 minutes first time for model download).

### 3. Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n llm-d-llama31-multinode

# Expected output: 16+ pods
# - 1 gateway pod
# - 1 master pod (decode)
# - 15 worker pods (workers-0 through workers-14)
```

### 4. Port Forward (Required for Testing)

```bash
kubectl port-forward -n llm-d-llama31-multinode \
  svc/ms-llama31-multinode-llm-d-modelservice-decode 8000:8000
```

### 5. Test Connection

```bash
# Check models endpoint
curl http://localhost:8000/v1/models

# Send a test request
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

### 6. Run Benchmark

```bash
./run-benchmark.sh
```

## Deployment Flow Summary

```bash
# 1. Deploy
./deploy.sh

# 2. Wait for all pods ready (16 worker pods + 1 master)
kubectl get pods -n llm-d-llama31-multinode -w

# 3. Port forward (in separate terminal)
kubectl port-forward -n llm-d-llama31-multinode \
  svc/ms-llama31-multinode-llm-d-modelservice-decode 8000:8000

# 4. Test
curl http://localhost:8000/v1/models

# 5. Benchmark
./run-benchmark.sh
```

## Monitoring & Debugging

### Check All Pods

```bash
kubectl get pods -n llm-d-llama31-multinode
```

**Expected**:
- 1 gateway pod
- 1 master pod (`ms-llama31-multinode-llm-d-modelservice-decode-*`)
- 15 worker pods (`ms-llama31-multinode-llm-d-modelservice-workers-0` through `workers-14`)

### Check Service

```bash
kubectl get svc -n llm-d-llama31-multinode ms-llama31-multinode-llm-d-modelservice-decode
```

**Expected**: Service with ports `8000/TCP,13345/TCP`

### Check Master (Rank 0) Logs

```bash
kubectl logs -n llm-d-llama31-multinode \
  -l app=ms-llama31-multinode-llm-d-modelservice-decode \
  -c vllm -f
```

**Look for**:
- `Started DP Coordinator process`
- `Rank 0 is connected to 15 peer ranks`
- `Uvicorn running on http://0.0.0.0:8000`

### Check Worker Logs

```bash
# Check specific worker (e.g., Rank 1)
kubectl logs -n llm-d-llama31-multinode \
  ms-llama31-multinode-llm-d-modelservice-workers-0 -f

# Check specific worker (e.g., Rank 15)
kubectl logs -n llm-d-llama31-multinode \
  ms-llama31-multinode-llm-d-modelservice-workers-14 -f
```

**Look for**:
- `Starting vLLM worker with rank <N>`
- `Rank <N> is connected to 15 peer ranks`
- `rank <N> in world size 16 is assigned as DP rank <N>`

### Verify DP Coordination

Check that all ranks are connected:

```bash
# Check master logs for peer connections
kubectl logs -n llm-d-llama31-multinode \
  -l app=ms-llama31-multinode-llm-d-modelservice-decode \
  -c vllm | grep "Rank 0 is connected"

# Check worker logs for peer connections
kubectl logs -n llm-d-llama31-multinode \
  ms-llama31-multinode-llm-d-modelservice-workers-0 | grep "Rank 1 is connected"
```

**Expected**: Each rank should report being connected to 15 peer ranks.

### Check GPU Utilization

```bash
# Check master GPU usage
kubectl exec -n llm-d-llama31-multinode \
  $(kubectl get pod -n llm-d-llama31-multinode -l app=ms-llama31-multinode-llm-d-modelservice-decode -o jsonpath='{.items[0].metadata.name}') \
  -c vllm -- nvidia-smi

# Check worker GPU usage
kubectl exec -n llm-d-llama31-multinode \
  ms-llama31-multinode-llm-d-modelservice-workers-0 -- nvidia-smi
```

**Expected**: During load, all GPUs should show utilization > 0%

### Test Direct Worker Access

You can verify workers are fully functional by accessing them directly:

```bash
# Port forward to a specific worker
kubectl port-forward -n llm-d-llama31-multinode \
  ms-llama31-multinode-llm-d-modelservice-workers-10 8001:8000

# Send request (bypasses DP coordinator, uses only 1 GPU)
curl -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "prompt": "Test", "max_tokens": 20}'
```

**Note**: This bypasses the DP coordinator and only uses that single worker's GPU. For production, always use Rank 0.

## Cleanup

```bash
./cleanup.sh

# To delete everything including namespace
kubectl delete namespace llm-d-llama31-multinode
```

## Troubleshooting

### Pods Not Scheduling

```bash
kubectl describe pod -n llm-d-llama31-multinode <pod-name>
```

Check for: GPU availability, node resources, insufficient GPUs

### Workers Can't Connect to Rank 0

```bash
# Verify service exists
kubectl get svc -n llm-d-llama31-multinode ms-llama31-multinode-llm-d-modelservice-decode

# Check worker logs for connection errors
kubectl logs -n llm-d-llama31-multinode \
  ms-llama31-multinode-llm-d-modelservice-workers-0 | grep -i "error\|connection"
```

### Model Download Issues

Each pod downloads the model independently (EmptyDir storage). First deployment takes longer:
- Rank 0: ~4-5 minutes
- Workers: 5-10 minutes (downloads happen in parallel)

### StatefulSet Issues

```bash
# Check StatefulSet status
kubectl get statefulset -n llm-d-llama31-multinode

# Check events
kubectl describe statefulset -n llm-d-llama31-multinode \
  ms-llama31-multinode-llm-d-modelservice-workers
```

## Key Architecture Decisions

1. **Why 16 pods × 1 GPU instead of 2 pods × 8 GPUs?**
   - Matches vllm-router and llm-d deployment patterns for fair comparison
   - Better resource distribution across cluster
   - Easier horizontal scaling

2. **Why StatefulSet for workers?**
   - Provides stable pod names with ordinals (workers-0, workers-1, etc.)
   - Enables dynamic rank calculation from pod ordinal
   - Better suited for stateful workloads like DP workers

## Files

- `values.yaml` - Helm values for Rank 0 (master)
- `workers-statefulset.yaml` - StatefulSet for Ranks 1-15
- `decode-service.yaml` - Service for Rank 0
- `helmfile.yaml.gotmpl` - Helmfile configuration
- `deploy.sh` - Deployment script
- `cleanup.sh` - Cleanup script
- `run-benchmark.sh` - Benchmark script