# Llama 3.1 8B - vllm-router P-D Disaggregation

Kubernetes deployment for Llama 3.1 8B with vllm-router handling Prefill-Decode disaggregation with KV cache transfer.

## Overview

- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Docker Image**: vllm/vllm-openai:latest
- **Parallelism**: Pure Kubernetes pod replication (8 independent prefill pods + 8 independent decode pods)
- **Architecture**: P-D disaggregation with NIXL KV cache transfer
- **Replicas**: 8 prefill pods, 8 decode pods
- **Resources per pod**: 1 GPU, 100Gi memory, 22 CPUs
- **Total**: 16 GPUs across 16 pods
- **Namespace**: vllm-router-pd-llama31

## Architecture

```
Client → vllm-router (P-D aware, consistent hashing)
              ↓
      ┌───────┴───────┐
      ▼               ▼
  Prefill Pods (×8)  Decode Pods (×8)
  • 1 GPU each       • 1 GPU each
  • Port 8000        • Port 8200
  • Prompt proc.     • Token gen.
      └───────────────┘
       KV Transfer (NIXL)
```

This architecture matches llm-d's pod-level parallelism (8+8 pods) for fair comparison of routing algorithms:
- **vllm-router**: Consistent hashing load balancing
- **llm-d (GAIE EPP)**: Queue scoring + prefix cache matching

## Prerequisites

1. **Kubernetes Cluster** with 16 GPUs + RDMA support
2. **Tools**: kubectl, helm, helmfile
3. **HuggingFace Token**: Set as environment variable before deploying

## Deployment Steps

### 1. Set HuggingFace Token

Export your HuggingFace token as an environment variable:

```bash
export HF_TOKEN=hf_your_token_here
```

Example:
```bash
export HF_TOKEN=your_actual_token
```

The `deploy.sh` script will automatically create the Kubernetes secret from this environment variable.

### 2. Deploy Everything

```bash
cd scripts/k8s/llama3/vllm-router/pd-disagg
./deploy.sh
```

This single command deploys:
- Namespace creation (if needed)
- HuggingFace token secret (from HF_TOKEN environment variable)
- 8 prefill pods with KV transfer enabled (1 GPU each)
- 8 decode pods with KV transfer enabled (1 GPU each)
- Backend Kubernetes Services (prefill & decode)
- vllm-router deployment (with consistent hashing)
- vllm-router service

### 3. Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n vllm-router-pd-llama31

# Check services exist
kubectl get svc -n vllm-router-pd-llama31

# View logs
kubectl logs -n vllm-router-pd-llama31 -l llm-d.ai/role=prefill -c vllm -f
kubectl logs -n vllm-router-pd-llama31 -l llm-d.ai/role=decode -c vllm -f
kubectl logs -n vllm-router-pd-llama31 -l app=vllm-router -f
```

## Running Benchmarks

### Performance Benchmarks

```bash
./run-benchmark.sh                # 100 prompts, 16 concurrency
./run-benchmark.sh 200 32         # 200 prompts, 32 concurrency
```

The benchmark script:
- Runs from inside a vLLM pod (has vllm bench command)
- Connects directly to vllm-router service
- Measures throughput, latency, and token generation metrics

### LM Evaluation

Run model evaluation tasks (requires `lm_eval` installed).
Please create a virtual environment and then run the installation command.

```bash
# Install lm_eval if needed
pip install lm-eval

# Set HuggingFace token
export HF_TOKEN=your_token_here

# Run evaluations
./run-eval.sh                     # GSM8K with 1 concurrent request
./run-eval.sh mmlu 4 50           # MMLU with 4 concurrent, 50 samples
./run-eval.sh hellaswag 1 100     # HellaSwag with 1 concurrent, 100 samples
```

The eval script:
- Port-forwards to vllm-router on localhost:10001
- Runs `lm_eval` with OpenAI-compatible completions API
- Supports various tasks: gsm8k, mmlu, hellaswag, truthfulqa, etc.

## Configuration

### values.yaml

Key configuration points:

**Prefill configuration** (8 independent pods):
```yaml
prefill:
  replicas: 8  # 8 Kubernetes pods
  containers:
  - name: "vllm"
    image: vllm/vllm-openai:latest
    args:
      - "--model"
      - "meta-llama/Llama-3.1-8B-Instruct"
      - "--kv-transfer-config"
      - '{"kv_connector":"NixlConnector", "kv_role":"kv_both"}'
      - "--block-size"
      - "128"
    env:
      - name: CUDA_VISIBLE_DEVICES
        value: "0"  # Single GPU per pod
    resources:
      limits:
        nvidia.com/gpu: "1"  # 1 GPU per pod
        memory: 100Gi
        cpu: "22"
```

**Decode configuration** (8 independent pods):
```yaml
decode:
  replicas: 8  # 8 Kubernetes pods
  containers:
  - name: "vllm"
    image: vllm/vllm-openai:latest
    args:
      - "--model"
      - "meta-llama/Llama-3.1-8B-Instruct"
      - "--kv-transfer-config"
      - '{"kv_connector":"NixlConnector", "kv_role":"kv_both"}'
      - "--block-size"
      - "128"
    env:
      - name: CUDA_VISIBLE_DEVICES
        value: "0"  # Single GPU per pod
    resources:
      limits:
        nvidia.com/gpu: "1"  # 1 GPU per pod
        memory: 100Gi
        cpu: "22"
```

**Note**: This uses pure Kubernetes replication (NOT vLLM's internal `--intra-node-data-parallel-size`), matching llm-d's architecture.

### router-deployment.yaml

vllm-router with P-D disaggregation:
```yaml
command:
  - vllm-router
  - --vllm-pd-disaggregation
  - --prefill
  - http://ms-llama31-llm-d-modelservice-prefill.vllm-router-pd-llama31.svc.cluster.local:8000
  - --decode
  - http://ms-llama31-llm-d-modelservice-decode.vllm-router-pd-llama31.svc.cluster.local:8200
  - --intra-node-data-parallel-size
  - "8"
  - --policy
  - consistent_hash
```

## Monitoring

### View Logs

```bash
# Prefill logs
kubectl logs -n vllm-router-pd-llama31 -l llm-d.ai/role=prefill -c vllm -f

# Decode logs
kubectl logs -n vllm-router-pd-llama31 -l llm-d.ai/role=decode -c vllm -f

# Router logs (INFO level)
kubectl logs -n vllm-router-pd-llama31 -l app=vllm-router -f

# Router logs with DEBUG (shows routing decisions)
kubectl logs -n vllm-router-pd-llama31 -l app=vllm-router -f | grep -E "(prefill|decode|PD retry)"
```

### Metrics

```bash
# Router metrics (Prometheus format)
kubectl port-forward -n vllm-router-pd-llama31 svc/vllm-router-llama31 29000:29000
curl http://localhost:29000/metrics

# vLLM prefill metrics
kubectl exec -n vllm-router-pd-llama31 -l llm-d.ai/role=prefill -c vllm -- curl -s http://localhost:8000/metrics

# vLLM decode metrics
kubectl exec -n vllm-router-pd-llama31 -l llm-d.ai/role=decode -c vllm -- curl -s http://localhost:8200/metrics
```

### Validate P-D Routing

Check that requests are being routed correctly:

```bash
# Watch router select different workers
kubectl logs -n vllm-router-pd-llama31 -l app=vllm-router -f | grep "PD retry attempt"

# You should see output like:
# PD retry attempt 0 using prefill=...@2 decode=...@0
# PD retry attempt 0 using prefill=...@5 decode=...@3
```

## Cleanup

```bash
./cleanup.sh vllm-router-pd-llama31
kubectl delete namespace vllm-router-pd-llama31
```

## How P-D Disaggregation Works

1. **Request arrives** at vllm-router
2. **Worker selection**: Router uses consistent hashing to select prefill and decode workers
3. **Prefill phase**: Router sends to selected prefill pod (1 of 8)
4. **KV transfer**: Prefill pod transfers KV cache to paired decode pod via NIXL
5. **Decode phase**: Router sends subsequent tokens to selected decode pod (1 of 8)
6. **Response**: Decode pod streams tokens back to client

## Key Differences vs llm-d

| Aspect | vllm-router (This Setup) | llm-d with GAIE EPP |
|--------|-------------------------|---------------------|
| **Pod Architecture** | 8 prefill + 8 decode pods | 8 prefill + 8 decode pods |
| **Load Balancing** | Consistent hashing | Queue scoring + prefix matching |
| **Routing Intelligence** | Hash-based affinity | Queue depth + KV cache awareness |
| **Gateway** | vllm-router (stateless) | GAIE EPP (stateful scheduling) |
| **P/D Coordination** | Router-managed | GAIE-managed with InferencePool |

Both use identical pod configurations (1 GPU per pod, same resources) for fair performance comparison.

## Troubleshooting

### RDMA/KV Transfer Issues

Check NIXL connectivity:
```bash
kubectl logs -n vllm-router-pd-llama31 -l role=prefill | grep NIXL
kubectl logs -n vllm-router-pd-llama31 -l role=decode | grep NIXL
```

Check RDMA devices:
```bash
kubectl exec -n vllm-router-pd-llama31 <pod-name> -- ibv_devices
```

### Router Not Routing Correctly

Check router logs for errors:
```bash
kubectl logs -n vllm-router-pd-llama31 -l app=vllm-router -f
```

Verify backend connectivity:
```bash
kubectl exec -n vllm-router-pd-llama31 <router-pod> -- curl http://ms-llama31-llm-d-modelservice-prefill:8000/v1/models
kubectl exec -n vllm-router-pd-llama31 <router-pod> -- curl http://ms-llama31-llm-d-modelservice-decode:8200/v1/models
```