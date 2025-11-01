# Llama 3.1 8B - vllm-router P-D Disaggregation

Kubernetes deployment for Llama 3.1 8B with vllm-router handling Prefill-Decode disaggregation with KV cache transfer.

## Overview

- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Docker Image**: vllm/vllm-openai:latest
- **Parallelism**: DP8 (Data Parallel 8, 8 workers across 8 GPUs)
- **Architecture**: P-D disaggregation with NIXL KV cache transfer
- **Replicas**: 1 prefill pod, 1 decode pod
- **Resources per pod**: 8 GPUs, 800Gi memory, 180 CPUs
- **Namespace**: vllm-router-pd-llama31

## Architecture

```
Client → vllm-router (P-D aware) → Prefill (8 GPUs DP8)
                                   ↓ KV transfer (NIXL)
                                   Decode (8 GPUs DP8)
```

Prefill handles initial token generation, transfers KV cache to decode for continuation.

## Prerequisites

1. **Kubernetes Cluster** with 16 GPUs + RDMA support
2. **Tools**: kubectl, helm, helmfile
3. **HuggingFace Token**:
```bash
kubectl create secret generic llm-d-hf-token \
  --from-literal=HF_TOKEN=hf_token \
  -n vllm-router-pd-llama31
```

## Deployment Steps

### 1. Deploy Everything

```bash
cd scripts/k8s/llama3/vllm-router/pd-disagg
./deploy.sh
```

This single command deploys:
- Namespace creation (if needed)
- HuggingFace token (auto-copied from llm-d-llama31 if available)
- 1 prefill pod with KV transfer enabled (8 GPUs DP8)
- 1 decode pod with KV transfer enabled (8 GPUs DP8)
- Backend Kubernetes Services (prefill & decode)
- vllm-router deployment
- vllm-router service

### 2. Verify Deployment

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

**Prefill configuration**:
```yaml
prefill:
  parallelism:
    tensor: 8  # Allocates 8 GPUs
    data: 1
  replicas: 1
  containers:
  - name: "vllm"
    image: vllm/vllm-openai:latest
    args:
      - "--tensor-parallel-size"
      - "1"
      - "--data-parallel-size"
      - "8"  # 8 data-parallel workers
      - "--kv-transfer-config"
      - '{"kv_connector":"NixlConnector", "kv_role":"kv_both"}'
    ports:
      - containerPort: 8000
```

**Decode configuration**:
```yaml
decode:
  parallelism:
    tensor: 8  # Allocates 8 GPUs
    data: 1
  replicas: 1
  containers:
  - name: "vllm"
    image: vllm/vllm-openai:latest
    args:
      - "--tensor-parallel-size"
      - "1"
      - "--data-parallel-size"
      - "8"  # 8 data-parallel workers
      - "--kv-transfer-config"
      - '{"kv_connector":"NixlConnector", "kv_role":"kv_both"}'
    ports:
      - containerPort: 8200
```

**Note**: The helm chart uses `parallelism.tensor` to allocate GPUs, even for data parallelism.

### router-deployment.yaml

vllm-router with P-D disaggregation:
```yaml
command:
  - vllm-router
  - --pd-disaggregation
  - --prefill
  - http://ms-llama31-llm-d-modelservice-prefill.vllm-router-pd-llama31.svc.cluster.local:8000
  - --decode
  - http://ms-llama31-llm-d-modelservice-decode.vllm-router-pd-llama31.svc.cluster.local:8200
  - --data-parallel-size
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
2. **Prefill phase**: Router sends to prefill pod
3. **KV transfer**: Prefill transfers KV cache to decode via NIXL
4. **Decode phase**: Router sends subsequent tokens to decode pod
5. **Response**: Decode streams tokens back to client

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