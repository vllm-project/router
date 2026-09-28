# Llama 3.1 8B - vllm-router Native Load Balancing

Kubernetes deployment for Llama 3.1 8B with vllm-router handling native load balancing across 2 backend pods.

## Overview

- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Docker Image**: vllm/vllm-openai:latest
- **Parallelism**: TP8 (Tensor Parallel 8)
- **Architecture**: 2 independent vLLM backends, vllm-router load balances
- **Replicas**: 2 decode pods (no prefill/decode separation)
- **Resources per pod**: 8 GPUs, 800Gi memory, 180 CPUs
- **Namespace**: vllm-router-llama31

## Architecture

```
Client → vllm-router → [Backend-0 (8 GPUs TP8), Backend-1 (8 GPUs TP8)]
```

Both backends handle full inference (prefill + decode). vllm-router distributes requests.

## Prerequisites

1. **Kubernetes Cluster** with 16 GPUs
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

### 2. Deploy Model Services and Router

```bash
cd scripts/k8s/llama3/vllm-router/native
./deploy.sh
```

This single command deploys:
- Namespace creation (if needed)
- HuggingFace token secret (from HF_TOKEN environment variable)
- 2 vLLM backend pods via helmfile (8 GPUs DP8 each)
- Backend Kubernetes Services (headless service for pod discovery)
- vllm-router deployment
- vllm-router service

### 3. Verify Deployment

```bash
kubectl get pods -n vllm-router-llama31
kubectl logs -n vllm-router-llama31 -l app=vllm-router -f
```

## Running Benchmarks

```bash
./run-benchmark.sh                # 100 prompts, 16 concurrency
./run-benchmark.sh 200 32         # 200 prompts, 32 concurrency
```

Or manually:
```bash
kubectl port-forward -n vllm-router-llama31 svc/vllm-router-llama31 10001:10001

# Then from a pod with vllm installed
vllm bench serve \
  --dataset-name random \
  --num-prompts 100 \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --random-input-len 1000 \
  --random-output-len 1000 \
  --endpoint /v1/completions \
  --max-concurrency 16 \
  --host localhost \
  --port 10001
```

## Configuration

### values.yaml

Key settings:
```yaml
decode:
  replicas: 2  # Number of backend pods
  containers:
  - name: "vllm"
    image: vllm/vllm-openai:latest
    args:
      - "--tensor-parallel-size"
      - "8"  # TP8 per pod
```

### router-deployment.yaml

Router uses headless service with data-parallel-size for load balancing:
```yaml
command:
  - vllm-router
  - --worker-urls
  - http://ms-llama31-llm-d-modelservice-decode.vllm-router-llama31.svc.cluster.local:8000
  - --intra-node-data-parallel-size
  - "8"
```

This configuration creates 8 logical workers (one per DP rank). The router establishes multiple HTTP connections (one per DP rank), and Kubernetes DNS round-robin distributes these connections across both backend pods.

## Monitoring

```bash
# Backend logs
kubectl logs -n vllm-router-llama31 -l llm-d.ai/role=decode -f

# Router logs
kubectl logs -n vllm-router-llama31 -l app=vllm-router -f

# Router metrics
kubectl port-forward -n vllm-router-llama31 svc/vllm-router-llama31 29000:29000
curl http://localhost:29000/metrics
```

## Cleanup

```bash
./cleanup.sh
kubectl delete namespace vllm-router-llama31
```

## Comparison Use Case

This setup is for comparing routing mechanisms:
- **vllm-router/native**: Native load balancing (this directory)
- **vllm-router/pd-disagg**: P-D disaggregation with vllm-router
- **llm-d/native**: llm-d native load balancing
- **llm-d/pd-disagg**: llm-d P-D disaggregation
