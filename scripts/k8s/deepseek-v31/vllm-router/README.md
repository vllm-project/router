# DeepSeek V3.1 with vllm-router (P/D Disaggregation)

This directory contains configurations for deploying DeepSeek V3.1 with Prefill/Decode disaggregation using vllm-router.

## Architecture

- **Model**: DeepSeek-V3.1 (~700GB)
- **Prefill Pod**: 1 pod with TP=8 (8 H100 GPUs)
- **Decode Pod**: 1 pod with TP=8 (8 H100 GPUs)
- **Routing**: vllm-router with consistent_hash policy
- **Total GPUs**: 16 H100 GPUs

## Prerequisites

1. Kubernetes cluster with:
   - At least 16 H100 GPUs available
   - GPU operator installed

2. HuggingFace token with access to DeepSeek-V3.1 model

## Deployment

### Step 1: Set HF Token (Optional)

```bash
export HF_TOKEN=hf_xxxxx
```

Or create secret manually:

```bash
kubectl create namespace vllm-router-deepseek-v31
kubectl create secret generic llm-d-hf-token \
  --from-literal=HF_TOKEN=hf_xxxxx \
  -n vllm-router-deepseek-v31
```

### Step 2: Deploy

```bash
./deploy.sh
```

This will:
- Create namespace and secret (if not exists)
- Deploy prefill and decode pods via Helm
- Create backend services
- Deploy vllm-router

### Step 3: Wait for Pods

The deployment takes 10-15 minutes on first run (model download ~700GB). Subsequent deployments take 3-5 minutes.

```bash
kubectl get pods -n vllm-router-deepseek-v31 -w
```

Expected pods:
- 1 prefill pod (1/1 ready)
- 1 decode pod (1/1 ready)
- 1 vllm-router pod (1/1 ready)

## Configuration

### Model Configuration

The `values.yaml` file contains the main configuration:

- **Tensor Parallelism**: 8 (uses all 8 GPUs per pod)
- **Data Parallelism**: 1 (single replica per role)
- **Max Model Length**: 32,000 tokens
- **Block Size**: 128
- **Memory**: 1500Gi per pod
- **KV Transfer**: NixlConnector for P/D communication

### DeepSeek V3.1 Optimizations

The configuration includes several optimizations:

1. **Chunked Prefill**: Enabled for better throughput
2. **Max Batched Tokens**: 32,768 for efficient batching
3. **Scheduler Steps**: 10 for better scheduling
4. **Expert Parallelism**: Distributes 256 MoE experts across 8 GPUs
5. **VLLM V1**: Enabled for CUDA graphs and better performance
6. **Shared Memory**: 32Gi for large model operations

### vllm-router Configuration

The router is configured with:
- **Policy**: consistent_hash for request routing
- **Data Parallel Size**: 1 (single instance per role)
- **Port**: 10001 for client requests
- **Metrics Port**: 29000 for monitoring

## Benchmarking

Run benchmarks with:

```bash
./run-benchmark.sh [num_prompts] [concurrency]
```

Examples:
```bash
./run-benchmark.sh           # 100 prompts, 16 concurrency
./run-benchmark.sh 200 32    # 200 prompts, 32 concurrency
./run-benchmark.sh 200 64    # 200 prompts, 64 concurrency
```

## Cleanup

```bash
./cleanup.sh
```

To completely remove including cached model:
```bash
kubectl delete namespace vllm-router-deepseek-v31
```

## Files

- `values.yaml` - vLLM backend configuration (prefill + decode)
- `helmfile.yaml` - Helmfile for deploying backends
- `deploy.sh` - Deployment script
- `cleanup.sh` - Cleanup script
- `run-benchmark.sh` - Benchmarking script
- `backend-services.yaml` - Service definitions for prefill/decode
- `router-deployment.yaml` - vllm-router deployment
- `router-service.yaml` - vllm-router service (ClusterIP + NodePort)

## Troubleshooting

### Check pod logs

```bash
# Prefill pod
kubectl logs -n vllm-router-deepseek-v31 -l llm-d.ai/role=prefill -f

# Decode pod
kubectl logs -n vllm-router-deepseek-v31 -l llm-d.ai/role=decode -f

# vllm-router
kubectl logs -n vllm-router-deepseek-v31 -l app=vllm-router -f
```

### Test router endpoint

```bash
# Port forward to router
kubectl port-forward -n vllm-router-deepseek-v31 svc/vllm-router-deepseek-v31 10001:10001

# Test in another terminal
curl -X POST "http://localhost:10001/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V3.1",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

### Check router health

```bash
curl http://localhost:10001/health
```

## Comparison with llm-d

This setup uses vllm-router instead of GAIE (Gateway API Inference Extension). Key differences:

| Feature | vllm-router | llm-d (GAIE) |
|---------|-------------|--------------|
| Routing | consistent_hash policy | Queue scoring + prefix matching |
| Components | 3 pods (prefill, decode, router) | 4 pods (prefill, decode, EPP, gateway) |
| Integration | Standalone router | Istio gateway integration |
| Metrics | vllm-router metrics | GAIE EPP metrics + Istio |

Both setups use the same backend configuration (TP=8, expert parallelism, etc.).
