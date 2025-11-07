# DeepSeek V3.1 with llm-d (GAIE P/D Disaggregation)

This directory contains configurations for deploying DeepSeek V3.1 with Prefill/Decode disaggregation using llm-d and GAIE (Gateway API Inference Extension).

## Architecture

- **Model**: DeepSeek-V3.1 (~700GB)
- **Prefill Pod**: 1 pod with TP=8 (8 H100 GPUs)
- **Decode Pod**: 1 pod with TP=8 (8 H100 GPUs)
- **Routing**: GAIE EPP with queue scoring and prefix matching
- **Total GPUs**: 16 H100 GPUs

## Prerequisites

1. Kubernetes cluster with:
   - At least 16 H100 GPUs available
   - GPU operator installed
   - Istio installed (for gateway)

2. HuggingFace token with access to DeepSeek-V3 model

## Deployment

### Step 1: Create Namespace and Secret

```bash
kubectl create namespace llm-d-deepseek-v3
kubectl create secret generic llm-d-hf-token \
  --from-literal=HF_TOKEN=hf_xxxxx \
  -n llm-d-deepseek-v3
```

### Step 2: Deploy

```bash
./deploy.sh
```

This will:
- Deploy infrastructure (gateway)
- Deploy GAIE components (InferencePool, EPP)
- Deploy model service (prefill + decode pods)
- Create routing resources

### Step 3: Wait for Pods

The deployment takes 10-15 minutes on first run (model download ~700GB). Subsequent deployments take 3-5 minutes.

```bash
kubectl get pods -n llm-d-deepseek-v3 -w
```

Expected pods:
- 1 prefill pod (1/1 ready)
- 1 decode pod (2/2 ready) - has sidecar
- 1 GAIE EPP pod (1/1 ready)
- 1 gateway pod (1/1 ready)

## Configuration

### Model Configuration

The `values.yaml` file contains the main configuration:

- **Tensor Parallelism**: 8 (uses all 8 GPUs per pod)
- **Max Model Length**: 32,000 tokens
- **Block Size**: 128
- **Memory**: 1500Gi per pod
- **KV Transfer**: NixlConnector for P/D communication

### Optimizations Applied

The configuration includes several DeepSeek V3 specific optimizations:

1. **Chunked Prefill**: Enabled for better throughput
2. **Max Batched Tokens**: 32,768 for efficient batching
3. **Scheduler Steps**: 10 for better scheduling
4. **VLLM V1**: Enabled for CUDA graphs and better performance
5. **Shared Memory**: 32Gi for large model operations

## Benchmarking

Run benchmarks with:

```bash
./run-benchmark.sh [num_prompts] [concurrency]
```

Examples:
```bash
./run-benchmark.sh           # 200 prompts, 32 concurrency
./run-benchmark.sh 200 64    # 200 prompts, 64 concurrency
./run-benchmark.sh 200 128   # 200 prompts, 128 concurrency
```

## Cleanup

```bash
./cleanup.sh
```

To completely remove including cached model:
```bash
kubectl delete namespace llm-d-deepseek-v3
```

## Files

- `values.yaml`: Main model service configuration
- `helmfile.yaml.gotmpl`: Helmfile template for deployment
- `deploy.sh`: Deployment script
- `cleanup.sh`: Cleanup script
- `run-benchmark.sh`: Benchmarking script
- `backend-services.yaml`: Service definitions
- `inferencepool.yaml`: InferencePool resource
- `httproute.yaml`: HTTPRoute configuration
- `gateway-configs/`: Gateway configuration files
- `gaie-deepseek-v3/`: GAIE-specific configuration

## Troubleshooting

### Check pod logs
```bash
# Prefill pod
kubectl logs -n llm-d-deepseek-v3 -l llm-d.ai/role=prefill

# Decode pod
kubectl logs -n llm-d-deepseek-v3 -l llm-d.ai/role=decode -c vllm

# GAIE EPP
kubectl logs -n llm-d-deepseek-v3 -l inferencepool=gaie-deepseek-v3-epp
```

### Check InferencePool status
```bash
kubectl describe inferencepool gaie-deepseek-v3 -n llm-d-deepseek-v3
```

### Test inference endpoint
```bash
kubectl exec -n llm-d-deepseek-v3 <decode-pod> -c vllm -- \
  curl -X POST "http://infra-deepseek-v3-inference-gateway-istio/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-ai/DeepSeek-V3", "prompt": "Hello", "max_tokens": 10}'
```
