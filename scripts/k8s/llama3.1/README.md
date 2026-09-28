# Llama 3.1 8B - Prefill/Decode Disaggregation Deployment

Kubernetes deployment manifests and scripts for deploying Llama 3.1 8B Instruct with Prefill-Decode disaggregation support using Data Parallelism (DP8TP1).

## Overview

This deployment uses:
- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Parallelism**: DP8TP1 (Data Parallel 8, Tensor Parallel 1) via `--intra-node-data-parallel-size 8` flag
- **Architecture**: Prefill-Decode disaggregation with NIXL KV cache transfer
- **Replicas**: 1 prefill pod, 1 decode pod
- **Resources per pod**: 8 GPUs, 800Gi memory, 180 CPUs

## Key Differences from DeepSeek-V3 Deployment

| Aspect | DeepSeek-V3 | Llama 3.1 8B |
|--------|-------------|--------------|
| Model | deepseek-ai/DeepSeek-V3 | meta-llama/Llama-3.1-8B-Instruct |
| Model Size | ~685B parameters | ~8B parameters |
| Parallelism | TP8DP1 | DP8TP1 (via --intra-node-data-parallel-size flag) |
| Tensor Parallel | 8 | 1 |
| Data Parallel | 1 | 8 |
| Replicas (prefill) | 1 | 1 |
| Replicas (decode) | 1 | 1 |
| GPUs per pod | 8 | 8 |
| Memory per pod | 1500Gi | 800Gi |
| Expert Parallelism | No | No |

## Files

- `values.yaml` - Model service configuration (prefill/decode workers)
- `gaie-llama31/values.yaml` - Gateway API Inference Extension configuration
- `helmfile.yaml.gotmpl` - Helmfile template for deployment orchestration
- `httproute.yaml` - HTTPRoute for istio/kgateway
- `httproute.gke.yaml` - HTTPRoute for GKE
- `deploy.sh` - Deployment script
- `cleanup.sh` - Cleanup script

## Prerequisites

1. **Kubernetes Cluster**
   - 16+ GPUs (8 for prefill, 8 for decode)
   - RDMA support (InfiniBand or RoCE) for KV cache transfer
   - Sufficient CPU and memory resources

2. **Client Tools**
   - kubectl configured with cluster access
   - helm 3.x
   - helmfile

3. **llm-d Infrastructure**
   - Gateway provider configured (Istio, KGateway, or GKE)
   - Monitoring stack (optional but recommended)
   - See [llm-d prerequisites](https://github.com/llm-d/llm-d) for details

4. **HuggingFace Token**
   - Create secret in target namespace:
     ```bash
     kubectl create secret generic llm-d-hf-token \
       --from-literal=HF_TOKEN=your_hf_token_here \
       -n llm-d-llama31
     ```

5. **Gateway Configurations**
   - Gateway provider configurations are stored locally in `gateway-configs/`
   - These files are copied from llm-d repository for self-contained deployment
   - If you need to update these configs, copy them from `~/llm-d/guides/prereq/gateway-provider/common-configurations/`

## Quick Start

### Deploy with Default Settings (Istio)

```bash
cd ~/router/scripts/k8s/llama3.1
./deploy.sh
```

This will deploy to namespace `llm-d-llama31` using the default (istioBench) gateway provider.

### Deploy with Custom Namespace and Gateway

```bash
# Deploy to custom namespace with KGateway
./deploy.sh my-namespace kgateway

# Deploy with GKE gateway
./deploy.sh llm-d-llama31 gke
```

### Install HTTPRoute

After deployment completes, install the HTTPRoute:

```bash
# For Istio/KGateway
kubectl apply -f httproute.yaml -n llm-d-llama31

# For GKE
kubectl apply -f httproute.gke.yaml -n llm-d-llama31
```

## Configuration

### Adjusting Parallelism

To modify the parallelism configuration, edit `values.yaml`:

```yaml
decode:
  parallelism:
    tensor: 1    # Tensor parallelism (GPUs per worker)
    data: 8      # Data parallelism (number of workers)
  replicas: 8    # Should match data parallelism

prefill:
  parallelism:
    tensor: 1
    data: 8
  replicas: 8
```

### Resource Limits

Adjust per-worker resources in `values.yaml`:

```yaml
resources:
  limits:
    memory: 100Gi
    cpu: "22"
    nvidia.com/gpu: "1"
  requests:
    memory: 100Gi
    cpu: "22"
    nvidia.com/gpu: "1"
```

### Node Selection

To pin workers to specific nodes, add nodeSelector or nodeName to `values.yaml`:

```yaml
decode:
  nodeSelector:
    node-type: gpu-node
```

## Accessing the Deployment

### Get Gateway External IP

```bash
kubectl get svc -n llm-d-llama31 infra-llama31-inference-gateway-istio
```

### Port Forward for Testing

```bash
# Forward to gateway
kubectl port-forward -n llm-d-llama31 \
  svc/infra-llama31-inference-gateway-istio 8000:80

# Test
curl http://localhost:8000/v1/models
```

### Send Inference Request

```bash
curl http://GATEWAY_IP/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "The capital of France is",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

## Monitoring

### View Logs

```bash
# Prefill pods
kubectl logs -n llm-d-llama31 -l app=ms-llama31-llm-d-modelservice-prefill -f

# Decode pods
kubectl logs -n llm-d-llama31 -l app=ms-llama31-llm-d-modelservice-decode -f

# GAIE EPP
kubectl logs -n llm-d-llama31 -l app=gaie-llama31-epp -f
```

### Check Pod Status

```bash
kubectl get pods -n llm-d-llama31
kubectl describe pod <pod-name> -n llm-d-llama31
```

### Prometheus Metrics

If monitoring is enabled, metrics are available at:
- Prefill: `http://<pod-ip>:8000/metrics`
- Decode: `http://<pod-ip>:8200/metrics`

## Running Benchmarks

You can use vllm bench serve to test the deployment:

```bash
# From a pod with vllm installed (e.g., one of the prefill pods)
GATEWAY_IP=$(kubectl get svc -n llm-d-llama31 \
  infra-llama31-inference-gateway-istio \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

vllm bench serve \
  --dataset-name random \
  --num-prompts 100 \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --random-input-len 1000 \
  --random-output-len 1000 \
  --endpoint /v1/completions \
  --max-concurrency 16 \
  --save-result \
  --ignore-eos \
  --served-model-name "meta-llama/Llama-3.1-8B-Instruct" \
  --host "$GATEWAY_IP" \
  --port 80
```

## Cleanup

### Remove Deployment Only

```bash
./cleanup.sh
```

### Remove Everything Including Namespace

```bash
./cleanup.sh
kubectl delete namespace llm-d-llama31
```

### Custom Namespace Cleanup

```bash
./cleanup.sh my-namespace kgateway
```

## Troubleshooting

### Pods Not Starting

Check events and logs:
```bash
kubectl describe pod <pod-name> -n llm-d-llama31
kubectl logs <pod-name> -n llm-d-llama31
```

### Out of Memory Errors

Increase memory limits in `values.yaml`:
```yaml
resources:
  limits:
    memory: 150Gi  # Increase from 100Gi
```

### Model Download Issues

Verify HuggingFace token:
```bash
kubectl get secret llm-d-hf-token -n llm-d-llama31 -o yaml
```

### RDMA/Network Issues

Check if RDMA is available:
```bash
kubectl exec -n llm-d-llama31 <pod-name> -- ibv_devices
```

Verify NIXL side channel connectivity:
```bash
kubectl logs -n llm-d-llama31 <pod-name> | grep NIXL
```

### Gateway Not Accessible

Check gateway status:
```bash
kubectl get gateway -n llm-d-llama31
kubectl describe gateway infra-llama31-inference-gateway -n llm-d-llama31
```

## Advanced Configuration

### Using Different Gateway Provider

The helmfile supports multiple gateway providers. Edit `helmfile.yaml.gotmpl` or use the `-e` flag:

```bash
# Available options: istio, istioBench, kgateway, gke, aks, xpu
helmfile apply -e kgateway -n llm-d-llama31
```

### Custom Release Name

Set the `RELEASE_NAME_POSTFIX` environment variable:

```bash
export RELEASE_NAME_POSTFIX=llama31-v2
./deploy.sh
```

This will create releases named `infra-llama31-v2`, `gaie-llama31-v2`, and `ms-llama31-v2`.

### Selective P/D

To enable selective P/D (routing some requests directly to decode), modify `gaie-llama31/values.yaml`:

```yaml
inferenceExtension:
  pluginsCustomConfig:
    pd-config.yaml: |
      plugins:
      - type: pd-profile-handler
        parameters:
          threshold: 100  # Route requests with <100 input tokens directly to decode
```

## Architecture Notes

### Data Parallelism (DP8TP1)

This deployment uses DP8TP1 instead of TP8DP1 because:
- Llama 3.1 8B is small enough to fit on a single GPU
- Data parallelism provides better throughput scaling for smaller models
- Each data parallel worker has the full model, enabling independent request processing
- Reduced complexity compared to tensor parallelism coordination

**Important**: Data parallelism is configured via the `--intra-node-data-parallel-size 8` flag in vLLM, NOT by creating 8 separate pods. Each pod (prefill and decode) runs a single vLLM server process that internally manages 8 data parallel workers across 8 GPUs.

### KV Cache Transfer

With 1 prefill pod and 1 decode pod (each running 8 DP workers internally), the GAIE EPP scheduler:
1. Routes prefill requests to the prefill pod
2. Transfers KV cache via NIXL from a prefill DP worker to the corresponding decode DP worker
3. Routes subsequent decode requests to maintain affinity with the appropriate DP worker

The `queue-scorer` plugin optimizes for KV cache reuse by preferring workers with matching prefix blocks.

## Performance Expectations

For Llama 3.1 8B with DP8TP1:
- **Higher throughput** than TP8 due to more parallel workers
- **Better GPU utilization** with single-GPU workers
- **Faster TTFT** with dedicated prefill workers
- **Stable decode latency** with dedicated decode workers

Expected metrics (approximate):
- TTFT: 100-300ms (depending on input length)
- TPOT: 15-25ms (per output token)
- Throughput: 2000-4000 tokens/s aggregate (with 16+ concurrent requests)

## References

- [llm-d P/D Disaggregation Guide](https://github.com/llm-d/llm-d/blob/main/guides/pd-disaggregation/README.md)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/)
