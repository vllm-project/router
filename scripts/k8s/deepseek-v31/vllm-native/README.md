# DeepSeek V3.1 with Kubernetes Native Load Balancing

vLLM deployment with 2 independent pods using Kubernetes native load balancing.

## Overview

This setup implements **Kubernetes-native load balancing** where:
- **2 independent vLLM pods**: Each pod runs a complete vLLM instance (8 GPUs with TP=8)
- **No DP coordinator**: Pods operate independently without coordination
- **Kubernetes Service**: Load balances requests across both pods
- **Total capacity**: 2 pods with 16 GPUs total (8 GPUs per pod)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Client                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           Kubernetes Service (Port 8000)                     │
│        Native Round-Robin Load Balancing                     │
└───────────────────────┬──────────────────────────────────┬──┘
                        │                                  │
                        ▼                                  ▼
┌─────────────────────────────────────────────────────────────┐
│              2 Independent vLLM Pods                         │
│  ┌───────────────────────────┐  ┌───────────────────────┐   │
│  │       Pod 1 (TP=8)        │  │      Pod 2 (TP=8)     │   │
│  │  8 GPUs (Tensor Parallel) │  │ 8 GPUs (Tensor Parallel)│ │
│  │  Complete Model Instance  │  │ Complete Model Instance │ │
│  └───────────────────────────┘  └───────────────────────┘   │
│                                                              │
│  • Each pod: standalone vLLM server with full model         │
│  • No inter-pod communication                               │
│  • Independent request processing                           │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Summary

**Pods:** 2 independent pods
- Each pod: 8 GPUs, complete vLLM instance with TP=8
- No master/worker relationship
- All pods are identical

**Resources:**
- 16 GPUs total (8 GPUs per pod)
- Tensor parallelism: TP=8 per pod (required for DeepSeek V3.1)
- No data parallelism (no DP flags)

**Model-Specific Settings:**
- DeepSeek V3.1 optimizations enabled:
  - `--enable-chunked-prefill`
  - `--enable-expert-parallel`
- Max model length: 32000 tokens
- Block size: 128

**Load Balancing:**
- Kubernetes Service handles request distribution
- Round-robin or session-affinity based
- Not queue-aware (limitation)

## How It Works

1. **Client sends request** → Kubernetes Service on port 8000
2. **K8s Service load balances** → Routes to one of 2 pods (round-robin)
3. **Pod processes independently** → No coordination with other pods
4. **Response returned** → Direct from the pod that processed the request

## Quick Start

### Prerequisites

- Kubernetes cluster with GPU nodes (16 GPUs total)
- `kubectl` and `helmfile` installed
- HuggingFace token with DeepSeek V3.1 access

### 1. Create Secret

```bash
kubectl create namespace llm-d-deepseek-v31-native
kubectl create secret generic llm-d-hf-token \
  --from-literal=HF_TOKEN=hf_xxxxxxxxxxxxx \
  -n llm-d-deepseek-v31-native
```

### 2. Deploy

```bash
cd scripts/k8s/deepseek-v31/vllm-native
./deploy.sh
```

This will:
- Deploy 2 independent vLLM pods via Helmfile (each with TP=8)
- Create Kubernetes service for load balancing
- Each pod downloads model independently

Wait for pods to be ready (15-20 minutes first time for model download ~700GB).

### 3. Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n llm-d-deepseek-v31-native -l llm-d.ai/role=decode

# Expected output: 2 pods in Running state
```

### 4. Port Forward

```bash
kubectl port-forward -n llm-d-deepseek-v31-native \
  svc/ms-deepseek-v31-native-llm-d-modelservice-decode 8000:8000
```

### 5. Test Connection

```bash
# Check models endpoint
curl http://localhost:8000/v1/models

# Send a test request
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V3.1",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

### 6. Run Benchmark

```bash
./run-benchmark.sh 200 32
```

## Deployment Flow Summary

```bash
# 1. Deploy
./deploy.sh

# 2. Wait for all pods ready (2 pods)
kubectl get pods -n llm-d-deepseek-v31-native -w

# 3. Port forward
kubectl port-forward -n llm-d-deepseek-v31-native \
  svc/ms-deepseek-v31-native-llm-d-modelservice-decode 8000:8000

# 4. Test
curl http://localhost:8000/v1/models

# 5. Benchmark
./run-benchmark.sh 200 32
```

## Monitoring & Debugging

### Check All Pods

```bash
kubectl get pods -n llm-d-deepseek-v31-native -l llm-d.ai/role=decode
```

**Expected**: 2 pods in Running/Ready state

### Check Service

```bash
kubectl get svc -n llm-d-deepseek-v31-native ms-deepseek-v31-native-llm-d-modelservice-decode
```

**Expected**: Service with port 8000/TCP and 2 endpoints

### Check Service Endpoints

```bash
kubectl get endpoints -n llm-d-deepseek-v31-native ms-deepseek-v31-native-llm-d-modelservice-decode
```

**Expected**: 2 pod IPs listed as endpoints

### Check Pod Logs

```bash
# Check logs from all pods
kubectl logs -n llm-d-deepseek-v31-native -l llm-d.ai/role=decode -c vllm --tail=50

# Check specific pod
kubectl logs -n llm-d-deepseek-v31-native <pod-name> -c vllm -f
```

**Look for**: `Uvicorn running on http://0.0.0.0:8000`

### Check GPU Utilization

```bash
# Check GPU usage on a specific pod
kubectl exec -n llm-d-deepseek-v31-native <pod-name> -c vllm -- nvidia-smi
```

**Expected**: During load, all 8 GPUs per pod should show utilization > 0%

## Cleanup

```bash
./cleanup.sh

# To delete everything including namespace
kubectl delete namespace llm-d-deepseek-v31-native
```

## Troubleshooting

### Pods Not Scheduling

```bash
kubectl describe pod -n llm-d-deepseek-v31-native <pod-name>
```

Check for: GPU availability, node resources, insufficient GPUs (need nodes with 8 GPUs each)

### Service Has No Endpoints

```bash
kubectl get endpoints -n llm-d-deepseek-v31-native
```

If no endpoints, check that pods have `llm-d.ai/role: decode` label

### Model Download Issues

Each pod downloads the model independently:
- First deployment: 15-20 minutes per pod (parallel downloads, ~700GB model)
- Subsequent deployments: Still need to download (no shared cache)
- Monitor with: `kubectl logs -n llm-d-deepseek-v31-native <pod-name> -c vllm -f`

### Uneven Load Distribution

K8s service load balancing is simple round-robin:
- Not queue-aware
- Doesn't consider pod load
- May result in uneven distribution under high load

## Key Architecture Decisions

### Why K8s Load Balancing?

Per vLLM documentation and best practices:
- **DP coordinator not needed** when using external load balancing
- **DP has overhead** that can make it slower than external routing
- **K8s native LB** is simpler and more performant for this use case
- **Fair comparison**: Allows direct comparison with vllm-router and llm-d approaches

### Comparison with Other Approaches

| Aspect | vLLM-Native (K8s LB) | vllm-router | llm-d |
|--------|---------------------|-------------|-------|
| **Pods** | 2 independent (TP=8 each) | 1 prefill + 1 decode (TP=8 each) | 1 prefill + 1 decode (TP=8 each) |
| **Total GPUs** | 16 GPUs | 16 GPUs | 16 GPUs |
| **Load balancing** | K8s service | External router | Gateway + sidecar |
| **Queue awareness** | No | Yes | Yes |
| **P/D disaggregation** | No | Yes | Yes |
| **Complexity** | Lowest | Medium | Highest |
| **vLLM mechanism** | Native (no custom routing) | Custom router | Gateway with sidecar |

### Limitations

1. **Not queue-aware**: K8s doesn't know pod queue lengths
2. **No P/D disaggregation**: All pods do both prefill and decode
3. **No shared model cache**: Each pod downloads independently
4. **Simple round-robin**: May not be optimal under varying load

### When to Use This

✅ **Good for:**
- Testing vLLM native performance baseline
- Comparing against external routing approaches
- Simple deployments without complex routing requirements
- Understanding K8s native load balancing behavior

❌ **Not recommended for:**
- Production with strict SLA requirements (use queue-aware routing)
- Scenarios requiring P/D disaggregation optimization
- High-variance workloads (queue-aware routing is better)

## vLLM Native Features

This implementation uses **vLLM's native request handling** without any custom routing logic:

- **Native OpenAI API**: Standard vLLM OpenAI-compatible server
- **Native scheduling**: vLLM's internal request scheduler
- **No custom routing**: Pure Kubernetes service load balancing
- **No DP coordinator**: Each pod is fully independent

This represents the baseline vLLM performance without any additional optimization layers.

## Files

- `values.yaml` - Helm values for 2 independent pods with TP=8
- `vllm-service.yaml` - Kubernetes service for load balancing
- `helmfile.yaml.gotmpl` - Helmfile configuration
- `deploy.sh` - Deployment script
- `cleanup.sh` - Cleanup script
- `run-benchmark.sh` - Benchmark script
- `gateway-configs/` - Gateway configuration files

## References

- [vLLM Data Parallel Deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)
- [Kubernetes Service Documentation](https://kubernetes.io/docs/concepts/services-networking/service/)
- Helm chart: `llm-d-modelservice` v0.2.11

---

**Simple, performant vLLM deployment with Kubernetes native load balancing for fair comparison with vllm-router and llm-d approaches.**
