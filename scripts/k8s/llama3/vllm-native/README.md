# Llama 3.1 8B with Kubernetes Native Load Balancing

vLLM deployment with 16 independent pods using Kubernetes native load balancing.

## Overview

This setup implements **Kubernetes-native load balancing** where:
- **16 independent vLLM pods**: Each pod runs a complete vLLM instance (1 GPU)
- **No DP coordinator**: Pods operate independently without coordination
- **Kubernetes Service**: Load balances requests across all 16 pods
- **Total capacity**: 16 pods with 16 GPUs total (1 GPU per pod)

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
└───┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬─┘
    │    │    │    │    │    │    │    │    │    │    │    │
    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
┌─────────────────────────────────────────────────────────────┐
│              16 Independent vLLM Pods                        │
│  ┌─────┐ ┌─────┐ ┌─────┐       ┌─────┐ ┌─────┐ ┌─────┐    │
│  │Pod 1│ │Pod 2│ │Pod 3│  ...  │Pod14│ │Pod15│ │Pod16│    │
│  │1 GPU│ │1 GPU│ │1 GPU│       │1 GPU│ │1 GPU│ │1 GPU│    │
│  └─────┘ └─────┘ └─────┘       └─────┘ └─────┘ └─────┘    │
│                                                              │
│  • Each pod: standalone vLLM server                         │
│  • No inter-pod communication                               │
│  • Independent request processing                           │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Summary

**Pods:** 16 independent pods
- Each pod: 1 GPU, complete vLLM instance
- No master/worker relationship
- All pods are identical

**Resources:**
- 16 GPUs total (1 GPU per pod)
- No tensor parallelism (TP=1)
- No data parallelism (no DP flags)

**Load Balancing:**
- Kubernetes Service handles request distribution
- Round-robin or session-affinity based
- Not queue-aware (limitation)

## How It Works

1. **Client sends request** → Kubernetes Service on port 8000
2. **K8s Service load balances** → Routes to one of 16 pods (round-robin)
3. **Pod processes independently** → No coordination with other pods
4. **Response returned** → Direct from the pod that processed the request

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
- Deploy 16 independent vLLM pods via Helmfile
- Create Kubernetes service for load balancing
- Each pod downloads model independently

Wait for pods to be ready (10-15 minutes first time for model download).

### 3. Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n llm-d-llama31-multinode -l llm-d.ai/role=decode

# Expected output: 16 pods in Running state
```

### 4. Port Forward

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

# 2. Wait for all pods ready (16 pods)
kubectl get pods -n llm-d-llama31-multinode -w

# 3. Port forward
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
kubectl get pods -n llm-d-llama31-multinode -l llm-d.ai/role=decode
```

**Expected**: 16 pods in Running/Ready state

### Check Service

```bash
kubectl get svc -n llm-d-llama31-multinode ms-llama31-multinode-llm-d-modelservice-decode
```

**Expected**: Service with port 8000/TCP and 16 endpoints

### Check Service Endpoints

```bash
kubectl get endpoints -n llm-d-llama31-multinode ms-llama31-multinode-llm-d-modelservice-decode
```

**Expected**: 16 pod IPs listed as endpoints

### Check Pod Logs

```bash
# Check logs from all pods
kubectl logs -n llm-d-llama31-multinode -l llm-d.ai/role=decode -c vllm --tail=50

# Check specific pod
kubectl logs -n llm-d-llama31-multinode <pod-name> -c vllm -f
```

**Look for**: `Uvicorn running on http://0.0.0.0:8000`

### Check GPU Utilization

```bash
# Check GPU usage on a specific pod
kubectl exec -n llm-d-llama31-multinode <pod-name> -c vllm -- nvidia-smi
```

**Expected**: During load, GPUs should show utilization > 0%

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

### Service Has No Endpoints

```bash
kubectl get endpoints -n llm-d-llama31-multinode
```

If no endpoints, check that pods have `llm-d.ai/role: decode` label

### Model Download Issues

Each pod downloads the model independently (EmptyDir storage):
- First deployment: 10-15 minutes per pod (parallel downloads)
- Subsequent deployments: Still need to download (no shared cache)

### Uneven Load Distribution

K8s service load balancing is simple round-robin:
- Not queue-aware
- Doesn't consider pod load
- May result in uneven distribution under high load

## Key Architecture Decisions

### Why K8s Load Balancing?

Per vLLM expert recommendation for non-MoE models like Llama 3.1:
- **DP coordinator not needed** for non-MoE models
- **DP has overhead** that makes it slower than external routing (#24461)
- **K8s native LB** is simpler and more performant

### Comparison with Other Approaches

| Aspect | vLLM-Native (K8s LB) | vllm-router | llm-d |
|--------|---------------------|-------------|-------|
| **Pods** | 16 independent | 8 prefill + 8 decode | 8 prefill + 8 decode |
| **Load balancing** | K8s service | External router | Gateway + sidecar |
| **Queue awareness** | No | Yes | Yes |
| **P/D disaggregation** | No | Yes | Yes |
| **Complexity** | Lowest | Medium | Highest |

### Limitations

1. **Not queue-aware**: K8s doesn't know pod queue lengths
2. **No P/D disaggregation**: All pods do both prefill and decode
3. **No shared model cache**: Each pod downloads independently

### When to Use This

✅ **Good for:**
- Simple deployments
- Testing vLLM native performance
- Comparing against external routing approaches

❌ **Not recommended for:**
- MoE models (use DP coordinator instead)
- Scenarios requiring P/D disaggregation
- Production with strict SLA requirements

## Files

- `values.yaml` - Helm values for 16 independent pods
- `vllm-service.yaml` - Kubernetes service for load balancing
- `helmfile.yaml.gotmpl` - Helmfile configuration
- `deploy.sh` - Deployment script
- `cleanup.sh` - Cleanup script
- `run-benchmark.sh` - Benchmark script

## References

- [vLLM Issue #24461](https://github.com/vllm-project/vllm/issues/24461) - DP overhead for non-MoE models
- [Kubernetes Service Documentation](https://kubernetes.io/docs/concepts/services-networking/service/)
- Helm chart: `llm-d-modelservice` v0.2.11

---

**Simple, performant vLLM deployment with Kubernetes native load balancing.**
