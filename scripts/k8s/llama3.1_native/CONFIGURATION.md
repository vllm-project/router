# Configuration Summary

## Setup: 2 Pods × DP8TP1 (vLLM Native Routing)

This configuration implements vLLM's external data parallelism pattern for benchmarking native routing.

### Key Configuration

```yaml
decode:
  replicas: 2              # 2 pods on 2 different nodes
  parallelism:
    tensor: 8              # Tells Helm to allocate 8 GPUs per pod
    data: 1                # External DP via multiple pods

  # Ensures pods go to different nodes
  podAntiAffinity: enabled

  vLLM args:
    --tensor-parallel-size 1   # TP1: Model fits on 1 GPU
    --data-parallel-size 8     # DP8: 8 workers per pod
    # No kv-transfer-config (no P/D)

prefill:
  create: false            # Disabled (no P/D disaggregation)
```

### GAIE Routing

```yaml
# Simple load balancing (no P/D logic)
plugins:
  - type: max-score-picker
```

## Architecture

```
User Request
    ↓
llm-d Gateway
    ↓
GAIE EPP (simple LB)
    ↓
┌─────────────────┐
│                 │
Pod 1          Pod 2
Node 1         Node 2
DP8 (vLLM)     DP8 (vLLM)
8 workers      8 workers
8 GPUs         8 GPUs
```

## Load Balancing

- **Between pods**: GAIE EPP (simple load balancing)
- **Within pod**: vLLM's internal scheduler (handles 8 workers)

## Differences from Original llama3.1

| Change | Original | This Configuration |
|--------|----------|-------------------|
| Prefill pods | 1 pod, 8 GPUs | **Disabled** |
| Decode pods | 1 pod, 8 GPUs | **2 pods, 8 GPUs each** |
| KV transfer | Yes (NIXL) | **No** |
| P/D scheduling | Yes | **No (simple LB)** |
| Container port | 8200 (decode) | **8000** |

## Files

- `values.yaml` - Main configuration (2 × DP8, no prefill)
- `gaie-llama31/values.yaml` - Simplified routing (no P/D)
- `deploy.sh` - Standard deployment script
- `cleanup.sh` - Cleanup script
- `httproute.yaml` / `httproute.gke.yaml` - HTTPRoute configs
- `helmfile.yaml.gotmpl` - Helmfile template (unchanged)
- `gateway-configs/` - Gateway configurations (unchanged)

## Deployment

```bash
cd ~/router/scripts/k8s/llama3.1_native

# Create secret
kubectl create namespace llm-d-llama31-native
kubectl create secret generic llm-d-hf-token \
  --from-literal=HF_TOKEN=your_token \
  -n llm-d-llama31-native

# Deploy
./deploy.sh llm-d-llama31-native

# Apply HTTPRoute
kubectl apply -f httproute.yaml -n llm-d-llama31-native

# Test
kubectl port-forward -n llm-d-llama31-native \
  svc/infra-llama31-inference-gateway-istio 8000:80
curl http://localhost:8000/v1/models
```

## What This Tests

✅ vLLM's internal load balancing (8 workers per pod)
✅ Multi-node deployment (2 nodes, 16 GPUs total)
✅ Simple external load balancing (GAIE between pods)
✅ No custom routing logic (standard vLLM + llm-d)
✅ Continuous batching within each pod

This is the standard pattern for vLLM multi-node deployment!
