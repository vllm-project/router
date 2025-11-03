# Llama 3.1 8B Prefill-Decode Disaggregation with llm-d

Production-ready deployment of Llama 3.1 8B with GAIE-powered Prefill/Decode (P/D) disaggregation using pure Kubernetes pod replication for data parallelism.

## Overview

This deployment implements **DP8TP1** (8-way data parallelism, 1-way tensor parallelism) with P/D disaggregation:
- **8 Prefill pods**: Handle prompt processing (1 GPU each)
- **8 Decode pods**: Handle token generation (1 GPU each) + routing-proxy sidecar
- **Total**: 16 GPUs across 16 pods
- **Routing**: GAIE (Gateway API Inference Extension) with EPP (Endpoint Picker Plugin) for intelligent P/D routing

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Request                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          Istio Gateway (infra-llama31-inference-gateway)     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                HTTPRoute (llm-d-llama31-pd-disaggregation)   │
│                      ↓                                        │
│                InferencePool (gaie-llama31)                  │
│                      ↓                                        │
│                  GAIE EPP (scheduling)                       │
└──────────────────┬───────────────┬──────────────────────────┘
                   │               │
         ┌─────────┘               └─────────┐
         ▼                                   ▼
┌─────────────────────┐           ┌─────────────────────┐
│  Prefill Pods (×8)  │           │  Decode Pods (×8)   │
│  • 1 GPU each       │           │  • 1 GPU each       │
│  • Port 8000        │←─────────→│  • Port 8200        │
│  • vLLM only        │  KV Cache │  • vLLM + proxy     │
│  • Prompt proc.     │  Transfer │  • Token gen.       │
└─────────────────────┘  (Nixl)   └─────────────────────┘
```

## Key Configuration Pattern

**Critical**: The llm-d-modelservice chart uses pure Kubernetes replication for data parallelism, NOT vLLM's internal `--data-parallel-size`.

### Correct Pattern (DP8TP1):
```yaml
decode/prefill:
  create: true
  replicas: 8              # 8 Kubernetes pods
  # NO parallelism section!
  containers:
  - name: vllm
    env:
      - name: CUDA_VISIBLE_DEVICES
        value: "0"         # Single GPU per pod
    resources:
      limits:
        nvidia.com/gpu: "1"  # 1 GPU per pod
        memory: 100Gi        # Per-pod resources
        cpu: "22"
```

### ❌ WRONG Pattern (causes crashes):
```yaml
decode/prefill:
  parallelism:
    data: 8    # This passes --data-parallel-size 8 to vLLM
  replicas: 8  # But each pod only has 1 GPU!
  resources:
    nvidia.com/gpu: "8"  # Doesn't work - chart divides by parallelism.data
```

## Prerequisites

- Kubernetes cluster with 16+ GPUs across multiple nodes
- `kubectl` and `helmfile` installed
- HuggingFace token with Llama 3.1 access

## Deployment

### 1. Create Secret

```bash
kubectl create namespace llm-d-llama31

kubectl create secret generic llm-d-hf-token \
  --from-literal=HF_TOKEN=hf_xxxxxxxxxxxxx \
  -n llm-d-llama31
```

### 2. Deploy

```bash
cd /home/congc/router/scripts/k8s/llm-d
helmfile apply
```

Wait for all pods to be ready (5-10 minutes first time for model download, ~2 minutes after).

### 3. Verify Deployment

```bash
# Check all 16 pods are running
kubectl get pods -n llm-d-llama31 -l llm-d.ai/inferenceServing=true

# Expected output:
# - 8 prefill pods (1/1 ready)
# - 8 decode pods (2/2 ready: vllm + routing-proxy)

# Check GAIE components
kubectl get inferencepool -n llm-d-llama31
kubectl get httproute -n llm-d-llama31
kubectl get gateway -n llm-d-llama31
```

### 4. Test Inference

```bash
# Get a pod to run test from
PREFILL_POD=$(kubectl get pods -n llm-d-llama31 -l llm-d.ai/role=prefill -o jsonpath='{.items[0].metadata.name}')

# Test inference through gateway
kubectl exec -n llm-d-llama31 $PREFILL_POD -c vllm -- \
  curl -s -X POST "http://infra-llama31-inference-gateway-istio/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "prompt": "Hello, what is AI?", "max_tokens": 50}'
```

## Benchmark Results

Performance with 1K input / 1K output tokens:

| Concurrency | Output Tok/s | Peak Tok/s | Mean TTFT | Mean TPOT | Duration |
|-------------|--------------|------------|-----------|-----------|----------|
| 32          | 3,710        | 5,463      | 829ms     | 7.41ms    | 53.9s    |
| 64          | 5,814        | 9,195      | 1.47s     | 8.46ms    | 34.4s    |
| **128**     | **7,590**    | **15,252** | 3.70s     | 9.01ms    | 26.4s    |

**Key Metrics:**
- ✅ **Throughput Scaling**: 57% increase from 32→64, 31% increase from 64→128 concurrency
- ✅ **Peak Performance**: 15,252 tok/s at concurrency 128
- ✅ **Stable Latency**: TPOT remains 7-9ms across all concurrency levels

## Running Benchmarks

```bash
# Get a decode pod to run benchmark from
DECODE_POD=$(kubectl get pods -n llm-d-llama31 -l llm-d.ai/role=decode -o jsonpath='{.items[0].metadata.name}')

# Run benchmark (200 prompts, concurrency 32)
kubectl exec -n llm-d-llama31 $DECODE_POD -c vllm -- \
  vllm bench serve \
    --dataset-name random \
    --num-prompts 200 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --random-input-len 1000 \
    --random-output-len 1000 \
    --endpoint /v1/completions \
    --max-concurrency 32 \
    --save-result \
    --ignore-eos \
    --served-model-name meta-llama/Llama-3.1-8B-Instruct \
    --host infra-llama31-inference-gateway-istio \
    --port 80
```

## Monitoring

### Check Pod Logs

```bash
# Prefill pod logs
kubectl logs -n llm-d-llama31 -l llm-d.ai/role=prefill -c vllm --tail=50

# Decode pod logs (vLLM container)
kubectl logs -n llm-d-llama31 -l llm-d.ai/role=decode -c vllm --tail=50

# Decode pod logs (routing-proxy sidecar)
kubectl logs -n llm-d-llama31 -l llm-d.ai/role=decode -c routing-proxy --tail=50

# GAIE EPP logs
kubectl logs -n llm-d-llama31 -l inferencepool=gaie-llama31-epp --tail=50
```

### Check InferencePool Status

```bash
kubectl describe inferencepool gaie-llama31 -n llm-d-llama31
```

### Monitor GPU Utilization

```bash
# Check GPU usage on prefill pod
PREFILL_POD=$(kubectl get pods -n llm-d-llama31 -l llm-d.ai/role=prefill -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n llm-d-llama31 $PREFILL_POD -c vllm -- nvidia-smi

# Check GPU usage on decode pod
DECODE_POD=$(kubectl get pods -n llm-d-llama31 -l llm-d.ai/role=decode -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n llm-d-llama31 $DECODE_POD -c vllm -- nvidia-smi
```

## Troubleshooting

### Pods Not Scheduling

```bash
kubectl describe pod -n llm-d-llama31 <pod-name>
```

Check for: GPU availability, node resources, memory constraints

### Pods Crashing (CrashLoopBackOff)

```bash
# Check logs for error
kubectl logs -n llm-d-llama31 <pod-name> -c vllm --tail=100
```

**Common issue**: If you see `RuntimeError: Engine core initialization failed` with `Failed core proc(s): {'EngineCore_DP1': 1, ...}`, this means:
- The chart is passing `--data-parallel-size N` to vLLM, but the pod only has 1 GPU
- **Solution**: Remove the `parallelism` section from values.yaml (use pure Kubernetes replication)

### InferencePool Not Found

```bash
# Check if InferencePool exists
kubectl get inferencepool -n llm-d-llama31

# If not found, manually create it
kubectl apply -f - <<EOF
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferencePool
metadata:
  name: gaie-llama31
  namespace: llm-d-llama31
spec:
  targetPortNumber: 8000
  selector:
    llm-d.ai/inferenceServing: "true"
  extensionRef:
    name: gaie-llama31-epp
    portNumber: 9002
    failureMode: FailClose
EOF
```

### No Inference Response / Gateway Timeout

1. Check if vLLM is ready:
```bash
PREFILL_POD=$(kubectl get pods -n llm-d-llama31 -l llm-d.ai/role=prefill -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n llm-d-llama31 $PREFILL_POD -c vllm -- \
  curl -s http://localhost:8000/v1/models
```

2. Check GAIE EPP is discovering pods:
```bash
kubectl logs -n llm-d-llama31 -l inferencepool=gaie-llama31-epp | grep -i "pod\|endpoint"
```

3. Check HTTPRoute status:
```bash
kubectl describe httproute llm-d-llama31-pd-disaggregation -n llm-d-llama31
```

## Cleanup

```bash
# Remove deployment
helmfile destroy

# Delete namespace (including PVCs with cached model)
kubectl delete namespace llm-d-llama31
```

## Key Differences from vllm-native Multi-Node DP

| Aspect | llm-d (This Setup) | vllm-native Multi-Node |
|--------|-------------------|------------------------|
| **DP Implementation** | Kubernetes pod replication | vLLM internal `--data-parallel-size` |
| **Coordination** | GAIE EPP for P/D routing | vLLM DP Coordinator (rank 0) |
| **Service Discovery** | InferencePool + Gateway | Kubernetes Service + RPC |
| **Load Balancing** | GAIE intelligent routing | vLLM internal round-robin |
| **P/D Disaggregation** | Native GAIE feature | Manual configuration |
| **Scaling** | Independent prefill/decode scaling | Must scale ranks together |

## Configuration Reference

### Model Configuration

Located in `values.yaml`:

```yaml
modelArtifacts:
  uri: "hf://meta-llama/Llama-3.1-8B-Instruct"
  size: 200Gi
  authSecretName: "llm-d-hf-token"
  name: "meta-llama/Llama-3.1-8B-Instruct"
```

### Routing Configuration

```yaml
routing:
  servicePort: 8000
  proxy:
    image: ghcr.io/llm-d/llm-d-routing-sidecar:v0.3.0
    connector: nixlv2
    secure: false
```

### Resource Limits

Per-pod (for 8 replicas):
- **GPU**: 1 × H100 (or similar)
- **Memory**: 100 Gi
- **CPU**: 22 cores
- **Shared Memory**: 16 Gi

Total for deployment:
- **GPUs**: 16 (8 prefill + 8 decode)
- **Memory**: 1.6 Ti (16 × 100 Gi)
- **CPUs**: 352 cores (16 × 22)

## References

- [llm-d GitHub](https://github.com/llm-d-incubation)
- [llm-d-modelservice Helm Chart](https://github.com/llm-d-incubation/llm-d-modelservice)
- [Gateway API Inference Extension (GAIE)](https://github.com/kubernetes-sigs/gateway-api-inference-extension)
- [vLLM P/D Disaggregation](https://docs.vllm.ai/en/latest/serving/disaggregated_prefill.html)

---

**Built for production-grade Prefill-Decode disaggregation with intelligent GAIE routing.** 🚀
