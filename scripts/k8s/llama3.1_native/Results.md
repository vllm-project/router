# Llama 3.1 8B Native Routing - Benchmark Results

Results from testing vLLM's native routing with external load balancing (2 × DP8TP1).

## Configuration

- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Setup**: 2 pods × DP8TP1 (8 data parallel workers per pod)
- **Total Resources**: 16 GPUs (8 per pod) across 2 nodes
- **Load Balancing**:
  - Internal: vLLM's DP-aware scheduler (within each pod)
  - External: GAIE EPP (between pods)
- **Cluster**: AWS EKS
- **GPU Type**: (Add your GPU type here, e.g., NVIDIA A100/H100)

## Performance Benchmarks

Performance results from `vllm bench serve` running inside a decode pod.

### Command to Replicate

```bash
cd ~/router/scripts/k8s/llama3.1_native
./run-benchmark.sh
```

This runs 100 requests with max concurrency of 16, using input length 2000 and output length 2000 tokens.

### Results

| Metric | Value | Unit |
|--------|------:|------|
| **Throughput** |
| Request throughput | 1.19 | req/s |
| Output token throughput | 2,380.22 | tok/s |
| Peak output token throughput | 2,329.00 | tok/s |
| Total token throughput | 4,757.60 | tok/s |
| **Latency** |
| Mean TTFT | 80.55 | ms |
| Median TTFT | 73.40 | ms |
| P99 TTFT | 135.64 | ms |
| Mean TPOT | 5.98 | ms |
| Median TPOT | 5.97 | ms |
| P99 TPOT | 6.14 | ms |
| Mean ITL | 7.32 | ms |
| Median ITL | 5.92 | ms |
| P99 ITL | 26.04 | ms |
| **Request Stats** |
| Successful requests | 100 | - |
| Maximum request concurrency | 16 | - |
| Peak concurrent requests | 32.00 | - |
| Benchmark duration | 84.03 | s |
| Total input tokens | 199,761 | - |
| Total generated tokens | 200,000 | - |

**Key Performance Insights:**
- ✅ Consistent low latency: Median TTFT of 73.40ms
- ✅ High throughput: 2,380 tokens/s output
- ✅ Stable performance: Low variance in TPOT (5.97-6.14ms)
- ✅ Efficient batching: Handled 32 peak concurrent requests

## Model Evaluation

Accuracy evaluation using lm-eval harness on GSM8K benchmark.

### Command to Replicate

```bash
# Terminal 1: Start port-forward
kubectl port-forward -n llm-d-llama31-native \
  svc/infra-llama31-inference-gateway-istio 8000:80

# Terminal 2: Run evaluation
cd ~/router/scripts/k8s/llama3.1_native
export HF_TOKEN=your_token_here
./run-eval.sh gsm8k 1
```

### Results

**GSM8K (Grade School Math)**

| Task | Filter | n-shot | Metric | Value | Stderr |
|------|--------|-------:|--------|------:|-------:|
| gsm8k | flexible-extract | 5 | exact_match | **77.48%** | ±1.15% |
| gsm8k | strict-match | 5 | exact_match | **70.36%** | ±1.26% |

**Configuration:**
- Model: meta-llama/Llama-3.1-8B-Instruct
- Shots: 5-shot
- Concurrent requests: 1
- Samples: 1,319 (full test set)

**Key Evaluation Insights:**
- ✅ Strong math reasoning: 77.48% accuracy (flexible matching)
- ✅ Reliable performance: Low standard error (±1.15%)
- ✅ Expected for Llama 3.1 8B on GSM8K benchmark

## Reproducing These Results

### 1. Deploy the Stack

```bash
cd ~/router/scripts/k8s/llama3.1_native

# Create namespace and secret
kubectl create namespace llm-d-llama31-native
kubectl create secret generic llm-d-hf-token \
  --from-literal=HF_TOKEN=your_token \
  -n llm-d-llama31-native

# Deploy
./deploy.sh llm-d-llama31-native

# Wait for pods to be ready
kubectl get pods -n llm-d-llama31-native -w
```

### 2. Run Performance Benchmark

```bash
# Run benchmark from inside decode pod
./run-benchmark.sh 100 16
```

### 3. Run Evaluation

```bash
# Terminal 1: Port-forward
kubectl port-forward -n llm-d-llama31-native \
  svc/infra-llama31-inference-gateway-istio 8000:80

# Terminal 2: Run eval
export HF_TOKEN=your_token
./run-eval.sh gsm8k 1
```
