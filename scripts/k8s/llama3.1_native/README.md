# Llama 3.1 8B - vLLM Native Routing (2 × DP8TP1)

Simple deployment for benchmarking vLLM's native routing with 2 pods across 2 nodes using external data parallelism.

## Configuration

- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Setup**: 2 pods × DP8TP1 (8 data parallel workers per pod, 1 tensor parallel)
- **Total**: 16 workers across 2 nodes, 16 GPUs (8 per pod)
- **Load Balancing**:
  - Within pod: vLLM's internal DP scheduler (8 workers with DP-aware mode)
  - Between pods: GAIE EPP (external load balancing)
- **No P/D disaggregation**: Standard decode pods only
- **External DP**: Follows vLLM's [external load balancing pattern](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html#external-load-balancing)

## Architecture

```
Request → llm-d Gateway → GAIE EPP
                             ↓
                    ┌────────────────┐
                    │                │
                 Pod 1            Pod 2
               (Node 1)         (Node 2)
                 DP8              DP8
              8 workers        8 workers
               8 GPUs           8 GPUs
```

## Quick Start

### 1. Create Secret

```bash
kubectl create namespace llm-d-llama31-native
kubectl create secret generic llm-d-hf-token \
  --from-literal=HF_TOKEN=your_token_here \
  -n llm-d-llama31-native
```

### 2. Deploy

```bash
cd ~/router/scripts/k8s/llama3.1_native
./deploy.sh llm-d-llama31-native
```

### 3. Verify

```bash
# Check 2 pods on different nodes
kubectl get pods -n llm-d-llama31-native -o wide

# Check logs show DP8
kubectl logs -n llm-d-llama31-native <pod-name> | grep "data-parallel-size"
```

### 4. Apply HTTPRoute

```bash
kubectl apply -f httproute.yaml -n llm-d-llama31-native
```

### 5. Test

```bash
kubectl port-forward -n llm-d-llama31-native \
  svc/infra-llama31-inference-gateway-istio 8000:80

curl http://localhost:8000/v1/models
```

## Benchmarking

### Performance Benchmarks

Run performance benchmarks using `vllm bench serve` from inside a decode pod:

```bash
# Basic benchmark (100 prompts, concurrency 16)
./run-benchmark.sh

# Custom configuration
./run-benchmark.sh 1000 32  # 1000 prompts, concurrency 32

# Set custom input/output lengths
INPUT_LEN=1024 OUTPUT_LEN=256 ./run-benchmark.sh 500 16
```

**How it works:**
- Executes `vllm bench serve` from inside a decode pod (which has vLLM installed)
- Sends requests to the gateway service (simulating external client traffic)
- Tests the full stack: Gateway → GAIE EPP → vLLM pods
- Measures throughput, latency (P50/P95/P99), and time-to-first-token

**Benchmark parameters:**
- `num_prompts`: Total number of requests (default: 100)
- `concurrency`: Maximum concurrent requests (default: 16)
- `INPUT_LEN`: Random input token length (default: 2000)
- `OUTPUT_LEN`: Random output token length (default: 2000)

### Model Evaluation

Run accuracy evaluations using lm-eval harness:

```bash
# First, start port-forward in a separate terminal
kubectl port-forward -n llm-d-llama31-native \
  svc/infra-llama31-inference-gateway-istio 8000:80

# Set your HuggingFace token (needed for Llama 3.1 tokenizer)
export HF_TOKEN=your_hf_token_here

# Install lm-eval if not already installed
pip install lm-eval

# Run evaluation
./run-eval.sh                    # GSM8K (default)
./run-eval.sh mmlu               # MMLU benchmark
./run-eval.sh hellaswag          # HellaSwag benchmark
./run-eval.sh mmlu 4 50          # MMLU with 4 concurrent, limit 50 samples
```

**How it works:**
- Runs from your local machine (requires port-forward)
- Uses lm-eval harness to test model accuracy
- Downloads tokenizer locally (requires HF token for gated models)
- Sends requests through port-forward to the deployed model

**Evaluation parameters:**
- `task`: Evaluation task (gsm8k, mmlu, hellaswag, etc.)
- `num_concurrent`: Concurrent requests (default: 1, increase with caution)
- `limit`: Limit number of samples (optional, for quick testing)

**Available tasks:**
- `gsm8k` - Grade school math problems
- `mmlu` - Massive multitask language understanding
- `hellaswag` - Common sense reasoning
- `arc_challenge` - Science questions
- `truthfulqa` - Truthfulness evaluation

## What's Different from P/D Stack

| Aspect | P/D Stack (llama3.1) | This (llama3.1_native) |
|--------|---------------------|---------------------------|
| **Prefill** | 1 pod, 8 GPUs | **Disabled** |
| **Decode** | 1 pod, 8 GPUs | **2 pods, 8 GPUs each** |
| **Total Pods** | 2 (prefill + decode) | **2 (decode only)** |
| **KV Transfer** | Yes (NIXL) | **No** |
| **Routing Logic** | P/D scheduling | **Simple load balancing** |
| **Use Case** | P/D disaggregation | **Benchmark vLLM native routing** |

## Cleanup

```bash
./cleanup.sh llm-d-llama31-native
kubectl delete namespace llm-d-llama31-native  # Optional
```

## Notes

- Each pod runs vLLM with `--data-parallel-size 8`
- vLLM handles internal load balancing across its 8 workers
- GAIE EPP provides simple external load balancing between the 2 pods
- Same pattern as P/D stack, just without prefill/decode separation
- Perfect for benchmarking vLLM's native routing capabilities

## Results

See [Results.md](Results.md) for detailed performance benchmarks and evaluation results, including:
- Throughput and latency metrics
- GSM8K evaluation accuracy
- Commands to reproduce all results
