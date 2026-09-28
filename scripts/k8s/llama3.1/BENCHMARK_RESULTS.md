# Llama 3.1 8B DP8TP1 P/D Disaggregation Benchmark Results

## Overview

This document contains benchmark results for Llama 3.1 8B Instruct deployed with Prefill-Decode (P/D) disaggregation using Data Parallelism (DP8) and vLLM router with DP-aware routing.

## Configuration

### Model
- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Model Size**: ~8B parameters
- **Max Model Length**: 32,000 tokens

### Parallelism Strategy
- **Data Parallel Size**: 8 (DP8)
- **Tensor Parallel Size**: 1 (TP1)
- **Configuration**: DP8TP1
- **Total Workers**: 16 (8 prefill + 8 decode)

### Infrastructure
- **Namespace**: llm-d-llama31
- **Prefill Pod**: 1 pod with 8 GPUs (8 DP workers internally)
- **Decode Pod**: 1 pod with 8 GPUs (8 DP workers internally)
- **Router**: vllm-router with DP-aware routing (--intra-node-data-parallel-size 8)
- **Resources per Pod**: 8 GPUs, 800Gi memory, 180 CPUs

### Router Configuration
```bash
vllm-router \
    --vllm-pd-disaggregation \
    --prefill http://10.2.1.187:8000 \
    --decode http://10.2.1.133:8200 \
    --intra-node-data-parallel-size 8 \
    --host 0.0.0.0 \
    --port 10001
```

## Benchmark Setup

### Test Parameters
- **Dataset**: Random synthetic data
- **Number of Prompts**: 100
- **Input Length**: 2000 tokens
- **Output Length**: 2000 tokens
- **Maximum Concurrency**: 32
- **Request Rate**: Unlimited (as fast as possible)
- **Burstiness Factor**: 1.0 (Poisson process)

### Command
```bash
vllm bench serve \
    --dataset-name random \
    --num-prompts 100 \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --random-input-len 2000 \
    --random-output-len 2000 \
    --endpoint /v1/completions \
    --max-concurrency 32 \
    --save-result \
    --ignore-eos \
    --served-model-name "meta-llama/Llama-3.1-8B-Instruct" \
    --host vllm-router-llama31 \
    --port 10001
```

## Results

### Overall Performance
| Metric | Value |
|--------|-------|
| Successful Requests | 100/100 (100%) |
| Benchmark Duration | 57.83 s |
| Request Throughput | 1.73 req/s |
| Output Token Throughput | 3,458.42 tok/s |
| Peak Output Token Throughput | 3,680.00 tok/s |
| Total Token Throughput | 6,912.71 tok/s |
| Peak Concurrent Requests | 64.00 |

### Token Statistics
| Metric | Value |
|--------|-------|
| Total Input Tokens | 199,761 |
| Total Generated Tokens | 200,000 |
| Average Input per Request | ~1,998 tokens |
| Average Output per Request | 2,000 tokens |

### Time to First Token (TTFT)
| Metric | Value |
|--------|-------|
| Mean TTFT | 280.18 ms |
| Median TTFT | 288.96 ms |
| P99 TTFT | 433.73 ms |

### Time per Output Token (TPOT)
Excluding first token:

| Metric | Value |
|--------|-------|
| Mean TPOT | 7.25 ms |
| Median TPOT | 7.26 ms |
| P99 TPOT | 7.37 ms |

### Inter-Token Latency (ITL)
| Metric | Value |
|--------|-------|
| Mean ITL | 9.70 ms |
| Median ITL | 6.75 ms |
| P99 ITL | 32.48 ms |

## Key Findings

### Success Rate
- **100% success rate** with all 100 requests completed successfully
- No failed requests or timeouts
- Responses were correct and not corrupted

### DP-Aware Routing
The router successfully expanded URLs to 16 DP-aware worker endpoints:
- 8 prefill workers: `http://10.2.1.187:8000@0` through `@7`
- 8 decode workers: `http://10.2.1.133:8200@0` through `@7`

This ensures proper KV cache affinity between matching prefill and decode worker ranks.

### Performance Characteristics
1. **Stable Latency**:
   - Consistent TPOT around 7.25ms
   - Low variance in inter-token latency (median 6.75ms)

2. **Good Throughput**:
   - Peak output token throughput of 3,680 tok/s
   - Efficient utilization of 8 DP workers

3. **Fast Prefill**:
   - Mean TTFT of 280ms for 2000-token prompts
   - Dedicated prefill workers enable quick prompt processing

## Comparison: Router vs GAIE Gateway

### GAIE Gateway (Without DP-Aware Routing)
- **Result**: Corrupted responses
- **Issue**: Gateway treats each pod as single endpoint, breaking KV cache affinity
- **Example Output**: "Ilie for. Ilie. Ilie..." (incorrect)

### vLLM Router (With DP-Aware Routing)
- **Result**: Correct responses ✓
- **Implementation**: Expands to 16 worker URLs with rank information
- **Example Output**: "The capital of France is Paris." (correct)

## Architecture Notes

### Why DP8 Instead of TP8?
For Llama 3.1 8B model:
- Model is small enough to fit on a single GPU
- Data parallelism provides better throughput scaling
- Each DP worker has full model, enabling independent request processing
- Reduced complexity compared to tensor parallelism coordination

### Internal DP Configuration
Important: DP8 is configured via `--intra-node-data-parallel-size 8` flag in vLLM, NOT by creating 8 separate pods. Each pod runs a single vLLM process that internally manages 8 data parallel workers across 8 GPUs.

### KV Cache Transfer Flow
1. GAIE EPP routes prefill request to prefill pod
2. Prefill worker #N processes prompt
3. KV cache transferred via NIXL from prefill worker #N to decode worker #N
4. Subsequent decode requests maintain affinity with decode worker #N
5. Router ensures correct rank matching throughout the process

## Deployment Files

- `values.yaml` - Model service configuration
- `gaie-llama31/values.yaml` - GAIE configuration
- `router-deployment.yaml` - Router deployment manifest
- `router-service.yaml` - Router service definitions
- `helmfile.yaml.gotmpl` - Helmfile orchestration
- `deploy.sh` - Deployment automation
- `cleanup.sh` - Cleanup script

## Conclusion

The Llama 3.1 8B deployment with DP8TP1 configuration demonstrates:
- ✓ **Functional Correctness**: 100% success rate with accurate responses
- ✓ **DP-Aware Routing**: Proper worker rank matching via router
- ✓ **Good Performance**: 3,458 tok/s output throughput
- ✓ **Stable Latency**: Consistent TPOT around 7.25ms
- ✓ **Efficient P/D**: Mean TTFT of 280ms, fast prefill processing

**Recommendation**: For DP-based deployments with P/D disaggregation, use the vLLM router with `--intra-node-data-parallel-size` set to match your workers' DP size to ensure correct KV cache affinity and prevent response corruption.

## Test Date
**Benchmark Run**: October 28, 2025
