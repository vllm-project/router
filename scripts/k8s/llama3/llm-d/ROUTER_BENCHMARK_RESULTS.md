# vLLM Router Benchmark Results - llm-d Deployment

## Deployment Configuration

**Model:** Llama 3.1 8B Instruct
**Mode:** Prefill-Decode (P/D) Disaggregation
**Setup:** 8 Prefill pods + 8 Decode pods (DP8TP1 via Kubernetes pod replication)
**Router:** vllm-router with dynamic pod discovery and consistent hashing
**Data Parallel Size:** 1 (each pod has 1 GPU)

## Benchmark Configuration

**Tool:** `vllm bench serve`
**Test:** Random prompts (200 prompts)
**Input Length:** 1000 tokens
**Output Length:** 1000 tokens
**Endpoint:** `/v1/completions`

## Performance Results

### vLLM Router (via vllm-router-llama31 service)

| Concurrency | Output Throughput (tok/s) | Peak Output (tok/s) | Request Throughput (req/s) | Mean TTFT (ms) | Median TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) |
|-------------|--------------------------|---------------------|---------------------------|----------------|------------------|----------------|------------------|
| 32          | 4,752                    | 5,495               | 4.75                      | 79.26          | 59.63            | 5.95           | 5.92             |
| 64          | 6,626                    | 10,469              | 6.63                      | 115.64         | 69.53            | 6.37           | 6.38             |
| **128**     | **15,221**               | **19,286**          | **15.22**                 | 231.47         | 215.26           | 7.15           | 7.08             |
| 256         | 8,000                    | 10,042              | 8.00                      | 380.61         | 381.37           | 6.90           | 6.89             |

**Optimal Performance:** Concurrency 128
- **15,221 tok/s** output throughput
- **19,286 tok/s** peak output throughput
- Consistent TPOT around 7ms

## Architecture Details

### Dynamic Pod Discovery
- Init container discovers all pod IPs at startup using kubectl
- Builds individual `--prefill` and `--decode` arguments for each of 16 pods
- RBAC: ServiceAccount with pod get/list/watch permissions

### Routing Strategy
- Policy: Consistent hash with 160 virtual nodes
- Hash key: Request prompt text
- Each request routed to specific prefill and decode pods
- No reliance on Kubernetes service load balancing

### Configuration
```yaml
--vllm-pd-disaggregation
--intra-node-data-parallel-size 1
--policy consistent_hash
--host 0.0.0.0
--port 10001
```

## Key Observations

1. **Best Performance:** 15,221 tok/s at concurrency 128
2. **Performance Drop:** Throughput decreases at concurrency 256, suggesting saturation
3. **Consistent TPOT:** 6-7ms across all concurrency levels
4. **TTFT Scaling:** Mean TTFT increases with concurrency (79ms → 381ms)

## Manual Validation

**Test Request:** "Explain what machine learning is in simple terms."

**Routing:**
- Prefill: `http://10.2.1.131:8000` (pod index 3)
- Decode: `http://10.2.1.179:8200` (pod index 5)
- Latency: 564ms
- Response: Valid, coherent output

## Deployment Files

- `vllm-router-services.yaml`: Headless services for pod discovery
- `vllm-router-deployment-with-discovery.yaml`: Router with init container for dynamic discovery
- `run-eval.sh`: Evaluation script for lm_eval harness

## Service Endpoint

**Internal:** `vllm-router-llama31.llm-d-llama31.svc.cluster.local:10001`
**Metrics:** Port 29000
