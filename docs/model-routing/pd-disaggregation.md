# Prefill-Decode Disaggregation

This guide covers prefill-decode (P/D) disaggregation routing, which separates the prefill and decode phases for optimized resource utilization.

## Overview

Prefill-Decode disaggregation allows you to:
- Use different hardware for prefill (compute-intensive) and decode (memory-intensive) phases
- Scale prefill and decode workers independently
- Apply different load balancing policies to each phase

## Architecture

```
Client Request
      ↓
  vLLM Router
      ↓
   ┌──────┴──────┐
   ↓             ↓
Prefill       Decode
Workers       Workers
(GPU-heavy)   (Memory-heavy)
```

## NIXL Connector Mode

When vLLM uses the NIXL connector, you specify prefill and decode URLs directly.

### Basic Configuration

```bash
./target/release/vllm-router \
    --policy consistent_hash \
    --vllm-pd-disaggregation \
    --prefill http://127.0.0.1:8081 \
    --prefill http://127.0.0.1:8082 \
    --decode http://127.0.0.1:8083 \
    --decode http://127.0.0.1:8084 \
    --host 127.0.0.1 \
    --port 8090
```

### With Bootstrap Ports

For Mooncake integration, specify bootstrap ports:

```bash
./target/release/vllm-router \
    --vllm-pd-disaggregation \
    --prefill http://prefill1:8000 9000 \
    --prefill http://prefill2:8000 9001 \
    --decode http://decode1:8001 \
    --decode http://decode2:8001 \
    --policy cache_aware
```

### Different Policies for Prefill and Decode

Apply different load balancing strategies:

```bash
./target/release/vllm-router \
    --vllm-pd-disaggregation \
    --prefill http://prefill1:8000 \
    --prefill http://prefill2:8000 \
    --decode http://decode1:8001 \
    --decode http://decode2:8001 \
    --prefill-policy cache_aware \
    --decode-policy power_of_two
```

**Policy Recommendations:**
- **Prefill**: `cache_aware` or `consistent_hash` for prefix cache optimization
- **Decode**: `power_of_two` or `round_robin` for load distribution

## NCCL Connector Mode

When vLLM uses the NCCL connector, the router uses ZMQ-based service discovery.

### Basic Configuration

```bash
./target/release/vllm-router \
    --policy consistent_hash \
    --vllm-pd-disaggregation \
    --vllm-discovery-address 0.0.0.0:30001 \
    --host 0.0.0.0 \
    --port 10001 \
    --prefill-policy consistent_hash \
    --decode-policy consistent_hash
```

### Configuration Options

- `--vllm-discovery-address`: ZMQ endpoint for worker discovery
- `--prefill-policy`: Load balancing policy for prefill workers
- `--decode-policy`: Load balancing policy for decode workers

## Complete Example

### 1. Start vLLM Workers

**Prefill Workers:**
```bash
# Prefill worker 1
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8081 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'

# Prefill worker 2
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8082 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
```

**Decode Workers:**
```bash
# Decode worker 1
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8083 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'

# Decode worker 2
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8084 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
```

### 2. Start Router

```bash
./target/release/vllm-router \
    --vllm-pd-disaggregation \
    --prefill http://127.0.0.1:8081 \
    --prefill http://127.0.0.1:8082 \
    --decode http://127.0.0.1:8083 \
    --decode http://127.0.0.1:8084 \
    --prefill-policy cache_aware \
    --decode-policy power_of_two \
    --host 0.0.0.0 \
    --port 8090
```

### 3. Send Request

```bash
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Explain quantum computing"}]
  }'
```

## Working Examples

See the `scripts/` directory for complete working examples:

- **NIXL Connector**: `scripts/llama3.1/`
- **NCCL Connector**: `scripts/install.sh`

## Kubernetes Deployment

For P/D disaggregation in Kubernetes, see:
- [Kubernetes Guide](../environment/kubernetes.md)
- `scripts/k8s/llama3.1/` - Complete K8s deployment example

## Performance Tuning

### Prefill Workers
- Use GPUs with high compute capability
- Optimize for throughput
- Consider `cache_aware` policy for repeated prompts

### Decode Workers
- Use GPUs with high memory bandwidth
- Optimize for low latency
- Consider `power_of_two` policy for load balancing

## Troubleshooting

### Workers Not Connecting

Check that:
1. vLLM workers are running with correct KV transfer config
2. Network connectivity between router and workers
3. Ports are not blocked by firewall

### High Latency

- Monitor prefill/decode worker loads
- Adjust policies (`--prefill-policy`, `--decode-policy`)
- Scale workers independently based on bottleneck

## Next Steps

- [Basic Routing](basic-routing.md) - Standard routing configuration
- [Model Protection](../model-protection/README.md) - Retries and circuit breakers
- [Load Balancing](load-balancing.md) - Policy details
