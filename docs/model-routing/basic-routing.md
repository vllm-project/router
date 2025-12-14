# Basic Routing

This guide covers standard routing configurations for vLLM Router.

## Overview

In basic routing mode, vLLM Router distributes requests across multiple vLLM worker instances using various load balancing policies.

## Quick Start

### Using the Binary

```bash
./target/release/vllm-router \
    --worker-urls http://localhost:8000 http://localhost:8001 \
    --policy round_robin \
    --host 0.0.0.0 \
    --port 8080
```

### Using Python Launcher

After installing the Python package (`make install-python`):

```bash
vllm-router \
    --worker-urls http://localhost:8000 http://localhost:8001 \
    --policy round_robin \
    --host 0.0.0.0 \
    --port 8080
```

## Configuration Options

### Worker URLs

Specify backend vLLM instances using `--worker-urls`:

```bash
vllm-router \
    --worker-urls http://worker1:8000 http://worker2:8000 http://worker3:8000
```

### Load Balancing Policies

Choose a policy with `--policy`:

```bash
vllm-router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy consistent_hash
```

Available policies:
- `round_robin` - Sequential distribution (default for even load)
- `random` - Random selection
- `consistent_hash` - Session affinity for multi-turn conversations
- `power_of_two` - Load-aware selection
- `cache_aware` - Optimizes for prefix cache hits

See [Load Balancing](load-balancing.md) for detailed policy descriptions.

### Host and Port

Configure the router's listening address:

```bash
vllm-router \
    --worker-urls http://worker1:8000 \
    --host 0.0.0.0 \
    --port 8080
```

- `--host`: IP address to bind (default: `127.0.0.1`)
- `--port`: Port number (default: `30000`)

## Data Parallelism

For intra-node data parallelism, use `--intra-node-data-parallel-size`:

```bash
vllm-router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy consistent_hash \
    --intra-node-data-parallel-size 8
```

When `--intra-node-data-parallel-size > 1`, the router automatically creates DP-aware workers and routes requests accordingly.

## Request ID Tracking

Track requests across distributed systems with custom headers:

```bash
vllm-router \
    --worker-urls http://localhost:8000 \
    --request-id-headers x-trace-id x-request-id
```

**Default headers:** `x-request-id`, `x-correlation-id`, `x-trace-id`, `request-id`

## Session Affinity Example

For multi-turn conversations with `consistent_hash` policy:

```bash
# Start router with consistent_hash
vllm-router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy consistent_hash
```

Send requests with session ID:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "X-Session-ID: user-session-123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

All requests with the same `X-Session-ID` will be routed to the same worker, maximizing KV cache reuse.

## Complete Example

```bash
# Start vLLM workers (in separate terminals)
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001

# Start router
vllm-router \
    --worker-urls http://localhost:8000 http://localhost:8001 \
    --policy cache_aware \
    --host 0.0.0.0 \
    --port 8080

# Send request
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

## Advanced Configuration

For advanced features like retries, circuit breakers, and metrics, see:
- [Model Protection](../model-protection/README.md) - Retries and circuit breakers
- [Model Monitoring](../model-monitoring/configuration.md) - Metrics and observability

## Next Steps

- [Prefill-Decode Disaggregation](pd-disaggregation.md) - Separate prefill and decode phases
- [Kubernetes Deployment](../environment/kubernetes.md) - Service discovery in K8s
- [Load Balancing Policies](load-balancing.md) - Detailed policy comparison
