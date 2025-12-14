# vLLM Router

A high-performance and light-weight request forwarding system for vLLM large scale deployments, providing advanced load balancing methods and prefill/decode disaggregation support.

## Key Features

- **Core Architecture**: Request routing framework and async processing patterns
- **Load Balancing**: Multiple algorithms (cache-aware, power of two, consistent hashing, random, round robin)
- **Prefill-Decode Disaggregation**: Specialized routing for separated processing phases
- **Service Discovery**: Kubernetes-native worker management and health monitoring
- **Enterprise Features**: Circuit breakers, retry logic, metrics collection

## Quick Start

### Installation

```bash
# One-command installation
make install

# Or step-by-step
make check-deps        # Check dependencies
make install-deps      # Install system dependencies
make build             # Build Rust binary and Python package
make install-python    # Install Python package
```

**Supported Systems:** Linux (Ubuntu/Debian, RHEL/CentOS/Fedora) and macOS

**Prerequisites:** Rust, Protocol Buffers compiler, Python 3.8+

📖 **Detailed installation guide:** [docs/installation.md](docs/installation.md)

### Basic Usage

#### Standard Routing

```bash
# Using the binary
./target/release/vllm-router \
    --worker-urls http://localhost:8000 http://localhost:8001 \
    --policy round_robin \
    --host 0.0.0.0 \
    --port 8080

# Using Python launcher (after make install-python)
vllm-router \
    --worker-urls http://localhost:8000 http://localhost:8001 \
    --policy consistent_hash
```

#### Data Parallelism

```bash
./target/release/vllm-router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy consistent_hash \
    --intra-node-data-parallel-size 8
```

#### Prefill-Decode Disaggregation

```bash
# NIXL connector mode
./target/release/vllm-router \
    --vllm-pd-disaggregation \
    --prefill http://127.0.0.1:8081 \
    --prefill http://127.0.0.1:8082 \
    --decode http://127.0.0.1:8083 \
    --decode http://127.0.0.1:8084 \
    --policy cache_aware

# NCCL connector mode (ZMQ discovery)
./target/release/vllm-router \
    --vllm-pd-disaggregation \
    --vllm-discovery-address 0.0.0.0:30001 \
    --prefill-policy consistent_hash \
    --decode-policy power_of_two
```

📖 **Usage guides:**
- [Basic Routing](docs/model-routing/basic-routing.md)
- [Prefill-Decode Disaggregation](docs/model-routing/pd-disaggregation.md)
- [Kubernetes Deployment](docs/environment/kubernetes.md)

## Load Balancing Policies

The router supports multiple load balancing policies:

| Policy | Description | Session Affinity | Use Case |
|--------|-------------|------------------|----------|
| `round_robin` | Sequential distribution across workers | No | General purpose, even distribution |
| `random` | Uniform random selection | No | Simple deployments |
| `consistent_hash` | Routes same session/user to same worker | Yes | Multi-turn chat, KV cache reuse |
| `power_of_two` | Picks least loaded of two random workers | No | Load-sensitive workloads |
| `cache_aware` | Optimizes for prefix cache hits | Yes | Repeated prompts, few-shot |

**Example:** Using `consistent_hash` with session affinity:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "X-Session-ID: user-123" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3", "messages": [{"role": "user", "content": "Hello!"}]}'
```

📖 **Learn more:** [Load Balancing Documentation](docs/model-routing/load-balancing.md)

## Documentation

- **[Installation Guide](docs/installation.md)** - Detailed installation instructions
- **[Basic Routing](docs/model-routing/basic-routing.md)** - Standard routing configuration
- **[Load Balancing](docs/model-routing/load-balancing.md)** - Policy details and comparison
- **[Prefill-Decode Disaggregation](docs/model-routing/pd-disaggregation.md)** - P/D separation
- **[Kubernetes Deployment](docs/environment/kubernetes.md)** - K8s service discovery
- **[Model Protection](docs/model-protection/README.md)** - Retries and circuit breakers
- **[Model Monitoring](docs/model-monitoring/configuration.md)** - Metrics and observability

## Development

For development setup, build instructions, and contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgement

This project is a fork of [SGLang Model Gateway](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway), and we would like to explicitly acknowledge and thank the original authors for their work. At this stage, our fork includes only minimal changes to preserve the existing interface and ensure compatibility with vLLM. We anticipate further divergence as we pursue the roadmap we have in mind, which is the reason for creating the fork.
