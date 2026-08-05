# vLLM Router

A high-performance and light-weight request forwarding system for vLLM large scale deployments, providing advanced load balancing methods and prefill/decode disaggregation support.

### Key Features

- **Core Architecture**: Request routing framework and async processing patterns
- **Load Balancing**: Multiple algorithms (cache-aware, power of two, consistent hashing, random, round robin)
- **Prefill-Decode Disaggregation**: Specialized routing for separated processing phases
- **Service Discovery**: Kubernetes-native worker management and health monitoring
- **Enterprise Features**: Circuit breakers, retry logic, metrics collection

## Quick Start

### Prerequisites

**Rust and Cargo:**
```bash
# Install rustup (Rust installer and version manager)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Follow the installation prompts, then reload your shell
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version

```

**Python with pip installed**

### Installation & Basic Usage

#### Rust Binary
```bash
# Build Rust components
cargo build --release
```

#### Python Package
Install from PyPI
```bash
pip install vllm-router                                                                                                                                                        ```

To build from source:
```bash    
pip install setuptools-rust wheel build
python -m build
pip install dist/*.whl

# Rebuild & reinstall in one step during development
python -m build && pip install --force-reinstall dist/*.whl
```

### Usage Examples

#### Standard Data Parallelism Routing
```bash
# Launch router with data parallelism (8 replicas per worker URL)
# When data-parallel-size > 1, the router automatically creates DP-aware workers
./target/release/vllm-router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy consistent_hash \
    --intra-node-data-parallel-size 8

# Alternative: using cargo run
cargo run --release -- \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy consistent_hash \
    --intra-node-data-parallel-size 8

# Alternative: using python launcher
vllm-router \
  --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy consistent_hash \
    --intra-node-data-parallel-size 8
```

#### Additional Generate Paths

Register extension endpoints that use the `/inference/v1/generate` request
schema. Configured paths get typed routing, active-load accounting, retries,
and circuit breaking. Built-in routes keep their existing handlers. Workers
must expose same paths.

```bash
vllm-router \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --policy cache_aware \
  --extra-generate-paths /custom/v1/generate
```

#### Prefill-Decode Disaggregation
```bash
# When vLLM runs the NIXL connector, prefill/decode URLs are required.
# See a working example in scripts/llama3.1/ folder.
cargo run --release -- \
    --policy consistent_hash \
    --vllm-pd-disaggregation \
    --prefill http://127.0.0.1:8081 \
    --prefill http://127.0.0.1:8082 \
    --decode http://127.0.0.1:8083 \
    --decode http://127.0.0.1:8084 \
    --decode http://127.0.0.1:8085 \
    --decode http://127.0.0.1:8086 \
    --host 127.0.0.1 \
    --port 8090 \
    --intra-node-data-parallel-size 1 \


# When vLLM runs the NCCL connector, ZMQ based discovery is supported.
# See a working example in scripts/install.sh
cargo run --release -- \
    --policy consistent_hash \
    --vllm-pd-disaggregation \
    --vllm-discovery-address 0.0.0.0:30001 \
    --host 0.0.0.0 \
    --port 10001 \
    --prefill-policy consistent_hash \
    --decode-policy consistent_hash

# When vLLM runs the Mooncake connector, pass --kv-connector mooncake.
# The router queries each prefill node's Mooncake bootstrap server at startup
# to learn engine_id per DP rank, and injects transfer_id / remote_bootstrap_addr /
# remote_engine_id into each request's kv_transfer_params for P/D coordination.
cargo run --release -- \
    --policy consistent_hash \
    --vllm-pd-disaggregation \
    --kv-connector mooncake \
    --prefill http://127.0.0.1:8081 \
    --prefill http://127.0.0.1:8082 \
    --decode http://127.0.0.1:8083 \
    --decode http://127.0.0.1:8084 \
    --host 127.0.0.1 \
    --port 8090 \
    --intra-node-data-parallel-size 1
```

## Configuration

### Authentication

Enable bearer-token validation by listing validation URLs (comma-separated) in `.env` via `API_KEY_VALIDATION_URLS` or passing `--api-key-validation-urls`.
When set, all HTTP endpoints require `Authorization: Bearer <token>` and tokens are validated with HTTP 200 responses.

```bash
# .env
API_KEY_VALIDATION_URLS=https://codebase.helmholtz.cloud/api/v4/user

# CLI override
vllm-router --api-key-validation-urls https://codebase.helmholtz.cloud/api/v4/user
```

### Metrics

Prometheus metrics endpoint available at `127.0.0.1:29000` by default.

```bash
# Custom metrics configuration
vllm-router \
    --worker-urls http://localhost:8080 http://localhost:8081 \
    --prometheus-host 0.0.0.0 \
    --prometheus-port 9000
```

### Retries and Circuit Breakers

#### Retry Configuration
Retries are enabled by default with exponential backoff and jitter:

```bash
vllm-router \
  --worker-urls http://localhost:8080 http://localhost:8081 \
  --retry-max-retries 3 \
  --retry-initial-backoff-ms 100 \
  --retry-max-backoff-ms 10000 \
  --retry-backoff-multiplier 2.0 \
  --retry-jitter-factor 0.1
```

#### Circuit Breaker Configuration
Circuit breakers protect workers and provide automatic recovery:

```bash
vllm-router \
  --worker-urls http://localhost:8080 http://localhost:8081 \
  --cb-failure-threshold 5 \
  --cb-success-threshold 2 \
  --cb-timeout-duration-secs 30 \
  --cb-window-duration-secs 60
```

**Circuit Breaker State Machine:**
- `Closed` → `Open` after N consecutive failures (failure-threshold)
- `Open` → `HalfOpen` after timeout (timeout-duration-secs)
- `HalfOpen` → `Closed` after M consecutive successes (success-threshold)

**Retry Policy:** Retries on HTTP status codes 408/429/500/502/503/504, with backoff/jitter between attempts.

### Request ID Tracking

Track requests across distributed systems with configurable headers:

```bash
# Use custom request ID headers
vllm-router \
    --worker-urls http://localhost:8080 \
    --request-id-headers x-trace-id x-request-id
```

**Default headers:** `x-request-id`, `x-correlation-id`, `x-trace-id`, `request-id`

### Load Balancing Policies

The router supports multiple load balancing policies:

| Policy | Description | Session Affinity | Use Case |
|--------|-------------|------------------|----------|
| `round_robin` | Sequential distribution across workers | No | General purpose, even distribution |
| `random` | Uniform random selection | No | Simple deployments |
| `consistent_hash` | Routes same session/user to same worker | Yes | Multi-turn chat, KV cache reuse |
| `power_of_two` | Picks least loaded of two random workers | No | Load-sensitive workloads |
| `cache_aware` | Optimizes for prefix cache hits | Yes | Repeated prompts, few-shot |

```bash
# Example: Using consistent_hash with HTTP header for session affinity
curl -X POST http://router:8000/v1/chat/completions \
  -H "X-Session-ID: my-session-123" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3", "messages": [{"role": "user", "content": "Hello!"}]}'
```

For detailed configuration options, hash key priorities, and usage examples, see [Load Balancing Documentation](docs/load_balancing/README.md).

## Advanced Features

### Kubernetes Service Discovery

Automatic worker discovery and management in Kubernetes environments.

#### Basic Service Discovery

```bash
vllm-router \
    --service-discovery \
    --selector app=vllm-worker role=inference \
    --service-discovery-namespace default
```

### Command Line Arguments Reference

#### Service Discovery
- `--service-discovery`: Enable Kubernetes service discovery
- `--service-discovery-port`: Port for worker URLs (default: 8000)
- `--service-discovery-namespace`: Kubernetes namespace to watch
- `--selector`: Label selectors for regular mode (format: `key1=value1 key2=value2`)

## Development

### Troubleshooting

**VSCode Rust Analyzer Issues:**
Set `rust-analyzer.linkedProjects` to the absolute path of `Cargo.toml`:

```json
{
  "rust-analyzer.linkedProjects": ["/workspaces/vllm/vllm-router/Cargo.toml"]
}
```

### CI/CD Pipeline

The continuous integration pipeline includes comprehensive testing, benchmarking, and publishing:

#### Build & Test
1. **Build Wheels**: Uses `cibuildwheel` for manylinux x86_64 packages
2. **Build Source Distribution**: Creates source distribution for pip fallback
3. **Rust HTTP Server Benchmarking**: Performance testing of router overhead
4. **Basic Inference Testing**: End-to-end validation through the router
5. **PD Disaggregation Testing**: Benchmark and sanity checks for prefill-decode load balancing

#### Publishing
- **PyPI Publishing**: Wheels and source distributions published when version changes in `pyproject.toml`
- **Container Images**: Docker images published using `/docker/Dockerfile.router`

## Acknowledgement

This project is a fork of [SGLang Model Gateway](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway), and we would like to explicitly acknowledge and thank the original authors for their work. At this stage, our fork includes only minimal changes to preserve the existing interface and ensure compatibility with vLLM. We anticipate further divergence as we pursue the roadmap we have in mind, which is the reason for creating the fork.
