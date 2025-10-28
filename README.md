# VLLM Router

A high-performance request routing system for vLLM deployments, providing advanced load balancing and specialized routing for modern LLM serving architectures.

## Release Roadmap

- **September 2025**: Internal testing with select vLLM developers. Core features implemented and tested:
  - Data Parallelism-aware Routing
  - Consistent Hash Load Balancing
  - Prefill/Decode Disaggregation Aware Routing
- **October 2025**: Development and testing of remaining functionalities ([progress tracking](https://docs.google.com/document/d/1d1gi5ex7yCpfMtmCQtwVcsmtOjDIevIOp8HLiA_WlPk/edit?tab=t.0))
- **November 2025**: Evaluation for potential integration into vLLM core repository

### Key Features

- **Core Architecture**: Request routing framework and async processing patterns
- **API Compatibility**: Seamless migration path between SGLang and vLLM ecosystems
- **Load Balancing**: Multiple algorithms (cache-aware, power of two, consistent hashing, random, round robin)
- **Prefill-Decode Disaggregation**: Specialized routing for separated processing phases
- **Service Discovery**: Kubernetes-native worker management and health monitoring
- **Enterprise Features**: Circuit breakers, retry logic, metrics collection, and tool parsing

This router adapted the [SGLang router](https://github.com/sgl-project/sglang/tree/main/sgl-router) API design and implementation for those sharing functionalities. Both SGLang and vLLM projects use the Apache-2.0 license, enabling this collaborative adaptation. This router maintains API compatibility with SGLang router and minimizes code changes to facilitate easy ecosystem transitions and foster potential future unification.

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
# Launch router with data parallelism awareness
./target/release/vllm-router \
    --worker-urls http://0.0.0.0:8000 \
    --dp-aware --policy consistent_hash \
    --data-parallel-size 1

# Alternative: using cargo run
cargo run --release -- \
    --worker-urls http://0.0.0.0:8000 \
    --dp-aware --policy consistent_hash \
    --data-parallel-size 1

# Alternative: using python launcher
vllm-router \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --policy cache_aware
```

#### Prefill-Decode Disaggregation with ZMQ Service Discovery
```bash
cargo run --release -- \
    --policy consistent_hash \
    --vllm-pd-disaggregation \
    --vllm-discovery-address 0.0.0.0:30001 \
    --host 0.0.0.0 \
    --port 10001 \
    --prefill-policy consistent_hash \
    --decode-policy consistent_hash
```

## Configuration

### Logging

Enable structured logging with optional file output:

```python
from vllm_router import Router

# Console logging (default)
router = Router(worker_urls=["http://worker1:8000", "http://worker2:8000"])

# File logging enabled
router = Router(
    worker_urls=["http://worker1:8000", "http://worker2:8000"],
    log_dir="./logs"  # Daily log files created here
)
```

Set log level with `--log-level` flag ([documentation](https://docs.vllm.ai/backend/server_arguments.html#logging)).

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
- Any failure in `HalfOpen` reopens immediately

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

#### RBAC Configuration

**Namespace-scoped (recommended):**
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: vllm-router
  namespace: vllm-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: vllm-system
  name: vllm-router
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: vllm-router
  namespace: vllm-system
subjects:
- kind: ServiceAccount
  name: vllm-router
  namespace: vllm-system
roleRef:
  kind: Role
  name: vllm-router
  apiGroup: rbac.authorization.k8s.io
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

## Key Features

### Performance & Scalability
- **High Performance**: Rust-based routing with connection pooling and optimized request handling
- **Scalability**: Handles thousands of concurrent connections with efficient resource utilization

### Advanced Load Balancing
- **Cache-Aware**: Intelligent routing based on cache locality for optimal performance
- **Power of Two**: Chooses the less loaded of two randomly selected workers
- **Random**: Distributes requests randomly across available workers
- **Round Robin**: Sequential distribution across workers in rotation
- **Consistent Hash**: Session-aware routing for stateful workloads

### Specialized Routing
- **Prefill-Decode Disaggregation**: Specialized load balancing for separated prefill and decode servers
- **Data Parallelism Awareness**: Optimized routing for distributed training and inference

### Enterprise Grade
- **Service Discovery**: Automatic Kubernetes worker discovery and health management
- **Monitoring**: Comprehensive Prometheus metrics and structured logging
- **Circuit Breakers**: Automatic fault tolerance and recovery
- **Retry Logic**: Intelligent request retry with exponential backoff
