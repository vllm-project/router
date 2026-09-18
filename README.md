# vLLM Router
<p align="center">
| <a href="docs/load_balancing/README.md"><b>Documentation</b></a> | <a href="https://deepwiki.com/vllm-project/router"><b>DeepWiki</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://vllm-dev.slack.com/archives/C085AUU43NK"><b>Developer Slack</b></a> | <a href="docs/assets/WeChat.png"><b>WeChat</b></a> |
</p>

A high-performance and light-weight request forwarding system for vLLM large scale deployments, providing advanced load balancing methods and prefill/decode disaggregation support.

### Key Features

- **Core Architecture**: Request routing framework and async processing patterns
- **Load Balancing**: Multiple algorithms (cache-aware, power of two, consistent hashing, random, round robin)
- **Prefill-Decode Disaggregation**: Specialized routing for separated processing phases
- **Service Discovery**: Kubernetes-native worker management and health monitoring
- **Enterprise Features**: Circuit breakers, retry logic, metrics collection
- **gRPC workers**: `grpc://` URLs speak vLLM rust `Inference` with router-side `token_ids` (see [gRPC worker backend](#grpc-worker-backend))

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

#### Optional WASM OnRequest middleware

Load an independently built WASM Component plugin (see `examples/wasm_middleware/` and [RFC #236](https://github.com/vllm-project/router/issues/236)). By default it attaches only to `POST /v1/chat/completions` and fails closed on plugin errors:

```bash
./examples/wasm_middleware/build.sh

./target/release/vllm-router \
    --worker-urls http://localhost:8000 \
    --wasm-middleware ./examples/wasm_middleware/wasm_middleware_example.component.wasm \
    --wasm-middleware-route /v1/chat/completions
```

Additional paths can be attached with repeated `--wasm-middleware-route` flags (must be one of the protected inference routes). Without `--wasm-middleware`, the Router does not initialize Wasmtime.

v0.1 resource / fail-closed defaults on attached routes (not configurable via CLI yet):

- **Input body cap**: `min(10 MiB, --max-payload-size)`. Requests larger than this get **413** before the plugin runs, even if the plugin would only `Continue`. This is intentionally tighter than the Router's default 512 MiB payload limit.
- **Execution deadline**: **100 ms** per invocation (Wasmtime epoch interruption). Deadline / trap failures fail closed with **500**.
- **Queue full**: when the bounded worker queue is saturated, matching requests get **503**.

Prometheus metrics for the WASM runtime are deferred to a later revision.

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

### gRPC worker backend

`grpc://` workers speak vLLM’s rust `Inference` API. Bindings come from
crates.io [`vllm-proto`](https://crates.io/crates/vllm-proto) (`0.2`).
HTTP `http://` workers stay a reverse proxy of OpenAI `messages`.
`grpc://` workers receive **`token_ids`** produced on the router
(`vllm-chat` + `vllm-tokenizer`, Cargo git tag `v0.29.0`, not
`pip install`). There is no in-repo tokenizer fallback.
`vllm-tokenizer` pulls `fastokens`, which depends on PCRE2; this repo
sets `PCRE2_SYS_STATIC=1` in `.cargo/config.toml` so plain `cargo build`
does not depend on a system `libpcre2-8` installation.

This is the **current default wire**, not a forever ban on a gRPC text
prompt or HTTP token-id input. Chat-only on `grpc://` is a **router
501** (`/v1/chat/completions` only); other OpenAI routes still work on
`http://`.

A worker pool is **all-`http(s)://` or all-`grpc(s)://`**. Mixed
schemes fail at init / `add_worker` (gRPC does not accept text).
All-HTTP keeps `policy.select` then reverse-proxy (no chat/tokenizer
frontend on the critical path). All-gRPC is
`Frontend.prepare(chat) → token_ids`, then `policy.select`, then
`Frontend.dispatch(ids, url)` (convert + GenerateStream + detok).
Policy still uses `extract_text_for_routing()` this version.

`TokenizerCache` (in `src/backend/preprocess.rs`) caches loaded
`vllm-chat` / `vllm-tokenizer` objects per model key in-process. It
does not reuse prior-request token ids or engine KV. Cold loads are
single-flight, and one model no longer pins the cache for later model keys.

The request lowering in `vllm_frontend.rs` intentionally tracks vLLM
0.29's private `prepare_chat_request` conversion. That upstream function
is `pub(super)` and coupled to `vllm-server`, so it cannot be imported by
this crate. Replace the local adapter if vLLM exposes a public
frontend-only lowering API.

Omitted sampling temperature is resolved from the loaded model's
`generation_config.json`, then falls back to OpenAI's `1.0`; protobuf
omission is not used because vLLM 0.29 gRPC rewrites it to greedy `0.0`.
When hidden stop strings/tokens are configured, the router requests worker
text so the exact character trim boundary is preserved; token IDs alone
cannot encode that boundary.

Reasoning/tool history, tool definitions, tool choice, template kwargs,
documents, and response format are preserved while rendering. Active
tool calling and `reasoning_effort` currently return a clear 400 because
the gRPC response adapter does not yet expose `vllm-chat`'s output
parsers; returning raw text as `content` would silently violate OpenAI
tool/reasoning response semantics. Multimodal content is likewise rejected
until media features are sent in `GenerateRequest.media`.

#### Backend modules (`src/backend/`)

Northbound is always HTTP `/v1/chat/completions` in
`routers/http/router.rs`. Detect runs at init (and `add_worker`) to
lock the pool kind and strip `@dp_rank` / tonic URI. Pipeline on an
all-`grpc://` pool:
`frontend.prepare → policy → frontend.dispatch` (dispatch =
`convert + grpc + openai`; all router-local; the worker is reached
inside `grpc.rs`). Startup and periodic probes use `health.rs`
(`grpc.health.v1`), not that generate path.

| File | Role |
|---|---|
| `mod.rs` | `stages_enabled()`, re-exports, `pb` = crates.io `vllm-proto` |
| `detect.rs` | `grpc://` / `grpcs://`, `WorkerPoolKind`, reject mixed schemes, strip `@dp_rank`, tonic URI (`grpc://host:port` → h2c `http://host:port`; still gRPC) |
| `frontend.rs` | `EngineFrontend`: `prepare(chat)` then `dispatch(ids, url)`. Wraps preprocess + convert + grpc |
| `health.rs` | `grpc.health.v1` Check on `--grpc-port` (empty service = overall `SERVING`). Used for worker + startup probes |
| `preprocess.rs` | `TokenizerCache` + chat template/encode → `token_ids`. Load key: valid local `VLLM_ROUTER_TOKENIZER`, else `VLLM_ROUTER_MODEL`, else request `model`. Required on `grpc://`; HTTP remains a transparent reverse proxy |
| `vllm_frontend.rs` | Private adapter onto `vllm-chat` / `vllm-tokenizer` / `vllm-text` (`load_model_backends`, render, encode, detok) |
| `convert.rs` | OpenAI fields → `GenerateRequest` with `prompt = TokenIds` (does not fill proto `media` / KV-transfer fields) |
| `grpc.rs` | Cached tonic `InferenceClient`, `GenerateStream` |
| `openai.rs` | Proto chunks → OpenAI JSON/SSE for the **client** (not a southbound hop) |

The diagnostic `engine_ms` stage is a first-output residual, not direct
EngineCore telemetry: it includes queueing/prefill/generation effects left
after subtracting router frontend and transfer spans. Compare it only when
the first-output boundary is equivalent across arms.

Set `VLLM_ROUTER_TOKENIZER` to a local tokenizer/model directory (or a
tokenizer file inside it, which is normalized to its parent directory)
when the northbound served model name is an alias. Otherwise
`VLLM_ROUTER_MODEL`, then request `model`, supplies the load key. The
request model remains the southbound protobuf model name; a tokenizer
filesystem path no longer overwrites that alias.

#### How to start `vllm-rs` (v0.29)

As of **vLLM 0.29**, `vllm-rs` is a binary **inside the Python wheel**, next
to the package (`…/site-packages/vllm/vllm-rs`). This router does not
vendor or Cargo-pull that binary.

Two different entrypoints, easy to mix up:

| How you start the worker | What actually runs | `--grpc-port` |
|---|---|---|
| `VLLM_USE_RUST_FRONTEND=1 vllm serve …` | Python `vllm serve` execs **`vllm-rs frontend`** (inherited listen fd, OpenAI HTTP) | **cannot** be set this way |
| `vllm-rs serve MODEL --grpc-port …` | The wheel binary in **`serve`** mode (HTTP `--port` **and** Inference gRPC) | **this** is the `grpc://` worker |

`VLLM_USE_RUST_FRONTEND` only switches the Python launcher onto rust HTTP.
It is ignored if you invoke `vllm-rs` yourself. There is no
`VLLM_RUST_FRONTEND` flag.

Locate the wheel binary (`./scripts/backend/check_vllm_rs.sh` prints
path, `--version`, and a copy-paste `export VLLM_RS=…`):

```bash
./scripts/backend/check_vllm_rs.sh
# then: export VLLM_RS=...
"$VLLM_RS" serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --grpc-port 50051 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --enable-prefix-caching
vllm-router \
  --host 127.0.0.1 \
  --port 30000 \
  --worker-urls grpc://127.0.0.1:50051
```

`--max-num-seqs`, `--max-model-len`, and prefix-cache flags are **worker**
options. `bench_prefix_kvhit.py` only tunes body length (`--chars` / `--tokens`)
and hit rate (`--hit-rate`). See [`scripts/backend`](scripts/backend).

`grpc://host:port` is rewritten to h2c `http://host:port` for tonic; that
is still gRPC, not the OpenAI `--port`. Worker liveness on that port is
`grpc.health.v1` Check (empty service = overall `SERVING`), not TCP
connect and not HTTP `GET /health` (that stays on `--port` for `http://`
workers). `--health-check-endpoint` is ignored for `grpc://`.

#### If the wheel has no `vllm-rs`

Some installs omit the rust artifact. Build it in a **separate** vLLM
checkout (not inside this router tree). Do **not** add `vllm-server` /
`vllm-rs` as a router Cargo dep.

The rust binary and the Python EngineCore it spawns must speak the same
**ZMQ/msgpack** handshake. Same release line (e.g. both 0.29.x) is the
safe default; an exact git tag is **not** required if the protocol has
not moved. Mixing a new `vllm-rs` with an old `vllm` package is what
breaks.

`build_rust.sh` → `tools/build_rust.py` (setuptools-rust) and a raw
`cargo build -p vllm-cmd --release` compile the **same** `vllm-cmd`
crate (`[[bin]] name = "vllm-rs"`). They are not two frontends. The
script additionally: vendors TLS (`native-tls-vendored`), builds
`vllm._rust_tool_parser`, copies the binary to `vllm/vllm-rs` in that
checkout, and ([#52593](https://github.com/vllm-project/vllm/pull/52593))
embeds the setuptools-scm version via `VLLM_RS_BUILD_VERSION`. Plain
`cargo` writes `target/release/vllm-rs` and reports crate `0.1.0` unless
you set `VLLM_RS_BUILD_VERSION` yourself.

Then `VLLM_RS=/path/to/vllm-rs` for `serve_rust_grpc.sh`. Prefer a 0.29
wheel that already ships the binary.

Worker + router launch examples (python HTTP, rust HTTP, rust gRPC)
and a prefix miss/hit client live in
[`scripts/backend`](scripts/backend).

#### Tests

These lock **this version’s wire**, not a policy that gRPC may never
grow a text prompt. No GPU and no live `vllm-rs` except `bench_prefix_kvhit.py`.

| What | Purpose | How |
|---|---|---|
| `tests/grpc_vs_http_e2e.rs` | HTTP body has `messages` (no `token_ids`); gRPC proto has `token_ids` (no text prompt). Same OpenAI JSON in | `cargo test --test grpc_vs_http_e2e` |
| `tests/common/mock_vllm_rs.rs` | In-process `Inference` + `grpc.health.v1` (SERVING) used by that e2e test | pulled in automatically |
| `src/backend/detect.rs` / `convert.rs` / `openai.rs` / `health.rs` | Scheme/URI, `TokenIds`-only proto, SSE shape, health helper | `cargo test --lib backend::` |
| `src/backend/preprocess.rs` + `tests/python_vllm_chat_ids.py` | rust `vllm-chat` ids vs Python `vllm.tokenizers` on the same messages. **Skips** unless `VLLM_ROUTER_MODEL` is a model dir with `tokenizer.json` and Python can `import vllm` | see below |
| `scripts/backend/bench_prefix_kvhit.py` | Live miss-then-hit against a running router | after `serve_*.sh` (see [`scripts/backend`](scripts/backend)) |

Always-on (CI):

```bash
cargo test --test grpc_vs_http_e2e
cargo test --lib backend::
```

Optional rust-vs-Python id check (skips if env/import missing):

```bash
export VLLM_ROUTER_MODEL=/path/to/hf-model   # dir must contain tokenizer.json
# optional: PYTHON=...  or activate VIRTUAL_ENV
cargo test --lib vllm_chat_tokenize_matches_python_vllm -- --nocapture
```

Hand-run the helper the preprocess tests spawn:

```bash
python tests/python_vllm_chat_ids.py /path/to/hf-model \
  '[{"role":"user","content":"hello"}]'
```

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
