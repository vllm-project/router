# Load Balancing Policies

The vLLM Router supports multiple load balancing policies for distributing requests across backend workers. Each policy is designed for different use cases and can be configured based on your deployment requirements.

Worker URLs may be `http(s)://` or `grpc://`. Examples below use HTTP;
the policy is the same for `grpc://host:port`.

## Available Policies

| Policy | Best For | Session Affinity | Load Aware |
|--------|----------|------------------|------------|
| `round_robin` | General purpose, even distribution | No | No |
| `random` | Simple deployments | No | No |
| `consistent_hash` | Multi-turn conversations, KV cache reuse | Yes | No |
| `power_of_two` | Load-sensitive workloads | No | Yes |
| `cache_aware` | Prefix caching optimization | Yes (cache-based) | Yes |
| `smetric` (alias of KV-aware SMetric selection) | Text-based agent workloads with a TTFT budget | Yes (conditional cache affinity) | Yes |
| `kv_aware` | Tiered KV reuse and live-load scoring; optional SMetric TTFT gate | Conditional in SMetric mode | Yes |

---

## Consistent Hash

The `consistent_hash` policy routes requests with the same session/user identifier to the same backend worker. This is essential for:

- **Multi-turn conversations**: Ensures conversation history stays on the same worker
- **KV cache reuse**: Maximizes cache hits by routing related requests together
- **Session affinity**: Maintains user-specific state on a single worker

### Configuration

```bash
# Using CLI
vllm-router --policy consistent_hash --worker-urls http://worker1:8000,http://worker2:8000

# Using Python
from vllm_router import Router
router = Router(
    policy="consistent_hash",
    worker_urls=["http://worker1:8000", "http://worker2:8000"]
)
```

### Hash Key Priority

The consistent hash policy extracts a routing key in the following priority order:

| Priority | Source | Header/Field | Example |
|----------|--------|--------------|---------|
| 1 | HTTP Header | `X-Session-ID` | `X-Session-ID: session-abc-123` |
| 2 | HTTP Header | `X-User-ID` | `X-User-ID: user-456` |
| 3 | HTTP Header | `X-Tenant-ID` | `X-Tenant-ID: tenant-xyz` |
| 4 | HTTP Header | `X-Request-ID` | `X-Request-ID: req-789` |
| 5 | HTTP Header | `X-Correlation-ID` | `X-Correlation-ID: corr-001` |
| 6 | HTTP Header | `X-Trace-ID` | `X-Trace-ID: trace-002` |
| 7 | Request Body | `session_params.session_id` | `{"session_params": {"session_id": "..."}}` |
| 8 | Request Body | `user` | `{"user": "..."}` (OpenAI format) |
| 9 | Request Body | `session_id` | `{"session_id": "..."}` (legacy) |
| 10 | Request Body | `user_id` | `{"user_id": "..."}` (legacy) |
| 11 | Fallback | Request body hash | Hash of entire request body |

### Usage Examples

#### Recommended: Using HTTP Headers

HTTP headers are the **recommended approach** for session affinity because:
- No JSON body parsing required (faster routing)
- Works with any request format
- Compatible with standard infrastructure tools (Nginx, Envoy, K8s Ingress)

```bash
# Using X-Session-ID header (recommended)
curl -X POST http://router:8000/v1/chat/completions \
  -H "X-Session-ID: conversation-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Using X-User-ID header for user-based routing
curl -X POST http://router:8000/v1/chat/completions \
  -H "X-User-ID: user-67890" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

#### Alternative: Using Request Body

For backward compatibility, you can also include session information in the request body:

```bash
# Using session_params.session_id in body
curl -X POST http://router:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3",
    "session_params": {"session_id": "conversation-12345"},
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Using OpenAI user field
curl -X POST http://router:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3",
    "user": "user-67890",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Behavior

- **Consistency**: Same session ID always routes to the same worker
- **Unhealthy fallback**: If the target worker is unhealthy, falls back to the first healthy worker
- **Virtual nodes**: Uses 160 virtual nodes per worker for even distribution
- **DP-aware routing**: Supports data-parallel worker URLs (e.g., `http://worker:8000@0`)

---

## Round Robin

The `round_robin` policy distributes requests evenly across all healthy workers in sequential order.

### Configuration

```bash
vllm-router --policy round_robin --worker-urls http://worker1:8000,http://worker2:8000
```

### Behavior

- Cycles through workers: worker1 → worker2 → worker3 → worker1 → ...
- Skips unhealthy workers automatically
- Simple and predictable distribution
- No session affinity (each request may go to a different worker)

### Best For

- Stateless workloads
- Single-turn requests
- When even distribution is more important than cache locality

---

## Random

The `random` policy selects a random healthy worker for each request.

### Configuration

```bash
vllm-router --policy random --worker-urls http://worker1:8000,http://worker2:8000
```

### Behavior

- Uniform random selection among healthy workers
- No session affinity
- Statistically even distribution over many requests

### Best For

- Simple deployments
- When you want to avoid any sequential patterns
- Testing and development

---

## Power of Two Choices

The `power_of_two` policy randomly selects two workers and routes to the one with lower load. This provides good load distribution with minimal coordination overhead.

### Configuration

```bash
vllm-router --policy power_of_two --worker-urls http://worker1:8000,http://worker2:8000,http://worker3:8000
```

### Behavior

1. Randomly pick two healthy workers
2. Query their current load (pending requests)
3. Route to the worker with lower load

### Best For

- Load-sensitive workloads
- When request processing times vary significantly
- Avoiding hot spots without full load tracking overhead

### Note

Requires at least 2 workers. With only 1 worker, behaves like direct routing.

---

## Cache Aware

The `cache_aware` policy optimizes for prefix caching by maintaining an approximate radix tree of request prefixes per worker.

### Configuration

```bash
vllm-router --policy cache_aware \
  --cache-threshold 0.5 \
  --balance-abs-threshold 32 \
  --balance-rel-threshold 1.1 \
  --worker-urls http://worker1:8000,http://worker2:8000
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cache_threshold` | 0.5 | Minimum prefix match ratio to use cache-based routing |
| `balance_abs_threshold` | 32 | Absolute load difference threshold for load balancing |
| `balance_rel_threshold` | 1.1 | Relative load ratio threshold for load balancing |
| `eviction_interval_secs` | 30 | Interval for cache eviction |
| `max_tree_size` | 10000 | Maximum nodes per radix tree |

### Behavior

1. **Balanced mode** (when load is even):
   - Find worker with highest prefix match for the request
   - If match rate > `cache_threshold`: route to that worker (cache hit)
   - Otherwise: route to worker with smallest tree (most cache capacity)

2. **Imbalanced mode** (when load is skewed):
   - Route to worker with lowest load (shortest queue)
   - Still updates the tree to maintain cache state

Transparent proxy requests now contribute to worker load and `running_requests`
for both `cache_aware` and SMetric, including while a streamed response is open.
This can affect `cache_aware` balancing for deployments using those endpoints.

### Best For

- Workloads with repeated prompt prefixes (system prompts, few-shot examples)
- When prefix caching is enabled on vLLM workers
- Multi-tenant deployments with distinct prompt patterns

---

## SMetric selection

SMetric is an optional selection mode of `KvAwarePolicy`. It balances estimated prefill work and reuses a cached prefix
only when the turn/history gate and estimated first-token latency budget allow it.
It supports text chat, including serialized tool metadata, and single-string
completion prompts for cache affinity.

These examples use the Rust CLI built from this repository. The Python launcher
installed by `pip` does not expose SMetric configuration.

```bash
cargo build --release --bin vllm-router
./target/release/vllm-router --policy kv_aware \
  --smetric-config examples/configs/smetric.yaml \
  --worker-urls http://worker1:8000 http://worker2:8000
```

`--smetric-config` selects SMetric within `kv_aware`; without it, the existing
multiplicative scorer and picker remain the default. `--policy smetric` remains
a compatibility alias for the same implementation. YAML router configuration
can select this mode with `policy: {type: kv_aware, smetric: {...}}`.

Worker binding, event queries and candidate collection are shared by both modes.
SMetric tracks queued prefill work in linear-token equivalents (`n + n * (L - n / 2) / L_eq`,
where `n = max(L - hit, 0)` is uncached prompt tokens and `ATTENTION_CROSSOVER_TOKENS` defaults
to `16384`, or `0` for a pure linear model) and learns a per-worker rolling P90 prefill rate
online from streaming/gRPC first-token or P/D prefill latencies (pure non-streaming HTTP
deployments should set an explicit `PREFILL_RATE`). Its `RequestTracker` uses the same router
request-lifecycle guard without adding a second slot reservation, and HTTP text-tree updates
are deferred until first generated output (or non-streaming 2xx completion) with per-scope
prefix isolation inside one shared Tree per model. Temperature and non-device transfer-credit
options apply to the default mode; SMetric uses its deterministic affinity/budget rule (50%
reusable-history gate) and local GPU prefix credit.

The YAML selects SMetric and sets your token-scale TTFT budget (`TTFT_SLO_SECS` and
`TTFT_SLO_SECS_PER_1K_TOKENS`). `MAX_TREE_SIZE` defaults to 67,108,864 characters per worker
(shared across all scopes on that model) and `EVICTION_INTERVAL_SECS` defaults to `30` (`0`
disables periodic eviction).

### Cache state and token input

HTTP workers (including HTTP P/D) use the request-history Tree. Regular gRPC workers reuse
the shared KV Events discovery and index on the prepared token IDs, selected per worker so
unbound peers fall back to their text tree without losing affinity on bound peers:

```bash
./target/release/vllm-router --policy kv_aware \
  --smetric-config examples/configs/smetric.yaml \
  --worker-urls grpc://worker1:8000 grpc://worker2:8000
```

An unpinned address needs Control metadata confirming a single DP rank; explicit `@rank`
addresses bind to that rank. SMetric credits contiguous local GPU blocks in cache group 0
(capped before the final prompt token), ignoring non-device tiers and nonempty `cache_salt`
requests. Bound live sources train the online P90 prefill rate on unsalted non-empty requests
including confirmed 0-block cold misses, while unbound or suspended sources skip rate training.

Single token-ID `/v1/completions` prompts route using tokenized KV affinity when bound KV Events
are available. Multimodal chat parts, unrecognized message roles, batched completion prompts,
and other endpoints fall back to **least-inflight balancing** (rotating ties) on HTTP without
populating the prefix tree or training rates (the token-ID-only gRPC chat frontend rejects
multimodal parts and unrecognized roles), logging one warning per bounded input category per
policy. SMetric request load is released on completion, error or cancellation; retry attempts
own separate load reservations.

The [archived reproduction](https://github.com/clateral912/smetric-router-repro)
contains a historical Power-of-Two **zero-load/load-feedback-limited baseline**.
It is not a measurement of a fully load-aware Power-of-Two implementation.
Correcting that baseline and remeasuring it are separate from this SMetric PR;
archived figures are unchanged.

---

## KV Aware

The `kv_aware` policy scores workers using the request's reusable KV blocks and current load. The default multiplicative selection is supported in regular, co-located routing mode only. In PD mode, prefill may use the SMetric selection described above; decode must use another supported policy. KV placement credit requires gRPC workers because the gRPC chat route supplies token IDs to the policy. With HTTP workers, the default selection uses load-only scoring, while SMetric selection uses the request-history Tree.

```bash
vllm-router --policy kv_aware \
  --router-temperature 0.2 \
  --kv-penalty-host 0.8 \
  --kv-penalty-external 0.6 \
  --kv-penalty-disk 0.3 \
  --worker-urls grpc://worker1:8000 grpc://worker2:8000
```

For each candidate, the scorer computes `ptoken = new_prefill_tokens + active_prefill_tokens` and projects the request count as `max(slot_manager.active_requests, Worker::load) + 1`. It multiplies `ptoken` by that projected count, so the score estimates the load after admitting this request and includes its remaining prefill work even when the worker is currently idle. Zero-cost cases, such as a request with no prefill work, are resolved by the KV-work and current-load tie-breaks.

`--router-temperature 0` selects a minimum-cost worker. A positive temperature samples among candidates by their relative costs. Device reuse has full credit; the host, external, and disk `--kv-penalty-*` flags configure transfer-credit ratios for those tiers. A higher credit means more reuse value and lower transfer cost.

---

## Choosing a Policy

```
                                    ┌─────────────────────┐
                                    │  Need session       │
                                    │  affinity?          │
                                    └─────────┬───────────┘
                                              │
                              ┌───────────────┴───────────────┐
                              │                               │
                             Yes                              No
                              │                               │
                              ▼                               ▼
                    ┌─────────────────┐             ┌─────────────────┐
                    │  Multi-turn or  │             │  Load-sensitive │
                    │  KV cache reuse?│             │  workload?      │
                    └────────┬────────┘             └────────┬────────┘
                             │                               │
                 ┌───────────┴───────────┐       ┌───────────┴───────────┐
                 │                       │       │                       │
                Yes                      No     Yes                      No
                 │                       │       │                       │
                 ▼                       ▼       ▼                       ▼
        ┌────────────────┐     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
        │ consistent_hash│     │  cache_aware   │ │  power_of_two  │ │  round_robin   │
        │                │     │                │ │                │ │  or random     │
        └────────────────┘     └────────────────┘ └────────────────┘ └────────────────┘
```

### Quick Reference

| Scenario | Recommended Policy |
|----------|-------------------|
| Chat applications with conversation history | `consistent_hash` |
| Batch inference with no state | `round_robin` |
| Variable request complexity | `power_of_two` |
| Repeated system prompts / few-shot | `cache_aware` |
| Simple testing / development | `random` |

---

## PD (Prefill-Decode) Mode

In prefill-decode disaggregated mode, you can configure separate policies for prefill and decode workers:

```bash
vllm-router \
  --mode pd \
  --prefill-policy consistent_hash \
  --decode-policy round_robin \
  --prefill-workers http://prefill1:8000,http://prefill2:8000 \
  --decode-workers http://decode1:8000,http://decode2:8000
```

This allows optimizing each stage independently:
- **Prefill**: Use `consistent_hash` for cache locality
- **Decode**: Use `round_robin` or `power_of_two` for load distribution
