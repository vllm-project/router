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
| `smetric` | Text-based agent workloads with a TTFT budget | Yes (conditional cache affinity) | Yes |
| `kv_aware` | Tiered KV reuse and live-load scoring | No | Yes |

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

## SMetric

SMetric balances estimated prefill work and reuses a cached prefix
only when the turn/history gate and estimated first-token latency budget allow it.
It supports text chat, including serialized tool metadata, and single-string
completion prompts for cache affinity.

```bash
vllm-router --policy smetric \
  --smetric-config examples/configs/smetric.yaml \
  --worker-urls http://worker1:8000 http://worker2:8000
```

The YAML is required; calibrate its character-based cost coefficients, rate and
TTFT budget for the deployment. Omit `PREFILL_RATE` to learn a per-worker rate
from the first generated output in a stream or successful PD prefill completion.
Stream metadata and error events do not train the rate estimate.
An explicit rate stays fixed. `MAX_TREE_SIZE` defaults to 67,108,864 characters
per worker. The optional `EVICTION_INTERVAL_SECS` defaults to 30; set it to `0`
to disable size eviction. Dropping the policy stops its eviction thread.

### Cache state and token input

HTTP workers, including the HTTP P/D path, continue to use the request-history
Tree. With regular gRPC chat workers, SMetric reuses the shared KV Events
discovery and index, querying the same prepared token IDs sent to the engine.
The engine must expose its KV Events configuration through the gRPC Control
service and publish events. No additional SMetric configuration is needed:

```bash
vllm-router --policy smetric \
  --smetric-config examples/configs/smetric.yaml \
  --worker-urls grpc://worker1:8000 grpc://worker2:8000
```

An unpinned address needs Control metadata confirming a single DP rank.
Explicit `@rank` addresses bind to that rank; an ambiguous target receives
no event-derived credit.

SMetric credits contiguous local GPU blocks in cache group 0, capped at the
complete blocks before the final prompt token that must be recomputed for
logits. Host and external cache tiers receive no credit. The index handles
block removals, cache clears and event sequence gaps; worker removal also
clears SMetric's binding. Token requests do not populate or consult the Tree.
Missing metadata or prefix evidence, and requests with a nonempty `cache_salt`,
receive zero cache credit. LoRA names use the shared index's hash namespace.
Event interpretation and cache-identity semantics follow the shared upstream index.

The cost model keeps its character units. For a prompt of `L` characters and
`N` prepared tokens, `H` reusable tokens become `floor(L * H / N)` equivalent
cached characters. This is a cost approximation based on event-derived token
residency; it does not recover exact token-to-character offsets. Existing cost
coefficients, TTFT budgets and rate parameters keep their meaning. Online rate
learning uses requests with observed event-prefix evidence; zero-match requests
release their pending work without training the rate, since the shared query
does not distinguish a cold miss from unavailable cache state. A configured
`PREFILL_RATE` remains fixed. Tree size and eviction settings apply only to
the request-history path.

Multimodal chat parts and tool results, unrecognized message roles,
batched/token-ID completion prompts and other endpoints fall back to
**least-inflight balancing**, with rotating ties among available
workers. They do not populate the prefix tree or train prefill rates. Native HTTP
forwarding preserves unsupported content parts; the token-ID-only gRPC path
still rejects multimodal input.
Re-encoded PD JSON responses regenerate body-length and framing headers.

Each policy instance warns once for each fixed input category: non-text chat,
batched or token-ID prompts, and unsupported endpoints or missing typed prompts.
Subsequent fallbacks in the same category log at debug level. Both levels include
`route`, input category, `fallback=least_inflight` and `selected_worker`, without
logging the input payload. SMetric request load
is released on completion, error or cancellation; retry attempts own separate
load reservations. This does not change Power-of-Two load accounting.

The [archived reproduction](https://github.com/clateral912/smetric-router-repro)
contains a historical Power-of-Two **zero-load/load-feedback-limited baseline**.
It is not a measurement of a fully load-aware Power-of-Two implementation.
Correcting that baseline and remeasuring it are separate from this SMetric PR;
archived figures are unchanged.

---

## KV Aware

The `kv_aware` policy scores workers using the request's reusable KV blocks and current load. It is supported in regular, co-located routing mode only; it is not supported with PD disaggregation. In PD mode, the effective prefill and decode policies must use other policies. KV placement credit requires gRPC workers because the gRPC chat route supplies token IDs to the policy. With HTTP workers, the policy degrades to load-only scoring and does not use KV event placement.

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
