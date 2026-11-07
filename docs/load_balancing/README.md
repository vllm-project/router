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
| `kv_aware` | KV reuse and stage-aware load scoring | No | Yes |

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
| `cache_threshold` | 0.5 | Minimum prefix match ratio over the full input |
| `balance_abs_threshold` | 32 | Absolute load difference threshold for load balancing |
| `balance_rel_threshold` | 1.1 | Relative load ratio threshold for load balancing |
| `eviction_interval_secs` | 30 | Interval for cache eviction |
| `max_tree_size` | 33,554,432 (`2^25`) | Character budget per worker per model |

Trees are trimmed to `max_tree_size` during periodic eviction and may grow beyond it between cycles. This character budget is not a memory limit; actual memory usage also includes tree overhead and scales with the number of workers and models.

### Behavior

1. **Balanced mode** (when load is even):
   - Find worker with highest prefix match for the request
   - If match rate > `cache_threshold`: route to that worker (cache hit)
   - Otherwise: route to worker with lowest load

2. **Imbalanced mode** (when load is skewed):
   - Route to worker with lowest load (shortest queue)
   - Still updates the tree to maintain cache state

### Best For

- Workloads with repeated prompt prefixes (system prompts, few-shot examples)
- When prefix caching is enabled on vLLM workers
- Multi-tenant deployments with distinct prompt patterns

---

## KV Aware

`kv_aware` uses KV cache reuse and worker load to route requests. It suits workloads with repeated prompt prefixes and supports HTTP, [gRPC](../backend/grpc.md), and P/D disaggregation over HTTP or gRPC.

HTTP Completion uses the existing [Direct/Worker input preparation](http_completion_kv.md), without local tokenizer assets. Unmodeled identities and unqualified targets stay routable without cache credit.

```bash
vllm-router --policy kv_aware \
  --worker-urls grpc://worker1:8000 grpc://worker2:8000
```

Enable KV events on the workers. HTTP text requests also need a render endpoint or, for simple prompts, `/tokenize`. Without usable KV information, routing falls back to load.

For regular pools without event discovery, configure `--kv-event-source WORKER_URL=tcp://HOST:PORT` per worker, plus matching `--kv-event-model` and `--kv-event-block-size`.

### P/D disaggregation

```bash
vllm-router --vllm-pd-disaggregation \
  --prefill http://prefill:8000 --decode http://decode:8000 \
  --prefill-policy kv_aware --decode-policy kv_aware
```

Prefill selection considers reusable KV and current work. Decode selection favors the least busy worker. Both stages inherit `--policy` unless overridden.

Configure the engines' P/D connector and enable prefix caching and KV events on prefill workers. For native Python vLLM metadata discovery, set `VLLM_SERVER_DEV_MODE=1`.

If event discovery is unavailable, add `--pd-kv-events http://prefill:8000=tcp://prefill:5557` for each prefill worker.

### Session affinity

Affinity is off by default. Add `--kv-aware-session-affinity-ttl-secs 1800` to retain bindings for 30 minutes of inactivity.

Send the same `x-session-id` header or `session_params.session_id` on each turn; agent metadata and framework session headers take precedence.

In P/D mode, affinity applies only to decode. A busy decoder may be replaced by a less busy one; prefill is selected independently for each request.

Multiple router replicas need ingress stickiness to preserve affinity.

---

## Per-worker admission capacity

Policies choose *which* worker serves a request; none of them refuses work.
Under sustained overload every request is still dispatched, and the excess
waits inside the engines' own queues, where latency grows without bound and
the backlog must drain before the pool recovers. When workers fail, the
survivors receive their share too.

`--max-inflight-requests-per-worker N` (Python: `max_inflight_requests_per_worker`)
is an opt-in hard limit on the requests this router has in flight on one
worker at a time. It is off by default; when unset nothing is counted and
routing is unchanged.

```bash
vllm-router --policy round_robin \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --max-inflight-requests-per-worker 64
```

### Behavior

- **Admission.** Selection skips full workers. A permit is then reserved with
  an atomic compare-and-swap, so concurrent arrivals cannot overshoot `N`. If
  the selected worker filled in the meantime, the attempt is retried with a
  fresh snapshot and holds no permit while it backs off.
- **Rejection.** When every eligible worker is full, the router answers
  immediately with `429 Too Many Requests` and `Retry-After: 1`. The request is
  not dispatched and not retried inside the router, so the router adds no
  hidden waiting. These rejections are not worker failures: they never count
  against circuit breakers.
- **Lifetime.** A permit tracks *worker* occupancy. It is held from dispatch
  until the worker finishes the response (end of stream or upstream error) or
  the client goes away (disconnect, timeout, abandoned retry), whichever comes
  first. Each permit is released exactly once, through the same guard that
  already owns worker load for the request. Bytes the router has already
  buffered for a slow client are not counted; bounding that buffer is a
  separate concern.
- **P/D.** The limit applies per stage endpoint: a prefill endpoint counts
  requests until their prefill finishes, and a decode endpoint counts pending
  plus active decode requests. A full prefill or decode stage gets the same
  429, and `/health_generate` probes bypass the limit so a busy pool is not
  reported unhealthy.
- **Scope.** Inference `POST` routes, including the transparent proxy, are
  covered. Requests that Program scheduling has already bound to a target are
  counted but never refused, because Program scheduling remains their
  admission controller.

### Signals and limitations

- Counts are owned by the router process. They never depend on worker
  telemetry, so a stale or missing metric cannot open the gate or block a
  worker permanently. They are also **local to one router replica** and do
  not see the engine's own queue. With `R` router replicas a worker can
  receive up to `R × N` requests.
- The limit counts requests, not tokens or bytes, and it is a logical bound,
  not a memory bound. It does not bound per-request
  memory such as large prompts; tenant quotas (#419) and token-cost fairness
  (#420) are separate controls.
- A worker removed and re-added under the same URL starts from zero, so it can
  briefly hold up to `2 × N` while requests admitted before the removal drain.
  It is never locked out.
- `--max-concurrent-requests` is a separate, router-wide admission token bucket
  that refills over time; it is not a per-worker limit.

### Metrics

| Metric | Labels | Meaning |
|--------|--------|---------|
| `vllm_router_admission_rejections_total` | `router` (`regular`, `pd`), `stage` (`worker`, `prefill`, `decode`), `reason` (`worker_capacity`) | Requests refused because every eligible worker or stage endpoint was full |
| `vllm_router_worker_admitted_requests` | `worker` | Permits held per regular worker |
| `vllm_router_pd_prefill_occupancy`, `vllm_router_pd_pending_decode_requests`, `vllm_router_pd_active_decode_requests` | `worker`, `rank` | Existing P/D occupancy that the limit is checked against |

Labels are fixed enums or worker URLs; no request content is recorded.

### Choosing N

Start near the engine's effective concurrent batch size (for vLLM,
`--max-num-seqs`), or a little above it so the engine's own queue never runs
dry. Watch `vllm_router_admission_rejections_total` together with latency: a
rising rejection rate at acceptable latency means the pool needs more
capacity, not a higher `N`.

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
- **Decode**: Inherits the main policy; override with `--decode-policy` (for example `round_robin`)

`grpc://` prefill and decode workers must run vLLM ≥ 0.30.0. The router
warns at startup when a worker reports an older `engine_version`, and
`power_of_two` uses the router's local load counts for them. See
[gRPC P/D disaggregation](../backend/grpc.md#pd-disaggregation).
