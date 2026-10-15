# Worker health checks and inference probes

The router runs two independent, optional mechanisms against workers. They look
similar and answer different questions, so they are documented together but
kept strictly separate:

| | Worker health check | Inference probe |
|---|---|---|
| What it asks | is the process serving? | can it actually generate a token? |
| Signal | `GET {--health-check-endpoint}` | `POST /v1/completions`, `max_tokens: 1` |
| Effect | **decides routing**: healthy workers get traffic | **reports only** |
| Flag | `--health-check-endpoint` (+ thresholds) | `--inference-probe` |
| Default | enabled | disabled |

The distinction is deliberate. Worker health drives node selection and is
therefore conservative, threshold-based, and operator-tuned. The inference probe
is a diagnostic: it produces metrics and log lines, and nothing else.

## Worker health check (drives routing)

The periodic checker probes every registered worker — including workers added
later by Kubernetes service discovery. A worker that accumulates
`--health-failure-threshold` consecutive failures is marked unhealthy and
excluded from routing until it passes `--health-success-threshold` consecutive
probes.

```bash
vllm-router \
    --worker-urls http://127.0.0.1:8000 http://127.0.0.1:8001 \
    --health-check-endpoint /health \
    --health-check-interval-secs 30 \
    --health-failure-threshold 3 \
    --health-success-threshold 2
```

A `GET /health` probe validates that the worker process is alive. It says
nothing about whether the worker can serve a request, which is the gap
[issue #158](https://github.com/vllm-project/router/issues/158) reports: the
router marks workers healthy while chat completions fail.

Probe requests never touch worker load counters, so the periodic checker does
not perturb load-based routing (see PR #216), and they are excluded from
OpenTelemetry server spans.

## Inference probe (reports only)

The inference probe asks the question `/health` cannot: it sends a real
one-token generation to every registered worker on each sweep.

```bash
vllm-router \
    --worker-urls http://127.0.0.1:8000 http://127.0.0.1:8001 \
    --inference-probe \
    --inference-probe-interval-secs 60 \
    --inference-probe-timeout-secs 30 \
    --inference-probe-concurrency 32
```

```bash
# Python launcher
vllm-router --inference-probe ...
python -m vllm_router.launch_router --inference-probe ...
```

The request is `POST {worker}/v1/completions` with
`{"prompt": "hi", "max_tokens": 1}` — the same endpoint the router forwards user
traffic to. A worker is reported as serving only when it **completes the
generation and returns a well-formed completion choice**. A 2xx status alone is
not enough: an engine that answers without producing a token is reported as
failing.

### What it does not do

The probe is a diagnostic, and this is enforced by construction:

- it never marks a worker unhealthy or healthy;
- it never feeds the circuit breaker;
- it never changes routing, pair selection, or drain state;
- it never touches worker load counters.

Worker health remains exclusively the business of the health check above. A
defect in the probe can therefore produce a wrong *report*, never an outage,
which is what makes it safe to enable on a running cluster. Acting on a report
— draining a node, filing a ticket — is a decision for an operator or an
external controller consuming the metrics.

### Authentication

When `--api-key` (the worker authorization key) is set, probes carry it as
`Authorization: Bearer <key>`, the same credential the proxy sends. Without it,
workers behind an API key would reject every probe and be reported as failing
for the wrong reason.

### Coverage

The sweep iterates over **all registered workers**, so every node of the P/D
topology is probed, including nodes that never receive traffic. This is
deliberately different from the router-level `GET /health_generate` endpoint,
which samples a single P/D pair per call.

### Cost and tuning

Each sweep costs one real generation per worker per interval. Size the interval
and timeout for inference latency, not for a socket round trip: the default
30s timeout reflects that, and a timeout tuned for a liveness probe will report
false failures on a busy or still-loading worker.

A sweep probes at most `--inference-probe-concurrency` workers at a time, so one
sweep lasts about `ceil(workers / concurrency) * timeout` in the worst case,
when every probe has to time out. Raise the concurrency on large fleets if a
sweep can outlast the interval: workers are probed in fixed batches, so a single
stalled batch delays the batches behind it and their gauges go stale. Being
bounded also keeps a sweep from firing thousands of simultaneous generations at
a cluster that is already struggling.

### What it cannot see

The probe is **worker-local**. It validates that a node can run inference and
return a token, and it does **not** validate prefill→decode coordination — KV
transfer or the bootstrap socket.

That boundary is deliberate. The P-side `kv_transfer_params` envelope is
connector-specific and is built from the *decode* peer's coordinates
(`VllmPDRouter::build_prefill_kv_transfer_params` takes `decode_base` and
`decode_dp_rank`; NIXL pull needs `remote_block_ids`/`remote_engine_id`/
`remote_host`/`remote_port`, Mooncake needs `transfer_id`, MoRIIO WRITE needs
the remote TP/DP sizes). A worker-level probe has no decode peer to target.
Verifying a full paired P/D path is separate work.

### Metrics

| Metric | Meaning |
|---|---|
| `vllm_router_worker_inference_probe_total{worker,result}` | probe outcomes; `result` is `ok`, `request_failed`, `http_error`, `unreadable_body`, or `no_choice` |
| `vllm_router_worker_inference_ok{worker}` | last sweep's result (`1` served, `0` failed) |
| `vllm_router_worker_inference_probe_latency_ms{worker}` | latency of successful probes |

`vllm_router_worker_inference_ok` is the metric to alert on: it is `0` exactly
for the workers that are up, that the health checker considers healthy, and that
cannot serve. Failures also log a warning naming the worker, the classifier, the
error detail, and whether the health checker currently considers the worker
healthy — which makes "healthy but cannot infer" immediately greppable.

### Other modes

- gRPC workers (`grpc://`) use the gRPC health service; the inference probe is
  an HTTP-worker feature.
- `--health-check-endpoint` is unrelated to the inference probe and does not
  affect it.
