# gRPC frontend L0 benchmark

Uses a local BPE fixture and mock gRPC worker to compare L0 off/on through
`EngineFrontend::prepare` and `dispatch`. No model download or GPU is needed.

```bash
cargo test --release --test grpc_frontend_bench -- --ignored --nocapture
```

Settings use the `VLLM_ROUTER_BENCH_` prefix, as in #290:

| Suffix | Default | Meaning |
|---|---|---|
| `SIZES` | `200,16384,131072` | User-message bytes; minimum 32 |
| `CONCURRENCY` | `1,8` | Concurrent clients |
| `MEASURE_SECS` | `2` | Measurement duration per case |

Each case starts a fresh frontend and warms up 64 prompts before timing.
`hot64` reuses those prompts; `cold` uses unique prompts. The harness checks
responses, hits and misses. It uses eight Tokio threads and the default L0
limits: 10,000 entries, 64 MiB total and 1 MiB per entry.

Each JSON result reports encode and request P50/P99, throughput, process CPU,
cache bytes, entries, hits and misses. Request latency covers prepare through
response-body consumption, excluding request construction. CPU includes
request construction and the mock worker. Occupancy includes warmup entries;
hit/miss counts exclude warmup.

Measurements exclude the HTTP listener, routing policies and model execution.
Increase `MEASURE_SECS` and repeat on an idle host for comparisons. Results
with this small fixture may differ from production tokenizers.
