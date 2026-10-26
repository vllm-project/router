# E2E performance tests

GPU-backed performance scenarios live in subdirectories (each with its own
`conftest.py`, cases, and docs).

| Directory | Description |
|-----------|-------------|
| [`grpc/`](grpc/README.md) | Prefix-KV hit/miss via router + vllm-rs / vllm (rust gRPC, rust HTTP, python HTTP) |

Run a suite:

```bash
export MODEL=/path/to/model
python -m pytest -sv tests/python/e2e/perf/grpc --perf-case=rust_grpc
```

Additional perf suites can be added as siblings of `grpc/` without sharing fixtures.
