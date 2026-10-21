# E2E performance tests

GPU-backed performance scenarios live in subdirectories (each with its own
`conftest.py`, cases, and docs).

| Directory | Description |
|-----------|-------------|
| [`grpc/`](grpc/README.md) | Prefix-KV hit/miss via router + vllm-rs / vllm (rust gRPC, rust HTTP, python HTTP) |

Run a suite:

```bash
python -m pytest -sv py_test/e2e/perf/grpc --perf-case=rust_grpc -m perf
```

Additional perf suites can be added as siblings of `grpc/` without sharing fixtures.
