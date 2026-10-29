# E2E performance tests

GPU-backed performance scenarios live in subdirectories (each with its own
`conftest.py`, cases, and docs).

| Directory | Description |
|-----------|-------------|
| [`grpc/`](grpc/README.md) | Prefix-KV hit/miss via router + vllm-rs / vllm (rust gRPC, rust HTTP, python HTTP) |
| [`l1_encode/`](l1_encode/README.md) | L1 prefix encoding cache (`encode_ms`, `encode_reuse`); one worker, router L1 on then L1 off |

Run the gRPC suite (**must** target `perf/grpc`: `--perf-case` is registered only in
`grpc/conftest.py`, not in this directory):

```bash
export MODEL=/path/to/model
python -m pytest -sv tests/python/e2e/perf/grpc --perf-case=rust_grpc
```

L1 prefix encoding cache (16k, `--shape turns`). One worker stays up. Router L1 on, then a router cold start with L1 off. `-s` streams `[vllm-rs]` and `[vllm-router]` logs. Each scenario prints its own `encode_ms`, `encode_reuse`, and JSON path.

```bash
# MODEL defaults to Qwen/Qwen3.8-27B; set it to a local directory to override
python -m pytest -sv tests/python/e2e/perf/l1_encode/test_l1_encode.py::test_l1_encode -k "l1on and hit030 and c4"
```

Additional perf suites can be added as siblings of `grpc/` without sharing fixtures.
