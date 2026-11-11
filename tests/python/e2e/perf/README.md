# E2E performance tests

GPU-backed performance scenarios live in subdirectories (each with its own
`conftest.py`, cases, and docs). They are not part of the default CI (`ready`) suite.

## Scenarios

| Directory | Description |
|-----------|-------------|
| [`grpc/`](grpc/README.md) | Prefix-KV hit/miss via router + vllm-rs / vllm (rust gRPC, rust HTTP, python HTTP) |
| [`router_policy/`](router_policy/README.md) | TraceLab cold replay: workers → router → bench; cells = 3 policies × 3 concurrency levels |
| [`l1_encode/`](l1_encode/README.md) | L1 prefix encoding cache (`encode_ms`, `encode_reuse`); one worker, router L1 on then L1 off |

## grpc

Run the gRPC suite (**must** target `perf/grpc`: `--perf-case` is registered only in
`grpc/conftest.py`, not in this directory):

```bash
export MODEL=/path/to/model
python -m pytest -sv tests/python/e2e/perf/grpc --perf-case=rust_grpc
```

## router_policy

Configure paths in each scenario JSON under `router_policy/cases/` (see
[`router_policy/README.md`](router_policy/README.md)). Run only that file (manual
GPU benchmark; nightly runs the three c16 cells):

```bash
export MODEL=/path/to/model
pytest tests/python/e2e/perf/router_policy/test_router_policy_scenarios.py -v
```

## l1_encode

L1 prefix encoding cache (16k, `--shape turns`). `test_qwen3_16k_l1on` and `test_qwen3_16k_l1off` are separate tests. Each starts one worker and one router and keeps them for that test's hit rates and concurrencies. `-s` streams `[vllm-rs]` and `[vllm-router]` logs. Each scenario prints its wave `ttft_ms`, `e2e_ms`, `encode_ms`, `frontend_ms`, and JSON path.

```bash
# MODEL defaults to Qwen/Qwen3-0.6B; set it to a local directory to override
python -m pytest -sv tests/python/e2e/perf/l1_encode/test_l1_encode.py::test_qwen3_16k_l1on -k "hit0.30 and c4"
```

Additional perf suites can be added as siblings of `grpc/` without sharing fixtures.
