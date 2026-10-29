# E2E performance tests

GPU-backed performance scenarios live in subdirectories (each with its own
`conftest.py`, cases, and docs). They are not part of the default CI (`ready`) suite.

## Scenarios

| Directory | Description |
|-----------|-------------|
| [`grpc/`](grpc/README.md) | Prefix-KV hit/miss via router + vllm-rs / vllm (rust gRPC, rust HTTP, python HTTP) |
| [`router_policy/`](router_policy/README.md) | Fixed AgentX trace replay: workers → router → bench; cells = 3 policies × 3 concurrency levels |

## grpc

Run a suite (**must** target `perf/grpc`: `--perf-case` is registered only in
`grpc/conftest.py`, not in this directory):

```bash
export MODEL=/path/to/model
python -m pytest -sv tests/python/e2e/perf/grpc --perf-case=rust_grpc
```

## router_policy

Set **`RUN_ROUTER_E2E_PERF=1`** and configure paths in each case JSON under
`router_policy/cases/<policy>/`. Then, for example:

```bash
export RUN_ROUTER_E2E_PERF=1
pytest tests/python/e2e/perf/router_policy/test_router_policy_scenarios.py -m perf -v
```

Shared pytest gate and marker: [`conftest.py`](conftest.py). Reusable helpers for
e2e live under [`../helpers/`](../helpers/).

Additional perf suites can be added as siblings without sharing fixtures.
