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

Configure paths in each case JSON under `router_policy/cases/<policy>/` (see
[`router_policy/README.md`](router_policy/README.md)). Run only that file (manual
GPU benchmark; not part of default CI):

```bash
export MODEL=/path/to/model
pytest tests/python/e2e/perf/router_policy/test_router_policy_scenarios.py -v
```

Shared path setup: [`conftest.py`](conftest.py). Reusable helpers live under
[`../helpers/`](../helpers/).

Additional perf suites can be added as siblings without sharing fixtures.
