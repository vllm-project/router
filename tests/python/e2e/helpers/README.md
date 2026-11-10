# E2E helpers

| Module | Role |
|--------|------|
| `resolve_bins.py` | Find `vllm-rs` and `vllm-router` (`resolve_vllm_rs`, `resolve_router_bin`) and turn case-JSON paths into absolute paths |
| `repo_root.py` | `repo_root()`, the vllm-router checkout root those paths are relative to |

Worker and router launch for `router_policy` and `perf/grpc` uses [`tests/python/e2e/perf/process.py`](../perf/process.py) (`launch_worker`, `launch_router`, `_wait_http`, `graceful_stop`). Suite-specific argv stays in the case JSON and that suite's `stack_launcher.py`.
