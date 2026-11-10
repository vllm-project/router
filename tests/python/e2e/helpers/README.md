# E2E helpers

| Module | Role |
|--------|------|
| `binaries.py` | `resolve_vllm_rs` (via `scripts/backend/check_vllm_rs.sh` when bash is available), `resolve_router_bin`, repo-relative paths |
| `paths.py` | `repo_root()`, router binary default path |

Worker and router launch for `router_policy` and `perf/grpc` uses [`tests/python/e2e/perf/process.py`](../perf/process.py) (`launch_worker`, `launch_router`, `_wait_http`, `graceful_stop`). Suite-specific argv stays in the case JSON and that suite's `stack_launcher.py`.
