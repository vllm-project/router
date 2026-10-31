# E2E helpers

Ported from the [`agentaware`](https://github.com/zhumingjue138/router/tree/agentaware) branch (`py_test/e2e/helpers`), adjusted for this repo layout (`tests/python/e2e/helpers`).

| Module | Role |
|--------|------|
| `binaries.py` | `resolve_vllm_rs` (via `scripts/backend/check_vllm_rs.sh` when bash is available), `resolve_router_bin`, repo-relative paths |
| `http_ready.py` | Poll HTTP until ready (workers, router) |
| `perf_process.py` | Shared perf stack subprocess helpers: `popen_merged_output`, `graceful_stop_subprocess`, `router_env_for_model`, `preflight_router_bin`, `wait_router_http_health`, `start_process_with_log` |
| `subprocess_io.py` | Stream subprocess stdout to stderr (+ optional log file) |
| `server.py` | `managed_vllm` — start/stop vLLM serve, log streaming |
| `router_server.py` | `managed_router` — start/stop router (program-scheduling argv today) |
| `case_loader.py` | JSON case → `E2EConfig` / deploy params |
| `replay_run.py` | Validate completed agentinfer replay directories |
| `paths.py` | `repo_root()`, router binary default path |

**Wired today:** `perf/grpc/stack_launcher.py`, `router_policy/casekit/` (`case_config`, `cell_runner`).

Perf suites under `e2e/perf/*` should reuse these for lifecycle and readiness; suite-specific argv (gRPC workers, kv_aware, etc.) belongs in case JSON + a small builder (see `router_policy/cases/`).
