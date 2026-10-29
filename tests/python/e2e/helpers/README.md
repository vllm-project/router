# E2E helpers

Ported from the [`agentaware`](https://github.com/zhumingjue138/router/tree/agentaware) branch (`py_test/e2e/helpers`), adjusted for this repo layout (`tests/python/e2e/helpers`).

| Module | Role |
|--------|------|
| `http_ready.py` | Poll HTTP until ready (workers, router) |
| `server.py` | `managed_vllm` — start/stop vLLM serve, log streaming |
| `router_server.py` | `managed_router` — start/stop router (program-scheduling argv today) |
| `case_loader.py` | JSON case → `E2EConfig` / deploy params |
| `replay_run.py` | Validate completed agentinfer replay directories |
| `paths.py` | `repo_root()`, router binary default path |

Perf suites under `e2e/perf/*` should reuse these for lifecycle and readiness; suite-specific argv (gRPC workers, kv_aware, etc.) belongs in case JSON + a small builder (see `router_policy/cases/`).
