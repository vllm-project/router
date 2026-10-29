# router_policy — AgentX replay, router policy × concurrency

Each **case JSON** under `cases/<policy>/` is self-contained (three policies × concurrency 8/16/32):

- **`runtime`** — paths **relative to vllm-router repo root** where applicable. **Env overrides JSON** (same as `scripts/backend/` and `perf/grpc`): `MODEL`, `VLLM_RS`, `ROUTER_BIN`, `AGENTX_PYTHON` (or `PYTHON`). Resolve `vllm-rs` with [`scripts/backend/check_vllm_rs.sh`](../../../../../scripts/backend/check_vllm_rs.sh).
- **agentinfer** — install via **`pip install …whl`** (or `uv pip install …whl`) into the **same env as `runtime.python`** (no source checkout). Pytest does not need agentinfer if `runtime.python` points at that env. Replay YAML is loaded from the installed package when `run_cell` starts. Optional: `runtime.replay_config`; optional `runtime.agent_infer` for source dev.
- **`workers`** — engine count and per-worker serve args (KV, max_model_len, optional `gpu_memory_utilization`, …)
- **`router_params`** — policy, gRPC, penalties, balance thresholds
- **`benchmark_params`** — concurrency, task-num
- **`workload`** — optional notes (e.g. `sessions`); pytest asserts **all** replay rows in `requests.jsonl` are `success` (no fixed request count)

Example: [cases/kv_aware/c16.json](cases/kv_aware/c16.json).

## Run

```bash
export MODEL=Qwen/Qwen3-4B-Instruct-2507   # or local model dir
export VLLM_RS=/path/to/vllm-rs            # check_vllm_rs.sh prints this
export ROUTER_BIN=$PWD/target/release/vllm-router
# optional if you launch pytest/run_cell with that venv’s python (case "python" → sys.executable):
# export AGENTX_PYTHON=/path/to/.venv/bin/python

export RUN_ROUTER_E2E_PERF=1
pytest tests/python/e2e/perf/router_policy/test_router_policy_scenarios.py -m perf -v
```

Single cell:

```bash
export AGENTX_CASE_FILE=tests/python/e2e/perf/router_policy/cases/kv_aware/c16.json
python -u tests/python/e2e/perf/router_policy/scripts/run_cell.py
```

## Analysis

At the end of a cell, `run_cell` prints **`ARTIFACTS`** (paths to `calibration.json`, `requests.jsonl`, logs). On failure it also prints **`LOG_TAIL`** snippets from engine/router/replay logs. Pytest adds **`results_summary.json`** under `test-results/router_policy/results/`.

```bash
python tests/python/e2e/perf/router_policy/scripts/collect_stats.py test-results/router_policy/runs
```
