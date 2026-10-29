# router_policy — AgentX replay, router policy × concurrency

Each **case JSON** under `cases/<policy>/` is self-contained (three policies × concurrency 8/16/32):

- **`runtime`** — paths **relative to vllm-router repo root** where applicable; **`model`** is the [Hugging Face model id](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (e.g. `Qwen/Qwen3-4B-Instruct-2507`).
- **agentinfer** — install via **`pip install …whl`** into the same env as `runtime.python` (no source checkout). Replay YAML is loaded from the installed package. Optional: set `runtime.replay_config` to a repo-relative YAML override; optional `runtime.agent_infer` only if you dev-install from source.
- **`workers`** — engine count and per-worker serve args (KV, max_model_len, …)
- **`router_params`** — policy, gRPC, penalties, balance thresholds
- **`benchmark_params`** — concurrency, task-num
- **`workload`** — expected request count for pytest asserts

Example: [cases/kv_aware/c16.json](cases/kv_aware/c16.json).

## Run

```bash
# Edit runtime.* paths in each case JSON you plan to run (or all nine).

export RUN_ROUTER_E2E_PERF=1
pytest tests/python/e2e/perf/router_policy/test_router_policy_scenarios.py -m perf -v
```

Single cell:

```bash
export AGENTX_CASE_FILE=tests/python/e2e/perf/router_policy/cases/kv_aware/c16.json
python -u tests/python/e2e/perf/router_policy/scripts/run_cell.py
```

## Analysis

```bash
python tests/python/e2e/perf/router_policy/scripts/collect_stats.py test-results/router_policy/runs
```
