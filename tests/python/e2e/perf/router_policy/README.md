# router_policy — TraceLab cold replay, router policy × concurrency

Each **scenario JSON** under `cases/` is one policy. `benchmark_params.max-concurrency` lists 8, 16, and 32; pytest expands those into cells `cache_aware/c16` and so on:

- **`runtime`** — paths **relative to vllm-router repo root** where applicable. **`MODEL`** overrides `runtime.model`. **`AGENTX_PYTHON`** (or `PYTHON`) overrides the pytest interpreter. The CI image runs `vllm-router` from `PATH` (`/usr/local/bin`) and `vllm-rs` from the installed `vllm` package.
- **agentinfer** — install via **`pip install …whl`** (or `uv pip install …whl`) into the **same env as pytest** (no source checkout). Replay YAML is loaded from the installed package when a cell starts. Optional: `runtime.replay_config`; optional `runtime.agent_infer` for source dev.
- **`workers`** — engine count and per-worker serve args (KV, max_model_len, optional `gpu_memory_utilization`, optional `kv_cache_metrics` / `kv_cache_metrics_sample` → vLLM `--kv-cache-metrics`, …)
- **`router_params`** — policy, gRPC, penalties, balance thresholds
- **`benchmark_params`** — `max-concurrency` (8, 16, 32), task-num, and the rest of the replay settings. pytest asserts **all** replay rows in `requests.jsonl` are `success` (no fixed request count)

Example: [cases/kv_aware.json](cases/kv_aware.json).

Worker and router processes use [`perf/process.py`](../process.py), the same launch path as `perf/grpc`.

## Run

```bash
export MODEL=Qwen/Qwen3-4B-Instruct-2507   # or local model dir
# optional if pytest’s python is not the CI image venv:
# export AGENTX_PYTHON=/path/to/.venv/bin/python

pytest tests/python/e2e/perf/router_policy/test_router_policy_scenarios.py -v
```

One concurrency / policy. `-sv` shows worker and router logs on stderr as they are written; replay stays in `replay.log`:

```bash
pytest tests/python/e2e/perf/router_policy/test_router_policy_scenarios.py -sv -k 'kv_aware/c16'
```

## Artifacts layout

Each run writes a **new** cell directory:

- `test-results/router_policy/runs/<case_id>-<run_suffix>/` — logs, metrics snapshots, `result/requests.jsonl`, `calibration.json`
- `test-results/router_policy/results/<case_id>-<run_suffix>.json` — copy of `calibration.json` for pytest

`<run_suffix>` is local time `YYYYMMDDhhmmss` unless you set **`AGENTX_RUN_SUFFIX`** (same suffix for all cells in one matrix, e.g. `export AGENTX_RUN_SUFFIX=202510291713`). Override the run dir with **`AGENTX_OUT`**.

Before replay, the cell runner **removes `result/`** if reusing the same `AGENTX_OUT`. The cell **fails if replay exit code ≠ 0**.

**Ports**: HTTP, gRPC, and the router use OS-assigned ports held until that process starts. KV event PUB/replay use a free port in **25100–25299**, also held until that worker starts. This vLLM reports the configured endpoint unchanged, so `tcp://*:0` comes back as port 0 and the router rejects it. After each worker is healthy, the runner calls gRPC `Control.GetKvEventSources` and fails the cell if an advertised endpoint still uses port 0. Between pytest cells, default **`AGENTX_CELL_GAP_SEC=5`**; before allocate, **`AGENTX_PORT_SETTLE_SEC=2`**. Do not run multiple matrix jobs on one host in parallel.

Worker and router processes are started with `perf/process.py` (`launch_worker` once per worker, then `launch_router`). The same output is streamed to stderr with prefixes `[engineN]` and `[vllm-router]`, and copied to `engineN.log` / `router.log` in the run directory. Replay stays in `replay.log`. Pytest hides the stderr stream unless the invocation uses **`-sv`**.

When replay exits, [`log_summary.py`](log_summary.py) writes **`calibration.json`** (TTFT, TPOT, JCT, and the other cell stats), then appends **`--- cell summary ---`**, **`STATS`**, and **`ARTIFACTS`** to **`replay.log`** immediately after agentinfer's `Replay result` line, and prints the same block on stdout. **`ARTIFACTS`** lists the run directory and **`results_summary.json`** when pytest sets `AGENTX_RESULTS_SUMMARY`. `calibration.json`, `requests.jsonl`, `ports.json`, `engineN.log`, `router.log`, and `replay.log` stay in the run directory and are not printed. Workers and the router are stopped after that. Worker and router lines already went to stderr while they were running.

Layout matches **`perf/grpc`**: case JSON + [`case_loader.py`](case_loader.py) + [`stack_launcher.py`](stack_launcher.py). [`log_summary.py`](log_summary.py) writes the stats and prints the cell summary.

## Buildkite nightly (L3)

Job **`router-policy-perf-e2e`** in [`.buildkite/test-nightly.yml`](../../../../../.buildkite/test-nightly.yml)
runs **`cache_aware/c16`**, **`kv_aware/c16`**, **`kv_aware_aff/c16`** on **`l4_4`** (4× L4).
Case JSON reserves 12 GiB GPU KV per worker, no CPU offload, and `max_model_len=32768`.

Nightly step runs `pip install agentinfer==0.2.0` then one pytest invocation (`-k 'cache_aware/c16 or kv_aware/c16 or kv_aware_aff/c16'`).
The trace is `data/tracelab-ci-uniform-28.jsonl` (8 sessions, 28 requests). Optional Buildkite env **`TRACELAB_JSONL_URL`** replaces that file when set.
See [`.buildkite/test-nightly.yml`](../../../../../.buildkite/test-nightly.yml) (`router-policy-perf-e2e`).
