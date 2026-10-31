# router_policy — AgentX replay, router policy × concurrency

Each **case JSON** under `cases/<policy>/` is self-contained (three policies × concurrency 8/16/32):

- **`runtime`** — paths **relative to vllm-router repo root** where applicable. **Env overrides JSON** (same as `scripts/backend/` and `perf/grpc`): `MODEL`, `VLLM_RS`, `ROUTER_BIN`, `AGENTX_PYTHON` (or `PYTHON`). Resolve `vllm-rs` with [`scripts/backend/check_vllm_rs.sh`](../../../../../scripts/backend/check_vllm_rs.sh).
- **agentinfer** — install via **`pip install …whl`** (or `uv pip install …whl`) into the **same env as pytest** (no source checkout). Replay YAML is loaded from the installed package when a cell starts. Optional: `runtime.replay_config`; optional `runtime.agent_infer` for source dev.
- **`workers`** — engine count and per-worker serve args (KV, max_model_len, optional `gpu_memory_utilization`, optional `kv_cache_metrics` / `kv_cache_metrics_sample` → vLLM `--kv-cache-metrics`, …)
- **`router_params`** — policy, gRPC, penalties, balance thresholds
- **`benchmark_params`** — concurrency, task-num
- **`workload`** — optional notes (e.g. `sessions`); pytest asserts **all** replay rows in `requests.jsonl` are `success` (no fixed request count)

Example: [cases/kv_aware/c16.json](cases/kv_aware/c16.json).

Binary resolution, HTTP readiness, and log streaming reuse [`tests/python/e2e/helpers/`](../../helpers/) (same as `perf/grpc`).

## Run

```bash
export MODEL=Qwen/Qwen3-4B-Instruct-2507   # or local model dir
export VLLM_RS=/path/to/vllm-rs            # check_vllm_rs.sh prints this
export ROUTER_BIN=$PWD/target/release/vllm-router
# optional if pytest’s python is not the case default (case "python" → sys.executable):
# export AGENTX_PYTHON=/path/to/.venv/bin/python

pytest tests/python/e2e/perf/router_policy/test_router_policy_scenarios.py -v
```

One concurrency / policy:

```bash
pytest tests/python/e2e/perf/router_policy/test_router_policy_scenarios.py -sv -k 'kv_aware/c16'
```

Live worker + router logs during a long cell (replay still file-only):

```bash
export AGENTX_STREAM_LOGS=1
pytest tests/python/e2e/perf/router_policy/test_router_policy_scenarios.py -sv -k 'kv_aware/c16'
```

## Artifacts layout

Each run writes a **new** cell directory:

- `test-results/router_policy/runs/<case_id>-<run_suffix>/` — logs, metrics snapshots, `result/requests.jsonl`, `calibration.json`
- `test-results/router_policy/results/<case_id>-<run_suffix>.json` — copy of `calibration.json` for pytest

`<run_suffix>` is local time `YYYYMMDDhhmmss` unless you set **`AGENTX_RUN_SUFFIX`** (same suffix for all cells in one matrix, e.g. `export AGENTX_RUN_SUFFIX=202510291713`). Override the run dir with **`AGENTX_OUT`**.

Before replay, the cell runner **removes `result/`** if reusing the same `AGENTX_OUT`. The cell **fails if replay exit code ≠ 0**.

**Ports**: HTTP, gRPC, and the router use OS-assigned ports held until that process starts. KV event PUB/replay use **`tcp://*:0`**: vLLM binds a free port at startup and the router reads the resolved address from `/kv_event_sources`. This needs a vLLM that resolves port 0 (it must not advertise `tcp://*:0` itself). Between pytest cells, default **`AGENTX_CELL_GAP_SEC=5`**; before allocate, **`AGENTX_PORT_SETTLE_SEC=2`**. Do not run multiple matrix jobs on one host in parallel.

Optional **`export AGENTX_STREAM_LOGS=1`**: stream **worker + router** logs to stderr in real time (prefix `[engineN]` / `[router]`); replay stays file-only. Use **pytest `-sv`** so stderr is visible.

At the end of a cell, the runner **stops workers/router first** (so live streams do not bury the footer), then prints **`STATS`**, **`ARTIFACTS`** (includes **`results_summary.json`** when pytest sets `AGENTX_RESULTS_SUMMARY`), **`PORTS`**, and **`LOG_SUMMARY`** (error scan + last N lines of every engine/router/replay log; `AGENTX_LOG_TAIL`, default 50). With **`AGENTX_STREAM_LOGS=1`**, a `--- cell summary ---` banner marks this block.

Layout matches **`perf/grpc`**: case JSON + [`casekit/case_config.py`](casekit/case_config.py) + stack runner (no separate `scripts/` tree).

## Buildkite nightly (L3)

Job **`router-policy-perf-e2e`** in [`.buildkite/test-nightly.yml`](../../../../../.buildkite/test-nightly.yml)
runs **`cache_aware/c16`**, **`kv_aware/c16`**, **`kv_aware_aff/c16`** on **`h100_4`** (4× H100).
Case JSON reserves ~48 GiB KV cache per worker and `max_model_len=196608`; **L4 (24 GB) is not sufficient**.

Nightly step runs `pip install agentinfer==0.2.0` then three pytest invocations (`-k …/c16`).
Optional Buildkite env **`AGENTX_TRACE_JSONL_URL`** to download `data/agentx-128-t4.jsonl` when the file is not in the checkout.
See [`.buildkite/test-nightly.yml`](../../../../../.buildkite/test-nightly.yml) (`router-policy-perf-e2e`).
