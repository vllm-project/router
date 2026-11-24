# router_policy — TraceLab and AgentX cold replay, router policy × concurrency

Each **scenario JSON** under `cases/` is one routing scenario. `benchmark_params.max-concurrency` lists 8, 16, and 32; pytest expands those into cells `pd_merge_cache_aware/c16` and so on:

- **`runtime`** — paths **relative to vllm-router repo root** where applicable. **`MODEL`** overrides `runtime.model`. **`AGENTX_PYTHON`** (or `PYTHON`) overrides the pytest interpreter. The CI image runs `vllm-router` from `PATH` (`/usr/local/bin`) and `vllm-rs` from the installed `vllm` package.
- **agentinfer** — install via **`pip install …whl`** (or `uv pip install …whl`) into the **same env as pytest** (no source checkout). Replay YAML is loaded from the installed package when a cell starts. Optional: `runtime.replay_config`; optional `runtime.agent_infer` for source dev.
- **`workers`** — engine count and per-worker serve args (KV, max_model_len, optional `gpu_memory_utilization`, optional `kv_cache_metrics` / `kv_cache_metrics_sample` → vLLM `--kv-cache-metrics`, …)
- **`router_params`** — policy, gRPC, penalties, balance thresholds
- **`benchmark_params`** — `max-concurrency` (8, 16, 32), task-num, and the rest of the replay settings. pytest asserts **all** replay rows in `requests.jsonl` are `success` (no fixed request count)

Example: [cases/pd_merge_kv_aware.json](cases/pd_merge_kv_aware.json).

Worker and router processes use [`perf/process.py`](../process.py), the same launch path as `perf/grpc`.

## Run

```bash
export MODEL=Qwen/Qwen3-4B-Instruct-2507   # or local model dir
# optional if pytest’s python is not the CI image venv:
# export AGENTX_PYTHON=/path/to/.venv/bin/python

pytest tests/python/e2e/perf/router_policy/test_router_policy_scenarios.py -v
```

One concurrency / policy. `-sv` shows worker, router, and replay lines on stderr:

```bash
pytest tests/python/e2e/perf/router_policy/test_router_policy_scenarios.py -sv -k 'pd_merge_kv_aware/c16'
```

## Artifacts layout

Performance JSON and the run directory are separate. The nightly dashboard copies `tests/python/e2e/results` only.

- `tests/python/e2e/results/perf/router_policy/<scenario>_<run_suffix>/c<concurrency>_t<task_num>.json` — copy of `calibration.json` when the cell is ok. One directory per scenario and session; one file per concurrency. The `concurrency` / `task_num` fields match the file name.
- `tests/python/e2e/run/perf/router_policy/<case_id>-<run_suffix>/` — cell files. Nightly uploads the complete cell directories matched by `run/perf/router_policy/c*-*/**/*`, including replay results, logs, Prometheus snapshots, and command files. The generated replay wrapper (`replay_entry.py`, including compatibility and stagger patches) is retained beside its command and logs after cleanup and uploaded for diagnosis and reproduction. Model caches stay under `tests/python/e2e/run/perf/router_policy/cache/` (shared across cells in this work root, excluded from nightly uploads). The same stdout is also printed (`[engineN]`, `[vllm-router]`, `[replay]`).

`<run_suffix>` is local time `YYYYMMDD_HHMMSS` (same format as grpc and l1_encode) unless you set **`AGENTX_RUN_SUFFIX`** (same suffix for all cells in one matrix, e.g. `export AGENTX_RUN_SUFFIX=20251029_171300`). The run directory is generated automatically from the case ID and run suffix.

Prometheus snapshots are saved under `before/`, `after/`, and `metrics/<elapsed_seconds>/` (approximately every 60 seconds during replay). They are uploaded with the cell directory for diagnosis. Replay logging uses the shared process stdout tee; cleanup joins the reader thread before closing the log.

Before replay, the cell runner **removes `result/`** if it already exists. The cell **fails if replay exit code ≠ 0**.

**Ports**: HTTP, gRPC, and the router use OS-assigned ports held until that process starts. KV event PUB/replay use a free port in **25100–25299**, also held until that worker starts. This vLLM reports the configured endpoint unchanged, so `tcp://*:0` comes back as port 0 and the router rejects it. After each worker is healthy, the runner calls gRPC `Control.GetKvEventSources` and fails the cell if an advertised endpoint still uses port 0. Between pytest cells, default **`AGENTX_CELL_GAP_SEC=5`**; before allocate, **`AGENTX_PORT_SETTLE_SEC=2`**. Do not run multiple matrix jobs on one host in parallel.

Worker and router processes are started with `perf/process.py` (`launch_worker` once per worker, then `launch_router`). Pytest hides the stderr stream unless the invocation uses **`-sv`**.

When replay exits, [`log_summary.py`](log_summary.py) writes **`calibration.json`** (TTFT, TPOT, JCT, concurrency, task count, and the other cell stats); when the cell is ok, the same JSON is copied to `tests/python/e2e/results/perf/router_policy/<scenario>_<run_suffix>/c<concurrency>_t<task_num>.json` and prints **`--- cell summary ---`**, **`STATS`**, and **`ARTIFACTS`** on stdout. Quantile blocks (`ttft_s`, `tpot_ms`, `jct_s`) include `n`, the actual number of samples used for that metric, which can differ from `success`. TTFT excludes missing timings; TPOT also excludes missing latency and outputs with at most one token. JCT counts the duration samples used; precomputed JCT preserves `n` or `count`, or reports `n: null` when unknown. Empty sample sets report `n: 0`. Cell-level request counts remain in `requests` and `success`. **`ARTIFACTS`** lists the run directory and the results JSON path (`results_json`). Workers and the router are stopped after that.

Layout matches **`perf/grpc`**: case JSON + [`case_loader.py`](case_loader.py) + [`stack_launcher.py`](stack_launcher.py). [`log_summary.py`](log_summary.py) writes the stats and prints the cell summary.

The sample-count regression tests run on CPU without model weights:

```bash
python -m pytest tests/python/unit/test_router_policy_log_summary.py -q
```

## Results schema

Each case filename and its JSON `id` use the same scenario name. The loader keeps
that name in `scenario` and adds concurrency to the run-directory ID so concurrent
cells have separate output files.

| Scenario | Routing policy | Topology |
| --- | --- | --- |
| `pd_merge_cache_aware` | `cache_aware` | `pd_merged` |
| `pd_merge_kv_aware` | `kv_aware` | `pd_merged` |
| `pd_merge_kv_aware_affinity` | `kv_aware_aff` | `pd_merged` |
| `pd_disagg_cache_aware` | `cache_aware` | `pd_disaggregated` |
| `pd_disagg_kv_aware` | `kv_aware` | `pd_disaggregated` |
| `pd_disagg_kv_aware_affinity` | `kv_aware_aff` | `pd_disaggregated` |

Result JSON includes `scenario`, `topology`, and `router_policy`. The existing
`arm` field also identifies the routing policy for historical consumers.
Publication requires a nonempty string `scenario`. Calibration files with a
missing or invalid scenario stay in the cell directory and are not copied to
the results directory. The routing policy is never used as a scenario fallback.
The dashboard uses policy and topology to retain historical merged series and
match each disaggregated policy with a disaggregated cache-aware baseline.

A cell must have successful requests and valid TTFT, TPOT, and JCT measurements
before it publishes a result. Every metric needs finite, nonnegative mean,
p50, p95, and p99 values. A known sample count must be positive. Precomputed
JCT aggregates with an unknown count (`n: null`) remain supported. Failed
measurement checks are recorded in `missing_latency_metrics` and fail the cell.
These checks apply to both merged and disaggregated cells. A replay with all
requests successful still fails if no valid TPOT samples remain (for example,
all outputs have at most one token) or JCT durations are unavailable. Its
`calibration.json` is retained for diagnosis.

## Buildkite nightly (L3)

Two steps in [`.buildkite/test-nightly.yml`](../../../../../.buildkite/test-nightly.yml)
run on `l4_4` (four L4 GPUs).

- `router-policy-perf-e2e` selects `pd_merge and c16` and runs the three merged scenarios.
- `router-policy-pd-disagg-perf-e2e` selects `pd_disagg and c16` and runs the three
  disaggregated scenarios. Failures in either step fail the Buildkite build.

The PD cases use two prefill workers and two decode workers, each on its own
GPU, with Rust gRPC frontends and NIXL KV transfer. Set `workers.prefill_count`
to the number of prefill workers. The remaining workers decode. Merged cases
use the default `prefill_count=0`. All cases reserve 12 GiB GPU KV per worker
and use no CPU offload. Merged cases use `max_model_len=32768`; PD cases use
`max_model_len=65536` for the pinned AgentX requests (maximum input plus output
65,339 tokens). For the configured BF16 Qwen3-4B model, one full 64K sequence
uses 9 GiB of KV, within the per-worker KV allocation.

The CI image supplies `agentinfer==0.2.0`, vLLM 0.30.0, NIXL, and the Rust
router. Merged cases use `data/tracelab-ci-uniform-28.jsonl` (eight sessions,
28 requests). The three PD cases share `data/agentx-ci-8x8.jsonl` (eight sessions,
eight turns each, 64 requests) with `trace_type=agentX`, `/v1/completions`,
and `gap_scale=0.002`. AgentInfer's `replay_agentX.yaml` supplies the matching
token-ID replay settings. See [data provenance and reproduction](data/README.md).

The shared cell launcher calls `prepare_trace()` before starting workers.
AgentX cases verify `runtime.trace_sha256` against the bundled slice. TraceLab
cases check the local file or download optional `TRACELAB_JSONL_URL` into the
cell directory as `trace.jsonl`; this override applies only to TraceLab.
Downloads replace the cell's trace atomically, and failures stop the cell
before GPU processes start. Local pytest runs use the same setup.

Run the three PD scenarios in an environment with four supported CUDA GPUs.

```bash
python -m pytest -sv tests/python/e2e/perf/router_policy/test_router_policy_scenarios.py -k 'pd_disagg and c16'
```

Earlier TraceLab runs of cache-aware and KV-aware affinity passed on two L20
GPUs with one prefill worker, one decode worker, and all 28 requests successful.
Those runs used the vLLM native sampler because FlashInfer CUDA compilation
failed in the test container. The current AgentX workload still needs GPU
validation with four GPUs and two prefill/two decode workers.
