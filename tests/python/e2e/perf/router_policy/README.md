# router_policy — TraceLab and AgentX cold replay, router policy × concurrency

Each **scenario JSON** under `cases/` is one routing scenario. `benchmark_params.max-concurrency` is `[8]` for merged cases and `[12]` for PD cases; pytest collects one cell per scenario, such as `pd_merge_cache_aware/c8` or `pd_disagg_cache_aware/c12`:

- **`runtime`** — paths **relative to vllm-router repo root** where applicable. **`MODEL`** overrides `runtime.model`. **`AGENTX_PYTHON`** (or `PYTHON`) overrides the pytest interpreter. The CI image runs `vllm-router` from `PATH` (`/usr/local/bin`) and `vllm-rs` from the installed `vllm` package.
- **agentinfer** — install via **`pip install …whl`** (or `uv pip install …whl`) into the **same env as pytest** (no source checkout). Replay YAML is loaded from the installed package when a cell starts. Optional: `runtime.replay_config`; optional `runtime.agent_infer` for source dev.
- **`workers`** — engine count and per-worker serve args (KV, max_model_len, optional `gpu_memory_utilization`, optional `kv_cache_metrics` / `kv_cache_metrics_sample` → vLLM `--kv-cache-metrics`, …)
- **`router_params`** — policy, gRPC, penalties, balance thresholds
- **`benchmark_params`** — `max-concurrency`, task-num, and the rest of the replay settings. pytest requires **all** replay requests to succeed and match the complete replay plan.

Merged scenarios use eight replay sessions at `c8`; PD scenarios use twelve
sessions at `c12`. Requests within each session are serial. These labels denote
the configured session-concurrency limit; request concurrency is at most eight
or twelve respectively and can be lower because of trace gaps and completed
sessions. The workload includes no `c16` or `c32` cells.

Example: [cases/pd_merge_kv_aware.json](cases/pd_merge_kv_aware.json).

Worker and router processes use [`helpers/process.py`](../../../helpers/process.py), the same launch path as `perf/grpc`.

## Run

```bash
export MODEL=Qwen/Qwen3-4B-Instruct-2507   # or local model dir
# optional if pytest’s python is not the CI image venv:
# export AGENTX_PYTHON=/path/to/.venv/bin/python

pytest tests/python/e2e/perf/router_policy/test_router_policy_scenarios.py -v
```

One concurrency / policy. `-sv` shows worker, router, and replay lines on stderr:

```bash
pytest tests/python/e2e/perf/router_policy/test_router_policy_scenarios.py -sv -k 'pd_merge_kv_aware/c8'
```

## Artifacts layout

Performance JSON and the run directory are separate. The nightly dashboard copies `tests/python/e2e/results` only.

- `tests/python/e2e/results/perf/router_policy/<scenario>_<run_suffix>/c<concurrency>_t<task_num>.json` — copy of `calibration.json` when the cell is ok. One directory per scenario and session; one file per concurrency. The `concurrency` / `task_num` fields match the file name.
- `tests/python/e2e/run/perf/router_policy/<case_id>-<run_suffix>/` — cell files. Nightly uploads the complete cell directories matched by `run/perf/router_policy/c*-*/**/*`, including replay results, logs, Prometheus snapshots, and command files. The generated replay wrapper (`replay_entry.py`, including compatibility and stagger patches) is retained beside its command and logs after cleanup and uploaded for diagnosis and reproduction. Model caches stay under `tests/python/e2e/run/perf/router_policy/cache/` (shared across cells in this work root, excluded from nightly uploads). The same stdout is also printed (`[engineN]`, `[vllm-router]`, `[replay]`).

`<run_suffix>` is local time `YYYYMMDD_HHMMSS` (same format as grpc and l1_encode) unless you set **`AGENTX_RUN_SUFFIX`** (same suffix for all cells in one matrix, e.g. `export AGENTX_RUN_SUFFIX=20251029_171300`). The run directory is generated automatically from the case ID and run suffix.

Prometheus snapshots are saved under `before/`, `after/`, and `metrics/<elapsed_seconds>/` (approximately every 60 seconds during replay). They are uploaded with the cell directory for diagnosis. Replay logging uses the shared process stdout tee; cleanup joins the reader thread before closing the log.

Before replay, the cell runner **removes `result/`** if it already exists. The cell **fails if replay exit code ≠ 0**.

**Ports**: HTTP, gRPC, and the router use OS-assigned ports held until that process starts. KV event PUB/replay use a free port in **25100–25299**, also held until that worker starts. This vLLM reports the configured endpoint unchanged, so `tcp://*:0` comes back as port 0 and the router rejects it. After each worker is healthy, the runner calls gRPC `Control.GetKvEventSources` and fails the cell if an advertised endpoint still uses port 0. Between pytest cells, default **`AGENTX_CELL_GAP_SEC=5`**; before allocate, **`AGENTX_PORT_SETTLE_SEC=2`**. Do not run multiple matrix jobs on one host in parallel.

Worker and router processes are started with `helpers/process.py` (`launch_worker` once per worker, then `launch_router`). Pytest hides the stderr stream unless the invocation uses **`-sv`**.

All six scenarios use the same vLLM 0.30.0 KV-event contract described in the
[KV-aware worker requirements](https://github.com/vllm-project/router/blob/main/docs/load_balancing/kv_aware.md#worker-requirements).
Each case sets `workers.per_worker.prefix_caching_hash_algo` to `sha256_cbor`;
the launcher passes `--prefix-caching-hash-algo sha256_cbor` after the `vllm-rs`
`--` separator. Workers receive `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0` so events
carry complete 32-byte hashes. Every worker start gets a fresh UUID topic in
`--kv-events-config`, alongside the reserved live and replay endpoints. This
also applies to both Prefill and Decode in PD cells. The newer `instance_topic`
option is not used because vLLM 0.30.0 does not support it. The effective flags
and topic are retained in `engine*.log.command.json`.

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
| `pd_disagg_round_robin` | `round_robin` | `pd_disaggregated` |

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

Before publication, `replay_validation` checks the persisted requests against
AgentInfer's `result/replay-source-analysis.json` and `result/replay-plan.json`,
which are written before measured requests start. Each planned task must contain
all measured request keys from its source session. The configured task count,
runtime session IDs, and request IDs must be valid; the pinned
cases must cover their complete source cycle. Only `request` nodes are counted;
timing dependencies do not generate measured request rows.

Actual rows must have nonempty, globally unique `request_id` values and exactly
match the planned `(session_id, request_id)` pairs. Missing or extra rows, wrong
IDs, duplicates, and missing or malformed plan/source artifacts fail the cell
and prevent dashboard publication. Expected counts come from the source
inventory and selected plan, independently of `requests.jsonl` or its summary.
The bundled workloads require 28 TraceLab requests or 144 AgentX requests; a
TraceLab URL override is checked against its own inventory and plan. Counts and
failure reasons are recorded in `replay_validation` in `calibration.json`.

Merged `kv_aware` / `kv_aware_aff` cells and disaggregated `kv_aware` cells also
validate router activity from the existing `after/router.prom` snapshot,
scraped from the router's dedicated Prometheus port after replay. Each cell
starts a fresh router, so these counters belong to that cell. The runner checks
snapshot HTTP status codes before saving metrics.

- Index liveness requires a nonzero
  `vllm_router_kv_placement_lookups_total{result="hit"}` or `result="miss"`.
  `index_unavailable`, `missing_tokens`, other fallback results, and nonzero
  `vllm_router_kv_routing_inputs_total` alone do not prove a verified index was
  queried. A miss is valid activity; a cache hit is not required.
- Merged affinity cells require a nonzero regular-router counter:
  `vllm_router_prefix_routing_decisions_total{policy="kv_aware",stage="regular",outcome="sticky_hit"}`.
  Merged routing has no separate Decode-stage affinity counter.
- `cache_aware` and `round_robin` cells skip both checks.

Counts and diagnostic errors are recorded in `router_policy_checks`. Missing,
zero, or invalid required counters fail the cell before dashboard publication,
even if every request succeeded. Index failures point to KV event subscriptions
and the engine's `sha256_cbor` discovery contract (see #493). These are functional
activity checks, with no latency thresholds or baseline comparisons. No extra
replay requests or metric scrapes are added.

The `prefill` block reports interval metrics computed once in
`write_calibration()` from the existing `before/` and `after/` Prometheus
snapshots. PD cells include only workers before `workers.prefill_count`;
merged cells include all workers, which also perform Prefill.

| Field | Meaning |
| --- | --- |
| `prefill.prefix_cache_hit_rate` | Sum of Prefill `prefix_cache_hits_total` deltas divided by the sum of `prefix_cache_queries_total` deltas. Worker percentages are not averaged. |
| `prefill.new_prompt_tokens` | Sum of Prefill `prompt_tokens_by_source_total{source="local_compute"}` deltas, the actual locally computed prompt tokens. |
| `prefill.new_prompt_tokens_max_mean` | Maximum per-worker new prompt tokens divided by their mean. |
| `prefill.workers` | Worker indices and individual token-counter deltas, including new prompt tokens per worker. |
| `prefill.unavailable_metrics` | Fields unavailable because a counter is missing, invalid, or reset between snapshots. |

Counter label series are summed within each worker. The `local_cache_hit` and
`external_kv_transfer` source series are excluded from locally computed tokens.
The counter definitions follow [vLLM 0.30.0](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/metrics/stats.py).
An aggregate is `null` if any contributing worker is unavailable. Zero counters
stay zero; a hit rate with no queries or skew with no new tokens is `null`.
These fields are report-only and do not add performance thresholds or affect
the existing latency-validation gate. The result publisher copies the same
`calibration.json`, including this block, without recomputing any metrics.

The historical `cache_hit_rate` field remains the cumulative, all-worker cache
lookup ratio. The new `prefill.prefix_cache_hit_rate` uses Prefill interval
counters.

## Buildkite nightly (L3)

The router policy steps in [`.buildkite/test-nightly.yml`](../../../../../.buildkite/test-nightly.yml)
run on `l4_4` (four L4 GPUs per job).

- `router-policy-perf-e2e` selects `pd_merge and c8` and runs the three merged scenarios.
- `router-policy-pd-disagg-perf-e2e` selects `pd_disagg and c12` and runs
  `cache_aware`, `kv_aware`, and `round_robin` sequentially in one four-GPU job.
  The three scenarios share a 180-minute timeout and artifact upload. This
  increases the previous 90-minute budget; the total runtime still needs
  confirmation on the actual four-L4 lane. Failures in either router policy
  job fail the Buildkite build.

The PD cases use two prefill workers and two decode workers, each on its own
GPU, with Rust gRPC frontends and NIXL KV transfer. Set `workers.prefill_count`
to the number of prefill workers. The remaining workers decode. Merged cases
use the default `prefill_count=0`. All cases reserve 12 GiB GPU KV per worker
and use no CPU offload. Merged cases use `max_model_len=32768`; PD cases use
`max_model_len=65536` for the pinned AgentX requests (maximum input plus output
65,181 tokens). For the configured BF16 Qwen3-4B model, one full 64K sequence
uses 9 GiB of KV, within the per-worker KV allocation.

The CI image supplies `agentinfer==0.2.0`, vLLM 0.30.0, NIXL, and the Rust
router. Merged cases use `data/tracelab-ci-uniform-28.jsonl` (eight sessions,
28 requests). The three PD cases share `data/agentx-ci-12x12.jsonl` (twelve sessions,
twelve turns each, 144 requests) with `trace_type=agentX`, `/v1/completions`,
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
python -m pytest -sv tests/python/e2e/perf/router_policy/test_router_policy_scenarios.py -k 'pd_disagg and c12'
```

Earlier TraceLab runs of cache-aware and KV-aware affinity passed on two L20
GPUs with one prefill worker, one decode worker, and all 28 requests successful.
Those runs used the vLLM native sampler because FlashInfer CUDA compilation
failed in the test container. The current AgentX workload still needs GPU
validation with four GPUs and two prefill/two decode workers.
