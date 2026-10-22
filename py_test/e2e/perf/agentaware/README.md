# Agent-aware router trace replay performance E2E (`agentaware`)

Cold vLLM serve, **vLLM Router** (program scheduling), then `vllm bench serve --agentinfer replay` (requires the **AgentInfer wheel** in the same venv as vLLM — see below).
**One case JSON → one pytest run → one result directory**; **no in-test compare** between router arms.

All commands assume the **vllm-router repository root** as the current working directory. Case JSON,
`benchmark_params.config`, L1 outputs under `results/agentaware/l1/`, and replay YAML paths such as
`results/agentaware/traces/requests.jsonl` are resolved relative to that root (pytest also `chdir`s there
for this suite).

Design reference: [vLLM Router PR 286](https://github.com/vllm-project/router/pull/286) End-to-end replay, with
the omissions and overrides below.

## Flow (PR steps mapped)

| PR E2E step | In this suite |
| ----------- | ------------- |
| Trace smoke (PR 319–331) | **Omitted** — no dedicated smoke gate or trace-hash check in the perf path. |
| Calibration (PR 333–355) | **L1 once** — [`prepare_l1.py`](prepare_l1.py) (serve → calibrate → stop serve). |
| Program JSON (PR 357–404) | **Same L1 run** — `results/agentaware/l1/program-scheduling*.json`; L2 cases reference paths only. |
| Start Router (PR 406–424) | **L2** — per case, one program config arm. |
| Health before replay (PR 426–431) | **L2** — poll Router (and serve) ready; **then** start bench. |
| `replay-full.yaml` (PR 433–476) | **L2** — [`configs/replay-full.yaml`](configs/replay-full.yaml); set trace and deployment paths locally. |
| Full replay (PR 478–497) | **L2** — `vllm bench serve --agentinfer replay`. |
| Compare arms (PR 499+) | **Out of scope** — run two case files separately; compare artifacts manually if needed. |

### L2 runtime sequence (single case)

1. Preflight (ports, router binary, program JSON, replay config, trace file exist).
2. Start vLLM serve → poll `http://<worker>:<port>/v1/models` until ready (or fail).
3. Start Router → poll `http://127.0.0.1:3001/health` (or configured URL) until ready (or fail).
4. Optional: fetch `/scheduling/diagnostics` and scrape metrics (PR step 5 evidence).
5. Run `vllm bench serve --agentinfer replay` (foreground subprocess; `result_dir` must not exist).
6. Validate completed run (`summary.json`, same helper family as other E2E replay).
7. Teardown: stop Router, stop vLLM.

No fixed long sleeps; readiness is poll-based with a timeout ceiling.

### Pytest completion line (match `swebench_run`)

When `run_benchmark.py` is implemented, print one line on success, for example:

```text
[agentaware:<arm>] completed: tasks=<n> requests=<m> dir=<result_dir>
```

`<arm>` comes from the case (e.g. `binding_only` or `program_scheduling`). No automatic baseline/candidate compare in pytest.

## L1 prepare script (run once per deployment)

L1 is **not** pytest. One script **starts vLLM serve**, runs **prefill + decode calibration**, writes **both
program JSON files** under a fixed directory, then **stops serve**. L2 cases only read those JSON paths.

Edit [`configs/l1-serve.json`](configs/l1-serve.json) (model path and `serve_args` aligned with replay cases),
build Router (`cargo build --release`), then from the **vllm-router** repo root:

```bash
# Same venv as vLLM (needs httpx, numpy, prometheus-client for calibration)
python py_test/e2e/perf/agentaware/prepare_l1.py \
  --serve-config-file py_test/e2e/perf/agentaware/configs/l1-serve.json \
  --output-root results/agentaware/l1
```

L1 runs **`python -m tools.calibration.calibrate_prefill|decode`** from this router repo ([`tools/calibration/`](../../../../tools/calibration/) vendored from [agent-infer](https://github.com/openJiuwen-ai/agent-infer)). The AgentInfer **wheel only ships `agentinfer/`** and does not include `tools/`; you do **not** need a source checkout unless you pass **`--agent-infer-root`** / **`AGENT_INFER_ROOT`** to use upstream calibration instead.

**Artifacts** (`calibration.json`, program JSON) land under **`results/agentaware/l1`** here.

| Variable / flag | Purpose |
| -------- | ------- |
| `VLLM_ROUTER_BINARY` | Optional override for `router_params.binary` (default `target/release/vllm-router`) |
| `AGENT_INFER_ROOT` / `--agent-infer-root` | Optional: use `tools/calibration` from an agent-infer git checkout |

You may point `--serve-config-file` at a full agentaware case JSON instead; only `serve_env` /
`server_params` (and optional timeouts) are used for L1.

Advanced: if a worker is already idle, omit `--serve-config-file` and pass `--base-url` + `--model` (no serve
lifecycle in the script).

Stable outputs (local deployment; keep on the machine where you run L1):

| Path | Purpose |
| ---- | ------- |
| `results/agentaware/l1/calibration/calibration.json` | Prefill + decode profile |
| `results/agentaware/l1/program-scheduling.json` | Program-scheduling arm |
| `results/agentaware/l1/program-scheduling-binding-only.json` | Binding-only arm |

Case templates point at the two JSON files under `results/agentaware/l1/`. **Re-run bootstrap only when**
the deployment or model changes; use `--force` to recalibrate. With default **`--skip-if-ready`**, if both program
JSON files already exist, the command exits immediately without touching calibration.

Optional: pass **`--token-capacity-per-dp-rank`**; otherwise capacity is taken from
`calibration/decode/run_config.json` → `observed_kv_capacity_tokens` after decode calibration.

Decode batch sizes default to **`1,2,4,8,12`** (suited to `max-num-seqs=12`).

## Case JSON (L2 — serve, router, replay load)

Step 7 **task count, concurrency, timeout**, and **vLLM serve startup** live in the case file (same idea as
`swebench_run` and `function/tracereplay`). Workload matrix (**task-num = 2 × max-concurrency**), same naming as
other GLM perf cases (`glm52_<tasks>x<concurrency>_router_replay`):

| File suffix | task-num | max-concurrency |
| ----------- | -------- | --------------- |
| `24x12` | 24 | 12 |
| `32x16` | 32 | 16 |
| `48x24` | 48 | 24 |
| `64x32` | 64 | 32 |

Each shape exists under **`cases/binding_only/`** and **`cases/program_scheduling/`** (8 case JSON files).
Fix deployment paths before running: `server_params.model`, trace file at
`replay.trace_path` in [`configs/replay-full.yaml`](configs/replay-full.yaml), and L1 program JSON under
`results/agentaware/l1/`. Router binary defaults to `target/release/vllm-router` in case JSON.

```bash
cargo build --release   # once per Router change

pytest -s -v py_test/e2e/perf/agentaware/run_benchmark.py \
  --test-config-file py_test/e2e/perf/agentaware/cases/binding_only/glm52_24x12_router_replay.json

pytest ... --task-num 16 --max-concurrency 8   # optional overrides (same as swebench_run)
```

| Block | PR / role |
| ----- | --------- |
| `serve_env` | vLLM subprocess environment (Ascend, HCCL, …). |
| `server_params.model` | Model path on disk. |
| `server_params.serve_args` | Full `vllm serve` CLI map: **port 8077**, **DP/TP**, parsers, `max-num-seqs`, quantization, scheduler, etc. |
| `router_params` | Router binary (`target/release/vllm-router` or `VLLM_ROUTER_BINARY`), **3001 / 29000**, `worker_urls`, `intra_node_data_parallel_size` (= serve DP), `program_scheduling_config` under `results/agentaware/l1/`, health poll URL/timeout. |
| `router_arm` | Printed in the completion line (`binding_only`, `program_scheduling`); not a second pytest compare. |
| `wait_for_vllm_ready` | Poll worker `/v1/models` (default true). |
| `result_root` | Parent for auto-generated run directories under `test-results/agentaware/`. |
| `benchmark_params.config` | Replay YAML (trace, prefixes, router `base_url` in yaml). |
| `benchmark_params.task-num` | Replay sessions (PR step 7); overrides yaml `experiment.task_num` when loader wires CLI. |
| `benchmark_params.max-concurrency` | Replay concurrency; overrides yaml `experiment.max_concurrency`. |
| `benchmark_params.timeout` | Bench subprocess / readiness ceiling (seconds). |
| `benchmark_params.model` | Served name passed to replay CLI (`glm-5`). |
| `benchmark_params.worker_metrics_url` / `worker_tokenizer_base_url` | Worker observability endpoints (yaml may mirror; case keeps worker URL explicit for single-host). |

Trace path and prefix tuning live in [`configs/replay-full.yaml`](configs/replay-full.yaml) (via `benchmark_params.config`).

## Replay configuration

[`configs/replay-full.yaml`](configs/replay-full.yaml):

- Traffic: `backend.base_url` → Router (`:3001`).
- Observability / tokenize: `metrics_url`, `tokenizer_base_url` → worker (often `:8077` on loopback for single-host).
- `endpoint: /v1/chat/completions` with `prompt_transport: completion_token_ids` for agentinfer trace replay.
- `experiment.result_dir` must **not** exist before the run; override per run via CLI/case like other perf suites.
- Prefix and trace fields are trace-specific; tune on the deployment copy, not in the main repo.

## Repository layout

| Path | Role |
| ---- | ---- |
| `run_benchmark.py` | Pytest entry (`test_router_replay_completes`) |
| `helpers/case_config.py` | Load case JSON → `RouterReplayE2EConfig` |
| `helpers/replay_cli.py` | Build `vllm bench … replay` argv (Router `base-url`) |
| `helpers/replay_runner.py` | `managed_vllm` → `managed_router` → replay → validate |
| `cases/…/*.json` | Cases (four task×concurrency shapes × two router arms) |
| `configs/replay-full.yaml` | Replay YAML (set trace path locally) |
| `prepare_l1.py` | L1 prerequisite: serve → calibrate → program JSON → stop serve |
| `configs/l1-serve.json` | Serve config for L1 |

Shared E2E helpers ([`../../helpers/`](../../helpers/)):

| Module | Role |
| ------ | ---- |
| `server.py` | vLLM serve lifecycle (`managed_vllm`) |
| `router_server.py` | Router lifecycle (`managed_router`, program JSON compaction) |
| `http_ready.py` | HTTP readiness polling (Router `/health`, etc.) |
| `replay_run.py` | `validate_completed_run` (AgentInfer replay artifacts) |
| `paths.py` | Repository root and `AGENT_INFER_ROOT` handling |
| `case_loader.py` | Deploy parsing reused by agentaware case loader |

Orchestration is **Python + case JSON only** (no checked-in shell launchers).
