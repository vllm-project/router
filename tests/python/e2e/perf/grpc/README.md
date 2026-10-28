# gRPC / HTTP prefix-KV performance (full stack)

End-to-end perf on A100-class GPU: **`stack_launcher.py`** starts worker + **`vllm-router`**
from `cases/*.json`, then runs **`bench_prefix_kvhit.py`** waves (warmup + concurrent load).

Reference workload: Qwen3.8-27B (`vllm-rs serve` / `vllm serve` + `vllm-router` + prefix-KV bench).

## Flow

```text
pytest tests/python/e2e/perf/grpc --perf-case=rust_grpc
  → per wave: cold start vllm-rs + vllm-router → bench → teardown (×8)
```

Repeat with `--perf-case=rust_http` and `python_http` (**3 backends × 8 waves = 24** cold starts total).

### Cold start (default)

Each parametrized **wave** gets its own **`perf_stack`**: worker and router are
**stopped after the test** and **launched again** before the next wave (fresh model
load and empty prefix KV). Expect **much longer** total time than one shared stack.

`PERF_STACK_STOP_SLEEP` (default `3`) seconds after teardown before the next launch.
`PERF_EXTERNAL=1` skips pytest launch; cold start is then your responsibility.

Wave-level **warmup** (`PERF_WARMUP=1`, default) still runs miss/hit requests after
each cold stack is up, before the concurrent wave.

## Run

Model path or HF id: set `"model"` in `cases/<name>.json`, or `export MODEL=...`
(**env overrides JSON**).

```bash
export MODEL=/path/to/model   # required unless case JSON defines model
export VLLM_RS=/path/to/vllm-rs          # rust_grpc
export ROUTER_BIN=$PWD/target/release/vllm-router   # optional
# results JSON (default on, dir created at run time): tests/python/e2e/perf/grpc/results/<backend>_<ts>/
# export PERF_SAVE_RESULTS=0   # disable file output

pytest tests/python/e2e/perf/grpc --perf-case=rust_grpc
pytest tests/python/e2e/perf/grpc --perf-case=rust_http
pytest tests/python/e2e/perf/grpc --perf-case=python_http
```

Debug with an existing stack: `PERF_EXTERNAL=1` + `ROUTER_URL`.

### No local stack / no pytest

If workers and router already run on a GPU machine (Slurm, manual `serve_*.sh`, etc.),
you do **not** need pytest or `stack_launcher.py` — only the bench CLI:

```bash
export ROUTER_URL=http://127.0.0.1:16002
export MODEL=/path/to/model
python benches/backend/bench_prefix_kvhit.py \
  --hit-rate 0.99 --tokens 131072 \
  --max-tokens 64 --warmup --concurrency 16 --requests 32 --timeout 1800
```

Optional pytest on that host: same env + `PERF_EXTERNAL=1` (skips launch, reuses bench
functions loaded from `bench_prefix_kvhit.py`).

## Wave matrix (per `--perf-case`)

| hit_rate | concurrency | requests |
|----------|-------------|----------|
| 0% | 4 | 8 |
| 0% | 8 | 16 |
| 0% | 12 | 24 |
| 0% | 16 | 32 |
| 99% | 4 | 8 |
| 99% | 8 | 16 |
| 99% | 12 | 24 |
| 99% | 16 | 32 |

Filter one wave: `-k hit99-c16-r32`.

### Looks stuck after the test name?

Pytest prints the node id, then runs **session fixtures** before the test body. With
`-s`, stderr now shows `[perf stack]` / `[perf bench]` progress. Typical timeline for
this case (27B-class, **131072** input tokens):

1. **`perf_stack`** — `vllm-rs serve` + router `/health` (often **10–30+ minutes** on first load).
2. **Tokenizer** — building the 131k-token prompt (minutes).
3. **Warmup** — two serial full-prefix requests (each up to `bench.timeout`, default **1800s**).
4. **Wave** — 32 requests at concurrency 16 (long; 99% hit still sends large bodies).

Quick smoke (not comparable to acceptance matrix):

```bash
export PERF_TOKENS=4096
export PERF_WARMUP=0
pytest -sv tests/python/e2e/perf/grpc --perf-case=rust_grpc -k hit99-c16-r32
```

Or attach to an existing stack: `PERF_EXTERNAL=1` + `ROUTER_URL=...` (skips launch).

## Acceptance (manual / downstream gate)

Let **G** = rust gRPC, **R** = rust HTTP, **P** = python HTTP. Compare `wave.summary`
**ttft_ms.avg** and **e2e_ms.avg** from `results/` after the three pytest runs above.

1. **hit_rate 0%:** G must not regress vs R and vs P on **ttft** and **e2e**.
2. **hit_rate 99%:** G must not regress vs R on **ttft** and **e2e**.
3. **hit_rate 99%:** G **e2e** must not regress vs P; G **ttft** should be ≥ **20% better** than P
   (`ttft_G <= 0.8 * ttft_P`).

Pytest only asserts each wave completes with valid latency stats; it does **not** auto-check
cross-backend SLOs.

## Case JSON

| Section | Content |
|---------|---------|
| `stack` | router **16002**, worker **18002**, gRPC **15002**, prom **29002/29001/29003** |
| `worker.cli` | **required** serve flags (Qwen3.8-27B example: TP=2, **`--max-num-seqs 128`**, prefix caching) |
| `bench` | **tokens 131072**, **max_tokens 64**, **timeout 1800**, warmup via `PERF_WARMUP` |
| `waves` | table above |

Override bench via env: `PERF_TOKENS`, `PERF_MAX_TOKENS`, `PERF_TIMEOUT`.

Each wave prints a **`perf summary`** block to stderr after the test (use `-sv`).
Set `PERF_QUIET=1` to disable terminal summary. Wave JSON is written under
`tests/python/e2e/perf/grpc/results/` **by default**; use `PERF_SAVE_RESULTS=0` to skip files.

One wave equals:

```bash
python benches/backend/bench_prefix_kvhit.py \
  --router-url http://127.0.0.1:16002 \
  --model "$MODEL" \
  --hit-rate <0.00|0.99> \
  --tokens 131072 \
  --max-tokens 64 \
  --warmup \
  --concurrency C \
  --requests R \
  --timeout 1800
```

Serve/router flags come from `cases/*.json` (`stack`, `worker.cli`).

## Layout

```text
cases/*.json
case_loader.py
stack_launcher.py   # optional if PERF_EXTERNAL=1
conftest.py
test_prefix_kvhit_perf.py
benches/backend/bench_prefix_kvhit.py   # HTTP bench + pytest helpers
```

## CI

Buildkite step **`:python: grpc perf test case`** (`key: grpc-perf-e2e` in
`.buildkite/test-nightly.yml`, L3 **nightly**): **`mithril-h100-pool`** (preset
`h100_2`). The job pulls the router CI image and does not build in-job.
`vllm-router` is on `PATH` (`/usr/local/bin/vllm-router`); the step does not set
`ROUTER_BIN`. `HF_TOKEN` comes from the `h100_2` pod secret. The step sets
`MODEL=Qwen/Qwen3.8-27B`, `PERF_STACK_TIMEOUT=1800`, and
`PERF_WORKER_STARTUP_TIMEOUT=1800`. Step timeout **90 min**, then:

`python -m pytest -sv tests/python/e2e/perf/grpc -k c16-r32 --perf-case=rust_grpc`

`python -m pytest -sv tests/python/e2e/perf/grpc -k c16-r32 --perf-case=python_http`

Wave JSON under `results/` is collected as build artifacts.

## Related

- `scripts/backend/README.md`
- `benches/backend/bench_prefix_kvhit.py`
