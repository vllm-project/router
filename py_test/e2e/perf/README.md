# gRPC / HTTP prefix-KV performance (full stack)

End-to-end perf on A100-class GPU: **`stack_launcher.py`** starts worker + **`vllm-router`**
from `cases/*.json`, then runs **`bench_prefix_kvhit.py`** waves (warmup + concurrent load).

Reference workload: Qwen3.8-27B (`vllm-rs serve` / `vllm serve` + `vllm-router` + prefix-KV bench).

## Flow

```text
export MODEL=/path/to/Qwen3.8-27B/snapshot
pytest py_test/e2e/perf --perf-case=rust_grpc -m perf
  → vllm-rs serve (grpc) + vllm-router → 8 waves → teardown
```

Repeat with `--perf-case=rust_http` and `python_http` (**3 backends × 8 waves = 24** runs total).

## Run

```bash
export MODEL=/path/to/model
export VLLM_RS=/path/to/vllm-rs          # rust_grpc
export ROUTER_BIN=$PWD/target/release/vllm-router   # optional
export PERF_SAVE_RESULTS=1               # optional, for cross-backend comparison

pytest py_test/e2e/perf --perf-case=rust_grpc -m perf
pytest py_test/e2e/perf --perf-case=rust_http -m perf
pytest py_test/e2e/perf --perf-case=python_http -m perf
```

Debug with an existing stack: `PERF_EXTERNAL=1` + `ROUTER_URL`.

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
| `worker.cli` | default Qwen3.8-27B flags, **`--max-num-seqs 128`**, TP=2, prefix caching, … |
| `bench` | **tokens 131072**, **max_tokens 64**, **timeout 1800**, warmup via `PERF_WARMUP` |
| `waves` | table above |

Override bench via env: `PERF_TOKENS`, `PERF_MAX_TOKENS`, `PERF_TIMEOUT`.

One wave equals:

```bash
python benches/backend/bench_prefix_kvhit.py \
  --router-url http://127.0.0.1:16002 \
  --model "$MODEL" \
  --hit-rate <0.00|0.99> \
  --tokens 131072 \
  --model-dir "$MODEL" \
  --max-tokens 64 \
  --warmup \
  --concurrency C \
  --requests R \
  --timeout 1800
```

Serve/router flags come from `cases/*.json` (`stack`, `worker.cli` defaults).

## Layout

```text
cases/*.json
case_loader.py
stack_launcher.py
prefix_kvhit_runner.py
conftest.py
test_prefix_kvhit_perf.py
```

## Related

- `scripts/backend/README.md`
- `benches/backend/bench_prefix_kvhit.py`
