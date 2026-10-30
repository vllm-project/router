# L1 prefix encoding cache (16k)

GPU perf for the router **L1 prefix encoding cache**. The `vllm-rs` worker
stays up through **L1 on**. Before **L1 off**, the worker and the router both
cold-start, so the off arm does not keep the on arm's KV cache. L0 stays off.
`--enable-prefix-caching` stays on inside each arm.

The number to compare is `wave.summary.encode_ms.p50` and
`wave.summary.encode_reuse.p50`. `ttft_ms` and `e2e_ms` include GPU prefill
and are the wrong column. `warmup` is the serial miss then hit and is not
inside `wave.summary`.

Context is **16k** (`bench.tokens` 16384), not the 128k manual run.
`--shape turns` is required. The default `split` cuts inside one message,
where L1 has no boundary, so encode time does not move.

## Run

The default model is the Hugging Face id `Qwen/Qwen3.8-27B` (`cases/qwen3_16k.json`).
Set `MODEL` to a local directory (or another id) to override it. That string is
the worker served name and the bench `--model`. `-s` (included in `-sv`) streams
worker and router logs. Run pytest on this file or directory.

```bash
export VLLM_RS=/path/to/vllm-rs
# optional local snapshot:
# export MODEL=/path/to/Qwen3.8-27B
# optional: export ROUTER_BIN=$PWD/target/release/vllm-router

python -m pytest -sv tests/python/e2e/perf/l1_encode/test_l1_encode.py::test_l1_encode \
  -k "l1on and hit030 and c4"
```

Omit `-k` for the full matrix (L1 on, then L1 off).

| `-k` | Selects |
|------|---------|
| `l1on` / `l1off` | router arm |
| `hit030` | hit rate 0.30 |
| `hit99999` | hit rate 0.99999 |
| `hit0-` | hit rate 0 (`hit0` also matches `hit030`) |
| `c1-` / `c4` / `c8` / `c16` | concurrency (`c1` also matches `c16`) |

Examples:

```bash
python -m pytest -sv tests/python/e2e/perf/l1_encode/test_l1_encode.py::test_l1_encode \
  -k "l1off and hit99999 and c1"

python -m pytest -sv tests/python/e2e/perf/l1_encode/test_l1_encode.py::test_l1_encode \
  -k "hit0- and c4"
```

Node ids look like `l1on-hit030-c4-r8`.

## What one session does

1. Start `vllm-rs` on GPU 0,1 (TP2, `--max-model-len 32768`, prefix caching on) and wait for `/v1/models`.
2. **L1 on:** for each hit rate, cold-start the router (`VLLM_ROUTER_L1_CACHE=1`, `VLLM_ROUTER_L0_CACHE=0`, `VLLM_ROUTER_STAGES=1`), then run that rate's concurrencies on the same router. The next hit rate gets a new router so the tree from one rate is not already warm. The worker stays up.
3. **L1 off:** cold-start the worker and the router (`VLLM_ROUTER_L1_CACHE=0`). Every off-arm hit rate and concurrency shares that worker and router.
4. Each wave is `benches/backend/bench_prefix_kvhit.py` with `--shape turns --turns 8 --tokens 16384 --max-tokens 64 --warmup`.

Hit rates `0`, `0.30`, and `0.99999`. Concurrency `1`, `4`, `8`, and `16` (requests are twice the concurrency: 2, 8, 16, 32). `--max-num-seqs` is 16 because that is the highest concurrency. `0.99999` uses the same shape as any rate ≥ 0.99: the whole paragraph is shared turns, and only the last-turn tag changes.

| hit rate | L1 on `encode_reuse` | encode time vs L1 off |
|----------|----------------------|------------------------|
| 0 | about 0 (one boundary token) | small drop |
| 0.30 | near 0.30 | about 30% lower |
| 0.99999 | above 0.90, near 1 | more than 90% lower |

Pytest asserts the reuse band and that the wave finished. It does not compare encode time across scenarios. L1 off must report `encode_reuse` 0. The on arm must log `L1 enabled`. `L1 disabled for this model` fails the on arm.

After the prompt is rendered, read `prompt_tokens` in the summary. If the worker rejects the prompt, raise `--max-model-len` and `--max-num-batched-tokens` in `cases/qwen3_16k.json`. Do not set them to 16384. Drop `--default-chat-template-kwargs` if this checkpoint's template does not use `enable_thinking`. A text Qwen does not need `--language-model-only`; set `L1_LANGUAGE_MODEL_ONLY=1` only for a VL checkpoint.

## Logs and artifacts

`-sv` prints `[vllm-rs]` and `[vllm-router]` lines as they arrive. When a scenario finishes it prints that wave's `ttft_ms`, `e2e_ms`, `encode_ms`, and `frontend_ms` (`min`, `p50`, `avg`, `max`) and the JSON path. Warmup is not printed. There is no cross-scenario summary. L1 Prometheus text is stored on the JSON (`metrics_l1`) and is not printed. A `vllm_tokenizer_l1_bypasses_total` value above 0 fails the on arm.

JSON is written under `tests/python/e2e/perf/l1_encode/results/<timestamp>/` (one object per wave, not the bench CLI's two-document stdout). `PERF_SAVE_RESULTS=0` skips the files. `PERF_OUTPUT_DIR` overrides the directory.

`PERF_TOKENS` overrides 16384. The worker flags still have to cover `tokens + max_tokens + template`. `PERF_EXTERNAL=1` plus `ROUTER_URL` skips launch and does **not** switch L1; select one arm with `-k`.

## Ports

Router `30000`, worker HTTP `8000`, gRPC `50051`, Prometheus `29000`. Edit `cases/qwen3_16k.json` if they are taken. These are not the ports used by `perf/grpc`.
