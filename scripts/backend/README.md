# Backend serve examples

vLLM 0.29 worker + this rust router. These scripts do not set Slurm,
Docker, or device IDs. Engine knobs (`--max-num-seqs`, …) are **worker
CLI flags**, not router flags. `benches/backend/bench_prefix_kvhit.py`
does not start engines.

Optional env: `HOST`, `WORKER_PORT`, `GRPC_PORT`, `ROUTER_PORT`,
`ROUTER_BIN`, `VLLM_RS`, `VLLM_ROUTER_MODEL`, `VLLM_ROUTER_TOKENIZER`.
`VLLM_ROUTER_TOKENIZER` may point at a model directory or tokenizer file;
the router normalizes tokenizer files to their parent model directory.
`VLLM_ROUTER_STAGES=1` turns on optional router stage clocks; these are
diagnostic boundaries/residuals, not direct network or EngineCore telemetry.

## Check `vllm-rs`

```bash
./scripts/backend/check_vllm_rs.sh
# then copy the printed:  export VLLM_RS=...
```

Prints path, `vllm-rs --version`, and a copy-paste `export VLLM_RS=…`
line (does not change your shell). Resolution: `VLLM_RS`, then `PATH`,
then the 0.29 wheel (`site-packages/vllm/vllm-rs`).

## Raw `vllm-rs serve` (gRPC worker)

`VLLM_USE_RUST_FRONTEND=1 vllm serve` runs **`vllm-rs frontend`** (OpenAI
HTTP only). It cannot set `--grpc-port`. Call **`serve`** yourself:

```bash
export MODEL=/path/or/hf-id
./scripts/backend/check_vllm_rs.sh
# copy the printed:  export VLLM_RS=...

"$VLLM_RS" serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --grpc-port 50051 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

Then the router (gRPC, token_ids):

```bash
export VLLM_ROUTER_MODEL="$MODEL"
vllm-router \
  --host 127.0.0.1 \
  --port 30000 \
  --worker-urls grpc://127.0.0.1:50051
```

`--port` is still OpenAI HTTP (unused on the `grpc://` path).
`--grpc-port` is `Inference.GenerateStream` plus `grpc.health.v1`
(the router Checks empty service = overall `SERVING`).

Worker flags we actually tune in tests (forwarded to EngineCore):

| Flag | Why |
|---|---|
| `--tensor-parallel-size` | Tensor-parallel width |
| `--max-model-len` | Must cover input + output tokens |
| `--max-num-seqs` | Scheduler cap; set **≥** client concurrency `C` |
| `--gpu-memory-utilization` | KV / weight budget |
| `--enable-prefix-caching` | Second request can KV-hit |

These are **not** in `benches/backend/bench_prefix_kvhit.py`. HTTP launchers
take them as extra args; gRPC needs `--` first (see below).

## Wrapper scripts

| Script | Worker | Router URL |
|---|---|---|
| `serve_python_http.sh` | `VLLM_USE_RUST_FRONTEND=0 vllm serve` | `http://` reverse proxy |
| `serve_rust_http.sh` | `VLLM_USE_RUST_FRONTEND=1 vllm serve` → `vllm-rs frontend` | `http://` reverse proxy |
| `serve_rust_grpc.sh` | `vllm-rs serve --grpc-port` | `grpc://` GenerateStream |

```bash
export MODEL=/path/or/hf-id

./scripts/backend/serve_python_http.sh \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --enable-prefix-caching

./scripts/backend/serve_rust_http.sh \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --enable-prefix-caching

# engine flags after --
./scripts/backend/serve_rust_grpc.sh -- \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

## `benches/backend/bench_prefix_kvhit.py` benchmark client

Client only: posts to `/v1/chat/completions` on an already-running router.
By default it runs one concurrent wave. Pass `--warmup` to first send a serial
miss/hit pair that primes prefix-cache state before the wave. Defaults:
`--hit-rate 0.99`, `--chars 8000`, `--max-tokens 16`,
`--concurrency 1`, and `--requests 2 * concurrency`.
`tests/test_bench_prefix_kvhit.py` is the unit test for this client script; it
does not test live prefix-cache behavior.

| What | Flag | Env | Meaning |
|---|---|---|---|
| Shared-prefix fraction | `--hit-rate` | `HIT_RATE` | `0.99` same body; `0.30` shared prefix + unique tail; `0.00` unique leading tag |
| Body size (no tokenizer) | `--chars` | `KVHIT_CHARS` | synthetic user text length |
| Exact input tokens | `--tokens` + `--model-dir` | `MODEL_DIR` | sizes via `transformers` chat template; `--tokens` requires a model dir |
| Decode length | `--max-tokens` | | output tokens (`max_tokens` in the JSON) |
| Wave concurrency | `--concurrency` | | simultaneous requests per batch |
| Wave request count | `--requests` | | total requests in the wave; use `2 * C` for two batches at concurrency `C` |
| Cache warmup | `--warmup` | | serial miss/hit before the concurrent wave |
| Router | `--router-url` | `ROUTER_URL` | default `http://127.0.0.1:30000` |
| Served name | `--model` | `MODEL` | request `model` field |
| HTTP timeout | `--timeout` | | seconds (raise for long ISL) |

```bash
python benches/backend/bench_prefix_kvhit.py \
  --router-url http://127.0.0.1:30000 \
  --model /path/or/hf-id \
  --hit-rate 0.99 \
  --tokens 131072 \
  --model-dir /path/or/hf-id \
  --max-tokens 512 \
  --warmup \
  --concurrency 8 \
  --requests 16 \
  --timeout 1800

python benches/backend/bench_prefix_kvhit.py \
  --router-url http://127.0.0.1:30000 \
  --model /path/or/hf-id \
  --hit-rate 0.00 \
  --tokens 131072 \
  --model-dir /path/or/hf-id \
  --max-tokens 512 \
  --warmup \
  --concurrency 8 \
  --requests 16 \
  --timeout 1800
```

`--max-num-seqs` lives on the **worker**, not here. For a concurrency-`C`
wave, start the worker with `--max-num-seqs >= C`.

Output is newline-delimited JSON: one metadata object, then the result object.
The result has optional `warmup.miss` / `warmup.hit` rows and a `wave` object:

```json
{
  "wave": {
    "concurrency": 8,
    "requests": 16,
    "summary": {"ttft_ms": {"min": 0, "avg": 0, "max": 0}},
    "requests_detail": [
      {
        "request_index": 0,
        "batch_index": 0,
        "ttft_ms": 0,
        "e2e_ms": 0,
        "tpot_ms": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "stages": "{}"
      }
    ]
  }
}
```

`requests_detail` contains one row per wave request. Use `summary` for a quick
read, and `requests_detail` for p50/p90/p99, per-batch behavior, token counts,
or router stage diagnostics. With `VLLM_ROUTER_STAGES=1`, `stages` may include
`frontend_ms`, `xfer_ms`, and `engine_ms`; these are diagnostic
boundary/residual clocks, not direct EngineCore or network telemetry.

For 0% hit-rate waves, the client generates a unique leading tag for every
wave request. This avoids accidentally measuring a warmed exact-prefix body.

## `e2e_chat_correctness.py` live observation

Client only: observes one short streaming chat request against one or more
already-running routers. It checks HTTP/SSE completion, usage fields, and
non-empty text, then prints generated text and SHA-256 hashes for manual
inspection. Golden text is intentionally not required by default because model,
parallelism, and kernel choices can produce small output differences. Use
`--expect-substring` or `--require-exact-match` only when a stricter focused
check is useful.

```bash
python scripts/backend/e2e_chat_correctness.py \
  --case rust_grpc=http://127.0.0.1:13002 \
  --case rust_http=http://127.0.0.1:13001 \
  --case python_http=http://127.0.0.1:13003 \
  --model /path/or/hf-id \
  --max-tokens 20
```

Example result from a live Qwen run: all three paths produced the same text and
the same SHA-256 hash:

```text
rust_grpc   93062e9c6fa82775bb048b9b00fb2f8d1e16949a2e9fecefa83ff877f83e55ea
rust_http   93062e9c6fa82775bb048b9b00fb2f8d1e16949a2e9fecefa83ff877f83e55ea
python_http 93062e9c6fa82775bb048b9b00fb2f8d1e16949a2e9fecefa83ff877f83e55ea
```

## If the wheel has no `vllm-rs`

Clone vLLM **outside** this repo. Same release line as the installed
EngineCore is enough (exact tag not required). `./build_rust.sh` →
`tools/build_rust.py` and `cargo build -p vllm-cmd --release` are the
same crate; the script installs into `vllm/vllm-rs` and stamps the
version. Then:

```bash
export VLLM_RS=/path/to/checkout/vllm/vllm-rs
./scripts/backend/check_vllm_rs.sh
```

Do not Cargo-depend on `vllm-rs` in the router.

## Tests (no live worker)

These live under `tests/` and `src/backend/`, not in this directory.
They lock this version’s HTTP=`messages` / gRPC=`token_ids` wire; they
are not a forever ban on a gRPC text prompt.

```bash
# always-on, in-process mock (no GPU, no vllm-rs)
cargo test --test grpc_vs_http_e2e
cargo test --lib backend::

# optional: rust vllm-chat ids vs Python vLLM (skips if unset)
export VLLM_ROUTER_MODEL=/path/to/hf-model   # tokenizer.json required
cargo test --lib vllm_chat_tokenize_matches_python_vllm -- --nocapture
```

`tests/python_vllm_chat_ids.py` is a helper spawned by the preprocess
tests (`tests/python_vllm_chat_ids.py MODEL_DIR MESSAGES_JSON`).
`tests/common/mock_vllm_rs.rs` is the in-process `Inference` server
used by `grpc_vs_http_e2e`. See the router README **Tests** section.
