# Backend serve examples

vLLM 0.29 worker + this rust router. These scripts do not set Slurm,
Docker, or device IDs. Engine knobs (`--max-num-seqs`, …) are **worker
CLI flags**, not router flags. `bench_prefix_kvhit.py` does not start engines.

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

These are **not** in `bench_prefix_kvhit.py`. HTTP launchers take them as extra
args; gRPC needs `--` first (see below).

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

## `bench_prefix_kvhit.py` benchmark client

Client only: one miss POST then one hit POST to
`/v1/chat/completions`. Defaults: `--hit-rate 0.99`, `--chars 8000`,
`--max-tokens 16`.
`tests/test_bench_prefix_kvhit.py` is the unit test for this client script; it
does not test live prefix-cache behavior.

| What | Flag | Env | Meaning |
|---|---|---|---|
| Shared-prefix fraction | `--hit-rate` | `HIT_RATE` | `0.99` same body twice; `0.30` shared prefix + unique tail; `0.00` unique tag on the second request |
| Body size (no tokenizer) | `--chars` | `KVHIT_CHARS` | synthetic user text length |
| Exact input tokens | `--tokens` + `--model-dir` | `MODEL_DIR` | sizes via `transformers` chat template; `--tokens` requires a model dir |
| Decode length | `--max-tokens` | | output tokens (`max_tokens` in the JSON) |
| Router | `--router-url` | `ROUTER_URL` | default `http://127.0.0.1:30000` |
| Served name | `--model` | `MODEL` | request `model` field |
| HTTP timeout | `--timeout` | | seconds (raise for long ISL) |

```bash
export ROUTER_URL=http://127.0.0.1:30000
export MODEL=/path/or/hf-id

python scripts/backend/bench_prefix_kvhit.py \
  --hit-rate 0.99 \
  --chars 8000 \
  --max-tokens 16

python scripts/backend/bench_prefix_kvhit.py \
  --hit-rate 0.30 \
  --tokens 131072 \
  --model-dir "$MODEL" \
  --max-tokens 512 \
  --timeout 1800
```

`--max-num-seqs` lives on the **worker**, not here. For a concurrency-`C`
wave, start the worker with `--max-num-seqs` ≥ `C` and run `C` clients
yourself.

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
