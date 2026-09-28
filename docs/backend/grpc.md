# gRPC Worker Backend

`grpc://` workers speak vLLM's Rust `Inference.GenerateStream` API through
the crates.io [`vllm-proto`](https://crates.io/crates/vllm-proto) bindings.
HTTP `http://` workers stay a transparent reverse proxy of OpenAI requests.

The current gRPC wire sends router-produced `token_ids`:

1. The router receives HTTP `/v1/chat/completions`.
2. `vllm-chat` and `vllm-tokenizer` render and tokenize the chat request.
3. The router selects a worker.
4. The router sends `GenerateRequest { prompt: TokenIds, ... }`.
5. Returned token IDs/text are adapted back into OpenAI JSON or SSE.

This is the current backend contract, not a policy that gRPC can never support
text prompts or richer payloads.

## Pool Behavior

A worker pool is all HTTP or all gRPC. Mixed schemes fail during startup or
`add_worker`. `grpc://host:port` is rewritten to h2c `http://host:port` for
tonic, but still calls the gRPC port, not the OpenAI HTTP port.

Worker liveness on `grpc://` uses `grpc.health.v1` Check with an empty service
name. `--health-check-endpoint` only applies to HTTP workers.

## Request Preparation

`TokenizerCache` in `src/backend/preprocess.rs` caches loaded `vllm-chat` and
`vllm-tokenizer` objects per model key. The load key is a valid local
`VLLM_ROUTER_TOKENIZER`, then `VLLM_ROUTER_MODEL`, then the request `model`.
The request `model` remains the protobuf model name sent to the worker.

The lowering in `src/backend/vllm_frontend.rs` tracks vLLM 0.29's private
`prepare_chat_request` conversion. Replace the local adapter if vLLM exposes a
public frontend-only lowering API.

Omitted temperature is resolved from the loaded model's `generation_config.json`
and then falls back to OpenAI's `1.0`; protobuf omission is not used because
vLLM 0.29 gRPC rewrites it to greedy `0.0`. When hidden stop strings/tokens are
configured, the router requests worker text so exact character trimming is
preserved.

## Capability Boundaries

Reasoning/tool history, tool definitions, tool choice, template kwargs,
documents, and response format are preserved while rendering.

Active tool calling and `reasoning_effort` currently return 400 on the gRPC
path. vLLM HTTP works because its OpenAI server owns the model-specific
tool-call parser and response shaping. The current vLLM gRPC/proto path does
not expose structured tool-call output from `GenerateStream`, so returning raw
text as `content` would silently violate OpenAI tool/reasoning semantics.
Full support needs vLLM gRPC/proto support for structured tool-call output, or
an equivalent parser/response adapter in this router.

Multimodal content also currently returns 400 on the gRPC path. The proto has
`GenerateRequest.media`, but the router does not yet populate it from OpenAI
chat content. Supporting it requires mapping OpenAI image/media parts into the
proto media fields and adding parity tests against vLLM HTTP behavior.

## Backend Modules

| File | Role |
|---|---|
| `src/backend/mod.rs` | Shared backend exports and `pb` aliases for `vllm-proto` |
| `src/backend/detect.rs` | Scheme detection, pool-kind locking, DP rank stripping, tonic URI conversion |
| `src/backend/frontend.rs` | `EngineFrontend`: `prepare(chat)` and `dispatch(ids, url)` |
| `src/backend/health.rs` | gRPC health checks for startup and worker probes |
| `src/backend/preprocess.rs` | Tokenizer/model cache and chat render/encode |
| `src/backend/vllm_frontend.rs` | Adapter around `vllm-chat`, `vllm-tokenizer`, and `vllm-text` |
| `src/backend/convert.rs` | OpenAI request fields to `GenerateRequest` |
| `src/backend/grpc.rs` | Cached tonic `InferenceClient` and `GenerateStream` |
| `src/backend/openai.rs` | Proto chunks to OpenAI JSON/SSE for clients |

`engine_ms` is a first-output residual diagnostic, not direct EngineCore
telemetry. Compare it only when the first-output boundary is equivalent across
paths.

## Starting vLLM 0.29

In vLLM 0.29, `vllm-rs` is a binary inside the Python wheel next to the
package. This router does not vendor or Cargo-depend on that binary.

Two entrypoints are easy to mix up:

| Start command | What runs | `--grpc-port` |
|---|---|---|
| `VLLM_USE_RUST_FRONTEND=1 vllm serve ...` | `vllm-rs frontend` for OpenAI HTTP | Not available |
| `vllm-rs serve MODEL --grpc-port ...` | Rust serve mode with HTTP and Inference gRPC | This is the `grpc://` worker |

Locate the wheel binary:

```bash
./scripts/backend/check_vllm_rs.sh
# then: export VLLM_RS=...
```

Start a gRPC worker and router:

```bash
"$VLLM_RS" serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --grpc-port 50051 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --enable-prefix-caching

vllm-router \
  --host 127.0.0.1 \
  --port 30000 \
  --worker-urls grpc://127.0.0.1:50051
```

If the wheel has no `vllm-rs`, build it in a separate vLLM checkout and set
`VLLM_RS=/path/to/vllm-rs`. Prefer the same vLLM release line as the installed
Python EngineCore package.

Launch helpers for python HTTP, rust HTTP, and rust gRPC are in
[`scripts/backend`](../../scripts/backend). The prefix-KV benchmark client is
[`benches/backend/bench_prefix_kvhit.py`](../../benches/backend/bench_prefix_kvhit.py).

## Tests

Always-on tests:

```bash
cargo test --test grpc_vs_http_e2e
cargo test --lib backend::
python3 -m unittest tests/test_bench_prefix_kvhit.py
```

Optional tokenizer parity check, skipped unless the model dir and Python vLLM
import are available:

```bash
export VLLM_ROUTER_MODEL=/path/to/hf-model
cargo test --lib vllm_chat_tokenize_matches_python_vllm -- --nocapture
```
