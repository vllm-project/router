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

## Prefill progress

Enable worker-reported prefill progress for more accurate `kv_aware` load estimates:

```bash
vllm-router --worker-urls grpc://worker:50051 \
  --policy kv_aware --prefill-progress-stale-after-secs 5
```

- Supported for regular gRPC pools; P/D mode is not supported.
- Disabled by default. Workers without progress support continue using estimated load.
- Set the stale threshold to at least 2 seconds and below `--request-timeout-secs`.
  This is a freshness threshold, not the worker's reporting interval.

## API Compatibility

Support status of the OpenAI HTTP surface when the worker pool is `grpc://`:

| API | Status | Behavior |
|---|---|---|
| `/v1/chat/completions` | Supported | The router renders the chat template and tokenizes, then sends `TokenIds`. |
| `/v1/completions` | Supported, token-id prompts only | `prompt` must be `list[int]`; ids go straight to the wire. Text prompts and fields the chat shape cannot express (`suffix`, `best_of`, `n > 1`, `json_schema`, `logprobs`, `echo`, `return_hidden_states`, `documents`) are rejected with `400`. |
| `/v1/responses`, `/v1/embeddings`, `/v1/rerank`, `/generate`, `/inference/v1/generate` | Not supported | Rejected with `400` before dispatch, so they are client errors and never reach a worker. |
| `/v1/models`, `/health` | Supported | Answered through the gRPC control/health services. |
| `/health_generate` | Not supported | Returns `501`; use `/health` for gRPC readiness. |

Chat-level gaps (tool calling, `reasoning_effort`, multimodal content) are
listed under [Capability Boundaries](#capability-boundaries).

### `/v1/completions` on a gRPC pool

A gRPC pool also serves `/v1/completions` with token-id prompts only:
`prompt: [int]` skips the chat template (step 2 above) and the client's ids go
straight to the worker. Text and nested prompts return 400. Routing honors
`session_params.session_id` like chat; without one, token-id prompts create no
prefix-cache affinity (cache_aware balances them on load alone).

The reply is shaped for completion clients — `text_completion` chunks and body
with `choices[].text`, `cmpl-` ids — while reusing the chat pipeline internally.
`return_token_ids` gates whether the reply writes `choices[].token_ids`; chat
responses never carry them. Streaming emits every id even from frames whose
text is empty. The non-streaming choice also carries `prompt_token_ids` (the
client's ids) under the same flag; streaming never echoes prompt ids, matching
vLLM HTTP where that happens only under `echo`.

`suffix`, `best_of`, `n > 1`, `json_schema`, `logprobs`, `echo`,
`return_hidden_states`, `documents`, negative prompt ids, and unknown
passthrough keys return 400 instead of being dropped.

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

## Optional L0 encoding cache

L0 is off by default. Enable it for gRPC chat requests:

```bash
VLLM_ROUTER_L0_CACHE=1 vllm-router --worker-urls grpc://127.0.0.1:50051
```

| Environment variable | Default | Meaning |
|---|---:|---|
| `VLLM_ROUTER_L0_CACHE` | `0` | Enable with `1` or `true`; disable with `0` or `false` |
| `VLLM_ROUTER_L0_MAX_ENTRIES` | `10000` | Entry limit across models |
| `VLLM_ROUTER_L0_MAX_BYTES` | `67108864` | Estimated byte limit across models |
| `VLLM_ROUTER_L0_MAX_ENTRY_BYTES` | `1048576` | Per-entry byte limit |

Settings are read at startup for gRPC and initially empty worker pools.
Nonempty HTTP pools skip them. Limits must be positive, with the per-entry
limit no larger than the total. Byte estimates are not RSS bounds.

Keys include rendered text, tokenizer instance, `add_special_tokens` and
`cache_salt`. Only successful deterministic encodings are cached; unknown
configurations and `Prompt::TokenIds` bypass L0. Model assets must stay fixed
during loading; restart the router to reload them. Rendering and validation
still run per request, and retries reuse `PreparedRequest`.

L0 shares the existing LRU and `vllm_tokenizer_cache_*` metrics. Rust callers
can use `EngineFrontend::with_tokenizer_cache(timeout, config)`. See the
[benchmark guide](../benchmarks/grpc_frontend_l0.md) for measurement details.

## Optional L1 prefix encoding cache

L1 reuses token IDs for the longest previously seen prefix of a rendered
prompt, so a growing conversation only encodes its new turns. It caches
router-side tokenizer output and is unrelated to the worker's KV prefix cache
(`enable_prefix_caching`). It is off by default:

```bash
VLLM_ROUTER_L1_CACHE=1 vllm-router --worker-urls grpc://127.0.0.1:50051
```

| Environment variable | Default | Meaning |
|---|---:|---|
| `VLLM_ROUTER_L1_CACHE` | `0` | Enable with `1` or `true` |
| `VLLM_ROUTER_L1_MAX_BYTES` | `268435456` | Estimated heap byte limit across models (contents and containers) |
| `VLLM_ROUTER_L1_MAX_NODES` | `65536` | Tree node limit across models |
| `VLLM_ROUTER_L1_MAX_INSERT_BYTES` | `33554432` | Bytes one request may add (at most the total) |
| `VLLM_ROUTER_L1_MAX_INSERT_SEGMENTS` | `4096` | Boundary segments one request may add |
| `VLLM_ROUTER_L1_HASH_STRATEGY` | `first_last_64` | Child-key hashing strategy: `first_last_64` or `full` |
| `VLLM_ROUTER_L1_VERIFY` | `0` | Also run a full encode and compare (debugging) |

`first_last_64` is the default because it avoids hashing an entire long
segment on the common path. Use `VLLM_ROUTER_L1_HASH_STRATEGY=full` when a
workload is likely to create many long sibling segments with the same length,
same first 64 bytes, and same last 64 bytes, for example fixed-size templated
inputs whose middle changes but whose head and tail are identical.

How it works:

- **Boundaries.** Tokenizers split input on added tokens before normalizing
  and pre-tokenizing. A prefix that ends right after a *boundary* added token
  therefore encodes the same alone as inside the full prompt. Cut positions
  come from an automaton over all added tokens that matches like the
  tokenizer (leftmost-longest). Boundary tokens have no `lstrip`/`rstrip`/
  `single_word` flags and no edge whitespace, and no longer added token
  contains them before its end. If any added token is `single_word`, a
  boundary must not end in a Unicode `\w` character (letters, digits,
  combining marks, `_`), the same test Hugging Face uses to accept a
  `single_word` match. `normalized` tokens qualify only when the
  normalizer is a no-op and they cannot overlap non-normalized tokens
  (otherwise fastokens and Hugging Face split differently, and stricter rules
  apply). For Qwen these are `<|im_start|>`, `<|im_end|>`, `<tool_call>`,
  `<think>` and the other added tokens. At load, each candidate is probed by
  comparing split and full encodes; failures are dropped. Rejected tokens are
  logged with their reason at debug level.
- **Tree.** A radix tree over rendered text, one namespace per model instance
  and `cache_salt`. Edges end at boundaries and store their token IDs and the
  offsets of inner boundaries, so a lookup can stop inside an edge: editing
  message `k` still reuses messages before it. Text after the last boundary
  (usually the generation prompt) is not cached.
- **Encode.** IDs for the matched prefix are copied from the tree. The rest
  is encoded in one call, and its new boundary segments are inserted as one
  edge. Lookups take a read lock. Inserts take a write lock and evict
  least-recently-used leaves until both limits hold again. The byte estimate
  counts node contents plus the tree's container capacity.

L1 runs after L0 when both are enabled. It is skipped for
`add_special_tokens=true`, `Prompt::TokenIds`, tiktoken/tekken tokenizers, and
`tokenizer.json` files with BPE dropout, truncation, padding or Metaspace
`prepend_scheme: first`. With `VLLM_ROUTER_L1_VERIFY=1`, a mismatch logs an
error, returns the full encode and disables L1 for that model.

Metrics: `vllm_tokenizer_l1_{lookups,hits,cached_tokens,encoded_tokens,evictions,verify_mismatches}_total`
and the `vllm_tokenizer_l1_{nodes,bytes}` gauges.
`vllm_tokenizer_l1_bypasses_total{reason}` counts prompts on an L1-enabled
model that skipped the tree: `token_ids` (e.g. Harmony), `add_special_tokens`,
`no_boundary`, or `disabled` after a verify mismatch. A model that cannot use
L1 at all logs `L1 disabled for this model` with the reason at load. With an
HTTP worker pool the router does not tokenize, so `VLLM_ROUTER_L0_CACHE` and
`VLLM_ROUTER_L1_CACHE` are ignored with a warning at startup. With
`VLLM_ROUTER_STAGES=1`, `x-router-stages` reports `encode_cached_tokens`.
Rust callers can use `EngineFrontend::with_encoding_caches(timeout, l0, l1)`.
See the [L1 benchmark guide](../benchmarks/grpc_frontend_l1.md).

Future work: multimodal placeholders (this path is text-only), tiktoken and
tekken tokenizers, boundaries inside long messages, and HTTP worker pools.

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
| `src/backend/frontend.rs` | `EngineFrontend`: `prepare(chat)`, `prepare_token_ids(chat, ids)`, and `dispatch` |
| `src/backend/health.rs` | gRPC health checks for startup and worker probes |
| `src/backend/preprocess.rs` | Tokenizer/model cache and chat render/encode |
| `src/backend/vllm_frontend.rs` | Adapter around `vllm-chat`, `vllm-tokenizer`, and `vllm-text` |
| `src/backend/convert.rs` | OpenAI request fields to `GenerateRequest`; `prepare_grpc_completion` lowers `/v1/completions` onto the chat shape |
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
cargo test --features cpu,ready --test grpc_vs_http_e2e
cargo test --features cpu,ready --lib backend::
python3 -m unittest tests/python/test_bench_prefix_kvhit.py
```

Optional tokenizer parity check, skipped unless the model dir and Python vLLM
import are available:

```bash
export VLLM_ROUTER_MODEL=/path/to/hf-model
cargo test --features cpu,ready --lib vllm_chat_tokenize_matches_python_vllm -- --nocapture
```
