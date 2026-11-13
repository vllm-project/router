# RFC: Full `/v1/completions` on the gRPC backend

- **Status:** Draft
- **Tracking PR:** [#563](https://github.com/vllm-project/router/pull/563) ships the
  token-id contract only. This RFC covers the follow-up that widens it to the
  vLLM v1 completions surface.
- **Prototype:** `KaisennHu/router@backup/grpc-completions-full` (three commits,
  ~1.8k lines incl. tests) implements everything below and was smoke-tested on
  an H20 node. It is not proposed for merge as one change.

## Summary

`/v1/completions` on a `grpc://` pool currently accepts one prompt form —
`prompt: list[int]` — and rejects text prompts, `n > 1`, `logprobs`, `echo`,
`json_schema`, batched prompts and `suffix` with `400`. This RFC proposes to
serve the vLLM v1 completions contract instead, in three independently
reviewable stages.

The wire protocol already carries almost everything needed. The work is in
lowering the OpenAI request shape onto `GenerateRequest`, and in rebuilding the
response path from "one prompt, one sequence" into "N sequences, each with its
own logprobs and finish reason".

## Motivation

The token-id contract was the minimum to unblock replay/benchmark harnesses that
send `prompt: [ids]`. Other completion clients cannot use gRPC P/D at all:

| Client behaviour | Today |
|---|---|
| Text prompts (OpenAI SDK, lm-eval, curl) | 400 |
| `logprobs` (perplexity, pass@k scoring, distillation) | 400 |
| `echo` (prompt scoring, joint probability) | 400 |
| `n > 1` (sampling diversity) | 400 |
| Batched prompts (throughput-oriented evaluation) | 400 |
| `json_schema` (structured extraction) | 400 |

## Non-goals

- Changing the chat path, HTTP pools, or the P/D two-leg protocol.
- Supporting features the engine cannot express (see
  [Engine constraints](#engine-constraints)).
- Making `suffix` work without engine-side FIM rendering.

## Design

### Request lowering

| OpenAI field | Wire mapping |
|---|---|
| `prompt: str` | router-side `vllm-tokenizer` encode (`add_special_tokens` default `true`) |
| `prompt: list[str]` / `list[list[int]]` | one `GenerateRequest` per prompt |
| `prompt: list[int]` | passthrough |
| `n` | one `GenerateRequest` per choice (see fan-out below) |
| `logprobs` / `top_logprobs` | `ResponseOptions.output_logprobs` + `output_candidates.top_n` |
| `prompt_logprobs` (or `echo` + `logprobs`) | `ResponseOptions.prompt_logprobs` + `prompt_candidates` |
| `logprob_token_ids` | `output_candidates` with `TokenIds` selection |
| `json_schema` | `DecodingParameters.structured_output.json` |
| `echo` | router-side: prefix the prompt text; shift `text_offset` |
| `max_tokens: 0` | router-side: ask for one token, drop it, answer with the prompt |
| `suffix` | not expressible (needs FIM rendering) |
| `best_of` | removed from the vLLM v1 completions API |

Text prompts are tokenized once, outside policy selection and retry, matching
how chat requests are prepared.

### Response path

The current adapter assumes one `SequenceOutput` stream. The proposal is a
per-index accumulator: text, token ids, `LogProbs`, finish reason, and the next
`text_offset` per sequence. `SequenceOutput.logprobs` / `ranks` /
`candidate_tokens` map onto OpenAI's `tokens`, `token_logprobs`, `top_logprobs`
and `text_offset`; `-inf` is clamped to `-9999` like the HTTP frontend.

`echo` merges `PromptInfo.logprobs` (when requested) in front of the output
entries. OpenAI's first prompt position has no preceding token, so it is
reported as `null`.

### Fan-out

The engine cannot produce multiple sequences today (see below), and one
`GenerateRequest` carries one prompt. Therefore:

- `n` fans out into `n` single-choice requests.
- Batched prompts fan out into one request per prompt.
- Both may combine: `prompts × n` requests.

The router merges the replies:

- Non-streaming: concatenate `choices` with sequential `index`, sum
  `completion_tokens`, and count `prompt_tokens` once per prompt.
- Streaming: merge SSE frames, rewrite `choices[].index` by
  `request_index × stride`, keep one `id`/`model`/`created`, swallow per-stream
  `[DONE]`, and emit one usage chunk plus one `[DONE]` at the end.

For `grpc://` P/D each fan-out unit is a full admission plus two-leg dispatch,
so a batched completion of `P` prompts with `n` choices costs `P × n`
admissions.

### Engine constraints

Measured against `vllm-rs` built from vLLM 0.28 + PR #63160 (`16f0960`):

- `rust/src/server/src/grpc/convert.rs` rejects `num_sequences > 1`
  (`"num_sequences > 1 is not supported"`) and hardcodes
  `SequenceOutput.index = 0`. Native multi-sequence therefore cannot be used
  until the engine supports it; fan-out is the portable path.
- `suffix` has no proto field, and vLLM only supports it for models that
  implement FIM rendering.
- `best_of` is absent from the vLLM v1 `CompletionRequest`; nothing to map.
- `echo` and `max_tokens: 0` are router-side transforms, no engine change.
- Explicit `stream + prompt_logprobs` is rejected by the HTTP frontend; the
  gRPC behaviour for streaming prompt logprobs still needs verification.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Use `RandomSampling.num_sequences` natively | Engine rejects it today; revisit when the engine lands multi-sequence support |
| Keep rejecting text prompts | Text is the default for OpenAI clients; tokenizing router-side is already implemented for chat |
| Serve text prompts via the proto `text` field | Router-side tokenization is needed anyway for admission scoring and `echo` |
| Route unsupported fields to the HTTP pool | Defeats the purpose of gRPC P/D pools, which have no HTTP endpoint |

## Operational impact

- **Extra dispatches.** `n` and batched prompts multiply engine requests. This
  is the price of the current engine limitation, not a design preference.
- **Response path cost.** Per-token logprobs require one detokenization per
  token on the router.
- **Usage semantics.** Prompt tokens are counted once per prompt even when `n`
  fans out; completion tokens accumulate across choices.
- **No new state on the hot path** when `n = 1`, a single prompt, and no
  logprobs/echo: the token-id behaviour is unchanged.

## Test plan

- Unit: prompt lowering (four forms), options mapping, logprobs assembly
  (offsets, top-k, clamping), merge accounting (indices, usage normalization).
- Integration (`tests/rust/grpc_pd_routing.rs`): text prompt tokenization,
  batched fan-out, `n` fan-out, stream merge.
- Real engine: text, `n=2`, `logprobs`, `echo`, `echo + max_tokens: 0`, batched
  prompts, streaming single and batched — 8/8 passed on H20 with the prototype.
- Still missing: P/D disaggregation with real 1P+1D engines (NIXL), `json_schema`,
  `logprob_token_ids`, `return_token_ids`, streaming prompt logprobs.

## Rollout

Three stacked PRs, each shippable on its own:

1. **Text prompts** — encode-only path in the frontend, `prompt: str`/`list[str]`
   lowering. Smallest user-visible win.
2. **Logprobs + echo** — per-sequence response state, logprob mapping, echo and
   `max_tokens: 0`.
3. **Batch and `n` fan-out** — the merge layer and router wiring.

`#563` (token-id only) can merge first and unblocks the original replay/benchmark
use case.

## Open questions

- Should the router use `num_sequences` when the engine advertises support, or
  always fan out? (Capability probing would need an engine-side signal.)
- Is the extra `P × n` admissions for P/D acceptable, or should batched
  completions stay HTTP-only until the engine supports multi-sequence?
- Do we need streaming prompt logprobs at all, and does the engine emit
  `PromptInfo.logprobs` while streaming?
- Should the wider surface be gated behind a router flag during rollout?
