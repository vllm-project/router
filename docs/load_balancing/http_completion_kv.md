# HTTP Completion inputs for KV-aware routing

Regular HTTP KV routing uses each Worker's existing input preparation contract.
Text is prepared by that Worker; explicit token IDs use the direct path, with
separate prefixes for separate prompts. Retries reuse prepared input and make a
fresh selection and reservation. Workers receive the validated ingress bytes.
Ordinary HTTP deployments do not need local tokenizer assets.

```sh
vllm-router --host 127.0.0.1 --port 19000 \
  --worker-urls http://127.0.0.1:8000 http://127.0.0.1:8001 \
  --policy kv_aware --intra-node-data-parallel-size 1
```

The Worker metadata/discovery contract supplies the event publishers and cache
identity. Explicit sources use `PolicyConfig.static_kv_event_sources`; this is
the existing configuration API, not a separate environment mapping. Unknown or
incompatible sources remain routing candidates without cache credit.

Worker input and identity contracts govern cache credit. Unknown extensions,
cache salt, adapters and other unmodeled identities remain routable without
cache credit; known input tokens still contribute their full input cost.
Session binding skips optional preparation RPCs, not known token accounting.

The Router does not load Completion tokenizer assets or validate IDs against a
local model vocabulary. The selected Worker retains input validation and serving
errors. Per-worker preparation may use render/tokenize RPCs; direct ID evidence
needs no RPC. Failed preparation removes cache evidence, not the original body
or ordinary forwarding path.

Observed placement estimates are not actual APC hit counts or complete
bootstrap/generation/recovery guarantees. See
[the HTTP consumer RFC](https://github.com/vllm-project/router/issues/405) for the
real-Worker acceptance requirements.

### Optional Completion history

`--kv-aware-completion-history` (off by default) can break a lowest-cost tie
between independently addressed Regular HTTP workers when `--router-temperature 0`
(the Router's selection temperature, not the request's sampling temperature).
A valid session binding wins first. If any retry-eligible target has positive
event credit, the usual scorer decides. Otherwise, among equally scored and
equally ranked targets, prefer the longest previously observed token prefix.
History never grants KV credit: a known 33-token request with no event credit
still reserves 33 tokens. Preparation uses the existing Direct or per-worker
tokens; enabling history sends no additional preparation RPC.

Only plain, single-prompt requests whose explicit model matches the registered
cache identity qualify. Successful upstream **headers**, not dispatch or body
completion, record the selected worker's prefix. A later body failure does not
undo that observation. An error or cancellation before headers records nothing.
Records expire after 60 seconds; each policy instance keeps the latest 128
observations, at most the first 256 token IDs each (32,768 retained IDs, not a
process memory limit). Replacement or removal fences the old registration.
KV Remove/Clear changes placement evidence, not the fact that the request was
routed. An unobserved same-address engine restart cannot be identified by this
local history. This option does not enable P/D, gRPC, IGW or Program history.

### Sequential Prefill history (P/D)

`--kv-aware-prefill-history` is a separate, default-off option for static,
independently addressed DP1 HTTP Prefill workers using `kv_aware` and sequential
NIXL pull. It accepts plain, single-prompt Completions for an explicitly matched
model. With zero **routing** temperature, no positive event credit, and equal
work and occupancy, the longest unexpired exact prefix can break the Prefill
tie. A hint does not reduce the reservation. Decode selection and affinity do
not use this history.

Prefill headers do not record a hint. The router consumes the bounded 2xx body
and checks the single Completion and its current attempt's NIXL pull metadata
(including request IDs, remote engine, endpoint, a nonempty block group, and
`remote_num_tokens` matching the entire prepared input, not just its first 256 IDs).
Only the selected, still-registered P can be observed. An invalid or unsupported
handoff does not train history or change ordinary forwarding. A later D failure
does not undo a valid P observation; a new attempt needs its own handoff.
This first slice covers one block group and TP1, not push, multi-rank, discovery,
EPD, Program, Chat or adapters. It shares the 60 s / 128 observations / first
256 IDs limits above; it does not establish residency or actual KV transfer.
