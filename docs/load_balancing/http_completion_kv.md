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

## Reading KV prediction metrics

For HTTP workers, the predicted and actual cached-token histograms count the
same attempts. The prediction counts the device prefix plus the longest indexed
continuation that touches its end, counting overlapping tokens once. This
estimates vLLM's total local and external cached tokens. Transfer discounts
apply only to routing costs. Use the outcomes counter to check how much
traffic contributed to the comparison.

Streaming clients should set `stream_options.include_usage=true` and the engine
must return cached-token details. A completed stream without those details
counts as `missing_usage`. The Router forwards the client's stream options.
Multi-prompt comparisons require aggregate usage. Regular HTTP Program requests
can be compared when the bound target is the selected worker handling the
request. If attribution is uncertain the outcome is `pd_usage_not_comparable`.

Native HTTP P/D admission can carry a verified single-prompt total-prefix
prediction for the selected prefill worker and rank. The Router compares it
with the cached-token count in the merged JSON response or rewritten SSE usage
when that count agrees with the same attempt's prefill report. Missing prefill
usage never falls back to decode usage. Opaque responses do not supply accuracy
samples. Program-bound P/D paths that bypass KV prediction and gRPC workers do
not supply these pairs. The metric assumes the backend reports total cached tokens. A local-only
backend counter has different semantics.

JSON and SSE share the same attempt receipt; a failed dispatch, cancellation,
or response observation can finalize it only once. Multi-prompt predictions
carry no aggregate cached-token count and cannot enter the accuracy histograms.
Conflicting SSE counts are `invalid_usage`, rather than a last-value-wins sample.

All colocated non-streaming requests with KV observation use the complete-body
buffering path, including Program requests. Observation is limited to 1 MiB;
a larger body is still returned intact but records `oversized_json`. This is
an observation budget, not a limit on the complete response buffer. SSE uses
one framer with the same 1 MiB budget for both ordinary observation and P/D
reconciliation. Ordinary observation forwards original fragments immediately;
P/D reconciliation buffers one frame to merge usage. Oversized or incomplete
frames do not contribute accuracy samples. Transport chunks and serialized
P/D frames occupy additional memory beyond the framing accumulator.
