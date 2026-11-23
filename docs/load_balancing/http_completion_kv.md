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

For compatibility versus opt-in strict event recovery, see
[`--kv-event-strict-recovery`](../cli_reference.md#workers-and-backend).

### Optional prefix history

Add `--kv-aware-completion-history` to a Regular `--policy kv_aware --router-temperature 0` invocation (off by default). With no usable session binding or positive event credit, a recently observed whole-block token prefix can break a load tie. A hint expires about 60 seconds after the last successful observation. This is a preference, not evidence that KV is still resident.

It requires a registered single-rank worker with a verified model, block size and KV source, and one eligible prompt with backend-effective IDs. HTTP workers support single-prompt Completion and Chat through the existing input preparation; gRPC workers support token-ID Completion. History is recorded only after verified generated output; HTTP echo or compressed responses are forwarded without training history. Responses, native generation and gRPC Chat keep ordinary routing without history.
