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

## Hybrid attention models

Models such as Qwen3.5 and Qwen4Exp cache more than attention KV. To route
them by prefix, pass `--kv-hybrid-layout` a JSON file keyed by worker URL that
describes each rank's cache groups. Start vLLM with
`--prefix-caching-hash sha256_cbor` and `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0`.
Event endpoints may be omitted; the Router then asks the worker for them.

Copy every value from the running engine. A layout that does not match it gives
that worker no cache credit, and the worker is still routed by load. Give each
publisher a new topic on every worker restart. The example below shows the
format only:

```json
{
  "http://127.0.0.1:8000": {
    "schema_version": 1,
    "ranks": [{
      "rank": 0, "model": "example-model", "weight_version": "revision",
      "publisher_topic": "worker-boot-id/rank-0", "instance_topic": true,
      "hash_algorithm": "sha256_cbor", "full_hashes": true,
      "root_hash": "4e1195df020de59e0d65a33a4279f1183e7ae4e5d980e309f8b55adff2e61c3e",
      "hash_block_size": 16, "hash_lookahead_tokens": 0,
      "speculative_tokens": 0, "prefix_caching": true,
      "groups": [
        {"group_idx": 0, "kind": "mamba", "spec_type": "MambaSpec",
         "block_size": 16, "prefix_cacheable": true, "prefix_replay_tokens": 0,
         "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 16},
        {"group_idx": 1, "kind": "full_attention", "spec_type": "FullAttentionSpec",
         "block_size": 16, "prefix_cacheable": true, "prefix_replay_tokens": 0}
      ]
    }]
  }
}
```

For P/D (experimental), pass `--pd-hybrid-config` a JSON file keyed by prefill
URL. Each entry holds `endpoints`, `replay_endpoints`, `block_size`, `model`,
`hybrid` (the layout above) and `compatible_decode_urls`, the decode workers
that prefill worker may hand off to. Only the listed pairs are used. Program
scheduling is not supported in this mode.

The Router does not load Completion tokenizer assets or validate IDs against a
local model vocabulary. The selected Worker retains input validation and serving
errors. Per-worker preparation may use render/tokenize RPCs; direct ID evidence
needs no RPC. Failed preparation removes cache evidence, not the original body
or ordinary forwarding path.

Observed placement estimates are not actual APC hit counts or complete
bootstrap/generation/recovery guarantees. See
[the HTTP consumer RFC](https://github.com/vllm-project/router/issues/405) for the
real-Worker acceptance requirements.
