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

## Hybrid layouts

For regular HTTP routing, `--kv-hybrid-layout` points to a JSON object keyed by
worker URL. The matching `--kv-event-source` and `--kv-event-replay-source`
entries list publisher and replay endpoints in DP-rank order. The Router checks
them against the worker's existing `/kv_event_sources` response.

Each layout must describe **every** cache group and rank for the exact model and
engine configuration. The deployment must assign a new nonempty topic to each
publisher on every worker restart and update this configuration before routing
traffic. Reusing a topic across restarts does not prove the publisher identity.
The hash root, block sizes and group configuration must match the engine; a
guess can produce incorrect routing estimates. A configured hybrid worker with
an unsupported layout or mismatched sources receives no hybrid credit. Do not
use the legacy full-attention index for a hybrid worker without a verified
layout: partial events alone cannot prove that all groups are present.

The following illustrates the JSON shape; its block sizes and groups are not a
model configuration to copy:

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

The hybrid adapter currently rejects sliding-window groups. Experimental P/D
configuration is supplied with `--pd-hybrid-config`: its JSON maps each
prefill base URL to `endpoints`, `replay_endpoints`, `block_size`, `model`,
`hybrid` (the layout above), and `compatible_decode_urls`. Each listed pair
must be verified by the operator for model/weights, engine version, connector,
and parallel configuration. Once enabled, only explicitly listed P/D pairs
are admitted. Program scheduling is not supported with hybrid P/D. The
connector remains responsible for KV transfer; end-to-end validation is in
progress.

The Router does not load Completion tokenizer assets or validate IDs against a
local model vocabulary. The selected Worker retains input validation and serving
errors. Per-worker preparation may use render/tokenize RPCs; direct ID evidence
needs no RPC. Failed preparation removes cache evidence, not the original body
or ordinary forwarding path.

Observed placement estimates are not actual APC hit counts or complete
bootstrap/generation/recovery guarantees. See
[the HTTP consumer RFC](https://github.com/vllm-project/router/issues/405) for the
real-Worker acceptance requirements.
