# Hybrid KV layout JSON

Enable prefix caching and start a ZMQ live and replay publisher on each prefill worker. The example uses Qwen3.5-4B with a 16-token hash unit.

```bash
env -u PYTHONHASHSEED VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 VLLM_SERVER_DEV_MODE=1 \
  vllm serve /models/Qwen3.5-4B --served-model-name qwen35 --host 127.0.0.1 --port 8000 \
  --max-model-len 4096 \
  --enable-prefix-caching --prefix-caching-hash-algo sha256_cbor \
  --mamba-cache-mode align --prefix-match-unit 16 \
  --kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://127.0.0.1:5557","replay_endpoint":"tcp://127.0.0.1:5558","topic":"qwen35-8000-start01","buffer_steps":30000}'
```

Use addresses the Router can reach. Give every worker distinct ports and every publisher launch a fresh topic. For a gRPC worker, use the [gRPC serve command](../backend/grpc.md) with the same vLLM options.

## Colocated layout

Save as `layout.json` and pass `--kv-hybrid-layout layout.json` to the [colocated Router](http_completion_kv.md#colocated-prefill-and-decode).

```json
{
  "http://127.0.0.1:8000": {
    "schema_version": 1,
    "ranks": [
      {
        "rank": 0,
        "model": "qwen35",
        "weight_version": "default",
        "publisher_topic": "qwen35-8000-start01",
        "instance_topic": true,
        "hash_algorithm": "sha256_cbor",
        "full_hashes": true,
        "root_hash": "9bd96a485ad84efdafb72ee48a1d7a69bcead0f8f0433173941b276b9581eef0",
        "hash_block_size": 16,
        "partial_hash_hits": true,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 0,
        "prefix_caching": true,
        "declared_groups": 4,
        "groups": [
          {"group_idx": 0, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 576, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 1, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 576, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 2, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 576, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 3, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 576, "prefix_cacheable": true, "prefix_replay_tokens": 0}
        ]
      }
    ]
  }
}
```

| Field | What to enter |
| --- | --- |
| Worker URL | The URL passed to the Router, including `grpc://` for a gRPC worker. Add one entry per hybrid worker. |
| `ranks` | One entry per data-parallel rank, with its served model name, weight version and publisher topic. |
| `root_hash` | The worker's 64-digit hexadecimal hash root. The sample value applies to vLLM builds with the fixed `vllm-none-hash` seed, `sha256_cbor`, and no `PYTHONHASHSEED`. |
| `hash_block_size`, `partial_hash_hits` | The engine's hash unit in tokens and whether it publishes partial pages at that unit. |
| `groups`, `declared_groups` | The worker's groups in engine order, including non-cacheable groups, and their count. Copy each group's actual geometry and replay settings. |

For LoRA, add `"adapters": [{"name": "my-adapter", "path": "/models/my-adapter"}]` alongside `ranks`. Use the exact path passed in vLLM's `--lora-modules`.

## P/D layout

Start matching prefill and decode engines with the roles and connector in the [P/D guide](http_completion_kv.md#pd-routing). Set `VLLM_SSM_CONV_STATE_LAYOUT=DS` on both Qwen3.5 engines when using NIXL. Put the worker layout object from above (the object containing `schema_version` and `ranks`) under `hybrid` in the prefill entry.

```json
{
  "http://127.0.0.1:8000": {
    "endpoints": ["tcp://127.0.0.1:5557"],
    "replay_endpoints": ["tcp://127.0.0.1:5558"],
    "block_size": 16,
    "model": "qwen35",
    "hybrid": {},
    "compatible_decode_urls": ["http://127.0.0.1:8001"]
  }
}
```

Replace the empty `hybrid` object with the complete layout before starting the Router. List endpoints in rank order and use base URLs without `@rank`. Save as `pd-layout.json` and pass `--pd-hybrid-config pd-layout.json`.

## Layout examples by checkpoint

These values are for the named checkpoints on matching vLLM 0.33 builds. Confirm the group order, sizes and hash settings against your engine, especially after changing parallelism or cache options.

| Checkpoint | Hash unit / partial pages | Groups in rank order |
| --- | --- | --- |
| Qwen3.5-4B | 16 / true | Three Mamba groups and one full-attention group, all with 576-token pages. |
| Qwen3.6-27B | 16 / true | Same fields as Qwen3.5, with 832-token pages. |
| Qwen3.8-27B | 16 / true | Same fields as Qwen3.5, with 832-token pages. |
| Qwen4Exp FP8 QSA | 1600 / false | Four Mamba groups with 1600-token pages and checkpoint alignment 1600, a 4-token scratch group, then a 1600-token MLA group with `tokens_per_state: "4"`. |
| DeepSeek V4 Flash | 4 / false | Sliding MLA (page/window) 64/128, 64/128, MLA page 256 with `tokens_per_state: "4"`, then sliding MLA 4/8 and 8/128. |
| DeepSeek V4 Pro | 4 / false | Sliding MLA 64/128, 64/128, MLA page 256 with `tokens_per_state: "128"`, then sliding MLA 8/128 and 4/8. |
| DeepSeek V4.1 Flash | 128 / false | Five 32-token sliding MLA groups with window and replay both 128, an MLA page of 128 with `tokens_per_state: "2"`, and an 8-token scratch group. |

Use `prefix_cacheable: true` and `prefix_replay_tokens: 0` for cacheable groups.

- **Mamba** uses `mamba` / `MambaSpec` with `mamba_cache_mode: "align"`. Qwen3.5, 3.6 and 3.8 use `--mamba-cache-mode align --prefix-match-unit 16` and checkpoint alignment 64 in these examples.
- **MLA** uses `mla_attention` / `MLAAttentionSpec` with `cache_role: "sparse"`. **Sliding MLA** uses `sliding_window_mla` / `SlidingWindowMLASpec` with `sliding_window` in tokens.
- **Scratch** uses `scratch` / `CircularBufferSpec` with `prefix_cacheable: false`, `prefix_replay_tokens: 0`, and `tokens_per_state: "1"`. For the V4.1 example, also set `prefix_cacheable: false` and `prefix_replay_tokens: 128` on the five sliding groups.
