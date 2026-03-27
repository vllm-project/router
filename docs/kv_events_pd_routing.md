# KV Events-Aware P/D Disaggregated Routing

## Summary

This PR introduces real-time KV cache event ingestion from vLLM workers,
enabling **precise cache-aware routing** for Prefill/Decode disaggregated
inference.  Instead of approximating cache state from routing history
(the existing `cache_aware` policy), the router now subscribes to actual
`BlockStored` / `BlockRemoved` / `AllBlocksCleared` events published by
vLLM over ZMQ, maintains a global KV block index, and uses it to:

1. **Route requests to workers with the highest KV cache prefix hit** –
   minimizing redundant prefill computation.
2. **Intelligently decide whether to skip P/D disaggregation** – when the
   selected decode worker already caches enough of the prompt, the
   overhead of a separate prefill stage is avoided.
3. **Speculatively index** the predicted cache state right after a routing
   decision, preventing duplicate prefill work when multiple requests
   with overlapping prefixes arrive in quick succession.

## Architecture

```
vLLM Prefill Pod₁  ─┐                         ┌──────────────────────┐
vLLM Prefill Pod₂  ─┤  ZMQ PUB (kv events)    │   Router (this PR)   │
vLLM Decode  Pod₁  ─┤ ─────────────────────►   │                      │
vLLM Decode  Pod₂  ─┘                         │  ┌────────────────┐  │
                                               │  │ KVEventPool    │  │
    HTTP request ──────────────────────────►   │  │  (ZMQ SUB ×N)  │  │
                                               │  └───────┬────────┘  │
                                               │          │ mpsc      │
                                               │  ┌───────▼────────┐  │
                                               │  │ KVBlockIndex   │  │
                                               │  │ (DashMap)      │  │
                                               │  └───────┬────────┘  │
                                               │          │           │
                                               │  ┌───────▼────────┐  │
                                               │  │ PrefixScorer   │  │
                                               │  │ + PreciseCache │  │
                                               │  │   AwarePolicy  │  │
                                               │  └───────┬────────┘  │
                                               │          │           │
                                               │     Route to best    │
                                               │     worker (or P/D   │
                                               │     disagg bypass)   │
                                               └──────────────────────┘
```

## New Modules

### `src/kv_events/` — Event Ingestion

| File | Purpose |
|------|---------|
| `mod.rs` | Module root and re-exports |
| `decoder.rs` | Msgpack decoder compatible with vLLM's `msgspec` wire format |
| `subscriber.rs` | Per-worker ZMQ SUB thread with graceful shutdown |
| `pool.rs` | Manages all subscriptions; single-channel event aggregation |

### `src/kv_index/` — Block Index & Scoring

| File | Purpose |
|------|---------|
| `mod.rs` | Module root and re-exports |
| `block_hash.rs` | Token-to-block-hash generator (chained content hashing) |
| `index.rs` | `DashMap<u64, Vec<BlockLocation>>` with speculative insert |
| `scorer.rs` | Longest contiguous prefix match scorer |
| `updater.rs` | Async task consuming events and updating the index |

### `src/policies/precise_cache_aware.rs` — New Policy

Implements `LoadBalancingPolicy` using the real KV index instead of
the approximate radix tree.

## Configuration

```yaml
mode:
  type: vllm_prefill_decode
  prefill_urls: [...]
  decode_urls: [...]
  kv_events:
    enabled: true
    topic_filter: "kv@"
    default_port: 5556
    index_max_entries: 100000000
    pd_uncached_token_threshold: 256
  prefill_policy:
    type: precise_cache_aware
    block_size: 16
    hash_seed: 12345
    enable_speculative: true
    speculative_ttl_ms: 2000
  decode_policy:
    type: precise_cache_aware
    block_size: 16
    hash_seed: 12345
```

### Critical Alignment Parameters

| Router Config | Must Match vLLM Config | Default |
|--------------|----------------------|---------|
| `block_size` | `--block-size` | 16 |
| `hash_seed` | `PYTHONHASHSEED` env | 0 |
| `topic_filter` | `--kv-events-config.topic` prefix | `kv@` |
| `default_port` | `--kv-events-config.endpoint` port | 5556 |

## P/D Disaggregation Bypass Logic

When KV events are enabled, the VllmPDRouter gains the ability to skip
the two-stage prefill/decode flow:

```
1. Select decode worker (highest prefix score)
2. Compute uncached_tokens = total_tokens - cached_blocks × block_size
3. If uncached_tokens ≤ pd_uncached_token_threshold:
     → Send directly to decode worker (skip prefill disaggregation)
4. Else:
     → Select prefill worker (also by prefix score)
     → Execute normal two-stage P/D flow
```

## Comparison with Existing `cache_aware` Policy

| Aspect | `cache_aware` (existing) | `precise_cache_aware` (this PR) |
|--------|-------------------------|--------------------------------|
| Data source | Router's own routing history (approximate radix tree) | Real KV events from vLLM engines |
| Accuracy | Approximate (may drift) | Near-exact (sub-second lag) |
| Tokenization | None (uses raw text chars) | Full HuggingFace tokenizer |
| Block-level matching | No (character-level prefix) | Yes (content-hashed blocks) |
| P/D bypass decision | Not supported | Supported via uncached token threshold |
| External dependency | None | vLLM KV events + ZMQ |

## Testing

- Unit tests for msgpack decoding (`decoder.rs`)
- Unit tests for block hash generation (`block_hash.rs`)
- Unit tests for index operations including speculative insert/expiry (`index.rs`)
- Unit tests for prefix scoring with gaps (`scorer.rs`)

## Future Work

- [ ] Wire `PreciseCacheAwarePolicy` into `VllmPDRouter::process_vllm_request`
      for end-to-end P/D bypass (requires refactoring the router init path)
- [ ] Integration tests with a mock vLLM ZMQ publisher
- [ ] Redis/Valkey-backed `KVBlockIndex` for multi-router deployments
- [ ] Metrics: cache hit ratio, speculative hit/miss, P/D bypass rate
- [ ] Bit-exact Python hash compatibility for block hashes
