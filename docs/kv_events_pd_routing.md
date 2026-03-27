# KV Events-Aware P/D Disaggregated Routing

## Summary

This feature introduces real-time KV cache event ingestion from vLLM workers,
enabling **KV-aware routing** for Prefill/Decode disaggregated
inference.  Instead of approximating cache state from routing history
(the existing `cache_aware` policy), the router subscribes to actual
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

## Quick Start

```bash
# vLLM PD mode with KV-aware routing (KV events enabled automatically)
vllm-router --vllm-pd-disaggregation \
  --vllm-discovery-address 0.0.0.0:30001 \
  --policy kv_aware \
  --kv-block-size 16 \
  --kv-hash-seed 12345 \
  --model-path Qwen/Qwen3-32B

# With separate policies for prefill and decode
vllm-router --vllm-pd-disaggregation \
  --vllm-discovery-address 0.0.0.0:30001 \
  --prefill-policy kv_aware \
  --decode-policy kv_aware \
  --kv-block-size 16 \
  --kv-hash-seed 12345 \
  --pd-uncached-token-threshold 256 \
  --model-path Qwen/Qwen3-32B

# With static prefill/decode workers
vllm-router --vllm-pd-disaggregation \
  --prefill http://10.0.0.1:8000 9001 \
  --prefill http://10.0.0.2:8000 9002 \
  --decode http://10.0.0.3:8000 \
  --decode http://10.0.0.4:8000 \
  --policy kv_aware \
  --kv-block-size 16 \
  --kv-hash-seed 12345
```

> **Note**: There is no `--kv-events-enabled` flag.  KV event ingestion is
> **automatically enabled** whenever `--policy`, `--prefill-policy`, or
> `--decode-policy` is set to `kv_aware`.

## CLI Parameters

### KV Events Configuration

KV event ingestion is **automatically enabled** when any policy is set to
`kv_aware`.  The following flags tune the event subsystem:

| Flag | Default | Description |
|------|---------|-------------|
| `--kv-events-topic-filter` | `kv@` | ZMQ topic prefix filter (must match vLLM `--kv-events-config` topic prefix) |
| `--kv-events-port` | `5556` | Default ZMQ port for KV event publishers on vLLM workers |
| `--kv-index-max-entries` | `100000000` | Maximum entries in the KV block index (advisory) |
| `--pd-uncached-token-threshold` | `256` | When the best decode worker's uncached tokens are below this value, skip prefill disaggregation |

### Precise Cache-Aware Policy Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--kv-block-size` | `16` | Tokens per KV block (must match vLLM `--block-size`) |
| `--kv-hash-seed` | `0` | Hash seed for block key computation (must match vLLM `PYTHONHASHSEED`) |
| `--kv-speculative-indexing` | `true` | Enable speculative index insertion after routing decisions |
| `--kv-speculative-ttl-ms` | `2000` | TTL in milliseconds for speculative index entries |

### Critical Alignment Requirements

| Router Flag | Must Match vLLM Setting | Example |
|------------|------------------------|---------|
| `--kv-block-size` | `--block-size` | `16` |
| `--kv-hash-seed` | `PYTHONHASHSEED` env var | `12345` |
| `--kv-events-topic-filter` | `--kv-events-config` topic prefix | `kv@` |
| `--kv-events-port` | `--kv-events-config` endpoint port | `5556` |

## Architecture

```
vLLM Prefill Pod₁  ─┐                         ┌──────────────────────┐
vLLM Prefill Pod₂  ─┤  ZMQ PUB (kv events)    │   Router             │
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
                                               │  │ + KvAware │  │
                                               │  │   AwarePolicy  │  │
                                               │  └───────┬────────┘  │
                                               │          │           │
                                               │     Route to best    │
                                               │     worker (or P/D   │
                                               │     disagg bypass)   │
                                               └──────────────────────┘
```

## P/D Disaggregation Bypass Logic

When KV events are enabled, the VllmPDRouter gains the ability to skip
the two-stage prefill/decode flow:

```
1. Select decode worker (highest prefix score)
2. Compute uncached_tokens = total_tokens - cached_blocks × block_size
3. If uncached_tokens ≤ --pd-uncached-token-threshold:
     → Send directly to decode worker (skip prefill disaggregation)
4. Else:
     → Select prefill worker (also by prefix score)
     → Execute normal two-stage P/D flow
```

## Comparison with Existing `cache_aware` Policy

| Aspect | `cache_aware` (existing) | `kv_aware` (this PR) |
|--------|-------------------------|--------------------------------|
| Data source | Router's own routing history (approximate radix tree) | Real KV events from vLLM engines |
| Accuracy | Approximate (may drift) | Near-exact (sub-second lag) |
| Tokenization | None (uses raw text chars) | Full HuggingFace tokenizer |
| Block-level matching | No (character-level prefix) | Yes (content-hashed blocks) |
| P/D bypass decision | Not supported | Supported via `--pd-uncached-token-threshold` |
| External dependency | None | vLLM KV events + ZMQ |

## vLLM Worker Configuration

Each vLLM worker must enable KV cache events:

```bash
# Example vLLM launch with KV events enabled
vllm serve Qwen/Qwen3-32B \
  --block-size 16 \
  --kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:5556","topic":"kv@${POD_IP}@Qwen/Qwen3-32B"}'

# Ensure PYTHONHASHSEED matches --kv-hash-seed on the router
export PYTHONHASHSEED=12345
```

## Policy Selection Guide (Updated)

```
 ┌─────────────────────┐
 │ Real-time KV cache  │
 │ state available?    │
 └─────────┬───────────┘
           │
   ┌───────┴──────────┐
   │                  │
  Yes                 No
   │                  │
   ▼                  ▼
 kv_aware  ┌─────────────────────┐
                      │ Need session        │
                      │ affinity?           │
                      └────────┬────────────┘
                               │
                      ┌────────┴────────┐
                     Yes               No
                      │                 │
                      ▼                 ▼
              consistent_hash     cache_aware / round_robin / ...
```
