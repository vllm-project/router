# KV-aware routing

If requests often start with the same prompt, such as a shared system message, `kv_aware` prefers a worker that can reuse that prefix and also considers worker load. vLLM publishes cache events so the Router can track which workers have the prefix. When cache information is unavailable, the Router routes by load.

Clients send OpenAI HTTP requests to the Router with either HTTP or gRPC workers.

## Requirements for KV-aware prefix reuse

These settings apply to all workers scored by `kv_aware`, including workers with a layout file.

| Requirement | Worker setting |
| --- | --- |
| Prefix caching and matching block hashes | Set `--enable-prefix-caching --prefix-caching-hash-algo sha256_cbor`. |
| Complete hashes in events | Set `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0`. |
| Incremental events | Keep incremental reporting and enable KV events. Give each publisher a fresh topic on restart. Live and replay addresses must be reachable from the Router. |

## When to pass a layout

| Deployment | Layout JSON to pass to the Router |
| --- | --- |
| [Ordinary single-group full attention](#colocated-prefill-and-decode), HTTP or gRPC | Omit `--kv-hybrid-layout` when the worker uses the default hash root and exposes its model, block size, and event sources. [Stock HTTP vLLM 0.30.0](#static-event-endpoints) uses static configuration. |
| Hybrid or sliding-window attention, speculative decoding, or custom hash seed | Use a worker-reported layout when available. Otherwise pass an [explicit layout](#explicit-layouts) with `--kv-hybrid-layout`. |
| [P/D disaggregation](#pd-routing), HTTP or gRPC | Use the same `--kv-hybrid-layout` file for both stages that need an [explicit layout](#pd-layout). |

## Colocated prefill and decode

Each worker handles the entire request. The walkthrough uses one Qwen3-4B-Instruct-2507 worker, vLLM 0.33.0, and the Rust `vllm-router` binary. Check [version compatibility](#versions) for other releases.

### 1. Prepare the worker settings

Run this before each worker start. Set the live port and let vLLM allocate a replay port. The command generates a fresh publisher topic.

```bash
MODEL=/models/Qwen3-4B-Instruct-2507
SERVED_MODEL=qwen3
export VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0
unset PYTHONHASHSEED
KV_TOPIC=$(python3 -c 'from uuid import uuid4; print(uuid4().hex)')
KV_LIVE_PORT=5557
EVENTS=$(printf '{"enable_kv_cache_events":true,"endpoint":"tcp://*:%s","replay_endpoint":"tcp://*:0","topic":"%s"}' "$KV_LIVE_PORT" "$KV_TOPIC")
```

For multiple worker processes on one host, set a distinct `KV_LIVE_PORT` per process. Set a fixed `replay_endpoint` if the network requires a reserved port. vLLM adds the DP rank to each fixed event port, so reserve non-overlapping ranges for live and replay. These ports are separate from HTTP and gRPC.

On vLLM 0.30.0, use a fixed replay endpoint such as `tcp://*:5558` for one rank.

### 2. Start the worker and Router

Choose one transport. **Run these commands with no layout JSON** for one full-attention group with no sliding window or speculative decoding. Leave `PYTHONHASHSEED` unset to use the default hash root. The Router checks that root and the reported block size against a first-block event. The `16` below is the example block size. The same rule applies to ordinary full-attention workers in P/D.

#### HTTP workers

`--enable-scale-out` exposes the worker's rendering endpoints for text and chat requests. Requests that already contain token IDs can use KV-aware routing without this flag.

```bash
vllm serve "$MODEL" --host 127.0.0.1 --port 8000 --served-model-name "$SERVED_MODEL" \
  --enable-scale-out --enable-prefix-caching --prefix-caching-hash-algo sha256_cbor --block-size 16 \
  --kv-events-config "$EVENTS"
```

```bash
vllm-router --host 127.0.0.1 --port 19000 --policy kv_aware \
  --worker-urls http://127.0.0.1:8000
```

#### gRPC workers

Use [`vllm-rs` from the same vLLM version](../backend/grpc.md#starting-a-vllm-grpc-worker). Activate that vLLM Python environment before starting the worker.

```bash
"${VLLM_RS:-vllm-rs}" serve "$MODEL" --host 127.0.0.1 --port 8000 --grpc-port 50051 --served-model-name "$SERVED_MODEL" -- \
  --enable-prefix-caching --prefix-caching-hash-algo sha256_cbor \
  --block-size 16 --kv-events-config "$EVENTS"
```

In the Router terminal, set `MODEL` to a local copy of the worker's model directory with its tokenizer files.

```bash
VLLM_ROUTER_TOKENIZER="$MODEL" vllm-router --host 127.0.0.1 --port 19000 \
  --policy kv_aware --worker-urls grpc://127.0.0.1:50051
```

#### Static event endpoints

This setup is for stock HTTP vLLM 0.30.0, which has no event discovery endpoint. Set the worker's replay port to `5558`. Save the example below as `layout.json` and replace the topic with the worker's `KV_TOPIC` value.

```bash
--kv-event-source http://127.0.0.1:8000=tcp://127.0.0.1:5557 \
--kv-event-replay-source http://127.0.0.1:8000=tcp://127.0.0.1:5558 \
--kv-event-model "$SERVED_MODEL" --kv-event-block-size 16 \
--kv-hybrid-layout layout.json
```

<details>
<summary>Complete layout.json for stock HTTP vLLM 0.30.0</summary>

```json
{
  "http://127.0.0.1:8000": {
    "schema_version": 1,
    "ranks": [
      {
        "rank": 0,
        "model": "qwen3",
        "publisher_topic": "replace-with-KV_TOPIC",
        "weight_version": "default",
        "instance_topic": true,
        "hash_algorithm": "sha256_cbor",
        "hash_seed": "vllm-none-hash",
        "full_hashes": true,
        "hash_block_size": 16,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 0,
        "prefix_caching": true,
        "declared_groups": 1,
        "groups": [
          {"group_idx": 0, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 16, "prefix_cacheable": true, "prefix_replay_tokens": 0}
        ]
      }
    ]
  }
}
```

</details>

## Explicit layouts

Use a layout file for hybrid attention, sliding-window attention, speculative decoding, or a custom hash seed when the worker cannot report its complete layout. Choose a recipe below and pass it with `--kv-hybrid-layout layout.json`. Use the worker URL as the JSON key, such as `grpc://127.0.0.1:50051` for the gRPC example. Include one entry per worker URL and one rank record per DP rank.

The Router discovers the served model, publisher topic, and live and replay addresses. For [static event sources](#static-event-endpoints), also declare `model` and `publisher_topic` per rank. Use the group IDs, block sizes, and reuse rules of the running worker. A different checkpoint, engine version, or speculation mode can change the layout.

For a hybrid worker with static events, set `--kv-event-block-size` to the recipe's `hash_block_size`.

The GPU examples use the default `sha256_cbor` root with `PYTHONHASHSEED` unset. If workers use a fixed seed, set `hash_seed` to that string. For LoRA, add `"adapters": [{"name": "my-adapter", "path": "/models/my-adapter"}]` beside `ranks`, with the exact path passed to `--lora-modules`.

## Hybrid attention models

Use the [HTTP](#http-workers) or [gRPC](#grpc-workers) worker command with the chosen checkpoint and served model name. Remove `--block-size 16` and add the engine options shown for the recipe. For gRPC, put these options after `--`. The GPU recipes use vLLM 0.33.0 and a context length of 4,096 tokens.

| Model | Available recipes |
| --- | --- |
| [Qwen3.5-4B](#qwen35-4b) | Base, [Ascend MTP](#qwen35-4b-with-mtp-on-ascend) |
| [Qwen3.6-27B](#qwen36-27b) | Base |
| [Qwen3.8-27B](#qwen38-27b) | Base, [MTP](#qwen38-27b-with-mtp), [DSpark](#qwen38-27b-with-dspark), [Ascend variants](#ascend-models) |
| [Qwen4Exp FP8 QSA](#qwen4exp-fp8-qsa) | Base |
| [DeepSeek V4 Flash](#deepseek-v4-flash) | [Base](#deepseek-v4-flash), [DSpark](#deepseek-v4-flash-with-dspark) |
| [DeepSeek V4 Pro](#deepseek-v4-pro) | Base |
| [DeepSeek V4.1 Flash](#deepseek-v41-flash) | Bounded replay |

### Qwen3.5-4B

```bash
--max-model-len 4096 --prefix-match-unit 16
```

<details>
<summary>Qwen3.5-4B layout.json</summary>

```json
{
  "http://127.0.0.1:8000": {
    "schema_version": 1,
    "ranks": [
      {
        "rank": 0,
        "weight_version": "default",
        "instance_topic": true,
        "hash_algorithm": "sha256_cbor",
        "hash_seed": "vllm-none-hash",
        "full_hashes": true,
        "hash_block_size": 16,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 0,
        "prefix_caching": true,
        "partial_hash_hits": true,
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

</details>

For NIXL P/D, set `VLLM_SSM_CONV_STATE_LAYOUT=DS` on both stages.

### Qwen3.6-27B

```bash
--max-model-len 4096 --prefix-match-unit 16
```

<details>
<summary>Qwen3.6-27B layout.json</summary>

```json
{
  "http://127.0.0.1:8000": {
    "schema_version": 1,
    "ranks": [
      {
        "rank": 0,
        "weight_version": "default",
        "instance_topic": true,
        "hash_algorithm": "sha256_cbor",
        "hash_seed": "vllm-none-hash",
        "full_hashes": true,
        "hash_block_size": 16,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 0,
        "prefix_caching": true,
        "partial_hash_hits": true,
        "declared_groups": 4,
        "groups": [
          {"group_idx": 0, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 1, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 2, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 3, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0}
        ]
      }
    ]
  }
}
```

</details>

### Qwen3.8-27B

**Without speculative decoding**

```bash
--max-model-len 4096 --prefix-match-unit 16
```

<details>
<summary>Qwen3.8-27B layout.json</summary>

```json
{
  "http://127.0.0.1:8000": {
    "schema_version": 1,
    "ranks": [
      {
        "rank": 0,
        "weight_version": "default",
        "instance_topic": true,
        "hash_algorithm": "sha256_cbor",
        "hash_seed": "vllm-none-hash",
        "full_hashes": true,
        "hash_block_size": 16,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 0,
        "prefix_caching": true,
        "partial_hash_hits": true,
        "declared_groups": 4,
        "groups": [
          {"group_idx": 0, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 1, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 2, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 3, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0}
        ]
      }
    ]
  }
}
```

</details>

#### Qwen3.8-27B with MTP

Pick the variant that matches the running worker. A P/D connector can disable lookahead hashes, so check Prefill and Decode separately. For Ascend, use its [MTP recipe](#qwen38-27b-with-mtp-on-ascend).

**One token of hash lookahead**

```bash
--max-model-len 4096 --prefix-match-unit 16 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

<details>
<summary>Qwen3.8 MTP layout.json · one token of hash lookahead</summary>

```json
{
  "http://127.0.0.1:8000": {
    "schema_version": 1,
    "ranks": [
      {
        "rank": 0,
        "weight_version": "default",
        "instance_topic": true,
        "hash_algorithm": "sha256_cbor",
        "hash_seed": "vllm-none-hash",
        "full_hashes": true,
        "hash_block_size": 16,
        "hash_lookahead_tokens": 1,
        "speculative_tokens": 3,
        "prefix_caching": true,
        "partial_hash_hits": true,
        "declared_groups": 4,
        "groups": [
          {"group_idx": 0, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 1, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 2, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 3, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0}
        ]
      }
    ]
  }
}
```

</details>

**Zero hash lookahead with a group 3 tail drop**

```bash
--max-model-len 4096 --prefix-match-unit 16 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"disable_lookahead_block_hashes":true}'
```

<details>
<summary>Qwen3.8 MTP layout.json · zero hash lookahead, group 3 tail drop</summary>

```json
{
  "http://127.0.0.1:8000": {
    "schema_version": 1,
    "ranks": [
      {
        "rank": 0,
        "weight_version": "default",
        "instance_topic": true,
        "hash_algorithm": "sha256_cbor",
        "hash_seed": "vllm-none-hash",
        "full_hashes": true,
        "hash_block_size": 16,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 3,
        "prefix_caching": true,
        "partial_hash_hits": true,
        "declared_groups": 4,
        "groups": [
          {"group_idx": 0, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 1, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 2, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 3, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0}
        ],
        "tail_drop_groups": [
          3
        ]
      }
    ]
  }
}
```

</details>

#### Qwen3.8-27B with DSpark

Download the [Qwen3.8 DSpark draft checkpoint](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark) to `/models/Qwen3.8-27B-speculator.dspark` or set the actual path in the launch option.

```bash
--max-model-len 4096 --prefix-match-unit 16 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":8,"model":"/models/Qwen3.8-27B-speculator.dspark"}'
```

<details>
<summary>Qwen3.8 DSpark layout.json</summary>

```json
{
  "http://127.0.0.1:8000": {
    "schema_version": 1,
    "ranks": [
      {
        "rank": 0,
        "weight_version": "default",
        "instance_topic": true,
        "hash_algorithm": "sha256_cbor",
        "hash_seed": "vllm-none-hash",
        "full_hashes": true,
        "hash_block_size": 16,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 8,
        "prefix_caching": true,
        "partial_hash_hits": false,
        "tail_drop_groups": [],
        "declared_groups": 15,
        "groups": [
          {"group_idx": 0, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 1, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 2, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 3, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 4, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 5, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 6, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 7, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 8, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 9, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 10, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0},
          {"group_idx": 11, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0},
          {"group_idx": 12, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0},
          {"group_idx": 13, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0},
          {"group_idx": 14, "kind": "sliding_window", "spec_type": "SlidingWindowSpec", "block_size": 832, "prefix_cacheable": true, "prefix_replay_tokens": 0, "sliding_window": 2048}
        ]
      }
    ]
  }
}
```

</details>

### Qwen4Exp FP8 QSA

Use the `Qwen3.8-Flash-Next-FP8` checkpoint with architecture `Qwen4ExpForConditionalGeneration`. Use a prompt longer than 1,600 tokens to check reuse.

```bash
--max-model-len 4096
```

<details>
<summary>Qwen4Exp FP8 QSA layout.json</summary>

```json
{
  "http://127.0.0.1:8000": {
    "schema_version": 1,
    "ranks": [
      {
        "rank": 0,
        "weight_version": "default",
        "instance_topic": true,
        "hash_algorithm": "sha256_cbor",
        "hash_seed": "vllm-none-hash",
        "full_hashes": true,
        "hash_block_size": 1600,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 0,
        "prefix_caching": true,
        "partial_hash_hits": false,
        "declared_groups": 6,
        "groups": [
          {"group_idx": 0, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 1600, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 1600},
          {"group_idx": 1, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 1600, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 1600},
          {"group_idx": 2, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 1600, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 1600},
          {"group_idx": 3, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 1600, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 1600},
          {"group_idx": 4, "kind": "scratch", "spec_type": "CircularBufferSpec", "block_size": 4, "prefix_cacheable": false, "prefix_replay_tokens": 0, "tokens_per_state": "1"},
          {"group_idx": 5, "kind": "mla_attention", "spec_type": "MLAAttentionSpec", "block_size": 1600, "prefix_cacheable": true, "prefix_replay_tokens": 0, "tokens_per_state": "4", "cache_role": "sparse"}
        ]
      }
    ]
  }
}
```

</details>

### DeepSeek V4 Flash

Use the V4 Flash 0731 checkpoint.

```bash
--max-model-len 4096
```

<details>
<summary>DeepSeek V4 Flash layout.json</summary>

```json
{
  "http://127.0.0.1:8000": {
    "schema_version": 1,
    "ranks": [
      {
        "rank": 0,
        "weight_version": "default",
        "instance_topic": true,
        "hash_algorithm": "sha256_cbor",
        "hash_seed": "vllm-none-hash",
        "full_hashes": true,
        "hash_block_size": 4,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 0,
        "prefix_caching": true,
        "partial_hash_hits": false,
        "declared_groups": 5,
        "groups": [
          {"group_idx": 0, "kind": "sliding_window_mla", "spec_type": "SlidingWindowMLASpec", "block_size": 64, "prefix_cacheable": true, "prefix_replay_tokens": 0, "sliding_window": 128},
          {"group_idx": 1, "kind": "sliding_window_mla", "spec_type": "SlidingWindowMLASpec", "block_size": 64, "prefix_cacheable": true, "prefix_replay_tokens": 0, "sliding_window": 128},
          {"group_idx": 2, "kind": "mla_attention", "spec_type": "MLAAttentionSpec", "block_size": 256, "prefix_cacheable": true, "prefix_replay_tokens": 0, "tokens_per_state": "4", "cache_role": "sparse"},
          {"group_idx": 3, "kind": "sliding_window_mla", "spec_type": "SlidingWindowMLASpec", "block_size": 4, "prefix_cacheable": true, "prefix_replay_tokens": 0, "sliding_window": 8},
          {"group_idx": 4, "kind": "sliding_window_mla", "spec_type": "SlidingWindowMLASpec", "block_size": 8, "prefix_cacheable": true, "prefix_replay_tokens": 0, "sliding_window": 128}
        ]
      }
    ]
  }
}
```

</details>

#### DeepSeek V4 Flash with DSpark

Use the 0731 checkpoint with DSpark support. For P/D, use NIXL.

```bash
--max-model-len 4096 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":4}'
```

<details>
<summary>DeepSeek V4 Flash DSpark layout.json</summary>

```json
{
  "http://127.0.0.1:8000": {
    "schema_version": 1,
    "ranks": [
      {
        "rank": 0,
        "weight_version": "default",
        "instance_topic": true,
        "hash_algorithm": "sha256_cbor",
        "hash_seed": "vllm-none-hash",
        "full_hashes": true,
        "hash_block_size": 4,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 4,
        "prefix_caching": true,
        "partial_hash_hits": false,
        "declared_groups": 5,
        "groups": [
          {"group_idx": 0, "kind": "sliding_window_mla", "spec_type": "SlidingWindowMLASpec", "block_size": 64, "prefix_cacheable": true, "prefix_replay_tokens": 0, "sliding_window": 128},
          {"group_idx": 1, "kind": "sliding_window_mla", "spec_type": "SlidingWindowMLASpec", "block_size": 64, "prefix_cacheable": true, "prefix_replay_tokens": 0, "sliding_window": 128},
          {"group_idx": 2, "kind": "mla_attention", "spec_type": "MLAAttentionSpec", "block_size": 256, "prefix_cacheable": true, "prefix_replay_tokens": 0, "tokens_per_state": "4", "cache_role": "sparse"},
          {"group_idx": 3, "kind": "sliding_window_mla", "spec_type": "SlidingWindowMLASpec", "block_size": 4, "prefix_cacheable": true, "prefix_replay_tokens": 0, "sliding_window": 8},
          {"group_idx": 4, "kind": "sliding_window_mla", "spec_type": "SlidingWindowMLASpec", "block_size": 8, "prefix_cacheable": true, "prefix_replay_tokens": 0, "sliding_window": 128}
        ],
        "tail_drop_groups": []
      }
    ]
  }
}
```

</details>

### DeepSeek V4 Pro

The example uses four GPUs. Set tensor parallelism to fit your deployment and confirm the group layout if you change it.

```bash
--max-model-len 4096 --tensor-parallel-size 4
```

<details>
<summary>DeepSeek V4 Pro layout.json</summary>

```json
{
  "http://127.0.0.1:8000": {
    "schema_version": 1,
    "ranks": [
      {
        "rank": 0,
        "weight_version": "default",
        "instance_topic": true,
        "hash_algorithm": "sha256_cbor",
        "hash_seed": "vllm-none-hash",
        "full_hashes": true,
        "hash_block_size": 4,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 0,
        "prefix_caching": true,
        "partial_hash_hits": false,
        "declared_groups": 5,
        "groups": [
          {"group_idx": 0, "kind": "sliding_window_mla", "spec_type": "SlidingWindowMLASpec", "block_size": 64, "prefix_cacheable": true, "prefix_replay_tokens": 0, "sliding_window": 128},
          {"group_idx": 1, "kind": "sliding_window_mla", "spec_type": "SlidingWindowMLASpec", "block_size": 64, "prefix_cacheable": true, "prefix_replay_tokens": 0, "sliding_window": 128},
          {"group_idx": 2, "kind": "mla_attention", "spec_type": "MLAAttentionSpec", "block_size": 256, "prefix_cacheable": true, "prefix_replay_tokens": 0, "tokens_per_state": "128", "cache_role": "sparse"},
          {"group_idx": 3, "kind": "sliding_window_mla", "spec_type": "SlidingWindowMLASpec", "block_size": 8, "prefix_cacheable": true, "prefix_replay_tokens": 0, "sliding_window": 128},
          {"group_idx": 4, "kind": "sliding_window_mla", "spec_type": "SlidingWindowMLASpec", "block_size": 4, "prefix_cacheable": true, "prefix_replay_tokens": 0, "sliding_window": 8}
        ]
      }
    ]
  }
}
```

</details>

### DeepSeek V4.1 Flash

Use the V4.1 Flash 1017 checkpoint. Bounded replay rebuilds the last 128 tokens for groups 0–4 after a cache hit.

```bash
--max-model-len 4096 --tensor-parallel-size 2 --swa-bounded-replay
```

<details>
<summary>DeepSeek V4.1 Flash layout.json</summary>

```json
{
  "http://127.0.0.1:8000": {
    "schema_version": 1,
    "ranks": [
      {
        "rank": 0,
        "weight_version": "default",
        "instance_topic": true,
        "hash_algorithm": "sha256_cbor",
        "hash_seed": "vllm-none-hash",
        "full_hashes": true,
        "hash_block_size": 128,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 0,
        "prefix_caching": true,
        "partial_hash_hits": false,
        "declared_groups": 7,
        "groups": [
          {"group_idx": 0, "kind": "sliding_window_mla", "spec_type": "SlidingWindowMLASpec", "block_size": 32, "prefix_cacheable": false, "prefix_replay_tokens": 128, "sliding_window": 128},
          {"group_idx": 1, "kind": "sliding_window_mla", "spec_type": "SlidingWindowMLASpec", "block_size": 32, "prefix_cacheable": false, "prefix_replay_tokens": 128, "sliding_window": 128},
          {"group_idx": 2, "kind": "sliding_window_mla", "spec_type": "SlidingWindowMLASpec", "block_size": 32, "prefix_cacheable": false, "prefix_replay_tokens": 128, "sliding_window": 128},
          {"group_idx": 3, "kind": "sliding_window_mla", "spec_type": "SlidingWindowMLASpec", "block_size": 32, "prefix_cacheable": false, "prefix_replay_tokens": 128, "sliding_window": 128},
          {"group_idx": 4, "kind": "sliding_window_mla", "spec_type": "SlidingWindowMLASpec", "block_size": 32, "prefix_cacheable": false, "prefix_replay_tokens": 128, "sliding_window": 128},
          {"group_idx": 5, "kind": "mla_attention", "spec_type": "MLAAttentionSpec", "block_size": 128, "prefix_cacheable": true, "prefix_replay_tokens": 0, "tokens_per_state": "2", "cache_role": "sparse"},
          {"group_idx": 6, "kind": "scratch", "spec_type": "CircularBufferSpec", "block_size": 8, "prefix_cacheable": false, "prefix_replay_tokens": 0, "tokens_per_state": "1"}
        ]
      }
    ]
  }
}
```

</details>

### Ascend models

These recipes use an Ascend build with the zero-lookahead cache lookup rule. Set `PYTHONHASHSEED=0` for these files and apply the Ascend checkpoint's launch options. Confirm group order and block size on the running worker. For Ascend Mooncake P/D, these layouts describe the Prefill worker. Use a separately verified layout for cache-aware Decode.

#### Qwen3.5-4B with MTP on Ascend

Use the 640-token cache layout on vLLM 0.32 with the matching Ascend build.

```bash
--max-model-len 4096 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

<details>
<summary>Ascend Qwen3.5 MTP layout.json</summary>

```json
{
  "http://127.0.0.1:8000": {
    "schema_version": 1,
    "ranks": [
      {
        "rank": 0,
        "weight_version": "default",
        "instance_topic": true,
        "hash_algorithm": "sha256_cbor",
        "hash_seed": "0",
        "full_hashes": true,
        "hash_block_size": 640,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 3,
        "prefix_caching": true,
        "tail_drop_groups": [],
        "declared_groups": 4,
        "groups": [
          {"group_idx": 0, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 640, "prefix_cacheable": true, "prefix_replay_tokens": 0},
          {"group_idx": 1, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 640, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 640},
          {"group_idx": 2, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 640, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 640},
          {"group_idx": 3, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 640, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 640}
        ]
      }
    ]
  }
}
```

</details>

#### Qwen3.8-27B with MTP on Ascend

Use the W8A8 checkpoint with `--quantization ascend` and the 896-token cache layout on vLLM 0.32 with the matching Ascend build.

```bash
--max-model-len 4096 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

<details>
<summary>Ascend Qwen3.8 MTP layout.json</summary>

```json
{
  "http://127.0.0.1:8000": {
    "schema_version": 1,
    "ranks": [
      {
        "rank": 0,
        "weight_version": "default",
        "instance_topic": true,
        "hash_algorithm": "sha256_cbor",
        "hash_seed": "0",
        "full_hashes": true,
        "hash_block_size": 896,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 3,
        "prefix_caching": true,
        "tail_drop_groups": [],
        "declared_groups": 4,
        "groups": [
          {"group_idx": 0, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0},
          {"group_idx": 1, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 896},
          {"group_idx": 2, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 896},
          {"group_idx": 3, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 896}
        ]
      }
    ]
  }
}
```

</details>

#### Qwen3.8-27B with DSpark on Ascend

Use the BF16 checkpoint, the Qwen3.8 DSpark draft, and the 896-token cache layout on vLLM 0.34 with the matching Ascend build. The example uses TP2.

```bash
--max-model-len 4096 --tensor-parallel-size 2 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":8,"model":"/models/Qwen3.8-27B-speculator.dspark"}'
```

<details>
<summary>Ascend Qwen3.8 DSpark layout.json</summary>

```json
{
  "http://127.0.0.1:8000": {
    "schema_version": 1,
    "ranks": [
      {
        "rank": 0,
        "weight_version": "default",
        "instance_topic": true,
        "hash_algorithm": "sha256_cbor",
        "hash_seed": "0",
        "full_hashes": true,
        "hash_block_size": 896,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 8,
        "prefix_caching": true,
        "tail_drop_groups": [],
        "declared_groups": 15,
        "groups": [
          {"group_idx": 0, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0},
          {"group_idx": 1, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0},
          {"group_idx": 2, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0},
          {"group_idx": 3, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0},
          {"group_idx": 4, "kind": "sliding_window", "spec_type": "SlidingWindowSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0, "sliding_window": 2048},
          {"group_idx": 5, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 896},
          {"group_idx": 6, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 896},
          {"group_idx": 7, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 896},
          {"group_idx": 8, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 896},
          {"group_idx": 9, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 896},
          {"group_idx": 10, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 896},
          {"group_idx": 11, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 896},
          {"group_idx": 12, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 896},
          {"group_idx": 13, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 896},
          {"group_idx": 14, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 896}
        ]
      }
    ]
  }
}
```

</details>

For zero-lookahead speculation, `tail_drop_groups` lists the cache groups that actually drop their last reusable unit. An empty array means the worker skips that drop. Use the lookup rule for each role. The predictor's draft length and the number of tokens hashed past a block are separate fields.

## P/D routing

P/D disaggregation puts prompt processing on Prefill workers and token generation on Decode workers. The example below uses NIXL and enables `kv_aware` on both stages.

1. Start two workers on different GPUs with the same checkpoint, vLLM version, connector and transport. Keep Prefill on HTTP port `8000` or gRPC port `50051`. Move Decode to HTTP port `8001` and, for a gRPC worker, gRPC port `50052`.
2. Set `VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1` on both workers. Set `VLLM_NIXL_SIDE_CHANNEL_PORT=5600` on Prefill and `5601` on Decode. For separate hosts, use reachable addresses.
3. Add `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'` to Prefill and the same option with `"kv_role":"kv_consumer"` to Decode. Enable prefix caching, `sha256_cbor`, and live and replay events on both stages. Give each process a distinct event port and topic. For gRPC, pass the engine options after `--`.
4. For hybrid models, pass the [P/D layout](#pd-layout). Ordinary single-group workers with event discovery can omit it.

Follow the transport setup for each scored worker and the Router ([HTTP](#http-workers) or [gRPC](#grpc-workers)).

```bash
PREFILL=http://127.0.0.1:8000
DECODE=http://127.0.0.1:8001
# For gRPC, use grpc://127.0.0.1:50051 and grpc://127.0.0.1:50052.
vllm-router --port 19000 --vllm-pd-disaggregation --prefill "$PREFILL" --decode "$DECODE" \
  --prefill-policy kv_aware --decode-policy kv_aware --kv-connector nixl
```

With HTTP workers, use explicit prompt token IDs to get Decode cache credit. With gRPC workers, the Router can also use tokens from its own tokenizer. For a Prefill-only setup, set `--decode-policy round_robin` and omit Decode's event settings and layout entry.

#### Ascend Mooncake

For HTTP P/D, replace the NIXL worker settings with these Mooncake options. Use different `kv_port` values on one host and set both stages' DP and TP sizes to match the deployment. The Qwen3.8 Ascend DSpark example uses `tp_size: 2` on both stages.

```bash
# Prefill
--kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_buffer_device":"npu","kv_port":38352,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":1,"tp_size":1}}}'
# Decode
--kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_buffer_device":"npu","kv_port":38353,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":1,"tp_size":1}}}'
```

Use `--kv-connector mooncake` in the Router command. For the Ascend recipes above, use `--decode-policy round_robin` and include only the Prefill layout. The optional bootstrap port can be omitted when Prefill returns the transfer parameters.

### P/D layout

For each hybrid Prefill, declare the Decode URLs compatible with its connector and layout. Use worker base URLs with the transport used by Router. If the worker reports its layout, the entry needs only the compatible Decode URLs.

```json
{
  "http://127.0.0.1:8000": {
    "compatible_decode_urls": ["http://127.0.0.1:8001"]
  }
}
```

Include a complete model entry for each scored worker that does not report its layout. The example configures both stages. Use `grpc://` URL keys and matching pairing values for gRPC. Use each stage's effective speculation and cache rules. Leave Program scheduling disabled for hybrid P/D layouts.

<details>
<summary>Complete P/D layout.json · Qwen3.5-4B with NIXL</summary>

```json
{
  "http://127.0.0.1:8000": {
    "schema_version": 1,
    "compatible_decode_urls": [
      "http://127.0.0.1:8001"
    ],
    "ranks": [
      {
        "rank": 0,
        "weight_version": "default",
        "instance_topic": true,
        "hash_algorithm": "sha256_cbor",
        "hash_seed": "vllm-none-hash",
        "full_hashes": true,
        "hash_block_size": 16,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 0,
        "prefix_caching": true,
        "partial_hash_hits": true,
        "declared_groups": 4,
        "groups": [
          {"group_idx": 0, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 576, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 1, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 576, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 2, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 576, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
          {"group_idx": 3, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 576, "prefix_cacheable": true, "prefix_replay_tokens": 0}
        ]
      }
    ]
  },
  "http://127.0.0.1:8001": {
    "schema_version": 1,
    "ranks": [
      {
        "rank": 0,
        "weight_version": "default",
        "instance_topic": true,
        "hash_algorithm": "sha256_cbor",
        "hash_seed": "vllm-none-hash",
        "full_hashes": true,
        "hash_block_size": 16,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 0,
        "prefix_caching": true,
        "partial_hash_hits": true,
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

</details>

If event discovery is unavailable, include `endpoints` and `replay_endpoints` in rank order, and include `model` and `publisher_topic` for each rank.

Pass the file once by appending `--kv-hybrid-layout layout.json` to the Router command.

## Check cache reuse

In an idle pool, send the same prompt twice. The first request fills the cache. The second should increase the Router's KV-aware hit counter. Set `SERVED_MODEL` to the worker's served name. For the hybrid recipes above, start with `PROMPT_TOKENS=2001`.

```bash
SERVED_MODEL=qwen3 PROMPT_TOKENS=129
payload=$(jq -nc --arg model "$SERVED_MODEL" --argjson n "$PROMPT_TOKENS" \
  '{model:$model,prompt:[range(1000;1000+$n)],max_tokens:1,temperature:0}')
request() {
  curl -fsS http://127.0.0.1:19000/v1/completions \
    -H 'Content-Type: application/json' --data-binary "$payload" >/dev/null
}
hits() {
  curl -fsS http://127.0.0.1:29000/metrics |
    awk '$1 == "vllm_router_kv_placement_lookups_total{result=\"hit\"}" {hits += $2} END {print hits+0}'
}
request && sleep 1 && before=$(hits) && request && after=$(hits) &&
  printf 'Router KV-aware hits: %s → %s\n' "$before" "$after"
```

Repeat the last request if events are still arriving. To confirm actual reuse, compare the worker's `prefix_cache_hits_total` before and after the repeat. For P/D, check Prefill.

```bash
curl -fsS http://127.0.0.1:8000/metrics | grep '^vllm:prefix_cache_hits_total'
```

To confirm NIXL transfer, send a fresh prompt and compare `vllm:nixl_bytes_transferred_sum` on Decode's metrics port before and after the request.

Both worker transports accept `/v1/chat/completions` through the Router. On gRPC, `/v1/completions` accepts one flat token-ID array as shown above.

## Versions

Use the same vLLM release for the engine and its gRPC frontend. In all versions below, gRPC discovers live and replay addresses through `GetKvEventSources`. The table covers the engine and event protocol. The Router's local gRPC chat renderer uses `vllm-chat` from vLLM 0.30.0. Check [gRPC API capabilities](../backend/grpc.md#api-compatibility) for model-specific rendering and output features.

| vLLM release | HTTP event discovery | Layout support | vLLM Ascend pairing |
| --- | --- | --- | --- |
| [0.30.0](https://github.com/vllm-project/vllm/tree/v0.30.0) | Pass a layout and [static addresses](#static-event-endpoints). | Zero hash lookahead. | [0.30.0 stable](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.30.0). |
| [0.31.0](https://github.com/vllm-project/vllm/tree/v0.31.0)–[0.32.0](https://github.com/vllm-project/vllm/tree/v0.32.0) | `/kv_event_sources` | Zero hash lookahead and V4.1 bounded replay. | [0.31.0rc1 wheel](https://pypi.org/project/vllm-ascend/0.31.0rc1/) or [0.32.0rc1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.32.0rc1). See [Ascend models](#ascend-models) for MTP and DSpark. |
| [0.33.0](https://github.com/vllm-project/vllm/tree/v0.33.0) | `/kv_event_sources` | Adds nonzero hash lookahead via `BlockStored.lookahead_token_ids`. The GPU recipes use this release. | Use an Ascend build compatible with vLLM 0.33. |
| [0.34.0](https://github.com/vllm-project/vllm/tree/v0.34.0) | `/kv_event_sources` | Defaults to 64-token blocks when the backend has no preferred size. Use the actual block size in an explicit layout. | Use an Ascend build compatible with vLLM 0.34. |

## CLI options

See [worker and KV event options](../cli_reference.md#workers-and-backend), [routing policy options](../cli_reference.md#routing-policies), and [P/D disaggregation options](../cli_reference.md#prefilldecode-disaggregation) for value types and defaults.

## Troubleshooting

- If the engine reports hits and the Router does not, check the layout's model and topic, the hash flags, and whether both event ports are reachable from the Router.
- For `unsupported tail-drop layout`, check the worker's [MTP recipe](#qwen38-27b-with-mtp).
- If vLLM reports `Address already in use`, check the live and replay port ranges before starting it again. Inspect the actual event addresses in `/kv_event_sources` for HTTP or `Control.GetKvEventSources` for gRPC.
- When a worker restarts, rerun the worker settings for a fresh topic. Update the layout only if it explicitly pins the topic or other layout fields changed. On a Router-only restart, replay recovers the events still in the worker's history.
