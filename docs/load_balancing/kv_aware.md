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
| [Qwen3.5-4B](#qwen35-4b) | Base, [Ascend modes](#qwen35-4b-with-mtp-on-ascend) |
| [Qwen3.6-27B](#qwen36-27b) | Base, [Ascend modes](#qwen36-27b-w8a8-on-ascend) |
| [Qwen3.8-27B](#qwen38-27b) | Base, [MTP](#qwen38-27b-with-mtp), [DSpark](#qwen38-27b-with-dspark), [Ascend variants](#ascend-models) |
| [Qwen4Exp](#qwen4exp-fp8-qsa) | GPU FP8, [Ascend W4A8](#qwen4exp-w4a8-on-ascend) |
| [DeepSeek V4 Flash](#deepseek-v4-flash) | Base, [GPU DSpark](#deepseek-v4-flash-with-dspark), [Ascend 0731 W8A8](#deepseek-v4-flash-0731-w8a8-on-ascend) |
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

Use the [vLLM Ascend 0.32.0rc1 image](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.32.0rc1) with its bundled vLLM 0.32.0 on Atlas 800I A2 (Ascend 910B, 64 GiB per NPU). The layouts below are from HTTP deployments. Use the same checkpoint and engine settings, then check the running worker's cache groups before reusing a layout.

#### Start an A2 worker

In the Ascend container, use the following worker command. The example starts Qwen3.5-4B with MTP on one NPU. For another model, replace the model path, served name, and option assignments in this command with the settings from its recipe below. For colocated serving, run the Router on the same host with the corresponding `layout.json`.

```bash
MODEL=/models/Qwen3.5-4B
SERVED_MODEL=qwen35-a2
MAX_LEN=10240 BATCH_TOKENS=4096 MEM_FRACTION=0.80
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0
KV_TOPIC=$(python3 -c 'from uuid import uuid4; print(uuid4().hex)')
EVENTS=$(printf '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:5557","replay_endpoint":"tcp://*:5558","topic":"%s"}' "$KV_TOPIC")
MODEL_OPTIONS=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')

vllm serve "$MODEL" --host 127.0.0.1 --port 8000 --served-model-name "$SERVED_MODEL" \
  --max-model-len "$MAX_LEN" --max-num-batched-tokens "$BATCH_TOKENS" --max-num-seqs 4 \
  --gpu-memory-utilization "$MEM_FRACTION" --enforce-eager --skip-mm-profiling \
  --enable-scale-out --enable-prefix-caching --enable-prompt-tokens-details \
  --prefix-caching-hash-algo sha256_cbor --kv-events-config "$EVENTS" "${MODEL_OPTIONS[@]}"
```

```bash
vllm-router --host 127.0.0.1 --port 19000 --prometheus-port 29000 \
  --policy kv_aware --worker-urls http://127.0.0.1:8000 --kv-hybrid-layout layout.json
```

The layouts use `hash_seed: "0"` to match the worker. Keep `sha256_cbor`, full-width hashes, and incremental reporting enabled. The examples rely on HTTP event discovery. For fixed endpoints without discovery, include the live and replay addresses in the layout and put the worker's served model and the generated topic in its rank record. Choose distinct HTTP, live and replay ports for each worker on the same host.

Select one method in `--speculative-config` per worker. To turn speculation off, omit the option and use the layout for that mode. The measured A2 recipes below publish zero hash lookahead. Set `speculative_tokens` to the configured draft length and `hash_lookahead_tokens` to the number of lookahead tokens actually used by the hash. Use an empty `tail_drop_groups` array only when the engine's cache lookup skips the last-unit drop.

| Checkpoint | Colocated modes below | NPU count | Prompt tokens for a reuse check |
| --- | --- | ---: | ---: |
| Qwen3.5-4B | [Off or MTP](#qwen35-4b-with-mtp-on-ascend), [DSpark](#qwen35-4b-with-dspark-on-ascend) | 1 | 1,927 |
| Qwen3.6-27B W8A8 | [Off, MTP or DSpark](#qwen36-27b-w8a8-on-ascend) | 1 | 2,695 |
| Qwen3.8-27B W8A8 | [Off or MTP](#qwen38-27b-with-mtp-on-ascend) | 1 | 2,695 |
| Qwen3.8-27B BF16 | [DSpark](#qwen38-27b-with-dspark-on-ascend) | 2 | 2,695 |
| Qwen3.8-Flash-Next W4A8 (Qwen4Exp) | [Off or MTP](#qwen4exp-w4a8-on-ascend) | 2 | 4,999 |
| DeepSeek V4 Flash 0731 W8A8 | [Off or DSpark](#deepseek-v4-flash-0731-w8a8-on-ascend) | 8 | 16,401 |

#### Qwen3.5-4B with MTP on Ascend

Use the worker command above. The four cache groups are the same with speculation off. To serve without speculation, set `MODEL_OPTIONS=()` and `"speculative_tokens": 0` in the layout.

<details>
<summary>A2 Qwen3.5-4B MTP layout.json</summary>

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
        "full_hashes": true,
        "hash_seed": "0",
        "hash_block_size": 640,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 3,
        "tail_drop_groups": [],
        "partial_hash_hits": false,
        "prefix_caching": true,
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

#### Qwen3.5-4B with DSpark on Ascend

Use the same target checkpoint and a compatible Ascend DSpark draft for Qwen3.5-4B at `/models/Qwen3.5-4B-DSpark`. The draft adds cache groups. Use the eight-group layout for this mode.

```bash
MAX_LEN=4096
MODEL_OPTIONS=(--speculative-config '{"method":"dspark","num_speculative_tokens":8,"model":"/models/Qwen3.5-4B-DSpark"}')
```

<details>
<summary>A2 Qwen3.5-4B DSpark layout.json</summary>

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
        "full_hashes": true,
        "hash_seed": "0",
        "hash_block_size": 640,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 8,
        "tail_drop_groups": [],
        "partial_hash_hits": false,
        "prefix_caching": true,
        "declared_groups": 8,
        "groups": [
          {"group_idx": 0, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 640, "prefix_cacheable": true, "prefix_replay_tokens": 0},
          {"group_idx": 1, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 640, "prefix_cacheable": true, "prefix_replay_tokens": 0},
          {"group_idx": 2, "kind": "sliding_window", "spec_type": "SlidingWindowSpec", "block_size": 640, "prefix_cacheable": true, "prefix_replay_tokens": 0, "sliding_window": 2048},
          {"group_idx": 3, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 640, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 640},
          {"group_idx": 4, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 640, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 640},
          {"group_idx": 5, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 640, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 640},
          {"group_idx": 6, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 640, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 640},
          {"group_idx": 7, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 640, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 640}
        ]
      }
    ]
  }
}
```

</details>

#### Qwen3.6-27B W8A8 on Ascend

Use [Eco-Tech/Qwen3.6-27B-w8a8](https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-27B-w8a8) and `--quantization ascend`. The measured runs used the same four cache groups in all three modes. Set `MODEL_OPTIONS` and `speculative_tokens` together.

| Mode | Worker options | `MAX_LEN` | `speculative_tokens` |
| --- | --- | ---: | ---: |
| Off | `--quantization ascend` | 10240 | 0 |
| MTP | `--quantization ascend --speculative-config '{"method":"mtp","num_speculative_tokens":3}'` | 10240 | 3 |
| DSpark | `--quantization ascend --speculative-config '{"method":"dspark","num_speculative_tokens":8,"model":"/models/Qwen3.6-27B-DSpark"}'` | 4096 | 8 |

<details>
<summary>A2 Qwen3.6-27B W8A8 MTP layout.json</summary>

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
        "full_hashes": true,
        "hash_seed": "0",
        "hash_block_size": 896,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 3,
        "tail_drop_groups": [],
        "partial_hash_hits": false,
        "prefix_caching": true,
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

#### Qwen3.8-27B with MTP on Ascend

Use the W8A8 checkpoint on one NPU with the defaults in the worker command.

```bash
MODEL_OPTIONS=(--quantization ascend --speculative-config '{"method":"mtp","num_speculative_tokens":3}')
```

For speculation off, keep `--quantization ascend` and set `"speculative_tokens": 0` in the layout.

<details>
<summary>A2 Qwen3.8-27B W8A8 MTP layout.json</summary>

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
        "full_hashes": true,
        "hash_seed": "0",
        "hash_block_size": 896,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 3,
        "tail_drop_groups": [],
        "partial_hash_hits": false,
        "prefix_caching": true,
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

Use the BF16 target with a matching Qwen3.8-27B DSpark draft. In the worker command, set `MODEL` to the BF16 checkpoint and expose two NPUs. This configuration was exercised with the same 0.32.0rc1 image.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
MAX_LEN=4096
MODEL_OPTIONS=(--tensor-parallel-size 2 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":8,"model":"/models/Qwen3.8-27B-speculator.dspark"}')
```

<details>
<summary>A2 Qwen3.8-27B BF16 DSpark layout.json</summary>

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
        "full_hashes": true,
        "hash_seed": "0",
        "hash_block_size": 896,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 8,
        "tail_drop_groups": [],
        "partial_hash_hits": false,
        "prefix_caching": true,
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

#### Qwen4Exp W4A8 on Ascend

Use [Eco-Tech/Qwen3.8-Flash-Next-w4a8](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-Flash-Next-w4a8) with the V2 model runner. Give the Ascend container 128 GiB of shared memory. The colocated test used TP2 and host offload for Engram.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ASCEND_PLE_BACKEND=torch
MEM_FRACTION=0.93
MODEL_OPTIONS=(--quantization ascend --tensor-parallel-size 2 --trust-remote-code \
  --enable-expert-parallel --mamba-cache-mode align \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}')
```

With MTP off, omit `--speculative-config`, set `speculative_tokens` to 0, set group 0's size to 4, remove group 6, and set `declared_groups` to 6. A2 validation covers the native MTP draft. See the [Ascend model support matrix](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/support_matrix/supported_models.md) for available speculative modes.

<details>
<summary>A2 Qwen4Exp W4A8 colocated MTP layout.json</summary>

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
        "full_hashes": true,
        "hash_seed": "0",
        "hash_block_size": 1664,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 3,
        "tail_drop_groups": [],
        "partial_hash_hits": false,
        "prefix_caching": true,
        "declared_groups": 7,
        "groups": [
          {"group_idx": 0, "spec_type": "CircularBufferSpec", "block_size": 8, "tokens_per_state": "1", "prefix_cacheable": false, "prefix_replay_tokens": 0, "kind": "scratch"},
          {"group_idx": 1, "spec_type": "FullAttentionSpec", "block_size": 1664, "tokens_per_state": "1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "kind": "full_attention"},
          {"group_idx": 2, "spec_type": "MambaSpec", "block_size": 1664, "tokens_per_state": "-1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "kind": "mamba"},
          {"group_idx": 3, "spec_type": "MambaSpec", "block_size": 1664, "tokens_per_state": "-1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "kind": "mamba"},
          {"group_idx": 4, "spec_type": "MambaSpec", "block_size": 1664, "tokens_per_state": "-1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "kind": "mamba"},
          {"group_idx": 5, "spec_type": "MambaSpec", "block_size": 1664, "tokens_per_state": "-1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "kind": "mamba"},
          {"group_idx": 6, "spec_type": "MambaSpec", "block_size": 1664, "tokens_per_state": "-1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "kind": "mamba"}
        ]
      }
    ]
  }
}
```

</details>

#### DeepSeek V4 Flash 0731 W8A8 on Ascend

Use the full W8A8 checkpoint in the [Ascend deployment guide](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/DeepSeek-V4-Flash.md#31-model-weight). The colocated configuration uses one 8-NPU node. Initialize the container's Ascend toolkit and set `HCCL_IF_IP`, `HCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, and `TP_SOCKET_IFNAME` for the node's communication interface.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_OP_EXPANSION_MODE=AIV OMP_NUM_THREADS=8 VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=16384
MAX_LEN=65536 BATCH_TOKENS=8192 MEM_FRACTION=0.90
MODEL_OPTIONS=(--quantization ascend --tensor-parallel-size 8 --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 --block-size 128 --attention_config.indexer_kv_dtype int8 \
  --no-disable-hybrid-kv-cache-manager --prefix-cache-retention-interval 16384 \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":16}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true}')
```

The measured layout applies with DSpark off by omitting `--speculative-config` and setting `"speculative_tokens": 0`. In both modes, a repeated 16,401-token prompt reused 16,384 tokens. Classic MTP uses the separate W8A8 MTP checkpoint and needs its own measured layout.

<details>
<summary>A2 DeepSeek V4 Flash 0731 W8A8 DSpark layout.json</summary>

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
        "full_hashes": true,
        "hash_seed": "0",
        "hash_block_size": 8,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 7,
        "tail_drop_groups": [],
        "partial_hash_hits": false,
        "prefix_caching": true,
        "declared_groups": 6,
        "groups": [
          {"group_idx": 0, "spec_type": "MLAAttentionSpec", "block_size": 512, "tokens_per_state": "4", "cache_role": "sparse", "prefix_cacheable": true, "prefix_replay_tokens": 0, "kind": "mla_attention"},
          {"group_idx": 1, "spec_type": "MLAAttentionSpec", "block_size": 16384, "tokens_per_state": "128", "cache_role": "sparse", "prefix_cacheable": true, "prefix_replay_tokens": 0, "kind": "mla_attention"},
          {"group_idx": 2, "spec_type": "SlidingWindowMLASpec", "block_size": 128, "tokens_per_state": "1", "sliding_window": 128, "prefix_cacheable": true, "prefix_replay_tokens": 0, "kind": "sliding_window_mla"},
          {"group_idx": 3, "spec_type": "SlidingWindowMLASpec", "block_size": 128, "tokens_per_state": "1", "sliding_window": 128, "prefix_cacheable": true, "prefix_replay_tokens": 0, "kind": "sliding_window_mla"},
          {"group_idx": 4, "spec_type": "SlidingWindowMLASpec", "block_size": 8, "tokens_per_state": "1", "sliding_window": 8, "prefix_cacheable": true, "prefix_replay_tokens": 0, "kind": "sliding_window_mla"},
          {"group_idx": 5, "spec_type": "SlidingWindowMLASpec", "block_size": 32, "tokens_per_state": "1", "sliding_window": 128, "prefix_cacheable": true, "prefix_replay_tokens": 0, "kind": "sliding_window_mla"}
        ]
      }
    ]
  }
}
```

</details>

The [Ascend V4.1 Flash guide](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/DeepSeek-V4.1-Flash.md#55-multi-node-colocated-deployment) uses two A2 nodes for the full-weight colocated DSpark example. Use that model's own runtime layout. The Flash 0731 layout above applies to V4 Flash.

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

On A2, the vLLM Ascend Mooncake connector transfers KV from Prefill to Decode. The Router's `kv_aware` policy scores reuse for Prefill selection, and `--kv-connector mooncake` selects the P/D handoff. Run the two workers on separate NPUs. Each worker used DP1. Configure MTP or DSpark through the worker's `--speculative-config`. The results below verify each connector and speculation mode together.

| Checkpoint | Ascend connector | Tested speculation modes | TP per worker |
| --- | --- | --- | ---: |
| Qwen3-4B (full attention) | `MooncakeConnectorV1` | Off | 1 |
| Qwen3.5-4B and Qwen3.6-27B W8A8 | `MooncakeConnectorV1` | Off, MTP, DSpark | 1 |
| Qwen3.8-27B W8A8 | `MooncakeConnectorV1` | Off, MTP | 1 |
| Qwen3.8-27B BF16 | `MooncakeConnectorV1` | DSpark | 2 |
| Qwen3.8-Flash-Next W4A8 | `MooncakeConnectorV2` | Off, MTP | 2 |

Add the matching transfer option to each worker command. The example uses V1 with TP1. Give each worker its own `engine_id` and `kv_port`. Set both `tp_size` values to match the selected worker configuration.

```bash
# Prefill
--kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_buffer_device":"npu","engine_id":"prefill-a2","kv_port":38352,"kv_load_failure_policy":"fail","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":1,"tp_size":1}}}'
# Decode
--kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_buffer_device":"npu","engine_id":"decode-a2","kv_port":38353,"kv_load_failure_policy":"fail","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":1,"tp_size":1}}}'
```

For W4A8 Qwen4Exp with V2, add `--mamba-ssm-cache-dtype bfloat16` on both workers. In the Prefill layout, the measured `hash_block_size` and the full attention and Mamba group sizes are 896. The scratch group keeps its mode-specific size from the [colocated recipe](#qwen4exp-w4a8-on-ascend).

Add `compatible_decode_urls` to each Prefill entry as shown in [P/D layout](#pd-layout). Use `--prefill-policy kv_aware --decode-policy round_robin --kv-connector mooncake` in the Router and supply the Prefill layout. The optional bootstrap port can be omitted when Prefill returns the transfer parameters.

For DeepSeek V4, follow the Ascend P/D recipe for your checkpoint and engine version. The A2 1P1D test used `MooncakeHybridConnector` with the four-layer V4 Flash mini4 checkpoint and speculation off. The full W8A8 DSpark checkpoint was verified in colocated mode.

Verify Decode reads using per-request Mooncake completion logs. With `MooncakeHybridConnector`, some Qwen3.5 and Qwen3.8 requests failed during Mamba block pairing before Decode could read the KV, even when the API returned HTTP 200. Use the tested connectors in the table for those Qwen configurations.

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

In an idle pool, send the same prompt twice. The first request fills the cache. The second should increase the Router's KV-aware hit counter. Set `SERVED_MODEL` to the worker's served name. For the GPU hybrid recipes above, start with `PROMPT_TOKENS=2001`. For Ascend, use the prompt length in the [A2 recipe table](#start-an-a2-worker).

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

For Ascend HTTP, `--enable-prompt-tokens-details` exposes the worker's cached-token count. To display it in the check above, replace `>/dev/null` with `| jq '.usage.prompt_tokens_details.cached_tokens'`. Once events have arrived, the repeated prompt should report positive cached tokens and increase the Router hit counter. For P/D, check the Prefill response for local reuse and the Decode connector log for remote reads. With MTP or DSpark enabled, verify that `vllm:spec_decode_num_draft_tokens_total` increases on the worker generating tokens.

To confirm NIXL transfer, send a fresh prompt and compare `vllm:nixl_bytes_transferred_sum` on Decode's metrics port before and after the request.

Both worker transports accept `/v1/chat/completions` through the Router. On gRPC, `/v1/completions` accepts one flat token-ID array as shown above.

## Versions

Pair the gRPC worker's `vllm-rs` and Python engine from the same release. The table covers worker KV event discovery and layout support.

| vLLM release | HTTP event discovery | Layout support | vLLM Ascend pairing |
| --- | --- | --- | --- |
| [0.30.0](https://github.com/vllm-project/vllm/tree/v0.30.0) | Pass a layout and [static addresses](#static-event-endpoints). | Zero hash lookahead. | [0.30.0 stable](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.30.0). |
| [0.31.0](https://github.com/vllm-project/vllm/tree/v0.31.0)–[0.32.0](https://github.com/vllm-project/vllm/tree/v0.32.0) | `/kv_event_sources` | Zero hash lookahead and V4.1 bounded replay. | [0.31.0rc1 wheel](https://pypi.org/project/vllm-ascend/0.31.0rc1/) or [0.32.0rc1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.32.0rc1). See [Ascend models](#ascend-models) for MTP and DSpark. |
| [0.33.0](https://github.com/vllm-project/vllm/tree/v0.33.0) | `/kv_event_sources` | Adds nonzero hash lookahead via `BlockStored.lookahead_token_ids`. The GPU recipes use this release. | Use an Ascend build compatible with vLLM 0.33. |
| [0.34.0](https://github.com/vllm-project/vllm/tree/v0.34.0) | `/kv_event_sources` | Defaults to 64-token blocks when the backend has no preferred size. Use the actual block size in an explicit layout. | Use an Ascend build compatible with vLLM 0.34. |

All listed gRPC worker releases discover live and replay addresses through `GetKvEventSources`. For chat routed to gRPC workers, the Router renders locally using `vllm-chat` pinned to vLLM 0.30.0. Model-specific rendering and output features follow the [Router's gRPC API compatibility](../backend/grpc.md#api-compatibility).

## CLI options

See [worker and KV event options](../cli_reference.md#workers-and-backend), [routing policy options](../cli_reference.md#routing-policies), and [P/D disaggregation options](../cli_reference.md#prefilldecode-disaggregation) for value types and defaults.

## Troubleshooting

- If the engine reports hits and the Router does not, check the layout's model and topic, the hash flags, and whether both event ports are reachable from the Router.
- If a worker-reported layout is rejected, run the Router with `--log-level debug` and look for `worker-reported KV layout rejected` to see the reason.
- If replay history is incomplete, an inferred full-attention layout resumes cache credit after the Router verifies a first-block Store against the default hash root.
- For `unsupported tail-drop layout`, check the worker's [MTP recipe](#qwen38-27b-with-mtp).
- If vLLM reports `Address already in use`, check the live and replay port ranges before starting it again. Inspect the actual event addresses in `/kv_event_sources` for HTTP or `Control.GetKvEventSources` for gRPC.
- When a worker restarts, rerun the worker settings for a fresh topic. Update the layout only if it explicitly pins the topic or other layout fields changed. On a Router-only restart, replay recovers the events still in the worker's history.
