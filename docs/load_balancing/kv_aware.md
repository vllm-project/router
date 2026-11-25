# KV-aware routing

`kv_aware` routes requests using prefix reuse and worker load. Use the Router's OpenAI API with HTTP or gRPC vLLM workers. The commands below add KV-aware routing to a working vLLM deployment.

## Versions

Pair `vllm-rs` with the same vLLM release as its Python engine. The examples use the `vllm-router` CLI.

| vLLM | Worker setup | Cache reuse support | vLLM Ascend |
| --- | --- | --- | --- |
| 0.30.0 | Automatic discovery for gRPC. Stock HTTP needs [static setup](#static-event-endpoints). | Speculation with zero hash lookahead. | [0.30.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.30.0) |
| 0.31.0–0.32.0 | Automatic discovery for HTTP and gRPC. | Speculation with zero hash lookahead. DeepSeek V4.1 bounded replay with the matching layout. | [0.31.0rc1](https://pypi.org/project/vllm-ascend/0.31.0rc1/) and [0.32.0rc1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.32.0rc1), respectively |
| 0.33.0–0.34.0 | Automatic discovery for HTTP and gRPC. | Speculation with zero or nonzero hash lookahead. DeepSeek V4.1 bounded replay with the matching layout. | Build against the matching vLLM version. |

Use the [model layout reference](#model-layout-reference) to choose a layout for your model. Use one worker interface throughout a pool. For gRPC P/D, specify the worker URLs directly. See [gRPC API compatibility](../backend/grpc.md#api-compatibility) for text, chat and tokenizer requirements.

## Choose a layout configuration

| Deployment | What to supply |
| --- | --- |
| Ordinary single-group full attention with event discovery, the default hash seed, and no sliding window or speculation | Worker URLs. Router derives the single-group layout and checks the hash contract against worker events. |
| Colocated worker that reports its complete layout | Worker URLs. Router discovers the layout and event addresses. |
| Hybrid P/D with worker-reported layouts | Worker URLs. Use [explicit pairs](#pd-worker-pairs) for a mixed P/D pool. |
| Worker with event discovery that needs a layout declaration | A [layout file](#layout-template) for the missing group and hash information. Router discovers the served model, publisher topic and event addresses. |
| Worker without event discovery, such as stock HTTP vLLM 0.30.0 | [Static event options](#static-event-endpoints). Supply a layout if the cache has multiple groups or specialized reuse rules. |

A manual layout describes the worker's cache groups, hashing and reuse rules. Router discovers the served model and event publisher when the worker reports them.

## Configure the workers

### Worker requirements

| Setting | What to configure and why |
| --- | --- |
| Prefix caching | Keep it enabled. Add `--enable-prefix-caching` if needed to turn it on. |
| Hash algorithm | Pass `--prefix-caching-hash-algo sha256_cbor` so Router and vLLM compute the same block keys. |
| Event hashes | Set `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0` to publish the full 32-byte hashes. |
| KV events | Enable events through `--kv-events-config` and keep incremental reporting. Router uses live events to track cache changes and replay to recover missed events. |
| HTTP text and chat | KV scoring uses the worker's render endpoint. Simple text completions can use `/tokenize`. If preparation is unavailable, the request still routes using load and keeps its original body. Explicit token ID prompts can be scored directly. |

Worker retries can reuse prepared inputs. If preparation fails, Router sends the original request body and skips cache scoring for that worker.

### Set the event ports

Run this in each worker terminal. The default live port is `5557`. Set `LIVE_PORT` to override it and choose a free `REPLAY_PORT` (the example uses `5558`). Reserve separate port ranges for workers on the same host because vLLM offsets fixed event ports by DP rank. For [P/D](#pd-routing), set each stage's ports first.

On vLLM 0.33–0.34, the worker generates a fresh topic with `instance_topic`. On vLLM 0.30–0.32, run `KV_TOPIC=$(python3 -c 'from uuid import uuid4; print(uuid4().hex)')` before each worker start.

```bash
export VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0
EVENTS=$(jq -nc --arg live "${LIVE_PORT:-}" --arg replay "${REPLAY_PORT:-5558}" \
  --arg topic "${KV_TOPIC:-}" '
  {enable_kv_cache_events:true,replay_endpoint:("tcp://*:" + $replay)}
  + (if $live == "" then {} else {endpoint:("tcp://*:" + $live)} end)
  + (if $topic == "" then {instance_topic:true} else {topic:$topic} end)')
KV_ARGS=(--prefix-caching-hash-algo sha256_cbor --kv-events-config "$EVENTS")
```

## Deploy

Set `MODEL` to your worker's checkpoint. Run worker and Router commands in separate terminals. The example addresses assume both run on the same host.

In the Router terminal, use `LAYOUT_ARGS=()` when Router can derive or discover the layout. For an explicit layout, use `LAYOUT_ARGS=(--kv-hybrid-layout layout.json)`.

### Colocated prefill and decode

#### HTTP workers

```bash
vllm serve "$MODEL" --host 127.0.0.1 "${KV_ARGS[@]}"
```

```bash
vllm-router --policy kv_aware \
  --worker-urls http://127.0.0.1:8000 "${LAYOUT_ARGS[@]}"
```

#### gRPC workers

| Platform | Worker launcher | Engine options |
| --- | --- | --- |
| GPU | `vllm-rs serve` | Pass model and KV engine flags after `--`. |
| Ascend A2 with vLLM 0.32.0 | `VLLM_USE_RUST_FRONTEND=1 vllm serve` | Use `--grpc-port` for vLLM's Rust inference and control services. |

```bash
# GPU
vllm-rs serve "$MODEL" --host 127.0.0.1 --grpc-port 50051 \
  -- "${KV_ARGS[@]}"
```

```bash
# Ascend A2
VLLM_USE_RUST_FRONTEND=1 vllm serve "$MODEL" --host 127.0.0.1 \
  --grpc-port 50051 "${KV_ARGS[@]}"
```

For gRPC text and chat, Router loads the tokenizer from the request's model path when available locally. Set `VLLM_ROUTER_TOKENIZER` in the Router terminal if that path is unavailable or the request uses a served alias.

```bash
vllm-router --policy kv_aware \
  --worker-urls grpc://127.0.0.1:50051 "${LAYOUT_ARGS[@]}"
```

### P/D routing

These P/D commands use GPU workers and `NixlConnector`. Use the same checkpoint and engine version on both stages. For separate hosts, set `VLLM_NIXL_SIDE_CHANNEL_HOST` to a reachable address on each worker.

#### Set worker ports and transfer parameters

| Setting | Prefill | Decode |
| --- | --- | --- |
| Worker HTTP port | `8000` | `8001` |
| Worker gRPC port, if used | `50051` | `50052` |
| NIXL side-channel port | `5600` | `5601` |

Prefill supplies the KV cache and Decode reads it. Use the transfer configuration for each stage below. Run the [event setup](#set-the-event-ports) on Prefill. For NIXL Mamba transfers, set `VLLM_SSM_CONV_STATE_LAYOUT=DS` on both stages.

```bash
# Prefill terminal
HTTP_PORT=8000 GRPC_PORT=50051
TRANSFER='{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
```

```bash
# Decode terminal
HTTP_PORT=8001 GRPC_PORT=50052
TRANSFER='{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
export VLLM_NIXL_SIDE_CHANNEL_PORT=5601
KV_ARGS=()
```

In the Router terminal, set `LAYOUT_ARGS=(--kv-hybrid-layout layout.json)` if a scored worker needs a manual layout. Otherwise use `LAYOUT_ARGS=()`. Leave Program scheduling disabled for hybrid P/D.

#### HTTP P/D

Run the worker command on Prefill first, then Decode.

```bash
vllm serve "$MODEL" --host 127.0.0.1 --port "$HTTP_PORT" \
  "${KV_ARGS[@]}" --kv-transfer-config "$TRANSFER"
```

```bash
vllm-router --vllm-pd-disaggregation \
  --prefill http://127.0.0.1:8000 --decode http://127.0.0.1:8001 \
  --prefill-policy kv_aware \
  "${LAYOUT_ARGS[@]}"
```

#### gRPC P/D

Run the GPU worker command on each stage with its own ports and `TRANSFER`. Use the [same tokenizer setup as colocated gRPC](#grpc-workers). Use gRPC worker URLs as keys in a layout file.

```bash
vllm-rs serve "$MODEL" --host 127.0.0.1 --port "$HTTP_PORT" --grpc-port "$GRPC_PORT" \
  -- "${KV_ARGS[@]}" --kv-transfer-config "$TRANSFER"
```

```bash
vllm-router --vllm-pd-disaggregation \
  --prefill grpc://127.0.0.1:50051 --decode grpc://127.0.0.1:50052 \
  --prefill-policy kv_aware \
  "${LAYOUT_ARGS[@]}"
```

For KV-aware Decode, add `--decode-policy kv_aware` and run the event setup on Decode with separate ports. Include its layout if needed. HTTP Decode scoring requires explicit prompt token IDs.

#### P/D worker pairs

`--prefill` and `--decode` define a pool of mutually compatible workers. Use `--pd-pair` when one Router serves more than one transfer-compatible group. Supply an entry for each Prefill, using base URLs without DP-rank suffixes.

```bash
--pd-pair 'http://p1:8000=http://d1:8000,http://d2:8000' \
--pd-pair 'http://p2:8000=http://d3:8000'
```

## Layout files

### Layout template

Save the layout as `layout.json` and pass `--kv-hybrid-layout layout.json`. Use your worker's HTTP or gRPC URL as the key.

Add one `ranks` entry per DP rank to score. Use `rank: 0` for one rank. Put the cache groups in `groups` and set `speculative_tokens: 0` when speculation is off. For a manual layout with a single full-attention group, add `"declared_groups": 1` to that rank.

Read `kv cache group sizes` in each worker's startup log. Copy each value to `groups[].block_size` in the printed order. For example, `kv cache group sizes (576, 576, 576, 576)` means groups 0 through 3 each use 576-token blocks.

This Qwen3.5-4B HTTP example uses those four groups and one DP rank. Router discovers the hash block size separately. After starting Router, [check cache reuse](#check-cache-reuse) with your layout.

```json
{
  "http://127.0.0.1:8000": {
    "ranks": [{
      "rank": 0,
      "speculative_tokens": 0,
      "groups": [
        {"kind": "mamba", "block_size": 576, "mamba_cache_mode": "align"},
        {"kind": "mamba", "block_size": 576, "mamba_cache_mode": "align"},
        {"kind": "mamba", "block_size": 576, "mamba_cache_mode": "align"},
        {"kind": "full_attention", "block_size": 576}
      ]
    }]
  }
}
```

For a manual layout, add only the fields the deployment needs.

- `hash_block_size` (tokens) when Router cannot discover it. HTTP event discovery can read it from worker metrics.
- `partial_hash_hits: true` when the worker can reuse prefixes at boundaries inside a cache block.
- `hash_seed` (string) when the worker uses a custom `PYTHONHASHSEED`. Use the same value.

### Model layout reference

Expand the profile for your checkpoint and mode, then replace the object inside `ranks` in the [layout template](#layout-template). The array order defines the group IDs. Keep every group, including scratch groups. Use the block sizes from your worker's startup log and the actual group settings for your deployment.

- GPU Qwen3.5-4B, Qwen3.6-27B and Qwen3.8-27B without speculation use the four groups in the layout template.
- The shared Ascend Qwen example shows 896-token blocks. Replace them with the sizes reported by your worker, such as 640 for Qwen3.5-4B.
- Set `hash_block_size` to your worker's actual value when Router cannot discover it.

<details>
<summary>GPU · Qwen3.8-27B with MTP and one hash lookahead token</summary>

```json
{
  "rank": 0,
  "hash_block_size": 16,
  "speculative_tokens": 3,
  "hash_lookahead_tokens": 1,
  "partial_hash_hits": true,
  "groups": [
    {"kind": "mamba", "block_size": 832, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 832, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 832, "mamba_cache_mode": "align"},
    {"kind": "full_attention", "block_size": 832}
  ]
}
```

</details>

<details>
<summary>GPU · Qwen3.8-27B with DSpark</summary>

```json
{
  "rank": 0,
  "hash_block_size": 16,
  "speculative_tokens": 8,
  "hash_lookahead_tokens": 0,
  "tail_drop_groups": [],
  "groups": [
    {"kind": "mamba", "block_size": 832, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 832, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 832, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 832, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 832, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 832, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 832, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 832, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 832, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 832, "mamba_cache_mode": "align"},
    {"kind": "full_attention", "block_size": 832},
    {"kind": "full_attention", "block_size": 832},
    {"kind": "full_attention", "block_size": 832},
    {"kind": "full_attention", "block_size": 832},
    {"kind": "sliding_window", "block_size": 832, "sliding_window": 2048}
  ]
}
```

</details>

<details>
<summary>GPU · Qwen3.8-Flash-Next-FP8 (Qwen4Exp QSA)</summary>

```json
{
  "rank": 0,
  "hash_block_size": 1600,
  "speculative_tokens": 0,
  "groups": [
    {"kind": "mamba", "block_size": 1600, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 1600, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 1600, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 1600, "mamba_cache_mode": "align"},
    {"kind": "scratch", "block_size": 4, "tokens_per_state": "1"},
    {"kind": "mla_attention", "block_size": 1600, "cache_role": "sparse", "tokens_per_state": "4"}
  ]
}
```

</details>

<details>
<summary>GPU · DeepSeek-V4-Flash-0731</summary>

```json
{
  "rank": 0,
  "hash_block_size": 4,
  "speculative_tokens": 0,
  "groups": [
    {"kind": "sliding_window_mla", "block_size": 64, "sliding_window": 128},
    {"kind": "sliding_window_mla", "block_size": 64, "sliding_window": 128},
    {"kind": "mla_attention", "block_size": 256, "cache_role": "sparse", "tokens_per_state": "4"},
    {"kind": "sliding_window_mla", "block_size": 4, "sliding_window": 8},
    {"kind": "sliding_window_mla", "block_size": 8, "sliding_window": 128}
  ]
}
```

</details>

<details>
<summary>GPU · DeepSeek-V4-Pro</summary>

```json
{
  "rank": 0,
  "hash_block_size": 4,
  "speculative_tokens": 0,
  "groups": [
    {"kind": "sliding_window_mla", "block_size": 64, "sliding_window": 128},
    {"kind": "sliding_window_mla", "block_size": 64, "sliding_window": 128},
    {"kind": "mla_attention", "block_size": 256, "cache_role": "sparse", "tokens_per_state": "128"},
    {"kind": "sliding_window_mla", "block_size": 8, "sliding_window": 128},
    {"kind": "sliding_window_mla", "block_size": 4, "sliding_window": 8}
  ]
}
```

</details>

<details>
<summary>GPU · DeepSeek-V4.1-Flash (1017) with bounded replay</summary>

```json
{
  "rank": 0,
  "hash_block_size": 128,
  "speculative_tokens": 0,
  "groups": [
    {"kind": "sliding_window_mla", "block_size": 32, "sliding_window": 128, "prefix_replay_tokens": 128},
    {"kind": "sliding_window_mla", "block_size": 32, "sliding_window": 128, "prefix_replay_tokens": 128},
    {"kind": "sliding_window_mla", "block_size": 32, "sliding_window": 128, "prefix_replay_tokens": 128},
    {"kind": "sliding_window_mla", "block_size": 32, "sliding_window": 128, "prefix_replay_tokens": 128},
    {"kind": "sliding_window_mla", "block_size": 32, "sliding_window": 128, "prefix_replay_tokens": 128},
    {"kind": "mla_attention", "block_size": 128, "cache_role": "sparse", "tokens_per_state": "2"},
    {"kind": "scratch", "block_size": 8, "tokens_per_state": "1"}
  ]
}
```

</details>

<details>
<summary>Ascend A2 · Qwen3-4B, manual full-attention layout</summary>

```json
{
  "rank": 0,
  "hash_block_size": 128,
  "speculative_tokens": 0,
  "declared_groups": 1,
  "groups": [
    {"kind": "full_attention", "block_size": 128}
  ]
}
```

</details>

<details>
<summary>Ascend A2 · Qwen3.5-4B, Qwen3.6-27B W8A8 and Qwen3.8-27B W8A8</summary>

```json
{
  "rank": 0,
  "hash_block_size": 896,
  "speculative_tokens": 0,
  "groups": [
    {"kind": "full_attention", "block_size": 896},
    {"kind": "mamba", "block_size": 896, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 896, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 896, "mamba_cache_mode": "align"}
  ]
}
```

</details>

<details>
<summary>Ascend A2 · Qwen3.5-4B with DSpark</summary>

```json
{
  "rank": 0,
  "hash_block_size": 640,
  "speculative_tokens": 8,
  "hash_lookahead_tokens": 0,
  "tail_drop_groups": [],
  "groups": [
    {"kind": "full_attention", "block_size": 640},
    {"kind": "full_attention", "block_size": 640},
    {"kind": "sliding_window", "block_size": 640, "sliding_window": 2048},
    {"kind": "mamba", "block_size": 640, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 640, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 640, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 640, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 640, "mamba_cache_mode": "align"}
  ]
}
```

</details>

<details>
<summary>Ascend A2 · Qwen3.8-27B BF16 with DSpark</summary>

```json
{
  "rank": 0,
  "hash_block_size": 896,
  "speculative_tokens": 8,
  "hash_lookahead_tokens": 0,
  "tail_drop_groups": [],
  "groups": [
    {"kind": "full_attention", "block_size": 896},
    {"kind": "full_attention", "block_size": 896},
    {"kind": "full_attention", "block_size": 896},
    {"kind": "full_attention", "block_size": 896},
    {"kind": "sliding_window", "block_size": 896, "sliding_window": 2048},
    {"kind": "mamba", "block_size": 896, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 896, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 896, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 896, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 896, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 896, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 896, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 896, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 896, "mamba_cache_mode": "align"},
    {"kind": "mamba", "block_size": 896, "mamba_cache_mode": "align"}
  ]
}
```

</details>

<details>
<summary>Ascend A2 · Qwen3.8-Flash-Next W4A8 (Qwen4Exp) with MTP</summary>

```json
{
  "rank": 0,
  "hash_block_size": 1664,
  "speculative_tokens": 3,
  "hash_lookahead_tokens": 0,
  "tail_drop_groups": [],
  "groups": [
    {"kind": "scratch", "block_size": 8, "tokens_per_state": "1"},
    {"kind": "full_attention", "block_size": 1664, "tokens_per_state": "1"},
    {"kind": "mamba", "block_size": 1664, "mamba_cache_mode": "align", "tokens_per_state": "-1"},
    {"kind": "mamba", "block_size": 1664, "mamba_cache_mode": "align", "tokens_per_state": "-1"},
    {"kind": "mamba", "block_size": 1664, "mamba_cache_mode": "align", "tokens_per_state": "-1"},
    {"kind": "mamba", "block_size": 1664, "mamba_cache_mode": "align", "tokens_per_state": "-1"},
    {"kind": "mamba", "block_size": 1664, "mamba_cache_mode": "align", "tokens_per_state": "-1"}
  ]
}
```

</details>

<details>
<summary>Ascend A2 · DeepSeek-V4-Flash-0731 W8A8</summary>

```json
{
  "rank": 0,
  "hash_block_size": 8,
  "speculative_tokens": 0,
  "groups": [
    {"kind": "mla_attention", "block_size": 512, "cache_role": "sparse", "tokens_per_state": "4"},
    {"kind": "mla_attention", "block_size": 16384, "cache_role": "sparse", "tokens_per_state": "128"},
    {"kind": "sliding_window_mla", "block_size": 128, "sliding_window": 128, "tokens_per_state": "1"},
    {"kind": "sliding_window_mla", "block_size": 128, "sliding_window": 128, "tokens_per_state": "1"},
    {"kind": "sliding_window_mla", "block_size": 8, "sliding_window": 8, "tokens_per_state": "1"},
    {"kind": "sliding_window_mla", "block_size": 32, "sliding_window": 128, "tokens_per_state": "1"}
  ]
}
```

</details>

### MTP and DSpark

Use a [model example](#model-layout-reference) that matches the worker's cache groups. Set the fields below next to `rank` and `groups`, replacing any existing values for those fields. Use the worker's `num_speculative_tokens` for `speculative_tokens`.

Use the worker's actual hash lookahead. For zero lookahead, include `tail_drop_groups` even when it is empty. In P/D, configure each scored rank using the rules for its stage. For Qwen3.8-27B W8A8 on Ascend A2 with gRPC, use the MTP profile.

| Worker and mode | Fields in the `ranks[]` object |
| --- | --- |
| GPU Qwen3.8, MTP with hash lookahead | `"speculative_tokens":3,"hash_lookahead_tokens":1` |
| GPU Qwen3.8, MTP with `disable_lookahead_block_hashes` | `"speculative_tokens":3,"hash_lookahead_tokens":0,"tail_drop_groups":[3]` |
| GPU Qwen3.8, DSpark | `"speculative_tokens":8,"hash_lookahead_tokens":0,"tail_drop_groups":[]` |
| GPU DeepSeek V4 Flash, DSpark | `"speculative_tokens":4,"hash_lookahead_tokens":0,"tail_drop_groups":[]` |
| Ascend A2 Qwen3.5, Qwen3.6, Qwen3.8 W8A8 or Qwen4Exp, MTP | `"speculative_tokens":3,"hash_lookahead_tokens":0,"tail_drop_groups":[]` |
| Ascend A2 Qwen3.5, Qwen3.6 or Qwen3.8 BF16, DSpark | `"speculative_tokens":8,"hash_lookahead_tokens":0,"tail_drop_groups":[]` |
| Ascend A2 DeepSeek V4 Flash, DSpark | `"speculative_tokens":7,"hash_lookahead_tokens":0,"tail_drop_groups":[]` |

For GPU Qwen3.8 MTP, use the four base groups. With zero hash lookahead, replace the rank fields as shown in the table. With DSpark, copy the full DSpark rank object because the groups change.

For Ascend Qwen4Exp with speculation off, use one Full group and four Mamba groups after scratch. In the example, set scratch `block_size` to `4`, remove the last Mamba group, set `speculative_tokens` to `0`, and omit `hash_lookahead_tokens` and `tail_drop_groups`.

## Static event endpoints

Use static event options when a worker cannot report its event sources, such as stock HTTP vLLM 0.30.0. This single-group example uses the default hash seed. Set `KV_TOPIC` to the current worker topic, `SERVED_MODEL` to its served model ID and the block size to its actual value.

```bash
vllm-router --policy kv_aware --worker-urls http://127.0.0.1:8000 \
  --kv-event-source 'http://127.0.0.1:8000=tcp://127.0.0.1:5557' \
  --kv-event-replay-source 'http://127.0.0.1:8000=tcp://127.0.0.1:5558' \
  --kv-event-topic "http://127.0.0.1:8000=$KV_TOPIC" \
  --kv-event-model "$SERVED_MODEL" --kv-event-block-size 16
```

In P/D, use `--pd-kv-events` for the live address. List publishers, replay addresses and topics in DP-rank order. Add a [layout file](#layout-template) only for group or hash rules the Router cannot derive.

## Check cache reuse

The Router estimates placement from the KV events it has applied. The worker reports actual reuse. Requests with an unrecognized adapter, cache salt or extension still route, with no cache credit.

In an idle worker pool, send the same token-ID prompt twice. Use the model ID returned by `/v1/models` and a prompt longer than the worker's cache-hit alignment. The example uses 129 tokens for a full-attention worker with 16-token blocks. Leave room for one output token.

```bash
SERVED_MODEL="SERVED_MODEL_ID"
PROMPT_TOKENS=129
payload=$(jq -nc --arg model "$SERVED_MODEL" --argjson n "$PROMPT_TOKENS" \
  '{model:$model,prompt:[range(1000;1000+$n)],max_tokens:1,temperature:0}')
for delay in 0 1; do
  sleep "$delay"
  curl -fsS http://127.0.0.1:30000/v1/completions \
    -H 'Content-Type: application/json' --data-binary "$payload"
  printf '\n'
done
```

| Check | Where to look | Success signal |
| --- | --- | --- |
| Router placement | Router `/metrics` on port `29000`. Look for `vllm_router_kv_placement_lookups_total{result="hit"}`. | Increases on the repeat after events arrive. |
| Worker local reuse | Worker `/metrics`. Look for `vllm:prefix_cache_hits_total`. | Increases on the repeat. For P/D, inspect Prefill. |
| P/D transfer | Send a fresh prompt. Compare Decode's `vllm:nixl_bytes_transferred_sum` counter. | Requests that need remote KV complete a read. |

## Reading KV prediction metrics

For HTTP workers, the predicted and actual cached-token histograms count the
same attempts. For full attention, the prediction counts the device prefix plus
the longest indexed continuation that touches its end, counting overlapping
tokens once. Hybrid layouts use the prefix reported by their restore rules.
The prediction estimates vLLM's total local and external cached tokens.
Transfer discounts apply only to routing costs. Use the outcomes counter to check how much
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
after reconciliation with the same attempt's prefill report. Missing prefill
usage never falls back to decode usage. Opaque responses do not supply accuracy
samples. Program-bound P/D paths that bypass KV prediction and gRPC workers do
not supply these pairs. The metric assumes the backend reports total cached tokens. A local-only
backend counter has different semantics.

JSON and SSE share the same attempt receipt; a failed dispatch, cancellation,
or response observation can finalize it only once. Multi-prompt predictions
carry no aggregate cached-token count and cannot enter the accuracy histograms.
Conflicting SSE counts are `invalid_usage`, rather than a last-value-wins sample.
A usage frame contributes an accuracy pair only after a clean stream EOF.
An SSE error envelope discards the provisional count even if the HTTP body
ends cleanly. The `stream_error` outcome also covers upstream body failures
and downstream client cancellations before EOF.

Ordinary transparent non-streaming responses forward each chunk immediately and
retain at most 1 MiB for observation. Larger bodies are returned intact and
record `oversized_json`; the observation buffer is released when exceeded.
Program-tracked and typed JSON responses retain their existing complete-body
buffering, with the same 1 MiB observation budget. SSE uses
one framer with the same 1 MiB budget for both ordinary observation and P/D
reconciliation. Ordinary observation forwards original fragments immediately;
P/D reconciliation buffers one frame to merge usage. Oversized or incomplete
frames do not contribute accuracy samples. Transport chunks and serialized
P/D frames occupy additional memory beyond the framing accumulator.

## Router options

Both the Rust and Python launchers accept these options. Configure vLLM workers with the separate [worker options](#worker-requirements).

### Routing and layout

| Option | Value | Default | Purpose |
| --- | --- | --- | --- |
| `--policy` | Policy name | `cache_aware` | Set `kv_aware` for colocated routing or as the default policy for both P/D stages. |
| `--worker-urls` | List of worker URLs | Empty list | Register a colocated pool of HTTP or gRPC workers. |
| `--intra-node-data-parallel-size` | Positive integer | `1` | Number of DP ranks per worker URL for DP-aware routing. |
| `--vllm-pd-disaggregation` | Flag | Off | Enable P/D routing. |
| `--prefill` | `URL [PORT or none]`, repeatable | No workers | Register a Prefill worker with an optional bootstrap port. |
| `--decode` | `URL`, repeatable | No workers | Register a Decode worker. |
| `--prefill-policy` | Policy name | Inherits `--policy` | Set `kv_aware` to score Prefill prefix reuse. |
| `--decode-policy` | Policy name | Inherits `--policy` | Set `kv_aware` to route Decode by context load. Enable Decode KV events to credit cache reuse. |
| `--kv-connector` | `nixl`, `mooncake`, or `moriio` | `nixl` | Match the connector used by the P/D workers. MoRI-IO requires vLLM ZMQ discovery. |
| `--vllm-discovery-address` | `HOST:PORT` | Unset | Listen for vLLM ZMQ worker registration when using P/D discovery. Use explicit URLs for gRPC P/D. |
| `--pd-pair` | `PREFILL=DECODE[,DECODE...]`, repeatable | All peers in one compatible pool | Restrict a mixed pool to the declared P/D pairs. Supply one entry per Prefill. |
| `--kv-hybrid-layout` | JSON file path | Unset | Supply cache groups, hashing, and reuse rules when the Router cannot derive them or read them from the worker. |

### Static KV events

Use these options when workers cannot report their event sources. Provide endpoint and topic lists in DP-rank order. See the [static setup example](#static-event-endpoints).

| Option | Value | Default | Purpose |
| --- | --- | --- | --- |
| `--kv-event-source` | `WORKER_URL=ENDPOINT[,ENDPOINT...]`, repeatable | Empty list | Set live ZMQ publisher addresses for colocated workers. |
| `--pd-kv-events` | `WORKER_URL=ENDPOINT[,ENDPOINT...]`, repeatable | Empty list | Set live ZMQ publisher addresses for workers in a stage that uses `kv_aware`. |
| `--kv-event-replay-source` | `WORKER_URL=ENDPOINT[,ENDPOINT...]`, repeatable | Empty list | Set replay addresses for the worker's `--kv-event-source` or `--pd-kv-events` entry. |
| `--kv-event-topic` | `WORKER_URL=TOPIC[,TOPIC...]`, repeatable | Empty list | Supply each static publisher's current event topic. |
| `--kv-event-model` | Served model ID | Unset | Set the model ID for static publishers. Use the ID from the worker's `/v1/models`. |
| `--kv-event-block-size` | Positive integer in tokens | `16` | Set the hash block size for static publishers. Match the worker's hash unit. |

### Scoring and recovery

Transfer credits are fractions of reusable tokens counted as work avoided. `0` ignores reuse from that tier and `1` gives full credit.

| Option | Value | Default | Purpose |
| --- | --- | --- | --- |
| `--router-temperature` | Nonnegative float | `0.0` | Set how strongly colocated and P/D Prefill selection favors lower cost. Zero picks a minimum-cost worker. |
| `--kv-penalty-host` | Float in `[0, 1]` | `0.8` | Set credit for reusable KV in worker-local host memory. |
| `--kv-penalty-external` | Float in `[0, 1]` | `0.6` | Set credit for confirmed external KV. Stock worker events alone cannot establish shared-pool residency. |
| `--kv-penalty-disk` | Float in `[0, 1]` | `0.3` | Set credit for reusable KV on disk. |
| `--kv-aware-session-affinity-ttl-secs` | Integer from `1` to `31536000` seconds | Off | Enable session affinity with an idle expiration. In P/D, apply a soft preference to the session's Decode worker. |
| `--kv-event-strict-recovery` | Flag | Off | After loss of event history, require complete recovery or a worker KV cache clear before restoring cache credit. Requests continue routing by load during recovery. |

### Verification

| Option | Value | Default | Purpose |
| --- | --- | --- | --- |
| `--port` | TCP port | `30000` | Set the Router's inference API port. |
| `--prometheus-host` | Bind address | `127.0.0.1` | Set the Router's metrics listening address. |
| `--prometheus-port` | TCP port | `29000` | Set the port for [Router cache metrics](#check-cache-reuse). |
| `--log-level` | Log level | `info` | Use `debug` to diagnose rejected layouts or event sources. |

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Worker reports reuse and Router stays at zero | Check that the worker publishes events and Router can reach both event ports. For a static source, confirm the model ID and publisher topic. |
| Router rejects the layout | Use `--log-level debug` and inspect `worker-reported KV layout rejected`. Match the JSON to the actual model, rank, groups and speculation settings. |
| Worker reports `Address already in use` | Check HTTP, gRPC, event and connector ports. Reserve separate ranges for DP ranks. |
| Worker restarted | Generate a new publisher topic. Update `--kv-event-topic` for a static source. A Router restart uses the available replay history. |
| Decode has no remote read | Check whether the prompt was already cached on Decode, the connector and role settings, and Decode's transfer log. |
