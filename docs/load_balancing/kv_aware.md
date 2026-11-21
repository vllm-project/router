# KV-aware routing

The `kv_aware` policy prefers workers that can reuse a request's prefix and accounts for worker load. Clients use the Router's OpenAI HTTP API with either HTTP or gRPC workers.

## Versions

Pair `vllm-rs` with the same vLLM release as its Python engine. Use the Rust `vllm-router` binary for the examples below.

| vLLM | HTTP event sources | gRPC event sources | Layout and hashing | vLLM Ascend pairing |
| --- | --- | --- | --- | --- |
| 0.30.0 | [Static addresses and layout](#static-event-endpoints) | `Control.GetKvEventSources` | Zero hash lookahead | [0.30.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.30.0) |
| 0.31.0–0.32.0 | `/kv_event_sources` | `Control.GetKvEventSources` | Zero hash lookahead. Supports V4.1 bounded replay with the corresponding model layout. | [0.31.0rc1](https://pypi.org/project/vllm-ascend/0.31.0rc1/) or [0.32.0rc1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.32.0rc1), respectively |
| 0.33.0 | `/kv_event_sources` | `Control.GetKvEventSources` | Supports nonzero hash lookahead carried by `BlockStored.lookahead_token_ids`. | Build against vLLM 0.33.0 |
| 0.34.0 | `/kv_event_sources` | `Control.GetKvEventSources` | Uses 64-token blocks when the backend has no preferred size. Use the worker's actual sizes in a layout. | Build against vLLM 0.34.0 |

For gRPC, use one interface for the whole worker pool. P/D uses static worker URLs. Router renders chat locally using `vllm-chat` pinned to vLLM 0.30.0. See the [gRPC API compatibility table](../backend/grpc.md#api-compatibility) for request features and tokenizer settings.

## Before deployment

### Worker requirements

| Setting | Value |
| --- | --- |
| Prefix caching | `--enable-prefix-caching --prefix-caching-hash-algo sha256_cbor` |
| Event hashes | `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0` publishes the full 32-byte hashes. |
| Event reporting | Keep incremental reporting and enable KV events through `--kv-events-config`. Router needs reachable live and replay endpoints. |
| Publisher identity | Generate a fresh topic on each worker start. |
| HTTP text prompts | Start the worker with `--enable-scale-out` to expose rendering endpoints. |

### Explicit layouts

| Worker configuration | What to pass to Router |
| --- | --- |
| Ordinary single-group full attention with event discovery, default hash root, no sliding window or speculative decoding | Omit `--kv-hybrid-layout`. Leave `PYTHONHASHSEED` unset. Router confirms the model, block size and hash root before granting reuse credit. |
| Hybrid attention, sliding-window attention, MTP, DSpark or a custom hash seed | Use the worker's complete layout metadata when available. Otherwise save the matching [model layout](#hybrid-attention-models) as `layout.json` and pass `--kv-hybrid-layout layout.json`. |
| Stock HTTP vLLM 0.30.0 | Use the [static event setup](#static-event-endpoints). |
| Hybrid P/D | Supply a [P/D layout file](#pd-layout) with compatible Decode addresses. Include layout details only for scored workers that cannot report them. |

Use the worker URL as the JSON key and one rank entry per DP rank. Use the group order, sizes and effective hash settings of the deployed worker. For fixed event addresses, include the actual served model and publisher topic per rank. For a custom seed, set the same `hash_seed` in the layout and `PYTHONHASHSEED` on the worker. To score LoRA reuse, add `"adapters": [{"name": "my-adapter", "path": "/models/my-adapter"}]` beside `ranks` using the path passed to `--lora-modules`.

### Set the event ports

Run in each worker terminal before launching vLLM. The [P/D port table](#pd-routing) gives separate values for each stage.

```bash
export VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0
LIVE_PORT=${LIVE_PORT:-5557}
REPLAY_PORT=${REPLAY_PORT:-5558}
KV_TOPIC=$(python3 -c 'from uuid import uuid4; print(uuid4().hex)')
EVENTS=$(printf '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:%s","replay_endpoint":"tcp://*:%s","topic":"%s"}' \
  "$LIVE_PORT" "$REPLAY_PORT" "$KV_TOPIC")
KV_ARGS=(--enable-prefix-caching --prefix-caching-hash-algo sha256_cbor --kv-events-config "$EVENTS")
```

Use distinct ports for workers on the same host. vLLM adds the DP rank to each fixed event port. Reserve separate ranges for live and replay when using multiple ranks. With event discovery on vLLM 0.31–0.34, set `REPLAY_PORT=0` to let vLLM allocate the replay port.

## Colocated prefill and decode

Set the event ports and copy one model's environment and `MODEL_ARGS`. Start with [Qwen3-4B on GPU](#qwen3-4b-on-gpu) for full attention or choose a [hybrid recipe](#hybrid-attention-models). Use the [A2 Qwen3.8 W8A8 MTP recipe](#qwen38-27b-with-mtp-on-ascend) for the Ascend gRPC command.

In the Router terminal, set `LAYOUT_ARGS=()` for automatic full-attention layout discovery. For an explicit layout, use `LAYOUT_ARGS=(--kv-hybrid-layout layout.json)`.

### HTTP workers

```bash
vllm serve "$MODEL" --host 127.0.0.1 --port 8000 --served-model-name "$SERVED_MODEL" \
  --enable-scale-out "${KV_ARGS[@]}" "${MODEL_ARGS[@]}"
```

```bash
vllm-router --host 127.0.0.1 --port 19000 --policy kv_aware \
  --worker-urls http://127.0.0.1:8000 "${LAYOUT_ARGS[@]}"
```

### gRPC workers

| Platform | Worker launcher | Engine options |
| --- | --- | --- |
| GPU | `vllm-rs serve` | Pass model and KV engine flags after `--`. |
| Ascend A2 with vLLM 0.32.0 | `VLLM_USE_RUST_FRONTEND=1 vllm serve` | Use `--grpc-port` for vLLM's Rust inference and control services. |

```bash
# GPU
vllm-rs serve "$MODEL" --host 127.0.0.1 --port 8000 --grpc-port 50051 \
  --served-model-name "$SERVED_MODEL" -- "${KV_ARGS[@]}" "${MODEL_ARGS[@]}"
```

```bash
# Ascend A2
VLLM_USE_RUST_FRONTEND=1 vllm serve "$MODEL" --host 127.0.0.1 \
  --port 8000 --grpc-port 50051 --served-model-name "$SERVED_MODEL" \
  --enable-scale-out "${KV_ARGS[@]}" "${MODEL_ARGS[@]}"
```

Set `MODEL` in the Router terminal to the local model directory with tokenizer files.

```bash
VLLM_ROUTER_TOKENIZER="$MODEL" vllm-router --host 127.0.0.1 --port 19000 \
  --policy kv_aware --worker-urls grpc://127.0.0.1:50051 "${LAYOUT_ARGS[@]}"
```

## P/D routing

### Choose a connector

Choose the worker connector from the model and platform. Use the same model checkpoint, engine version and connector on both stages.

| Platform | Models and modes | Worker interface | Worker connector | Router `--kv-connector` |
| --- | --- | --- | --- | --- |
| GPU | Qwen3-4B, Qwen3.5-4B, DeepSeek V4 Flash 0731 with DSpark | HTTP | `NixlConnector` | `nixl` |
| GPU | Qwen3.8-27B MTP with zero hash lookahead, or DSpark | HTTP | `MooncakeConnector` | `mooncake` |
| Ascend A2 | Qwen3-4B, Qwen3.5-4B, Qwen3.6-27B W8A8, Qwen3.8-27B BF16 with DSpark | HTTP | `MooncakeConnectorV1` | `mooncake` |
| Ascend A2 | Qwen3.8-27B W8A8 | HTTP; gRPC with MTP | `MooncakeConnectorV1` | `mooncake` |
| Ascend A2 | Qwen4Exp W4A8 with MTP or off | HTTP | `MooncakeConnectorV2` | `mooncake` |

Use `CONNECTOR=nixl` or `CONNECTOR=mooncake` in the Router terminal. HTTP and gRPC describe the Router-to-worker interface. The connector transfers KV between Prefill and Decode.

### Set worker ports and connector parameters

Start Prefill and Decode in separate terminals. After loading the same recipe in each terminal, assign disjoint devices with `CUDA_VISIBLE_DEVICES` or `ASCEND_RT_VISIBLE_DEVICES`. The examples bind to localhost. For separate hosts, use reachable bind and advertised addresses.

| Setting | Prefill | Decode |
| --- | --- | --- |
| Worker HTTP port | `8000` | `8001` |
| Worker gRPC port, if used | `50051` | `50052` |
| `LIVE_PORT` and `REPLAY_PORT` | `5557` / `5558` | `5559` / `5560` |
| Connector role | `kv_producer` | `kv_consumer` |
| NIXL side-channel port | `5600` | `5601` |
| Ascend Mooncake `kv_port` | `38352` | `38353` |

Copy the settings for each stage into its terminal, then run the [event setup](#set-the-event-ports) to build `KV_ARGS`. For Ascend, set `TP` from the model recipe and choose `MOONCAKE_CONNECTOR` from the connector table.

```bash
# Prefill terminal
HTTP_PORT=8000 GRPC_PORT=50051 LIVE_PORT=5557 REPLAY_PORT=5558
KV_ROLE=kv_producer NIXL_PORT=5600 KV_PORT=38352 ENGINE_ID=prefill-a2
```

```bash
# Decode terminal
HTTP_PORT=8001 GRPC_PORT=50052 LIVE_PORT=5559 REPLAY_PORT=5560
KV_ROLE=kv_consumer NIXL_PORT=5601 KV_PORT=38353 ENGINE_ID=decode-a2
```

Set `TRANSFER` in each terminal using the connector below.

<details>
<summary>GPU NIXL</summary>

Set `NIXL_PORT` to the side-channel port for this worker. For Qwen3.5, export `VLLM_SSM_CONV_STATE_LAYOUT=DS` on both stages.

```bash
export VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1
export VLLM_NIXL_SIDE_CHANNEL_PORT="$NIXL_PORT"
TRANSFER=$(printf '{"kv_connector":"NixlConnector","kv_role":"%s"}' "$KV_ROLE")
```

</details>

<details>
<summary>GPU Mooncake</summary>

Install the [Mooncake Transfer Engine for your CUDA version](https://docs.vllm.ai/en/latest/features/mooncake_connector_usage/#installation). Use the same bootstrap port in the Prefill and Router commands.

```bash
export VLLM_MOONCAKE_BOOTSTRAP_PORT=8998
TRANSFER=$(printf '{"kv_connector":"MooncakeConnector","kv_role":"%s"}' "$KV_ROLE")
```

</details>

<details>
<summary>Ascend Mooncake</summary>

Set `MOONCAKE_CONNECTOR` to the worker connector from the table. Set `KV_PORT` from the port table and assign different `ENGINE_ID` values, such as `prefill-a2` and `decode-a2`. Set `TP` to the recipe's tensor parallel size for each stage. The following config uses DP1 and the same TP on both stages.

```bash
TRANSFER=$(printf '{"kv_connector":"%s","kv_role":"%s","kv_buffer_device":"npu","engine_id":"%s","kv_port":%s,"kv_load_failure_policy":"fail","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":%s},"decode":{"dp_size":1,"tp_size":%s}}}' \
  "$MOONCAKE_CONNECTOR" "$KV_ROLE" "$ENGINE_ID" "$KV_PORT" "$TP" "$TP")
```

</details>

### HTTP P/D

Run on Prefill first, then Decode, with the port, role and event settings for each stage. Set `LAYOUT_ARGS=(--kv-hybrid-layout layout.json)` when using a [P/D layout](#pd-layout), or `LAYOUT_ARGS=()` for ordinary full-attention discovery. Set `PREFILL_BOOTSTRAP=()` for NIXL and Ascend Mooncake, or `PREFILL_BOOTSTRAP=(8998)` for GPU Mooncake.

```bash
vllm serve "$MODEL" --host 127.0.0.1 --port "$HTTP_PORT" --served-model-name "$SERVED_MODEL" \
  --enable-scale-out "${KV_ARGS[@]}" "${MODEL_ARGS[@]}" --kv-transfer-config "$TRANSFER"
```

```bash
vllm-router --port 19000 --vllm-pd-disaggregation \
  --prefill http://127.0.0.1:8000 "${PREFILL_BOOTSTRAP[@]}" --decode http://127.0.0.1:8001 \
  --prefill-policy kv_aware --decode-policy round_robin --kv-connector "$CONNECTOR" \
  "${LAYOUT_ARGS[@]}"
```

### gRPC P/D

For Ascend A2, use [Qwen3.8 W8A8 MTP](#qwen38-27b-with-mtp-on-ascend). Set `MOONCAKE_CONNECTOR=MooncakeConnectorV1` on both workers before building `TRANSFER`. Run the worker command on Prefill first and then Decode with the stage settings above.

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve "$MODEL" --host 127.0.0.1 \
  --port "$HTTP_PORT" --grpc-port "$GRPC_PORT" --served-model-name "$SERVED_MODEL" \
  --enable-scale-out "${KV_ARGS[@]}" "${MODEL_ARGS[@]}" --kv-transfer-config "$TRANSFER"
```

Set `MODEL` in the Router terminal to the local tokenizer directory. Save the model's [gRPC P/D layout](#qwen38-27b-with-mtp-on-ascend) as `layout.json`.

```bash
VLLM_ROUTER_TOKENIZER="$MODEL" vllm-router --port 19000 --vllm-pd-disaggregation \
  --prefill grpc://127.0.0.1:50051 --decode grpc://127.0.0.1:50052 \
  --prefill-policy kv_aware --decode-policy round_robin --kv-connector mooncake \
  --kv-hybrid-layout layout.json
```

To score Decode reuse as well, set `--decode-policy kv_aware`, enable Decode's events and include its layout when required. For HTTP Decode scoring, supply explicit prompt token IDs. Each gRPC stage uses the token IDs produced by Router. Use one worker interface throughout the P/D pool.

### P/D layout

Save the model recipe as `layout.json` and pass it to Router with `--kv-hybrid-layout`. Use the Qwen3.5-4B GPU recipe below for a two-stage NIXL example.

| Entry | Required contents |
| --- | --- |
| Hybrid Prefill | `compatible_decode_urls` plus the Prefill layout when the worker does not report it |
| Decode scored by `kv_aware` | The Decode layout when the worker does not report it |
| Decode using `round_robin` | List its URL under the Prefill's `compatible_decode_urls` |

Use the same URL scheme and address as the Router command. Leave Program scheduling disabled for hybrid P/D layouts.

<details>
<summary>Pairing-only layout.json when the workers report their layouts</summary>

```json
{
  "http://127.0.0.1:8000": {
    "compatible_decode_urls": [
      "http://127.0.0.1:8001"
    ]
  }
}
```

</details>

## Check cache reuse

In an idle worker pool, send the same token-ID prompt twice. Set `SERVED_MODEL` to the worker's served name and `PROMPT_TOKENS` to the length in its recipe.

```bash
SERVED_MODEL=qwen3
PROMPT_TOKENS=129
payload=$(jq -nc --arg model "$SERVED_MODEL" --argjson n "$PROMPT_TOKENS" \
  '{model:$model,prompt:[range(1000;1000+$n)],max_tokens:1,temperature:0}')
for delay in 0 1; do
  sleep "$delay"
  curl -fsS http://127.0.0.1:19000/v1/completions \
    -H 'Content-Type: application/json' --data-binary "$payload"
  printf '\n'
done
```

| Check | Where to look | Success signal |
| --- | --- | --- |
| Router placement | Router `/metrics` on port `29000`. Look for `vllm_router_kv_placement_lookups_total{result="hit"}`. | Increases on the repeat after events arrive. |
| Worker local reuse | Worker `/metrics`. Look for `vllm:prefix_cache_hits_total`. With `--enable-prompt-tokens-details` on HTTP, inspect `usage.prompt_tokens_details.cached_tokens`. | Increases on the repeat. For P/D, inspect Prefill. |
| P/D transfer | Send a fresh prompt. Read Decode's Mooncake completion log or compare its `vllm:nixl_bytes_transferred_sum` counter. | Requests that need remote KV complete a read. |
| MTP or DSpark | On the worker generating tokens, inspect `vllm:spec_decode_num_draft_tokens_total` or the speculative-decoding log. | Drafted tokens increase during generation. |

## Model recipes

Copy the shell block for the selected model before starting each worker. Set `MODEL` to the local checkpoint path. Copy the matching JSON when the routing path needs a layout. The examples use one rank. To use a gRPC worker, use its `grpc://` address as the JSON key.

### GPU full attention

#### Qwen3-4B on GPU

Use Qwen3-4B-Instruct-2507 with the default hash root. Automatic discovery uses one 16-token full-attention group.

```bash
MODEL=/models/Qwen3-4B-Instruct-2507
SERVED_MODEL=qwen3
PROMPT_TOKENS=129
unset PYTHONHASHSEED
MODEL_ARGS=(--block-size 16)
```

### Hybrid attention models

The GPU recipes use vLLM 0.33.0. The layout and limits apply to the named checkpoint and mode.

#### Qwen3.5-4B

<details>
<summary>Worker settings and layout.json for colocated or HTTP P/D</summary>

```bash
MODEL=/models/Qwen3.5-4B
SERVED_MODEL=qwen35
PROMPT_TOKENS=2001
unset PYTHONHASHSEED
MODEL_ARGS=(--max-model-len 4096 --prefix-match-unit 16)
```

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

#### Qwen3.6-27B

<details>
<summary>Worker settings and layout.json</summary>

```bash
MODEL=/models/Qwen3.6-27B
SERVED_MODEL=qwen36
PROMPT_TOKENS=2001
unset PYTHONHASHSEED
MODEL_ARGS=(--max-model-len 4096 --prefix-match-unit 16)
```

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

#### Qwen3.8-27B

| Mode | Setup |
| --- | --- |
| Off | Use the base model profile below. |
| MTP, one token of hash lookahead | Use the colocated MTP profile. |
| MTP, zero hash lookahead | Use the zero-lookahead profile for colocated serving or Mooncake P/D. |
| DSpark | Use the matching draft checkpoint for colocated serving or Mooncake P/D. |

<details>
<summary>Off</summary>

```bash
MODEL=/models/Qwen3.8-27B
SERVED_MODEL=qwen38
PROMPT_TOKENS=2001
unset PYTHONHASHSEED
MODEL_ARGS=(--max-model-len 4096 --prefix-match-unit 16)
```

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

##### Qwen3.8-27B with MTP

Use the profile for the worker's effective hash lookahead.

<details>
<summary>MTP with one token of hash lookahead</summary>

```bash
MODEL=/models/Qwen3.8-27B
SERVED_MODEL=qwen38
PROMPT_TOKENS=2001
unset PYTHONHASHSEED
MODEL_ARGS=(--max-model-len 4096 --prefix-match-unit 16 --speculative-config '{"method":"mtp","num_speculative_tokens":3}')
```

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

<details>
<summary>MTP with zero hash lookahead</summary>

```bash
MODEL=/models/Qwen3.8-27B
SERVED_MODEL=qwen38
PROMPT_TOKENS=2001
unset PYTHONHASHSEED
MODEL_ARGS=(--max-model-len 4096 --prefix-match-unit 16 --speculative-config '{"method":"mtp","num_speculative_tokens":3,"disable_lookahead_block_hashes":true}')
```

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
    ],
    "compatible_decode_urls": [
      "http://127.0.0.1:8001"
    ]
  }
}
```

</details>

##### Qwen3.8-27B with DSpark

Place the [DSpark draft checkpoint](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark) at `/models/Qwen3.8-27B-speculator.dspark`.

<details>
<summary>Worker settings and layout.json</summary>

```bash
MODEL=/models/Qwen3.8-27B
SERVED_MODEL=qwen38
PROMPT_TOKENS=2001
unset PYTHONHASHSEED
MODEL_ARGS=(--max-model-len 4096 --prefix-match-unit 16 --speculative-config '{"method":"dspark","num_speculative_tokens":8,"model":"/models/Qwen3.8-27B-speculator.dspark"}')
```

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
    ],
    "compatible_decode_urls": [
      "http://127.0.0.1:8001"
    ]
  }
}
```

</details>

#### Qwen4Exp FP8 QSA

Use the Qwen3.8-Flash-Next-FP8 checkpoint with architecture Qwen4ExpForConditionalGeneration.

<details>
<summary>Worker settings and layout.json</summary>

```bash
MODEL=/models/Qwen3.8-Flash-Next-FP8
SERVED_MODEL=qwen4exp
PROMPT_TOKENS=1701
unset PYTHONHASHSEED
MODEL_ARGS=(--max-model-len 4096)
```

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

#### DeepSeek V4 Flash

<details>
<summary>Worker settings and layout.json</summary>

```bash
MODEL=/models/DeepSeek-V4-Flash-0731
SERVED_MODEL=v4-flash
PROMPT_TOKENS=1057
unset PYTHONHASHSEED
MODEL_ARGS=(--max-model-len 4096)
```

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

##### DeepSeek V4 Flash with DSpark

Use the 0731 checkpoint with its DSpark draft. For P/D, select NIXL.

<details>
<summary>Worker settings and layout.json</summary>

```bash
MODEL=/models/DeepSeek-V4-Flash-0731
SERVED_MODEL=v4-flash
PROMPT_TOKENS=2001
unset PYTHONHASHSEED
MODEL_ARGS=(--max-model-len 4096 --speculative-config '{"method":"dspark","num_speculative_tokens":4}')
```

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
    ],
    "compatible_decode_urls": [
      "http://127.0.0.1:8001"
    ]
  }
}
```

</details>

#### DeepSeek V4 Pro

<details>
<summary>Worker settings and layout.json</summary>

```bash
MODEL=/models/DeepSeek-V4-Pro
SERVED_MODEL=v4-pro
PROMPT_TOKENS=2001
unset PYTHONHASHSEED
MODEL_ARGS=(--max-model-len 4096 --tensor-parallel-size 4)
```

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

#### DeepSeek V4.1 Flash

Use the 1017 checkpoint with bounded replay.

<details>
<summary>Worker settings and layout.json</summary>

```bash
MODEL=/models/DeepSeek-V4.1-Flash
SERVED_MODEL=v41-flash
PROMPT_TOKENS=1057
unset PYTHONHASHSEED
MODEL_ARGS=(--max-model-len 4096 --tensor-parallel-size 2 --swa-bounded-replay)
```

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

Use the [vLLM Ascend 0.32.0rc1 A2 image](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.32.0rc1) on Atlas 800I A2. Initialize the Ascend toolkit before launching workers.

| Checkpoint | Modes | NPUs per worker | Deployment |
| --- | --- | ---: | --- |
| [Qwen3-4B](#qwen3-4b-on-ascend) | Off | 1 | HTTP colocated or Mooncake V1 P/D |
| [Qwen3.5-4B](#qwen35-4b-with-mtp-on-ascend) | Off, MTP, [DSpark](#qwen35-4b-with-dspark-on-ascend) | 1 | HTTP colocated or Mooncake V1 P/D |
| [Qwen3.6-27B W8A8](#qwen36-27b-w8a8-on-ascend) | Off, MTP, DSpark | 1 | HTTP colocated or Mooncake V1 P/D |
| [Qwen3.8-27B W8A8](#qwen38-27b-with-mtp-on-ascend) | Off, MTP | 1 | HTTP colocated or Mooncake V1 P/D. Use the MTP profile for gRPC. |
| [Qwen3.8-27B BF16](#qwen38-27b-with-dspark-on-ascend) | DSpark | 2 | HTTP colocated or Mooncake V1 P/D |
| [Qwen4Exp W4A8](#qwen4exp-w4a8-on-ascend) | Off, MTP | 2 | HTTP colocated or Mooncake V2 P/D |
| [DeepSeek V4 Flash 0731 W8A8](#deepseek-v4-flash-0731-w8a8-on-ascend) | Off, DSpark | 8 | HTTP colocated |

#### Qwen3-4B on Ascend

<details>
<summary>Worker settings and layout.json</summary>

```bash
MODEL=/models/Qwen3-4B
SERVED_MODEL=qwen3-a2
PROMPT_TOKENS=391
TP=1
export PYTHONHASHSEED=0
export ASCEND_RT_VISIBLE_DEVICES=0
MODEL_ARGS=(--max-model-len 10240 --max-num-batched-tokens 4096 --max-num-seqs 4
  --gpu-memory-utilization 0.80 --enforce-eager --skip-mm-profiling --enable-prompt-tokens-details
  --block-size 16)
```

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
        "hash_block_size": 128,
        "hash_lookahead_tokens": 0,
        "speculative_tokens": 0,
        "prefix_caching": true,
        "declared_groups": 1,
        "groups": [
          {"group_idx": 0, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 128, "prefix_cacheable": true, "prefix_replay_tokens": 0}
        ]
      }
    ]
  }
}
```

</details>

#### Qwen3.5-4B with MTP on Ascend

<details>
<summary>Off</summary>

```bash
MODEL=/models/Qwen3.5-4B
SERVED_MODEL=qwen35-a2
PROMPT_TOKENS=1927
TP=1
export PYTHONHASHSEED=0
export ASCEND_RT_VISIBLE_DEVICES=0
MODEL_ARGS=(--max-model-len 10240 --max-num-batched-tokens 4096 --max-num-seqs 4
  --gpu-memory-utilization 0.80 --enforce-eager --skip-mm-profiling --enable-prompt-tokens-details)
```

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
        "speculative_tokens": 0,
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
    ],
    "compatible_decode_urls": [
      "http://127.0.0.1:8001"
    ]
  }
}
```

</details>

<details>
<summary>MTP</summary>

```bash
MODEL=/models/Qwen3.5-4B
SERVED_MODEL=qwen35-a2
PROMPT_TOKENS=1927
TP=1
export PYTHONHASHSEED=0
export ASCEND_RT_VISIBLE_DEVICES=0
MODEL_ARGS=(--max-model-len 10240 --max-num-batched-tokens 4096 --max-num-seqs 4
  --gpu-memory-utilization 0.80 --enforce-eager --skip-mm-profiling --enable-prompt-tokens-details
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}')
```

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
    ],
    "compatible_decode_urls": [
      "http://127.0.0.1:8001"
    ]
  }
}
```

</details>

#### Qwen3.5-4B with DSpark on Ascend

Place a compatible Ascend DSpark draft for Qwen3.5-4B at `/models/Qwen3.5-4B-DSpark`.

<details>
<summary>DSpark</summary>

```bash
MODEL=/models/Qwen3.5-4B
SERVED_MODEL=qwen35-a2
PROMPT_TOKENS=1927
TP=1
export PYTHONHASHSEED=0
export ASCEND_RT_VISIBLE_DEVICES=0
MODEL_ARGS=(--max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 4
  --gpu-memory-utilization 0.80 --enforce-eager --skip-mm-profiling --enable-prompt-tokens-details
  --tensor-parallel-size 1 --speculative-config '{"method": "dspark", "num_speculative_tokens": 8, "model": "/models/Qwen3.5-4B-DSpark"}')
```

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
    ],
    "compatible_decode_urls": [
      "http://127.0.0.1:8001"
    ]
  }
}
```

</details>

#### Qwen3.6-27B W8A8 on Ascend

Use [Eco-Tech/Qwen3.6-27B-w8a8](https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-27B-w8a8). Mount its Ascend DSpark draft at `/models/Qwen3.6-27B-DSpark` for DSpark.

<details>
<summary>Off</summary>

```bash
MODEL=/models/Qwen3.6-27B-w8a8
SERVED_MODEL=qwen36-a2
PROMPT_TOKENS=2695
TP=1
export PYTHONHASHSEED=0
export ASCEND_RT_VISIBLE_DEVICES=0
MODEL_ARGS=(--max-model-len 10240 --max-num-batched-tokens 4096 --max-num-seqs 4
  --gpu-memory-utilization 0.80 --enforce-eager --skip-mm-profiling --enable-prompt-tokens-details
  --quantization ascend)
```

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
        "speculative_tokens": 0,
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
    ],
    "compatible_decode_urls": [
      "http://127.0.0.1:8001"
    ]
  }
}
```

</details>

<details>
<summary>MTP</summary>

```bash
MODEL=/models/Qwen3.6-27B-w8a8
SERVED_MODEL=qwen36-a2
PROMPT_TOKENS=2695
TP=1
export PYTHONHASHSEED=0
export ASCEND_RT_VISIBLE_DEVICES=0
MODEL_ARGS=(--max-model-len 10240 --max-num-batched-tokens 4096 --max-num-seqs 4
  --gpu-memory-utilization 0.80 --enforce-eager --skip-mm-profiling --enable-prompt-tokens-details
  --quantization ascend --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}')
```

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
    ],
    "compatible_decode_urls": [
      "http://127.0.0.1:8001"
    ]
  }
}
```

</details>

<details>
<summary>DSpark</summary>

```bash
MODEL=/models/Qwen3.6-27B-w8a8
SERVED_MODEL=qwen36-a2
PROMPT_TOKENS=2695
TP=1
export PYTHONHASHSEED=0
export ASCEND_RT_VISIBLE_DEVICES=0
MODEL_ARGS=(--max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 4
  --gpu-memory-utilization 0.80 --enforce-eager --skip-mm-profiling --enable-prompt-tokens-details
  --quantization ascend --tensor-parallel-size 1 --speculative-config '{"method": "dspark", "num_speculative_tokens": 8, "model": "/models/Qwen3.6-27B-DSpark"}')
```

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
        "declared_groups": 4,
        "groups": [
          {"group_idx": 0, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0},
          {"group_idx": 1, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 896},
          {"group_idx": 2, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 896},
          {"group_idx": 3, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 896, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 896}
        ]
      }
    ],
    "compatible_decode_urls": [
      "http://127.0.0.1:8001"
    ]
  }
}
```

</details>

#### Qwen3.8-27B with MTP on Ascend

<details>
<summary>Off (HTTP)</summary>

```bash
MODEL=/models/Qwen3.8-27B-W8A8
SERVED_MODEL=qwen38-a2
PROMPT_TOKENS=2695
TP=1
export PYTHONHASHSEED=0
export ASCEND_RT_VISIBLE_DEVICES=0
MODEL_ARGS=(--max-model-len 10240 --max-num-batched-tokens 4096 --max-num-seqs 4
  --gpu-memory-utilization 0.80 --enforce-eager --skip-mm-profiling --enable-prompt-tokens-details
  --quantization ascend)
```

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
        "speculative_tokens": 0,
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
    ],
    "compatible_decode_urls": [
      "http://127.0.0.1:8001"
    ]
  }
}
```

</details>

<details>
<summary>MTP worker settings and HTTP layout.json</summary>

```bash
MODEL=/models/Qwen3.8-27B-W8A8
SERVED_MODEL=qwen38-a2
PROMPT_TOKENS=2695
TP=1
export PYTHONHASHSEED=0
export ASCEND_RT_VISIBLE_DEVICES=0
MODEL_ARGS=(--max-model-len 10240 --max-num-batched-tokens 4096 --max-num-seqs 4
  --gpu-memory-utilization 0.80 --enforce-eager --skip-mm-profiling --enable-prompt-tokens-details
  --quantization ascend --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}')
```

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
    ],
    "compatible_decode_urls": [
      "http://127.0.0.1:8001"
    ]
  }
}
```

</details>

<details>
<summary>MTP gRPC colocated layout.json</summary>

```json
{
  "grpc://127.0.0.1:50051": {
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

<details>
<summary>MTP gRPC P/D layout.json</summary>

```json
{
  "grpc://127.0.0.1:50051": {
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
    ],
    "compatible_decode_urls": [
      "grpc://127.0.0.1:50052"
    ]
  }
}
```

</details>

#### Qwen3.8-27B with DSpark on Ascend

Use the BF16 target and place its compatible DSpark draft at `/models/Qwen3.8-27B-speculator.dspark`.

<details>
<summary>DSpark</summary>

```bash
MODEL=/models/Qwen3.8-27B
SERVED_MODEL=qwen38-a2
PROMPT_TOKENS=2695
TP=2
export PYTHONHASHSEED=0
export ASCEND_RT_VISIBLE_DEVICES=0,1
MODEL_ARGS=(--max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 4
  --gpu-memory-utilization 0.80 --enforce-eager --skip-mm-profiling --enable-prompt-tokens-details
  --tensor-parallel-size 2 --speculative-config '{"method": "dspark", "num_speculative_tokens": 8, "model": "/models/Qwen3.8-27B-speculator.dspark"}')
```

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
    ],
    "compatible_decode_urls": [
      "http://127.0.0.1:8001"
    ]
  }
}
```

</details>

#### Qwen4Exp W4A8 on Ascend

Use [Eco-Tech/Qwen3.8-Flash-Next-w4a8](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-Flash-Next-w4a8) with 128 GiB of container shared memory and host offload for Engram.

<details>
<summary>Colocated, off</summary>

```bash
MODEL=/models/Qwen3.8-Flash-Next-w4a8
SERVED_MODEL=qwen4exp-a2
PROMPT_TOKENS=4999
TP=2
export PYTHONHASHSEED=0
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ASCEND_PLE_BACKEND=torch
MODEL_ARGS=(--max-model-len 10240 --max-num-batched-tokens 4096 --max-num-seqs 4
  --gpu-memory-utilization 0.93 --enforce-eager --skip-mm-profiling --enable-prompt-tokens-details
  --quantization ascend --tensor-parallel-size 2 --trust-remote-code --enable-expert-parallel
  --mamba-cache-mode align --engram-config '{"cpu_offload": true, "dp_shared_memory": false}')
```

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
        "speculative_tokens": 0,
        "tail_drop_groups": [],
        "partial_hash_hits": false,
        "prefix_caching": true,
        "declared_groups": 6,
        "groups": [
          {"group_idx": 0, "spec_type": "CircularBufferSpec", "block_size": 4, "tokens_per_state": "1", "prefix_cacheable": false, "prefix_replay_tokens": 0, "kind": "scratch"},
          {"group_idx": 1, "spec_type": "FullAttentionSpec", "block_size": 1664, "tokens_per_state": "1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "kind": "full_attention"},
          {"group_idx": 2, "spec_type": "MambaSpec", "block_size": 1664, "tokens_per_state": "-1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "kind": "mamba"},
          {"group_idx": 3, "spec_type": "MambaSpec", "block_size": 1664, "tokens_per_state": "-1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "kind": "mamba"},
          {"group_idx": 4, "spec_type": "MambaSpec", "block_size": 1664, "tokens_per_state": "-1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "kind": "mamba"},
          {"group_idx": 5, "spec_type": "MambaSpec", "block_size": 1664, "tokens_per_state": "-1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "kind": "mamba"}
        ]
      }
    ]
  }
}
```

</details>

<details>
<summary>Colocated, MTP</summary>

```bash
MODEL=/models/Qwen3.8-Flash-Next-w4a8
SERVED_MODEL=qwen4exp-a2
PROMPT_TOKENS=4999
TP=2
export PYTHONHASHSEED=0
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ASCEND_PLE_BACKEND=torch
MODEL_ARGS=(--max-model-len 10240 --max-num-batched-tokens 4096 --max-num-seqs 4
  --gpu-memory-utilization 0.93 --enforce-eager --skip-mm-profiling --enable-prompt-tokens-details
  --quantization ascend --tensor-parallel-size 2 --trust-remote-code --enable-expert-parallel
  --mamba-cache-mode align --engram-config '{"cpu_offload": true, "dp_shared_memory": false}'
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}')
```

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

<details>
<summary>P/D, off with Mooncake V2</summary>

```bash
MODEL=/models/Qwen3.8-Flash-Next-w4a8
SERVED_MODEL=qwen4exp-a2
PROMPT_TOKENS=2695
TP=2
export PYTHONHASHSEED=0
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ASCEND_PLE_BACKEND=torch
MODEL_ARGS=(--max-model-len 10240 --max-num-batched-tokens 4096 --max-num-seqs 4
  --gpu-memory-utilization 0.93 --enforce-eager --skip-mm-profiling --enable-prompt-tokens-details
  --quantization ascend --tensor-parallel-size 2 --trust-remote-code --enable-expert-parallel
  --mamba-cache-mode align --engram-config '{"cpu_offload": true, "dp_shared_memory": false}'
  --mamba-ssm-cache-dtype bfloat16)
```

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
        "speculative_tokens": 0,
        "tail_drop_groups": [],
        "partial_hash_hits": false,
        "prefix_caching": true,
        "declared_groups": 6,
        "groups": [
          {"group_idx": 0, "spec_type": "CircularBufferSpec", "block_size": 4, "tokens_per_state": "1", "prefix_cacheable": false, "prefix_replay_tokens": 0, "kind": "scratch"},
          {"group_idx": 1, "spec_type": "FullAttentionSpec", "block_size": 896, "tokens_per_state": "1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "kind": "full_attention"},
          {"group_idx": 2, "spec_type": "MambaSpec", "block_size": 896, "tokens_per_state": "-1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "kind": "mamba"},
          {"group_idx": 3, "spec_type": "MambaSpec", "block_size": 896, "tokens_per_state": "-1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "kind": "mamba"},
          {"group_idx": 4, "spec_type": "MambaSpec", "block_size": 896, "tokens_per_state": "-1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "kind": "mamba"},
          {"group_idx": 5, "spec_type": "MambaSpec", "block_size": 896, "tokens_per_state": "-1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "kind": "mamba"}
        ]
      }
    ],
    "compatible_decode_urls": [
      "http://127.0.0.1:8001"
    ]
  }
}
```

</details>

<details>
<summary>P/D, MTP with Mooncake V2</summary>

```bash
MODEL=/models/Qwen3.8-Flash-Next-w4a8
SERVED_MODEL=qwen4exp-a2
PROMPT_TOKENS=2695
TP=2
export PYTHONHASHSEED=0
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ASCEND_PLE_BACKEND=torch
MODEL_ARGS=(--max-model-len 10240 --max-num-batched-tokens 4096 --max-num-seqs 4
  --gpu-memory-utilization 0.93 --enforce-eager --skip-mm-profiling --enable-prompt-tokens-details
  --quantization ascend --tensor-parallel-size 2 --trust-remote-code --enable-expert-parallel
  --mamba-cache-mode align --engram-config '{"cpu_offload": true, "dp_shared_memory": false}'
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}'
  --mamba-ssm-cache-dtype bfloat16)
```

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
        "declared_groups": 7,
        "groups": [
          {"group_idx": 0, "spec_type": "CircularBufferSpec", "block_size": 8, "tokens_per_state": "1", "prefix_cacheable": false, "prefix_replay_tokens": 0, "kind": "scratch"},
          {"group_idx": 1, "spec_type": "FullAttentionSpec", "block_size": 896, "tokens_per_state": "1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "kind": "full_attention"},
          {"group_idx": 2, "spec_type": "MambaSpec", "block_size": 896, "tokens_per_state": "-1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "kind": "mamba"},
          {"group_idx": 3, "spec_type": "MambaSpec", "block_size": 896, "tokens_per_state": "-1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "kind": "mamba"},
          {"group_idx": 4, "spec_type": "MambaSpec", "block_size": 896, "tokens_per_state": "-1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "kind": "mamba"},
          {"group_idx": 5, "spec_type": "MambaSpec", "block_size": 896, "tokens_per_state": "-1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "kind": "mamba"},
          {"group_idx": 6, "spec_type": "MambaSpec", "block_size": 896, "tokens_per_state": "-1", "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "kind": "mamba"}
        ]
      }
    ],
    "compatible_decode_urls": [
      "http://127.0.0.1:8001"
    ]
  }
}
```

</details>

#### DeepSeek V4 Flash 0731 W8A8 on Ascend

Set HCCL_IF_IP, HCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME and TP_SOCKET_IFNAME for the node's communication interface.

<details>
<summary>Off</summary>

```bash
MODEL=/models/DeepSeek-V4-Flash-0731-w8a8
SERVED_MODEL=v4-flash-a2
PROMPT_TOKENS=16401
TP=8
export PYTHONHASHSEED=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_OP_EXPANSION_MODE=AIV OMP_NUM_THREADS=8 VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=16384
MODEL_ARGS=(--max-model-len 65536 --max-num-batched-tokens 8192 --max-num-seqs 4
  --gpu-memory-utilization 0.90 --enforce-eager --skip-mm-profiling --enable-prompt-tokens-details
  --quantization ascend --tensor-parallel-size 8 --enable-expert-parallel --tokenizer-mode deepseek_v4
  --block-size 128 --attention_config.indexer_kv_dtype int8 --no-disable-hybrid-kv-cache-manager
  --prefix-cache-retention-interval 16384 --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}')
```

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
        "speculative_tokens": 0,
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

<details>
<summary>DSpark</summary>

```bash
MODEL=/models/DeepSeek-V4-Flash-0731-w8a8
SERVED_MODEL=v4-flash-a2
PROMPT_TOKENS=16401
TP=8
export PYTHONHASHSEED=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_OP_EXPANSION_MODE=AIV OMP_NUM_THREADS=8 VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=16384
MODEL_ARGS=(--max-model-len 65536 --max-num-batched-tokens 8192 --max-num-seqs 4
  --gpu-memory-utilization 0.90 --enforce-eager --skip-mm-profiling --enable-prompt-tokens-details
  --quantization ascend --tensor-parallel-size 8 --enable-expert-parallel --tokenizer-mode deepseek_v4
  --block-size 128 --attention_config.indexer_kv_dtype int8 --no-disable-hybrid-kv-cache-manager
  --prefix-cache-retention-interval 16384 --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}'
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 7, "enforce_eager": true}')
```

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

## CLI options

These defaults are for `vllm-router`. See the [CLI reference](../cli_reference.md) for other Router options.

| Option | Type | Default | Use |
| --- | --- | --- | --- |
| `--policy` | Policy name | `cache_aware` | Set `kv_aware` for colocated routing. Provides the default policy for P/D stages. |
| `--worker-urls` | URL list | Empty | Worker endpoints for colocated routing. |
| `--vllm-pd-disaggregation` | Flag | `false` | Use separate Prefill and Decode pools. |
| `--prefill` | URL plus optional port, repeatable | Empty | Add a Prefill URL and, for Mooncake bootstrap, its port. |
| `--decode` | URL, repeatable | Empty | Add a Decode URL. |
| `--prefill-policy` | Policy name | Inherit `--policy` | Set `kv_aware` to score Prefill reuse. |
| `--decode-policy` | Policy name | Inherit `--policy` | Select the Decode policy, such as `round_robin` or `kv_aware`. |
| `--kv-connector` | `nixl`, `mooncake`, `moriio` | `nixl` | Select the P/D handoff protocol. |
| `--kv-hybrid-layout` | File path | Unset | Read the layout and P/D compatibility pairs keyed by worker URL. |
| `--kv-event-source` | `URL=ENDPOINT[,ENDPOINT...]`, repeatable | Empty | Static live event addresses. |
| `--kv-event-replay-source` | `URL=ENDPOINT[,ENDPOINT...]`, repeatable | Empty | Static replay addresses. |
| `--kv-event-block-size` | `u32` | `16` | Hash unit in tokens for static event sources. |
| `--kv-event-model` | String | Unset | Served model ID for static event sources. |
| `--pd-kv-events` | `URL=ENDPOINT[,ENDPOINT...]`, repeatable | Empty | Static P/D live endpoints in DP-rank order for 16-token blocks. |
| `--pd-hybrid-config` | File path | Unset | Read a legacy P/D layout. New deployments use `--kv-hybrid-layout`. |
| `--kv-event-strict-recovery` | Flag | `false` | Require complete recovery or a worker KV clear before scoring reuse. |
| `--router-temperature` | `f64` | `0.0` | KV-aware selection temperature for colocated and Prefill placement. Zero selects the lowest cost. |
| `--kv-penalty-host` | `f64` | `0.8` | Reuse credit for host-resident KV. |
| `--kv-penalty-external` | `f64` | `0.6` | Reuse credit for KV in an external tier. |
| `--kv-penalty-disk` | `f64` | `0.3` | Reuse credit for disk-resident KV. |
| `--kv-aware-session-affinity-ttl-secs` | `u64`, seconds | Unset | Enable idle session affinity for 1–31,536,000 seconds. In P/D, applies to Decode. |
| `--prometheus-port` | `u16` | `29000` | Port for Router metrics. |
| `VLLM_ROUTER_TOKENIZER` | Path, environment variable | Unset | Local model directory for gRPC text and chat rendering. |

### Static event endpoints

For stock HTTP vLLM 0.30.0, use fixed live and replay ports. Save this [Qwen3-4B](#qwen3-4b-on-gpu) example as `layout.json` and replace `replace-with-KV_TOPIC` with the topic from the worker terminal.

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

```bash
vllm-router --port 19000 --policy kv_aware --worker-urls http://127.0.0.1:8000 \
  --kv-event-source http://127.0.0.1:8000=tcp://127.0.0.1:5557 \
  --kv-event-replay-source http://127.0.0.1:8000=tcp://127.0.0.1:5558 \
  --kv-event-model qwen3 --kv-event-block-size 16 --kv-hybrid-layout layout.json
```

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Worker reports reuse and Router stays at zero | Check the served model, hash settings, publisher topic and reachability of both event ports. Inspect `/kv_event_sources` or `Control.GetKvEventSources`. |
| Router rejects the layout | Use `--log-level debug` and inspect `worker-reported KV layout rejected`. Match the JSON to the actual model, rank, groups and speculation settings. |
| Worker reports `Address already in use` | Check HTTP, gRPC, event and connector ports. Reserve separate ranges for DP ranks. |
| Worker restarted | Generate a new publisher topic. Update a static layout that pins the topic. A Router restart uses the available replay history. |
| Decode has no remote read | Check whether the prompt was already cached on Decode, the connector and role settings, and Decode's transfer log. |
