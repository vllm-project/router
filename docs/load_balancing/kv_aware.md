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
| Hybrid attention, sliding-window attention, MTP, DSpark or a custom hash seed | Use the worker's complete layout metadata when available. Otherwise save the matching [model layout](#layout-template) as `layout.json` and pass `--kv-hybrid-layout layout.json`. |
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

Set the event ports and put the selected model's worker arguments in the Bash array `MODEL_ARGS`. Start with [Qwen3-4B on GPU](#gpu) for full attention or choose a [hybrid recipe](#layout-template). Use the [A2 Qwen3.8 W8A8 MTP recipe](#ascend-a2) for the Ascend gRPC command.

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

For Ascend A2, use [Qwen3.8 W8A8 MTP](#ascend-a2). Set `MOONCAKE_CONNECTOR=MooncakeConnectorV1` on both workers before building `TRANSFER`. Run the worker command on Prefill first and then Decode with the stage settings above.

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve "$MODEL" --host 127.0.0.1 \
  --port "$HTTP_PORT" --grpc-port "$GRPC_PORT" --served-model-name "$SERVED_MODEL" \
  --enable-scale-out "${KV_ARGS[@]}" "${MODEL_ARGS[@]}" --kv-transfer-config "$TRANSFER"
```

Set `MODEL` in the Router terminal to the local tokenizer directory. Save the model's [gRPC P/D layout](#ascend-a2) as `layout.json`.

```bash
VLLM_ROUTER_TOKENIZER="$MODEL" vllm-router --port 19000 --vllm-pd-disaggregation \
  --prefill grpc://127.0.0.1:50051 --decode grpc://127.0.0.1:50052 \
  --prefill-policy kv_aware --decode-policy round_robin --kv-connector mooncake \
  --kv-hybrid-layout layout.json
```

To score Decode reuse as well, set `--decode-policy kv_aware`, enable Decode's events and include its layout when required. For HTTP Decode scoring, supply explicit prompt token IDs. Each gRPC stage uses the token IDs produced by Router. Use one worker interface throughout the P/D pool.

### P/D layout

Save the entries in `layout.json` and pass it to Router with `--kv-hybrid-layout`. Use the [layout template](#layout-template) for workers that need an explicit layout.

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

Use the checkpoint and flags from the appropriate platform table. Set `MODEL` to its local path and `SERVED_MODEL` to the name you want clients to use. Use the probe length in [Check cache reuse](#check-cache-reuse).

### GPU

These recipes use vLLM 0.33.0. Leave `PYTHONHASHSEED` unset.

| Checkpoint | Worker arguments in addition to the [common options](#worker-requirements) | Probe tokens | Layout |
| --- | --- | ---: | --- |
| Qwen3-4B-Instruct-2507 | `--block-size 16` | 129 | Automatic |
| Qwen3.5-4B | `--max-model-len 4096 --prefix-match-unit 16` | 2001 | GPU Qwen, group size 576 |
| Qwen3.6-27B | `--max-model-len 4096 --prefix-match-unit 16` | 2001 | GPU Qwen, group size 832 |
| Qwen3.8-27B | `--max-model-len 4096 --prefix-match-unit 16` | 2001 | GPU Qwen, group size 832. For DSpark use GPU Qwen3.8 DSpark. |
| Qwen3.8-Flash-Next-FP8 (Qwen4Exp QSA) | `--max-model-len 4096` | 1701 | GPU Qwen4Exp |
| DeepSeek-V4-Flash-0731 | `--max-model-len 4096` | 1057, or 2001 with DSpark | GPU V4 Flash |
| DeepSeek-V4-Pro | `--max-model-len 4096 --tensor-parallel-size 4` | 2001 | GPU V4 Pro |
| DeepSeek-V4.1-Flash (1017) | `--max-model-len 4096 --tensor-parallel-size 2 --swa-bounded-replay` | 1057 | GPU V4.1 Flash |

### Ascend A2

Use the [vLLM Ascend A2 image](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.32.0rc1) with the paired versions above. Initialize the toolkit, export `PYTHONHASHSEED=0`, and set `ASCEND_RT_VISIBLE_DEVICES` for each worker. The Qwen recipes use `--max-num-batched-tokens 4096 --max-num-seqs 4 --enforce-eager --skip-mm-profiling --enable-prompt-tokens-details`.

| Checkpoint | Deployment | TP | Max model length | GPU memory utilization flag | Probe tokens | Layout |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-4B | HTTP colocated or Mooncake V1 P/D | 1 | 10240 | 0.80 | 391 | A2 Qwen3 |
| Qwen3.5-4B | HTTP colocated or Mooncake V1 P/D | 1 | 10240, or 4096 with DSpark | 0.80 | 1927 | A2 Qwen, group size 640. For DSpark use A2 Qwen3.5 DSpark. |
| [Qwen3.6-27B W8A8](https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-27B-w8a8) | HTTP colocated or Mooncake V1 P/D | 1 | 10240 | 0.80 | 2695 | A2 Qwen, group size 896 |
| Qwen3.8-27B W8A8 | HTTP colocated or Mooncake V1 P/D. Use the MTP profile for gRPC. | 1 | 10240 | 0.80 | 2695 | A2 Qwen, group size 896 |
| Qwen3.8-27B BF16 | HTTP colocated or Mooncake V1 P/D with DSpark | 2 | 4096 | 0.80 | 2695 | A2 Qwen3.8 DSpark |
| [Qwen3.8-Flash-Next W4A8](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-Flash-Next-w4a8) | HTTP colocated or Mooncake V2 P/D | 2 | 10240 | 0.93 | 4999 colocated, 2695 P/D | A2 Qwen4Exp |
| DeepSeek-V4-Flash-0731 W8A8 | HTTP colocated | 8 | 65536 | 0.90 | 16401 | A2 V4 Flash |

Set `--tensor-parallel-size`, `--max-model-len` and `--gpu-memory-utilization` from the table.

| Checkpoint | Extra settings |
| --- | --- |
| Qwen3-4B | `--block-size 16`. Use the effective 128-token hash blocks in the layout. |
| W8A8 or W4A8 checkpoints | `--quantization ascend` |
| Qwen4Exp | Export `VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ASCEND_PLE_BACKEND=torch`. Use 128 GiB of container shared memory. Add `--trust-remote-code --enable-expert-parallel --mamba-cache-mode align --engram-config '{"cpu_offload":true,"dp_shared_memory":false}'`. For P/D add `--mamba-ssm-cache-dtype bfloat16`. |
| V4 Flash | Export `HCCL_OP_EXPANSION_MODE=AIV OMP_NUM_THREADS=8 VLLM_ASCEND_ENABLE_FLASHCOMM1=1 VLLM_PREFIX_CACHE_RETENTION_INTERVAL=16384`. Set `HCCL_IF_IP`, `HCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` and `TP_SOCKET_IFNAME` for the node. Add `--max-num-batched-tokens 8192 --max-num-seqs 4 --enforce-eager --skip-mm-profiling --enable-prompt-tokens-details --enable-expert-parallel --tokenizer-mode deepseek_v4 --block-size 128 --attention_config.indexer_kv_dtype int8 --no-disable-hybrid-kv-cache-manager --prefix-cache-retention-interval 16384 --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":16}'`. |

### MTP and DSpark

Pass the JSON below to the worker's `--speculative-config`. Use one mode per worker. For DSpark with a separate draft, point `model` to the matching local checkpoint. In the Router layout set `speculative_tokens` to `num_speculative_tokens` and use the hash lookahead and tail-drop values below. Omit `--speculative-config` and use zero for both token counts when speculation is off.

| Worker | Deployment or mode | Speculative config | Hash lookahead tokens | Tail-drop groups |
| --- | --- | --- | ---: | --- |
| GPU Qwen3.8 | Colocated, MTP | `{"method":"mtp","num_speculative_tokens":3}` | 1 | Omit |
| GPU Qwen3.8 | Colocated or Mooncake P/D, MTP | `{"method":"mtp","num_speculative_tokens":3,"disable_lookahead_block_hashes":true}` | 0 | `[3]` |
| GPU Qwen3.8 or A2 Qwen3.8 BF16 | Colocated or Mooncake P/D, DSpark | `{"method":"dspark","num_speculative_tokens":8,"model":"/models/Qwen3.8-27B-speculator.dspark"}` | 0 | `[]` |
| GPU V4 Flash | Colocated or NIXL P/D, DSpark | `{"method":"dspark","num_speculative_tokens":4}` | 0 | `[]` |
| A2 Qwen3.5, Qwen3.6 or Qwen3.8 W8A8 | MTP | `{"method":"mtp","num_speculative_tokens":3}` | 0 | `[]` |
| A2 Qwen3.5 | DSpark | `{"method":"dspark","num_speculative_tokens":8,"model":"/models/Qwen3.5-4B-DSpark"}` | 0 | `[]` |
| A2 Qwen3.6 | DSpark | `{"method":"dspark","num_speculative_tokens":8,"model":"/models/Qwen3.6-27B-DSpark"}` | 0 | `[]` |
| A2 Qwen4Exp | MTP | `{"method":"qwen3_5_mtp","num_speculative_tokens":3}` | 0 | `[]` |
| A2 V4 Flash | DSpark | `{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true}` | 0 | `[]` |

### Layout template

This example is GPU Qwen3.5-4B with event discovery. Replace the URL with your HTTP or gRPC worker address. Set `declared_groups` to the number of entries in `groups`, including scratch groups. For P/D, apply the [pairing rules](#pd-layout) to the Prefill entry and add a Decode entry if it needs an explicit layout. For fixed publishers, set the actual `model` and `publisher_topic` on each rank.

```json
{
  "http://127.0.0.1:8000": {
    "schema_version": 1,
    "ranks": [{
      "rank": 0,
      "weight_version": "default",
      "instance_topic": true,
      "hash_algorithm": "sha256_cbor",
      "hash_seed": "vllm-none-hash",
      "full_hashes": true,
      "prefix_caching": true,
      "hash_block_size": 16,
      "partial_hash_hits": true,
      "speculative_tokens": 0,
      "hash_lookahead_tokens": 0,
      "declared_groups": 4,
      "groups": [
        {"group_idx": 0, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 576, "mamba_cache_mode": "align", "prefix_cacheable": true, "prefix_replay_tokens": 0},
        {"group_idx": 1, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 576, "mamba_cache_mode": "align", "prefix_cacheable": true, "prefix_replay_tokens": 0},
        {"group_idx": 2, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 576, "mamba_cache_mode": "align", "prefix_cacheable": true, "prefix_replay_tokens": 0},
        {"group_idx": 3, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 576, "prefix_cacheable": true, "prefix_replay_tokens": 0}
      ]
    }]
  }
}
```

For Ascend layouts, set `"hash_seed":"0"`. Set `partial_hash_hits` to `false` unless the table specifies `true`. For each group, assign its `group_idx` and `block_size` from the layout row and copy the matching fields below. Groups use `prefix_cacheable: true` and `prefix_replay_tokens: 0` unless specified.

| Group | `kind` | `spec_type` | Extra fields |
| --- | --- | --- | --- |
| Full | `full_attention` | `FullAttentionSpec` |  |
| Mamba | `mamba` | `MambaSpec` | `"mamba_cache_mode":"align"` |
| SWA | `sliding_window` | `SlidingWindowSpec` | `sliding_window` from the row |
| MLA SWA | `sliding_window_mla` | `SlidingWindowMLASpec` | `sliding_window` from the row |
| Sparse MLA | `mla_attention` | `MLAAttentionSpec` | `"cache_role":"sparse"` and `tokens_per_state` as a string |
| Scratch | `scratch` | `CircularBufferSpec` | `"prefix_cacheable":false,"tokens_per_state":"1"` |

| Layout | `hash_block_size` | `partial_hash_hits` | Groups in index order, with block size and extra values |
| --- | ---: | --- | --- |
| GPU Qwen | 16 | true | 0–2 Mamba, 3 Full, same size on all groups |
| GPU Qwen3.8 DSpark | 16 | false | 0–9 Mamba, 10–13 Full, 14 SWA with window 2048. All size 832. |
| GPU Qwen4Exp | 1600 | false | 0–3 Mamba 1600, 4 Scratch 4, 5 Sparse MLA 1600 with `"4"` |
| GPU V4 Flash | 4 | false | 0–1 MLA SWA 64 with window 128, 2 Sparse MLA 256 with `"4"`, 3 MLA SWA 4 with window 8, 4 MLA SWA 8 with window 128 |
| GPU V4 Pro | 4 | false | 0–1 MLA SWA 64 with window 128, 2 Sparse MLA 256 with `"128"`, 3 MLA SWA 8 with window 128, 4 MLA SWA 4 with window 8 |
| GPU V4.1 Flash | 128 | false | 0–4 MLA SWA 32 with window 128, `"prefix_cacheable":false,"prefix_replay_tokens":128`. 5 Sparse MLA 128 with `"2"`, 6 Scratch 8. |
| A2 Qwen3 | 128 | false | 0 Full 128 |
| A2 Qwen | Group size from model row | false | 0 Full, 1–3 Mamba, same size on all groups |
| A2 Qwen3.5 DSpark | 640 | false | 0–1 Full, 2 SWA with window 2048, 3–7 Mamba. All size 640. |
| A2 Qwen3.8 DSpark | 896 | false | 0–3 Full, 4 SWA with window 2048, 5–14 Mamba. All size 896. |
| A2 Qwen4Exp | 1664 colocated, 896 P/D | false | 0 Scratch 4 with speculation off, 8 with MTP. 1 Full and 2–5 Mamba (2–6 with MTP), each the hash block size. Set Full `tokens_per_state` to `"1"` and Mamba to `"-1"`. |
| A2 V4 Flash | 8 | false | 0 Sparse MLA 512 with `"4"`, 1 Sparse MLA 16384 with `"128"`, 2–3 MLA SWA 128 with window 128, 4 MLA SWA 8 with window 8, 5 MLA SWA 32 with window 128. Set `tokens_per_state` to `"1"` on the MLA SWA groups. |


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
| `--kv-event-strict-recovery` | Flag | `false` | Require complete recovery or a worker KV clear before scoring reuse. |
| `--router-temperature` | `f64` | `0.0` | KV-aware selection temperature for colocated and Prefill placement. Zero selects the lowest cost. |
| `--kv-penalty-host` | `f64` | `0.8` | Reuse credit for host-resident KV. |
| `--kv-penalty-external` | `f64` | `0.6` | Reuse credit for KV in an external tier. |
| `--kv-penalty-disk` | `f64` | `0.3` | Reuse credit for disk-resident KV. |
| `--kv-aware-session-affinity-ttl-secs` | `u64`, seconds | Unset | Enable idle session affinity for 1–31,536,000 seconds. In P/D, applies to Decode. |
| `--prometheus-port` | `u16` | `29000` | Port for Router metrics. |
| `VLLM_ROUTER_TOKENIZER` | Path, environment variable | Unset | Local model directory for gRPC text and chat rendering. |

### Static event endpoints

For stock HTTP vLLM 0.30.0, use fixed live and replay ports. Save this [Qwen3-4B](#gpu) example as `layout.json` and replace `replace-with-KV_TOPIC` with the topic from the worker terminal.

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
