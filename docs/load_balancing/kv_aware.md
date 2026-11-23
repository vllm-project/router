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

## Choose a layout configuration

| Deployment | What to supply |
| --- | --- |
| Ordinary single-group full attention with event discovery, the default hash seed, and no sliding window or speculation | Worker URLs. Router derives the single-group layout and checks the hash contract against worker events. |
| Colocated worker that reports its complete layout | Worker URLs. Router discovers the layout and event addresses. |
| Hybrid P/D with worker-reported layouts | Worker URLs and a [pairing-only file](#pd-layout). |
| Worker with event discovery that needs a layout declaration | A [layout file](#layout-template) for the missing group and hash information. Router discovers the served model, publisher topic and event addresses. Add Prefill-to-Decode pairs for hybrid P/D. |
| Worker without event discovery, such as stock HTTP vLLM 0.30.0 | [Static event addresses and a layout](#static-event-endpoints). |

Supply an explicit layout for hybrid attention, sliding windows, speculative decoding or a custom hash seed when the worker cannot report its actual layout.

## Configure the workers

### Worker requirements

| Setting | What to configure and why |
| --- | --- |
| Prefix caching | Keep it enabled. Add `--enable-prefix-caching` if needed to turn it on. |
| Hash algorithm | Pass `--prefix-caching-hash-algo sha256_cbor` so Router and vLLM compute the same block keys. |
| Event hashes | Set `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0` to publish the full 32-byte hashes. |
| KV events | Enable events through `--kv-events-config` and keep incremental reporting. Router uses live events to track cache changes and replay to recover missed events. |
| HTTP text prompts | Add `--enable-scale-out` so Router can use the worker's rendering endpoints. |

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

### Colocated prefill and decode

Set `MODEL` to the checkpoint path and put the [model-specific worker options](#model-recipes) in `MODEL_ARGS`. Use `MODEL_ARGS=()` if the recipe has no extra options.

In the Router terminal, use `LAYOUT_ARGS=()` when Router can derive or discover the layout. For an explicit layout, use `LAYOUT_ARGS=(--kv-hybrid-layout layout.json)`.

#### HTTP workers

```bash
vllm serve "$MODEL" --host 127.0.0.1 \
  --enable-scale-out "${KV_ARGS[@]}" "${MODEL_ARGS[@]}"
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
  -- "${KV_ARGS[@]}" "${MODEL_ARGS[@]}"
```

```bash
# Ascend A2
VLLM_USE_RUST_FRONTEND=1 vllm serve "$MODEL" --host 127.0.0.1 \
  --grpc-port 50051 "${KV_ARGS[@]}" "${MODEL_ARGS[@]}"
```

For gRPC text and chat, Router loads the tokenizer from the request's model path when available locally. Set `VLLM_ROUTER_TOKENIZER` in the Router terminal if that path is unavailable or the request uses a served alias.

```bash
vllm-router --policy kv_aware \
  --worker-urls grpc://127.0.0.1:50051 "${LAYOUT_ARGS[@]}"
```

### P/D routing

These P/D commands use GPU workers and `NixlConnector`. Set `MODEL` and `MODEL_ARGS` in each worker terminal as in the colocated setup. Run the same model checkpoint, engine version and connector on both stages. Assign disjoint GPUs with `CUDA_VISIBLE_DEVICES`. For separate hosts, set `VLLM_NIXL_SIDE_CHANNEL_HOST` to a reachable address on each worker.

#### Set worker ports and transfer parameters

| Setting | Prefill | Decode |
| --- | --- | --- |
| Worker HTTP port | `8000` | `8001` |
| Worker gRPC port, if used | `50051` | `50052` |
| Live and replay event ports | `5557` / `5558` | `5559` / `5560` if Decode uses KV-aware |
| Connector role | `kv_producer` | `kv_consumer` |
| NIXL side-channel port | `5600` | `5601` |

Set each stage's ports and role in its terminal. Run the [event setup](#set-the-event-ports) on Prefill. On Decode, run it only when using `--decode-policy kv_aware`. For Qwen3.5, export `VLLM_SSM_CONV_STATE_LAYOUT=DS` on both stages.

```bash
# Prefill terminal
HTTP_PORT=8000 GRPC_PORT=50051
KV_ROLE=kv_producer
```

```bash
# Decode terminal
HTTP_PORT=8001 GRPC_PORT=50052
KV_ROLE=kv_consumer
export VLLM_NIXL_SIDE_CHANNEL_PORT=5601
KV_ARGS=(--prefix-caching-hash-algo sha256_cbor)
```

To enable Decode events, set `LIVE_PORT=5559 REPLAY_PORT=5560` before running the event setup in the Decode terminal.

```bash
# Both terminals
TRANSFER=$(printf '{"kv_connector":"NixlConnector","kv_role":"%s"}' "$KV_ROLE")
```

In the Router terminal, use `LAYOUT_ARGS=(--kv-hybrid-layout layout.json)` when a [P/D layout](#pd-layout) is required and `LAYOUT_ARGS=()` otherwise.

#### HTTP P/D

Run the worker command on Prefill first, then Decode.

```bash
vllm serve "$MODEL" --host 127.0.0.1 --port "$HTTP_PORT" \
  --enable-scale-out "${KV_ARGS[@]}" "${MODEL_ARGS[@]}" --kv-transfer-config "$TRANSFER"
```

```bash
vllm-router --vllm-pd-disaggregation \
  --prefill http://127.0.0.1:8000 --decode http://127.0.0.1:8001 \
  --prefill-policy kv_aware --decode-policy round_robin \
  "${LAYOUT_ARGS[@]}"
```

#### gRPC P/D

Run the GPU worker command on each stage with its own ports and `TRANSFER`. Use the [same tokenizer setup as colocated gRPC](#grpc-workers). Use gRPC worker URLs as keys in a layout file.

```bash
vllm-rs serve "$MODEL" --host 127.0.0.1 --port "$HTTP_PORT" --grpc-port "$GRPC_PORT" \
  -- "${KV_ARGS[@]}" "${MODEL_ARGS[@]}" --kv-transfer-config "$TRANSFER"
```

```bash
vllm-router --vllm-pd-disaggregation \
  --prefill grpc://127.0.0.1:50051 --decode grpc://127.0.0.1:50052 \
  --prefill-policy kv_aware --decode-policy round_robin \
  "${LAYOUT_ARGS[@]}"
```

These commands use Router's default NIXL connector and select Decode by round robin. To score Decode reuse as well, set `--decode-policy kv_aware`, enable Decode's events and include its layout when required. For HTTP Decode scoring, supply explicit prompt token IDs. Each gRPC stage uses the token IDs produced by Router. Use one worker interface throughout the P/D pool.

#### P/D layout

Save the entries in `layout.json` and pass it to Router with `--kv-hybrid-layout`. Use the [layout template](#layout-template) for workers that need an explicit layout.

| Entry | Required contents |
| --- | --- |
| Hybrid Prefill | `compatible_decode_urls` restricts the pair selection to Decode workers that can receive its KV. Add the Prefill layout if the worker cannot report it. |
| Decode scored by `kv_aware` | The Decode layout when the worker does not report it |
| Decode using `round_robin` | List its URL under the Prefill's `compatible_decode_urls` |

Use the same worker URL as the Router command, without a DP-rank suffix. Leave Program scheduling disabled for hybrid P/D layouts.

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

## Model recipes

Use the worker options for your checkpoint and platform. Choose context length and batching limits for your workload and device memory. Use the layout reference if your worker needs a manual declaration.

### GPU

These recipes use vLLM 0.33.0. Leave `PYTHONHASHSEED` unset.

| Checkpoint | Worker arguments beyond the [common options](#worker-requirements) | Probe tokens | Layout reference |
| --- | --- | ---: | --- |
| Qwen3-4B-Instruct-2507 | None | 129 | Automatic |
| Qwen3.5-4B | None | 2001 | GPU Qwen, group size 576 |
| Qwen3.6-27B | None | 2001 | GPU Qwen, group size 832 |
| Qwen3.8-27B | None | 2001 | GPU Qwen, group size 832. For DSpark use GPU Qwen3.8 DSpark. |
| Qwen3.8-Flash-Next-FP8 (Qwen4Exp QSA) | None | 1701 | GPU Qwen4Exp |
| DeepSeek-V4-Flash-0731 | None | 1057, or 2001 with DSpark | GPU V4 Flash |
| DeepSeek-V4-Pro | `--tensor-parallel-size 4` | 2001 | GPU V4 Pro |
| DeepSeek-V4.1-Flash (1017) | `--tensor-parallel-size 2` | 1057 | GPU V4.1 Flash with bounded replay |

### Ascend A2

Use the [vLLM Ascend A2 image](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.32.0rc1) with the paired versions above. Initialize the toolkit and set `ASCEND_RT_VISIBLE_DEVICES` for each worker. Use `--enforce-eager` for the Qwen A2 recipes.

| Checkpoint | Deployment | TP | GPU memory utilization flag | Probe tokens | Layout |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen3-4B | HTTP colocated | 1 | 0.80 | 391 | A2 Qwen3 |
| Qwen3.5-4B | HTTP colocated | 1 | 0.80 | 1927 | A2 Qwen, group size 640. For DSpark use A2 Qwen3.5 DSpark. |
| [Qwen3.6-27B W8A8](https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-27B-w8a8) | HTTP colocated | 1 | 0.80 | 2695 | A2 Qwen, group size 896 |
| Qwen3.8-27B W8A8 | HTTP or gRPC colocated. Use the MTP profile for gRPC. | 1 | 0.80 | 2695 | A2 Qwen, group size 896 |
| Qwen3.8-27B BF16 | HTTP colocated | 2 | 0.80 | 2695 | A2 Qwen3.8 DSpark |
| [Qwen3.8-Flash-Next W4A8](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-Flash-Next-w4a8) | HTTP colocated | 2 | 0.93 | 4999 | A2 Qwen4Exp |
| DeepSeek-V4-Flash-0731 W8A8 | HTTP colocated | 8 | 0.90 | 16401 | A2 V4 Flash |

Pass the TP and memory values to `--tensor-parallel-size` and `--gpu-memory-utilization`.

| Checkpoint | Extra settings |
| --- | --- |
| Qwen3-4B | `--block-size 16`. Use the effective 128-token hash blocks in the layout. |
| W8A8 or W4A8 checkpoints | `--quantization ascend` |
| Qwen4Exp | Export `VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ASCEND_PLE_BACKEND=torch`. Use 128 GiB of container shared memory. Add `--trust-remote-code --enable-expert-parallel --mamba-cache-mode align --engram-config '{"cpu_offload":true,"dp_shared_memory":false}'`. |
| V4 Flash | Export `HCCL_OP_EXPANSION_MODE=AIV OMP_NUM_THREADS=8 VLLM_ASCEND_ENABLE_FLASHCOMM1=1 VLLM_PREFIX_CACHE_RETENTION_INTERVAL=16384`. Set `HCCL_IF_IP`, `HCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` and `TP_SOCKET_IFNAME` for the node. Add `--enforce-eager --enable-expert-parallel --tokenizer-mode deepseek_v4 --block-size 128 --attention_config.indexer_kv_dtype int8 --no-disable-hybrid-kv-cache-manager --prefix-cache-retention-interval 16384 --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":16}'`. |

### MTP and DSpark

Pass the JSON below to the worker's `--speculative-config`. Use one mode per worker. For DSpark with a separate draft, point `model` to the matching local checkpoint. In the Router layout set `speculative_tokens` to `num_speculative_tokens` and use the hash lookahead and tail-drop values below. Omit `--speculative-config` and use zero for both token counts when speculation is off.

| Worker | Deployment or mode | Speculative config | Hash lookahead tokens | Tail-drop groups |
| --- | --- | --- | ---: | --- |
| GPU Qwen3.8 | Colocated, MTP | `{"method":"mtp","num_speculative_tokens":3}` | 1 | Omit |
| GPU Qwen3.8 | Zero hash lookahead, MTP | `{"method":"mtp","num_speculative_tokens":3,"disable_lookahead_block_hashes":true}` | 0 | `[3]` |
| GPU Qwen3.8 or A2 Qwen3.8 BF16 | DSpark | `{"method":"dspark","num_speculative_tokens":8,"model":"/models/Qwen3.8-27B-speculator.dspark"}` | 0 | `[]` |
| GPU V4 Flash | Colocated or NIXL P/D, DSpark | `{"method":"dspark","num_speculative_tokens":4}` | 0 | `[]` |
| A2 Qwen3.5, Qwen3.6 or Qwen3.8 W8A8 | MTP | `{"method":"mtp","num_speculative_tokens":3}` | 0 | `[]` |
| A2 Qwen3.5 | DSpark | `{"method":"dspark","num_speculative_tokens":8,"model":"/models/Qwen3.5-4B-DSpark"}` | 0 | `[]` |
| A2 Qwen3.6 | DSpark | `{"method":"dspark","num_speculative_tokens":8,"model":"/models/Qwen3.6-27B-DSpark"}` | 0 | `[]` |
| A2 Qwen4Exp | MTP | `{"method":"qwen3_5_mtp","num_speculative_tokens":3}` | 0 | `[]` |
| A2 V4 Flash | DSpark | `{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true}` | 0 | `[]` |

## Supply a layout if required

### Layout template

Save the layout as `layout.json` and pass `--kv-hybrid-layout layout.json`. Use your worker's HTTP or gRPC URL as the key. For hybrid P/D, add the [pairing fields](#pd-layout).

Use the worker's effective layout. On vLLM 0.33.0, use `kv cache group sizes` from the startup log for `groups[].block_size`. For HTTP workers with event discovery, omit `hash_block_size` when `vllm:cache_config_info` reports it. For other manual layouts, supply the worker's effective hash block size. Set `partial_hash_hits` only if the worker uses sub-block prefix hits.

This example is for a GPU Qwen3.5-4B HTTP worker whose metrics report the hash size. It has one DP rank, 576-token group blocks and speculation disabled. Use your worker's group sizes and add `partial_hash_hits` only if it uses sub-block hits.

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

List groups in engine order. Use the hash sizes below as references for manual declarations.

| Group | `kind` | Extra fields |
| --- | --- | --- |
| Full | `full_attention` |  |
| Mamba | `mamba` | `"mamba_cache_mode":"align"` |
| SWA | `sliding_window` | `sliding_window` from the row |
| MLA SWA | `sliding_window_mla` | `sliding_window` from the row |
| Sparse MLA | `mla_attention` | `"cache_role":"sparse"` and `tokens_per_state` as a string |
| Scratch | `scratch` | `"tokens_per_state":"1"` |

| Layout | Hash size (tokens) | Groups in index order, with block size and extra values |
| --- | ---: | --- |
| GPU Qwen | Worker-reported size | 0–2 Mamba, 3 Full, same size on all groups |
| GPU Qwen3.8 DSpark | Worker-reported size | 0–9 Mamba, 10–13 Full, 14 SWA with window 2048. All size 832. |
| GPU Qwen4Exp | 1600 | 0–3 Mamba 1600, 4 Scratch 4, 5 Sparse MLA 1600 with `"4"` |
| GPU V4 Flash | 4 | 0–1 MLA SWA 64 with window 128, 2 Sparse MLA 256 with `"4"`, 3 MLA SWA 4 with window 8, 4 MLA SWA 8 with window 128 |
| GPU V4 Pro | 4 | 0–1 MLA SWA 64 with window 128, 2 Sparse MLA 256 with `"128"`, 3 MLA SWA 8 with window 128, 4 MLA SWA 4 with window 8 |
| GPU V4.1 Flash | 128 | 0–4 MLA SWA 32 with window 128, `"prefix_replay_tokens":128`. 5 Sparse MLA 128 with `"2"`, 6 Scratch 8. |
| A2 Qwen3 | 128 | 0 Full 128 |
| A2 Qwen | Group size from model row | 0 Full, 1–3 Mamba, same size on all groups |
| A2 Qwen3.5 DSpark | 640 | 0–1 Full, 2 SWA with window 2048, 3–7 Mamba. All size 640. |
| A2 Qwen3.8 DSpark | 896 | 0–3 Full, 4 SWA with window 2048, 5–14 Mamba. All size 896. |
| A2 Qwen4Exp | 1664 | 0 Scratch 4 with speculation off, 8 with MTP. 1 Full and 2–5 Mamba (2–6 with MTP), each the hash block size. Set Full `tokens_per_state` to `"1"` and Mamba to `"-1"`. |
| A2 V4 Flash | 8 | 0 Sparse MLA 512 with `"4"`, 1 Sparse MLA 16384 with `"128"`, 2–3 MLA SWA 128 with window 128, 4 MLA SWA 8 with window 8, 5 MLA SWA 32 with window 128. Set `tokens_per_state` to `"1"` on the MLA SWA groups. |

### Static event endpoints

For stock HTTP vLLM 0.30.0, point Router to the worker's live and replay addresses in `layout.json`. Use the worker's model ID returned by `/v1/models`, the topic generated in the worker terminal, and its actual ports and block size.

```json
{
  "http://127.0.0.1:8000": {
    "endpoints": ["tcp://127.0.0.1:5557"],
    "replay_endpoints": ["tcp://127.0.0.1:5558"],
    "ranks": [
      {
        "rank": 0,
        "model": "/models/Qwen3-4B-Instruct-2507",
        "publisher_topic": "replace-with-KV_TOPIC",
        "hash_block_size": 16,
        "speculative_tokens": 0,
        "declared_groups": 1,
        "groups": [
          {"kind": "full_attention", "block_size": 16}
        ]
      }
    ]
  }
}
```

```bash
vllm-router --policy kv_aware --worker-urls http://127.0.0.1:8000 \
  --kv-hybrid-layout layout.json
```

### Layout field reference

The top-level object maps Router worker URLs to entries. Use the same scheme, host, and port as the configured worker URL. Set only the fields needed for the deployment. Use effective worker values for every token size.

| Worker entry field | JSON type | When to set it |
| --- | --- | --- |
| `ranks` | Array of rank objects | Required for a layout declaration. Include each DP rank to score. |
| `adapters` | Array of objects | Optional. Include entries for LoRA reuse. |
| `adapters[].name` | String | Required in each adapter entry. Use the served adapter name. |
| `adapters[].path` | String | Required in each adapter entry. Use the exact path string passed to `--lora-modules`. |
| `endpoints` | Array of TCP address strings | Required when supplying static live addresses. List them in DP-rank order. |
| `replay_endpoints` | Array of TCP address strings | Required with static live addresses. Supply one replay address for each live address in the same order. |
| `compatible_decode_urls` | Array of worker URL strings | Required on hybrid Prefill entries. List the Decode workers able to receive their KV transfers. |

| Rank field | JSON type and unit | When to set it |
| --- | --- | --- |
| `rank` | Unsigned integer, zero-based DP rank | Required. Use `0` for a single DP rank. |
| `model` | String | Required with static event addresses. Otherwise discovered from the worker. |
| `publisher_topic` | String | Required with static event addresses. Otherwise discovered per rank. Use the topic of the current worker instance. |
| `hash_seed` | String | Set only when the worker uses a custom seed. Use its exact `PYTHONHASHSEED` value. |
| `root_hash` | String, 64 hex digits representing 32 bytes | Use only to supply a custom worker root directly. Supply either the root or the seed. |
| `hash_block_size` | Positive unsigned integer, tokens | Use the worker's hash step. Omit with HTTP event discovery when metrics report it. |
| `partial_hash_hits` | Boolean | Set to `true` only when the worker publishes and reuses partial prefixes at hash-block granularity. |
| `speculative_tokens` | Unsigned integer, tokens | Required. Use `num_speculative_tokens` from the worker configuration, or `0` when speculation is disabled. |
| `hash_lookahead_tokens` | Unsigned integer, tokens | Set only when speculation is enabled. Use the actual hash lookahead, including `0`. |
| `tail_drop_groups` | Array of unsigned group IDs | Required with speculation and zero hash lookahead. List the groups whose lookup drops a trailing unit. Supply `[]` if the worker skips that step. |
| `dense_lookup_groups` | Array of arrays of unsigned group IDs | Set only when declaring ordered dense lookup batches for fine-grained speculation. Include each full-attention group once in the engine's lookup order. |
| `declared_groups` | Unsigned integer, group count | Set to `1` for a manually declared single full-attention group. |
| `groups` | Array of group objects | Required. Use the complete cache group list, including groups that publish no events. |

| Group field | JSON type and unit | When to set it |
| --- | --- | --- |
| `kind` | String | Required. Use the matching entry in the [group table](#layout-template). |
| `block_size` | Positive unsigned integer, tokens | Required. Use the group block size. |
| `sliding_window` | Positive unsigned integer, tokens | Required for sliding-window groups. Use the engine's window length. |
| `prefix_replay_tokens` | Unsigned integer, tokens | Set the replay window only for a sliding group rebuilt by bounded replay. |
| `mamba_cache_mode` | String | Required for a Mamba checkpoint group. Use `"align"`. |
| `tokens_per_state` | String containing a token count | Required for sparse MLA and scratch. Use the number of tokens represented by one state for sparse MLA and `"1"` for scratch. Follow the recipe for other engine-reported values. |
| `cache_role` | String | Set to `"sparse"` for sparse MLA groups. Omit when the engine reports no special cache role. |

## Check cache reuse

The Router estimates placement from the KV events it has applied. The worker reports actual reuse. Requests with an unrecognized adapter, cache salt or extension still route, with no cache credit.

In an idle worker pool, send the same token-ID prompt twice. Use the served model ID in `SERVED_MODEL` (the checkpoint path by default) and the recipe's probe length in `PROMPT_TOKENS`. The worker's context must fit the prompt and one output token. If your cache-hit alignment differs from the recipe, use a prompt longer than your worker's alignment.

```bash
SERVED_MODEL="$MODEL"
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
| Worker local reuse | Worker `/metrics`. Look for `vllm:prefix_cache_hits_total`. With `--enable-prompt-tokens-details` on HTTP, inspect `usage.prompt_tokens_details.cached_tokens`. | Increases on the repeat. For P/D, inspect Prefill. |
| P/D transfer | Send a fresh prompt. Compare Decode's `vllm:nixl_bytes_transferred_sum` counter. | Requests that need remote KV complete a read. |
| MTP or DSpark | On the worker generating tokens, inspect `vllm:spec_decode_num_draft_tokens_total` or the speculative-decoding log. | Drafted tokens increase during generation. |

## CLI options

Use these options when you need a layout, a static event source or an override. See the [CLI reference](../cli_reference.md) for shared routing and metrics options.

| Option | Type | Default | Use |
| --- | --- | --- | --- |
| `--kv-hybrid-layout` | File path | Unset | Read the layout and P/D compatibility pairs keyed by worker URL. |
| `--kv-event-source` | `URL=ENDPOINT[,ENDPOINT...]`, repeatable | Empty | Static live event addresses. |
| `--kv-event-replay-source` | `URL=ENDPOINT[,ENDPOINT...]`, repeatable | Empty | Static replay addresses. |
| `--kv-event-block-size` | `u32` | `16` | Hash unit in tokens for static event sources. Match the layout's `hash_block_size` when supplying both. |
| `--kv-event-model` | String | Unset | Served model ID for static event sources. |
| `--pd-kv-events` | `URL=ENDPOINT[,ENDPOINT...]`, repeatable | Empty | Static P/D live endpoints in DP-rank order for 16-token blocks. |
| `--kv-event-strict-recovery` | Flag | `false` | Require complete recovery or a worker KV clear before scoring reuse. |
| `--router-temperature` | `f64` | `0.0` | KV-aware selection temperature for colocated and Prefill placement. Zero selects the lowest cost. |
| `--kv-penalty-host` | `f64` | `0.8` | Reuse credit for host-resident KV. |
| `--kv-penalty-external` | `f64` | `0.6` | Reuse credit when a placement provider reports an external tier. |
| `--kv-penalty-disk` | `f64` | `0.3` | Reuse credit for disk-resident KV. |
| `--kv-aware-session-affinity-ttl-secs` | `u64`, seconds | Unset | Enable idle session affinity for 1–31,536,000 seconds. In P/D, applies to Decode. |
| `VLLM_ROUTER_TOKENIZER` | Path, environment variable | Unset | Local model directory for gRPC text and chat rendering. |

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Worker reports reuse and Router stays at zero | Check the served model, hash settings, publisher topic and reachability of both event ports. Inspect `/kv_event_sources` or `Control.GetKvEventSources`. |
| Router rejects the layout | Use `--log-level debug` and inspect `worker-reported KV layout rejected`. Match the JSON to the actual model, rank, groups and speculation settings. |
| Worker reports `Address already in use` | Check HTTP, gRPC, event and connector ports. Reserve separate ranges for DP ranks. |
| Worker restarted | Generate a new publisher topic. Update a static layout that pins the topic. A Router restart uses the available replay history. |
| Decode has no remote read | Check whether the prompt was already cached on Decode, the connector and role settings, and Decode's transfer log. |
