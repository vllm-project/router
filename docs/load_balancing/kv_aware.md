# KV-aware routing

`kv_aware` routes requests using prefix reuse and worker load. Use the Router's OpenAI API with HTTP or gRPC vLLM workers. The commands below add KV-aware routing to a working vLLM deployment.

## Versions

Pair `vllm-rs` with the same vLLM release as its Python engine. Use the Rust `vllm-router` binary for the examples below.

| vLLM | HTTP event sources | gRPC event sources | Layout and hashing | vLLM Ascend pairing |
| --- | --- | --- | --- | --- |
| 0.30.0 | [Static addresses and layout](#static-event-endpoints) | `Control.GetKvEventSources` | Zero hash lookahead | [0.30.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.30.0) |
| 0.31.0–0.32.0 | `/kv_event_sources` | `Control.GetKvEventSources` | Zero hash lookahead. Supports V4.1 bounded replay with the corresponding model layout. | [0.31.0rc1](https://pypi.org/project/vllm-ascend/0.31.0rc1/) or [0.32.0rc1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.32.0rc1), respectively |
| 0.33.0 | `/kv_event_sources` | `Control.GetKvEventSources` | Supports nonzero hash lookahead carried by `BlockStored.lookahead_token_ids`. | Build against vLLM 0.33.0 |
| 0.34.0 | `/kv_event_sources` | `Control.GetKvEventSources` | Uses 64-token blocks when the backend has no preferred size. Use the worker's actual sizes in a layout. | Build against vLLM 0.34.0 |

Use one worker interface throughout a pool. gRPC P/D uses explicitly configured worker URLs. Router renders chat locally using `vllm-chat` pinned to vLLM 0.30.0. See the [gRPC API compatibility table](../backend/grpc.md#api-compatibility) for request features and tokenizer settings.

## Choose a layout configuration

| Deployment | What to supply |
| --- | --- |
| Ordinary single-group full attention with event discovery, the default hash seed, and no sliding window or speculation | Worker URLs. Router derives the single-group layout and checks the hash contract against worker events. |
| Colocated worker that reports its complete layout | Worker URLs. Router discovers the layout and event addresses. |
| Hybrid P/D with worker-reported layouts | Worker URLs and a [pairing-only file](#pd-layout). |
| Worker with event discovery that needs a layout declaration | A [layout file](#layout-template) for the missing group and hash information. Router discovers the served model, publisher topic and event addresses. Add Prefill-to-Decode pairs for hybrid P/D. |
| Worker without event discovery, such as stock HTTP vLLM 0.30.0 | [Static event addresses and a layout](#static-event-endpoints). |

A manual layout describes the worker's cache groups, hashing and reuse rules. With event discovery, leave live and replay addresses, publisher topic and model ID out of the file.

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

Set `MODEL` to your worker's checkpoint. Run worker and Router commands in separate terminals. The example addresses assume both run on the same host.

In the Router terminal, use `LAYOUT_ARGS=()` when Router can derive or discover the layout. For an explicit layout, use `LAYOUT_ARGS=(--kv-hybrid-layout layout.json)`.

### Colocated prefill and decode

#### HTTP workers

```bash
vllm serve "$MODEL" --host 127.0.0.1 \
  --enable-scale-out "${KV_ARGS[@]}"
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
| Connector role | `kv_producer` | `kv_consumer` |
| NIXL side-channel port | `5600` | `5601` |

Set the ports and transfer role in each stage's terminal. Run the [event setup](#set-the-event-ports) on Prefill. For NIXL Mamba transfers, set `VLLM_SSM_CONV_STATE_LAYOUT=DS` on both stages.

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
KV_ARGS=()
```

```bash
# Both terminals
TRANSFER=$(printf '{"kv_connector":"NixlConnector","kv_role":"%s"}' "$KV_ROLE")
```

In the Router terminal, use `LAYOUT_ARGS=(--kv-hybrid-layout layout.json)` when a [P/D layout](#pd-layout) is required and `LAYOUT_ARGS=()` otherwise.

#### HTTP P/D

Run the worker command on Prefill first, then Decode.

```bash
vllm serve "$MODEL" --host 127.0.0.1 --port "$HTTP_PORT" \
  --enable-scale-out "${KV_ARGS[@]}" --kv-transfer-config "$TRANSFER"
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

#### P/D layout

For hybrid P/D, list the Decode workers that can receive KV from each Prefill. Add a [layout declaration](#layout-template) for each scored worker that needs one.

Use the same worker URL as the Router command, without a DP-rank suffix. Leave Program scheduling disabled for hybrid P/D layouts.

If each scored worker reports its layout, the file only needs the Prefill-to-Decode pairs.

```json
{
  "http://127.0.0.1:8000": {
    "compatible_decode_urls": [
      "http://127.0.0.1:8001"
    ]
  }
}
```

## Layout files

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

### Model layout reference

Use these profiles for manual layouts. Use token sizes from the running worker.

| Platform and vLLM | Checkpoint | Layout |
| --- | --- | --- |
| GPU, 0.33.0 | Qwen3-4B-Instruct-2507 | Automatic for single-group full attention |
| GPU, 0.33.0 | Qwen3.5-4B | GPU Qwen, 576-token groups |
| GPU, 0.33.0 | Qwen3.6-27B | GPU Qwen, 832-token groups |
| GPU, 0.33.0 | Qwen3.8-27B | GPU Qwen, 832-token groups. With DSpark, use GPU Qwen3.8 DSpark. |
| GPU, 0.33.0 | Qwen3.8-Flash-Next-FP8 (Qwen4Exp QSA) | GPU Qwen4Exp |
| GPU, 0.33.0 | DeepSeek-V4-Flash-0731 | GPU V4 Flash |
| GPU, 0.33.0 | DeepSeek-V4-Pro | GPU V4 Pro |
| GPU, 0.33.0 | DeepSeek-V4.1-Flash (1017) | GPU V4.1 Flash with bounded replay |
| Ascend A2, 0.32.0 | Qwen3-4B | A2 Qwen3 |
| Ascend A2, 0.32.0 | Qwen3.5-4B | A2 Qwen, 640-token groups. With DSpark, use A2 Qwen3.5 DSpark. |
| Ascend A2, 0.32.0 | Qwen3.6-27B W8A8 | A2 Qwen, 896-token groups |
| Ascend A2, 0.32.0 | Qwen3.8-27B W8A8 | A2 Qwen, 896-token groups. Use the MTP profile for gRPC. |
| Ascend A2, 0.32.0 | Qwen3.8-27B BF16 with DSpark | A2 Qwen3.8 DSpark |
| Ascend A2, 0.32.0 | Qwen3.8-Flash-Next W4A8 | A2 Qwen4Exp |
| Ascend A2, 0.32.0 | DeepSeek-V4-Flash-0731 W8A8 | A2 V4 Flash |

List groups in engine order. Use the hash sizes below as references for manual declarations.

| Group | `kind` | Required fields for this group |
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

### MTP and DSpark

When a worker runs MTP or DSpark and requires a manual layout, set `speculative_tokens` to the worker's configured count. Match the hash lookahead and tail-drop groups to the worker's runtime. Workers that report their layouts supply these values automatically.

| Worker profile | `hash_lookahead_tokens` | `tail_drop_groups` |
| --- | ---: | --- |
| GPU Qwen3.8, MTP with hash lookahead | 1 | Omit |
| GPU Qwen3.8, MTP with hash lookahead disabled | 0 | `[3]` |
| GPU Qwen3.8 or V4 Flash, DSpark | 0 | `[]` |
| Ascend A2 Qwen3.5, Qwen3.6, Qwen3.8 W8A8 or Qwen4Exp, MTP | 0 | `[]` |
| Ascend A2 Qwen3.5, Qwen3.6, Qwen3.8 BF16 or V4 Flash, DSpark | 0 | `[]` |

### Layout field reference

The top-level object maps Router worker URLs to entries. Use the same scheme, host, and port as the configured worker URL. Set only the fields needed for the deployment. Use effective worker values for every token size.

| Worker entry field | JSON type | When to set it |
| --- | --- | --- |
| `ranks` | Array of rank objects | Required for a layout declaration. Include each DP rank to score. |
| `adapters` | Array of objects | Optional. Include entries for LoRA reuse. |
| `adapters[].name` | String | Required in each adapter entry. Use the served adapter name. |
| `adapters[].path` | String | Required in each adapter entry. Use the exact path string passed to `--lora-modules`. |
| `compatible_decode_urls` | Array of worker URL strings | Required on hybrid Prefill entries. List the Decode workers able to receive their KV transfers. |

| Rank field | JSON type and unit | When to set it |
| --- | --- | --- |
| `rank` | Unsigned integer, zero-based DP rank | Required. Use `0` for a single DP rank. |
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

## Static event endpoints

Use this fallback only if the worker cannot report event sources, such as stock HTTP vLLM 0.30.0. In colocated mode, `--kv-event-source` and `--kv-event-replay-source` can supply the addresses. The static setup also needs the publisher topic and hash contract, which the example supplies in the same file. Supply each address once.

<details>
<summary>Static event source for a single full-attention worker</summary>

Replace `SERVED_MODEL_ID` with the worker's `id` from `/v1/models` and `replace-with-KV_TOPIC` with its current event topic. Use the worker's actual event addresses and block size.

```json
{
  "http://127.0.0.1:8000": {
    "endpoints": ["tcp://127.0.0.1:5557"],
    "replay_endpoints": ["tcp://127.0.0.1:5558"],
    "ranks": [
      {
        "rank": 0,
        "model": "SERVED_MODEL_ID",
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

</details>

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

## Router options

| Option | Type | When to use it |
| --- | --- | --- |
| `--policy kv_aware` | Policy name | Score reuse in a colocated pool. |
| `--prefill-policy kv_aware` | Policy name | Score Prefill reuse in P/D. |
| `--decode-policy kv_aware` | Policy name | Score Decode reuse in P/D when Decode publishes KV events. |
| `--kv-hybrid-layout` | JSON file path | Supply missing layout data, static event addresses or hybrid P/D pairs. |

See the [CLI reference](../cli_reference.md#routing-policies) for the full option list.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Worker reports reuse and Router stays at zero | Check the served model, hash settings, publisher topic and reachability of both event ports. Inspect `/kv_event_sources` or `Control.GetKvEventSources`. |
| Router rejects the layout | Use `--log-level debug` and inspect `worker-reported KV layout rejected`. Match the JSON to the actual model, rank, groups and speculation settings. |
| Worker reports `Address already in use` | Check HTTP, gRPC, event and connector ports. Reserve separate ranges for DP ranks. |
| Worker restarted | Generate a new publisher topic. Update a static layout that pins the topic. A Router restart uses the available replay history. |
| Decode has no remote read | Check whether the prompt was already cached on Decode, the connector and role settings, and Decode's transfer log. |
