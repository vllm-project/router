# KV-aware routing

`kv_aware` routes requests using prefix reuse and worker load. Use the Router's OpenAI API with HTTP or gRPC vLLM workers. The commands below add KV-aware routing to a working vLLM deployment.

## Versions

Pair `vllm-rs` with the same vLLM release as its Python engine. Use the Rust `vllm-router` binary for the examples below.

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
| Hybrid P/D with worker-reported layouts | Worker URLs and a [pairing-only file](#hybrid-pd-pairing). |
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
| HTTP text and chat | Add `--enable-scale-out` so Router can use the worker's tokenizer and chat template to score reuse. Workers serving pretokenized prompts without rendering can omit it. |

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

In the Router terminal, use `LAYOUT_ARGS=(--kv-hybrid-layout layout.json)` when using [hybrid P/D](#hybrid-pd-pairing) or a manual layout and `LAYOUT_ARGS=()` otherwise.

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

#### Hybrid P/D pairing

`--prefill` and `--decode` register the worker pools. For hybrid KV transfer, add `compatible_decode_urls` under each Prefill URL in the file passed to `--kv-hybrid-layout`. Router selects only the declared pairs. Use the worker URLs from the command, without DP-rank suffixes.

If a scored worker cannot report its cache layout, add its [cache groups](#layout-template) to the same file. Leave Program scheduling disabled for hybrid P/D.

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

Save the layout as `layout.json` and pass `--kv-hybrid-layout layout.json`. Use your worker's HTTP or gRPC URL as the key. For hybrid P/D, add the [pairing fields](#hybrid-pd-pairing).

Add one `ranks` entry per DP rank to score. Use `rank: 0` for one rank. Set `speculative_tokens: 0` when speculation is off.

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

Choose the row for your checkpoint and mode when writing a manual layout. The token sizes are examples. Use the worker's actual sizes.

| Group | `kind` | Required fields for this group |
| --- | --- | --- |
| Full | `full_attention` |  |
| Mamba | `mamba` | `"mamba_cache_mode":"align"` |
| SWA | `sliding_window` | `sliding_window` from the row |
| MLA SWA | `sliding_window_mla` | `sliding_window` from the row |
| Sparse MLA | `mla_attention` | `"cache_role":"sparse"` and `tokens_per_state` as a string |
| Scratch | `scratch` | `"tokens_per_state":"1"` |

| Platform | Checkpoint or mode | Example hash size (tokens) | Cache groups in index order |
| --- | --- | ---: | --- |
| GPU | Qwen3.5-4B, Qwen3.6-27B, Qwen3.8-27B | Worker-reported size | 0–2 Mamba, 3 Full, same size on all groups |
| GPU | Qwen3.8-27B with DSpark | Worker-reported size | 0–9 Mamba, 10–13 Full, 14 SWA with window 2048 |
| GPU | Qwen3.8-Flash-Next-FP8 (Qwen4Exp QSA) | 1600 | 0–3 Mamba 1600, 4 Scratch 4, 5 Sparse MLA 1600 with `"4"` |
| GPU | DeepSeek-V4-Flash-0731 | 4 | 0–1 MLA SWA 64 with window 128, 2 Sparse MLA 256 with `"4"`, 3 MLA SWA 4 with window 8, 4 MLA SWA 8 with window 128 |
| GPU | DeepSeek-V4-Pro | 4 | 0–1 MLA SWA 64 with window 128, 2 Sparse MLA 256 with `"128"`, 3 MLA SWA 8 with window 128, 4 MLA SWA 4 with window 8 |
| GPU | DeepSeek-V4.1-Flash (1017) with bounded replay | 128 | 0–4 MLA SWA 32 with window 128, `"prefix_replay_tokens":128`. 5 Sparse MLA 128 with `"2"`, 6 Scratch 8. |
| Ascend A2 | Qwen3-4B | 128 | 0 Full 128 |
| Ascend A2 | Qwen3.5-4B, Qwen3.6-27B W8A8, Qwen3.8-27B W8A8 | Worker-reported size | 0 Full, 1–3 Mamba, same size on all groups |
| Ascend A2 | Qwen3.5-4B with DSpark | Worker-reported size | 0–1 Full, 2 SWA with window 2048, 3–7 Mamba |
| Ascend A2 | Qwen3.8-27B BF16 with DSpark | Worker-reported size | 0–3 Full, 4 SWA with window 2048, 5–14 Mamba |
| Ascend A2 | Qwen3.8-Flash-Next W4A8 (Qwen4Exp) | 1664 | 0 Scratch 4 with speculation off, 8 with MTP. 1 Full and 2–5 Mamba (2–6 with MTP), each the hash block size. Set Full `tokens_per_state` to `"1"` and Mamba to `"-1"`. |
| Ascend A2 | DeepSeek-V4-Flash-0731 W8A8 | 8 | 0 Sparse MLA 512 with `"4"`, 1 Sparse MLA 16384 with `"128"`, 2–3 MLA SWA 128 with window 128, 4 MLA SWA 8 with window 8, 5 MLA SWA 32 with window 128. Set `tokens_per_state` to `"1"` on the MLA SWA groups. |

### MTP and DSpark

For Qwen3.8-27B W8A8 on Ascend A2 with gRPC, use the MTP profile.

When a worker runs MTP or DSpark and requires a manual layout, set `speculative_tokens` to the worker's configured count. Match the hash lookahead and tail-drop groups to the worker's runtime. Workers that report their layouts supply these values automatically.

| Worker profile | `hash_lookahead_tokens` | `tail_drop_groups` |
| --- | ---: | --- |
| GPU Qwen3.8, MTP with hash lookahead | 1 | Omit |
| GPU Qwen3.8, MTP with hash lookahead disabled | 0 | `[3]` |
| GPU Qwen3.8 or V4 Flash, DSpark | 0 | `[]` |
| Ascend A2 Qwen3.5, Qwen3.6, Qwen3.8 W8A8 or Qwen4Exp, MTP | 0 | `[]` |
| Ascend A2 Qwen3.5, Qwen3.6, Qwen3.8 BF16 or V4 Flash, DSpark | 0 | `[]` |

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
| Worker reports reuse and Router stays at zero | Check that the worker publishes events and Router can reach both event ports. For a static source, confirm the model ID and publisher topic. |
| Router rejects the layout | Use `--log-level debug` and inspect `worker-reported KV layout rejected`. Match the JSON to the actual model, rank, groups and speculation settings. |
| Worker reports `Address already in use` | Check HTTP, gRPC, event and connector ports. Reserve separate ranges for DP ranks. |
| Worker restarted | Generate a new publisher topic. Update a static layout that pins the topic. A Router restart uses the available replay history. |
| Decode has no remote read | Check whether the prompt was already cached on Decode, the connector and role settings, and Decode's transfer log. |
