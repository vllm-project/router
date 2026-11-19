# KV-aware routing

If requests often start with the same prompt, such as a shared system message, `kv_aware` prefers a worker that can reuse that prefix and also considers worker load. vLLM publishes cache events so the Router can track which workers have the prefix. When cache information is unavailable, the Router routes by load.

Clients send OpenAI HTTP requests to the Router with either HTTP or gRPC workers. Pick the setup that matches your deployment.

| Deployment | Layout JSON to pass to the Router |
| --- | --- |
| [Full attention on standard vLLM](#colocated-prefill-and-decode), HTTP or gRPC | Pass the [one-group layout](#full-attention-layout) with `--kv-hybrid-layout`. |
| Colocated HTTP worker that exposes `/kv_routing_capabilities` for one full-attention group | Omit the layout when [the worker meets the discovery requirements](#full-attention-layout). |
| [Hybrid attention](#hybrid-attention-models), HTTP or gRPC | Pass the checkpoint's group layout with `--kv-hybrid-layout`. |
| [P/D disaggregation](#pd-routing), HTTP or gRPC | Pass [`pd-layout.json`](#pd-layout) with `--pd-hybrid-config` for this guide's `sha256_cbor` Prefill setup. |

## Colocated prefill and decode

Each worker handles the entire request. The walkthrough uses one Qwen3-4B-Instruct-2507 worker, vLLM 0.33.0, and the Rust `vllm-router` binary. Check [version compatibility](#versions) if you use another release.

### 1. Prepare the worker settings

In the worker terminal, set a local model directory and publish full 32-byte hashes in KV events. Use `tcp://*:5557` to bind the live publisher. The replay port lets the Router recover recent events after a disconnect. Keep the default incremental reporting.

```bash
MODEL=/models/Qwen3-4B-Instruct-2507
export VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0
unset PYTHONHASHSEED
EVENTS='{"enable_kv_cache_events":true,"endpoint":"tcp://*:5557","replay_endpoint":"tcp://127.0.0.1:5558","topic":"qwen3-start01"}'
```

#### Full-attention layout

For one full-attention group without speculative decoding, LoRA or a KV connector, a colocated HTTP worker with `/kv_routing_capabilities` supplies the layout automatically. Skip this file and omit `--kv-hybrid-layout` from the Router command in that case.

For standard vLLM, save this one-group example as `layout.json`. Copy `group_idx: 0` as shown for a 16-token block size. Set the model name and topic to match the worker. Use `http://127.0.0.1:8000` as the top-level key for HTTP or keep the `grpc://` key for gRPC.

```json
{
  "grpc://127.0.0.1:50051": {
    "schema_version": 1,
    "ranks": [
      {
        "rank": 0,
        "model": "qwen3",
        "weight_version": "default",
        "publisher_topic": "qwen3-start01",
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

The sample uses the vLLM default seed with `PYTHONHASHSEED` unset. If the worker sets `PYTHONHASHSEED=42`, use `"hash_seed": "42"`. You can also supply the engine's 64-digit hexadecimal `root_hash`. The two fields must agree when both are present. For LoRA, add `"adapters": [{"name": "my-adapter", "path": "/models/my-adapter"}]` alongside `ranks` and use the path from `--lora-modules`.

### 2. Start the worker and Router

Choose one transport. Run the worker and Router in separate terminals.

#### HTTP workers

```bash
vllm serve "$MODEL" --host 127.0.0.1 --port 8000 --served-model-name qwen3 \
  --enable-prefix-caching --prefix-caching-hash-algo sha256_cbor --block-size 16 \
  --kv-events-config "$EVENTS"
```

Set the top-level key in `layout.json` to `http://127.0.0.1:8000`, then start the Router.

```bash
vllm-router --host 127.0.0.1 --port 19000 --policy kv_aware \
  --worker-urls http://127.0.0.1:8000 --kv-hybrid-layout layout.json
```

#### gRPC workers

Use [`vllm-rs` from the same vLLM version](../backend/grpc.md#starting-vllm-029). Activate that vLLM Python environment before starting the worker.

```bash
"${VLLM_RS:-vllm-rs}" serve "$MODEL" --host 127.0.0.1 --port 8000 --grpc-port 50051 --served-model-name qwen3 -- \
  --enable-prefix-caching --prefix-caching-hash-algo sha256_cbor \
  --block-size 16 --kv-events-config "$EVENTS"
```

Keep the layout key `grpc://127.0.0.1:50051`. In the Router terminal, set `MODEL` to a local copy of the worker's model directory with its tokenizer files.

```bash
VLLM_ROUTER_TOKENIZER="$MODEL" vllm-router --host 127.0.0.1 --port 19000 \
  --policy kv_aware --worker-urls grpc://127.0.0.1:50051 --kv-hybrid-layout layout.json
```

#### Static event endpoints

On stock HTTP vLLM 0.30.0, append these options to the Router command. Change the worker URL, model and block size if you changed the example.

```bash
--kv-event-source http://127.0.0.1:8000=tcp://127.0.0.1:5557 \
--kv-event-replay-source http://127.0.0.1:8000=tcp://127.0.0.1:5558 \
--kv-event-model qwen3 --kv-event-block-size 16
```

### 3. Check cache reuse

Send the same prompt twice with one output token each. The first request fills the cache. The second should increase the Router's KV-aware hit counter.

```bash
payload=$(jq -nc '{model:"qwen3",prompt:[range(1000;1129)],max_tokens:1,temperature:0}')
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

An increase such as `0 → 1` confirms that KV-aware routing found a cached prefix. If the counts are equal, retry the final request after the Router has received the cache events. Use an idle pool for the check so the counter reflects your requests. To confirm vLLM reuse as well, compare this worker counter before and after the repeat. For P/D, check the Prefill worker.

```bash
curl -fsS http://127.0.0.1:8000/metrics | grep '^vllm:prefix_cache_hits_total'
```

Both worker transports accept `/v1/chat/completions` through the Router. On gRPC, `/v1/completions` accepts one flat token-ID array as shown above.

## Hybrid attention models

[Qwen3.5](#qwen35-4b) · [Qwen3.6](#qwen36-27b) · [Qwen3.8](#qwen38-27b) · [Qwen4Exp](#qwen4exp-fp8-qsa) · [DeepSeek V4 Flash](#deepseek-v4-flash) · [DeepSeek V4 Pro](#deepseek-v4-pro) · [DeepSeek V4.1 Flash](#deepseek-v41-flash)

Start with the [full-attention JSON](#full-attention-layout). Set the worker URL, served model and publisher topic for your deployment. In each rank, replace the four fields shown for your checkpoint. Keep the engine's full cache-group order, including groups that do not publish cache events.

Use the same HTTP or gRPC commands with the selected checkpoint and model settings. Drop `--block-size 16` from the worker command for the hybrid examples. Keep `--prefix-caching-hash-algo sha256_cbor`, the full-width event hashes and the live and replay publishers. The configurations below use vLLM 0.33.0 and a 4,096-token context. Adjust memory and parallelism for your hardware and confirm the layout again if the engine changes its cache groups.

### Qwen3.5-4B

Add `--max-model-len 4096 --prefix-match-unit 16` to vLLM. The layout uses aligned Mamba checkpoints. For NIXL P/D, set `VLLM_SSM_CONV_STATE_LAYOUT=DS` on Prefill and Decode.

```json
{
  "hash_block_size": 16,
  "partial_hash_hits": true,
  "declared_groups": 4,
  "groups": [
    {"group_idx": 0, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 576, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
    {"group_idx": 1, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 576, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
    {"group_idx": 2, "kind": "mamba", "spec_type": "MambaSpec", "block_size": 576, "prefix_cacheable": true, "prefix_replay_tokens": 0, "mamba_cache_mode": "align", "prefill_checkpoint_alignment": 64},
    {"group_idx": 3, "kind": "full_attention", "spec_type": "FullAttentionSpec", "block_size": 576, "prefix_cacheable": true, "prefix_replay_tokens": 0}
  ]
}
```

### Qwen3.6-27B

Use `--max-model-len 4096 --prefix-match-unit 16`. Copy the Qwen3.5 rank fields and set all four `groups[].block_size` values to `832`. Keep `prefill_checkpoint_alignment: 64`, `hash_block_size: 16` and `partial_hash_hits: true`.

### Qwen3.8-27B

For the base checkpoint, use the [Qwen3.6 settings](#qwen36-27b) with the Qwen3.8 weights and served model name. Update the rank when enabling speculation.

Enable MTP or DSpark on the worker with `--speculative-config`. In Router's layout, `speculative_tokens` is the configured draft length and `hash_lookahead_tokens` is the number of tokens after each block that enter its hash. For lookahead 1 and a 16-token block, token 17 is part of the hash of tokens 1–16. For lookahead 0, set `tail_drop_groups` to the groups for which the engine rolls back the last matched unit. DSpark uses `[]` in the examples below. Use the values in effect on Prefill for P/D disaggregation.

| vLLM configuration | Rank settings |
| --- | --- |
| `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` in colocated routing | Keep the four base groups. Set `speculative_tokens: 3` and `hash_lookahead_tokens: 1`. |
| MTP with `"disable_lookahead_block_hashes":true` added to the worker's `--speculative-config`, or the tested Mooncake P/D setup | Keep the four base groups. Set `speculative_tokens: 3`, `hash_lookahead_tokens: 0` and `tail_drop_groups: [3]`. |
| `--speculative-config '{"method":"dspark","num_speculative_tokens":8,"model":"/models/Qwen3.8-27B-speculator.dspark"}'` | Set `speculative_tokens: 8`, `hash_lookahead_tokens: 0`, `tail_drop_groups: []` and `partial_hash_hits: false`. Keep `hash_block_size: 16`. |

With the [Qwen3.8 DSpark draft](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark), set `declared_groups: 15`. Groups 0–9 use the 832-token Mamba settings above, and groups 10–13 use the 832-token full-attention settings. Group 14 uses `kind: "sliding_window"`, `spec_type: "SlidingWindowSpec"`, `block_size: 832`, `sliding_window: 2048`, `prefix_cacheable: true` and `prefix_replay_tokens: 0`.

### Qwen4Exp FP8 QSA

Use the FP8 QSA checkpoint with architecture `Qwen4ExpForConditionalGeneration`. The reference checkpoint has the name `Qwen3.8-Flash-Next-FP8`. Use `--max-model-len 4096` and let the engine choose cache block sizes. Use a prompt longer than 1,600 tokens to check reuse.

```json
{
  "hash_block_size": 1600,
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
```

### DeepSeek V4 Flash

Use the V4 Flash 0731 checkpoint with `--max-model-len 4096` and the engine-selected cache block sizes. For DSpark, add `--speculative-config '{"method":"dspark","num_speculative_tokens":4}'`. Keep these groups and set `speculative_tokens: 4`, `hash_lookahead_tokens: 0` and `tail_drop_groups: []`. The reference DSpark P/D setup uses NIXL.

```json
{
  "hash_block_size": 4,
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
```

### DeepSeek V4 Pro

Use `--max-model-len 4096` and the engine-selected cache block sizes. The reference V4 Pro checkpoint used `--tensor-parallel-size 4`.

```json
{
  "hash_block_size": 4,
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
```

### DeepSeek V4.1 Flash

Use `--max-model-len 4096`, keep `--swa-bounded-replay` enabled, and let vLLM choose cache block sizes. The V4.1 Flash 1017 reference used `--tensor-parallel-size 2`. Groups 0–4 rebuild their last 128 tokens after a cache hit. Keep their replay settings in the layout.

```json
{
  "hash_block_size": 128,
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
```

To [check reuse](#3-check-cache-reuse) for a hybrid model, use its served name and a repeated prompt long enough to span its cache pages. The configurations describe the tested CUDA layouts, including the declared speculative modes. Check connector compatibility when deploying a model in P/D.

## P/D routing

P/D disaggregation puts prompt processing on Prefill workers and token generation on Decode workers. This example uses NIXL and enables KV-aware routing for Prefill.

1. Start two workers on different GPUs with the same checkpoint, vLLM version, connector and transport. Keep Prefill on HTTP port `8000` or gRPC port `50051`. Move Decode to HTTP port `8001` and, for a gRPC worker, gRPC port `50052`.
2. Set `VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1` on both workers. Set `VLLM_NIXL_SIDE_CHANNEL_PORT=5600` on Prefill and `5601` on Decode. For separate hosts, use reachable addresses.
3. Add `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'` to Prefill and the same option with `"kv_role":"kv_consumer"` to Decode. Enable events and replay on Prefill. Omit `--kv-events-config` on Decode for this setup. For gRPC, pass these engine options after `--`.
4. Create [`pd-layout.json`](#pd-layout) with the Prefill layout, its live and replay addresses, and the compatible Decode URL. Leave Program scheduling disabled for declared P/D layouts.

For gRPC, export `VLLM_ROUTER_TOKENIZER` with the model directory in the Router terminal.

```bash
PREFILL=http://127.0.0.1:8000
DECODE=http://127.0.0.1:8001
# For gRPC, use grpc://127.0.0.1:50051 and grpc://127.0.0.1:50052.
vllm-router --port 19000 --vllm-pd-disaggregation --prefill "$PREFILL" --decode "$DECODE" \
  --prefill-policy kv_aware --decode-policy round_robin --kv-connector nixl \
  --pd-hybrid-config pd-layout.json
```

Use the same client URL and [reuse check](#3-check-cache-reuse). To check NIXL transfer, send a fresh prompt and compare `vllm:nixl_bytes_transferred_sum` on Decode's HTTP metrics port `8001` before and after the request.

### P/D layout

Save this as `pd-layout.json`. Replace the empty `hybrid` object with the complete value under the Prefill worker URL in your colocated layout. Match `model` and `block_size` to the Prefill rank's model and `hash_block_size`. List addresses in rank order. For gRPC, use the Prefill and Decode gRPC URLs with their service ports.

```json
{
  "http://127.0.0.1:8000": {
    "endpoints": ["tcp://127.0.0.1:5557"],
    "replay_endpoints": ["tcp://127.0.0.1:5558"],
    "block_size": 16,
    "model": "qwen3",
    "hybrid": {},
    "compatible_decode_urls": ["http://127.0.0.1:8001"]
  }
}
```

## Versions

Use the same vLLM release for the engine and its gRPC frontend. The table tells you how to supply the live and replay addresses in **colocated** deployments. Replay recovers missed events from the worker's recent history.

| vLLM release | Colocated HTTP | Colocated gRPC | Hybrid or speculative decoding |
| --- | --- | --- | --- |
| [0.30.0](https://github.com/vllm-project/vllm/tree/v0.30.0) | Pass [static live and replay addresses](#static-event-endpoints). | `GetKvEventSources` supplies both addresses. | Use layouts with zero hash lookahead. The V4.1 bounded-replay recipe requires 0.31 or newer. |
| [0.31.0](https://github.com/vllm-project/vllm/tree/v0.31.0) | `/kv_event_sources` supplies both addresses. | Same as 0.30.0. | Adds V4.1 bounded replay. Use layouts with zero hash lookahead. |
| [0.32.0](https://github.com/vllm-project/vllm/tree/v0.32.0) | Same as 0.31.0. | Same as 0.30.0. | Same KV event and layout options as 0.31.0. |
| [0.33.0](https://github.com/vllm-project/vllm/tree/v0.33.0) | Same as 0.31.0. | Same as 0.30.0. | Adds events for nonzero hash lookahead. Use this release for the model recipes above. |
| [0.34.0](https://github.com/vllm-project/vllm/tree/v0.34.0) | Same as 0.31.0. | Same as 0.30.0. | Same Router setup as 0.33.0. Check the layout against your running model. |

Standard workers on all five versions need layout JSON for the `sha256_cbor` setup in this guide. Address discovery covers the event connection. The JSON supplies the hash root and cache groups. In P/D disaggregation, include the Prefill addresses in [`pd-layout.json`](#pd-layout). Hash lookahead only matters if the worker uses future tokens when hashing cached blocks, such as the [Qwen3.8 MTP example](#qwen38-27b).

## CLI options

| Option | Value format or example | Rust type | Default | Description |
| --- | --- | --- | --- | --- |
| `--policy` | `kv_aware` | enum | `cache_aware` | Selects the colocated policy and the fallback policy for both P/D stages. |
| `--worker-urls` | `http://worker:8000` or `grpc://worker:50051` | list of strings | `[]` | Registers colocated workers. Use one transport per pool. |
| `--kv-hybrid-layout` | `layout.json` | path | unset | Loads a colocated full- or hybrid-attention layout keyed by worker URL. |
| `--kv-event-source` | `WORKER_URL=ENDPOINT[,ENDPOINT...]` | string, repeatable | `[]` | Supplies live ZMQ addresses when event discovery is unavailable. |
| `--kv-event-replay-source` | `WORKER_URL=ENDPOINT[,ENDPOINT...]` | string, repeatable | `[]` | Supplies replay addresses for a matching static live source. |
| `--kv-event-block-size` | `16` | `u32` | `16` | Sets the block size in tokens for static colocated sources. |
| `--kv-event-model` | `qwen3` | string | unset | Sets the served model name when using static colocated sources. |
| `--intra-node-data-parallel-size` | `4` | `usize` | `1` | Sets the data-parallel rank count per worker URL. List event endpoints in rank order. |
| `--vllm-pd-disaggregation` | no value | flag | `false` | Enables vLLM P/D disaggregation. |
| `--prefill` | `URL [PORT or none]` | URL with optional `u16` port, repeatable | `[]` | Adds a Prefill worker. The optional port is for connector bootstrap. |
| `--decode` | `URL` | string, repeatable | `[]` | Adds a Decode worker. |
| `--prefill-policy` | `kv_aware` | enum | inherits `--policy` | Enables KV-aware prefix routing on Prefill. |
| `--decode-policy` | `round_robin` or `kv_aware` | enum | inherits `--policy` | Sets the Decode policy. See [Decode load and affinity](README.md#pd-decode-load). |
| `--kv-connector` | `nixl`, `mooncake`, `moriio` | enum | `nixl` | Matches the workers' P/D transfer connector. MoRI-IO requires vLLM ZMQ service discovery. |
| `--vllm-discovery-address` | `0.0.0.0:30001` | string | unset | Listens for vLLM ZMQ worker registrations in P/D. |
| `--pd-hybrid-config` | `pd-layout.json` | path | unset | Supplies Prefill sources, their layouts and compatible Decode URLs. |
| `--pd-kv-events` | `WORKER_URL=ENDPOINT[,ENDPOINT...]` | string, repeatable | `[]` | Supplies live publishers for 16-token P/D sources without a declared layout. |
| `--router-temperature` | finite value ≥ `0` | `f64` | `0.0` | Samples more broadly with higher values in colocated routing and P/D Prefill. Zero selects the lowest-cost worker. |
| `--kv-penalty-host` | `0.0` to `1.0` | `f64` | `0.8` | Sets the fraction of full cache credit for a hit in host memory. |
| `--kv-penalty-external` | `0.0` to `1.0` | `f64` | `0.6` | Sets the fraction for a hit in external storage. |
| `--kv-penalty-disk` | `0.0` to `1.0` | `f64` | `0.3` | Sets the fraction for a hit on disk. |
| `--kv-aware-session-affinity-ttl-secs` | `1` to `31536000` seconds | `u64` | unset | Enables session affinity until the idle timeout. In P/D it applies to Decode. |
| `--prometheus-port` | `29000` | `u16` | `29000` | Sets the Router metrics port used in the reuse check. |

For session affinity, send the same `x-session-id` or `session_params.session_id` on each turn. See the [CLI reference](../cli_reference.md) for launcher availability and other Router options.

## Troubleshooting

- If the engine reports hits and the Router does not, check the layout's model and topic, the hash flags, and whether both event ports are reachable from the Router. Use distinct ports for workers on the same host.
- When a worker restarts, give it a fresh event topic and update the layout. The Router can recover after its own restart when the worker still has the replay history.
