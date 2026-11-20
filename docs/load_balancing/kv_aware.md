# KV-aware routing

If requests often start with the same prompt, such as a shared system message, `kv_aware` prefers a worker that can reuse that prefix and also considers worker load. vLLM publishes cache events so the Router can track which workers have the prefix. When cache information is unavailable, the Router routes by load.

Clients send OpenAI HTTP requests to the Router with either HTTP or gRPC workers. Pick the setup that matches your deployment.

| Deployment | Layout JSON to pass to the Router |
| --- | --- |
| [Ordinary single-group full attention](#colocated-prefill-and-decode), HTTP or gRPC | Omit the layout. The Router derives group 0 and verifies the default hash root. |
| Hybrid or sliding-window attention, speculative decoding, or custom hash seed | Pass an [explicit layout](#explicit-layouts) with `--kv-hybrid-layout`. |
| [P/D disaggregation](#pd-routing), HTTP or gRPC | Pass [`pd-layout.json`](#pd-layout) with `--pd-hybrid-config` for this guide's `sha256_cbor` Prefill setup. |

To upgrade, apply the worker settings below and restart workers whose hash or publisher settings change. The Router rebuilds its in-memory index from live and replay events.

## Colocated prefill and decode

Each worker handles the entire request. The walkthrough uses one Qwen3-4B-Instruct-2507 worker, vLLM 0.33.0, and the Rust `vllm-router` binary. Check [version compatibility](#versions) for other releases.

Omitting the layout declares ordinary single-group full attention. With a standard worker, the Router validates the default hash root and block size against a first-block event from live or replay. Until it can verify the source, it routes by load with zero cache credit.

### 1. Prepare the worker settings

Run this before each worker start. Set the live port and let vLLM allocate a replay port. The command generates a fresh publisher topic. Keep the default incremental reporting.

```bash
MODEL=/models/Qwen3-4B-Instruct-2507
export VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0
unset PYTHONHASHSEED
KV_TOPIC=$(python3 -c 'from uuid import uuid4; print(uuid4().hex)')
KV_LIVE_PORT=5557
EVENTS=$(printf '{"enable_kv_cache_events":true,"endpoint":"tcp://*:%s","replay_endpoint":"tcp://*:0","topic":"%s"}' "$KV_LIVE_PORT" "$KV_TOPIC")
```

For multiple worker processes on one host, set a distinct `KV_LIVE_PORT` per process. Set a fixed `replay_endpoint` if the network requires a reserved port. vLLM adds the DP rank to each fixed event port, so reserve non-overlapping ranges for live and replay. These ports are separate from HTTP and gRPC.

On vLLM 0.30.0, use a fixed replay endpoint such as `tcp://*:5558` for one rank.

### 2. Start the worker and Router

Choose one transport. Run the worker and Router in separate terminals. For an [explicit layout](#explicit-layouts), append `--kv-hybrid-layout layout.json` to the Router command.

#### HTTP workers

`--enable-scale-out` exposes the worker's rendering endpoints for text and chat requests. Requests that already contain token IDs can use KV-aware routing without this flag.

```bash
vllm serve "$MODEL" --host 127.0.0.1 --port 8000 --served-model-name qwen3 \
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
"${VLLM_RS:-vllm-rs}" serve "$MODEL" --host 127.0.0.1 --port 8000 --grpc-port 50051 --served-model-name qwen3 -- \
  --enable-prefix-caching --prefix-caching-hash-algo sha256_cbor \
  --block-size 16 --kv-events-config "$EVENTS"
```

In the Router terminal, set `MODEL` to a local copy of the worker's model directory with its tokenizer files.

```bash
VLLM_ROUTER_TOKENIZER="$MODEL" vllm-router --host 127.0.0.1 --port 19000 \
  --policy kv_aware --worker-urls grpc://127.0.0.1:50051
```

#### Static event endpoints

On stock HTTP vLLM 0.30.0, supply an explicit layout and append these options to the Router command. Change the worker URL, model and block size if you changed the example.

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

The Router counter measures prefix matches in its event index. Use an idle pool and repeat the last request if events are still arriving. Requests with an unsupported cache identity still route by load with zero cache credit. To measure actual reuse, compare the worker counter before and after the repeat. For P/D disaggregation, check Prefill.

```bash
curl -fsS http://127.0.0.1:8000/metrics | grep '^vllm:prefix_cache_hits_total'
```

Both worker transports accept `/v1/chat/completions` through the Router. On gRPC, `/v1/completions` accepts one flat token-ID array as shown above.

## Explicit layouts

Use this template to write `layout.json` with the generated topic. Set `WORKER_URL` to `grpc://127.0.0.1:50051` for gRPC. Adapt the fields to the running model and its cache groups.

```bash
WORKER_URL=http://127.0.0.1:8000
cat > layout.json <<EOF
{
  "$WORKER_URL": {
    "schema_version": 1,
    "ranks": [
      {
        "rank": 0,
        "model": "qwen3",
        "weight_version": "default",
        "publisher_topic": "$KV_TOPIC",
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
EOF
```

The sample uses the vLLM default seed with `PYTHONHASHSEED` unset. If the worker sets `PYTHONHASHSEED=42`, use `"hash_seed": "42"`. You can also supply the engine's 64-digit hexadecimal `root_hash`. The two fields must agree when both are present. For LoRA, add `"adapters": [{"name": "my-adapter", "path": "/models/my-adapter"}]` alongside `ranks` and use the path from `--lora-modules`.

## Hybrid attention models

[Qwen3.5](#qwen35-4b) · [Qwen3.6](#qwen36-27b) · [Qwen3.8](#qwen38-27b) · [Qwen4Exp](#qwen4exp-fp8-qsa) · [DeepSeek V4 Flash](#deepseek-v4-flash) · [DeepSeek V4 Pro](#deepseek-v4-pro) · [DeepSeek V4.1 Flash](#deepseek-v41-flash)

Start with the [layout JSON](#explicit-layouts). Set the worker URL, served model and publisher topic for your deployment. In each rank, replace the four fields shown for your checkpoint. Keep the engine's full cache-group order, including groups that do not publish cache events.

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

Enable MTP or DSpark on the worker with `--speculative-config`. In Router's layout, `speculative_tokens` is the configured draft length and `hash_lookahead_tokens` is the number of tokens after each block that enter its hash. For lookahead 1 and a 16-token block, token 17 is part of the hash of tokens 1–16. For lookahead 0, list the groups where the worker actually drops a trailing unit during cache lookup in `tail_drop_groups`. Use `[]` when the worker skips the drop. For P/D disaggregation, declare the rules in effect on Prefill.

| vLLM configuration | Rank settings |
| --- | --- |
| `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` in colocated routing | Keep the four base groups. Set `speculative_tokens: 3` and `hash_lookahead_tokens: 1`. |
| MTP with `"disable_lookahead_block_hashes":true` added to the worker's `--speculative-config`, or the tested Mooncake P/D setup | Keep the four base groups. Set `speculative_tokens: 3`, `hash_lookahead_tokens: 0` and `tail_drop_groups: [3]`. |
| `--speculative-config '{"method":"dspark","num_speculative_tokens":8,"model":"/models/Qwen3.8-27B-speculator.dspark"}'` | Set `speculative_tokens: 8`, `hash_lookahead_tokens: 0`, `tail_drop_groups: []` and `partial_hash_hits: false`. Keep `hash_block_size: 16`. |

With the [Qwen3.8 DSpark draft](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark), set `declared_groups: 15`. Groups 0–9 use the 832-token Mamba settings above, and groups 10–13 use the 832-token full-attention settings. Group 14 uses `kind: "sliding_window"`, `spec_type: "SlidingWindowSpec"`, `block_size: 832`, `sliding_window: 2048`, `prefix_cacheable: true` and `prefix_replay_tokens: 0`.

### Ascend with vLLM 0.32

On vllm-ascend builds with the [Ascend tail-drop rule](https://github.com/vllm-project/vllm-ascend/blob/450e556980df9f1b09392b77952d552d75e4cba2/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py#L42-L61), colocated workers and pure Prefill producers skip the tail drop. Keep the block sizes and group IDs reported by each worker. Use the following rank fields for MTP or DSpark workers that publish zero-lookahead hashes. For P/D disaggregation, put the Prefill layout in the [`hybrid` field](#pd-layout).

| Worker configuration | `speculative_tokens` | `hash_lookahead_tokens` | `tail_drop_groups` |
| --- | ---: | ---: | --- |
| Qwen3.5 or Qwen3.8 MTP with three draft tokens | `3` | `0` | `[]` |
| Qwen3.8 DSpark with eight draft tokens | `8` | `0` | `[]` |

Use a bind address such as `tcp://*:5557` for the worker's live publisher. Use a reachable worker address when configuring a static Router subscriber. Decode workers that perform a tail drop require their own layout. Use load-based Decode routing for tail-drop rules outside the supported layout.

For HTTP P/D, set `kv_connector: "MooncakeConnectorV1"` in the Ascend workers' KV transfer config and use `--kv-connector mooncake` on Router. Use the [P/D layout](#pd-layout) and omit the optional bootstrap port when Prefill returns the transfer parameters. Router uses the Prefill lookup boundary for Mamba and bounded-replay layouts.

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

Use `--max-model-len 4096` and the engine-selected cache block sizes. The reference V4 Pro checkpoint used `--tensor-parallel-size 4`. Copy the [V4 Flash rank fields](#deepseek-v4-flash) and update three groups.

- Group 2 uses `"tokens_per_state": "128"`.
- Group 3 uses `"block_size": 8` and `"sliding_window": 128`.
- Group 4 uses `"block_size": 4` and `"sliding_window": 8`.

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

To [check reuse](#3-check-cache-reuse) for a hybrid model, use its served name and a repeated prompt long enough to span its cache pages. Check connector compatibility when deploying a model in P/D.

## P/D routing

P/D disaggregation puts prompt processing on Prefill workers and token generation on Decode workers. This example uses NIXL and enables KV-aware routing for Prefill.

1. Start two workers on different GPUs with the same checkpoint, vLLM version, connector and transport. Keep Prefill on HTTP port `8000` or gRPC port `50051`. Move Decode to HTTP port `8001` and, for a gRPC worker, gRPC port `50052`.
2. Set `VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1` on both workers. Set `VLLM_NIXL_SIDE_CHANNEL_PORT=5600` on Prefill and `5601` on Decode. For separate hosts, use reachable addresses.
3. Add `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'` to Prefill and the same option with `"kv_role":"kv_consumer"` to Decode. Enable events and replay on Prefill. Omit `--kv-events-config` on Decode for this setup. For gRPC, pass these engine options after `--`.
4. Create [`pd-layout.json`](#pd-layout) with the Prefill layout, its live and replay addresses, and the compatible Decode URL. Leave Program scheduling disabled for declared P/D layouts.

For text or chat over HTTP, add `--enable-scale-out` to each worker scored by `kv_aware`. For gRPC, export `VLLM_ROUTER_TOKENIZER` with the model directory in the Router terminal.

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

Save this as `pd-layout.json`. Replace the empty `hybrid` object with the value under the Prefill URL in your colocated layout. Match `model` and `block_size` to the Prefill rank's model and `hash_block_size`. List addresses in rank order. Read the allocated replay endpoints from Prefill's `/kv_event_sources` for HTTP or `Control.GetKvEventSources` for gRPC. For gRPC workers, use their gRPC URLs as the Prefill key and Decode URL.

```json
{
  "http://127.0.0.1:8000": {
    "endpoints": ["tcp://127.0.0.1:5557"],
    "replay_endpoints": ["<advertised-prefill-replay-endpoint>"],
    "block_size": 16,
    "model": "qwen3",
    "hybrid": {},
    "compatible_decode_urls": ["http://127.0.0.1:8001"]
  }
}
```

## Versions

Use the same vLLM release for the engine and its gRPC frontend. In all versions below, gRPC discovers live and replay addresses through `GetKvEventSources`.

| vLLM release | HTTP event discovery | Layout support | vLLM Ascend pairing |
| --- | --- | --- | --- |
| [0.30.0](https://github.com/vllm-project/vllm/tree/v0.30.0) | Pass a layout and [static addresses](#static-event-endpoints). | Zero hash lookahead. | [0.30.0 stable](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.30.0). |
| [0.31.0](https://github.com/vllm-project/vllm/tree/v0.31.0)–[0.32.0](https://github.com/vllm-project/vllm/tree/v0.32.0) | `/kv_event_sources` | Zero hash lookahead and V4.1 bounded replay. | [0.31.0rc1 wheel](https://pypi.org/project/vllm-ascend/0.31.0rc1/) or [0.32.0rc1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.32.0rc1). See [0.32 Ascend layouts](#ascend-with-vllm-032) for MTP and DSpark. |
| [0.33.0](https://github.com/vllm-project/vllm/tree/v0.33.0) | `/kv_event_sources` | Adds nonzero hash lookahead via `BlockStored.lookahead_token_ids`. Model recipes use this release. | Use an Ascend build compatible with vLLM 0.33. |
| [0.34.0](https://github.com/vllm-project/vllm/tree/v0.34.0) | `/kv_event_sources` | Defaults to 64-token blocks when the backend has no preferred size. Use the actual block size in an explicit layout. | Use an Ascend build compatible with vLLM 0.34. |

## CLI options

| Option | Value format or example | Rust type | Default | Description |
| --- | --- | --- | --- | --- |
| `--kv-hybrid-layout` | `layout.json` | path | unset | Loads a colocated full- or hybrid-attention layout keyed by worker URL. |
| `--kv-event-source` | `WORKER_URL=ENDPOINT[,ENDPOINT...]` | string, repeatable | `[]` | Supplies live ZMQ addresses when event discovery is unavailable. |
| `--kv-event-replay-source` | `WORKER_URL=ENDPOINT[,ENDPOINT...]` | string, repeatable | `[]` | Supplies replay addresses for a matching static live source. |
| `--kv-event-block-size` | `16` | `u32` | `16` | Sets the block size in tokens for static colocated sources. |
| `--kv-event-model` | `qwen3` | string | unset | Sets the served model name when using static colocated sources. |
| `--pd-hybrid-config` | `pd-layout.json` | path | unset | Supplies Prefill sources, their layouts and compatible Decode URLs. |
| `--pd-kv-events` | `WORKER_URL=ENDPOINT[,ENDPOINT...]` | string, repeatable | `[]` | Supplies live publishers for 16-token P/D sources without a declared layout. |
| `--router-temperature` | finite value ≥ `0` | `f64` | `0.0` | Samples more broadly with higher values in colocated routing and P/D Prefill. Zero selects the lowest-cost worker. |
| `--kv-penalty-host` | `0.0` to `1.0` | `f64` | `0.8` | Sets the fraction of full cache credit for a hit in host memory. |
| `--kv-penalty-external` | `0.0` to `1.0` | `f64` | `0.6` | Sets the fraction for a hit in external storage. |
| `--kv-penalty-disk` | `0.0` to `1.0` | `f64` | `0.3` | Sets the fraction for a hit on disk. |
| `--kv-aware-session-affinity-ttl-secs` | `1` to `31536000` seconds | `u64` | unset | Enables session affinity until the idle timeout. In P/D it applies to Decode. |

For session affinity, send the same `x-session-id` or `session_params.session_id` on each turn. See the [CLI reference](../cli_reference.md) for launcher availability and other Router options.

## Troubleshooting

- If the engine reports hits and the Router does not, check the layout's model and topic, the hash flags, and whether both event ports are reachable from the Router.
- For `unsupported tail-drop layout`, check the effective lookup rule on the worker. Use `tail_drop_groups: []` if it skips the drop. An EAGLE group ID alone indicates model membership.
- If vLLM reports `Address already in use`, check the live and replay port ranges before starting it again. Inspect the actual event addresses in `/kv_event_sources` for HTTP or `Control.GetKvEventSources` for gRPC.
- When a worker restarts, rerun the worker settings for a fresh topic. If you use a layout file, regenerate it with the new topic and restart the Router to load it. On a Router-only restart, replay recovers the events still in the worker's history.
