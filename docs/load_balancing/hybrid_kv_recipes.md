# Hybrid model recipes for KV-aware routing

These examples use matching vLLM 0.33 builds for the selected checkpoints. Set the model paths and tensor-parallel sizes for your hardware. The launcher needs Bash, jq and Python 3.

## 1. Prepare the launcher

Save the exporter as `kv-layout-export/sitecustomize.py`. It writes one layout file per data-parallel rank at engine startup. An equivalent export from your engine is also sufficient.

<details>
<summary>Layout exporter for vLLM 0.33</summary>

```python
import os
from pathlib import Path
import json

output = os.environ.get("KV_LAYOUT_OUTPUT")
if output:
    from vllm import envs
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.engine.core import EngineCore

    kinds = {
        "FullAttentionSpec": "full_attention",
        "MambaSpec": "mamba",
        "MLAAttentionSpec": "mla_attention",
        "SlidingWindowSpec": "sliding_window",
        "SlidingWindowMLASpec": "sliding_window_mla",
        "CircularBufferSpec": "scratch",
    }
    init = EngineCore.__init__

    def init_and_export(self, *args, **kwargs):
        init(self, *args, **kwargs)
        config = self.vllm_config
        scheduler = self.scheduler
        manager = scheduler.kv_cache_manager
        groups = []
        for group_idx, group in enumerate(manager.kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            spec_type = type(spec).__name__
            ratio = getattr(spec, "tokens_per_state", None)
            role = getattr(spec, "cache_role", None)
            groups.append({
                "group_idx": group_idx,
                "spec_type": spec_type,
                "kind": kinds[spec_type],
                "block_size": spec.block_size,
                "prefix_cacheable": spec.prefix_cacheable,
                "prefix_replay_tokens": spec.prefix_replay_tokens,
                "sliding_window": getattr(spec, "sliding_window", None),
                "mamba_cache_mode": getattr(spec, "mamba_cache_mode", None),
                "prefill_checkpoint_alignment": getattr(spec, "prefill_checkpoint_alignment", None),
                "tokens_per_state": str(ratio) if ratio is not None else None,
                "cache_role": getattr(role, "value", role),
            })

        rank = config.parallel_config.data_parallel_rank
        layout = {
            "rank": rank,
            "model": os.environ["KV_LAYOUT_MODEL"],
            "weight_version": self._weight_version,
            "publisher_topic": config.kv_events_config.topic,
            "instance_topic": True,
            "hash_algorithm": config.cache_config.prefix_caching_hash_algo,
            "full_hashes": not envs.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES,
            "root_hash": bytes(kv_cache_utils.NONE_HASH).hex(),
            "hash_block_size": scheduler.hash_block_size,
            "partial_hash_hits": manager.coordinator.enable_partial_hash_hits,
            "hash_lookahead_tokens": scheduler.num_hash_lookahead_tokens,
            "speculative_tokens": scheduler.num_spec_tokens,
            "prefix_caching": config.cache_config.enable_prefix_caching,
            "declared_groups": len(groups),
            "groups": groups,
        }
        path = Path(output) / f"rank-{rank}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    EngineCore.__init__ = init_and_export
```

</details>

Save the launcher as `serve-hybrid.sh` next to `kv-layout-export/`, then run `source ./serve-hybrid.sh`. Rank files are written under `layouts/<served-name>-<HTTP-port>/`.

```bash
serve_hybrid() {
  local checkpoint=$1 model=$2 tp=$3 http=$4 pub=$5 replay=$6
  shift 6
  local topic events
  topic="kv-$http-$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
  events=$(jq -nc --arg pub "tcp://127.0.0.1:$pub" --arg replay "tcp://127.0.0.1:$replay" \
    --arg topic "$topic" '{enable_kv_cache_events:true,publisher:"zmq",
      endpoint:$pub,replay_endpoint:$replay,topic:$topic,buffer_steps:30000}') || return
  local -a engine=(vllm serve "$checkpoint" --host 127.0.0.1 --port "$http")
  if [[ -n ${GRPC_PORT:-} ]]; then
    engine=("${VLLM_RS:-vllm-rs}" serve "$checkpoint" --host 127.0.0.1 --port "$http" --grpc-port "$GRPC_PORT" --)
  fi
  local output="$PWD/layouts/$model-$http"
  mkdir -p "$output" || return
  PYTHONPATH="$PWD/kv-layout-export${PYTHONPATH:+:$PYTHONPATH}" \
  KV_LAYOUT_OUTPUT="$output" KV_LAYOUT_MODEL="$model" \
  VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 VLLM_SERVER_DEV_MODE=1 \
    "${engine[@]}" --served-model-name "$model" --tensor-parallel-size "$tp" \
      --max-model-len 4096 --enable-prefix-caching --prefix-caching-hash-algo sha256_cbor \
      --kv-events-config "$events" "$@"
}
```

## 2. Start a model worker

Run one example per worker. For gRPC, prefix the command with `GRPC_PORT=50051`. Set `VLLM_RS` if the matching `vllm-rs` binary is outside `PATH`. Give additional workers unique HTTP, gRPC, publisher and replay ports. Remove stale rank files before changing DP size.

### Qwen

```bash
# Qwen3.5
serve_hybrid /models/Qwen3.5-4B qwen35 1 8000 5557 5558 \
  --mamba-cache-mode align --prefix-match-unit 16

# Qwen3.6
serve_hybrid /models/Qwen3.6-27B qwen36 1 8000 5557 5558 \
  --mamba-cache-mode align --prefix-match-unit 16

# Qwen3.8
serve_hybrid /models/Qwen3.8-27B qwen38 1 8000 5557 5558 \
  --mamba-cache-mode align --prefix-match-unit 16

# Qwen4Exp
serve_hybrid /models/Qwen4Exp-FP8 qwen4 1 8000 5557 5558
```

Qwen3.5, 3.6 and 3.8 can use a 16-token hash unit alongside larger Mamba pages. Keep the engine's exported `partial_hash_hits` value in each rank entry. For a Qwen4Exp QSA checkpoint, retain its Mamba, compressed MLA and non-cacheable scratch groups.

### DeepSeek

```bash
# DeepSeek V4 Flash
serve_hybrid /models/DeepSeek-V4-Flash-0731 dsflash 1 8000 5557 5558

# DeepSeek V4 Pro
serve_hybrid /models/DeepSeek-V4-Pro dspro 4 8000 5557 5558

# DeepSeek V4.1 Flash
serve_hybrid /models/DeepSeek-V4.1-Flash-1017 ds41 2 8000 5557 5558
```

Export each checkpoint's actual MLA and sliding-window groups. V4.1 Flash also requires the scratch groups and `prefix_replay_tokens` for non-cacheable sliding groups. The Router deducts that replay window from credited reuse.

## 3. Build the Router configuration

### Colocated layout

This example uses the Qwen3.5 rank files. Use the exact Router worker URL as the key, including `grpc://` for gRPC.

```bash
WORKER_URL=${WORKER_URL:-http://127.0.0.1:8000}
jq -s --arg worker "$WORKER_URL" \
  '{($worker): {schema_version: 1, ranks: sort_by(.rank)}}' \
  layouts/qwen35-8000/rank-*.json > layout.json
```

For multiple workers, generate one JSON file per worker and merge them with `jq -s 'add' worker1.json worker2.json > layout.json`. Include every rank and verify each topic against the worker's event source. For LoRA, add an `adapters` array to the worker entry with each adapter's `name` and exact `--lora-modules` path.

Pass the file with `--kv-hybrid-layout layout.json`. [Start the colocated Router](http_completion_kv.md#colocated-prefill-and-decode).

### Hybrid P/D example

Run these commands on separate GPUs. For gRPC, set `GRPC_PORT=50051` on prefill and `GRPC_PORT=50052` on decode.

```bash
# Prefill
VLLM_SSM_CONV_STATE_LAYOUT=DS VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
serve_hybrid /models/Qwen3.5-4B qwen35 1 8000 5557 5558 \
  --mamba-cache-mode align --prefix-match-unit 16 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'

# Decode
VLLM_SSM_CONV_STATE_LAYOUT=DS VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT=5602 \
serve_hybrid /models/Qwen3.5-4B qwen35 1 8001 5567 5568 \
  --mamba-cache-mode align --prefix-match-unit 16 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
```

Set `PREFILL` and `DECODE` to the Router's worker base URLs without `@rank`. List only decoders that can accept KV from the prefill checkpoint and connector.

```bash
PREFILL=${PREFILL:-http://127.0.0.1:8000}
DECODE=${DECODE:-http://127.0.0.1:8001}
jq -s --arg prefill "$PREFILL" --arg decode "$DECODE" \
  --arg pub tcp://127.0.0.1:5557 --arg replay tcp://127.0.0.1:5558 \
  'sort_by(.rank) | {($prefill): {endpoints: [$pub], replay_endpoints: [$replay],
    block_size: .[0].hash_block_size, model: .[0].model,
    hybrid: {schema_version: 1, ranks: .}, compatible_decode_urls: [$decode]}}' \
  layouts/qwen35-8000/rank-0.json > pd-layout.json
```

For multi-rank prefill, include every rank file and put its event endpoints in both arrays in rank order.

Pass the file with `--pd-hybrid-config pd-layout.json`. [Start the P/D Router](http_completion_kv.md#pd-routing).
