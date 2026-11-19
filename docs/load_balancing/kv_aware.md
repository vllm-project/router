# KV-aware prefix routing

Choose [HTTP](#http-workers) or [gRPC](#grpc-workers) for the worker pool. For both, clients send OpenAI HTTP requests to the Router.

## Versions

Use the Rust `vllm-router` binary. For a gRPC worker, build `vllm-rs` from the same vLLM tag as its Python engine. Keep the vLLM version, model checkpoint, connector and worker transport consistent across both sides of a P/D pair.

| vLLM release | HTTP event discovery | gRPC event discovery | Events with nonzero hash lookahead | Verification |
| --- | --- | --- | --- | --- |
| [0.30.0](https://github.com/vllm-project/vllm/tree/v0.30.0) | Supply both endpoints | Discovered | Unavailable | Source inspected |
| [0.31.0](https://github.com/vllm-project/vllm/tree/v0.31.0) | Discovered | Discovered | Unavailable | Source inspected |
| [0.32.0](https://github.com/vllm-project/vllm/tree/v0.32.0) | Discovered | Discovered | Unavailable | Source inspected |
| [0.33.0](https://github.com/vllm-project/vllm/tree/v0.33.0) | Discovered | Discovered | Available | B300 reference version |
| [0.34.0](https://github.com/vllm-project/vllm/tree/v0.34.0) | Discovered | Discovered | Available | Source inspected |

For `sha256_cbor` on a standard worker, supply the cache layout JSON for every version above. Event discovery provides live and replay addresses, topics and ranks. The layout provides the hash root and cache groups. The [colocated HTTP exception](#http-workers) applies to worker builds that expose both. For P/D, include the prefill endpoints in [`pd-layout.json`](hybrid_kv_recipes.md#pd-layout).

The commands below use vLLM 0.33.0. See the [model recipes](hybrid_kv_recipes.md#layout-examples-by-checkpoint) for checkpoint-specific settings.

## 1. Configure vLLM

These single-GPU examples use Qwen3-4B-Instruct-2507. Set the variables in each terminal and replace the model path. Use the engine's actual [hash seed or root](hybrid_kv_recipes.md#full-attention-layout) if it differs from the example.

```bash
MODEL=/models/Qwen3-4B-Instruct-2507
export VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0
unset PYTHONHASHSEED
EVENTS='{"enable_kv_cache_events":true,"endpoint":"tcp://127.0.0.1:5557","replay_endpoint":"tcp://127.0.0.1:5558","topic":"qwen3-start01"}'
```

Use incremental event reporting, the default. Replay lets the Router catch up on recent events after a disconnect. Use a fresh topic on every worker launch and update the layout JSON to match. For multiple workers, use distinct ports and one transport per pool. Across hosts, use addresses reachable from the Router.

## 2. Start the worker and Router

### Colocated prefill and decode

Choose one of the following setups and save the [full-attention example](hybrid_kv_recipes.md#full-attention-layout) as `layout.json`.

#### HTTP workers

Start vLLM.

```bash
vllm serve "$MODEL" --host 127.0.0.1 --port 8000 --served-model-name qwen3 \
  --enable-prefix-caching --prefix-caching-hash-algo sha256_cbor --block-size 16 \
  --enable-prompt-tokens-details --kv-events-config "$EVENTS"
```

Set the worker key in `layout.json` to `http://127.0.0.1:8000`. In a second terminal, start the Router.

```bash
vllm-router --host 127.0.0.1 --port 19000 --policy kv_aware \
  --worker-urls http://127.0.0.1:8000 --kv-hybrid-layout layout.json
```

For a single full-attention group without speculative decoding, LoRA or a KV connector, omit `--kv-hybrid-layout` if the worker build exposes its resolved contract at `/kv_routing_capabilities`.

#### gRPC workers

Use [a vLLM build with the Rust gRPC service](../backend/grpc.md#starting-vllm-029).

```bash
vllm-rs serve "$MODEL" --host 127.0.0.1 --port 8000 --grpc-port 50051 --served-model-name qwen3 -- \
  --enable-prefix-caching --prefix-caching-hash-algo sha256_cbor \
  --block-size 16 --kv-events-config "$EVENTS"
```

Use `grpc://127.0.0.1:50051` as the worker key in `layout.json`. In a second terminal, point the Router at the worker's gRPC port and use its tokenizer.

```bash
VLLM_ROUTER_TOKENIZER="$MODEL" vllm-router --host 127.0.0.1 --port 19000 \
  --policy kv_aware --worker-urls grpc://127.0.0.1:50051 --kv-hybrid-layout layout.json
```

The Router discovers the event endpoints from the worker. For a build without event discovery, append `--kv-event-source "$WORKER_URL=tcp://127.0.0.1:5557"` and `--kv-event-replay-source "$WORKER_URL=tcp://127.0.0.1:5558"`, where `WORKER_URL` is the worker URL in `layout.json`.

### Hybrid attention models

Use the worker command and `layout.json` from the [hybrid recipes](hybrid_kv_recipes.md). Keep the model name, hash settings, publisher topic, and cache groups aligned with the engine. For gRPC, use the same engine flags after `vllm-rs serve ... --` and key the JSON by the gRPC URL.

### P/D routing

Start one prefill and one decode engine on different GPUs using the same model and worker transport. For NIXL, add these settings to the worker commands above.

| Setting | Prefill | Decode |
| --- | --- | --- |
| Service port | HTTP `8000` or gRPC `50051` | HTTP `8001` or gRPC `50052` |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` | `127.0.0.1` | `127.0.0.1` |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | `5600` | `5601` |
| `--kv-transfer-config` | `'{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'` | `'{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'` |
| `--kv-events-config` | `"$EVENTS"` | Omit |

For gRPC, also move Decode's worker HTTP port to `8001` and export `VLLM_ROUTER_TOKENIZER="$MODEL"` in the Router terminal. Write [`pd-layout.json`](hybrid_kv_recipes.md#pd-layout) with the prefill layout and the compatible decode URL, then start the Router.

```bash
PREFILL=http://127.0.0.1:8000
DECODE=http://127.0.0.1:8001
# For gRPC, use grpc://127.0.0.1:50051 and grpc://127.0.0.1:50052.
vllm-router --port 19000 --vllm-pd-disaggregation --prefill "$PREFILL" --decode "$DECODE" \
  --prefill-policy kv_aware --decode-policy round_robin --kv-connector nixl \
  --pd-hybrid-config pd-layout.json
```

Leave Program scheduling disabled with declared P/D layouts. See the [CLI reference](../cli_reference.md#routing-policies) for additional settings.

## 3. Verify reuse

For the Qwen3 example, send the same token-ID prompt twice. Allow the Router to receive the first request's KV events before sending the second.

```bash
curl -fsS http://127.0.0.1:19000/health
jq -n '{model:"qwen3",prompt:[range(1000;1129)],max_tokens:1,temperature:0}' > request.json
curl -fsS http://127.0.0.1:19000/v1/completions \
  -H 'Content-Type: application/json' --data-binary @request.json
```

Run the request again and check both sides.

- **Router** — The `result="hit"` counter should increase for the warm request.
  ```bash
  curl -fsS http://127.0.0.1:29000/metrics | grep '^vllm_router_kv_placement_lookups_total{result="hit"}'
  ```
- **Worker** — For HTTP, check `usage.prompt_tokens_details.cached_tokens` when available. For either transport, compare the worker's prefix-cache hit counter before and after the warm request. Use the prefill worker in P/D.
  ```bash
  curl -fsS http://127.0.0.1:8000/metrics | grep '^vllm:prefix_cache_hits_total'
  ```

Both transports also support `/v1/chat/completions`. With gRPC, `/v1/completions` requires a flat array of token IDs. If Router lookups stay cold while the engine reports hits, check that the layout model, hash settings, and topic match the worker and that both event endpoints are reachable.
