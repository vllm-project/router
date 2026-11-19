# KV-aware routing

If requests often start with the same prompt, such as a shared system message, `kv_aware` prefers a worker that can reuse that prefix and also considers worker load. vLLM publishes cache events so the Router can track which workers have the prefix. When cache information is unavailable, the Router routes by load.

- [Full attention](#colocated-prefill-and-decode) starts one HTTP or gRPC worker. Clients use the Router's OpenAI HTTP API either way.
- [Hybrid attention](#hybrid-attention-models) uses a model-specific cache layout with the same Router commands.
- [Separate prefill and decode](#pd-routing) runs the two stages on different workers.

## Colocated prefill and decode

Each worker handles the entire request. The walkthrough uses one Qwen3-4B-Instruct-2507 worker, vLLM 0.33.0, and the Rust `vllm-router` binary. Check [version compatibility](#versions) if you use another release.

### 1. Prepare the worker settings

In the worker terminal, set a local model directory and publish full 32-byte hashes in KV events. The replay port lets the Router recover recent events after a disconnect. Keep the default incremental reporting.

```bash
MODEL=/models/Qwen3-4B-Instruct-2507
export VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0
unset PYTHONHASHSEED
EVENTS='{"enable_kv_cache_events":true,"endpoint":"tcp://127.0.0.1:5557","replay_endpoint":"tcp://127.0.0.1:5558","topic":"qwen3-start01"}'
```

Save the [full-attention JSON](hybrid_kv_recipes.md#full-attention-layout) as `layout.json`. It supplies the cache block size and hash settings for this worker. Keep its model name, topic and worker URL in sync with the commands below.

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

A worker build with `/kv_routing_capabilities` can supply the layout automatically for one full-attention group without speculative decoding, LoRA or a KV connector. Omit `--kv-hybrid-layout` in that case.

#### gRPC workers

Use [`vllm-rs` from the same vLLM version](../backend/grpc.md#starting-vllm-029).

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

With `curl` and `jq`, send this Qwen3 request once to fill the cache. Run the final `curl` again after the worker's events arrive.

```bash
curl -fsS http://127.0.0.1:19000/health
jq -n '{model:"qwen3",prompt:[range(1000;1129)],max_tokens:1,temperature:0}' > request.json
curl -fsS http://127.0.0.1:19000/v1/completions \
  -H 'Content-Type: application/json' --data-binary @request.json
```

Compare these counters before and after the repeat request. The Router counter shows that routing saw the cached prefix. The worker counter shows actual reuse. For P/D use the Prefill worker's metrics.

```bash
curl -fsS http://127.0.0.1:29000/metrics | grep '^vllm_router_kv_placement_lookups_total{result="hit"}'
curl -fsS http://127.0.0.1:8000/metrics | grep '^vllm:prefix_cache_hits_total'
```

Both worker transports accept `/v1/chat/completions` through the Router. On gRPC, `/v1/completions` accepts one flat token-ID array as shown above.

## Hybrid attention models

Use the [settings for your checkpoint](hybrid_kv_recipes.md#layout-examples-by-checkpoint) and the [hybrid JSON example](hybrid_kv_recipes.md#colocated-layout). Start the worker with those model settings, then use the same HTTP or gRPC Router command. For gRPC, use the engine options after `--` and key the layout by the `grpc://` URL.

To check reuse, repeat a prompt that spans several cache pages for that model. Keep the hash unit, cache groups and any MTP or DSpark settings aligned with the running engine.

## P/D routing

In P/D, Prefill processes the prompt and Decode generates the output. This example uses NIXL and enables KV-aware routing for Prefill.

1. Start two workers on different GPUs with the same checkpoint, vLLM version, connector and transport. Keep Prefill on HTTP port `8000` or gRPC port `50051`. Move Decode to HTTP port `8001` and, for a gRPC worker, gRPC port `50052`.
2. Set `VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1` on both workers. Set `VLLM_NIXL_SIDE_CHANNEL_PORT=5600` on Prefill and `5601` on Decode. For separate hosts, use reachable addresses.
3. Add `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'` to Prefill and the same option with `"kv_role":"kv_consumer"` to Decode. Enable events and replay on Prefill. Omit `--kv-events-config` on Decode for this setup. For gRPC, pass these engine options after `--`.
4. Create [`pd-layout.json`](hybrid_kv_recipes.md#pd-layout) with the Prefill layout, its live and replay addresses, and the compatible Decode URL. Leave Program scheduling disabled for declared P/D layouts.

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

## Versions

The table lists event discovery in standard vLLM releases. The `sha256_cbor` examples use layout JSON on all five versions. Event discovery supplies live and replay addresses, topics and ranks. The layout supplies the hash root and cache groups.

| vLLM release | HTTP event discovery | gRPC event discovery | Events with nonzero hash lookahead | Verification |
| --- | --- | --- | --- | --- |
| [0.30.0](https://github.com/vllm-project/vllm/tree/v0.30.0) | Static endpoints | Automatic | Unavailable | Source inspected |
| [0.31.0](https://github.com/vllm-project/vllm/tree/v0.31.0) | Automatic | Automatic | Unavailable | Source inspected |
| [0.32.0](https://github.com/vllm-project/vllm/tree/v0.32.0) | Automatic | Automatic | Unavailable | Source inspected |
| [0.33.0](https://github.com/vllm-project/vllm/tree/v0.33.0) | Automatic | Automatic | Available | B300 reference version |
| [0.34.0](https://github.com/vllm-project/vllm/tree/v0.34.0) | Automatic | Automatic | Available | Source inspected |

## CLI options

**Worker registration**

- `--worker-urls` and `--policy kv_aware` select a colocated pool. Use one transport per pool.
- `--kv-hybrid-layout` loads a colocated layout for full or hybrid attention.
- `--kv-event-source`, `--kv-event-replay-source`, `--kv-event-model` and `--kv-event-block-size` describe static colocated sources. Repeat endpoint options per worker. For data parallelism, list endpoints in rank order and set `--intra-node-data-parallel-size` (default `1`).

**P/D**

- `--vllm-pd-disaggregation`, `--prefill` and `--decode` declare the stages. Repeat the URL flags for more workers.
- `--prefill-policy` and `--decode-policy` override `--policy` for each stage. Use `kv_aware` on Prefill for prefix reuse. See [Decode load and affinity](README.md#pd-decode-load) for Decode policy behavior.
- `--kv-connector` selects NIXL (default), Mooncake or MoRI-IO. Match the connector on both workers. MoRI-IO requires vLLM ZMQ service discovery.
- `--pd-hybrid-config` supplies the Prefill layout and compatible pairs. `--pd-kv-events` supplies a static publisher for a 16-token P/D source using event-based load accounting.

**Tuning and metrics**

| Flag | Default | Use |
| --- | --- | --- |
| `--router-temperature` | `0` | Select the lowest-cost worker. Higher values distribute requests among candidates in colocated routing and P/D Prefill. |
| `--kv-penalty-host` | `0.8` | Credit for a hit in host memory. |
| `--kv-penalty-external` | `0.6` | Credit for a hit in external storage. |
| `--kv-penalty-disk` | `0.3` | Credit for a hit on disk. |
| `--kv-aware-session-affinity-ttl-secs` | Disabled | Enable an idle timeout for session affinity. Send the same `x-session-id` or `session_params.session_id` on each turn. In P/D this applies to Decode. |
| `--prometheus-port` | `29000` | Router metrics port. |

See the [CLI reference](../cli_reference.md) for values and CLI availability.

## Troubleshooting

- If the engine reports hits and the Router does not, check the layout's model and topic, the hash flags, and whether both event ports are reachable from the Router. Use distinct ports for workers on the same host.
- When a worker restarts, give it a fresh event topic and update the layout. The Router can recover after its own restart when the worker still has the replay history.
