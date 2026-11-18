# KV-aware prefix routing

`kv_aware` routes using worker KV events and current load. The examples use the [Rust binary](../cli_reference.md). Clients send OpenAI HTTP requests to the Router.

| Deployment | HTTP workers | gRPC workers |
| --- | --- | --- |
| Colocated prefill and decode | Chat and Completion | Chat and flat token-ID Completion |
| Separate prefill and decode (P/D) | Chat and Completion | Chat |

Use one worker transport per pool. For gRPC, export `VLLM_ROUTER_TOKENIZER` with the model directory used by the workers. See the [gRPC worker setup](../backend/grpc.md).

## 1. Start workers

### Full attention models

For a local worker, set the model path and the HTTP, publisher and replay ports. Use a fresh event topic on every launch.

```bash
MODEL=/models/Qwen3-4B-Instruct-2507
SERVED=qwen3
HTTP=8000
EVENTS=$(jq -nc --arg topic "kv-$(python3 -c 'import uuid; print(uuid.uuid4().hex)')" \
  '{enable_kv_cache_events:true,publisher:"zmq",endpoint:"tcp://127.0.0.1:5557",
    replay_endpoint:"tcp://127.0.0.1:5558",topic:$topic,buffer_steps:30000}')

VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" --host 127.0.0.1 --port "$HTTP" \
  --served-model-name "$SERVED" --enable-prefix-caching --kv-events-config "$EVENTS"
```

For a gRPC worker, replace the serve command.

```bash
"${VLLM_RS:-vllm-rs}" serve "$MODEL" --host 127.0.0.1 --port "$HTTP" --grpc-port 50051 -- \
  --served-model-name "$SERVED" --enable-prefix-caching --kv-events-config "$EVENTS"
```

Use distinct ports and GPU placement for additional workers. Across hosts, bind addresses the Router can reach.

### Hybrid attention models

Use the [hybrid model recipes](hybrid_kv_recipes.md) to start the engine and export its layout. HTTP `/kv_event_sources` and gRPC `GetKvEventSources` discover publisher and replay endpoints. The layout supplies each rank's hash root and all cache groups, including groups that emit no events.

## 2. Start the Router

### Colocated prefill and decode

Set `WORKERS` to the worker URLs from the previous step.

```bash
WORKERS=(http://127.0.0.1:8000 http://127.0.0.1:8001)
# For gRPC:
# WORKERS=(grpc://127.0.0.1:50051 grpc://127.0.0.1:50052)
vllm-router --host 127.0.0.1 --port 19000 --policy kv_aware --worker-urls "${WORKERS[@]}"
```

Use one served model per regular pool. For separate model pools, enable `--enable-igw` and register each worker through `POST /workers` with its `model_id`.

- **Hybrid workers.** Add `--kv-hybrid-layout layout.json`. [Build the file from exported rank layouts](hybrid_kv_recipes.md#colocated-layout).
- **Static event sources.** If discovery is unavailable, supply `--kv-event-source 'WORKER_URL=tcp://HOST:5557'` and `--kv-event-replay-source 'WORKER_URL=tcp://HOST:5558'` per worker, plus `--kv-event-block-size` and `--kv-event-model`. For a hybrid worker, the layout supplies the topic.

### P/D routing

Start separate engines with the same model and compatible transfer settings. For NIXL, pass `--kv-transfer-config` with `"kv_connector":"NixlConnector"` and set `kv_role` to `kv_producer` on prefill and `kv_consumer` on decode. Give each engine a reachable `VLLM_NIXL_SIDE_CHANNEL_HOST` and its own `VLLM_NIXL_SIDE_CHANNEL_PORT`. Enable KV events and replay on prefill. See the [complete NIXL example](hybrid_kv_recipes.md#hybrid-pd-example).

```bash
PREFILL=${PREFILL:-http://127.0.0.1:8000}
DECODE=${DECODE:-http://127.0.0.1:8001}
# For gRPC:
# PREFILL=grpc://127.0.0.1:50051
# DECODE=grpc://127.0.0.1:50052
vllm-router --port 19000 --vllm-pd-disaggregation --prefill "$PREFILL" --decode "$DECODE" \
  --prefill-policy kv_aware --decode-policy kv_aware --kv-connector nixl
```

Repeat `--prefill` and `--decode` for multiple workers. Match `--kv-connector` to the engines. The [gRPC P/D API](../backend/grpc.md#pd-disaggregation) requires vLLM 0.30.0 or later.

- **Full attention without event discovery.** Add `--pd-kv-events 'PREFILL_URL=tcp://HOST:5557'` for 16-token blocks.
- **Hybrid attention.** Add `--pd-hybrid-config pd-layout.json` with [explicit compatible pairs](hybrid_kv_recipes.md#hybrid-pd-example). Disable Program scheduling.

### Optional CLI settings

| Setting | Effect |
| --- | --- |
| `--intra-node-data-parallel-size` | Rank count per worker URL. Default `1`. |
| `--router-temperature` | Selection temperature for colocated routing and P/D prefill. Default `0` selects the lowest cost. |
| `--kv-penalty-host`, `--kv-penalty-external`, `--kv-penalty-disk` | Cache credit by storage tier. Defaults `0.8`, `0.6`, `0.3`. |
| `--kv-aware-session-affinity-ttl-secs` | Optional session affinity. Applies to decode only in P/D. |

See [all routing options](../cli_reference.md#routing-policies).

### Reporting and restarts

Use incremental reporting on monitored workers. Explicit `vllm_xargs.kv_cache_report_mode=full` is rejected. Hybrid cache recovery requires ordered replay from an empty cache. After a hybrid worker restarts or its layout changes, export a fresh layout and restart the Router with the updated file.

## Verify reuse

Put a prompt spanning several cache pages in `long-prompt.txt` and set the served model name. Send the request twice, allowing cache events from the first request to arrive.

```bash
curl -fsS http://127.0.0.1:19000/health
jq -n --arg model qwen3 --rawfile prompt long-prompt.txt \
  '{model:$model,messages:[{role:"user",content:$prompt}],max_tokens:1}' > request.json
curl -fsS http://127.0.0.1:19000/v1/chat/completions -H 'Content-Type: application/json' --data-binary @request.json
curl -fsS http://127.0.0.1:29000/metrics | grep vllm_router_kv_placement_lookups_total
```

A rise in `result="hit"` shows Router-observed reuse. Check `usage.prompt_tokens_details.cached_tokens` for actual engine reuse when the backend exposes it. If cache credit stays zero, check the served model, rank, topic and replay endpoint. For HTTP text input, also check worker rendering or `/tokenize` for supported simple prompts. The Router falls back to load-based routing when cache evidence is unavailable.
