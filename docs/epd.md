# Encoder + PD / E/P/D routing (experimental)

The native Rust router can orchestrate remote encoders before forwarding a chat
request to a combined prefill/decode worker or separate P and D workers. Embeddings move through vLLM's EC
connector; the router only handles JSON placeholder metadata and transfer handles.
No PyTorch dependency or tensor deserialization is needed in the router.

Use vLLM with JSON metadata inputs (vllm-project/vllm#56090). Configure E workers
as EC producers and PD workers as matching EC consumers, as for the vLLM encoder
disaggregation example. The example connector needs a shared filesystem; NIXL
and Mooncake need their usual backend dependencies and network connectivity.

```bash
cargo build --release --bin vllm-router
target/release/vllm-router \
  --host 0.0.0.0 --port 8000 \
  --worker-urls http://pd0:8000 http://pd1:8000 \
  --policy random \
  --epd-config '{
    "encoder_urls": ["http://e0:8000", "http://e1:8000"],
    "consumer_zmq_addrs": {
      "http://pd0:8000": "tcp://pd0:14579",
      "http://pd1:8000": "tcp://pd1:14579"
    }
  }'
```

`consumer_zmq_addrs` is for Mooncake PUSH and must map each configured PD HTTP URL
to that worker's configured EC control address. For NIXL PULL or the shared-file
example connector, omit this map. The PD worker is selected before encoding so
the producer and the HTTP request use the same destination. Encoder requests
are distributed round-robin per media item.

Send ordinary OpenAI chat requests to `/v1/chat/completions`. Each raw media item
is sent to an encoder. Published metadata replaces that item with an embeds
reference; missing metadata preserves the original media for vLLM's fallback.
Repeated images keep their original positions and independent transfer IDs.
Other request fields, including sampling options, pass through unchanged.
Text-only and existing embeds inputs do not call the encoders. PD responses,
including SSE streams, use the existing transparent forwarding path.

Encoder failures are returned without submitting a partially prepared PD
request. There is no automatic replay of EPD requests in this prototype.
Abandoned Mooncake pushes receive best-effort cancellation over their existing
ZMQ control channel; backend expiration remains the fallback if cancellation
cannot reach the consumer. After a successful PD response header, the consumer
owns request completion/abort cleanup.

## Scope

- Native binary CLI only; the Python launcher does not expose `--epd-config` yet.
- Regular E+PD or static vLLM E/P/D mode, DP=1, no IGW.
- Keep the existing backend's TP/PP constraints; this does not add parallelism support.
- Static E URLs; E health-aware routing and dynamic consumer discovery are not added.
- JSON metadata only, not legacy tensor-base64 metadata from old E servers.
- No early metadata publication, encoder batching changes, or engine/scheduler changes.
- Image E2E validation is performed separately; do not infer audio/video coverage
  or production readiness from the metadata rewrite code alone.

## Separate prefill and decode

Add `--epd-config` to the existing static P/D router. Its EC control address map
must reference **P**, not D. P is an EC consumer and KV producer; D is only a KV
consumer. Both accept JSON placeholder metadata (`--enable-mm-embeds`).
For Qwen3.5 with current vLLM NIXL KV transfer, P and D also require
`VLLM_SSM_CONV_STATE_LAYOUT=DS` for Mamba state transfer.

```bash
target/release/vllm-router --host 0.0.0.0 --port 8000 \
  --vllm-pd-disaggregation --kv-connector nixl \
  --prefill http://p0:8000 --prefill http://p1:8000 \
  --decode http://d0:8000 --decode http://d1:8000 --policy random \
  --epd-config '{
    "encoder_urls": ["http://e0:8000", "http://e1:8000"],
    "consumer_zmq_addrs": {
      "http://p0:8000": "tcp://p0:14579",
      "http://p1:8000": "tcp://p1:14579"
    }
  }'
```

The router selects P/D, prepares E metadata for P, then uses the existing KV
handoff. D receives the same placeholder layout with KV transfer parameters,
without P's EC handles. P errors or missing NIXL KV handles stop the pipeline.
Dynamic P/D discovery is not supported with EPD yet.

```bash
cargo test --test test_pd_routing test_epd_handoff
```

Focused tests:

```bash
cargo test --lib routers::http::epd::tests
```
