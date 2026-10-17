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
    "encode_policy": "round_robin",
    "consumer_zmq_addrs": {
      "http://pd0:8000": "tcp://pd0:14579",
      "http://pd1:8000": "tcp://pd1:14579"
    }
  }'
```

`consumer_zmq_addrs` is for Mooncake PUSH and must map each configured PD HTTP URL
to that worker's configured EC control address. For NIXL PULL or the shared-file
example connector, omit this map. The PD worker is selected before encoding so
the producer and the HTTP request use the same destination. The Encode policy
assigns each media item to an available encoder; images landing on the same
encoder are batched into a single request, like the Python EPD proxy.

## Encode policy, health, and load

`encode_policy` in `--epd-config` accepts `round_robin` (the default when omitted
or null) or `random`. The native CLI can override the JSON setting with
`--encode-policy random`; this flag requires `--epd-config`. Unsupported Encode
policies are rejected at startup. Rust callers set `EpdConfig.encode_policy`
to `Some(EncodePolicy::RoundRobin)` or `Some(EncodePolicy::Random)`;
`None` uses round-robin.

The Encode policy is shared across requests and independent of `--policy`,
`--prefill-policy`, `--decode-policy`, and per-model consumer policies. For
example, the command above uses random consumer selection and round-robin
Encode selection. Round-robin reserves the media assignments for one request
together, following the configured encoder order among available workers.
Random assigns each media item uniformly among available workers. Both retain
image batching by encoder; audio and video items remain singleton subrequests.

Static encoders are registered at startup with the `Encode` role. They use the
router's health-check endpoint (default `/health`), timeout, interval, and
success/failure thresholds, together with its circuit-breaker configuration.
Selection requires a healthy worker whose circuit breaker permits execution.
If no encoder is available, a raw-media request returns `503` before consumer
submission. Text-only and existing-embeds requests still bypass Encode.
Encode workers appear in management listings but stay outside inference
consumer pools and cannot satisfy consumer readiness on their own.

Encode `load` counts in-flight encoder HTTP batches. A batch of several images
adds one, and the count stays held until the response body/JSON is consumed or
the attempt fails. Completion, failure, or dropping the encoder future releases
that batch's load once; periodic load resets preserve active Encode batches.
This counter measures router-side HTTP work, not media-item count or the
engine's GPU queue. RR/random selection does not balance by this load.

Encode `processed_requests` counts completed HTTP batch attempts, including
HTTP, transport, and response-JSON failures. Dropping an unfinished future
releases load without adding a completion or a circuit-breaker outcome.
Successful JSON responses and HTTP 4xx count as reachable-worker outcomes;
transport errors, HTTP 5xx, and unreadable JSON count as failed outcomes.

## Request and metadata behavior

Send ordinary OpenAI chat requests to `/v1/chat/completions`. Each raw media item
is sent to an encoder. Published metadata replaces that item with an embeds
reference; when an encoder reports no metadata, the original media is kept with
its content-derived uuid pinned and its transfer is forwarded, so the consumer
re-preprocesses into the same mm_hash and can still reuse the published
embedding. Repeated images keep their original positions and independent
transfer IDs.

Encoder responses use the existing JSON metadata contract. When `item_indices`
is present it must be a batch-local integer array with in-range, non-repeated
positions; malformed indices or non-numeric metadata fail the request with
`502` before the consumer is called. Three missing-data cases remain valid
fallbacks: no EC report, no entry for an item, or an entry without non-empty
placeholder metadata. In each case the router keeps the raw media with its
content-derived UUID and forwards the reported transfer handle when one exists.
Connector-specific fields under `ec_transfer_params` remain opaque and are
passed through unchanged.
Other request fields, including sampling options, pass through unchanged.
Text-only and existing embeds inputs do not call the encoders. PD responses,
including SSE streams, use the existing transparent forwarding path.

The native Rust configuration supports static HTTP encoders with DP=1; Python
launcher, dynamic discovery, DP>1, IGW, gRPC encoder transport, full
cancellation, and real GPU performance are separate follow-up work or
validation environments.

Encoder failures are returned without submitting a partially prepared PD
request. There is no automatic replay of EPD requests in this prototype.
Abandoned Mooncake pushes receive best-effort cancellation over their existing
ZMQ control channel; backend expiration remains the fallback if cancellation
cannot reach the consumer. After a successful PD response header, the consumer
owns request completion/abort cleanup.

## Scope

- Native binary CLI only; the Python launcher does not expose `--epd-config` yet.
- Regular E+PD or static vLLM E/P/D mode, DP=1, no IGW.
- HTTP(S) backends only; gRPC encoder and consumer URLs are rejected at startup.
- Keep the existing backend's TP/PP constraints; this does not add parallelism support.
- Static E URLs with health- and circuit-breaker-aware selection. Dynamic
  Encode/consumer discovery is not supported. The discovery service registry
  has no encoder role, and vLLM does not publish EC control addresses, so
  Mooncake PUSH cannot resolve a newly registered consumer. Dynamic discovery
  for NIXL PULL or the example connector is a smaller follow-up.
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
