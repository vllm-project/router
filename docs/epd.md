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

E, P, and D use the same `LoadBalancingPolicy::select_worker_with_headers`
interface. Encode constructs its own policy instance through the common factory:
its counter, hash ring, and load cache are independent of consumer instances.
All requests, registry/context clones, and EncoderStage instances for the same
router share the Encode instance and its synchronous selection context.

Configuration priority is native CLI `--encode-policy` > JSON `epd.encode_policy`
> the complete main `RouterConfig.policy`. Omitted or null Encode configuration
inherits the main configuration, including its parameters, and creates a separate
instance. P/D overrides and per-model consumer policies do not change that
resolution. `--encode-policy` requires `--epd-config` and uses the existing CLI
policy parameters. JSON is parsed before the override: invalid JSON, unknown
names, and incorrect types remain errors. A known policy that is incompatible
with Encode may be replaced by a compatible CLI override.

| Policy | Encode behavior |
| --- | --- |
| `round_robin` | One ordinary selection per item, using the configured encoder order. |
| `random` | One independent uniform choice per item. |
| `power_of_two` | Compare two candidates using local in-flight HTTP batch load; no new encoder polling or item reservations. |
| `consistent_hash`, `rendezvous_hash` | Request/session affinity within the independent Encode pool. |
| `cache_aware` | Rejected: Encode prefix/cache affinity and initialization are not defined. |
| `kv_aware` | Rejected: KV placement and token transfer costs do not apply to Encode. |

Role compatibility is declared by each policy. Startup validation rejects an
incompatible role before constructing any policy, including the main policy's
cache eviction thread. A direct selector call with an unsupported role returns
`None` before modifying state or metrics; EPD selection failure remains `503`.

The CLI still defaults to main `cache_aware`, while Rust
`RouterConfig::default()` still uses `random`. Thus enabling EPD without an
Encode override on the default CLI now fails at startup. Preserve the consumer
policy and explicitly configure Encode when using either cache-aware or
KV-aware consumers:

```bash
vllm-router --worker-urls http://pd0:8000 --policy cache_aware \
  --epd-config '{"encoder_urls":["http://e0:8000"]}' \
  --encode-policy round_robin
vllm-router --worker-urls http://pd0:8000 --policy kv_aware \
  --epd-config '{"encoder_urls":["http://e0:8000"]}' \
  --encode-policy round_robin
```

The KV-aware example is Regular E+PD; existing static P/D KV-aware restrictions
still apply. With no override, `--policy random` uses independent Random
instances for consumers and Encode. Explicit JSON strings `"round_robin"` and
`"random"` retain their existing meaning and serialization. All common policy
names accept shorthand defaults, and parameterized policies also accept tagged
objects:

```json
{
  "encoder_urls": ["http://e0:8000", "http://e1:8000"],
  "encode_policy": {"type":"consistent_hash", "virtual_nodes":317}
}
```

The resolver and factory entry retain the full `PolicyConfig`. The existing
factories currently ignore `consistent_hash.virtual_nodes` (the ring still uses
160 virtual nodes per worker) and `power_of_two.load_check_interval_secs`
(which does not control polling). These fields round-trip in tagged JSON and
must be positive, but retaining them does not make them operational. Use a
tagged object for custom parameters; parameterized policies serialize as objects.

Startup errors distinguish an explicit `--encode-policy`, explicit
`epd.encode_policy`, and inheritance from `--policy` because Encode is unset.
Rust/JSON validation names `RouterConfig.policy` as the inherited source.
For example, inherited `kv_aware` reports that source, the unsupported Encode
role, its reason, and the suggestion to set `--encode-policy round_robin`.
Explicit incompatible configurations name the override that needs correction.

A shared synchronous selection context keeps a request's RR choices consecutive
when the candidate pool and availability are stable. Every item calls the
ordinary selector, which rechecks health/circuit state; changing availability
can change the sequence. The lock covers only synchronous selections, and
policy replacement uses the same lock order. HTTP calls, response processing,
and batch load guards run after it is released. All item assignments must
succeed before HTTP dispatch; a failed selection does not rewind the counter.
Images assigned to one encoder are combined into one HTTP request; audio/video
items remain singleton calls. Common selection may scan health once per item.

For hashing, provide a stable `X-Session-ID` or another existing session header.
The existing priority is headers (`x-session-id`, `x-user-id`, `x-tenant-id`,
`x-correlation-id`, `x-request-id`, `x-trace-id`) before body session/user fields,
then request-content fallback. Per-request IDs do not provide cross-request
session affinity. Without a matching header, Encode serializes and extracts or
hashes the original body once, outside the selection lock. Each item receives
only the prepared canonical key through internal policy data, with no repeated
body scan and no changed key prefix. The internal key cannot be imported from
an HTTP header and is never forwarded to encoders or consumers. The ordinary
consumer key-extraction path is unchanged.

With a stable pool and availability, all items in one hash request have the
same key and select the same encoder, producing one image batch. A health or
registry change may redirect selection. This is request/session affinity;
media content cache affinity is outside this implementation. Canonical key
preparation is at most one body scan/hash, while each item's hash policy still
computes its ring lookup or worker scores. Long session/user keys still cost
proportionally to their key length; one body serialization can still use a
large allocation. The consumer's own selection cost is separate.

### Rust API migration

This revision changes public Rust source APIs. Replace
`Some(EncodePolicy::Random)` with `Some(PolicyConfig::Random)` (likewise for RR);
`EncodePolicy` is removed and `EpdConfig.encode_policy` is now
`Option<PolicyConfig>`. Existing JSON string compatibility does not preserve the
old Rust type. For example:

```rust
use vllm_router_rs::config::{EpdConfig, PolicyConfig};
let epd = EpdConfig {
    encoder_urls: vec!["http://encoder:8000".into()],
    encode_policy: Some(PolicyConfig::Random),
    consumer_zmq_addrs: Default::default(),
};
let main = PolicyConfig::RoundRobin;
let effective: &PolicyConfig = epd.effective_encode_policy(&main);
```

`effective_encode_policy()` becomes `effective_encode_policy(&main_policy)` and
returns a configuration reference. `PolicyRegistry::set_encode_policy` takes
`&PolicyConfig` and returns `Result<(), PolicyRoleError>`;
`get_encode_policy` returns `Option<Arc<dyn LoadBalancingPolicy>>` with no
consumer fallback. Non-EPD registries leave that slot empty.
`LoadBalancingPolicy::select_encode_batch` and the RR/random overrides are
removed; external callers and trait implementations must migrate to repeated
`select_worker_with_headers` calls. To keep a request's sequence together, use
the registry's shared `get_encode_selection_context().with_selection(...)`,
read the current policy inside that closure, and keep HTTP work outside it.
Do not call `select_and_reserve` inside the closure: it would relock the context
and create per-item reservations. No version bump is included in this Draft.

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
engine's GPU queue. RR/random selection does not balance by this load; power-of-two reads it.
Item selection does not increment Encode processed/load or existing consumer
processed/policy metrics. HTTP batch execution owns that accounting.

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
Text-only and existing embeds inputs do not call the encoders. Mooncake consumers
with rank-qualified URLs reject caller-provided nonempty `ec_transfer_params`
with HTTP 400: those handles may belong to a different consumer. Send raw media
through the router instead. Unranked DP=1 and address-map-free NIXL/Example inputs
retain their existing behavior. PD responses,
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
- Regular E+PD or static vLLM E/P/D mode, including consumer DP>1, no IGW.
- HTTP(S) backends only; gRPC encoder and consumer URLs are rejected at startup.
- Encoder TP/PP constraints remain backend-specific and unchanged.
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

## Data-parallel consumers

E and P/PD instance counts are independent. For example, 2E+1PD or 4E+2PD
can use different numbers of encoder and worker URLs with DP left at 1/unset.

For combined PD, set `--intra-node-data-parallel-size` to the server's DP size.
For separate P/D, each endpoint's DP size is discovered from `/metrics`; the flag
is only a fallback when metrics are unavailable. The router
selects a replica before encoding and pins the HTTP request to that replica
using `X-data-parallel-rank`. Encoders remain separate instances; this flag does
not expand encoder URLs or override the EC backend's producer constraints.

For Mooncake PUSH, provide one control address per replica using `@rank` keys:

```bash
target/release/vllm-router --host 0.0.0.0 --port 8000 \
  --worker-urls http://pd0:8000 --intra-node-data-parallel-size 2 \
  --epd-config '{
    "encoder_urls": ["http://e0:8000", "http://e1:8000"],
    "consumer_zmq_addrs": {
      "http://pd0:8000@0": "tcp://pd0:14579",
      "http://pd0:8000@1": "tcp://pd0:14580"
    }
  }'
```

Mooncake assigns replica `r` the control port `ec_port + r * TP`; list each
replica's first TP shard. The example above assumes TP=1. With separate P/D,
the map references P replicas, not D, including all discovered ranks when P
endpoints have different DP sizes. With DP>1, URLs explicitly ending in `@rank`
remain pinned to that replica rather than being expanded again. Startup and
worker-add operations reject ranks outside the endpoint's DP size, including
separate P and D endpoints. The map is checked against the resolved workers
during router construction, after DP discovery. `ConfigValidator::validate()`
checks configuration syntax, not map coverage. Missing rank mappings fail before
encoder dispatch and identify the missing URL. NIXL PULL and
ECExampleConnector need no address map. DP=1 keeps the unsuffixed keys shown above.
The shared P/D discovery path also rejects invalid candidate snapshots instead
of routing through a partially filtered pool.

```bash
cargo test --features cpu,ready --test test_pd_routing test_epd_handoff
```

Focused tests:

```bash
cargo test --features cpu,ready --lib routers::http::epd::tests
```
