# Hybrid KV-aware routing: implementation handoff

Status: **draft, not production-ready**. The design is in the [English RFC](hybrid-kv-aware-rfc.md) and [Chinese RFC](hybrid-kv-aware-rfc.zh.md). The implementation uses existing upstream vLLM events; it does not require a vLLM patch.

## Implemented

- Router-managed layouts describe all required groups and per-rank topics. Replay and source checks gate the multi-group index; missing checkpoint hashes, rank mismatches, gaps and layout mismatches must not create hybrid credit.
- Regular HTTP routing accepts `--kv-hybrid-layout`. Experimental P/D routing accepts `--pd-hybrid-config`, binds an index to each configured P, and restricts P/D admission to operator-declared pairs. P/D Program scheduling is rejected for this configuration.
- An H20 test with upstream main Python code, vLLM 0.33 binary dependencies and a reduced Qwen4Exp dummy model observed 0 then 1,920 cached tokens on two identical 2,048-token requests; Router estimated 0 then 1,920. This is one rank and one worker, not a production workload.
- Hybrid P/D unit tests cover pair filtering, replay and publisher-topic invalidation. An H20 1P1D test with NIXL 1.5.2 and an intentionally incorrect 320-token Router layout transferred 4,505,600 bytes; the engine reported 2,016 cached tokens for the repeated request and Router granted zero credit. With the matching 288-token layout, both Router and the engine reported 2,016 cached tokens and NIXL transferred 4,505,600 bytes. This used a reduced Qwen4Exp model with dummy weights, upstream main Python code and vLLM 0.33 binary dependencies.

### This revision

- **Group-complete credit.** Credit is the boundary that every declared group
  can restore, so one group with a non-empty hash namespace cannot carry the
  prefix alone. A checkpoint group that publishes an empty hash list for a
  reused chunk keeps the stream valid but contributes no residency. Sliding
  groups only have to cover the window that ends at the boundary, and each
  group's own block size and checkpoint alignment are applied before the
  boundary is rounded to the hash block.
- **Layout replay window.** `prefix_replay_tokens` is a property of the whole
  layout, not of one group. All bounded-replay sliding/MLA groups must declare
  the same window or the rank is rejected; groups that are not prefix-cacheable
  are excluded from the index but still set the window. Credited tokens are
  `align_down(boundary - replay, hash_block)`, matching the engine's own rewind.
- **Per-rank state.** The provider keeps one index per admitted `dp_rank` and a
  lookup only reads the rank that will serve the request, so another rank's
  events, ownership or history can never answer for it. Admission is per rank:
  a rank with a missing capability, another model or an invalid layout is
  skipped with a warning and gets no subscriber (never legacy group-0 credit),
  while its siblings stay routable. This stops normal rank differences from
  being reported as a deployment-wide layout inconsistency.
- **Advisory affinity.** Requests whose exact tokens are unavailable (tools,
  reasoning, a backend that cannot render them) use a separate digest namespace.
  The advisory branch can follow and create multi-turn affinity, and a later
  tokenized turn may follow an advisory binding, but an advisory binding never
  reduces estimated prefill work and is never treated as a cache hit.
- **Discovery is not a prerequisite.** KV event sources reach the router through
  the operator's declaration (`--kv-event-source`, `--kv-event-replay-source`,
  and `--pd-kv-event-source` for P/D), through the gRPC Control RPC
  `GetKvEventSources`, or through the HTTP `/kv_event_sources` route that
  **upstream vLLM added in v0.31.0** (`entrypoints/serve/instrumentator/basic.py`;
  absent in v0.29.0 and v0.30.0, confirmed on the v0.29.0 install as a live 404,
  and answered `200` on the v0.33.0 install). On engines without the route the
  probe result is recorded once per worker, logged at `debug`, and never
  re-probed, while a transport failure still warns and retries. A declared
  static layout stays authoritative either way. Removing a worker clears the
  verdict, so a restart (or `--vllm-discovery-address` re-registration) probes
  again; enabling the extension on a worker that is never removed is not
  detected until it is.
- **P/D handoff checks.** Before the decode leg is dispatched, the decode
  transfer parameters must exist, echo the request's `transfer_id`, and carry a
  connector identity for the chosen engine. A rejected handoff answers `502`
  without dispatching to decode, so the decode engine never starts cold work
  for a transfer that did not happen. A decode failure after a successful
  prefill handoff is counted separately
  (`vllm_router_pd_handoff_failures_total`), logged with both endpoints, and
  drops the cached NIXL push identity so a retry performs a fresh handshake
  instead of reusing stale transfer state.

## Live checks on the H20 cluster

Run on `h20` with vLLM 0.29.0, Qwen3.5-4B, `--data-parallel-size 2`,
`--block-size 528`, `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0`,
`--prefix-caching-hash sha256_cbor`, ZMQ KV events, and a router started with
`--worker-urls http://127.0.0.1:8000 --intra-node-data-parallel-size 2`,
one static per-rank source per rank, and a four-group layout (three mamba
groups plus full attention, all block 528, root
`9bd96a48…`, the engine's `sha256_cbor("vllm-none-hash")`).

- Both ranks were admitted, one subscriber per rank was spawned, and requests
  were dispatched with `X-data-parallel-rank` so the engine pinning is real:
  the engine's per-rank series show `prefix_cache_queries_total` on the rank
  the router chose and zero on the other.
- The first request for a 601-token prompt credited 0 and the next five
  credited 528 (5 x 528), leaving `new_prefill_tokens=73`; the engine reported
  `vllm:prefix_cache_hits_total 2640` on the same rank. This is the first
  end-to-end confirmation that router credit equals engine cache hits for a
  hybrid engine.
- **The hash contract must match.** The same engine on its default
  `--prefix-caching-hash` (`sha256`, which pickles the tuple) publishes
  32-byte digests that a `sha256_cbor` layout cannot reproduce, and the router
  credited 0 while the engine still reported 2640 cached tokens. Both are
  32 bytes, so only recomputing the digest distinguishes them; the captured
  batch is checked in as a fixture that must credit nothing.
- vLLM 0.28 cannot exercise this path at all: it publishes 64-bit integer
  block hashes (`block_hashes: [1862206544370044968]`) and has no
  `full_hashes`/`root_hash` contract.
- A per-rank URL in `--worker-urls` without `--intra-node-data-parallel-size`
  is not a DP declaration. `base@rank` parses as URL userinfo, so the router
  forwards to `http://0.0.0.0/...`; configuration now rejects that combination
  instead of failing at request time.

### vLLM 0.33 model matrix

The same check ran on vLLM 0.33.0 (torch 2.14.0+cu130) with
`--prefix-caching-hash sha256_cbor` and `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0`:
without that variable the engine publishes 64-bit integer block hashes, which
no CBOR layout can reproduce, and the router credits 0 while the engine still
reports hits. Each run sends one prompt three times and then one unrelated
prompt; "credit/repeat" is what the router subtracted from `new_prefill_tokens`
on a repeat, "engine hits" is `vllm:prefix_cache_hits_total` at the end.

| model | how it ran | block | group layout (idx: kind) | credit/repeat | engine hits | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B | 2x DP, real weights | 576 | 0-2 mamba, 3 full | 1152 | 3456 (3 repeats) | equal per repeat; the first request credited 0 before events arrived |
| Qwen3.5-35B-A3B | real weights | 1088 | 0-2 mamba, 3 full | 2176 | 4352 (2 repeats) | equal |
| Qwen3.6-35B-A3B | real weights | 1088 | 0-2 mamba, 3 full | 2176 | 4352 (2 repeats) | equal |
| Qwen3.8-27B | real weights | 832 | 0-2 mamba, 3 full | 1664 | 3328 (2 repeats) | equal |
| Qwen4Exp | dummy weights, real config | 192 | 0-2 mamba, 3 scratch ring (no events), 4 compressed MLA | 576 | 1152 (2 repeats) | equal |
| DeepSeek-V4-Flash | real weights, TP=2 | 256 | 0,1,4 sliding-window MLA (64), 2 compressed MLA (256), 3 sliding-window MLA (4) | 0 | 1536 (2 repeats) | rejected, see below |

The four Qwen3.x runs used a layout with declared endpoints. The Qwen4Exp and
V4 runs used a layout with **no** endpoints, so the router discovered the
publishers from the engine itself; two defects had to be fixed for that path to
work at all on 0.33:

- `StaticKvEventSources::validate` rejected the empty endpoint list the
  layout-only configuration produces (`fix(config)`, `cf5de3a`), so the router
  refused to start.
- `/kv_event_sources` on 0.33 answers without the optional `encoding` and
  `schema_version` fields (`fix(kv-index)`, `491a9d9`), so every rank was
  filtered out and the worker fell back to load-only scoring with
  `error=hybrid capability has no event sources`.

**DeepSeek-V4-Flash is not yet representable.** Its engine does serve the same
contract (five groups, prefix reuse of 768 tokens per repeat) but two layout
properties break the single-hash-chain model this PR implements:

- Page sizes are per group (`64, 64, 256, 4, 8`) and three of them are below
  the engine's own 256-token hash block. A layout is only admitted when one
  `hash_block_size` divides every group block, which here forces
  `hash_block_size=4`. With the natural `256` the rank is rejected with
  `invalid group layout`.
- Replay windows differ per group (`128` for the sparse-attention and compressor
  groups, `8` for the indexer group). The layout contract allows one replay
  window shared by every replay group, so declaring the real windows is
  rejected with `unsupported group restore contract`.

Forcing `hash_block_size=4` and zero replay windows admits the rank, and then
the events are refused: every batch is dropped with `error=InvalidBatch`. A ZMQ
capture of the same engine shows why the shape is different, not just the
sizes:

| group | kind | window | block | published hashes for a 1024-token store |
| --- | --- | --- | --- | --- |
| 0, 1 | sliding_window_mla | 128 | 64 | 2 (the resident window), plus one event with 0 hashes |
| 2 | mla_attention | - | 256 | 4 (the whole store) |
| 2 | mla_attention | - | 4 | 1 (a 4-token compressed page, same group index) |
| 3 | sliding_window_mla | 8 | 4 | 2 (the resident window), plus one event with 0 hashes |
| 4 | sliding_window_mla | 128 | 8 | 16 (the resident window) |

So a sliding-window group publishes `window/block` hashes, not one hash per
block of the token range, and the same group index can publish two page sizes
(`256` and `4`). The event carries `parent_block_hash`, which is what a
bounded publication would have to anchor on: the chain does not start at the
root, and this implementation only derives keys from the root. The router
therefore credits 0 while the engine reports 1536 cached tokens: fail-closed,
no false hits, but no reuse either.

**DeepSeek-V4.1-Flash is not runnable on this hardware.** The repository
publishes 510.3 GB of weights in 48 shards (`DeepSeek-V4-Flash`, the model that
does fit, is 159.6 GB in 46 shards), so it needs at least six H20-3e cards
before any KV cache is allocated. vLLM 0.33 does ship the config class
(`vllm.transformers_utils.configs.deepseek_v41.DeepseekV41Config`, model type
`deepseek_v41`), so the layout can be written against it, but no engine-side
evidence is possible on a two-card node.

### Advisory affinity, without a model

Two stub backends answered `/v1/completions` but returned 404 for `/render` and
`/tokenize`, which is the tools/reasoning shape: no exact tokens exist for the
request. With `--policy kv_aware --kv-aware-session-affinity-ttl-secs 300` and
three requests carrying the same `session_id`:

```
outcome="advisory_bind" worker="http://127.0.0.1:9101" input_tokens=0 credited_prefix_tokens=0 used_binding=false
outcome="advisory_hit"  worker="http://127.0.0.1:9101" input_tokens=0 credited_prefix_tokens=0 used_binding=true
outcome="advisory_hit"  worker="http://127.0.0.1:9101" input_tokens=0 credited_prefix_tokens=0 used_binding=true
```

The first turn binds an advisory digest, the next two follow it to the same
worker, and every turn reports zero credited prefix tokens: the affinity is
carried, the cache claim is not. Without
`--kv-aware-session-affinity-ttl-secs` the same requests stay
`outcome="missing_tokens"`, which is the opt-in boundary for session affinity.

### P/D handoff, without a model

One stub prefill and one stub decode in direct-URL mode
(`--vllm-pd-disaggregation --prefill … --decode … --kv-connector nixl`,
no `--vllm-discovery-address`, which would switch the router to pure discovery
and ignore the static URLs). Four requests, with the stub decode reporting how
many times it was called:

| case | prefill reply | client | prefill calls | decode calls | evidence |
| --- | --- | --- | --- | --- | --- |
| `ok-d` | `kv_transfer_params.remote_block_ids` | 200 | 1 | 1 | normal handoff |
| `fail-d` | same as above | 500 | 2 | 2 | `WARN … reason="500"`, `vllm_router_pd_handoff_failures_total` |
| `no-handoff` | no `kv_transfer_params` | 502 | 3 | 2 | `WARN … handoff validation failed before decode dispatch`, reason `missing_decode_transfer_params` |
| `mismatch-id` | echoed `remote_block_ids` | 200 | 4 | 3 | NIXL pull mode has no router-side transfer id to compare |

The second row is the reported DeepSeek failure: prefill really handed off, then
decode failed. The router keeps the request's 500, counts the handoff failure
with both endpoints and the reason, and drops the cached push identity. The
third row is the fail-closed case: decode's call count does not move, so a
missing handoff can never be served as a normal P/D request.

## Remaining work and acceptance criteria

1. **P/D:** Automate the observed matching and mismatched-layout assertions and verify a fully built upstream main with real weights. Enabling P/D changed the Qwen4Exp Mamba/attention block size from 320 to 288 in the test; verify each deployment's effective layout. Compare outputs against a direct reference, and exercise nonmatching pairs and multiple DP ranks. Handoff validation, the pre-dispatch 502 and post-handoff decode failures are covered against stub backends; a real NIXL P/D pair still needs the same four cases.
2. **Lifecycle and unknown layouts:** Integrate trusted deployment generation changes and automatic rebinding. A static topic and periodic HTTP checks cannot establish that a worker has not restarted with the same topic. Ensure unknown hybrid models cannot obtain legacy group-0 credit, while retaining existing full-attention behavior.
3. **Model layouts:** Capture stock-vLLM event and restore behavior for Qwen3.5/3.6/3.8/4 and DeepSeek V4/V4.1. The sliding-window/bounded-replay path now models one agreed layout window and rejects disagreeing groups, but the DeepSeek V4 compressed/shared-state layout still needs captured events from a real deployment before it can be enabled. Do not enable unknown groups or infer completeness from observed events. Concretely, enabling DeepSeek-V4-Flash needs three changes beyond a layout file: a hash chain per group page size (its groups publish 4, 8, 64 and 256-token pages while the engine hashes at 256), acceptance of a bounded publication that anchors on `parent_block_hash` instead of the root (a sliding-window group publishes only `window/block` hashes, and 0-hash stores occur), and tolerance for one group index publishing two page sizes. Qwen4Exp only needs the layout that is now recorded: three `mamba` groups, one compressed-MLA group, and the scratch ring excluded.
4. **Advisory affinity:** The advisory namespace binds, follows and never credits on stub backends. Bounding history by worker/rank/generation and tenant still needs tests, and the tools/reasoning entry points need documenting per framework.
5. **Production checks:** Test feature-off and existing full-attention/gRPC paths, worker restart and replay truncation, DP isolation, recovery, races, capacity, multi-worker routing, actual engine hit counts and output equivalence. Run full Rust tests, strict Clippy and performance tests; document the supported version/model/connector matrix. Also reconcile the startup message "Using hybrid mode" with discovery mode actually ignoring static P/D URLs.

The currently checked-in static layouts are operator assertions. They are not automatic verification of model weights, complete groups or publisher generation. Do not deploy an unverified combination as a production feature.
