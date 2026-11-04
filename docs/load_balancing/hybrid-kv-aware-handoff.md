# Hybrid KV-aware routing: implementation handoff

Status: **draft, not production-ready**. The design is in the [English RFC](hybrid-kv-aware-rfc.md) and [Chinese RFC](hybrid-kv-aware-rfc.zh.md). The implementation uses existing upstream vLLM events; it does not require a vLLM patch.

## Implemented

- Router-managed layouts describe all required groups and per-rank topics. Replay and source checks gate the multi-group index; missing checkpoint hashes, rank mismatches, gaps and layout mismatches must not create hybrid credit.
- Regular HTTP routing accepts `--kv-hybrid-layout`. Experimental P/D routing accepts `--pd-hybrid-config`, binds an index to each configured P, and restricts P/D admission to operator-declared pairs. P/D Program scheduling is rejected for this configuration.
- An H20 test with upstream main Python code, vLLM 0.33 binary dependencies and a reduced Qwen4Exp dummy model observed 0 then 1,920 cached tokens on two identical 2,048-token requests; Router estimated 0 then 1,920. This is one rank and one worker, not a production workload.
- Hybrid P/D unit tests cover pair filtering, replay and publisher-topic invalidation. An H20 1P1D test with NIXL 1.5.2 and an intentionally incorrect 320-token Router layout transferred 4,505,600 bytes; the engine reported 2,016 cached tokens for the repeated request and Router granted zero credit. With the matching 288-token layout, both Router and the engine reported 2,016 cached tokens and NIXL transferred 4,505,600 bytes. This used a reduced Qwen4Exp model with dummy weights, upstream main Python code and vLLM 0.33 binary dependencies.

### This revision

- **Group-complete credit.** Credit is the smallest hit any declared group
  reports, so one group with a non-empty hash namespace cannot carry the prefix
  alone, and a group that proves nothing drops the credit to zero. A checkpoint
  group that publishes an empty hash list for a reused chunk keeps the stream
  valid but contributes no residency. Sliding groups cover the window that ends
  at their own hit, each group's own block size and checkpoint alignment are
  applied to its hit, and the smallest hit is then pulled down by the cumulative
  groups' granularity before the replay window is subtracted.
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
on a repeat, "engine hits" is `vllm:prefix_cache_hits_total` at the end. Every
run used a two-card allocation; the partition itself is six nodes of eight
H20-3e (143.8 GB each, 48 cards), so the larger V4 models fit on one node when
a whole node is requested.

| model | how it ran | block | group layout (idx: kind) | credit/repeat | engine hits | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B | real weights, TP1 | 576 | 0-2 mamba, 3 full | 1728 | 3456 (2 repeats) | equal per repeat; the first request credited 0 before events arrived |
| Qwen3.5-35B-A3B | real weights | 1088 | 0-2 mamba, 3 full | 3264 | 6528 (2 repeats) | equal |
| Qwen3.6-35B-A3B | real weights | 1088 | 0-2 mamba, 3 full | 2176 | 4352 (2 repeats) | equal |
| Qwen3.8-27B | real weights | 832 | 0-2 mamba, 3 full | 2496 | 4992 (2 repeats) | equal |
| Qwen4Exp | dummy weights, real config | 192 | 0-2 mamba, 3 scratch ring (no events), 4 compressed MLA | 576 | 1152 (2 repeats) | equal |
| DeepSeek-V4-Flash | real weights, TP=2 | 256 | 0,1,4 sliding-window MLA (64), 2 compressed MLA (256), 3 sliding-window MLA (4) | 768 | 1536 (2 repeats) | equal |
| DeepSeek-V4-Pro | real weights, TP=8 on B300 | 256 | 0,1,3,4 sliding-window MLA (64, 64, 8, 4), 2 compressed MLA (256) | 768 | 768 per repeat | equal |

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

**DeepSeek-V4-Flash needed a different credit rule, not a different
chain.** Its engine serves the same `sha256_cbor` contract and reuses 768 tokens
per repeat, but three properties broke the first implementation:

- Page sizes are per group (`64, 64, 256, 4, 8`) and three are below the
  engine's own 256-token request block. A layout is only admitted when one
  `hash_block_size` divides every group block, which forces
  `hash_block_size=4`; with the natural `256` the rank is rejected with
  `invalid group layout`. Four is the engine's real chain granularity: every
  published hash equals the chain hash at the end of the group's block, which a
  re-computation with vLLM's own `sha256_cbor` confirmed for all five groups.
- Sliding groups publish only the blocks they keep. A 1,024-token store
  publishes 2 hashes for the 64-token groups, 2 for the 8-token-window group and
  16 for the 8-token-block group, next to events with none at all, so the shape
  rule `len(block_hashes) * block_size == len(token_ids)` refused every batch
  with `error=InvalidBatch`.
- One group index publishes two page sizes: group 2 stores its 256-token blocks
  and also 4-token compressor pages.

Accepting the tail shape and ignoring the secondary pages made the events apply.
Credit was still 0, and the calibration runs show why: the router required one
boundary that every group accepts, while the engine measures each group on its
own and keeps the smallest. With 1,024-token prompts the sliding tails only
cover the window that ends at 1,024, so a shared boundary of 768 (what the
256-token group restores) was rejected by the router and credited 0, while the
engine reported 768 per repeat. The prefix-subset runs pin the rule down: after a
1,024-token prompt is cached, a 704-token prefix of it hits 0 on its first
attempt (the sliding tail does not cover 704) and then 512 on its own repeat
(the 256-token group's granularity), so the engine keeps each group's own hit
and does not re-check the window at a shorter boundary.

`common_reusable_tokens` now does the same: each group reports the longest
prefix its own rule accepts, rounded to its own block; the credit is the
smallest of those, capped at `len(tokens) - 1`, pulled down by the cumulative
groups' granularity, and reduced by the replay window. Driving the router with
three lengths, three attempts each and comparing the engine's per-request hit
delta with `credited_prefix_tokens` gives equality on all nine requests, and the
same rule leaves the Qwen rows above unchanged:

| prompt | attempt 1 | attempt 2 | attempt 3 |
| --- | --- | --- | --- |
| 769 | 0 / 0 | 768 / 768 | 768 / 768 |
| 1,025 | 768 / 768 | 1,024 / 1,024 | 1,024 / 1,024 |
| 1,537 | 1,024 / 1,024 | 1,536 / 1,536 | 1,536 / 1,536 |

Each cell is the engine's hit delta over the router's credit for that request.
The overlapping prefixes are why attempt 1 is not always zero: a 1,025-token
request finds the 768 tokens the previous 769-token prompt stored, and a
1,537-token request finds the 1,024 the 1,025 one stored.

A separate engine-only calibration pins the rule where the two models disagree:
after a 1,024-token prompt is cached, a 704-token prefix of it hits 0 on its
first attempt (the sliding tail does not cover 704) and 512 on its own repeat
(the 256-token group's granularity). Lengths 1,536 and 2,048 behave the same way
(1,280 and 1,792 per repeat), so the engine keeps each group's own hit and does
not re-check the window at a shorter boundary.

**DeepSeek-V4-Pro is the same shape as V4-Flash with different numbers.**
`deepseek-ai/DeepSeek-V4-Pro` is `model_type: deepseek_v4` and runs the same
`vllm/models/deepseek_v4/` package, so the group mechanism is identical; the
shapes move to 61 layers, hidden 7168, 128 attention heads, 384 routed experts,
`index_topk` 1024, `o_groups` 16 and 864.7 GB of weights in 64 shards. Captured
on the real model (eight B300 cards, vLLM 0.33) the layout is five groups, all
publishing:

| group | kind | block | window | publication for a 1,024-token store |
| --- | --- | --- | --- | --- |
| 0, 1 | sliding_window_mla | 64 | 128 | 2 hashes (the resident tail) |
| 2 | mla_attention | 256 | - | 4 hashes (the whole store), plus 4-token secondary pages |
| 3 | sliding_window_mla | 8 | 128 | 16 hashes (the resident tail) |
| 4 | sliding_window_mla | 4 | 8 | 2 hashes |

Neither `compress_ratios` difference reaches the cache: the pages are the same
`64, 64, 256, 8, 4` family as V4-Flash, the chain runs at 4 tokens (re-computing
it with vLLM's own `sha256_cbor` matched the compressed group's four hashes at
granularity 4 and none at 8 or above), and the engine's hit is
`align_down(tokens - 1, 256)`: a 1,024-token prompt reports 768. With
`hash_block_size` 4 the router credits 768 on the repeat, equal to the engine's
own count, and 0 on the first attempt before the events arrive.

**DeepSeek-V4.1-Flash is a third shape: bounded replay with one publishing
group.** It runs on the same vLLM 0.33 from eight B300 SXM6 cards (510.3 GB in
48 shards, real weights). The engine reports seven cache groups —
`kv cache group sizes (32, 32, 32, 32, 32, 128, 8)`, `kv lcm block sizes 128` —
and publishes events for **one** of them:

| group | kind | block | prefix cacheable | publishes |
| --- | --- | --- | --- | --- |
| 0-4 | sliding_window_mla | 32 | no (bounded replay, window 128) | nothing |
| 5 | mla_attention | 128 | yes | 8 hashes for 1,024 tokens, then 1 hash per 128-token store with `parent_block_hash` |
| 6 | sliding_window_mla | 8 | no | nothing |

Two measurements pin the contract:

- The hash chain runs at **128 tokens**, not at the smallest page. Re-computing
  the chain with vLLM's own `sha256_cbor` from `sha256_cbor("vllm-none-hash")`
  matched all eight published hashes at granularity 128 and none at 4, 8, 16, 32
  or 64, because only the cacheable group takes part in hashing.
- The engine's hit is `align_down(tokens - 1, 128)`: 1,024 tokens report 896 and
  1,537 report 1,536, which is the same rule as V4-Flash with a coarser block.

With `hash_block_size` 128, groups 0-4 declared as bounded-replay (window 128,
`prefix_replay_tokens` 128, not cacheable) and group 5 as a compressed attention
group with one token per state, the layout is admitted and the events apply. The
router then credits 768 for a 1,024-token repeat where the engine counts 896:
the 128-token difference is exactly the replay window. The scheduler computes
`num_replay_tokens = min(prefix_replay_tokens, num_hit_tokens)`, sets
`Request.replay_start` to the hit minus that window, and the worker rebuilds the
sliding-window KV for those tokens while leaving their cached KV alone — so the
window is prefilled again even though `prefix_cache_hits_total` counts it. Credit
is deliberately the smaller, real number, and the same scheduler asserts one
window for all groups (`Prefix replay windows should agree`), which is why the
layout carries a single `prefix_replay_tokens` rather than one per group.


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
