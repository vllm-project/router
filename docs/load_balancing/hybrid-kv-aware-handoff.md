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
- **P/D handoff checks.** Before the decode leg is dispatched, the decode
  transfer parameters must exist, echo the request's `transfer_id`, and carry a
  connector identity for the chosen engine. A decode failure after a successful
  prefill handoff is counted separately
  (`vllm_router_pd_handoff_failures_total`), logged with both endpoints, and
  drops the cached NIXL push identity so a retry performs a fresh handshake
  instead of reusing stale transfer state.

## Remaining work and acceptance criteria

1. **P/D:** Automate the observed matching and mismatched-layout assertions and verify a fully built upstream main with real weights. Enabling P/D changed the Qwen4Exp Mamba/attention block size from 320 to 288 in the test; verify each deployment's effective layout. Compare outputs against a direct reference, and exercise nonmatching pairs and multiple DP ranks. Handoff validation and post-handoff decode failures now fail closed; an automated P-fails and P-succeeds/D-fails pair test is still required.
2. **Lifecycle and unknown layouts:** Integrate trusted deployment generation changes and automatic rebinding. A static topic and periodic HTTP checks cannot establish that a worker has not restarted with the same topic. Ensure unknown hybrid models cannot obtain legacy group-0 credit, while retaining existing full-attention behavior.
3. **Model layouts:** Capture stock-vLLM event and restore behavior for Qwen3.5/3.6/3.8/4 and DeepSeek V4/V4.1. The sliding-window/bounded-replay path now models one agreed layout window and rejects disagreeing groups, but the DeepSeek V4 compressed/shared-state layout still needs captured events from a real deployment before it can be enabled. Do not enable unknown groups or infer completeness from observed events.
4. **Advisory affinity:** The advisory namespace exists and never credits; bounding history by worker/rank/generation and tenant, and documenting the tools/reasoning entry points, still need end-to-end coverage.
5. **Production checks:** Test feature-off and existing full-attention/gRPC paths, worker restart and replay truncation, DP isolation, recovery, races, capacity, multi-worker routing, actual engine hit counts and output equivalence. Run full Rust tests, strict Clippy and performance tests; document the supported version/model/connector matrix.

The currently checked-in static layouts are operator assertions. They are not automatic verification of model weights, complete groups or publisher generation. Do not deploy an unverified combination as a production feature.
