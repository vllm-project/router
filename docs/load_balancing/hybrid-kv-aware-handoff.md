# Hybrid KV-aware routing: implementation handoff

Status: **draft, not production-ready**. The design is in the [English RFC](hybrid-kv-aware-rfc.md) and [Chinese RFC](hybrid-kv-aware-rfc.zh.md). The implementation uses existing upstream vLLM events; it does not require a vLLM patch.

## Implemented

- Router-managed layouts describe all required groups and per-rank topics. Replay and source checks gate the multi-group index; missing checkpoint hashes, rank mismatches, gaps and layout mismatches must not create hybrid credit.
- Regular HTTP routing accepts `--kv-hybrid-layout`. Experimental P/D routing accepts `--pd-hybrid-config`, binds an index to each configured P, and restricts P/D admission to operator-declared pairs. P/D Program scheduling is rejected for this configuration.
- An H20 test with upstream main Python code, vLLM 0.33 binary dependencies and a reduced Qwen4Exp dummy model observed 0 then 1,920 cached tokens on two identical 2,048-token requests; Router estimated 0 then 1,920. This is one rank and one worker, not a production workload.
- Hybrid P/D unit tests cover pair filtering, replay and publisher-topic invalidation. An H20 1P1D test with NIXL 1.5.2 and an intentionally incorrect 320-token Router layout transferred 4,505,600 bytes; the engine reported 2,016 cached tokens for the repeated request and Router correctly granted zero credit. A matching 288-token layout test is in progress.

## Remaining work and acceptance criteria

1. **P/D:** Complete the real upstream-main NIXL 1P1D test. Verify the effective layout from each deployment: enabling P/D changed the Qwen4Exp Mamba/attention block size from 320 to 288 in the test. Confirm that a mismatched layout gets zero credit, the matching layout gives the same reusable prefix as P, NIXL transfers state, and outputs match a direct reference. Exercise nonmatching pairs and multiple DP ranks. P/D handoff-failure behavior was explicitly deferred by the user.
2. **Lifecycle and unknown layouts:** Integrate trusted deployment generation changes and automatic rebinding. A static topic and periodic HTTP checks cannot establish that a worker has not restarted with the same topic. Ensure unknown hybrid models cannot obtain legacy group-0 credit, while retaining existing full-attention behavior.
3. **Model layouts:** Capture stock-vLLM event and restore behavior for Qwen3.5/3.6/3.8/4 and DeepSeek V4/V4.1. Sliding-window, bounded replay, compressed and shared states require their own verified restore and alignment rules; do not enable unknown groups or infer completeness from observed events.
4. **Advisory affinity:** Add the separate tools/reasoning path for requests without exact token inputs. Scope and bound history by worker/rank/generation and model/tenant, respect load and health, do not persist raw request text, and never turn text similarity into KV credit.
5. **Production checks:** Test feature-off and existing full-attention/gRPC paths, worker restart and replay truncation, DP isolation, recovery, races, capacity, multi-worker routing, actual engine hit counts and output equivalence. Run full Rust tests, strict Clippy and performance tests; document the supported version/model/connector matrix.

The currently checked-in static layouts are operator assertions. They are not automatic verification of model weights, complete groups or publisher generation. Do not deploy an unverified combination as a production feature.
