# router_policy replay traces

## TraceLab (PD merged)

Merged case `runtime.trace_path` points at the TraceLab subset:

`tests/python/e2e/perf/router_policy/data/tracelab-ci-uniform-28.jsonl`

`tracelab-ci-uniform-28.jsonl` is eight Codex sessions (28 requests) from
`UW-SyFI/TraceLab` revision `7256bbf77e76d5cb83f4ddbd5ad31d8faee9dc7a`, file
`data/v0.0.2/syfi_coding_trace.jsonl.gz`.

Regenerate it with [`extract_tracelab.py`](extract_tracelab.py):

```bash
python tests/python/e2e/perf/router_policy/data/extract_tracelab.py syfi_coding_trace.jsonl.gz
```

The script checks the compressed source SHA-256 and the extracted file SHA-256.

## TraceLab (PD disaggregated)

The three disaggregated scenarios share `tracelab-l4-6.jsonl`, the context-capped
subset in [PR #615's four-H20 measurements and reproduction script](https://github.com/vllm-project/router/pull/615#issuecomment-7207247874).
It contains **six complete sessions and 68 requests** from the same TraceLab
v0.0.2 source as the merged subset. No request token counts or timing fields
are changed.

The extractor identifies the first sixteen `(provider, session_id)` keys in
source order, collects all their rows across the complete source file, and
keeps the six sessions whose maximum `input_tokens_total + output_tokens` is
at most 65,536. All 68 rows retain source order; sessions are not truncated.
The maximum input plus output is 60,228 tokens.

Each PD case uses `task-num=6`, `max-concurrency=[6]`, `max_inflight=1`, and
`gap_scale=0.002`. Session dependencies remain serial, so observed request
concurrency can be below six. The model and three policies are unchanged;
AgentInfer 0.2.0 uses `replay_tracelab.yaml` and `/v1/chat/completions`.

Each worker uses a 65,536-token context and 12 GiB of GPU KV cache. CI runs
2P2D on four L4 GPUs with vLLM 0.30.0 and NIXL. The review's three-policy run
on four H20s completed all 68 requests per policy in 18m 17s total, including
startup and teardown. This is evidence for that H20 run; L4 runtime remains
to be measured.

Regenerate the committed subset from the official release asset:

```bash
curl -fL -o syfi_coding_trace.jsonl.gz \
  https://github.com/uw-syfi/TraceLab/releases/download/v0.0.2/syfi_coding_trace.jsonl.gz
python tests/python/e2e/perf/router_policy/data/extract_tracelab.py \
  syfi_coding_trace.jsonl.gz --subset pd_disagg
```

The source SHA-256 is
`11ce51ec0a25e3d1d95b025bca2f7d1647e47571eb7cc968acd5fc64d4b4fb65`.
The output SHA-256 is
`105c375c30634d8191a22cfbc56a6e2d63ce3e2ddc571ffe7519581b8086e2e0`.
The extractor verifies both digests before replacing the PD fixture. All three
PD cases check the output digest before GPU startup, and publication requires
exactly six distinct source sessions and 68 matched requests. Nightly uses the
committed subset and does not download the full source.

`TRACELAB_JSONL_URL` overrides only the unpinned merged trace. It cannot replace
the pinned PD fixture or relax the PD session/request counts.
