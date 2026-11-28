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

## AgentX (PD disaggregated)

The three disaggregated scenarios share `agentx-l4-12x12.jsonl`, the context-capped
AgentX subset in [PR #615's four-H20 measurements and reproduction script](https://github.com/vllm-project/router/pull/615#issuecomment-7207247874).
It contains **twelve sessions with twelve direct calls each: 144 requests**.
Original token counts, hash IDs, and timing fields are retained.

The source is the Apache-2.0 [SemiAnalysis AgentX 256k dataset](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/tree/8fecd2fc56694469f758f0afbbb6335ad3043740),
file `traces.jsonl` at revision `8fecd2fc56694469f758f0afbbb6335ad3043740`.
This is the context-capped 256k source used by the comment's L4-sized slice.

The extractor keeps the first twelve top-level entries of each candidate
session. It requires local hash IDs, positive input, more than one output
token, valid hash-block counts, and at most 65,536 input plus output tokens
for every call, including nested subagent calls. It sorts the 57 eligible
sessions by total input tokens, breaking ties by session ID, then selects
one-based ranks 5-8, 27-30, and 50-53. The selected sessions contain no subagent
groups and expand into exactly 144 model requests with AgentInfer 0.2.0.
Maximum input plus output is 65,477 tokens.

Each PD case uses `task-num=12`, `max-concurrency=[12]`, `max_inflight=1`, and
`gap_scale=0.002`. Requests within each session are serial, so observed request
concurrency can be below twelve. AgentInfer 0.2.0 uses `replay_agentX.yaml`
and sends hash snapshots as token IDs through `/v1/completions`.

Each worker uses a 65,536-token context and 12 GiB of GPU KV cache. CI runs
2P2D on four L4 GPUs with vLLM 0.30.0 and NIXL. The review's three-policy run
on four H20s completed all 144 requests per policy in 45m 46s total, including
startup and teardown. This is evidence for that H20 run; L4 runtime remains
to be measured.

Regenerate the committed subset with [`extract_agentx_ci.py`](extract_agentx_ci.py):

```bash
curl -fL -o agentx-source.jsonl \
  https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/resolve/8fecd2fc56694469f758f0afbbb6335ad3043740/traces.jsonl
python tests/python/e2e/perf/router_policy/data/extract_agentx_ci.py agentx-source.jsonl
```

The source SHA-256 is
`e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef`.
The output SHA-256 is
`1f37cb060e4609c5d52db9ca2d3674b23d32c6c42f3ed4a2cb4861e78b1f772b`.
The extractor verifies both digests before writing the fixture. All three
PD cases check the output digest before GPU startup, normalizing CRLF to LF
without rewriting the file. Other content changes still fail verification.
Publication requires
exactly twelve distinct source sessions and 144 matched requests. Nightly
uses the committed subset and does not download the full source. Reproduction
instructions and hashes live here and in the extractor; no manifest is needed.

`TRACELAB_JSONL_URL` overrides only the merged TraceLab workload. It cannot
replace the pinned AgentX fixture or relax the PD session/request counts.
