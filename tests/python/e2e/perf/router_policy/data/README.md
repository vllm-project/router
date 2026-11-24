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

The three disaggregated scenarios share `agentx-ci-8x8.jsonl`, a fixed slice from the public
[SemiAnalysis dataset](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/tree/23f152f6f0f9399a85901b89a6458def0ef16729),
published under Apache-2.0. The slice contains hash-block recipes and timing
metadata from real requests. Every selected session retains its first eight
request entries verbatim, including their original hash IDs and token counts.

`agentx-ci-8x8.manifest.json` records the upstream revision and file digest,
the selected session IDs, source line numbers and row digests, and the slice
digest. The eight sessions contain 64 requests, 2,617,664 input tokens, and
48,850 output tokens. Maximum input plus output is 65,339 tokens. The PD cases
therefore use a 65,536-token context and `gap_scale=0.002`; recorded request
timings are compressed while per-session request dependencies remain intact.

AgentInfer 0.2.0 renders the full hash snapshots directly as token IDs through
`/v1/completions`. It preserves the recorded input lengths. All three cases check
the same slice SHA256 before starting any GPU processes. The fixture stays in
the repository, so nightly jobs do not download the full upstream dataset.

To reproduce the slice, obtain the pinned upstream `traces.jsonl` or a byte
prefix containing its first 30 complete lines, then run:

```bash
python tests/python/e2e/perf/router_policy/data/extract_agentx_ci.py \
  /path/to/traces.jsonl
```

The extractor verifies every selected source row before writing and verifies
the final slice digest. It does not need to read the rest of the 1.85 GB source
file. The `.gitattributes` rule keeps the slice's line endings fixed across
platforms so the digest is stable.
