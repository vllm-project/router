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

The three disaggregated scenarios share `agentx-ci-12x12.jsonl`, a fixed slice from the public
[SemiAnalysis dataset](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/tree/23f152f6f0f9399a85901b89a6458def0ef16729),
published under Apache-2.0. The slice contains hash-block recipes and timing
metadata from real requests. Every selected session retains its first twelve
request entries verbatim, including their original hash IDs and token counts.

The twelve sessions contain 144 requests, 5,900,864 input tokens, and
98,003 output tokens. Maximum input plus output is 65,181 tokens. The PD cases
therefore use a 65,536-token context and `gap_scale=0.002`; recorded request
timings are compressed while per-session request dependencies remain intact.
Each PD case sets both `task-num` and `max-concurrency` to 12, with one
in-flight request per session. This permits up to twelve concurrent requests;
trace gaps and completed sessions can reduce the observed concurrency.

AgentInfer 0.2.0 renders the full hash snapshots directly as token IDs through
`/v1/completions`. It preserves the recorded input lengths. All three cases check
the same slice SHA256 before starting any GPU processes. The fixture stays in
the repository, so nightly jobs do not download the full upstream dataset.

To reproduce the slice, use [`extract_agentx_ci.py`](extract_agentx_ci.py).
Obtain `traces.jsonl` at the revision linked above, or a byte prefix containing
its first 84 complete lines (375,809,283 bytes), then run:

```bash
python tests/python/e2e/perf/router_policy/data/extract_agentx_ci.py \
  /path/to/traces.jsonl
```

The script selects the first twelve distinct eligible sessions in source
order and retains their first twelve request entries. An eligible session has
local hash IDs, at least twelve turns, no subagent entries in those turns,
more than one output token per request, valid hash-block counts, and input plus
output within 65,536 tokens. It preserves the original request fields.

The extraction rules and expected slice digest are defined in the script;
there is no separate manifest file. The expected SHA-256 is
`faf79824c90ddb72774a6f5f018f67821f26c48743cd5171cfbcfe9e00855537`.
The script verifies this digest before writing, and the case JSONs use the
same value at replay startup. It stops after the twelfth eligible session and
does not need to read the rest of the 1.85 GB source file.
