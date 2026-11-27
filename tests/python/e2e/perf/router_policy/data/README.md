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

The three disaggregated scenarios share `agentx-hetero12x12.jsonl`, the fixed
heterogeneous fixture referenced in [PR #615's dataset review](https://github.com/vllm-project/router/pull/615#discussion_r4677145345).
Its session IDs and serialization match the [reference extractor at commit
`d35dec404b1132f0504ec7cb165aa750d208b9c2`](https://github.com/KaisennHu/router/blob/d35dec404b1132f0504ec7cb165aa750d208b9c2/tests/python/e2e/perf/router_policy/agentx_pd_hetero/extract_agentx_hetero.py).

The source is the Apache-2.0 [SemiAnalysis dataset](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/tree/23f152f6f0f9399a85901b89a6458def0ef16729),
file `traces.jsonl` (393 sessions). Its SHA-256 is
`29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`.
The fixture retains four sessions from each of the reference's small, medium,
and large input-size cohorts, in the reference order. Each session keeps its
first twelve top-level entries, including nested subagent requests, original
hash IDs, token counts, and timing metadata. Session input totals span about
415K to 930K tokens (2.24 times), exercising placement of differently sized
sessions and reuse within each session.

The 144 top-level entries expand into **315 model requests** with AgentInfer
0.2.0. Including subagents, they contain 20,554,048 input tokens and 210,665
output tokens; maximum input plus output per request is 151,962 tokens.
They are not 144 independent HTTP requests. Each PD case uses
`task-num=12`, `max-concurrency=[12]`, `max_inflight=1`, and `gap_scale=0.002`.
There are up to twelve in-flight requests across the twelve session tasks;
dependencies, trace gaps, and completed tasks can reduce observed concurrency.

The PD cases use a 163,840-token context and 26 GiB of GPU KV cache per worker,
matching the reference workload's memory profile. CI uses four H100 GPUs for
2P2D; the previous four-L4 profile cannot hold this per-worker KV allocation.
The model, vLLM 0.30.0, and three policies remain unchanged. AgentInfer 0.2.0
replays hash snapshots as token IDs through `/v1/completions`.

To reproduce the fixture, obtain `traces.jsonl` at the pinned revision above
and use [`extract_agentx_ci.py`](extract_agentx_ci.py):

```bash
python tests/python/e2e/perf/router_policy/data/extract_agentx_ci.py \
  /path/to/traces.jsonl
```

The fixed session IDs, turn cap, and expected SHA-256 are in the script;
there is no separate manifest. The expected output SHA-256 is
`e7e7fd7cd31f178483e2027281ce4592fca38bd35252c09b88e2a9dee1078638`.
The script verifies it before writing, and all three cases check the same
digest before GPU processes start. Nightly jobs use the committed fixture
without downloading the 1.85 GB source file.
