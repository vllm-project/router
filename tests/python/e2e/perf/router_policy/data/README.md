# router_policy replay traces

Case `runtime.trace_path` points at the AgentX stress file:

`tests/python/e2e/perf/router_policy/data/agentx-stress-68.jsonl`

`agentx-stress-68.jsonl` is the synthetic two-session AgentX trace from the
router PR 477 cold-replay note (4× H20). It is two session records and 68
requests. SHA-256 is
`452cf96ad95b321ec9b907ef96fed007910cf9d39790bc1a4474c55f520409b5`.
Do not edit it; the hash is the identity of that run.

`tracelab-ci-uniform-28.jsonl` is the earlier eight-session TraceLab subset.
Regenerate that file with [`extract_tracelab.py`](extract_tracelab.py). It is
not the case default.
