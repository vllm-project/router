# TraceLab cold-replay subset

`tracelab-ci-uniform-28.jsonl` is eight Codex sessions (28 requests) from
`UW-SyFI/TraceLab` revision `7256bbf77e76d5cb83f4ddbd5ad31d8faee9dc7a`, file
`data/v0.0.2/syfi_coding_trace.jsonl.gz`.

Regenerate it with [`extract_tracelab.py`](extract_tracelab.py):

```bash
python tests/python/e2e/perf/router_policy/data/extract_tracelab.py syfi_coding_trace.jsonl.gz
```

The script checks the compressed source SHA-256 and the extracted file SHA-256.
Case `runtime.trace_path` points at:

`tests/python/e2e/perf/router_policy/data/tracelab-ci-uniform-28.jsonl`
