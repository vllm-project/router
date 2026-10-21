# Perf case JSON

Each `*.json` file drives one backend (`rust_grpc`, `rust_http`, `python_http`).

## Model

Set **`model`** to a HuggingFace id or local snapshot directory (same as `export MODEL=...`).
Bench token sizing and worker launch both use this path; **`export MODEL=...` overrides JSON**.

## Worker (`worker.cli`)

**Required.** Flat argv list appended to `vllm-rs serve` / `vllm serve` (TP, parsers,
`--max-model-len`, prefix caching, etc.). Edit per case when the model or GPU layout changes;
do not rely on Python defaults.

```json
"worker": {
  "cli": [
    "--tensor-parallel-size", "2",
    "--max-model-len", "262144",
    "--enable-prefix-caching"
  ]
}
```
