# Perf case JSON

Each `*.json` file drives one backend (`rust_grpc`, `rust_http`, `python_http`).

## Model

Set **`model`** to a HuggingFace id or local snapshot directory (same as `export MODEL=...`).
Bench token sizing and worker launch both use this path; **`export MODEL=...` overrides JSON**.

Replace the placeholder in committed files before running, or use env only:

```json
"model": "/nvme1n1p1/you/Qwen3.8-27B/snapshot"
```

Optional env **`MODEL_DIR`** (not in JSON) only if the served `model` name differs from the
local directory used for tokenizer loading.
