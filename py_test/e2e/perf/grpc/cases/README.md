# Perf case JSON

Each `*.json` file drives one backend (`rust_grpc`, `rust_http`, `python_http`).

## Model

Optional **`model`** in JSON (HF id or local path). For perf runs, set **`export GRPC_MODEL=...`**
(**env overrides JSON**). Do not commit placeholder paths; with no `GRPC_MODEL`, tests skip until
you set the env var or add a real `model` in your local case file.

## Binaries (`stack.router_bin`, `stack.vllm_rs_bin`)

Optional paths to **`vllm-router`** and **`vllm-rs`**. Relative paths are resolved from the
repository root (where `Cargo.toml` lives). **`ROUTER_BIN` / `VLLM_RS` env override JSON.**

```json
"stack": {
  "router_bin": "target/release/vllm-router",
  "vllm_rs_bin": "/path/to/vllm-rs"
}
```

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
