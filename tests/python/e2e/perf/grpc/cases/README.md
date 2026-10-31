# Perf case JSON

Each `*.json` file drives one backend (`rust_grpc`, `rust_http`, `python_http`).

## `--perf-case` selector

| Form | Behavior |
|------|----------|
| `rust_grpc` | Loads bundled `cases/rust_grpc.json` |
| `cases/rust_grpc.json` | Same (relative to repo) |
| `/abs/path/to/rust_grpc.json` | Must exist; **no** silent fallback |
| `rust_grpc.json` only (basename) | Must exist in **cwd**; if missing but `cases/rust_grpc.json` exists, pytest errors and asks for stem `rust_grpc` |

Use the stem (`rust_grpc`) in CI/docs for bundled cases.

## Model

Optional **`model`** in JSON (HF id or local path). For perf runs, set **`export MODEL=...`**
(**env overrides JSON**). Do not commit placeholder paths; with no `MODEL`, tests skip until
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
