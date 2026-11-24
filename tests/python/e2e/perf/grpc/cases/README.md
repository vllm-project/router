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

Optional **`model`** in JSON (HF id or local path). The `rust_grpc` and `python_http`
cases default to `Qwen/Qwen3.8-27B`. Set **`export MODEL=...`** to override the JSON.
Tests skip when neither the environment nor the case file supplies a model.

## Binaries (`router.bin`, `worker.bin`)

Optional paths to **`vllm-router`** and **`vllm-rs`**. Relative paths are resolved from the
repository root (where `Cargo.toml` lives). **`ROUTER_BIN` / `VLLM_RS` env override JSON.**

```json
"router": { "bin": "target/release/vllm-router", "port": 16002 },
"worker": { "bin": "/path/to/vllm-rs", "http_port": 18002 }
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
