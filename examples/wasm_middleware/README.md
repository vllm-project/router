# Example WASM OnRequest middleware

Small demo guest for the vLLM Router pluggable middleware runtime
([RFC #236](https://github.com/vllm-project/router/issues/236)).

## Build

```bash
./build.sh
```

Produces `wasm_middleware_example.component.wasm` in this directory.

## Behavior

- `Reject(400)` when the request body contains `__wasm_reject__`
- Otherwise `Modify` with header `x-wasm-middleware: example` (body unchanged)

## Run with Router

```bash
cargo run --release -- \
  --worker-urls http://localhost:8000 \
  --wasm-middleware ./examples/wasm_middleware/wasm_middleware_example.component.wasm
```

By default the host only invokes the plugin for `POST /v1/chat/completions`.
Use repeated `--wasm-middleware-route` flags to attach additional paths.
