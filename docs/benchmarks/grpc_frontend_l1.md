# gRPC frontend L1 benchmark

Same mock worker and JSON fields as the
[L0 benchmark](grpc_frontend_l0.md). No model download or GPU is needed.
The prompt is one shared chat history plus a unique last user turn, rendered
with the Qwen template so the cut falls on `<|im_end|>`.

```bash
cargo test --release --test grpc_frontend_bench grpc_frontend_l1_overhead -- --ignored --nocapture
```

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_ROUTER_BENCH_TOKENS` | `2048,8192` | Rendered prompt length, in tokens |
| `VLLM_ROUTER_BENCH_REUSE` | `0,30,70,99.99` | Target percent of that prompt that is the shared history |
| `VLLM_ROUTER_BENCH_CONCURRENCY` | `1,4,8,16,32,64` | Requests in flight |
| `VLLM_ROUTER_BENCH_MEASURE_SECS` | `2` | Measurement duration per case |

The test pins eight Tokio threads, the same harness as the L0 bench.
`vllm-router` itself does not: `Runtime::new()` uses one worker thread per
CPU, and `TOKIO_WORKER_THREADS` overrides that count. Concurrency 64 here is
64 requests on those eight threads.

`0` uses a unique system message, so nothing matches. Any other percent keeps
the system prompt and earlier turns identical and changes only the last user
message. The tail is `s{index} measured: name the function.` padded with
` note`. When that sentence is longer than the uncached budget (the 99.99%
case on a short prompt), the tail is the fixed-width marker `q{index}`
instead. `reuse_pct` in the output is the measured cached-token share;
`reuse_target_pct` is what was requested. The new user turn also renders
`<|im_start|>user`, `<|im_end|>`, and the generation prompt, about 25 tokens
on this fixture. A literal 99.99% (one uncached token per 10,000) therefore
needs a long prompt; at 8192 tokens the shortest turn measures about 99.7%.

L0 is left off, so the comparison is L1 alone. Each case warms one request,
then times L1 off and on. A shape check,
`l1_bench_prompts_hit_the_requested_reuse`, runs in the normal test suite and
checks that 0%, 70%, and 99.99% land in the expected reuse bands.

```bash
cargo test --test grpc_frontend_bench l1_bench_prompts_hit_the_requested_reuse -- --nocapture
```

Example of a shorter timing run:

```bash
VLLM_ROUTER_BENCH_TOKENS=2048 \
VLLM_ROUTER_BENCH_REUSE=0,70,99.99 \
VLLM_ROUTER_BENCH_CONCURRENCY=1 \
cargo test --release --test grpc_frontend_bench grpc_frontend_l1_overhead -- --ignored --nocapture
```
