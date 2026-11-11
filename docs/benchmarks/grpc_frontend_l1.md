# gRPC frontend L1 benchmark

Same mock worker and JSON fields as the
[L0 benchmark](grpc_frontend_l0.md). No model download or GPU is needed.
The prompt is one shared chat history plus a unique last user turn, rendered
with the Qwen template so the cut falls on `<|im_end|>`.

```bash
cargo test --features cpu,ready --release --test grpc_frontend_bench grpc_frontend_l1_overhead -- --ignored --nocapture
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
cargo test --features cpu,ready --test grpc_frontend_bench l1_bench_prompts_hit_the_requested_reuse -- --nocapture
```

Example of a shorter timing run:

```bash
VLLM_ROUTER_BENCH_TOKENS=2048 \
VLLM_ROUTER_BENCH_REUSE=0,70,99.99 \
VLLM_ROUTER_BENCH_CONCURRENCY=1 \
cargo test --features cpu,ready --release --test grpc_frontend_bench grpc_frontend_l1_overhead -- --ignored --nocapture
```

## End to end with a real worker

[`bench_prefix_kvhit.py`](../../benches/backend/bench_prefix_kvhit.py)
`--shape turns` sends the same shape to a running router: shared earlier
turns, then a unique last user message. The default `--shape split` cuts
inside one message, where L1 has no boundary to reuse, so use `turns`.

Start the same gRPC worker once, then run the client against a router started
with `VLLM_ROUTER_STAGES=1`, first without and then with
`VLLM_ROUTER_L1_CACHE=1`:

```bash
VLLM_ROUTER_STAGES=1 VLLM_ROUTER_L1_CACHE=1 \
  vllm-router --worker-urls grpc://127.0.0.1:50051 --port 30000

python benches/backend/bench_prefix_kvhit.py \
  --router-url http://127.0.0.1:30000 --model /path/or/hf-id \
  --shape turns --hit-rate 0.70 \
  --tokens 131072 --model-dir /path/or/hf-id \
  --max-tokens 128 --warmup --concurrency 8 --requests 16 --timeout 1800
```

`wave.summary` reports `ttft_ms` and `e2e_ms` from the client, and from the
router stages `template_ms`, `encode_ms`, `frontend_ms` and `encode_reuse`
(cached prompt tokens divided by prompt tokens). Worker KV reuse is the same
in both runs, so TTFT and e2e should stay level while `encode_ms` and
`frontend_ms` drop with L1 on, more at higher `--hit-rate` and longer
prompts. `--tokens` sizes the text for a one-message chat; with `turns` the
rendered prompt is a little longer, and `prompt_tokens` in each row is the
exact count. `vllm_tokenizer_l1_*` on the Prometheus port (default
`127.0.0.1:29000/metrics`) gives run totals, and
`vllm_tokenizer_l1_bypasses_total{reason}` should stay near zero.
