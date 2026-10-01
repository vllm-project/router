# Router CPU overhead benchmarks

These benchmarks measure router overhead before the tokenizer integration
in #244. Inputs are seeded and offline; workers return immediately.
The results measure router cost. Measuring KV-cache benefits requires real workers.

## Run

```bash
# In-process benchmarks
cargo bench --bench routing_input
cargo bench --bench routing_input -- edge/

# End-to-end harness (ignored by default)
cargo test --release --test router_overhead_bench -- --ignored --nocapture

# CI checks for shared fixtures, corpus and mock workers
cargo test --test routing_edge_cases_test --test bench_corpus_test --test router_overhead_bench
```

CI builds both benchmarks but runs only the non-ignored tests. The tools
are also available through `scripts/run_benchmarks.py` with
`--bench routing_input` or `--router-overhead`.

## Inputs

`tests/common/bench_corpus.rs` generates the same requests from seed `0x244`,
input size and request index.

| Corpus | Prompts |
|---|---|
| `hot64` | Cycle through 64 fixed prompts |
| `cold` | Unique marker at the start; no shared prefix |
| `mixed90` | 90% hot, 10% cold |
| `short_shared_prefix` | Share up to 600 bytes and at most half the prompt, then a unique tail |
| `long_shared_prefix` | Share everything except a unique 64-byte tail |
| `utf8_hot64` | Hot prompts with CJK, kana and emoji after an ASCII marker |

Default text sizes are 200 B, 2 KiB and 16 KiB; long-input cases use
128 KiB, 512 KiB and 1 MiB. Sizes count prompt bytes, including for UTF-8.
Requests use `max_tokens: 16`, no streaming, and either a completion prompt
or one chat user message without a session ID.

Non-shared text comes from up to 256 distinct bodies. The pool shrinks for
large inputs toward a 64 MiB budget, with a minimum of eight bodies. Distinct
content matters: repeating one body made 16 KiB rendezvous selection about
three times cheaper in an earlier measurement.

`IdCorpus` generates pre-tokenized prompts, sized by ID count. Criterion
uses 16,384 and 131,072 IDs; the harness enables them through
`VLLM_ROUTER_BENCH_PROMPT_IDS`. IDs are five-digit integers, making request
size constant within each cell. Reports distinguish `text<bytes>` from
`ids<count>` and include the serialized `body_bytes`.

## Criterion

| Group | Timed work |
|---|---|
| `routing_key/extract_text_for_routing` | Extract routing text from completion or chat requests, including dropping the result |
| `policy/{cache_aware,rendezvous_hash}` | Select among four healthy workers using a warmed hot set |
| `cache_aware_key_format` | Insert and prefix-match raw text, one character per token, or tagged decimal IDs at 128–8192 tokens; tree operations only |
| `edge/cache_aware` | Select from long keys, threshold cases, imbalanced loads and UTF-8 keys |
| `edge/long_input` | Parse through `axum::Json`, extract routing text and serialize completion/chat bodies; also parse equal-sized CJK inputs |
| `edge/rendezvous` | Select using long prompts, session headers or JSON-like fields in prompt text |
| `edge/token_ids` | Parse and extract routing text from ID prompts |
| `edge/workers` | Select among 1, 4, 16 and 64 workers |

Compare commits on the same machine:

```bash
cargo bench --bench routing_input -- --save-baseline main  # base commit
cargo bench --bench routing_input -- --baseline main       # changed commit
```

### Edge fixtures

`tests/common/routing_edge.rs` supplies both CI tests and Criterion. Each
cache-aware benchmark asserts its expected worker before timing. CI also
checks exact character counts and selection after a fixture reset.

| Coverage | Expected behavior |
|---|---|
| Long ASCII and CJK keys | Full hits use the warmed tenant; cold or 45% matches use the least-loaded worker |
| Match threshold | 45% misses, 55% hits; exactly 50% and one character below miss, one above hits |
| Load thresholds | Imbalanced only when both `max - min > abs` and `max > min × rel`; test equality and each condition alone |
| UTF-8 boundaries | Count characters when byte and character ratios disagree, including a fork inside a multibyte character |
| Unhealthy tenant | Fall back to the first healthy worker, remove the stale tenant, and miss after it recovers |
| Session header | Override two long prompts that otherwise select different workers; ignore an empty header |
| Worker count | Every hot prompt hits its assigned tenant at 1, 4, 16 and 64 workers |
| HTTP body limit | Route a body of exactly 1 MiB; reject one byte more with 413 |

Exact threshold and stale-tenant cases run in CI only. Criterion also
measures JSON-like prompt text and token-ID parsing; those cases check
input shape without fixing the current routing-key format as a contract.

The fixture configuration differs from library and CLI defaults:

| Source | Cache threshold | Absolute load threshold | Relative load threshold | Eviction |
|---|---:|---:|---:|---|
| Edge fixtures | 0.5 | 5 | 2.0 | Off |
| `CacheAwareConfig::default()` | 0.5 | 32 | 1.1 | 30 s |
| CLI / overhead harness | 0.3 | 64 | 1.5 | 120 s / 3600 s |

Branch cases warm W1 and normally use loads `[1, 1, 1, 0]`: W1 identifies
a cache hit, W3 a low-match or imbalanced selection, and W0 the fallback
when W1 is unhealthy. Small load thresholds make boundary cases easy to set up.

Routing inserts the probe into the tree. Before each timed call, the
benchmark clears and re-warms the fixture outside the timer. Long inputs
are reused one case at a time. Cleanup prunes tree nodes to break their
parent/child `Arc` cycles. Worker-count cases need no reset because repeated
hits leave the warmed tree unchanged.

### Current behavior exposed by the inputs

- A session header skips rendezvous text scanning and hashing. Parsing,
  copying the completion prompt and serializing the request still cost time;
  read the header rows alongside `edge/long_input`.
- `PromptInput` tries three untagged variants before `String`. Each failed
  attempt formats the whole text into an error. The CJK parsing rows expose
  the added cost for non-ASCII text.
- Rendezvous uses a `"user": "…"` field found inside prompt text as its key.
- Equal-length ID prompts all produce `token_ids:<count>`. Rendezvous sends
  them to one worker; cache-aware routing also keeps them together until
  load imbalance triggers selection of the least-loaded worker.

These benchmarks record existing behavior; they do not change routing or parsing.

## End-to-end harness

Each routed cell starts a fresh router process in front of four mock workers.
Defaults are concurrency 1 and 64, 5 s warmup, 20 s measurement, and one
repeat. The client sends its next request after reading the previous
response. A rate cap can pace each task while keeping at most one request
in flight per task.

| Scenario | Policy | Route |
|---|---|---|
| `direct_completions` | Direct to mock worker | `/v1/completions` |
| `direct_chat` | Direct to mock worker | `/v1/chat/completions` |
| `completions_off` | `cache_aware` | `/v1/completions` |
| `completions_rendezvous` | `rendezvous_hash` | `/v1/completions` |
| `chat_off` | `cache_aware` | `/v1/chat/completions` |

### Configuration

All variables below have the prefix `VLLM_ROUTER_BENCH_`.

| Suffix | Default | Meaning |
|---|---|---|
| `SCENARIOS` | All five above | Comma-separated scenario names |
| `SIZES` | `200,2048,16384` | Text prompt bytes |
| `CORPORA` | `hot64,cold` | Comma-separated corpus names |
| `PROMPT_IDS` | None | ID counts; completions only, separate from text size/corpus combinations |
| `CONCURRENCY` | `1,64` | Requests in flight; each value must be positive |
| `RATE` | `0` | Total requests/s cap; zero means closed loop |
| `REPEATS` | `1` | Full matrix repeats; scenario order rotates each time |
| `WARMUP_SECS` / `MEASURE_SECS` | `5` / `20` | Per-cell durations |
| `WORKERS` | `4` | Mock workers |
| `ROUTER_CPUS` | Unpinned | Linux CPU list passed to `taskset -c` for the router only |
| `OUT_DIR` | `target/router_overhead` | Reports and router logs |

Quick smoke test:

```bash
VLLM_ROUTER_BENCH_SCENARIOS=direct_completions,completions_off \
VLLM_ROUTER_BENCH_SIZES=2048 VLLM_ROUTER_BENCH_CONCURRENCY=4 \
VLLM_ROUTER_BENCH_WARMUP_SECS=1 VLLM_ROUTER_BENCH_MEASURE_SECS=3 \
cargo test --release --test router_overhead_bench -- --ignored --nocapture
```

For long inputs, name the corpora explicitly. The harness rejects `cold`,
`mixed90` and `short_shared_prefix` at 128 KiB and above because they can add
nearly a full prompt to the cache-aware tree per request.

```bash
VLLM_ROUTER_BENCH_SIZES=131072,524288,1048576 \
VLLM_ROUTER_BENCH_CORPORA=hot64,utf8_hot64,long_shared_prefix \
VLLM_ROUTER_BENCH_PROMPT_IDS=131072 \
VLLM_ROUTER_BENCH_CONCURRENCY=1,16 \
cargo test --release --test router_overhead_bench -- --ignored --nocapture
```

### Read the results

- **Latency:** `mean_minus_direct_us` subtracts the direct mean for the same
  route, input, corpus, concurrency and repeat. Without that direct row it
  stays unset. Percentiles are shown side by side; subtracting P99s does not
  measure the router's P99.
- **Router resources:** `wait4` reads only the router child's CPU and peak
  RSS. RSS is normalized to bytes. CPU per 1,000 requests includes startup,
  warmup and measurement, divided by successful requests across warmup and
  measurement. Startup matters more in low-throughput cells.
- **Tree growth:** eviction is set to 3600 s to keep it out of normal cells.
  Non-repeating inputs grow the tree; `long_shared_prefix` adds short tails,
  while cold inputs add nearly the whole prompt. CPU and RSS include this work.
- **Client load:** client and mocks share eight runtime threads; the router
  uses its default thread count. Body construction happens outside request
  latency timing but still uses client CPU. `harness_cores` is harness CPU
  divided by cell wall time, including router startup and shutdown when present. High usage suggests a
  client limit; low usage alone does not rule one out. A run with bodies
  prebuilt can test this.
- **Load spread:** `hotspot` is `workers × largest worker share`. It ranges
  from 1 (even) to the worker count (all requests on one worker).
- **Noise:** use at least three repeats. The summary shows per-cell medians
  and mean-latency spread, `(max - min) / median`. Treat differences within
  that spread as inconclusive.

To limit router CPUs on Linux, use `ROUTER_CPUS`; pinning the entire test
also pins the client and mocks. `TOKIO_WORKER_THREADS` sets the router's
async thread count, not its CPU affinity. Leave enough cores for the client
and mocks to avoid contention.

The output directory contains `summary.json`, `summary.md` and a router log
per cell. JSON includes every run, typed input size, body size, commit and
dirty state, toolchain, machine details, thread counts, pinning, tree
settings and run configuration. Markdown includes per-run and median tables.
When reporting results, include that metadata and the summary table. Compare
commits on the same machine in one session, retaining both direct scenarios.

## Comparing a token-aware routing change

Run these arms with the same requests, offered load, output length, model
and tokenizer version:

| Arm | Routing key | Router-side encoding |
|---|---|---|
| A | Current policy | None |
| B | Text-prefix hash | None |
| C | Token-prefix hash | Every request, cache off |
| D | Token-prefix hash | Cache on |

B versus A measures the effect of prefix routing; C versus B adds token
boundaries; D versus C measures the encoding cache. Compare D with A and B
to judge the complete change. The harness currently supplies A and the
direct baselines; the change under evaluation must add B–D.
