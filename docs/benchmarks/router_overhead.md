# Router CPU fast-path baseline

Two tools measure what the router itself costs on the request path, so
that changes to that path (tokenizer integration, routing-key derivation,
policy work) can be compared against a fixed "before".

| Tool | What it measures | In CI |
|---|---|---|
| `cargo bench --bench routing_input` | Per-call cost of deriving the routing key and selecting a worker, in process, with criterion | built by the clippy job (`--all-targets`); never run |
| `cargo test --release --test router_overhead_bench -- --ignored --nocapture` | End-to-end router-added latency, throughput, load spread across workers, router CPU and peak RSS with mock workers | built by the integration-test and clippy jobs; ignored, never run |

Both take their inputs from `tests/common/bench_corpus.rs`: seeded,
offline, byte-identical on every machine. `tests/bench_corpus_test.rs`
keeps the corpus and the bench mock worker building and deterministic in CI.

## What these numbers can and cannot say

- They measure router **cost**. The mock workers answer instantly and keep
  no state, so no row here shows a routing **benefit** (KV-cache hits,
  time to first token). That needs real workers with prefix caching on.
- Client-observed latency includes the client, the loopback network and the
  mock worker. The `direct_mock` scenario drives a mock worker with no
  router in the path and is the floor to read the other rows against.
- Means subtract; percentiles do not. The table reports
  `mean - direct_mock mean`, and lists the router and direct percentiles
  side by side. A "router P99" obtained by subtracting two P99 values is
  not a meaningful quantity and is not printed.
- Router CPU and RSS are per process, read with `wait4(2)` for the router
  child only, after it has been stopped. `ru_maxrss` is bytes on macOS and
  kibibytes on Linux; the harness normalizes to bytes. `router cpu ms/1k
  req` is the router's CPU over its whole life divided by every request it
  served in the cell (warmup and measurement); startup is a small fixed
  cost included in it, noticeable only in low-throughput cells.
- The cache-aware tree is never evicted inside a cell: every router gets
  `--eviction-interval 3600` and starts empty. For corpora that do not
  repeat (`cold`, `long_shared_prefix`, `short_shared_prefix`), each
  `cache_aware` request adds to the tree, so its CPU and RSS in those rows
  include tree growth, which grows with requests times prompt size. That is
  what the policy does between eviction passes in production, stated here
  so it is not mistaken for per-request parsing cost.
- The client and the mock workers run on one 8-thread runtime inside the
  harness; the router is a separate process with its default thread count.
  On a machine without that many spare cores, router rows include CPU
  contention the `direct_mock` row does not. The report records both thread
  counts and the harness's own CPU per cell.
- `hotspot` is the largest worker's share of the generation requests
  divided by the even share `1 / workers`: `1.0` is a perfectly even
  spread, `workers` means every request landed on one worker. It shows
  whether a routing policy concentrates load; it says nothing about
  whether that concentration was useful.
- Run-to-run noise is real. With `VLLM_ROUTER_BENCH_REPEATS` above one the
  summary reports the median per cell and the spread of the mean
  (`(max - min) / median`); differences inside that spread are not
  findings.

## Comparing routing designs: four arms

When evaluating a token-aware routing change, run four arms on the same
request set, at the same offered load and output length, with the same
model and tokenizer version:

| Arm | Routing key | Router-side encoding |
|---|---|---|
| A | the current policy | none |
| B | text-prefix hash | none |
| C | token-prefix hash | every request, cache off |
| D | token-prefix hash | cache on |

B against A isolates prefix routing itself, C against B the cost of token
boundaries, D against C the cache, and only D against A and B decides
whether the whole change is worth it. D faster than C but slower than A or
B only shows that the cache offsets a cost the change introduced.

Today the harness ships arm A (`completions_off`, `completions_rendezvous`,
`chat_off`) and the floor (`direct_mock`); arms B to D are added by the
change under evaluation.

## Corpus

| Kind | Requests | Question it answers |
|---|---|---|
| `hot64` | 64 fixed prompts, cycled | the upper bound of an exact-match cache; every request after the first 64 repeats one |
| `long_shared_prefix` | all but the last 64 bytes shared, unique tail | same prefix, varying suffix: an exact-match cache misses every time while a prefix router sees one prefix |
| `short_shared_prefix` | shared prefix (at most 600 bytes, at most half the prompt) then a unique tail | many requests sharing one system prompt: does the policy create a hotspot |
| `cold` | unique marker at the start of every prompt | no repeats and no shared prefix: the pure added cost |
| `mixed90` | 90% `hot64`, 10% `cold` | a mostly-warm workload |

Text that is not meant to be shared comes from a pool of 256 distinct
seeded bodies. Routing-key code is content dependent (substring search,
hashing, parsing), and one repeated body lets the CPU's branch predictors
learn it: with a single body, `rendezvous_hash` selection on 16 KiB prompts
measured about three times cheaper than on distinct prompts.

Sizes: 200 B, 2 KiB, 16 KiB of ASCII text. Requests are non-streaming
`/v1/completions` (text prompt) or `/v1/chat/completions` (one user
message, no session id) with `max_tokens: 16`.

## Criterion groups (`benches/routing_input.rs`)

- `routing_key/extract_text_for_routing/{completion,chat}/{size}`: the one
  call per request that produces today's routing text.
- `policy/{cache_aware,rendezvous_hash}/{size}`: `select_worker` over four
  healthy workers, tree warmed with the hot set.
- `cache_aware_key_format/{raw_text,one_char_per_token,digit_tagged}/{tokens}`:
  `Tree::insert` and `Tree::prefix_match_with_counts` for three candidate
  key encodings at 128 to 8192 tokens. Input for the discussion of
  token-id routing keys; not a router code path.

```text
cargo bench --bench routing_input
cargo bench --bench routing_input -- --save-baseline main      # on the base commit
cargo bench --bench routing_input -- --baseline main           # on the change
```

## Overhead harness (`tests/router_overhead_bench.rs`)

One `vllm-router` process per scenario cell (the real binary, built by
cargo for the test), four in-process mock workers with zero delay, a
client at concurrency 1 and 64, 5 s warmup and 20 s measurement per cell.
The client is closed loop by default; `VLLM_ROUTER_BENCH_RATE` caps the
offered load so arms can be compared at equal load (each task then paces
itself to `rate / concurrency`, still with one request in flight).

Scenarios:

| Scenario | Router policy | Route |
|---|---|---|
| `direct_mock` | none (client to mock worker) | `/v1/completions` |
| `completions_off` | `cache_aware` (default) | `/v1/completions` |
| `completions_rendezvous` | `rendezvous_hash` | `/v1/completions` |
| `chat_off` | `cache_aware` | `/v1/chat/completions` |

```text
cargo test --release --test router_overhead_bench -- --ignored --nocapture
```

Knobs (environment variables): `VLLM_ROUTER_BENCH_SCENARIOS`,
`VLLM_ROUTER_BENCH_SIZES`, `VLLM_ROUTER_BENCH_CORPORA`,
`VLLM_ROUTER_BENCH_CONCURRENCY`, `VLLM_ROUTER_BENCH_RATE`,
`VLLM_ROUTER_BENCH_REPEATS`, `VLLM_ROUTER_BENCH_WARMUP_SECS`,
`VLLM_ROUTER_BENCH_MEASURE_SECS`, `VLLM_ROUTER_BENCH_WORKERS`,
`VLLM_ROUTER_BENCH_ROUTER_CPUS`, `VLLM_ROUTER_BENCH_OUT_DIR`. A quick
smoke:

```text
VLLM_ROUTER_BENCH_SCENARIOS=direct_mock,completions_off \
VLLM_ROUTER_BENCH_SIZES=2048 VLLM_ROUTER_BENCH_CONCURRENCY=4 \
VLLM_ROUTER_BENCH_WARMUP_SECS=1 VLLM_ROUTER_BENCH_MEASURE_SECS=3 \
cargo test --release --test router_overhead_bench -- --ignored --nocapture
```

Output: `target/router_overhead/summary.json` (every run, plus commit and
whether tracked files were modified, rustc version, OS, architecture, CPU
count, profile, thread counts, router pinning, tree settings, repeats and
rate), `summary.md` (a per-run table and a median-over-runs table) and one
`*.router.log` per router run. With repeats, the scenario order is rotated
every repeat. Request bodies are built the same way for every corpus,
outside the timed region.

To limit only the router to some CPUs on Linux, set
`VLLM_ROUTER_BENCH_ROUTER_CPUS=0,1`: the harness runs the router under
`taskset -c 0,1` and leaves the client and the mock workers unpinned.
Pinning the whole `cargo test` process instead would put the client on the
same cores and measure a client-bound system. `TOKIO_WORKER_THREADS` only
sets the router's async worker count and is not a CPU limit; the report
records it as `router_worker_threads`.

Both tools are also reachable through `scripts/run_benchmarks.py`
(`--bench routing_input`, `--router-overhead`).

## Reporting

Report the median-over-runs table the harness prints and this header:

```text
commit: <sha> (dirty: <bool>)   rustc: <version>   profile: release
os/arch/cpus: <os> <arch> <n>   router cpus: <none | taskset list>   router threads: <default | n>
corpus seed: 0x244   warmup/measure: <w> s / <m> s   workers: 4   repeats: <n>   rate: <closed loop | n rps>
```

When comparing a change against `main`, run both on the same machine in
the same session with nothing else running, keep `direct_mock` in both
runs, use at least three repeats, and treat differences inside the
reported spread as no change.
