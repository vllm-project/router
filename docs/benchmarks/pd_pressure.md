# P/D pressure routing: per-PR stress and benchmark evidence

`scripts/benchmarks/pd_pressure/run.py` is a standard-library runner. It does not
start a router, a worker, a connector, or Program scheduling. CPU stress and live
benchmarks are different evidence: CPU tests establish routing/accounting
behavior; `vllm bench serve` can measure client-observed TTFT through the public
HTTP API. HTTP P/D is the primary validation path; gRPC P/D remains a
compatibility check. The same public-API benchmark can exercise either backend
transport when the corresponding deployment is explicitly configured.

The actual P/D connector, engine capabilities, and backend readiness must be
established separately. The runner does not infer them from an HTTP URL.

## Entry points by PR

| Stage | CPU stress filter | Current status |
|---|---|---|
| PR1 admission | Library `routers::http::pd_router::admission_tests::pd_cache_aware_pr1_stress` | Shared accounting check. |
| PR1 HTTP P/D | Target `test_pd_routing`, exact filter `test_pd_routing::http_pd_cache_aware_pr1_stress` | Primary routing stress. |
| PR1 gRPC P/D | Target `grpc_pd_routing`, exact filter `grpc_pd_cache_aware_pr1_stress` | Compatibility stress. |
| PR2 | Not registered | Exits 3 with `not_implemented`; no success claim. |
| PR3 | Not registered | Exits 3 with `not_implemented`; no success claim. |
| PR4 | Not registered | Exits 3 with `not_implemented`; no success claim. |

Each later PR must implement its stress test, register its real filter in
`STAGES`, update the standard-library tests, and document the new evidence before
its entry point can succeed. There is no arbitrary filter override that would
silently turn an unimplemented stage into a passing stage.

All commands below run from the repository root in Linux/WSL. Replace the
explicit paths and comparison commit with the artifacts actually under test.
Every `--out-dir` must be new: existing evidence is never overwritten. Keep
evidence outside the checkout/build target so cleanup does not remove it.

### PR1 CPU stress

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN=1.95.0
export CARGO_TARGET_DIR="$HOME/.cache/router-kv-pressure-pd-target"
export CARGO_BUILD_JOBS=2

python3 scripts/benchmarks/pd_pressure/run.py stress \
  --stage 1 --base-ref BASE_COMMIT_SHA \
  --cargo "$HOME/.cargo/bin/cargo" --profile release \
  --repeat 2 --stress-rounds 24 --stress-concurrency 16 --timeout 3600 \
  --out-dir /absolute/path/to/benchmark-artifacts/pr1-stress-run1
```

Each repeat runs admission, HTTP P/D, and gRPC P/D in that order. All three
suites must pass; a missing filter resulting in zero tests is a failure.
The actual Cargo commands include `--locked
--release --features cpu,ready --message-format=json`, the appropriate `--lib`
or `--test test_pd_routing` / `--test grpc_pd_routing` target, and its exact test
filter from the table above. Libtest
receives `--ignored --exact --nocapture --test-threads=1`. Internal concurrency
remains available; libtest runs one selected test at a time. Two repetitions are
the default. `--stress-rounds` and `--stress-concurrency` override the corresponding
`ROUTER_PD_STRESS_ROUNDS` and `ROUTER_PD_STRESS_CONCURRENCY` environment values
(defaults 24 and 16).

All three suites emit `PD_PRESSURE_STRESS_RESULT` JSON, which is retained separately
and in the manifest. Pair-admission stress checks concurrent reservations,
cleanup and sentinel preservation. HTTP P/D mock stress exercises candidate
tokenization, P/D prefix hits, per-request handoff parameters, completion counts,
and zero final occupancy. The gRPC P/D mock stress retains compatibility checks
for one P and one D dispatch per request, KV handoff parameters, token consistency,
completion counts, and zero final occupancy. The result kinds are `pair_admission`,
`http_pd`, and `grpc_pd`; HTTP and gRPC records share the request/completion/handoff
counts and occupancy fields. Recorded rounds and concurrency must match the
requested values; admission `attempts` and HTTP/gRPC `requests` must each equal
`rounds * concurrency` exactly. The runner rejects a missing result, failed
invariants, zero passed tests, or a missing test-executable hash. Cargo's JSON
build messages provide the executable path; libtest's raw output and summaries
are retained in the same log. Compilation errors remain failures; the runner
does not install dependencies or rewrite a lockfile.

These are CPU mock stresses. Their wall time, pair-admission time, and mock
completion time are **not TTFT**, do not measure GPU prefill/decode, and do not
prove a capacity-policy performance improvement.

For an initial functional check, use `--profile test`; this omits `--release`
and uses Cargo's test profile. Optional low-memory build settings are
`CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0`.
The selected profile and these environment values are recorded. Test-profile
timings are functional evidence only and must not be used to assess performance.

Use separate target directories for incompatible branches/toolchains rather
than overwriting a retained reference binary. The global Cargo download cache
may be reused. A 32-core machine with only 8 GiB assigned to WSL should not use
32 concurrent compiler jobs without a separate memory check.

For later stages, the same command with `--stage 2`, `3`, or `4` currently
records `not_implemented` and exits 3. That is an incomplete validation result,
not a skipped success.

### Probe the installed vLLM CLI

```bash
python3 scripts/benchmarks/pd_pressure/run.py probe \
  --base-ref BASE_COMMIT_SHA \
  --vllm /absolute/path/to/vllm \
  --out-dir /absolute/path/to/benchmark-artifacts/cli-probe-run1
```

This records `vllm --version`, `vllm bench --help`, and the `serve` and `forward`
help output. It starts no model or server. Exit 3 means at least one requested
capability is unavailable; the manifest identifies each capability separately.
An executable that cannot start is an error, not evidence that a benchmark ran.
Grouped help is expanded with `--help=all` when the CLI advertises that option.

A local inspection on 2026-11-09 found vLLM **0.31.0** with `bench serve` but no
`bench forward`. The installed binary, rather than the latest documentation,
decides availability. No compatible vLLM version is installed automatically.

### Explicit paired TTFT runs through HTTP P/D

First start and validate baseline and candidate routers and their homogeneous
HTTP P/D backends outside this runner. Keep Program scheduling disabled. Specify the
actual connector; unknown connector behavior is not a validated P/D setup.
Both router URLs must support `/v1/chat/completions`; supply router addresses,
not individual worker addresses. gRPC P/D compatibility measurements can use
the same command with a separately validated gRPC deployment. Record backend
transport in deployment evidence; the runner does not detect it from the URL.

```bash
python3 scripts/benchmarks/pd_pressure/run.py serve \
  --stage 1 --base-ref BASE_COMMIT_SHA \
  --vllm /absolute/path/to/vllm --model /absolute/path/to/model \
  --baseline-url http://127.0.0.1:30001 \
  --candidate-url http://127.0.0.1:30002 \
  --baseline-binary /absolute/path/to/baseline/vllm-router \
  --candidate-binary /absolute/path/to/candidate/vllm-router \
  --input-len 1024 --output-len 64 --num-prompts 32 \
  --request-rate 4 --seed 17 --repeat 3 --timeout 1800 \
  --out-dir /absolute/path/to/benchmark-artifacts/pr1-serve-run1
```

This explicitly executes `vllm bench serve` with the `openai-chat` backend,
random token-length inputs, the public chat endpoint, and `--save-result --save-detailed`.
It requests p50/p95/p99 for TTFT, TPOT, ITL, and end-to-end latency explicitly.
Each pair uses the same seed, requested input/output lengths, prompt count, and
arrival-rate parameters. Baseline/candidate execution order alternates between
pairs. Every CLI invocation and its raw output are retained. `completed` must
equal the requested prompt count in each saved result; exit 0 alone is not
sufficient.

Binary hashes are computed from the files supplied by the caller. The runner
does **not** attest that a remote URL is running that file: preserve deployment
commands, engine/router startup logs, connector settings, model revisions,
hardware topology, and readiness evidence alongside the run. Avoid concurrent
baseline and candidate engine workloads competing for the same GPU. Establish
and record comparable warm-up/cold-cache conditions yourself; the runner never
flushes caches or restarts workers.

These small defaults exercise the measurement pipeline. They do not guarantee
KV pressure, a particular prefix-hit rate, controlled output termination, or
coverage of the two-long-versus-three-short scenario. A complete capacity study
must add fixed workload traces, pressure telemetry, candidate-selection counts,
multiple pressure regimes, length groups, and enough repetitions for confidence
intervals. PR1's unchanged behavior is a regression/overhead comparison, not a
claim that an unimplemented capacity override improves TTFT.

The status `completed_unassessed` means the paired runs completed and results
exist. It is deliberately not a performance acceptance decision. Compare the
same TTFT statistic on each arm, retain absolute milliseconds and per-run raw
values, and predeclare the allowed regression before evaluation.

### Optional model-forward diagnostic

```bash
python3 scripts/benchmarks/pd_pressure/run.py forward \
  --stage 1 --base-ref BASE_COMMIT_SHA \
  --vllm /absolute/path/to/vllm --model /absolute/path/to/model \
  --out-dir /absolute/path/to/benchmark-artifacts/pr1-forward-run1
```

This action probes `bench forward --help` before doing anything GPU-related. An
unsupported subcommand or incompatible interface produces `unsupported`, exit
3, and saved help/error output. A supported invocation explicitly runs the
installed CLI's forward defaults for the supplied model; inspect the saved help
to understand those defaults.

Forward benchmarks bypass the serving scheduler and do not traverse the router.
They cannot measure router TTFT, P/D transfer, candidate selection, queueing, or
end-to-end benefits. Use them only for model/GPU compute diagnosis. Do not
substitute `bench throughput` or `latency` and label it as a forward result.

## Evidence layout and exit codes

Each new artifact directory contains `manifest.json`, ordered raw command logs,
a source patch log, version output, resolved base/head commits and dirty status,
the runner/tool hash, and (when available) Cargo test-executable hashes. Serve
runs also contain per-arm `metrics.json` and the caller-supplied binary hashes.
The manifest records argv without shell interpretation, working directory,
start/end timestamps, elapsed seconds, command exit codes, and timeouts.
The `machine` object records CPU model, visible logical CPU count, and memory
bytes using standard-library OS observations. Linux uses `/proc/cpuinfo`, CPU
affinity, and `sysconf`; memory describes the exposed machine/VM memory, not a
container cgroup limit. Unavailable fields are `null` (including CPU model and
memory on Windows); values are not inferred. `test_binaries` retains its existing
path-to-SHA-256 format, while `test_binary_metadata` adds each file's `size_bytes`
and `mtime_ns` (Unix epoch nanoseconds) alongside the hash.
Git metadata and patch commands use invocation-local `core.autocrlf=true` so
Windows checkout line endings do not appear as full-repository changes under
WSL; no global Git setting is changed. Unimplemented stages record a status and
the supplied parameters without executing provenance or benchmark commands.
Required provenance commands must succeed. The patch contains tracked changes;
untracked files under `src`, `tests`, this runner's directory, and this document
(plus root Cargo/build files) are copied into `source-snapshot` with SHA-256
hashes. Other untracked files are listed in Git status but are not archived.

Only non-secret build environment settings are recorded. URLs with embedded
credentials or query strings are rejected. This initial runner does not accept
API keys; use a controlled benchmark deployment that does not require them.

| Exit | Meaning |
|---|---|
| 0 | CPU stress passed, both probe capabilities available, diagnostic completed, or serve completed unassessed; inspect `status`. |
| 1 | Execution, compilation, zero-test, missing artifact, incomplete request, or evidence error. |
| 2 | Invalid command-line arguments or attempted reuse of an evidence directory. |
| 3 | Stage not implemented or installed CLI capability unavailable. |

Run the runner's own tests without any GPU/model/Cargo build:

```bash
python3 -B -m unittest discover -s scripts/benchmarks/pd_pressure/tests -v
```

The tests cover all three PR1 suites and their default two-repeat orchestration,
tiny sample output, command construction, invalid parameters,
zero/ignored tests, artifact extraction, evidence preservation, unsupported
forward handling, and unimplemented stages.

References:

- [vLLM benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/)
- [vLLM forward benchmark](https://docs.vllm.ai/en/latest/cli/bench/forward/)
