# KV-event routing for two independent text workers

`kv_aware` uses KV blocks reported by each worker to route a request to a worker
with its longest cached prefix. This initial implementation supports the native
`vllm-router` CLI, static Regular HTTP routing, and independently addressable
DP=1 workers with a compatible **Qwen3 Dense** tokenizer profile. It does not
require a worker patch. The finite acceptance matrix uses **Qwen/Qwen3-0.6B**
and immutable assets; compatibility checks do not establish that other model
sizes have been tested.

The supported inputs are a Completion string, a single Completion token-ID
array, and the verified Qwen3 text Chat template. Chat permits an optional
initial system message, alternating user/assistant string messages, and a final
user message. Generation prompting is enabled; `enable_thinking` may be true
or false. Tools, multimodal content, reasoning history, adapters, cache salt,
prompt truncation, batched prompts, and unknown input transformations do not
receive an approximate KV score. They use the policy's non-affinity fallback
while the original HTTP request bytes are forwarded on the KV path.
Internal typed calls and non-KV policies retain their existing behavior.

Only complete blocks count. Worker lifecycle changes, publisher discontinuities,
and cache-clear events invalidate ownership. A retained subscriber runtime is
owned by the Router. Missing positive evidence uses fair fallback among eligible
workers. This is a correctness feature, not a multi-GPU performance claim.

KV Events are asynchronous PUB/SUB observations, not a synchronized cache
snapshot. This slice does not replay missed history. Observed gaps or disconnects
purge ownership; a lost final event cannot be detected until a subsequent
sequence or connection event reveals the discontinuity. Affinity never changes
the request's inference semantics, and a selected worker may still need to
recompute an evicted prefix.

Publication-review validation and the original-candidate evidence boundary are
recorded in [kv_aware_review.md](kv_aware_review.md).

## Runtime contract

Use an isolated environment with vLLM **0.29.0**. Both workers must use the same
model/tokenizer assets, block size **16**, hash algorithm **sha256_cbor**, and
**PYTHONHASHSEED=0**. Set `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0` so events carry
full hash bytes. The Router reads local `tokenizer.json`, `tokenizer_config.json`,
and `config.json` from the same directory. It checks Qwen3 Dense model metadata
and tokenizer semantics, not a hard-coded model revision or tokenizer file hash.
The standard verified Chat template enables exact Chat affinity; an unknown
template retains Completion support but falls back without Chat affinity.
`--kv-model` is the workers' served model alias.

For reproducible **0.6B validation**, the harness pins the tokenizer profile to
the public Hugging Face revision `c1899de289a04d12100db370d81485cdf75e47ca` and
requires this `tokenizer.json` SHA-256:

```text
aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
```

The pinned public assets are available from
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca).
The Router does not download assets at startup. Matching local metadata does not
by itself establish the workers' configuration: validate the running workers too.
Keep the CPU oracle environment separate: its pinned Transformers dependency
must not overwrite vLLM's runtime dependencies.

These example commands are for a user-provisioned host and an explicitly
authorized GPU. Choose memory limits that fit two copies on that GPU. Launch
each worker as its own process; two URLs for one worker are not sufficient.
The validation script will check separate HTTP and EngineCore processes. On one
GPU, let the first worker finish initialization before starting the second so
their memory profiling does not overlap.

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 \
VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 VLLM_SERVER_DEV_MODE=1 VLLM_USE_RUST_FRONTEND=0 \
vllm serve Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --tokenizer-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --served-model-name Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8000 \
  --data-parallel-size 1 --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --gpu-memory-utilization 0.40 --max-model-len 4096 --enforce-eager \
  --enable-log-requests \
  --enable-prefix-caching --prefix-caching-hash-algo sha256_cbor --block-size 16 \
  --kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:5557","topic":"kv"}'
```

Launch the second process with HTTP port **8001** and event endpoint
**tcp://*:5558**; retain the other model/hash settings. Keep worker HTTP and
Router HTTP bound to **127.0.0.1**. Development mode is used only for the isolated
real-cache-clear test; do not expose those HTTP endpoints publicly.

In vLLM 0.29, `tcp://*:PORT` makes the PUB socket **bind**, whereas
`tcp://127.0.0.1:PORT` makes it **connect**. The Router SUB socket also connects,
so using fixed loopback publisher addresses here leaves neither side listening.
The worker wildcard KV bindings are only for an explicitly authorized test host
whose management-network/firewall rules block public access to ports 5557/5558.
Confirm that protection before launch; never expose these unauthenticated KV
ports publicly. Router endpoint mappings remain `tcp://127.0.0.1:5557` and
`tcp://127.0.0.1:5558`.
The harness mirrors the pinned publisher's bind/connect heuristic and rejects
connect-only TCP publishers for this direct-worker fixture, which has no broker.
That configuration check is not proof of a live listener or received events;
the first-route positive-score and backend-count checks remain mandatory.

Start the candidate native Router after both workers are ready:

```sh
env -u RUST_LOG /absolute/path/to/candidate/vllm-router \
  --host 127.0.0.1 --port 3001 --backend vllm \
  --worker-urls http://127.0.0.1:8000 http://127.0.0.1:8001 \
  --policy kv_aware --kv-model Qwen/Qwen3-0.6B \
  --kv-tokenizer-path /absolute/path/to/pinned/tokenizer.json \
  --kv-hash-algo sha256_cbor --kv-block-size 16 --kv-hash-seed 0 \
  --kv-events-topic-filter kv --kv-events-port 5557 \
  --kv-events-endpoint http://127.0.0.1:8000=tcp://127.0.0.1:5557 \
  --kv-events-endpoint http://127.0.0.1:8001=tcp://127.0.0.1:5558 \
  --log-level debug
```

Capture this process's stdout/stderr in a new task-owned log. Explicit endpoint
mapping takes precedence over `--kv-events-port`, which remains a host/port
fallback for unmapped workers. A shared endpoint and `base_port + dp_rank` are
not supported.

### Reproducible ModelScope validation snapshot

Download and hash-verify the fixed public acceptance snapshot without accessing
Hugging Face. The authorized root must already exist; the output directory must
be new and inside it:

```sh
python3 scripts/kv_qwen3_modelscope.py \
  --allowed-root /absolute/path/to/authorized/workspace \
  --output /absolute/path/to/authorized/workspace/qwen3-0.6b-validation
```

This standard-library helper verifies all eight required files against the
immutable ModelScope file listing and separately checks the tokenizer goldens.
It writes the download manifest consumed by the validator below. These pins
belong to the acceptance fixture, not the production Router's model allowlist.

The same 0.6B matrix also accepts the immutable ModelScope snapshot
`Qwen/Qwen3-0.6B` at commit `09b42cad3d112e832108974449ccb5e8e0f5b5d1` through
`validate --model-manifest /absolute/model/directory/download-manifest.json`.
This records **ModelScope weight provenance**, not equality with weights at the
Hugging Face commit. Its tokenizer files match the fixed tokenizer fixture.
Both workers must use the verified directory as their absolute `vllm serve`
positional model argument, retain `--served-model-name Qwen/Qwen3-0.6B`, and set
both `--revision` and `--tokenizer-revision` to that ModelScope commit. These flags
record the source pin; the manifest hashes verify the locally loaded artifacts.
Use that directory's `tokenizer.json` for the Router and validation arguments.

The download manifest must have `status: "verified"`, `source: "ModelScope"`,
`model`, `source_commit`, absolute `output`, and a `required_files` array covering
`config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`,
`merges.txt`, `vocab.json`, `LICENSE`, and `model.safetensors`. Each `files` entry
must contain `path`, `status: "verified"`, `source_commit`, `bytes`, `sha256`, and
the matching ModelScope `api_sha256`. The local `modelscope-api-files.json` must
match `source_listing_sha256`. Only these eight model files plus the three
download evidence files (`download-manifest.json`, `download-events.jsonl`, and
`modelscope-api-files.json`) may be present in the snapshot directory.

The harness rehashes all eight files before and after validation, checks the
four approved config/weight/tokenizer artifact hashes, and rejects changed
files, unexpected files, or a worker loading a different directory. Omitting
`--model-manifest` preserves the original pinned Hugging Face validation mode.

## Reproducible validation

Run the focused Rust tests, formatting, check, and Clippy against the actual
candidate checkout on Linux. Run the full applicable regression once the
candidate is stable. The local development policy may reserve `cargo test` and
`cargo build` for the user; in that case record them as **NOT RUN** until the
user supplies results. Do not substitute another checkout's build or historical
downstream GPU results.

The standard-library harness offers an offline `self-check` plus two GPU-host
commands. `python3 scripts/kv_aware_cuda_validate.py self-check` tests evidence
parsing without contacting any server; it does not establish CUDA correctness.
`build` compiles a
clean, committed candidate and records its commit, tree, command, toolchain,
architecture, native path and SHA-256. Run it only after tests pass. Use a fresh
evidence directory outside the source tree:

```sh
python3 scripts/kv_aware_cuda_validate.py build \
  --source /absolute/path/to/clean/candidate \
  --candidate FULL_CANDIDATE_COMMIT_SHA \
  --output /absolute/path/to/new/build-evidence
```

Start the Router using the resulting
`build-evidence/target/release/vllm-router`. The Linux GPU build is specific to
that host architecture; an ARM64 development artifact is not an x86_64 artifact.
After launch, run the finite acceptance matrix with the actual task-owned PIDs.
Use the vLLM environment's Python and the same Linux process namespace; the
harness needs its package metadata as well as access to the five `/proc` entries:

```sh
python3 scripts/kv_aware_cuda_validate.py validate \
  --source /absolute/path/to/clean/candidate \
  --candidate FULL_CANDIDATE_COMMIT_SHA \
  --native /absolute/path/to/build-evidence/target/release/vllm-router \
  --build-manifest /absolute/path/to/build-evidence/build.json \
  --tokenizer /absolute/path/to/pinned/tokenizer.json \
  --router-log /absolute/path/to/new/router.log \
  --worker0-log /absolute/path/to/worker0.log \
  --worker1-log /absolute/path/to/worker1.log \
  --router-pid ROUTER_PID \
  --worker0-pid HTTP0_PID --engine0-pid ENGINE0_PID \
  --worker1-pid HTTP1_PID --engine1-pid ENGINE1_PID \
  --allow-cache-reset \
  --output /absolute/path/to/new/cuda-evidence
```

This validation is deliberately sequential and requires exclusive use of the
two workers. For each input form it warms only W0 directly, then sends the first
Router request for that prefix; a fresh prefix verifies the reverse direction.
`/tokenize` does not warm the KV cache. PASS requires all of the following:

- Both workers return identical exact tokens; their token digest matches the
  Router's decision. The digest is SHA-256 of concatenated big-endian u32 IDs.
- The selected worker has a positive prefix score, and the other has zero.
- Worker request-counter deltas independently prove the actual backend.
- The first direction returns JSON; the reverse direction returns valid SSE.
- A fresh cold prefix has zero scores. A real worker cache reset removes the
  warmed ownership. vLLM queues its clear event until a scheduler step, so the
  harness sends a fresh, nonmatching one-token request directly to W0 after
  reset, verifies that backend, then tests the old prefix through the Router.
  Backend errors and an observed aborted stream release load.
- The live Router `/proc/PID/exe` hash matches the candidate build manifest;
  EngineCore PIDs have vLLM's DP=1 EngineCore process title and belong to the
  corresponding HTTP process. All five processes retain their original
  PID/start-time, command and executable identities.
- Each running worker's `/version` response reports exactly `0.29.0`. These
  responses are saved separately from the harness Python environment's package
  metadata. Missing endpoints, version mismatches and unreachable workers fail
  preflight; an installed local package cannot establish a worker's version.

The script never provisions hardware, uses SSH, starts/stops servers, injects
KV events, or pushes code. `--allow-cache-reset` authorizes resetting the owned
W0 prefix cache for the clear-event case. Without it that case is NOT RUN and
the overall result is INCOMPLETE. A missing counter, missing decision log,
ambiguous backend delta, token mismatch, or missing abort evidence cannot pass.
The default request counter is `vllm:request_success_total`; an override must
name an equivalent completed-request counter, never an approximate cache metric.

For the cancellation case, enable `--enable-log-requests` on both isolated
workers and redirect each HTTP process's stdout/stderr directly to its own log.
Provide both `--worker0-log` and `--worker1-log`, or neither. The harness checks
each log's device/inode against that HTTP PID's stdout/stderr, reads only bytes
after the probe's recorded offset, and matches the selected API-server PID and
the exact SSE response ID in vLLM's `Aborted request(s)` record. A single-prompt
Completion adds `-0` to the response ID; vLLM may append its eight-hex-character
internal suffix. Similar prefixes or another worker's log cannot establish abort.

PASS also requires a complete explicitly nonterminal SSE event, matching Router
and worker active counts before client shutdown, unchanged completed-request
counts at that point, and healthy zero loads afterwards with no natural
completion. Stock vLLM 0.29 can remove externally aborted requests before its
finished-request statistics are updated, so an abort counter need not increase.
When no worker logs are supplied, the strict matching-worker abort-counter
increase remains mandatory; idle state alone never passes. The case JSON keeps
the event, counters and timestamps, with bounded-run worker/Router log excerpts.

`summary.json` is the acceptance result. `worker_versions.json` retains both
live version responses, including failed preflight observations. Per-case JSON files and log excerpts
contain only this run's synthetic requests and observations. Keep full build
logs and machine metadata outside the public patch. Review the evidence before
sharing it. `model_source.json` distinguishes the actual model source commit and
artifact hashes from the tokenizer-profile fixture revision. Worker
restart/new generation, late events, sequence gaps, and
compatibility with the disabled feature require the corresponding deterministic
CPU tests; a real cache-clear case alone does not prove those behaviors.
