# KV-aware routing and trusted selection

`kv_aware` is an opt-in policy for regular, co-located, normal-dense group-0
attention with worker-owned Device KV. It combines verified prefix overlap
with router-local active request load. The existing `cache_aware` policy keeps
its behavior.

## Running the policy

Build the current branch and select the policy explicitly:

```bash
cargo build --locked --bin vllm-router
./target/debug/vllm-router --host 127.0.0.1 --port 3001 \
  --worker-urls grpc://127.0.0.1:50051 \
  --policy kv_aware --kv-cache-weight 0.5 \
  --kv-max-event-age-secs 30 --kv-max-sources 1024 --kv-max-blocks 65536
```

These are example service commands; the CPU verification below does not start
a real inference engine. Rebuild the Rust extension before selecting
`PolicyType.KvAware` through the Python API or `--policy kv_aware` through the
Python launcher. The Python entry uses the same default limits.

For gRPC workers, discovery uses Control.GetServerInfo, GetModelInfo and
GetKvEventSources. Chat requests reuse the token IDs already produced by
EngineFrontend.prepare, including across retries. DP-aware gRPC workers receive
the selected rank through the existing gRPC routing metadata.

HTTP workers must expose `/get_server_info` with `instance_id`, `model_id` (or
`model_path`) and `kv_block_size`, plus `/kv_event_sources` keyed by DP rank.
Each enabled ZMQ publisher must have its own endpoint. Wildcard bind hosts are
resolved against the worker origin and connect endpoints are validated. The
configured API key is used for HTTP discovery. An engine without these
endpoints remains eligible for load fallback.

HTTP `/v1/completions` requests with a single token-ID prompt can use exact
selection. Text-only HTTP requests use load fallback, preserving transparent
proxy behavior. The selected DP rank overrides a client-supplied
`X-data-parallel-rank`; request tokens are forwarded unchanged.

## Completeness, identity and lifecycle

Registration starts untrusted. A publisher-emitted `AllBlocksCleared` observed
after subscription is the supported completeness barrier. Merely receiving a
live event, replay delta, or sequence zero does not establish a complete view.
The router does not clear engine caches itself. Operators must arrange a valid
barrier before expecting exact selection.

Event decoding, normal-dense shape admission, rank binding, sequence state,
index publication and selection are fenced by one runtime lock. A dropped
advisory signal cannot leave stale data visible as exact. Gaps, rollbacks,
invalid batches, wrong ranks, disconnect errors and stale timestamps revoke
trust. Repeated sequence numbers do not refresh freshness. Recovery requires
a new valid clear barrier; replay/snapshot reconciliation is not implemented
by this runtime. Publisher and router clocks must be synchronized within the
configured freshness window; timestamps over one second into the future are
rejected.

Worker metadata binds each source to an engine instance, model, block size and
addressable DP rank. Duplicate instances/endpoints and duplicate canonical
ranks are rejected. Removing or replacing a worker revokes its discovery lease
and stops its subscriber tasks; late discovery responses and queued events
cannot reattach the removed generation. Unhealthy/circuit-open workers are
excluded from selection.

The supported deployment must be normal-dense full attention in group zero.
Shape admission does not discover an engine's complete model architecture.
Hybrid/linear attention, sliding windows, LoRA, cache salts, shared external
ownership and multimodal requests are outside this runtime contract. Such
request namespaces or observed event extensions use safe fallback. This mode
must not be used to claim complete hybrid KV reuse. P/D, EPD and IGW combinations
are rejected at configuration validation.

## Selection, load and observability

All healthy candidates must share a cache identity and have fresh, reconciled
sources. Otherwise the entire eligible pool uses deterministic least-load
fallback without querying untrusted residency. Exact selection credits only
contiguous Device prefixes of complete blocks; the final prompt token is not
credited. It blends uncached-prefix fraction and normalized active request
load using `cache_weight`. This is a routing heuristic, not a calibrated
latency model or a measured performance gain.

One load guard covers each KV attempt. Errors, retries, cancellation before
headers, non-streaming completion and stream termination release that attempt
once. HTTP streaming polls upstream with downstream demand and does not create
an unbounded forwarding queue. The existing gRPC producer holder handles
terminal and dropped-body cleanup.

Responses expose `x-router-kv-mode` (`exact` or `fallback`) and
`x-router-kv-reason`. Debug route receipts contain target/rank and candidate
scores, never prompt text or token IDs. Prometheus counters are
`router_kv_decisions_total{mode,reason}` and
`router_kv_ingestion_failures_total{reason}`; reason labels are bounded constants.

Source count is capped globally. `max_blocks` conservatively charges stored
block announcements, including aliases and duplicates, because retained index
metadata can outlive residency. Removals do not refund this historical budget.
At exhaustion the affected identity's index is discarded and its sources lose
trust; new clear barriers are required. The last source's removal releases the
identity and its budget. This bounds index growth, not the entire process heap
or transport buffers.

## Library component

`kv_index::routing` still exposes KvTrustRegistry and TrustedKvSelector for
other engine-owned integrations. That selector can discount storage tiers and
remote locality; the registered runtime admits only Device events. Callers of
the library must independently prove complete reconciliation and cache identity.
Client headers/hints must never establish trust.

## CPU verification

```bash
cargo +1.95.0 test --locked --test kv_aware_runtime \
  --test kv_trusted_selection --test agent_hint_request_hints
cargo +1.95.0 test --locked --lib --bin vllm-router
cargo +1.95.0 clippy --locked --lib --bin vllm-router \
  --test kv_aware_runtime --test kv_trusted_selection \
  --test agent_hint_request_hints -- -D warnings
cargo +1.95.0 fmt --all -- --check
```

The integration tests use TCP ZMQ, HTTP and tonic services, actual index
residency/eviction, DP metadata and cancellation. They establish software
behavior; real model reuse, engine acceptance, NPU/GPU performance, hybrid
aggregation and HA remain separate acceptance work.

## Real Ascend acceptance, 2026-10-15

The HTTP path was exercised with real Qwen3-8B weights on two Ascend 910B4-1
cards: two independent TP1/DP1 engines, BF16, eager execution, block size 128.
The router binary used Rust 1.95.0's **dev profile**. The engines used vLLM
`ced6857afa0ea7b2e3f0846a62e1394e90f15607` (`0.30.0+empty`) and the installed
Ascend wheel `0.1.dev1+g4ea839b81`. The base image was
`quay.nju.edu.cn/ascend/vllm-omni:v0.30.0`, with repository digest
`quay.io/ascend/vllm-omni@sha256:274d65369c912c14cc241f53c9c900d95e844b91652ca604cc71f07a9e1b3aaf`.

| Three paired rounds, 48 measured requests per policy | Round robin | KV-aware |
| --- | --- | --- |
| Native cached-token fraction | 53.61% | 98.96% |
| First generated token, median | 789.83 ms | 198.22 ms |
| First generated token, observed p95 | 2023.71 ms | 268.29 ms |
| Completed requests per second | 4.245 | 21.192 |

This is a fixed prefix-placement workload: 16 independent 7,243-token prompts,
one generated token, concurrency four. Both engines were cleared and given the
same alternating prefix placement before each policy arm; order was AB/BA/AB.
Warmup is excluded. All 48 KV-aware requests used exact mode; all 96 measured
outputs matched direct generation token IDs. The approximately 4.99x request
throughput applies to this workload, not general Agent task throughput.

The functional checks observed 7,168 native cached tokens only on the selected
second worker, zero credit after native clear, long-prompt unreconciled fallback,
stale fallback after a five-second window, and restoration after a new clear.
Three known-answer cases matched direct token IDs. Agent hints were exercised
with `binding_only`; calibrated admission, priority/deadline utility and physical
KV retention were not evaluated. Native gRPC, DP16, real worker restart, transport
gap injection, hybrid, P/D and HA remain untested on devices.

The pinned Python HTTP launcher lacks these discovery routes.
[`serve_native.py`](../examples/kv_aware_acceptance/serve_native.py) exposes
configuration read from its live engine state, restricted to dense Qwen3/DP1.
It leaves generation, cache lookup and native ZMQ publishing unchanged.
This is an acceptance adapter, not a general production discovery plugin.
[`accept_native.py`](../examples/kv_aware_acceptance/accept_native.py) records
HTTP responses, native counters and original event bytes.
The [machine-readable receipt](../examples/kv_aware_acceptance/results-20261015.json)
contains per-request/per-arm measurements, all actual model-file SHA256 values,
package versions and script hashes. Model download revision was not recorded;
actual files are fixed by that manifest. Original logs and event bytes are
retained in the task archive, not embedded in this repository.

The following reproduces the measured setup using the existing image, wheel,
model directory and Rust toolchain. It was not rebuilt from an empty image.
Choose currently idle cards and unused ports before executing; the numbers below
are the actual task allocation, not an ongoing resource reservation.

```bash
mkdir -p /data2/router-kv-agent-real-20261015
cp examples/kv_aware_acceptance/serve_native.py /data2/router-kv-agent-real-20261015/
cp examples/kv_aware_acceptance/accept_native.py /data2/router-kv-agent-real-20261015/
for task_card in 2 3; do
  docker run -d --name router-kv-real-npu${task_card}-20261015 \
    --label router.task=kv-agent-real-20261015 --network host \
    --cpus 8 --memory 32g --shm-size 8g \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /data2/router-tokenizer-cache-20261011:/work:ro \
    -v /data2/models/Qwen/Qwen3-8B:/models/qwen3-8b:ro \
    -v /data2/router-kv-agent-real-20261015:/experiment \
    --device /dev/davinci${task_card}:/dev/davinci${task_card}:rwm \
    --device /dev/davinci_manager:/dev/davinci_manager:rwm \
    --device /dev/devmm_svm:/dev/devmm_svm:rwm \
    --device /dev/hisi_hdc:/dev/hisi_hdc:rwm \
    -e VLLM_PLUGINS=ascend -e ASCEND_RT_VISIBLE_DEVICES=0 \
    -e OMP_NUM_THREADS=4 -e TOKENIZERS_PARALLELISM=false \
    quay.nju.edu.cn/ascend/vllm-omni:v0.30.0 sleep infinity
  docker exec router-kv-real-npu${task_card}-20261015 python3 -m pip install \
    --no-deps --force-reinstall \
    /work/wheels/vllm_ascend-0.1.dev1+g4ea839b81-cp312-cp312-linux_aarch64.whl
done

# Run one command per engine in separate terminals; task_index is 0 then 1.
task_index=0
task_card=$((task_index + 2))
docker exec -e VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 \
  -e VLLM_SERVER_DEV_MODE=1 router-kv-real-npu${task_card}-20261015 \
  python3 /experiment/serve_native.py serve /models/qwen3-8b \
  --host 127.0.0.1 --port $((28610 + task_index)) \
  --served-model-name router-qwen3-8b --dtype bfloat16 --enforce-eager \
  --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.5 \
  --disable-uvicorn-access-log --enable-prefix-caching --enable-scale-out \
  --kv-events-config "{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:$((28620 + task_index))\",\"replay_endpoint\":\"tcp://*:$((28630 + task_index))\",\"topic\":\"router-kv-real\",\"hwm\":100000}" \
  --prefix-caching-hash-algo sha256_cbor

unset CARGO_TARGET_DIR
export PYO3_PYTHON=/work/.venv-render/bin/python
cargo +1.95.0 build --locked --bin vllm-router \
  --target-dir /work/runs/20261015-kv-agent/target-isolated

# Separate terminals, same binary and workers for both policies.
/work/runs/20261015-kv-agent/target-isolated/debug/vllm-router \
  --host 127.0.0.1 --port 28640 \
  --worker-urls http://127.0.0.1:28610 http://127.0.0.1:28611 \
  --policy kv_aware --prometheus-port 28650 --log-level debug \
  --retry-max-retries 1 --kv-max-event-age-secs 300 \
  --enable-program-scheduling \
  --program-scheduling-config-json '{"program_scheduling_enable_key":"auto","binding_only":true}'
/work/runs/20261015-kv-agent/target-isolated/debug/vllm-router \
  --host 127.0.0.1 --port 28641 \
  --worker-urls http://127.0.0.1:28610 http://127.0.0.1:28611 \
  --policy round_robin --prometheus-port 28651 --log-level debug --retry-max-retries 1

docker exec router-kv-real-npu2-20261015 python3 /experiment/accept_native.py \
  --model-path /models/qwen3-8b \
  --workers http://127.0.0.1:28610 http://127.0.0.1:28611 \
  --publishers tcp://127.0.0.1:28620 tcp://127.0.0.1:28621 \
  --kv-router http://127.0.0.1:28640 --baseline-router http://127.0.0.1:28641 \
  --output /experiment/paired-new --rounds 3 --contexts 16 --concurrency 4
```

For the independent recovery check, start a fresh KV router on port 28642 with
metrics port 28652 and `--kv-max-event-age-secs 5`; retain the same other options.
Run the acceptance command against it with a fresh output directory and
`--rounds 0 --expect-unreconciled --verify-stale-after-seconds 6`. Do this after
the benchmark, because it clears the same native engines.

Device process auditing covered 85 samples through completed acceptance, with
no other NPU processes on the selected cards. Another task was observed on card3
after both result files were complete. Only task-owned router processes and
containers were stopped; selected cards had no running processes at cleanup.
This was not whole-host CPU isolation, a confidence-interval study, full model
precision certification or an Agent benchmark.
