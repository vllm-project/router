# Backend calibration tools

Vendored from [openJiuwen-ai/agent-infer](https://github.com/openJiuwen-ai/agent-infer) for vllm-router agent-aware perf L1 (`prepare_l1.py`). Run from the **router repository root**: `python -m tools.calibration.calibrate_prefill …`.

These standalone commands measure the cold-prefill cost model and warm-prefix decode throughput surface against an
already running vLLM server. They do not run during AgentInfer startup and do not modify Scheduling or native vLLM
behavior.

## Supported deployment scope

The initial implementation reads stable metadata from `/version`, `/v1/models`, and `/metrics`. Dynamic Prometheus
counter values are measurement evidence and never enter `metadata_hash`. For homogeneous vLLM internal DP, both
commands automatically select the first EngineCore reported by `/metrics`. Multi-DP requests are pinned with
`X-data-parallel-rank`; single-DP requests omit that header. The coefficients are reusable by every listed engine.

Each DP rank remains a separate Router instance for request placement, KV capacity, and cache affinity. Shared EP makes
decode performance depend on the other DP ranks in the EP group. The initial decode surface does not model this
coupling and can therefore overestimate group-batch gain. This may leave optimization headroom unused but does not add
a more aggressive rejection term that would underestimate batching benefit. A later group-level model can replace it
without changing the profile envelope.

The service should be otherwise idle during calibration. `/metrics` must expose numeric `engine` labels and homogeneous
`vllm:cache_config_info` values. The endpoint must support token-ID prompts, `cache_salt`, streamed usage, and
`return_token_ids`. If the server sets `--max-num-seqs`, it must be no smaller than the largest requested
`--batch-sizes` value; otherwise vLLM admission limits the measured batch and invalidates the decode surface.
The deterministic exact-length prompts use token IDs through 4112, so the served model must have a vocabulary containing
those IDs. An opaque HTTP 400 response may indicate that the selected model's vocabulary is too small.

## Commands

Run both commands from the repository root against the same deployment. If `--output-dir` is omitted, both stages use
`tools/calibration/results` and assemble the final Profile there:

```bash
python -m tools.calibration.calibrate_prefill \
  --base-url http://127.0.0.1:8000 \
  --model MODEL_NAME \
  --prompt-lengths 1024,2048,4096,8192,16384,32768,65536 \
  --repeats 2

python -m tools.calibration.calibrate_decode \
  --base-url http://127.0.0.1:8000 \
  --model MODEL_NAME \
  --batch-sizes 1,2,4,8,16,32 \
  --context-lengths 1024,8192,16384,32768 \
  --output-tokens 128 \
  --repeats 1
```

Prefill requests use unique cache salts and generate one token. Decode cases warm every private prefix in sequence,
wait briefly for EngineCore to publish the last completed request's cache blocks, then measure only the common interval
in which all requests are decoding. Sequential warmup avoids cold-prefill admission and preemption within the warmup
batch from leaving only a subset of the intended private prefixes reusable. `total_context_tokens` is the logical sum
of every request context and does not subtract shared physical prefix blocks.

## Common parameters

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--base-url` | `http://127.0.0.1:8000` | Base URL of the running vLLM server. The tools append `/version`, `/v1/models`, `/metrics`, and `/v1/completions`. |
| `--model` | Required | Served model ID accepted by `/v1/completions`. It must match an `id` returned by `/v1/models`. |
| `--output-dir` | `tools/calibration/results` | Shared root for Metadata, prefill artifacts, decode artifacts, and the final Profile. Use a different directory for each deployment or independent calibration run. |
| `--data-parallel-rank` | Auto | Optional internal-DP EngineCore rank. Auto selects the first engine from `/metrics`; the rank header is sent only when multiple engines exist. |
| `--timeout` | `3600` seconds | HTTP timeout for one request. Long-context prefill and large decode cases may require a high value. |
| `--overwrite` | Disabled | Replace only the current command's stage directory. It does not delete the other completed stage, but final assembly still requires matching Metadata and DP scope. |

## Prefill parameters

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--prompt-lengths` | `1024,2048,4096,8192,16384,32768,65536` | Exact cold-prompt Token counts. At least three distinct lengths are required for the quadratic fit; production calibration should cover the workload's supported context range. |
| `--repeats` | `2` | Cold requests measured at each prompt length. Increase this when backend noise is visible in the raw samples. |
| `--shuffle-seed` | `260901` | Seed used only to shuffle the prompt-length/repeat order, reducing systematic warmup or clock-order bias while preserving reproducibility of the case order. |

## Decode parameters

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--batch-sizes` | `1,2,4,8,16,32` | Concurrent request counts used to sample batch gain and saturation. Values should cover both the low-batch region and the expected throughput wall. |
| `--context-lengths` | `1024,8192,16384,32768` | Exact logical Context length per request. Combined with batch size, this produces `total_context_tokens` for the fitted surface. |
| `--output-tokens` | `128` | Tokens generated per measurement request. This normally provides a stable common Decode interval without dominating calibration time. |
| `--warmup-settle-seconds` | `1.0` | Delay after every case's sequential prefix warmup. It prevents the measured batch from racing EngineCore's completed-request KV release and accidentally mixing cold Prefill into the Decode sample. |
| `--repeats` | `1` | Measurements for every valid batch/context pair. The default surface already contains many independently varying points; increase this for a higher-confidence Profile. |
| `--max-kv-fraction` | `0.9` | Maximum fraction of the observed per-rank HBM KV capacity that a candidate case may occupy. Cases above the threshold are skipped before requests are sent. |
| `--shuffle-seed` | `260901` | Seed used to shuffle valid batch/context/repeat cases and reduce execution-order bias. |

`kv-capacity-tokens` is intentionally not a user parameter. The tool reads `num_gpu_blocks` and `block_size` from the
common `vllm:cache_config_info` exposed by `/metrics`. For internal DP, that endpoint repeats the deployment-wide block
count under every engine label, so the tool divides `num_gpu_blocks * block_size` by the homogeneous `engine_count` to
obtain one EngineCore rank's capacity. It records the result as `observed_kv_capacity_tokens` and applies
`--max-kv-fraction`. Requiring a second manually entered capacity would duplicate an authoritative backend value and
could silently filter cases with a stale or TP/DP-confused number. The command fails explicitly if the backend does not
expose the required fields.

The default `2/1` Prefill/Decode repeat counts with 128 Decode tokens target a complete run in roughly ten minutes on
the reference deployment. Use a reduced matrix with `1/1` repeats for functional smoke validation. Use `3/2` repeats
and 256 or 512 Decode tokens for a higher-confidence Profile when additional runtime is acceptable.

## Output

Each request is flushed to its stage `requests.jsonl` immediately. Decode also updates `points.partial.json` after each
case. Stage results are written to `prefill/result.json` and `decode/result.json`. After both stages finish with the same
Metadata Hash, the tools atomically create `calibration.json` with exactly three top-level fields:

```json
{
  "metadata": {},
  "metadata_hash": "sha256:<64 lowercase hex characters>",
  "calibration": {
    "scope": {
      "kind": "homogeneous_internal_dp",
      "calibrated_engine": 0,
      "reusable_engine_ids": [0, 1]
    },
    "prefill": {},
    "decode": {}
  }
}
```

The first version's Hash describes only facts observable from the three HTTP endpoints. It is not a complete hardware
or launch-configuration identity. A future backend middleware may calculate and return a full deployment Hash without
exposing raw configuration or credentials.

## Smoke validation

The following matrices validate request generation, endpoint discovery, DP pinning, fitting, and profile assembly.
Their coefficients are not performance results.

```bash
python -m tools.calibration.calibrate_prefill \
  --base-url http://127.0.0.1:8000 \
  --model MODEL_NAME \
  --output-dir /tmp/agentinfer-calibration-smoke \
  --prompt-lengths 128,256,512 \
  --repeats 1

python -m tools.calibration.calibrate_decode \
  --base-url http://127.0.0.1:8000 \
  --model MODEL_NAME \
  --output-dir /tmp/agentinfer-calibration-smoke \
  --batch-sizes 1,2 \
  --context-lengths 128,256 \
  --output-tokens 64 \
  --repeats 1
```
