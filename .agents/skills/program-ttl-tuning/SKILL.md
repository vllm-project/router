---
name: program-ttl-tuning
description: Ordered procedure for tuning Program scheduling and Progress-TTL on vLLM Router (PR #286), portable across model, accelerator, node count, and agent workload. Use it to establish a pressure point, calibrate capacity and cost coefficients, then tune admission, TTL, and placement in order. Covers enabling the feature on a new deployment, re-tuning after a model or workload change, and diagnosing a stalled token cache-hit ratio, a TTFT tail, or job timeouts. Complements docs/program_scheduling/program-scheduling.md.
---

# Program scheduling tuning

Program scheduling only moves the end-to-end objective when reusable KV is under
eviction pressure. When capacity is comfortable, the feature can add wall-clock
cost without a hit-rate gain. This skill sequences the work so that a negative
result is attributable to the algorithm rather than to a wrongly set knob.

Field semantics, defaults, and metric definitions are in
[`docs/program_scheduling/program-scheduling.md`](../../../docs/program_scheduling/program-scheduling.md). This skill
does not restate them; it covers order of operations, decision thresholds, and
the failure modes that are easy to mistake for "no benefit".

## Scope and deployment shapes

This procedure applies to Program scheduling on the Router's internal-DP path,
with one or more workers and either single-node or multi-node placement. In the
formulas below, `dp` is the total number of DP Ranks across all workers, and
`per_rank_capacity` is read separately for each Rank.

It does not apply to PD-disaggregated deployments. Those use a different
scheduling path and different capacity boundaries; see
[`docs/program_scheduling/pd-program-scheduling.md`](../../../docs/program_scheduling/pd-program-scheduling.md).

Nothing below depends on a specific accelerator family, device count, model
size, or agent implementation. Capacity comes from the engine, cost coefficients
from calibration on the target deployment, and gap scales from the workload.
Treat every absolute number here as a starting hypothesis to be re-measured, not
as a constant.

## Workload classes

Progress-TTL only pays when a Program returns before recomputing its context
would have been cheaper. The class of workload therefore decides whether tuning
can produce anything at all.

| Workload | Program shape | Expected outcome | Start with |
|---|---|---|---|
| Single-turn, no tool calls | no reuse across turns | no benefit; the feature is inert | confirm identity is not attached where it has no meaning |
| Interactive assistant, sub-second to seconds gaps | many short returns | retention can pay when eviction pressure exists | section 3 as written |
| Agent with build or test loops, seconds to minutes gaps | fewer, longer returns | TTL must cover the gap p95 and the cost model must reflect long prefill | the gap analysis in 3.2 before touching TTL |
| Minutes-to-hours jobs | rare returns, large contexts | TTL cannot cover the gap; expect affinity-only gains | the `binding_only` diagnostic; treat as non-regression |
| Batch or offline, no gaps | one request per Program | no reuse to protect; a large TTL only wastes capacity | leave Progress-TTL off |

If the class cannot show reuse — no gaps, or gaps far beyond any practical TTL —
record the result as a non-regression case rather than a failed experiment.

## Facts that drive most decisions

1. `max_acting_ttl_seconds` is an upper bound. The acting TTL is
   `min(max_acting_ttl_seconds, cache_miss_impact_seconds)`, where the impact is
   the calibrated cold-prefill cost for the uncached prefix multiplied by
   `(2 - decode_throughput_alpha)`. Raising the bound alone changes
   nothing while the cost model stays cheap.
2. Before `stats_window_size` samples accumulate, the automatic TTL is `0` and
   the growth reserve is `0`. A cold Router therefore pauses on the first tool
   gap.
3. Progress-TTL retains KV that would otherwise be evicted. It cannot create
   reuse when the working set fits in capacity; in that regime only the affinity
   component can help.

## 1. Preconditions

Do not tune before all three hold; a violation invalidates every later
comparison.

### 1.1 Program identity reaches the Router

Only requests carrying the configured `program_scheduling_enable_key` enter
Program scheduling. The default, `vllm_xargs.agentic_context`, requires a body
object; header-only harnesses need `auto`, and harnesses that post to
`/v1/messages` need a translation bridge, because only `/v1/chat/completions`
and `/v1/completions` enter the Program path.

```bash
curl -s "<router-url>/scheduling/diagnostics" |
  python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["enabled"], [r["dp_rank"] for r in d["ranks"]])'
grep -c program_admit "<router-log>"     # expect >= 1
```

Zero `program_admit` events mean the metrics will be identical for both arms and
say nothing about the algorithm. Confirm `--intra-node-data-parallel-size`
equals the engine's data-parallel size, otherwise the per-Rank view is wrong.

### 1.2 A comparable baseline exists

- The only difference between arms is `--enable-program-scheduling` plus the
  config JSON.
- Keep the baseline on the default request-level policy (`cache_aware`). A
  degraded baseline (`round_robin`) measures the baseline, not the feature.
- Restart the engine before each arm so prefix cache and cumulative counters
  start clean, and keep the start-of-run metrics as cold-start evidence.
- Hold the task set (same IDs and order), concurrency, sampling, engine
  context-length and memory-utilization settings, and node count fixed. Once
  concurrency is chosen, it is no longer a variable.

### 1.3 The baseline has eviction pressure

Measure `vllm:prompt_tokens_cached_total / vllm:prompt_tokens_total` on the
baseline. Aim for roughly 80–88%: above ~95% there is nothing to protect, below
~75% both arms sit near timeouts and the measurement is noisy. Adjust
concurrency or KV capacity until the band is hit.

Do not use `vllm:prefix_cache_queries_total` / `vllm:prefix_cache_hits_total`;
they describe lookup accounting, not served prompt tokens.

## 2. Three quantities to measure first

`CC` is the client concurrency and `dp` the number of DP Ranks in the
deployment.

```
working-set ratio  = CC * mean_context / (dp * per_rank_capacity)      target 1.0 - 1.5
in-flight ratio rho = r * CC * mean_context / (dp * per_rank_capacity) target 0.4 - 0.8
gap coverage        = TTL > p95(tool-call gap)
```

- `r` is in-flight requests divided by concurrency. It depends on the workload:
  interactive coding agents have been observed around 0.28–0.37 at low pressure,
  while batch or single-shot traffic sits lower and long deliberation loops
  higher. It approaches 1 near saturation, so linear extrapolation overestimates
  headroom. Measure it per concurrency step instead of assuming a value.
- The working-set ratio decides whether retention has anything to save. The
  in-flight ratio decides whether retention will throttle useful backend
  concurrency.
- The gap distribution is a property of the agent and its tools, not of the
  model. Compile-and-test agents can produce minute-scale gaps on the same
  engine where an interactive agent produces second-scale gaps. Measure; do not
  copy a number from another deployment.

## 3. Calibration order

### 3.1 Capacity per DP Rank

Read the per-Rank KV capacity from the engine's startup output (vLLM prints
`GPU KV cache size`), not from block counters. Block-count arithmetic has
disagreed with the engine's own number by more than 2x on real deployments,
which shifts every admission decision. Set
`token_capacity_per_dp_rank = 0.95 * log_value` and leave `low_watermark_ratio`
at 1.0; discounting twice removes the headroom twice.

Re-read the value after any change to the engine's memory-utilization or
context-length setting (`--gpu-memory-utilization`, `--max-model-len` in vLLM),
parallelism, or engine version. Capacity is a property of the deployment, not of
the hardware.

### 3.2 Workload profile

Run a short baseline arm and derive `r`, mean context, and the gap distribution
from the request log. Two useful analyses:

- gap-to-hit-rate buckets: if long gaps return with a materially lower hit rate
  than short gaps, the KV was lost during the gap and a longer TTL can pay. If
  all buckets look alike, the hits come from first turns or backend eviction,
  and lengthening TTL will not help.
- in-flight ratio: `requests * mean_latency / wall_time` divided by concurrency.

### 3.3 Concurrency and task count

Choose concurrency so both axes hold. Task count must be at least concurrency;
add tasks rather than over-subscribing a smaller set. If the working-set ratio
stays below 1.0 while the in-flight ratio is already near its ceiling, do not
add concurrency: shrink the pool with the engine's own knob (vLLM:
`--num-gpu-blocks-override`) so the working set exceeds capacity at the current
concurrency.

### 3.4 Cost coefficients

Calibrate `prefill_cost_model` and `decode_throughput_model` on the target
deployment with the AgentInfer calibration tooling listed under References.
Keep the fit-quality evidence; a prefill fit below R² ≈ 0.98 is not usable.
Write all seven fields explicitly, otherwise startup reports
`program_scheduling_reference_calibration` and the reference coefficients
silently drive TTL decisions.

Idle-engine calibration systematically understates recomputation cost: queueing,
chunked-prefill contention, and preemption are not exercised. Understated cost
shrinks the fitted TTL to seconds, so every tool gap pauses. Two remedies,
either acceptable if recorded in the report:

- calibrate against a loaded engine with representative concurrency, or
- fit the coefficients from a baseline-arm request log (the Program arm's Router
  queue time contaminates client-side TTFT, so it must not be the source).

`decode_throughput_alpha` is configured separately and is not produced by
calibration. Keep 0.15 unless a single sweep of 0.10 / 0.15 / 0.25 is part of
the experiment.

### 3.5 Pressure point

On the baseline arm, sweep a few concurrency values with a small task subset and
pick the value that lands the token cache hit in the 80–88% band. The band is a
heuristic for workloads capable of high reuse at low pressure; if the workload
cannot reach it, record a non-regression result instead of forcing eviction.
Each probe needs its own result directory, and probe runs are not evidence:
restart the engine before the recorded arms. Freeze concurrency here.

## 4. Tuning order

Change one field per iteration and record the config hash. The order below is
deliberate: capacity and lifetime errors masquerade as TTL errors, and placement
changes dominate everything.

### 4.1 Capacity and admission

| Field | Default | Guidance |
|---|---:|---|
| `token_capacity_per_dp_rank` | `null` (count guard only) | `0.95 * log capacity`; wrong-by-a-lot values show up as constant queueing or `unexpected_cache_discontinuity` |
| `max_active_programs_per_target` | `64` | Scale with concurrency: about `CC / dp * 1.2`. Leaving it at a value sized for a smaller run makes the count guard bind before token admission |
| `high_watermark_ratio` / `low_watermark_ratio` | `1.0` / `1.0` | Keep 1.0 when capacity already carries the 0.95 discount |
| `admission_waiting_request_threshold` | `1` | Raise only with evidence of request avalanches; it batches admits and adds latency |
| `decode_buffer_tokens` | `100` | Raise toward the p95 output length when harnesses omit `expected_output_length` |

### 4.2 Queue timeouts and Program lifetime

| Field | Default | Guidance |
|---|---:|---|
| `queue_timeout_seconds` | `600` | Must stay well below the client's own timeout. A multi-minute pool timeout shows up as a TTFT tail, not as engine queueing |
| `force_resume_timeout_seconds` | `300` | Anti-starvation bound; keep it below the task timeout and above normal queueing |
| `paused_retention_ttl_seconds` | `1800` | About 1–2x the p95 task duration |

### 4.3 Progress-TTL

| Field | Default | Guidance |
|---|---:|---|
| `max_acting_ttl_seconds` | `10` | Only a bound. Change it after, not before, the cost model is trustworthy |
| `stats_window_size` | `100` | Lower it for short experiments; otherwise the first minutes run with TTL = 0 |
| `prefill_cost_model.*` | reference values | Calibrated values; this is what makes TTL non-zero |
| `decode_throughput_alpha` | `0.15` | Lower values weight cache-miss impact higher and extend TTL |
| request-level `kv_retention_ttl_seconds` | unset | Overrides the computed TTL without the impact cap. Use only when gaps are known to exceed the impact |

### 4.4 Placement

| Field | Default | Guidance |
|---|---:|---|
| `binding_strategy` | `consistent_hash` | Keep it; most measurable benefit comes from returning to the same Rank |
| `global_queue` | `false` | Enable only when the in-flight ratio is high enough that cross-Rank moves buy capacity. At low pressure they only destroy per-Rank prefix cache |
| `cross_rank_headroom_ratio` | `1.2` | Raise to make migration rarer when `global_queue` is on |
| `binding_only` | `false` | Set `true` for the affinity-only diagnostic arm |

### 4.5 Fairness and isolation

| Field | Default | Guidance |
|---|---:|---|
| `resume_order` | `mru` | `fcfs` is a fairness experiment, not a fix for low hit rate |
| `max_segment_rounds` | `14` | Lower under heavy contention; raise if segment yields dominate the transition counters |
| `enable_batch_gain_admission` | `true` | Turn off only to isolate another variable |

## 5. Symptom to first knob

Mechanism side, read from `vllm_router_agent_aware_*` metrics:

| Observation | Likely cause | First change |
|---|---|---|
| `armed_ttl_seconds` near zero | window not matured, gaps longer than TTL, or understated cost | let the run reach maturity, then lower `stats_window_size`, then recalibrate |
| many `max_segment_yield` pauses | one Program consumes full segments | raise `max_segment_rounds`, or lower `force_resume_timeout_seconds` |
| `unexpected_cache_discontinuity` | physical eviction | recheck capacity; step concurrency down |
| `capacity_accounting_source` falls back | missing engine labels or stale observations | verify per-engine metric labels, single API server per worker, idle run before traffic |
| admits but no pauses and no hit gain | pressure too low, or Programs too short | return to section 3.5 |

End-to-end side, from the run summary plus `rank_queue_size` /
`queue_wait_seconds` (`JCT` is job completion time):

| Observation | Meaning | Ordered changes |
|---|---|---|
| hit and JCT both rise, queue wait high | retention is throttling useful concurrency | restore watermarks, lower `decode_buffer_tokens`, raise the waiting threshold, raise the Program count guard |
| hit flat, queue wait high | capacity too small for the workload | recalibrate capacity, then sweep it ±10% |
| p50 improves, p95 worsens | mean/median trade-off | disable `global_queue` or raise headroom ratio |
| cross-Rank moves appear | paused Programs are being relocated | `global_queue: false` |
| task timeouts rise | pool lifetime or starvation | align `queue_timeout_seconds` and `force_resume_timeout_seconds` with the task timeout |

If the mechanism is visibly active but the hit rate stays flat, run a TTL sweep
before changing anything else: `0` (admission only), then 0.5x, 1x, and 4x the
measured gap p95, all else equal. Gains should appear at the first arm whose TTL
covers the gap. A flat sweep means TTL is not the limiting factor; return to
section 3. Fixed-second values are only meaningful once the gap scale is known.

## 6. Evidence standard and stop conditions

Three layers are required. A hit-rate increase alone is a mechanism result, not
an end-to-end result.

| Layer | Evidence |
|---|---|
| Mechanism | token cache hit ratio moves; armed TTL is non-zero and consistent with the calibration; pause reasons match the workload's gap distribution |
| End to end | job completion time p50/p95 and wall time do not regress; TTFT p95 does not regress |
| Correctness and cost | resolved/completed tasks do not degrade (paired test); input and output token volume stays comparable |

Stop and record a negative result when any of these hold:

- two cold-start paired rounds show no directional change;
- the mechanism is active but the end-to-end objective is flat after a TTL
  sweep;
- the hit rate improves only alongside a wall-time regression large enough to
  erase it.

Report the limitation instead of tuning further. A deployment with no eviction
pressure is a non-regression case, not a failure of the algorithm.

## 7. Recalibration matrix

| Change | Recalibrate | Reusable |
|---|---|---|
| Model, parallelism (including node count), engine version, serving config | all seven cost coefficients and capacity | semantic switches (resume order, watermark ratios, metric interval) |
| Accelerator or hardware class | coefficients, capacity, and the pressure point | component flags and this procedure |
| Concurrency only | nothing; adjust the Program count guard and re-check the pressure band | coefficients and capacity |
| Workload or agent | window size, queue and starvation timeouts, paused retention, segment rounds, output buffer | coefficients on the same deployment |
| Capacity settings (engine memory utilization, context length, block override) | capacity and the pressure point | coefficients |

Portability checklist for a new deployment: read capacity (3.1), re-measure `r`
and the gap distribution (3.2), recalibrate coefficients (3.4), then re-find the
pressure point (3.5). The knob order in section 4 does not change between
deployments; only the numbers do.

Fast sanity check without a full recalibration: if startup does not warn about
reference coefficients, spot-check that a 50k-token cold prefill lands within
~10% of the deployed measurement. A larger gap means the fit does not describe
the deployment.

## 8. Ablations to run before claiming a gain

| Arm | Configuration | What it isolates |
|---|---|---|
| A | feature disabled | baseline |
| B | `binding_only: true` | Program-to-Rank affinity alone |
| C | full configuration | affinity plus retention |

Run each direction-finding arm on a small task subset; keep the full task set
for the paired baseline/candidate rounds. Report affinity and retention
separately, otherwise a threshold change and a scheduling change cannot be
distinguished.

## 9. Anti-patterns

- Tuning TTL before confirming eviction pressure. No pressure means no effect.
- Raising `max_acting_ttl_seconds` without fixing the cost model. The impact cap
  absorbs it.
- Comparing arms without restarting the engine. Prefix cache and cumulative
  counters leak across arms and bias the hit-rate comparison in favor of the
  second arm.
- Changing concurrency and TTL in the same round. The two effects are then
  inseparable.
- Enabling `global_queue` at low in-flight ratio. Cross-Rank moves discard the
  per-Rank prefix cache that the feature exists to protect.
- Using block-counter arithmetic as capacity. It disagrees with the engine's own
  accounting.
- Reporting a higher hit rate as the result. Retention that does not reach the
  end-to-end objective is an intermediate observation.

## References

- [`docs/program_scheduling/program-scheduling.md`](../../../docs/program_scheduling/program-scheduling.md) —
  configuration reference, calibration models, request metadata, and the
  upstream tuning workflow.
- [PR #286](https://github.com/vllm-project/router/pull/286) — Program-level
  Progress-TTL.
- [AgentInfer calibration tooling](https://github.com/openJiuwen-ai/agent-infer/tree/main/tools/calibration)
  — offline prefill and decode calibration.
