---
name: program-ttl-tuning
description: >-
  Ordered procedure for tuning Program scheduling and Progress-TTL on vLLM Router (PR #286), portable across model, accelerator, node count, and agent workload. Use it to establish a pressure point, calibrate capacity and cost coefficients, then tune admission, TTL, and placement in order. Covers enabling the feature on a new deployment, re-tuning after a model or workload change, and diagnosing a stalled token cache-hit ratio, a TTFT tail, or job timeouts. Complements docs/program_scheduling/program-scheduling.md.
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

For a sample invocation, replay commands, and a deployment decision table, see
[Tuning DeepSeek V4 Flash on Ascend A3](../../../docs/program_scheduling/ascend-a3-deepseek-v4-flash.md).

## Scope and deployment shapes

This procedure applies to Program scheduling on the Router's internal-DP path,
with one or more workers and either single-node or multi-node placement. In the
formulas below, `dp` is the total number of DP Ranks across all workers, and
`per_rank_capacity` assumes a common measured capacity across homogeneous Ranks.
Read every Rank's startup capacity to verify that assumption. The Router's
`token_capacity_per_dp_rank` is one scalar applied to every Rank, not a map of
per-Rank limits; use the conservative mixed-Rank procedure in section 3.1 when
the measured capacities differ.

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

1. The automatic acting TTL is bounded by
   `min(max_acting_ttl_seconds, cache_miss_impact_seconds)`, where the impact is
   the calibrated cold-prefill cost for the uncached prefix multiplied by
   `(2 - prefill_cost_model.decode_throughput_alpha)`. Once the continuity window
   is mature, the interval fit selects the TTL with the best estimated utility
   within that bound; it can select a smaller TTL or zero even when the bound is positive.
   Raising `max_acting_ttl_seconds` alone has no effect while the impact cap
   binds, and a higher cap does not guarantee a longer selected TTL.
2. Before `stats_window_size` samples accumulate, the automatic TTL is `0` and
   the growth reserve is `0`. A cold Router therefore pauses on the first tool
   gap.
3. Progress-TTL controls Router admission and logical token reservations to
   reduce competing KV demand across tool gaps. It does not pin physical KV
   blocks or execute engine eviction, offload, or prefetch. An engine can still
   evict blocks before the logical TTL expires. Armed TTL and a hit-rate gain
   establish logical scheduling behavior and measured reuse, not physical
   residency. With a working set that already fits, there may be little reuse
   to recover; measure affinity and full-scheduler effects separately.

## 1. Preconditions

Do not tune before all three hold; a violation invalidates every later
comparison.

### 1.1 Program identity reaches the Router

Requests must satisfy the configured `program_scheduling_enable_key` to enter
Program scheduling. The default, `vllm_xargs.agentic_context`, requires canonical
identity metadata in the JSON body; header-only harnesses need `auto` and a
supported session identity header. In addition to `/v1/chat/completions` and
`/v1/completions`, POST requests handled by the default transparent-proxy
fallback, including `/v1/messages`, run Program intake without a path filter.
They still need valid identity metadata. A protocol translation bridge is needed
only if the backend does not support the harness's API; any bridge must preserve
that metadata. Backend protocol support and Program-path entry are separate
checks.

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
- For every recorded cold-start arm, including C0/C and TTL-sweep arms, restart
  both the Router and the engines. The engine restart clears prefix cache and
  its counters; the Router restart clears Program/affinity bindings, queued
  state, and learned rank windows. Refreshing unchanged rank registrations or
  only waiting for Programs to expire does not clear learned rank windows.
- Use the same readiness and idle-observation procedure before traffic. Save
  startup logs, start-of-run metrics, and enabled-arm scheduling diagnostics;
  verify no leftover Programs or queued requests and zero rank
  `rolling.request_samples` / `rolling.continuity_samples`. For a separate
  warm-state comparison, document equivalent warmup histories and scheduler
  state in both arms; do not present an engine-only restart as a cold start.
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

The capacity ratios assume homogeneous Ranks and roughly balanced load. With
mixed capacities, the smallest measured Rank capacity can be used for a
conservative screening calculation, but `dp * min_capacity` is only a lower
bound on total physical capacity. Check each Rank's load, cache pressure, and
hit ratio; an aggregate ratio or the common logical limit does not establish
physical eviction pressure. Gap coverage refers to logical TTL, not a guarantee
that the engine keeps the blocks resident.

- `r` is in-flight requests divided by concurrency. It depends on the workload:
  interactive coding agents have been observed around 0.28–0.37 at low pressure,
  while traffic with no gaps can approach 1 and long tool or deliberation gaps
  reduce it. It approaches 1 near saturation, so linear extrapolation overestimates
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
which shifts every admission decision. For homogeneous Ranks, start with
`token_capacity_per_dp_rank = floor(0.95 * common_log_capacity)`.

For mixed Ranks, this Router still applies one scalar to every Rank: use
`floor(0.95 * min(all_rank_log_capacities))`, including every worker's Ranks.
Do not use the largest Rank, an average, or a summed capacity as that per-Rank
limit. This conservative starting point leaves some capacity on larger Ranks
unused by the logical guard; record the per-Rank measurements and that trade-off,
and verify pressure per Rank rather than assuming uniform utilization. Leave
`low_watermark_ratio` at 1.0 when capacity already includes the 0.95 discount;
discounting twice removes the headroom twice. Logical admission limits do not
guarantee physical KV residency.

Re-read the value after any change to the engine's memory-utilization or
context-length setting (`--gpu-memory-utilization`, `--max-model-len` in vLLM),
parallelism, or engine version. Capacity is a property of the deployment, not of
the hardware.

### 3.2 Workload profile

Run a short baseline arm and derive `r`, mean context, and the gap distribution
from the request log. Two useful analyses:

- gap-to-hit-rate buckets: lower hit rates after long gaps show reduced reuse,
  not proof of eviction or a benefit from a longer logical TTL. Check request
  prefix stability, placement, and backend cache pressure, then use the C/C0
  control in section 8 to test retention. A flat bucket profile alone does not
  identify where the hits came from or rule out a TTL effect.
- active-request fraction: estimate `r = requests * mean_latency / (wall_time * CC)`
  over the same baseline observation window, with latency and wall time in the
  same units. Then estimate KV pressure as `rho = r * working-set ratio`
  using section 2. Apply the 0.4–0.8 target band to `rho`, not `r`: for example,
  `r = 0.60` and a working-set ratio of `1.5` give `rho = 0.90`, above the band.

### 3.3 Concurrency and task count

Choose concurrency so both axes hold. Task count must be at least concurrency;
add tasks rather than over-subscribing a smaller set. Before increasing
concurrency or reducing the KV pool (vLLM: `--num-gpu-blocks-override`), project
both the working-set ratio and `rho = r * working-set ratio`. At fixed
concurrency, reducing capacity raises both projected ratios for unchanged
`r` and mean context. Consider a smaller pool only if the projected operating
point stays within both target bands, then remeasure `r`, mean context, and
per-Rank pressure in the baseline. For example, if `r = 0.90`, moving the
working-set ratio from `0.88` to `1.0` raises projected `rho` from `0.792` to
`0.90`. At a measured `r > 0.80`, the two bands cannot overlap at that
operating point: try a different baseline concurrency and remeasure, or
record the workload as outside the target bands rather than forcing eviction.

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

`prefill_cost_model.decode_throughput_alpha` is configured within
`prefill_cost_model`, separately from the fitted coefficients, and is not
produced by calibration. Keep 0.15 unless a single sweep of 0.10 / 0.15 / 0.25
is part of the experiment.

### 3.5 Pressure point

On the baseline arm, sweep a few concurrency values with a small task subset and
pick the value that lands the token cache hit in the 80–88% band. The band is a
heuristic for workloads capable of high reuse at low pressure; if the workload
cannot reach it, record a non-regression result instead of forcing eviction.
Each probe needs its own result directory, and probe runs are not evidence:
restart both the Router and engines before the recorded arms, following section
1.2. Freeze concurrency here.

## 4. Tuning order

Change one field per iteration and record the config hash. The order below is
deliberate: capacity and lifetime errors masquerade as TTL errors, and placement
changes dominate everything.

### 4.1 Capacity and admission

| Field | Default | Guidance |
|---|---:|---|
| `token_capacity_per_dp_rank` | `null` (count guard only) | One scalar for all Ranks: start with `floor(0.95 * smallest rank log capacity)`; see section 3.1. Recheck it when persistent queueing or INFO `unexpected_cache_discontinuity` events appear, but neither alone proves a capacity error |
| `max_active_programs_per_target` | `64` | Scale with concurrency: about `CC / dp * 1.2`. Leaving it at a value sized for a smaller run makes the count guard bind before token admission |
| `high_watermark_ratio` / `low_watermark_ratio` | `1.0` / `1.0` | Keep 1.0 when capacity already carries the 0.95 discount |
| `admission_waiting_request_threshold` | `1` | Backend waiting-queue ceiling: ordinary admission/resume blocks when `waiting >= threshold`; `0` disables the gate. Raising it relaxes backpressure and can worsen backend queueing. Increase only when evidence shows the gate suppresses useful concurrency, not in response to a request avalanche |
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
| `prefill_cost_model.*` | reference values | Calibrated values determine cache-miss impact; a non-zero TTL also requires a mature interval window and a useful fitted candidate |
| `prefill_cost_model.decode_throughput_alpha` | `0.15` | Lower values increase cache-miss impact and may extend the fitted TTL; check the selected TTL rather than assuming it increases |
| request-level `kv_retention_ttl_seconds` | unset | Overrides the computed TTL without the impact cap on this internal-DP path. Use for the controlled retention diagnostic in section 5; measure queueing and end-to-end cost before keeping an override |

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

Mechanism side, use `vllm_router_agent_aware_*` Prometheus metrics,
`/scheduling/diagnostics`, and Router INFO logs as identified below:

| Observation | Likely cause | First change |
|---|---|---|
| `armed_ttl_seconds` near zero | window not matured, gaps longer than TTL, or understated cost | let the run reach maturity, then lower `stats_window_size`, then recalibrate |
| many `max_segment_yield` pauses | one Program consumes full segments | raise `max_segment_rounds`, or lower `force_resume_timeout_seconds` |
| `unexpected_cache_discontinuity` (Router INFO log event) | explicitly reported cached tokens fall below the continuity threshold relative to the previous context within one placement; this alone does not prove physical eviction | inspect the log's `cached_prompt_tokens`, `previous_context_tokens`, and `cache_hit_vs_previous_context`; check whether the request prefix changed and correlate with backend cache pressure before changing capacity or concurrency |
| `capacity_accounting_source` falls back (diagnostics) | missing engine labels or stale observations | verify per-engine metric labels, single API server per worker, idle run before traffic |
| admits but no pauses and no hit gain | pressure too low, or Programs too short | return to section 3.5 |

End-to-end side, from the run summary plus `rank_queue_size` /
`queue_wait_seconds` (`JCT` is job completion time):

| Observation | Meaning | Ordered changes |
|---|---|---|
| hit and JCT both rise, queue wait high | retention or admission gates may be throttling useful concurrency | check the blocking reason and backend waiting count; restore watermarks and correct output reservations first. Raise the waiting threshold or Program count guard only if that gate is suppressing useful concurrency and the backend has headroom |
| hit flat, queue wait high | capacity too small for the workload | recalibrate capacity, then sweep it ±10% |
| p50 improves, p95 worsens | mean/median trade-off | disable `global_queue` or raise headroom ratio |
| cross-Rank moves appear | paused Programs are being relocated | `global_queue: false` |
| task timeouts rise | pool lifetime or starvation | align `queue_timeout_seconds` and `force_resume_timeout_seconds` with the task timeout |

If the mechanism is visibly active but the hit rate stays flat, run a controlled
TTL sweep before changing anything else. On this internal-DP path, attach
request-level `kv_retention_ttl_seconds` to every request in each arm: `0`
(admission without post-request TTL retention), then 0.5x, 1x, and 4x the measured
gap p95, all else equal. This override isolates retention duration from the
automatic interval fit and impact cap; sweeping `max_acting_ttl_seconds` alone
does not establish that the acting TTL changed. Verify the armed TTL and pause
reasons in each arm, since capacity or segment yields can still end retention
early, and the engine can evict physical blocks while logical retention remains
active. A flat sweep is informative only if logical retention actually changed;
then return to section 3 to investigate other limits. Restore automatic TTL selection
after the diagnostic. Fixed-second values are only meaningful once the gap scale
is known.

## 6. Evidence standard and stop conditions

Three layers are required. A hit-rate increase alone is a mechanism result, not
an end-to-end result.

| Layer | Evidence |
|---|---|
| Mechanism | token cache hit ratio moves; logical armed TTL is non-zero and consistent with the calibration; pause reasons match the workload's gap distribution. These signals do not establish physical KV residency; a residency claim needs independent engine evidence |
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
| C0 | same full configuration as C, `binding_only: false`, with request-level `kv_retention_ttl_seconds: 0` on every request | full scheduler without post-request TTL retention; admission, pause/resume, and fairness remain enabled |
| C | full configuration, `binding_only: false`, with the candidate TTL policy | full scheduler including post-request TTL retention |

Run each direction-finding arm on a small task subset; keep the full task set
for the paired baseline/candidate rounds. Report B minus A as the affinity
delta and C minus B as the full-scheduler delta: `binding_only` bypasses
RequestPool admission, Progress-TTL pause/resume scheduling, and fairness, so
C minus B cannot establish a retention-only gain.

Before attributing a change to retention, compare C with C0, holding the workload,
placement policy, admission thresholds, fairness settings, and all other
configuration fixed. Only the request-level TTL override differs: C0 overrides
every request to zero, while C uses the candidate automatic TTL or explicit
override. Verify the armed TTL and pause reasons as in section 5; if C also
selects zero TTL, the pair has not exercised different retention. Report this
controlled TTL-policy delta separately from the full-scheduler delta, including
queueing, task completion, and end-to-end cost.

## 9. Anti-patterns

- Tuning TTL before confirming eviction pressure. No pressure means no effect.
- Raising `max_acting_ttl_seconds` without fixing the cost model. The impact cap
  absorbs it.
- Comparing cold-start arms after restarting only the engine. Router affinity
  and learned rank windows can carry over and mature automatic TTL earlier in
  a later arm. Restart both components as in section 1.2; record equivalent
  histories explicitly for a separate warm-state experiment.
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
