# Program scheduling

Program scheduling is an optional layer before native vLLM Router forwarding. It groups related requests into a long-lived Program, retains requests in a Router RequestPool when a backend has insufficient logical capacity, and protects reusable KV across short tool calls with Progress-TTL. Requests that do not carry the configured Program metadata continue through the existing request-level routing policy.

## Enablement

Library users configure Program scheduling through `RouterConfig.program_scheduling`. The standalone Rust and Python CLIs accept the same object through `--program-scheduling-config-json`; individual scheduling fields are not exposed as separate CLI flags. The `program_scheduling_enable_key` field controls which requests opt in:

| Value | Behavior |
|---|---|
| `vllm_xargs.agentic_context` | Default. Only the canonical body object enables Program scheduling. |
| `auto` | Also accepts `agent_hint`, Claude Code Headers, Codex/OpenCode Headers, and `x-session-id`. |

```json
{
  "program_scheduling_enable_key": "vllm_xargs.agentic_context",
  "global_queue": true,
  "resume_order": "mru",
  "token_capacity_per_target": 266864,
  "metrics_interval_seconds": 1.0,
  "prefill_cost_model": {
    "intercept_seconds": 0.06600061907132926,
    "linear_seconds_per_1k_tokens": 0.05702024012166613,
    "quadratic_seconds_per_1k_tokens_squared": 0.0044057347937978475,
    "decode_throughput_alpha": 0.15
  },
  "decode_throughput_model": {
    "fixed_step_seconds": 0.010635218423115973,
    "batch_step_seconds_per_request": 0.0004192803698834784,
    "context_step_seconds_per_token": 1.420458414996201e-7
  }
}
```

`global_queue: true` is the recommended deployment mode. The Router first tries to resume a paused Program on its previous Rank and considers cross-Rank placement only when the configured capacity and headroom checks allow it. Set it to `false` only when strict Rank-local scheduling is required.

To adapt Program scheduling to a different model or hardware configuration, operators generally only need to recalibrate `prefill_cost_model` and `decode_throughput_model`. The [AgentInfer backend calibration README](https://github.com/openJiuwen-ai/agent-infer/blob/main/tools/calibration/README.md) describes how to measure these coefficients for the target deployment.

## Launch the Router

Save the configuration above as `program-scheduling.json`, compact it into one CLI argument, and start the Router with the backend API endpoints. One Worker URL may expose multiple internal DP Ranks; `--intra-node-data-parallel-size` is the number of internal Ranks behind each Worker URL.

```bash
export PROGRAM_CONFIG_JSON="$(python -c \
  'import json,sys; print(json.dumps(json.load(open(sys.argv[1])), separators=(",", ":")))' \
  program-scheduling.json)"

vllm-router \
  --host 0.0.0.0 \
  --port 3001 \
  --worker-urls http://worker-0:8000 http://worker-1:8000 \
  --intra-node-data-parallel-size 2 \
  --policy consistent_hash \
  --prometheus-host 0.0.0.0 \
  --prometheus-port 29000 \
  --program-scheduling-config-json "${PROGRAM_CONFIG_JSON}"
```

`--policy` remains the native request-level fallback for requests without Program identity. `binding_strategy` in `program-scheduling.json` independently selects the initial Rank for a new Program. After startup, verify that the diagnostics contain the expected number of concrete DP Ranks and fresh backend observations before sending production traffic.

```bash
curl -fsS http://127.0.0.1:3001/health
curl -fsS http://127.0.0.1:3001/scheduling/diagnostics | python -m json.tool
```

The diagnostics report `observation_age_seconds`, `observation_fresh`, and `capacity_accounting_source` for each target. If a capacity observation is missing, stale (older than three polling intervals), or cannot produce a token estimate, scheduling preserves its existing Router-ledger fallback. The Router emits one `capacity_observation_fallback` warning when the source changes into a fallback state and one `capacity_observation_recovered` event when usable backend accounting returns; it does not repeat the same warning every polling interval.

At INFO level, Program scheduling logs committed `program_admit` and `program_pause` transitions plus `unexpected_cache_discontinuity` when an explicitly observed cache hit unexpectedly breaks within one placement. High-frequency `capacity_observation`, `cache_observation`, `ttl_armed`, `request_dispatch`, and `continuity_sample` events remain available at DEBUG. Their normal-path numerical data is also exposed through the `vllm_router_agent_aware_*` Prometheus metrics, including cache-miss impact, request intervals, armed TTL, explicit shared-prefix observations, and admission capacity snapshots. Metric labels are limited to target IDs and bounded enums.

Identity-free requests do not create Program state, enter RequestPool, or trigger Program backend polling. Existing `x-session-id` Consistent Hash routing therefore remains the fallback unless `auto` is explicitly configured.

## Configuration reference

All fields below belong to `program_scheduling`. Values omitted from the JSON object use the listed defaults.

| Parameter | Default | Effect on scheduling |
|---|---:|---|
| `program_scheduling_enable_key` | `vllm_xargs.agentic_context` | Selects which identity inputs opt a request into Program scheduling. The default accepts only canonical `vllm_xargs.agentic_context`; `auto` additionally accepts the documented `agent_hint` and framework Header schemas. |
| `binding_only` | `false` | Preserves Program identity and initial Program-to-Rank binding but bypasses RequestPool admission, Progress-TTL pause, and resume scheduling. This isolates the effect of Program affinity. |
| `global_queue` | `false` | When `false`, a paused Program can resume only on its bound Rank. When `true`, resume first tries the previous Rank and may then select another Rank that passes the cross-Rank capacity gates. |
| `resume_order` | `mru` | Orders ordinary Rank-local resume candidates within the same force-resume, request-priority, and segment-yield tier. Supported values are `mru` and `fcfs`; cross-Rank resume is always FCFS. |
| `cross_rank_headroom_ratio` | `1.2` | Controls cross-Rank migration conservatism. A destination must have at least the source Rank's free capacity plus this multiple of the Program's complete estimated context. It has no effect when `global_queue` is `false`. |
| `binding_strategy` | `consistent_hash` | Selects the initial home Rank for each new Program generation. Supported values are `consistent_hash`, `program_round_robin`, `least_program_count`, `reasoning_token_balance`, and `cache_aware`; it does not replace admission or resume decisions. |
| `hash_virtual_nodes` | `160` | Sets the number of virtual nodes used by `consistent_hash` initial binding. It has no effect on the other binding strategies. |
| `token_capacity_per_target` | `null` | Supplies the logical KV-token capacity of each concrete DP Rank. A value enables token-based admission, growth reserve, capacity repair, and cross-Rank headroom checks; `null` falls back to the Program-count guard. `reasoning_token_balance` requires this value. |
| `max_active_programs_per_target` | `64` | Limits active Programs on one Rank as a coarse concurrency guard. It is the primary capacity guard when token capacity is absent and remains a count guard when token capacity is configured; forced resume may override it. |
| `metrics_interval_seconds` | `1.0` | Sets the backend metrics polling and periodic scheduling interval. It also defines observation freshness and the minimum residence used by parts of cross-Rank resume. |
| `admission_waiting_request_threshold` | `1` | Blocks ordinary admission or resume to a Rank when its native vLLM waiting-request count reaches this threshold. `0` disables this waiting gate. |
| `queue_timeout_seconds` | `600.0` | Sets the maximum time a retained request may remain in the Router RequestPool. A request-scoped deadline can shorten, but not extend, this limit. |
| `force_resume_timeout_seconds` | `300.0` | Sets the fallback and upper bound for the workload-derived force-resume deadline that prevents starvation. The actual deadline may be shorter based on queued work and measured request throughput. |
| `paused_retention_ttl_seconds` | `1800.0` | Releases an idle paused Program generation after this retention period so stale Program state and logical capacity do not persist indefinitely. |
| `shared_prefix_freshness_warmup_seconds` | `100.0` | Sets the shared-prefix observation lifetime before rolling statistics are mature, and acts as the fallback lifetime when the dynamic estimate cannot be computed. |
| `shared_prefix_freshness_kv_turnovers` | `2.0` | Scales the mature shared-prefix freshness interval by the estimated time required for this many whole-KV-pool turnovers. Larger values retain an observation longer and reduce prefix queries, but increase stale-observation risk. |
| `decode_buffer_tokens` | `100` | Adds an immediate output-growth allowance to a Program's private-token demand when the request does not provide `expected_output_length`. It affects admission, resume, and logical occupancy accounting. |
| `max_acting_ttl_seconds` | `10.0` | Caps the request-specific acting TTL chosen from the mature interval window and cache-miss impact. Before the window is complete, the automatic TTL is `0`; request-scoped `kv_retention_ttl_seconds` overrides the computed value. |
| `high_watermark_ratio` | `1.0` | Multiplies token capacity to define when capacity repair starts pausing reasoning Programs. It must be greater than or equal to `low_watermark_ratio`. |
| `low_watermark_ratio` | `1.0` | Multiplies token capacity to define the admission/resume limit and the occupancy target of capacity repair. Lower values leave more free KV headroom. |
| `max_segment_rounds` | `14` | Caps the rolling estimate of protected continuous-service rounds. An idle Program that reaches this limit yields under local queue pressure and receives one-shot lower resume priority. |
| `stats_window_size` | `100` | Sets the fixed number of valid samples required before request-growth and continuity estimates become active, and bounds their rolling windows. Until then, growth reserve is zero and automatic acting TTL is zero. |
| `enable_batch_gain_admission` | `true` | Allows modeled decode batch gain to bypass only the continuous-growth reserve when it covers candidate recovery and continuity-loss costs. It never bypasses immediate capacity or Program-count limits. |
| `prefill_cost_model.intercept_seconds` | `0.06600061907132926` | Adds the fixed component of estimated cold-prefill cost used by TTL utility and batch-gain decisions. |
| `prefill_cost_model.linear_seconds_per_1k_tokens` | `0.05702024012166613` | Adds the linear cold-prefill cost for each 1K uncached prompt tokens. |
| `prefill_cost_model.quadratic_seconds_per_1k_tokens_squared` | `0.0044057347937978475` | Adds the quadratic cold-prefill cost for long uncached prompts. |
| `prefill_cost_model.decode_throughput_alpha` | `0.15` | Estimates the fraction of decode throughput retained while prefill is mixed into the batch. Lower values increase the cache-miss impact used by TTL and batch-gain decisions. |
| `decode_throughput_model.fixed_step_seconds` | `0.010635218423115973` | Sets the batch-independent decode-step latency in the throughput surface used to estimate admission batch gain. |
| `decode_throughput_model.batch_step_seconds_per_request` | `0.0004192803698834784` | Sets the additional decode-step latency contributed by each request in the batch. |
| `decode_throughput_model.context_step_seconds_per_token` | `1.420458414996201e-7` | Sets the additional decode-step latency contributed by each token in the sum of full Program contexts. |

## Canonical request metadata

The canonical contract is `vllm_xargs.agentic_context`. Identity semantics match AgentInfer EngineCore.

```json
{
  "vllm_xargs": {
    "agentic_context": {
      "program_id": "task-a:researcher",
      "task_id": "task-a",
      "session_id": "session-a",
      "agent_id": "researcher",
      "parent_program_id": "task-a:lead",
      "root_program_id": "task-a:lead",
      "blocks_parent": false,
      "expected_resume": true,
      "agent_role": "subagent",
      "spawn_reason": "research",
      "request_id": "request-a",
      "step_id": 7,
      "priority": 10,
      "deadline_seconds": 30,
      "expected_output_length": 2048,
      "kv_retention_ttl_seconds": 5
    }
  }
}
```

`program_id` is stable for one Program generation. `parent_program_id` and `root_program_id` form a model-scoped parent forest that supports multi-level agent trees but does not grant scheduling privilege. An omitted first `step_id` becomes zero; a later omitted value becomes the previous observed value plus one.

`request_id`, `priority`, `deadline_seconds`, `expected_output_length`, and `kv_retention_ttl_seconds` are request-scoped. They are retained with exactly one RequestPool entry; dispatch transfers them only to that exact in-flight request, and completion or cancellation drops them. Priority affects resume ordering, deadline overrides the force-resume wait, expected output length replaces the default output allowance in capacity demand, and KV-retention TTL replaces the automatic acting TTL after that request completes.

## Compatibility identity inputs

`auto` uses this strict precedence: canonical context, `agent_hint`, then one known Header schema. Claude Code uses `x-claude-code-session-id`, `x-claude-code-agent-id`, and `x-claude-code-parent-agent-id`; Codex/OpenCode uses `session-id` and `thread-id`; the generic schema uses `x-session-id`. Recognized schemas are not mixed.

For `agent_hint`, `session_id` becomes both Program ID and session ID, `task_id` remains absent, and `blocks_parent` defaults to true. A blocking hint requires `parent_session_id`. Explicit `expected_resume` always overrides the blocking-subagent default.

## Resume ordering and capacity

Resume first selects expired force-resume deadlines, then larger request priorities. Within each equal force/priority cohort, a Program paused because it reached `max_segment_rounds` ranks behind other candidates. Rank-local ordering defaults to MRU and can be changed to FCFS, while cross-Rank candidates always use FCFS. Successful resume clears the one-shot yield marker. Priority and yield ordering only change selection order and never grant a new capacity bypass; established force-resume admission behavior remains unchanged.

The first implementation contains no automatic privileged Program, relationship handoff, privileged TTL, or privileged capacity demotion. Programs that consume a complete continuous segment yield once to other waiters, while force resume bounds starvation.

Acting Programs are not reclaimed by admission or resume. Their TTL and terminal lifecycle are responsible for moving them out of ACTIVE state; capacity repair pauses reasoning Programs under pressure.

## Offline calibration models

The prefill and decode coefficients depend on the model, accelerator, parallel configuration, inference-engine version, and execution settings. Operators should measure them on the deployed backend rather than copying values calibrated for a different system. Follow the [AgentInfer backend calibration procedure](https://github.com/openJiuwen-ai/agent-infer/blob/main/tools/calibration/README.md), retain its fit-quality evidence, and copy the constrained coefficients from `calibration.json` into the Router configuration:

| AgentInfer `calibration.json` coefficient | Router field |
|---|---|
| `ttl_prefill_model_intercept_seconds` | `prefill_cost_model.intercept_seconds` |
| `ttl_prefill_model_linear_seconds_per_1k_tokens` | `prefill_cost_model.linear_seconds_per_1k_tokens` |
| `ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared` | `prefill_cost_model.quadratic_seconds_per_1k_tokens_squared` |
| `decode_step_fixed_seconds` | `decode_throughput_model.fixed_step_seconds` |
| `decode_step_seconds_per_request` | `decode_throughput_model.batch_step_seconds_per_request` |
| `decode_step_seconds_per_context_token` | `decode_throughput_model.context_step_seconds_per_token` |

`prefill_cost_model.decode_throughput_alpha` represents retained decode throughput while prefill is mixed into the batch and is configured separately; the current cold-prefill and warm-prefix decode calibration artifact does not derive it.

The built-in values remain backward-compatible defaults. When Program scheduling is enabled with `binding_only: false` and one or more coefficient fields are omitted, startup emits `program_scheduling_reference_calibration`. The warning lists exactly which fields used defaults, reports the effective reference estimates (about `13.93` seconds for a 50,000-token cold prefill and `98.23` aggregate tokens/second for batch size 4 with 200,000 total context tokens when all reference defaults apply), and warns that a deployment mismatch can degrade scheduling decisions. Explicitly configuring all seven fields suppresses the warning; invalid explicit values remain configuration errors.

For `x = uncached_prompt_tokens / 1000`, the cold-prefill estimate is `prefill_seconds = intercept_seconds + linear_seconds_per_1k_tokens * x + quadratic_seconds_per_1k_tokens_squared * x^2`. Progress-TTL includes mixed-batch decode interference as `cache_miss_impact_seconds = prefill_seconds * (2 - decode_throughput_alpha)`.

| `prefill_cost_model` field | Unit | Constraint | Default |
|---|---|---|---:|
| `intercept_seconds` | seconds | finite and >= 0 | `0.06600061907132926` |
| `linear_seconds_per_1k_tokens` | seconds per 1K uncached tokens | finite and >= 0 | `0.05702024012166613` |
| `quadratic_seconds_per_1k_tokens_squared` | seconds per squared 1K-token unit | finite and >= 0 | `0.0044057347937978475` |
| `decode_throughput_alpha` | retained decode-throughput ratio during prefill | finite and within `[0, 1]` | `0.15` |

For batch size `B` and the sum of full Program contexts `C`, aggregate decode throughput is `B / (fixed_step_seconds + batch_step_seconds_per_request * B + context_step_seconds_per_token * C)`. Full Program contexts are used even when Programs share prefixes because the fitted surface models execution cost rather than physical KV occupancy.

| `decode_throughput_model` field | Unit | Constraint | Default |
|---|---|---|---:|
| `fixed_step_seconds` | seconds per decode step | finite and > 0 | `0.010635218423115973` |
| `batch_step_seconds_per_request` | additional step seconds per batched request | finite and >= 0 | `0.0004192803698834784` |
| `context_step_seconds_per_token` | additional step seconds per total context token | finite and >= 0 | `1.420458414996201e-7` |

## Engine KV-control boundary

Progress-TTL currently protects KV indirectly by controlling request admission. Physical eviction, offload, and prefetch are not executed by this feature. A future Router-to-engine KV hint must receive explicit `accepted`, `rejected`, or `unsupported` feedback and an eventual `applied`, `failed`, or `expired` result. Router residency accounting must not change until the engine reports `applied`.
