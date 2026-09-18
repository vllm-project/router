# Program scheduling

Program scheduling is an optional layer before native vLLM Router forwarding. It groups related requests into a long-lived Program, retains requests in a Router RequestPool when a backend has insufficient logical capacity, and protects reusable KV across short tool calls with Progress-TTL. Requests that do not carry the configured Program metadata continue through the existing request-level routing policy.

## Enablement

Program scheduling is configured by `program_scheduling` or `--program-scheduling-config-json`. The `program_scheduling_enable_key` field controls which requests opt in:

| Value | Behavior |
|---|---|
| `vllm_xargs.agentic_context` | Default. Only the canonical body object enables Program scheduling. |
| `auto` | Also accepts `agent_hint`, Claude Code Headers, Codex/OpenCode Headers, and `x-session-id`. |

```json
{
  "program_scheduling_enable_key": "vllm_xargs.agentic_context",
  "global_queue": false,
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

Identity-free requests do not create Program state, enter RequestPool, or trigger Program backend polling. Existing `x-session-id` Consistent Hash routing therefore remains the fallback unless `auto` is explicitly configured.

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

The prefill and decode coefficients depend on the model, accelerator, parallel configuration, inference-engine version, and execution settings. Operators should measure them on the deployed backend rather than copying values calibrated for a different system. The built-in values remain backward-compatible defaults.

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
