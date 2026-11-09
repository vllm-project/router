# Agent integration for workflow TTL prediction

This guide defines the request contract that an agent or agent harness should
use when it wants Program scheduling to predict return time and retain reusable
KV across tool calls. It applies to native, non-PD HTTP routing. In vLLM PD
mode, workflow prediction may run in `shadow` mode for observation, but
`active` return-time prediction is not supported.

The Router measures the interval from one successful model request completing
to the next request for the same Program arriving. The agent does not send that
interval, a predicted TTL, or prefill-cost coefficients. It supplies stable
Program identity, lifecycle intent, and a bounded workflow classification so
the Router can learn the appropriate distribution.

## Minimum agent contract

Send `vllm_xargs.agentic_context` on every model request that should participate
in Program scheduling. For reliable workflow prediction, provide at least:

| Field | Requirement |
| --- | --- |
| `program_id` | Use one stable, unique ID for all turns in one live Program generation. Do not reuse it for an unrelated concurrent Program. |
| `expected_resume` | Set `true` when another model request is expected after this request and `false` only when the agent knows this is the final model request. |
| `step_id` | Use the logical turn number and increment it for each new turn. Keep it unchanged when retrying the same logical turn. |

The Router can derive a Program ID from some combinations of task, session, and
agent fields, and can infer missing step IDs. An explicit `program_id` and
`step_id` are nevertheless recommended because they prevent framework-specific
defaults from changing the prediction identity.

`expected_resume` is a request-time declaration. If the agent cannot know
whether the model will request another tool before it sends the request, it
should use `true`; it must not guess `false`, because `false` terminates the
Program generation after completion. A completion declared with
`expected_resume=false` contributes an explicit `no_return` observation. An
unexpected client exit, Router shutdown, or TTL expiry is censored instead of
being mislabeled as `no_return`.

## Workflow classification

The following fields are strongly recommended. They let observations from
different Program generations share a useful, bounded prediction profile:

| Field | Meaning and stability requirement |
| --- | --- |
| `workflow_profile_id` | A low-cardinality workflow family such as `swe-agent`, shared by comparable runs. It is not a task or session ID. |
| `profile_version` | The version of that workflow contract. Keep it stable within a run and change it when stage semantics or lifecycle behavior changes materially. |
| `agent_role` | A stable role category such as `lead`, `researcher`, or `reviewer`. |
| `spawn_reason` | A bounded stage or invocation category such as `repository_search`, `implementation`, or `test`. Change the value only when the semantic stage changes. |
| `blocks_parent` | Whether this Program blocks its parent. If it is `true`, also provide a valid `parent_program_id`. |

Canonical field names are preferred. The parser also accepts `workflow_type`
for `workflow_profile_id`, and `subagent_type` or `agent_type` for
`spawn_reason`.

The exact return-time key contains the workflow profile ID and version, served
model pool, agent role, spawn reason, parent-blocking state, resume intent, and
a bucket derived from `step_id`. When an exact key lacks enough observations,
the Router backs off in stages: first without the step bucket, then to broader
workflow/role state, model-and-role state, model state, and finally the global
profile.

Use a small taxonomy so each key can reach `return_time_min_support`. Do not put
request IDs, timestamps, prompt text, tool arguments, user IDs, or random
values in workflow classification fields. High-cardinality values fragment
support, fill `max_keys`, and force predictions to broader fallback levels.

## Request example

An intermediate request that is expected to resume after a tool call can use:

```json
{
  "model": "my-model",
  "messages": [],
  "vllm_xargs": {
    "agentic_context": {
      "program_id": "task-123:researcher",
      "task_id": "task-123",
      "session_id": "session-123",
      "agent_id": "researcher",
      "parent_program_id": "task-123:lead",
      "root_program_id": "task-123:lead",
      "agent_role": "researcher",
      "spawn_reason": "repository_search",
      "workflow_profile_id": "swe-agent",
      "profile_version": "1",
      "blocks_parent": false,
      "expected_resume": true,
      "step_id": 3,
      "request_id": "request-456"
    }
  }
}
```

The next logical turn keeps `program_id` and the stable workflow fields, and
uses `step_id: 4`. If it moves to a genuinely different workflow stage, it may
change `spawn_reason` to another value from the same bounded taxonomy.

When the orchestrator knows before dispatch that a request is the final model
turn, it sends the same Program identity with:

```json
{
  "vllm_xargs": {
    "agentic_context": {
      "program_id": "task-123:researcher",
      "workflow_profile_id": "swe-agent",
      "profile_version": "1",
      "agent_role": "researcher",
      "spawn_reason": "final_response",
      "blocks_parent": false,
      "expected_resume": false,
      "step_id": 8
    }
  }
}
```

## Fields that do not define return-time behavior

`request_id` is for correlation and never becomes Program identity.
`priority` and `deadline_seconds` affect RequestPool ordering and wait limits.
`expected_output_length` affects capacity reservation and is also an explicit
override for output-length prediction. Parent and root IDs describe the agent
tree; they do not grant scheduling privilege.

Do not normally send `kv_retention_ttl_seconds` when using workflow TTL
prediction. It is an authoritative request-scoped override and prevents the
predicted TTL from controlling that completion. Use it only when the agent
intentionally owns the TTL decision.

The agent also does not send `prefill_cost_model` coefficients. Those values
depend on the model, hardware, parallel layout, and engine configuration and
must be calibrated and configured by the Router operator.

## KV reuse requirements

Return-time prediction controls how long the Router keeps a logical Program
seat. It does not reserve physical vLLM KV. To make a retained seat useful:

- preserve the reusable prompt prefix between turns instead of rewriting early
  messages or changing tokenization;
- keep one `program_id` across those turns so Program affinity remains intact;
- configure realistic logical capacity and backend KV capacity; and
- verify actual cached prompt tokens, because physical eviction can still
  remove a logically retained prefix.

## Router rollout

Enable Program scheduling with the canonical opt-in contract and start workflow
prediction in `shadow` mode:

```json
{
  "program_scheduling_enable_key": "vllm_xargs.agentic_context",
  "prediction_profile": {
    "mode": "shadow",
    "return_time_bucket_bounds_seconds": [1, 2, 5, 10],
    "return_time_min_support": 20,
    "return_time_horizon_seconds": 5,
    "step_bucket_size": 4,
    "max_keys": 4096
  }
}
```

`shadow` collects the same observations and exposes predictions without
changing the fitted acting TTL. After the metadata taxonomy has enough support
and the predicted TTLs match observed return behavior, a non-PD deployment can
switch to `active`. In `active` mode, a mature workflow prediction replaces the
rank-level fitted TTL. Insufficient support falls back to the fitted TTL, and a
request-scoped TTL remains authoritative.

An offline profile is optional. If `profile_path` is configured,
`workflow_profile_id` and `profile_version` select compatible seeded entries;
the bounded online overlay continues learning from current traffic.

## Integration validation

Before enabling `active`, verify all of the following:

1. Requests appear as Programs in `GET /scheduling/diagnostics`, with the
   expected `program_id`, workflow fields, `step_id`, and `expected_resume`.
2. Repeated turns retain one Program identity rather than creating a new
   Program per request.
3. `return_prediction_support` reaches the configured minimum, and
   `return_prediction_backoff_level` is not unexpectedly broad.
4. DEBUG `return_time_prediction` events show plausible
   `predicted_ttl_seconds`, `baseline_ttl_seconds`, and support.
5. In `active` mode, `prediction_ttl_applied=true` and
   `vllm_router_agent_aware_armed_ttl_seconds{source="return_prediction"}`
   appears for eligible completions.
6. Backend `cached_prompt_tokens` confirms that retained Programs actually
   preserve reusable KV.

Useful aggregate metrics include
`vllm_router_workflow_return_prediction_total`,
`vllm_router_workflow_return_prediction_support`,
`vllm_router_workflow_predicted_ttl_seconds`,
`vllm_router_workflow_return_actual_seconds`, and
`vllm_router_workflow_prediction_profile_keys`.

If predictions remain absent, first check canonical opt-in, stable Program
identity, and minimum support. If predictions repeatedly use broad backoff,
reduce classification cardinality. If a mature prediction is visible but does
not control TTL, check for `shadow` mode, PD mode, a request-scoped TTL
override, or a terminal completion.
