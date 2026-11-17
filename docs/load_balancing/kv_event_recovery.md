# KV event recovery

See the [CLI reference](../cli_reference.md#workers-and-backend) for
`--kv-event-strict-recovery`, its syntax, and defaults.

## Recovery behavior

Compatibility recovery can credit fresh observations after lost history; its
`observed` state does not mean the complete publisher history was recovered.
With strict recovery, incomplete history withholds placement credit, and a
source can remain in `fallback`. Requests remain routable in either mode.

A validated worker clear or sequence-zero history can restore `worker_exact`:
worker-owned cache tiers only. This assumes the worker cache was reset or fully
re-announced at sequence zero. External pool tiers remain uncredited until an
independent pool baseline exists; the current protocol provides none. Deployments
using Mooncake/LMCache pool tiers therefore lose that part of KV-aware routing
in strict mode. A worker clear restores worker-local credit, not pool credit.

The protocol has no snapshot watermark or producer-issued boot ID. Router
restart, worker re-add, or replay-buffer rollover can make history unavailable;
a publisher-only restart that preserves cache or passes the old cursor may not
be identifiable. Strict recovery does not prove instantaneous physical residency.
Keep it disabled unless these recovery limitations are acceptable.

Failed or unavailable replays have a three-attempt budget, with 2- and 4-second
waits before the second and third attempts. After exhaustion, a healthy
contiguous stream alone does not re-arm retries. An operator-initiated worker
KV clear can restore worker-local credit in the same publisher lifetime. Do not
automatically clear a live worker merely to restore Router credit. A replay
cancelled by the local queue limit retains received events and does not consume
the publisher retry budget, although it records a `failed` replay outcome.

## Memory during recovery

Each `(worker, rank)` subscriber buffers at most 1,024 live frames and 64 MiB of
frame data during replay, plus the single already-received frame crossing the
byte limit. With N simultaneously buffering subscribers, frame data can approach
N × 64 MiB plus those crossing frames; 100 subscribers can approach 6.25 GiB.
There is no process-wide shared byte cap. Transport queues and frame containers
add memory, and small events usually reach the frame limit first.

## Monitoring

Recovery metrics aggregate across workers and ranks; identify the affected
source in logs. The state gauge counts registered sources, rather than exposing
a per-source one-hot series. Retirement removes the source's gauge contribution.

| Metric | Labels and meaning |
|---|---|
| `vllm_router_kv_source_recovery_state` | `state`, `reason`; number of registered sources in each state |
| `vllm_router_kv_source_recovery_transitions_total` | `from`, `to`, `reason`; state transitions |
| `vllm_router_kv_source_recovery_duration_seconds` | `outcome`; time from loss until credit returns or the source retires |
| `vllm_router_kv_replay_outcomes_total` | `outcome`; `success`, `failed`, `unavailable`, or `superseded` |
| `vllm_router_kv_replay_duration_seconds` | `outcome`; duration when a replay endpoint was attempted |

`reason` describes the latest accepted transition or boundary. In particular,
`fallback` with `retry_exhausted` identifies exhausted recovery. A `superseded`
replay was ended by invalidation or a subscriber lifecycle change, including
shutdown; it is neither successful recovery nor a transport failure.

For example, alert on a loss of exact credit:

```promql
sum(increase(vllm_router_kv_source_recovery_transitions_total{from=~"exact|worker_exact",to=~"degraded|fallback"}[5m])) > 0
```

Or on sources remaining in fallback:

```promql
sum(vllm_router_kv_source_recovery_state{state="fallback"}) > 0
```

Choose windows, severity, and a sustained alert duration for the deployment so
brief recovery does not page an operator. Inspect logs for the source/rank,
replay endpoint availability, timeout or buffer loss, rejected events, and
supervisor invalidation.
