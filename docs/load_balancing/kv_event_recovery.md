# KV event recovery

## Compatibility and strict recovery

The default is compatibility recovery (`kv_event_strict_recovery: false`). After
an unrecoverable gap the Router clears the affected source/rank's claims and
may credit fresh, validated observations again. Its state is `observed`, not
`exact`: this does not assert that the entire publisher history was recovered.
Other workers remain available and ordinary routing continues without KV credit
when placement evidence is unavailable.

Opt in to authoritative-history recovery using `--kv-event-strict-recovery`
(Rust and Python CLI), the Python `kv_event_strict_recovery=True` argument, or
the corresponding RouterConfig boolean. It defaults to false in old JSON
configurations as well. The setting applies to regular and P/D event subscribers.

Strict recovery preserves the scope of its baseline:

| Evidence | Scope after successful recovery |
|---|---|
| Validated worker sequence zero, live or replayed from zero | `worker_exact`: worker-owned tiers only, under the worker reset assumption below |
| Validated worker clear after loss | `worker_exact`; the clear does not cover shared pools |
| Contiguous replay of a gap on an intact baseline in the same stream | Preserve that baseline: `worker_exact` stays worker-only; an already `exact` baseline stays exact |
| Worker clear on an uninterrupted exact stream | Preserve exact: the clear only changes worker-owned records |

These conditions apply on initial attachment as well as after detected sequence
rollback. A new subscriber cannot tell whether a pool predates its stream.
An external pool can retain X across worker restart and only announce Y after
zero. Replaying that stream cannot reconstruct X. Strict mode therefore hides
external credit, including fresh pool stores, until there is a proven pool
baseline. The current wire protocol supplies no pool-wide baseline. A pool
snapshot or complete inventory with a consistent watermark, or a certified
pool clear followed by contiguous updates, could establish one. None of these
is implied by a worker clear or sequence zero. Compatibility mode retains its
existing fresh-observation and sequence-zero behavior.

The current wire protocol has no snapshot/watermark or producer-issued boot ID.
After Router restart, worker re-add, or replay-buffer rollover, sequence zero
may no longer be retained. Full reconstruction then cannot succeed, including
when it exceeds the fixed five-second replay deadline. A healthy contiguous
stream alone does not repair this strict-mode fallback. Keep strict recovery
disabled unless that availability trade-off is acceptable. Do not clear a live
worker's KV automatically just to make the Router exact; a clear must be an
explicit operational action against the publisher. Snapshot/anchor support is
follow-up work under #426, not implemented by this flag.

### Legacy protocol assumptions

Strict recovery is conservative within the existing event contract, not an
unconditional guarantee about physical KV residency. It assumes one ordered
publisher stream per binding and that sequence zero represents the beginning
of worker-owned cache history (worker KV reset or a complete re-announcement).
Router cannot independently verify that precondition. Sequence rollback is a local
restart observation, not a producer-authenticated identity. A publisher-only
restart that retains cache state, or a replacement whose sequence has already
passed the old cursor, cannot be fully identified by the current wire format.
This PR does not claim to solve those producer-identity cases. `exact` means
complete event evidence through the consumed watermark under that contract;
normal asynchronous event delay can still make it differ from instant device
residency.

The snapshot/watermark follow-up must define a source-owned consistent metadata
snapshot at W, bind snapshot and replay to the same producer incarnation, retain
the required changes after W, and publish recovered credit only after contiguous
catch-up. Snapshot data must include enough prefix structure to reconstruct the
index without mistaking structural ancestors for resident blocks. A Router's
already-incomplete local index is not an authoritative source snapshot.

## Retry and lifecycle rules

A failed or unavailable replay consumes one of three attempts per subscriber.
The first two failures impose delays of 2 and 4 seconds before another attempt;
a subsequent gap or rejected frame drives the retry. After the third failure
the source falls back with reason `retry_exhausted`. A superseded attempt
consumes no retry budget and does not itself advance the signal-loss epoch.

Repeated bad frames in the same loss episode do not repeatedly flush placement
and Program warm candidates. After a validated owned batch, a new gap must
again discard any intervening observations, including while retries are suppressed.
A successful replay or authoritative boundary resets the budget. A healthy
contiguous stream alone does not re-arm an exhausted budget. If the publisher
continues in the same incarnation, an operator can explicitly clear its KV cache
and have the router observe the validated clear event to restore worker-local credit
and re-arm retries.

Each recovery has a unique local attempt ID. Before applying each replay batch,
the subscriber checks the attempt under the same mutation lock used for source
invalidation. A superseded replay stops before writing another batch. Finishing a stale attempt cannot
restore placement or replace the reason set by an external invalidation. State
mutators and replay completion serialize inside the indexer; callers do not
need to rely solely on the subscriber lease convention. Source leases still
fence event application by retired subscribers. The supervisor must retire and
forget a previous registration before starting its replacement. Registration
occurs only after local socket creation, connect and subscription succeed;
ZMQ connect success does not prove the remote publisher is reachable.

## Metrics and diagnosis

Metrics aggregate across source IDs and ranks to avoid retaining a new label
series for every worker incarnation. Source/rank details belong in logs.

| Metric | Labels / meaning |
|---|---|
| `vllm_router_kv_source_recovery_state` | `state`, `reason`; current count of registered sources, not a per-source one-hot gauge |
| `vllm_router_kv_source_recovery_transitions_total` | `from`, `to`, `reason`; transition count |
| `vllm_router_kv_source_recovery_duration_seconds` | `outcome`; time from loss until visibility is restored or source retired |
| `vllm_router_kv_replay_outcomes_total` | `outcome`; `success`, `failed`, `unavailable`, or `superseded` |
| `vllm_router_kv_replay_duration_seconds` | `outcome`; replay duration when an endpoint was attempted |

`reason` describes the latest accepted state transition/boundary. An Exact clear
updates it to `clear_barrier` without invalidating queries merely for a reason
change. Replay is recorded once after commit validation. `superseded` means
another invalidation or lifecycle change won; it is neither successful recovery
nor a transport failure. Its existing invalidation reason is preserved.

Example alert expressions (choose windows and severity for the deployment):

```promql
sum(increase(vllm_router_kv_source_recovery_transitions_total{from=~"exact|worker_exact",to=~"degraded|fallback"}[5m])) > 0
```

```promql
sum(vllm_router_kv_source_recovery_state{state="fallback"}) > 0
```

Use a sustained alert duration for fallback; inspect logs for the source/rank,
whether the replay endpoint is configured, timeout/buffer loss, schema or block
size errors, and supervisor invalidations. Worker retirement decrements its
state count. Hashing an unbounded source ID is not a cardinality fix and is not
used here.

## Release note

This change adds opt-in strict KV event recovery and source recovery telemetry.
Default recovery continues to use fresh observations after lost history, now
identified as `observed` rather than `exact`. Strict mode withholds cache credit
until an authoritative recovery boundary and can retain fallback when historical
events are unavailable. Recovery metrics use aggregate bounded labels; consumers
of earlier draft metrics must remove `source`/`dp_rank` selectors and treat the
state gauge as a count. Routing availability is independent of placement credit.
