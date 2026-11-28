# KV-aware decision receipts

With the existing opt-in OpenTelemetry configuration enabled, each actual
`kv_aware` selection creates a `kv_route` span. Its application attributes are
the fixed fields of `KvDecisionReceipt`:

| Field | Meaning |
| --- | --- |
| `policy` | The bounded policy name, currently `kv_aware`. |
| `stage` | `regular`, `prefill`, or `decode`. |
| `outcome` | A bounded selection or fallback reason. |
| `credited_tokens` | Router-estimated credited work when available; never engine-confirmed reuse. |
| `new_prefill_tokens` | The known remaining Prefill work; unavailable for Decode. |
| `limiting_group` | A cache-group index when the placement provider supplies it. |
| `target_index` | The selected index in this decision's supplied candidate slice. |
| `source_state` | The source recovery state when supplied by the placement provider. |
| `overhead_micros` | Elapsed local selection-call time, in microseconds. |

No prompt, token IDs, request identity, or worker URL is part of this contract.
The serializable receipt contains exactly these nine fields. Tracing exporters
may additionally attach their standard code, thread, and timing metadata.
Existing HTTP transport spans retain their own HTTP attributes.

The receipt keeps its full unsigned integer types and JSON numeric values.
OTel integer attributes are signed: known values through `i64::MAX` are
recorded as integers, and larger values as exact decimal strings. This applies
to all numeric fields, including `target_index` and `overhead_micros`; an
out-of-range value is never truncated, saturated, or marked unavailable.

## Unavailable values and credit

An unavailable numeric value is `null` in the receipt and the string
`unavailable` in the span, rather than a fabricated zero. Known zero credited
work has an explicit reason, such as `no_reusable_prefix`; unavailable credit
retains a fallback such as `missing_tokens`, `cache_salt`, or
`unverified_identity`. Explicit cache-read exclusions retain their own reason,
including `skip_prefix_cache`, `skip_external_kv`, and
`unsupported_prompt_scoring`, even though selection consumes their boolean
qualification. Preparation facts travel with the same selection so each span
attribute is recorded once with its final value, without relying on duplicate
attribute keys being overwritten. A P/D pool missing a candidate's work estimate
reports `pool_estimate_unavailable`, rather than borrowing another candidate's
identity failure. Unknown fallback strings collapse to the bounded
`preparation_unavailable` outcome.

This also changes existing observable fallback labels: `StageDecision.fallback_reason`
and the `fallback` label of `vllm_router_pd_prefill_decisions_total` report
`pool_estimate_unavailable` when a candidate's work estimate is missing, instead
of a reason taken from an arbitrary candidate. Even when every candidate has
`unsupported_cache_identity`, that surface now reports the more general pool
reason. This trades input-specific diagnostic detail for correct pool-level
attribution; it does not change selection or admission.

The current placement-provider interface does not return a source recovery
state or the group limiting a restore. Consequently, `source_state` is
`unavailable` and `limiting_group` is unavailable. A compatible cache identity
or an indexed placement hit is not used to invent an `exact` source state.
There is no extra index query for observation. These receipts do not yet
diagnose the exact source state or limiting cache group.

Lookup availability is a separate, smaller fact: built-in providers carry it
with the same placement query and request-local memo. A visible, current empty
source can confirm zero. An unpublished/replaying source, retired lease,
untrusted or expired layout, or a grant lost during the lookup cannot; these
produce unavailable credit and `placement_unavailable` when no more specific
input exclusion applies. Existing custom providers that only return match
vectors default to unavailable observation. They can implement the compatible
`observed_matches_with_cache` method to supply this fact explicitly.

The shared domain-miss shortcut does not inspect each source's usable snapshot.
It retains the same scoring result and lookup count, but conservatively reports
unavailable observation. A negative page memo that did not validate the source
is treated the same way. Availability never implies an `exact` recovery state.

The receipt uses the credit that survives the existing commit-time validation.
Revoked or unpinnable placement falls back to full input without asserting a
known cache miss. The Regular retry limit retains its bounded
`placement_pin_failed` outcome when pin failures exhausted the attempts.
Observation does not add queries, retries, source pins, or reservations.

`credited_tokens` describes the routing estimate used for Prefill work. It is
separate from worker-reported cache reuse and from Decode's marginal-context
occupancy estimate. An unqualified identity or skipped placement lookup has
unavailable credit; its known input size can still establish Prefill work.

## Span relationships

The existing HTTP request span is the parent of each `kv_route`. A selected
stage's `prefill_dispatch` or `decode_dispatch` is a child of that decision;
existing HTTP client spans are children of dispatch. Regular colocated requests
use `prefill_dispatch` for the single combined worker call, without claiming a
separate engine Decode measurement. HTTP and gRPC use the same receipt contract.

With OTel enabled, gRPC connection and buffer tasks do not inherit a request's
span: cached channels and reconnects can outlive that request. Each RPC still
runs under its selected dispatch span. With OTel disabled, connection tasks
retain the existing tracing-context inheritance for ordinary logs.

Reqwest does not expose a connection-task executor. With OTel enabled, the
bounded request paths use an owned send task that carries the subscriber but
does not enter the request span. This covers generation, input preparation,
Program warm-remote operations, EPD encoding, and P/D profiling. Explicit HTTP
client spans retain their dispatch parents and injected trace headers. Dropping
the caller aborts its send task; reqwest errors are returned unchanged and task
panics keep their payload. Task cancellation while its awaiter is still alive
is an unexpected runtime failure, since the abort handle is private.

This adds a Tokio task, scheduling, and task allocation per enabled HTTP send.
Internal connection logs retain their subscriber but no longer inherit the
request span. Disabled sends keep the direct path. These costs have not been
benchmarked. This does not claim complete health-probe span coverage.

P/D has one receipt per actual stage selection. A deferred Decode placeholder
creates no receipt: the final commitment supplies the Decode receipt and its
current candidate index. Retries keep separate decision spans. A failed Prefill
that never commits Decode therefore has no Decode decision or dispatch span.

Program scheduling binds Decode through its own scheduler, so it produces no
native KV-aware Decode receipt. When its configured Prefill policy is
`kv_aware`, the same pure selection returns its observation without reserving
load a second time. Warm-remote and cold fallback carry that single selection's
receipt; a warm decline does not select Prefill again. The generic policy API
has no prepared tokens, so its credit is explicitly unavailable.

`kv_route` is the logical decision parent retained for its dispatch. Its
duration is not a standalone routing-latency measurement. Use
`overhead_micros` for local selection-call time; P/D's previously prepared token
and placement estimates and all network rendering are outside that timer.
Turning OTel off creates no decision or dispatch spans and preserves passive
HTTP trace-header forwarding.

The provider's existing match methods and policy implementations remain usable
through default observation methods. Runtime selection and reservation behavior
is preserved. This is not complete Rust source compatibility for direct struct
construction: `KvSelectionReceipt` and `StageDecision` add `trace`, and
`PrefillWorkEstimate` adds `lookup_available`. External code constructing those
structs must supply the new fields; unknown lookup availability must be `false`.
External policy implementations can create a trace handle with
`KvDecisionTrace::new(receipt)` from their own explicit decision facts. This
observation factory does not choose workers, query placement, reserve load, or
authenticate the supplied facts.

Each decision's receipt and span handle have fixed size. Additional CPU state
for lookup availability grows linearly with the candidate count and the number
of request-local memo entries.

## CPU validation

The receipt contract and policy observation tests are in the library's
`kv_decision` and `policies::kv_aware` modules. The integration matrix exercises
real HTTP/gRPC dispatches through local workers and an in-memory OTel exporter:

```bash
cargo test --features cpu,ready --lib kv_decision
cargo test --features cpu,ready --lib policies::kv_aware
cargo test --features cpu,ready --test kv_decision_trace_test
cargo test --features cpu,ready --test test_pd_warm_remote
```

These tests establish Router behavior and span relationships. They do not
establish engine-confirmed KV reuse or production instrumentation overhead.
