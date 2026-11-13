# KV event recovery

- Compatibility recovery is the default. With `--kv-event-strict-recovery`,
  placement credit is withheld until an authoritative recovery boundary;
  a source can stay in `fallback`. Requests remain routable. After the three
  replay attempts are exhausted, an operator-initiated worker KV clear is
  needed to restore worker-local credit in the same publisher lifetime.
- External pool tiers remain uncredited in strict mode until an independent
  pool baseline exists. The current protocol provides none, so deployments
  using Mooncake/LMCache pool tiers lose that part of KV-aware routing.
  A worker clear restores worker-local credit, not pool credit.
- Monitor `vllm_router_kv_source_recovery_state{state="fallback",reason="retry_exhausted"}`
  and `vllm_router_kv_source_recovery_transitions_total`. The state gauge counts
  sources across all workers/ranks; inspect logs to identify the affected source.
