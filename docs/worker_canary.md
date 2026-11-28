# Idle worker generation canary

`--worker-canary-interval-secs N` (default `0`) enables a generation check in
the regular HTTP worker pool. The first check waits a full idle interval.
The Router sends a non-streaming, one-token `/v1/completions` request directly
to an idle worker with the credential bound to that registration for HTTP
probes, if any, and the pinned DP rank when known. Unbound workers receive no
authorization header. The canary never falls back to the router-wide API key.
It does not pass through user admission, routing, or request
metrics. There are at most four concurrent probes Router-wide, each bounded
by the existing health-check timeout. The check is local to the worker and
never sends a KV transfer envelope.

Passive health and canary health are independent gates. A 2xx is insufficient
to verify standalone support. A complete, normal one-token response verifies
it. Before then, failures are reported without excluding the worker. A 404,
405 or 501 before verification marks the capability unsupported for that
registration. Once verified, consecutive generation failures exclude the
worker at `--health-failure-threshold`. Recovery needs
`--health-success-threshold` generation successes and a passing passive health
gate. A passive 200 alone does not clear generation failure. Removing or
replacing a worker retires that registration's evidence and in-flight probe.

The Router only starts checks after its own worker traffic has been idle for
the interval. New user traffic is never delayed by a canary and invalidates
an overlapping probe's verdict. Requests sent directly to an engine or through
a different Router are not visible. P/D, EPD, gRPC, external OpenAI providers,
and unpinned multi-rank workers are excluded.

- `vllm_router_worker_canary_total{result}` counts `ok`, `failed`, `timeout`,
  `unsupported` and `superseded` attempts.
- `vllm_router_worker_idle_seconds` is the oldest observed idle age among
  eligible workers at the last sweep. It is zero if there are no idle
  eligible workers. Neither series has a worker URL label.

Review points are the thresholds, the separate recovery gate, and whether any
dedicated P/D role should ever opt into standalone verification. In particular,
the Router cannot safely withdraw a worker that never demonstrated standalone
generation solely on the basis of a failed canary.
