# Command-line reference

The router has two CLI entry points:

- The **Rust binary**, built with `cargo build --release --bin vllm-router`, uses
  [clap declarations in `src/main.rs`](../src/main.rs).
- The **Python launcher**, installed by `pip install vllm-router`, uses
  [argparse declarations in `router_args.py`](../py_src/vllm_router/router_args.py).
  `python -m vllm_router.launch_router` invokes the same launcher. It starts the
  Rust extension in-process by default; `--mini-lb` selects the Python debug
  load balancer instead.

Both install an executable named `vllm-router`. Check which executable is on
your `PATH` before copying flags between installations. Their options are
similar, but they are not interchangeable. In particular:

- Rust accepts `rendezvous_hash`; Python does not. Rust also exposes backend
  selection, tracing, API-key validation URLs, history storage and profiling.
- Python exposes `--mini-lb`, queue settings and the token bucket refill rate.
  The Rust CLI fixes queue size at `100`, queue timeout at `60` seconds, and
  leaves the refill rate unset (it falls back to `max_concurrent_requests`).
- Cache eviction uses `--eviction-interval` in Rust and
  `--eviction-interval-secs` in Python. Log-level choices also differ.

## Reading the tables

The defaults below are **parser defaults**, not `RouterArgs()` dataclass or
library constructor defaults. `unset` means no value is supplied, `[]` is an
empty list, `{}` is an empty selector map, and **—** means the option is absent
from that CLI. Mode-specific fallbacks are described alongside the tables.

`flag` takes no value; supplying it enables the option. A scalar or choice
takes one value. `[0..]` takes zero or more space-separated values in one
occurrence, and `[1..]` requires at least one. Options marked `(repeatable)`
accumulate values across occurrences, including Rust `Vec<T>` options with
clap's implicit append action. Put non-repeatable list values after one flag.
Rust rejects repeated scalar options; Python keeps the last value for scalar
and non-append list options.

Rust integer types are unsigned: `u16` is 0–65535, `u32` is 0–4294967295,
`u64` is 0–18446744073709551615, and `usize` depends on the target pointer
width. `f32` and `f64` are floating-point values. Python's `integer` and
`float` use `int` and `float`; runtime configuration validation still applies.

## General and server

| Option | Rust value | Rust default | Python value | Python default | Description |
| --- | --- | --- | --- | --- | --- |
| `--host` | `string` | `127.0.0.1` | `string` | `127.0.0.1` | Address on which the router listens. |
| `--port` | `u16` | `30000` | `integer` | `30000` | Router HTTP port. |
| `--help` | `flag` | `n/a` | `flag` | `n/a` | Print help and exit; `-h` is also accepted. Rust `--help` shows long help. |
| `--version` | `flag` | `n/a` | — | — | Print the Rust binary version and exit; `-V` is also accepted. |
| `--max-payload-size` | `usize` | `536870912` | `integer` | `536870912` | Maximum request body size in bytes; the default is 512 MiB. |
| `--request-timeout-secs` | `u64` | `1800` | `integer` | `1800` | Request timeout in seconds. |
| `--max-concurrent-requests` | `usize` | `32768` | `integer` | `32768` | Token bucket capacity for request admission; does not enforce a hard limit on in-flight requests. |
| `--queue-size` | — | — | `integer` | `100` | Pending request queue capacity; `0` disables queuing and returns 429 when full. |
| `--queue-timeout-secs` | — | — | `integer` | `60` | Maximum time in seconds spent waiting in the request queue. |
| `--rate-limit-tokens-per-second` | — | — | `integer` | `unset` | Token bucket refill rate for request admission; unset uses `max_concurrent_requests`. These are admission tokens, not generated model tokens. |
| `--cors-allowed-origins` | `string [0..] (repeatable)` | `[]` | `string [0..]` | `[]` | Allowed browser origins, e.g. `https://example.com`; see CORS behavior below. |

With no CORS origins, the server permits any origin. Supplying origins restricts
the allowed list. For Python-only queue and rate-limit controls, use the Python
launcher; there are no corresponding Rust CLI flags.

Admission tokens refill with elapsed time, so another request can be admitted
while an earlier request is still running, even with `--max-concurrent-requests 1`.

## Workers and backend

| Option | Rust value | Rust default | Python value | Python default | Description |
| --- | --- | --- | --- | --- | --- |
| `--worker-urls` | `string [0..] (repeatable)` | `[]` | `string [0..]` | `[]` | Worker URLs for regular routing, e.g. `http://worker:8000`; Rust also supports `grpc://` inference workers. |
| `--worker-startup-timeout-secs` | `u64` | `600` | `integer` | `600` | Time in seconds to wait for workers to become ready. |
| `--worker-startup-check-interval` | `u64` | `30` | `integer` | `30` | Seconds between worker startup checks; also controls regular-router `power_of_two` load polling. |
| `--intra-node-data-parallel-size` | `usize` | `1` | `integer` | `1` | DP replicas per worker URL; values greater than 1 enable DP-aware routing. |
| `--backend` | `vllm, trtllm, openai, anthropic` | `vllm` | — | — | Select the backend; `trtllm` and `anthropic` currently warn and fall back to regular routing. |
| `--runtime` | `vllm, trtllm, openai, anthropic` | `vllm` | — | — | Alias for `--backend`, with the same choices and default. |
| `--mini-lb` | — | — | `flag` | `false` | Use the Python debug load balancer instead of the Rust extension; it does not implement the full router feature set. |
| `--enable-igw` | `flag` | `false` | `flag` | `false` | Enable Inference Gateway mode for multi-model routing. |
| `--history-backend` | `memory, none` | `memory` | — | — | Inference Gateway history storage: in-memory storage or no storage. |
| `--profile` | `flag` | `false` | — | — | Enable profiling calls to vLLM workers; the Rust CLI uses a 10-second profiling timeout. |

The Rust CLI requires worker URLs in regular mode unless service discovery is
enabled. The Python launcher permits an empty worker list without service
discovery; workers can be registered later through `/add_worker`. Gateway mode
and the Rust OpenAI backend use different configuration paths. See
[gRPC workers](backend/grpc.md) for supported gRPC worker capabilities.

## Routing policies

| Option | Rust value | Rust default | Python value | Python default | Description |
| --- | --- | --- | --- | --- | --- |
| `--policy` | `random, round_robin, cache_aware, power_of_two, consistent_hash, rendezvous_hash` | `cache_aware` | `random, round_robin, cache_aware, power_of_two, consistent_hash` | `cache_aware` | Request-level routing policy; also the fallback for both PD stages. |
| `--prefill-policy` | `random, round_robin, cache_aware, power_of_two, consistent_hash, rendezvous_hash` | `unset` | `random, round_robin, cache_aware, power_of_two, consistent_hash` | `unset` | PD prefill policy; unset inherits `--policy`. |
| `--decode-policy` | `random, round_robin, cache_aware, power_of_two, consistent_hash, rendezvous_hash` | `unset` | `random, round_robin, cache_aware, power_of_two, consistent_hash` | `unset` | PD decode policy; unset inherits `--policy`. |
| `--cache-threshold` | `f32` | `0.3` | `float` | `0.3` | Cache-match threshold from 0.0 to 1.0 for cache-aware routing. |
| `--balance-abs-threshold` | `usize` | `64` | `integer` | `64` | Cache-aware load balancing absolute threshold; used together with the relative threshold. |
| `--balance-rel-threshold` | `f32` | `1.5` | `float` | `1.5` | Cache-aware load balancing relative threshold; used together with the absolute threshold. |
| `--eviction-interval` | `u64` | `120` | — | — | Seconds between cache-aware approximation-tree eviction operations (Rust spelling). |
| `--eviction-interval-secs` | — | — | `integer` | `120` | Seconds between cache-aware approximation-tree eviction operations (Python spelling). |
| `--max-tree-size` | `usize` | `67108864` | `integer` | `67108864` | Maximum cache-aware approximation-tree size. |

Cache-aware tuning applies only to policies using `cache_aware`. Load balancing
is triggered when both `(max_load - min_load) > balance_abs_threshold` and
`max_load > min_load * balance_rel_threshold` hold. The CLI does not expose the
consistent-hash virtual-node count, which is `160`. In the regular router,
`power_of_two` load polling uses `--worker-startup-check-interval` (default
`30` seconds) from either CLI. The internal policy configuration's
`load_check_interval_secs` value is not used by this polling loop. See
[routing policies](load_balancing/README.md).

Without service discovery, `power_of_two` requires at least two workers. A
PD stage using an explicit power-of-two policy needs two workers for that stage.

## Prefill/decode disaggregation

| Option | Rust value | Rust default | Python value | Python default | Description |
| --- | --- | --- | --- | --- | --- |
| `--vllm-pd-disaggregation` | `flag` | `false` | `flag` | `false` | Enable vLLM prefill/decode (PD) routing. |
| `--prefill` | `URL [PORT or none] (repeatable)` | `[]` | `string [1..] (repeatable)` | `unset` | Add one prefill URL and optional bootstrap port: `--prefill URL [PORT]`. Omit the port or pass `none` for no bootstrap port. |
| `--decode` | `string (repeatable)` | `[]` | `string (repeatable)` | `unset` | Add one decode URL per occurrence: `--decode URL`. |
| `--vllm-discovery-address` | `string` | `unset` | `string` | `unset` | ZMQ registration address for vLLM workers, e.g. `0.0.0.0:30001`. |
| `--kv-connector` | `nixl, mooncake, moriio` | `nixl` | `nixl, mooncake, moriio` | `nixl` | PD KV transfer connector: NIXL, Mooncake or MoRI-IO. MoRI-IO requires vLLM ZMQ discovery. |

PD workers can come from static URLs, vLLM ZMQ discovery, or Kubernetes
discovery. For static configuration, provide both prefill and decode workers:

```bash
vllm-router --vllm-pd-disaggregation \
  --prefill http://prefill-1:8000 9000 \
  --prefill http://prefill-2:8000 none \
  --decode http://decode-1:8001 --decode http://decode-2:8001 \
  --prefill-policy cache_aware --decode-policy power_of_two
```

Python converts omitted `--prefill` and `--decode` values to empty worker lists.
Although Python's parser accepts one or more strings per `--prefill`, its
supported format is one URL and an optional port, not several URLs after one
flag. Repeat the flag for each worker.

The Rust binary consumes `--prefill` before clap parsing, so it does not appear
in clap's option list in `--help`.

## Program scheduling

| Option | Rust value | Rust default | Python value | Python default | Description |
| --- | --- | --- | --- | --- | --- |
| `--enable-program-scheduling` | `flag` | `false` | `flag` | `false` | Enable Program-level scheduling independently of the request-level routing policy. |
| `--program-scheduling-config-json` | `string` | `unset` | `string` | `unset` | JSON object overriding Program scheduling defaults; requires `--enable-program-scheduling`. |

For JSON fields, defaults and request opt-in behavior, see
[Program scheduling](program_scheduling.md). The JSON option configures the
feature; it does not enable it on its own.

## Kubernetes service discovery

| Option | Rust value | Rust default | Python value | Python default | Description |
| --- | --- | --- | --- | --- | --- |
| `--service-discovery` | `flag` | `false` | `flag` | `false` | Discover workers from Kubernetes pods. |
| `--selector` | `string [0..] (repeatable)` | `[]` | `string [1..]` | `{}` | Regular worker pod labels as space-separated `key=value` pairs. |
| `--service-discovery-port` | `u16` | `80` | `integer` | `80` | Port used in discovered worker URLs; the parser default is `80` in both CLIs. |
| `--service-discovery-namespace` | `string` | `unset` | `string` | `unset` | Namespace to watch; unset watches all namespaces and needs cluster-wide permissions. |
| `--prefill-selector` | `string [0..] (repeatable)` | `[]` | `string [1..]` | `{}` | PD prefill pod labels as space-separated `key=value` pairs. |
| `--decode-selector` | `string [0..] (repeatable)` | `[]` | `string [1..]` | `{}` | PD decode pod labels as space-separated `key=value` pairs. |

Selectors are converted to maps. An omitted selector is empty in both CLIs;
when the same key appears more than once, the last value wins. In PD mode,
use `--prefill-selector` and `--decode-selector` for the two worker roles.
The bootstrap port annotation is fixed to `vllm.ai/bootstrap-port`.

## Retries

| Option | Rust value | Rust default | Python value | Python default | Description |
| --- | --- | --- | --- | --- | --- |
| `--retry-max-retries` | `u32` | `5` | `integer` | `5` | Maximum retry count. |
| `--retry-initial-backoff-ms` | `u64` | `50` | `integer` | `50` | Initial retry backoff in milliseconds. |
| `--retry-max-backoff-ms` | `u64` | `30000` | `integer` | `30000` | Maximum retry backoff in milliseconds. |
| `--retry-backoff-multiplier` | `f32` | `1.5` | `float` | `1.5` | Multiplier for exponential retry backoff. |
| `--retry-jitter-factor` | `f32` | `0.2` | `float` | `0.2` | Jitter factor applied to retry backoff. |
| `--disable-retries` | `flag` | `false` | `flag` | `false` | Disable retries; overrides the retry configuration. |

## Circuit breakers

| Option | Rust value | Rust default | Python value | Python default | Description |
| --- | --- | --- | --- | --- | --- |
| `--cb-failure-threshold` | `u32` | `10` | `integer` | `10` | Consecutive failures before opening the circuit breaker. |
| `--cb-success-threshold` | `u32` | `3` | `integer` | `3` | Successes needed to close the circuit breaker. |
| `--cb-timeout-duration-secs` | `u64` | `60` | `integer` | `60` | Seconds before an open breaker can transition to half-open. |
| `--cb-window-duration-secs` | `u64` | `120` | `integer` | `120` | Currently unused by the circuit breaker; changing it has no effect on failure counting. |
| `--disable-circuit-breaker` | `flag` | `false` | `flag` | `false` | Disable circuit breaking; overrides the circuit breaker configuration. |

## Health checks

| Option | Rust value | Rust default | Python value | Python default | Description |
| --- | --- | --- | --- | --- | --- |
| `--health-failure-threshold` | `u32` | `3` | `integer` | `3` | Consecutive failed health checks before marking a worker unhealthy. |
| `--health-success-threshold` | `u32` | `2` | `integer` | `2` | Consecutive successful health checks before marking a worker healthy. |
| `--health-check-timeout-secs` | `u64` | `5` | `integer` | `5` | Health check request timeout in seconds. |
| `--health-check-interval-secs` | `u64` | `60` | `integer` | `60` | Seconds between runtime health checks. |
| `--health-check-endpoint` | `string` | `/health` | `string` | `/health` | Worker health endpoint path. |

## Logging and metrics

| Option | Rust value | Rust default | Python value | Python default | Description |
| --- | --- | --- | --- | --- | --- |
| `--log-dir` | `string` | `unset` | `string` | `unset` | Log file directory; unset logs only to the console. |
| `--log-level` | `debug, info, warn, error` | `info` | `debug, info, warning, error, critical` | `info` | Log severity; Rust accepts `warn`, while Python accepts `warning` and `critical`. |
| `--prometheus-port` | `u16` | `29000` | `integer` | `29000` | Prometheus listener port. Both CLI parsers enable metrics on `29000` by default. |
| `--prometheus-host` | `string` | `127.0.0.1` | `string` | `127.0.0.1` | Prometheus listener bind address. |
| `--request-id-headers` | `string [0..] (repeatable)` | `[]` | `string [0..]` | `unset` | Custom request ID header names; omitted values use common headers, described below. |

The common request ID headers are `x-request-id`, `x-correlation-id`,
`x-trace-id` and `request-id`. Rust also uses these when an explicitly supplied
list is empty; Python distinguishes an omitted list from an explicit empty
list. The Python CLI's metrics defaults differ from `RouterArgs()`, whose
Prometheus host and port start unset.

## OpenTelemetry tracing (Rust)

| Option | Rust value | Rust default | Python value | Python default | Description |
| --- | --- | --- | --- | --- | --- |
| `--enable-trace` | `flag` | `false` | — | — | Enable OpenTelemetry tracing. |
| `--otlp-traces-endpoint` | `string` | `unset` | — | — | OTLP collector endpoint (`host:port`); unset respects `OTEL_EXPORTER_OTLP_ENDPOINT`. |
| `--otel-sampling-ratio` | `f64` | `1.0` | — | — | Parent-based sampling ratio from 0.0 to 1.0; applies when tracing is enabled. |
| `--otel-excluded-paths` | `string [0..] (repeatable)` | `[]` | — | — | Exact HTTP paths excluded from server spans; omitted or empty uses the default health paths below. |

The default excluded paths are `/health`, `/health_generate`, `/liveness` and
`/readiness`. A nonempty `--otel-excluded-paths` list replaces them. These
tracing flags are not exposed by the Python CLI, even though the Rust library
has tracing support.

## Authentication

| Option | Rust value | Rust default | Python value | Python default | Description |
| --- | --- | --- | --- | --- | --- |
| `--api-key` | `string` | `unset` | `string` | `unset` | Authorization key used for requests to workers. |
| `--api-key-validation-urls` | `string [0..] (repeatable)` | `[]` | — | — | Validation URLs; an empty list falls back to comma-separated `API_KEY_VALIDATION_URLS` from the environment or `.env`. |

## WASM middleware

| Option | Rust value | Rust default | Python value | Python default | Description |
| --- | --- | --- | --- | --- | --- |
| `--wasm-middleware` | `string` | `unset` | `string` | `unset` | Path to a WASM Component Model OnRequest artifact; plugin errors reject the request. |
| `--wasm-middleware-sha256` | `string` | `unset` | `string` | `unset` | Expected artifact SHA-256 hex digest; requires `--wasm-middleware`. |
| `--wasm-middleware-route` | `string (repeatable)` | `[]` | `string (repeatable)` | `[]` | Exact HTTP path invoking middleware; repeat for multiple paths. Empty uses `/v1/chat/completions` when middleware is configured. |

For the middleware contract, resource limits and artifact build instructions,
see [WASM middleware](../README.md#optional-wasm-onrequest-middleware).

## Updating this reference

After changing either CLI, update the corresponding table rows and run:

```bash
python scripts/check_cli_reference.py
```

The check compares option coverage (including the Rust `--runtime` alias),
types, accepted choices, defaults and list/repeat syntax against the Rust
declarations and the actual Python argument parser. It runs in CI and needs
only Python's standard library. Descriptions and runtime fallbacks still need
review when behavior changes.
