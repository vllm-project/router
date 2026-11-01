# P/D error responses

The vLLM prefill/decode (P/D) router preserves upstream failure statuses and
returns a JSON error response that identifies the failing phase. Upstream means
the Prefill or Decode worker; downstream means the client calling the router.

This contract covers request failures in direct and service-discovery P/D
routing, with sequential or concurrent dispatch, for `/v1/chat/completions`,
`/v1/completions`, and `/v1/responses`. Opt-in Program scheduling uses the same
contract for chat completions and completions. Worker-management errors and
other router backends have their own error handling.

## Status mapping

The following mapping applies before the router commits the downstream response
(sends its HTTP status and headers). A successful upstream status does not rule
out a later body-read or required protocol-validation failure.

| Upstream response or failure | Downstream HTTP status | Public `error.type` | Default `error.code` |
| --- | --- | --- | --- |
| HTTP 400 | 400 | `invalid_request_error` | `"400"` |
| HTTP 422 | 422 | `invalid_request_error` | `"422"` |
| HTTP 429 | 429 | `rate_limit_error` | `"429"` |
| Other HTTP 4xx | Preserve the received status | `invalid_request_error` | Status as a string |
| HTTP 500 | 500 | `server_error` | `"500"` |
| HTTP 503 | 503 | `server_error` | `"503"` |
| Other HTTP 5xx | Preserve the received status | `server_error` | Status as a string |
| Other non-2xx response received by the error handler | Preserve the received status | `server_error` | Status as a string |
| Timeout without a received upstream HTTP error status to preserve | 504 | `server_error` | `"504"` |
| Connection refusal or other transport failure without a received upstream HTTP error status to preserve | 502 | `server_error` | `"502"` |
| Required upstream protocol or JSON validation failure | 500 | `server_error` | `"500"` |
| Internal request/response construction, serialization, or configuration failure | 500 | `server_error` | `"500"` |

Once an upstream non-2xx status has been received, an error-body read failure,
timeout, or size limit does not replace that status. For example, a Prefill 503
whose error body times out still produces a downstream 503, rather than 504.
The table describes responses received by the error handler; it does not change
the HTTP client's redirect behavior.

## Error fields

The router generates `Content-Type: application/json` and an `error` object.
For example, a Prefill 503 produces HTTP 503 with:

```json
{
  "error": {
    "message": "Upstream prefill request failed (HTTP 503).",
    "type": "server_error",
    "code": "503"
  }
}
```

| Field | Behavior |
| --- | --- |
| `message` | Identifies `prefill` or `decode` for upstream failures. Eligible 4xx diagnostics are appended to the router-generated phase/status prefix. Internal failures use `The router could not complete the request.` |
| `type` | Generated from the downstream status: 429 maps to `rate_limit_error`, other 4xx to `invalid_request_error`, and other failures to `server_error`. An upstream `type` is not forwarded. |
| `code` | Always a string. Defaults to the downstream HTTP status; an eligible upstream 4xx string or numeric code can override it. |
| `param` | Included only when an eligible upstream 4xx contains a string `param`. Otherwise omitted. |

### Preserving 4xx diagnostics

The router retains selected upstream diagnostics only when all of these
conditions hold:

- The upstream status is 4xx.
- The body was read completely and is at most 64 KiB (65,536 bytes).
- The media type is `application/json`, compared without case sensitivity and
  allowing parameters such as `charset=utf-8`.
- The body parses as JSON with an `error` object containing a string `message`.

The router appends the upstream message to its phase/status prefix, retains a
string `param`, and retains a string `code` or converts a numeric `code` to a
string. A missing, null, or otherwise unsupported `code` keeps the default from
the status table. Unknown fields are omitted.

For example, an HTTP 400 Prefill response with `Content-Type: application/json`
and this complete body:

```json
{
  "error": {
    "message": "max_tokens exceeds the model limit.",
    "type": "BadRequestError",
    "param": "max_tokens",
    "code": "context_length_exceeded"
  }
}
```

produces HTTP 400 with:

```json
{
  "error": {
    "message": "Upstream prefill request failed (HTTP 400). max_tokens exceeds the model limit.",
    "type": "invalid_request_error",
    "param": "max_tokens",
    "code": "context_length_exceeded"
  }
}
```

5xx bodies, non-JSON bodies, malformed envelopes, incomplete bodies, and bodies
over the size limit use the router-generated fallback message and code. The
upstream body is never forwarded wholesale.

## Headers and body capture

The only upstream error response header forwarded to the client is a valid
`Retry-After`: delay-seconds or a supported HTTP-date. Invalid values are omitted.
The upstream `Content-Type` is used to validate 4xx diagnostics; the router
generates its own JSON content type. Other upstream error headers, including
`Content-Encoding`, `Set-Cookie`, and `WWW-Authenticate`, are not forwarded.

Error-body capture saves at most 64 KiB incrementally. It uses the existing total
request timeout, without a separate error-body timeout or a deadline reset per
chunk. At exactly 64 KiB the router can still wait for EOF or the request
deadline to distinguish a complete body from an oversized or incomplete one.

## Failure precedence and response commitment

In sequential dispatch, a Prefill failure prevents Decode dispatch. In
concurrent dispatch, the router waits for Prefill body consumption and required
validation before exposing the Decode response. If both phases fail, the
Prefill failure takes precedence, even when Decode finishes first.

Receiving Prefill non-2xx headers drops the pending local Decode future or held
Decode response before collecting the Prefill error body. Dropping local HTTP
work does not guarantee that the remote engine immediately stops computation.

After the downstream status and headers have been sent, a body transport failure
terminates the body. The committed status cannot change, and the router does
not synthesize an SSE error event or a `[DONE]` marker. This also applies to
oversized non-streaming responses forwarded through the byte-stream path.
Body-read or validation failures detected before commitment use the status
mapping above.

Successful Prefill bodies are bounded to 64 MiB. Non-streaming Decode bodies
are buffered up to 64 MiB for usage reconciliation; larger bodies are forwarded
as streams. Optional JSON enrichment, including logprobs merging, is limited
to 8 MiB. These successful-response limits are separate from the 64 KiB
error-body capture limit.

## Operational diagnostics

Internal error classifications remain separate from public `error.type` values:

| Internal classification | Meaning |
| --- | --- |
| `upstream_http_error` | An upstream non-2xx response was received. |
| `upstream_timeout` | An upstream operation timed out without an HTTP error status to preserve. |
| `upstream_transport_error` | An upstream connection or other transport operation failed. |
| `upstream_protocol_error` | A required upstream response protocol check failed. |
| `internal_error` | Router construction, serialization, or configuration failed. |

`vllm_router_pd_phase_errors_total` records the returned primary failure with
`phase` and internal `error_type` labels; internal failures use `phase=router`.
`vllm_router_pd_error_body_total` records capture outcomes as `complete`,
`limit_exceeded`, or `read_failed`. `vllm_router_pd_response_body_errors_total`
records transport failures after response commitment.

When DEBUG logging is enabled, a completely captured upstream error body can
produce an escaped preview of at most 256 Unicode characters. Incomplete or
oversized bodies produce no preview. This preview is excluded from the
production OpenTelemetry span filter; it does not change the public response.

## Tests

Focused error-contract coverage:

```bash
cargo test --offline --locked --features cpu,ready --lib pd_request_error
cargo test --offline --locked --features cpu,ready --lib pd_dispatch
cargo test --offline --locked --features cpu,ready --test test_pd_error_passthrough
cargo test --offline --locked --features cpu,ready --test test_pd_error_observability
cargo test --offline --locked --features cpu,ready --test test_pd_program_scheduling
```

The passthrough tests include 120 combinations of direct/discovery routing,
sequential/concurrent dispatch, three endpoints, two failure phases, and
400/422/429/500/503 statuses. Additional cases cover transport failures, bounded
capture, failure precedence, cancellation, and failures after commitment.
These tests use mock workers; they do not establish real-model or connector
behavior. Omit `--offline` if dependencies must first be downloaded.
