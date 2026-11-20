# P/D error responses

The vLLM prefill/decode (P/D) router preserves upstream HTTP errors and returns
a JSON error that identifies whether Prefill or Decode failed.

This contract covers direct and service-discovery HTTP P/D routing for
`/v1/chat/completions`, `/v1/completions`, and `/v1/responses`. It also covers
cold HTTP Program chat and completion requests. Program warm-remote, gRPC P/D,
and worker-management requests have separate error handling.

## Status and error type

| Failure before response headers are sent | HTTP status | `error.type` |
| --- | --- | --- |
| Worker returns 429 | 429 | `rate_limit_error` |
| Worker returns another 4xx | Original status | `invalid_request_error` |
| Worker returns 5xx | Original status | `server_error` |
| Upstream timeout before an error status is received | 504 | `server_error` |
| Connection or other transport failure before an error status is received | 502 | `server_error` |
| Required upstream protocol validation or internal router failure | 500 | `server_error` |

If a worker returns an error status and its body stalls or cannot be read,
the router keeps the status it received. For example, a Prefill 503 still
returns 503 if its error body times out.

Other non-2xx statuses, including 3xx, are preserved if they reach the error
handler. This is defensive handling. The HTTP client's redirect behavior is
unchanged, and a 304 response has no body.

## Response body and retry hints

The router returns `Content-Type: application/json` with an `error` object.
For example, a Prefill 503 returns:

```json
{
  "error": {
    "message": "Upstream prefill request failed (HTTP 503).",
    "type": "server_error",
    "code": "503"
  }
}
```

`error.code` is a string and defaults to the HTTP status. The message identifies
the failing worker phase when known. Router failures may return a generic
message.

Details from a worker's 4xx are best-effort. When the router receives a complete
`application/json` body with an `error` object and a string `message`, it appends
the message and can include a string `param` and a string or numeric `code`.
Numeric codes are converted to strings. Each retained string is capped at
2 KiB including a trailing ellipsis if truncated. Slow, incomplete, oversized,
or invalid worker bodies fall back to the generic diagnostic for that status.
Clients can always use the returned HTTP status to identify the error class.

The router preserves a valid upstream `Retry-After` as delay-seconds from 0
through 86,400 or as a supported HTTP date. Other upstream error headers and
raw error bodies are omitted.

## Concurrent requests and streaming

If both phases fail in concurrent routing, the Prefill failure takes precedence.
The router waits for Prefill validation before sending the Decode response.

Once the client has received the status and headers, a transport failure ends
the response body. The status stays the same. A broken stream does not receive
a synthetic SSE error or `[DONE]` marker. Clients should handle an interrupted
stream as an incomplete response.
