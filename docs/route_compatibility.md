# Route compatibility matrix

This page lists how the router handles each inference route. It is a doc,
not a ci check. Engine routes were checked against **vllm v0.32.0**
(`vllm/entrypoints`, released 2026-10-18). Re-check it when the pinned
version moves.

## Handling modes

- **first class**: the router parses the body into a typed request and sends
  it through `route_typed_request`. Policies see routing text, and retries,
  circuit breaker, load tracking, concurrency limit, wasm middleware and per
  route metrics all apply.
- **transparent**: the body is forwarded as json through `route_transparent`.
  Policies see no request text.
- **unsupported**: the router has no route. The request returns 404 unless
  `--enable-transparent-proxy` is set, in which case it is transparent.

## Inference routes (http workers)

| route | regular router | vllm pd router | notes |
|---|---|---|---|
| `/generate` | first class | first class | |
| `/inference/v1/generate` | first class | first class | |
| `/v1/chat/completions` | first class | first class | transparent when epd is configured |
| `/v1/completions` | first class | first class | |
| `/v1/embeddings` | first class | first class | |
| `/rerank`, `/v1/rerank` | first class | first class | |
| `/v1/responses` | transparent | transparent | get, cancel, delete and list input items also routed |
| `/v1/messages` | first class | transparent | anthropic messages api, passthrough only, see below |
| `/v1/messages/count_tokens` | unsupported | unsupported | router side counting waits for the tokenizer path (#348) |
| `/v1/score`, `/score`, `/pooling`, `/classify`, `/v2/embed`, `/v2/rerank` | unsupported | unsupported | |
| `/tokenize`, `/detokenize` | unsupported | unsupported | |
| `/v1/audio/transcriptions`, `/v1/audio/translations` | unsupported | unsupported | multipart bodies |
| `/v1/chat/completions/render`, `/v1/completions/render`, `/v1/messages/render` | unsupported | unsupported | |

gRPC workers only serve `/v1/chat/completions`. Every other typed route,
including `/v1/messages`, is rejected with 400 before a worker is picked.
Anthropic messages over grpc needs a messages rpc in the vllm proto first.

## `/v1/messages`

| topic | handling |
|---|---|
| engines | passthrough to engines that serve `/v1/messages` natively (vllm). no translation to chat completions. engines without the route return their own 404 |
| body | unknown fields are kept and forwarded, so the worker gets the same json the client sent |
| routing text | `system` (string or text blocks), then `tools[].name` and `tools[].description`, then message text in order. image, document, tool_use and tool_result blocks add a short marker, not their payload |
| session affinity | headers first (`x-session-id`, `x-user-id`, ...), then nested `metadata.user_id` |
| router errors | anthropic shape `{"type":"error","error":{"type":...,"message":...}}`, including invalid json and missing fields (400, `invalid_request_error`) |
| worker errors | passed through unchanged |
| non streaming | first class |
| streaming | forwarded as sse; anthropic event parsing for ttft and usage is a follow up |
| `x-api-key` auth, `anthropic-version` / `anthropic-beta` forwarding | follow up |
| pd prefill patch (`max_tokens = 1`) | follow up |

## Other backends

| backend | `/v1/messages` |
|---|---|
| `--backend openai` | not routed (404) |
| `--backend anthropic`, `--backend trtllm` | not implemented yet, these fall back to the regular router, so the first class route applies. trtllm engines without the route return their own 404 |
