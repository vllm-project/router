# Route compatibility

How the router handles each inference route on HTTP workers. Engine routes
were checked against vLLM v0.32.0; re-check when the pinned version moves.
Routing behaviour per policy is in [load_balancing](load_balancing/README.md).

- **first class**: parsed into a typed request; policies see the request
  text, and retries, circuit breaker, load tracking, concurrency limit, WASM
  middleware and per route metrics apply.
- **transparent**: forwarded as JSON; policies see no request text.
- **unsupported**: no route; 404 unless `--enable-transparent-proxy` is set,
  which makes it transparent.

| route | regular router | vLLM P/D router | notes |
|---|---|---|---|
| `/generate`, `/inference/v1/generate` | first class | first class | |
| `/v1/chat/completions` | first class | first class | transparent when EPD is configured |
| `/v1/completions`, `/v1/embeddings`, `/rerank`, `/v1/rerank` | first class | first class | |
| `/v1/responses` | transparent | transparent | get, cancel, delete and list input items also routed |
| `/v1/messages` | first class | transparent | Anthropic Messages API, see below; `--backend openai` does not route it, `--backend anthropic` and `--backend trtllm` use the regular router |
| `/v1/messages/count_tokens` | unsupported | unsupported | |
| `/v1/score`, `/score`, `/pooling`, `/classify`, `/v2/embed`, `/v2/rerank` | unsupported | unsupported | |
| `/tokenize`, `/detokenize`, `*/render` | unsupported | unsupported | |
| `/v1/audio/transcriptions`, `/v1/audio/translations` | unsupported | unsupported | multipart bodies |

gRPC pools serve `/v1/chat/completions` and `/v1/completions` (a gRPC P/D
pool serves chat only); other typed routes, including `/v1/messages`, are
rejected with 400.

## `/v1/messages`

- The body is passed through unchanged, unknown fields included, to engines
  that serve `/v1/messages` natively (vLLM). There is no translation to chat
  completions; engines without the route return their own 404.
- Prefix-cache aware policies (`cache_aware`, `kv_aware`) work for text-only
  requests; requests with images or documents route least loaded.
- Errors raised by the regular router use the Anthropic shape
  `{"type":"error","error":{"type":...,"message":...}}`. Worker errors, and
  errors from the P/D router, are passed through unchanged.
- Streaming responses are forwarded as SSE.
- Not supported yet: `x-api-key` auth, and TTFT and usage metrics for
  streaming responses.
