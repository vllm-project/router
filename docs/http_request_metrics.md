# HTTP request metrics

These Prometheus series use the `vllm_router_` prefix. This covers known HTTP
generation routes on the regular and vLLM P/D routers, not the gRPC, EPD,
OpenAI-provider proxy or arbitrary transparent paths.

- `active_requests{model}` starts before HTTP admission and lasts until the
  response body ends or is discarded. It uses the selected worker's registered
  model (or `unknown` for an unlabeled worker), with `unresolved` before
  selection. Caller-supplied model names are not metric labels.
- `requests_cancelled_total{stage}` attributes abandonment once. A real
  admission queue wait is `queued`; HTTP preprocessing, selection and retry
  backoff are `routing`; waits on P or D are `prefill` or `decode`. While
  both P and D are outstanding, P wins because it gates the response. After
  handing a successful streaming body to HTTP, the stage is `streaming`.
  An observed terminal event, upstream error or timeout is not cancellation.
- `ttft_seconds{route,stream}` observes the first complete SSE event with
  generated output at the Router's outgoing body boundary. It includes
  admission queueing and Router overhead and is neither client wire latency
  nor engine sampling time. There is no non-stream TTFT. The legacy
  `first_token_ms` diagnostic is unchanged.
- `itl_seconds{route}` is proposed as the Router-observed time between
  adjacent output-bearing SSE deliveries that each confirm exactly one token
  from the same choice via token IDs or corresponding logprobs. It needs
  maintainer confirmation. There is no sample across a multi-token,
  uncountable or co-delivered batch, and no inferred in-frame timing.
- `input_tokens{route}` and `output_tokens{route}` use final engine-reported
  usage on successful responses. JSON usage is recorded on clean completion.
  Chat and Completion SSE require a final usage-only event, `[DONE]` and
  clean EOF. Responses SSE requires `response.completed` and clean EOF.
  An error after usage suppresses it. Missing usage is not zero. The metrics
  do not ask engines for additional usage or change what a caller receives.

The SSE observer understands Chat content, reasoning, refusal and tool/function
**argument** deltas, Completion text, and Responses text, refusal, reasoning and
function-argument deltas. Roles, tool names/IDs without arguments, usage,
terminal events and metadata do not count as generated output. With echo
enabled, timing is omitted because echoed and generated text cannot generally
be separated reliably. Encoded payloads the Router does not decode and
unrecognized native formats have no output timing or usage observations.
