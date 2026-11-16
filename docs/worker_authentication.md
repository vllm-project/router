# Worker authentication

`--api-key` is the bearer key the regular router presents to its workers. It
does not authenticate clients to the router; `--api-key-validation-urls` does.
One key applies to all workers and is read at startup, so changing it takes a
restart.

With the key set, an HTTP worker receives the following `Authorization` header.

| Request | `Authorization` the worker receives |
| --- | --- |
| Health checks: the startup wait, `add_worker` and the periodic check | The key |
| KV-aware routing queries, EPD encoder requests and Program scheduling `/metrics` polling | The key |
| Typed routes such as `/v1/chat/completions`, `/v1/completions` and `/v1/embeddings`, and GET routes such as `/v1/models` | The client's header |
| The transparent route: `POST /v1/responses`, chat completions with `--epd-config`, and paths without a typed handler | The key, in place of the client's header |
| `/flush_cache` and `/get_load` | None ([#586](https://github.com/vllm-project/router/issues/586)) |

A route that forwards the client's header does not add the key, so a worker
that checks the key on those paths needs clients to send it.

gRPC workers receive no credential, on health checks or on inference requests
([#587](https://github.com/vllm-project/router/issues/587)). PD and
`--enable-igw` worker health checks do not carry the key
([#578](https://github.com/vllm-project/router/issues/578)).
