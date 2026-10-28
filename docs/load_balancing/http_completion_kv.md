# HTTP Completion inputs for KV-aware routing

The opt-in Regular HTTP path prepares a single text prompt or token-ID sequence
once before selection. Retries reuse the IDs but select and reserve separately;
Workers receive the original validated JSON bytes and perform normal serving.
The supported cohort is independently addressed DP1 Workers using Normal Full
Attention. Chat and unsupported inputs keep ordinary forwarding.

```sh
export VLLM_ROUTER_MODEL=/models/example
export VLLM_ROUTER_TOKENIZER=/tokenizers/example
export VLLM_ROUTER_COMPLETION_MODEL_ALIAS=public-example
export VLLM_ROUTER_COMPLETION_CACHE_MODEL=example
export VLLM_ROUTER_KV_EVENT_SOURCES='{
  "http://127.0.0.1:8000": {
    "endpoints": ["tcp://127.0.0.1:5557"],
    "replay_endpoints": ["tcp://127.0.0.1:5559"],
    "block_size": 16,
    "model": "example"
  },
  "http://127.0.0.1:8001": {
    "endpoints": ["tcp://127.0.0.1:5558"],
    "block_size": 16,
    "model": "example"
  }
}'
vllm-router --host 127.0.0.1 --port 19000 \
  --worker-urls http://127.0.0.1:8000 http://127.0.0.1:8001 \
  --policy kv_aware --intra-node-data-parallel-size 1
```

Use immutable local model and tokenizer assets matching the Workers' effective
Completion encoding. The model's `config.json` must provide a nonzero
`vocab_size`; the tokenizer directory must include `tokenizer.json` and its
required frontend configuration. `VLLM_ROUTER_TOKENIZER` defaults to the model
directory. Asset changes require restart. The served alias must match the request
model, and the cache model must match event discovery (it defaults to the alias).

The event mapping is JSON, not a file path; model and block size describe each
Worker. Replay, `topic` (default empty) and `hwm` are optional. Do not combine
nonempty compatibility and explicit source maps. Without a mapping, metadata
discovery remains available.

Only matching Dense HTTP sources receive placement credit. Unknown or
incompatible sources remain ordinary candidates; unsupported modifiers lose
credit without losing request fields or output observation. Invalid IDs fail
before dispatch, and tokenizer failures are service errors rather than fallback.

Operators must verify complete Worker input tokens for their deployment: aliases
and bounded probes do not authenticate remote objects or establish full-domain
equivalence. These are observed placement estimates, not actual APC hit counts or
complete bootstrap/generation/recovery guarantees. See
[the HTTP consumer RFC](https://github.com/vllm-project/router/issues/405) for the
design and real-Worker acceptance requirements.
