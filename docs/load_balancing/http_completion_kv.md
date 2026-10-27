# HTTP Completion inputs for KV-aware routing

The Regular HTTP router can prepare one text prompt or one token-ID sequence
before KV-aware selection. Preparation runs once; retries reuse those IDs while
selection and request accounting run for each attempt. The Worker still receives
the original validated JSON bytes and performs its usual Completion serving.
Chat, batches and unsupported input extensions retain their ordinary path.

This is an opt-in contract for a static, independently addressable DP1 cohort
using Normal Full Attention. The operator must deploy immutable local frontend
assets that match the Workers' effective Completion tokenizer and model
vocabulary. A served-model alias is not remote attestation. Compare complete
Worker input token arrays for the deployed configuration before enabling credit;
bounded probes alone do not prove all inputs or authenticate a remote object.

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

`VLLM_ROUTER_MODEL` must name an immutable local model directory whose
`config.json` supplies a nonzero `vocab_size`. The existing model-keyed frontend
cache resolves the served alias and loads frontend assets from
`VLLM_ROUTER_TOKENIZER`, or from `VLLM_ROUTER_MODEL` when no override is set.
The tokenizer directory may be separate from the model directory; it must
contain `tokenizer.json` and the frontend's other required configuration.
Startup validates the actual loaded frontend source, including cache hits, not
a newly guessed source. The operator must establish that both asset sets match
the Workers. Asset changes require a restart; there is no request-time reload.

The legacy global alias variable configures one verified input group, not a
multi-model registry. `VLLM_ROUTER_COMPLETION_CACHE_MODEL` explicitly names the
model identity reported by KV discovery; it defaults to the exact served alias
for compatibility. Neither identity is inferred from a directory basename.

`VLLM_ROUTER_KV_EVENT_SOURCES` is a JSON string, not a file path. Its explicit
model and block size must describe each actual Worker. The static source shape
uses the shared source configuration; it skips HTTP metadata calls for listed
Workers. `topic` and `hwm` are optional; the default topic remains empty.
Replay endpoints are optional. Without static entries the existing metadata
discovery remains available; absent or incompatible metadata gives ordinary
selection without fabricated KV credit.

The request model must be absent or equal to the configured alias to prepare
Completion tokens. Every available candidate must also have a Dense HTTP source
contract, matching discovered model identity and rank zero before receiving placement credit. Qualification
is per Worker: missing or incompatible sources receive no positive credit but
remain ordinary routing candidates with the full prepared input cost. One
unknown source does not disable credit for verified peers. Other request aliases
use ordinary per-Worker preparation rather than this local input group. LoRA,
salt, session extensions, truncation
and unproved modifiers receive no credit;
legal requests still reach the Worker without dropping their fields. An active
local contract's unsupported request cannot regain credit through the generic
render path. Known
invalid explicit IDs fail before dispatch, and tokenizer failures are service
errors, not an unsupported-input fallback. Scheduling-only integer `priority`
does not change the input tokens; Worker-side scheduling validation still applies.

Dense ingestion rejects a whole batch before mutation when an extension cannot
be represented by its token-only query identity. It also rejects contradictory
residency metadata rather than converting a remote observation into local Device
credit. Live and replay ingestion use the shared strict wire decoder and validate
complete data topics separately from the empty replay terminal. Valid generic
integer hashes, ranks and storage tiers remain accepted.

This provides observed placement estimates with the existing cost model. It does
not establish complete bootstrap, remote generation authority, durable recovery
or actual APC hit counts. Debug logs distinguish prepared input from the tokens
actually admitted to selection; logs never maintain the safety state.
