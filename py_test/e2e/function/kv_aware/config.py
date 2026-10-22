"""Model and prompt constants for kv_aware functional e2e.

Workers: **``vllm-rs serve`` + ``grpc://``** only (see ``scripts/backend``).

Optional environment (see also ``function/helpers/launch.py``):

- ``ROUTER_BIN``: path to ``vllm-router`` after ``cargo build --release``
  (same as ``scripts/backend`` / Buildkite). Alias: ``VLLM_ROUTER_BIN``.
- ``VLLM_RS``: path to ``vllm-rs`` binary (else ``PATH`` or wheel under ``vllm``).
- ``VLLM_RS_SERVE_EXTRA``: extra args to ``vllm-rs serve`` (shell-split), e.g.
  ``"--max-model-len 8192 --enable-prefix-caching"``.
"""

# HuggingFace id or local path for ``vllm-rs serve`` and ``VLLM_ROUTER_MODEL``.
KV_AWARE_E2E_MODEL = "Qwen/Qwen3-0.6B"

KV_AWARE_SHARED_PREFIX = (
    "You are a concise assistant. Answer in one short sentence. Context block: "
    + ("token-" * 48)
)
