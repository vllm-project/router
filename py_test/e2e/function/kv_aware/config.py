"""Model and prompt constants for kv_aware functional e2e.

Optional environment (see also ``function/helpers/launch.py``):

- ``ROUTER_BIN``: path to ``vllm-router`` after ``cargo build --release``
  (same as ``scripts/backend`` / Buildkite). Alias: ``VLLM_ROUTER_BIN``.
- Workers: vLLM **0.3x** uses ``vllm serve`` (auto). Override with
  ``VLLM_E2E_WORKER_MODE=serve`` or ``launch_server`` for older vLLM.
"""

# HuggingFace id or local path passed to vLLM --model-path and API "model".
KV_AWARE_E2E_MODEL = "Qwen/Qwen3-0.6B"

KV_AWARE_SHARED_PREFIX = (
    "You are a concise assistant. Answer in one short sentence. Context block: "
    + ("token-" * 48)
)
