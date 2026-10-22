"""Model and prompt constants for kv_aware functional e2e."""

# HuggingFace id or local path passed to vLLM --model-path and API "model".
KV_AWARE_E2E_MODEL = "Qwen/Qwen3-0.6B"

KV_AWARE_SHARED_PREFIX = (
    "You are a concise assistant. Answer in one short sentence. Context block: "
    + ("token-" * 48)
)
