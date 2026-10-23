"""Model and prompt constants for kv_aware functional e2e.

Workers: **``vllm-rs serve`` + ``grpc://``** only (see ``scripts/backend``).

Optional environment (see also ``function/helpers/launch.py``):

- ``ROUTER_BIN``: path to ``vllm-router`` after ``cargo build --release``
  (same as ``scripts/backend`` / Buildkite). Alias: ``VLLM_ROUTER_BIN``.
- ``VLLM_RS``: path to ``vllm-rs`` binary (else ``PATH`` or wheel under ``vllm``).
- ``VLLM_RS_SERVE_EXTRA``: extra args to ``vllm-rs serve`` (shell-split), e.g.
  ``"--max-model-len 8192 --enable-prefix-caching"``.
- Router logs: run pytest with ``-s``; default ``RUST_LOG=vllm_router_rs=info`` on
  the router subprocess (override in the shell for more/less noise).

Worker GPUs (two ``vllm-rs serve`` processes):

- If ``CUDA_VISIBLE_DEVICES`` is set (e.g. ``0,1,2,3`` or ``2,3,4,5``), worker A/B
  use the **first two entries** in that list (physical ids).
- If unset, query the machine via ``nvidia-smi`` and use physical ``0`` and ``1``
  (requires at least two GPUs on the host).

Prefix-cache scenario (``test_kv_aware_prefix_cache_reuse``):

- ``KV_AWARE_INDEX_SETTLE_SEC``: wait after prime for KV events / index (default 5).
- ``KV_AWARE_MIN_TTFT_GAP_MS``: reuse TTFT must beat control by at least this many ms
  (default 5). gRPC router ``usage`` often omits ``cached_tokens``; the prefix test
  uses client TTFT + optional ``x-router-stages`` instead.
"""

from __future__ import annotations

import os
import subprocess

# HuggingFace id or local path for ``vllm-rs serve`` and ``VLLM_ROUTER_MODEL``.
KV_AWARE_E2E_MODEL = "Qwen/Qwen3-0.6B"

KV_AWARE_SHARED_PREFIX = (
    "You are a concise assistant. Answer in one short sentence. Context block: "
    + ("token-" * 48)
)


def kv_aware_index_settle_sec() -> float:
    raw = os.environ.get("KV_AWARE_INDEX_SETTLE_SEC", "5").strip()
    return float(raw)


def kv_aware_min_ttft_gap_ms() -> float:
    raw = os.environ.get("KV_AWARE_MIN_TTFT_GAP_MS", "5").strip()
    return float(raw)


def _parse_cuda_visible_device_list(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return []
    ids: list[int] = []
    for part in parts:
        if part.isdigit() or (part.startswith("-") and part[1:].isdigit()):
            ids.append(int(part))
        else:
            raise ValueError(f"invalid CUDA_VISIBLE_DEVICES entry {part!r} in {raw!r}")
    return ids


def _machine_physical_gpu_indices() -> list[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is unset and nvidia-smi failed; "
            "set CUDA_VISIBLE_DEVICES to at least two GPU ids"
        ) from exc
    indices = [int(line.strip()) for line in out.splitlines() if line.strip()]
    if len(indices) < 2:
        raise RuntimeError(
            f"kv_aware e2e needs at least two GPUs; nvidia-smi reported {indices!r}"
        )
    return indices


def kv_aware_gpu_pool() -> list[int]:
    """Ordered physical GPU ids available for picking worker A/B."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None and raw.strip():
        pool = _parse_cuda_visible_device_list(raw.strip())
        if len(pool) < 2:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES={raw!r} lists fewer than two GPUs; "
                "kv_aware e2e needs two worker GPUs"
            )
        return pool
    return _machine_physical_gpu_indices()


def kv_aware_worker_gpu_ids() -> tuple[int, int]:
    """Physical GPU indices for worker A and worker B (first two in the pool)."""
    pool = kv_aware_gpu_pool()
    return pool[0], pool[1]
