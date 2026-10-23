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
- Worker logs: subprocess PIPE → log file + stderr echo when ``pytest -s`` unless
  ``KV_AWARE_WORKER_LOG_QUIET=1`` (file-only under ``[e2e] worker logs: ...``).

Worker GPUs (two ``vllm-rs serve`` processes):

- If ``CUDA_VISIBLE_DEVICES`` is set (e.g. ``0,1,2,3`` or ``2,3,4,5``), worker A/B
  use the **first two entries** in that list (physical ids).
- If unset, query the machine via ``nvidia-smi`` and use physical ``0`` and ``1``
  (requires at least two GPUs on the host).

Routing scenario (``test_kv_aware_*`` under ``kv_aware/``):

- ``KV_AWARE_INDEX_SETTLE_SEC``: wait after prime for KV events / index (default 5).
- ``KV_AWARE_REQUEST_LOG_POLL_SEC``: max wait for ``request_id`` in worker logs (default 60).
- ``KV_AWARE_REUSE_ROUNDS`` / ``KV_AWARE_CONTROL_ROUNDS``: counts for the multi-round
  routing probe (default 5 each).
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


def kv_aware_request_log_poll_sec() -> float:
    raw = os.environ.get("KV_AWARE_REQUEST_LOG_POLL_SEC", "60").strip()
    return float(raw)


def kv_aware_worker_logs_quiet() -> bool:
    raw = os.environ.get("KV_AWARE_WORKER_LOG_QUIET", "").strip().lower()
    return raw in ("1", "true", "yes")


def kv_aware_reuse_rounds() -> int:
    raw = os.environ.get("KV_AWARE_REUSE_ROUNDS", "5").strip()
    value = int(raw)
    if value < 1:
        raise ValueError("KV_AWARE_REUSE_ROUNDS must be >= 1")
    return value


def kv_aware_control_rounds() -> int:
    raw = os.environ.get("KV_AWARE_CONTROL_ROUNDS", "5").strip()
    value = int(raw)
    if value < 1:
        raise ValueError("KV_AWARE_CONTROL_ROUNDS must be >= 1")
    return value


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
