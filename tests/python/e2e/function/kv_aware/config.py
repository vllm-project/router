"""Model and prompt constants for kv_aware functional e2e.

Workers: ``mock-grpc-worker`` (``tests/python/fixtures/mock_grpc_worker``) on
``grpc://`` URLs. The fixture builds it with ``cargo build`` unless
``MOCK_GRPC_WORKER`` points at a binary. The router still tokenizes with
``kv_aware_e2e_model()`` / ``VLLM_ROUTER_MODEL``. The ``vllm-rs`` launcher
below is unused by this fixture.

Optional environment (see also ``function/helpers/launch.py``):

- ``ROUTER_BIN``: path to ``vllm-router`` after ``cargo build --release``
  (same as ``scripts/backend`` / Buildkite). Alias: ``VLLM_ROUTER_BIN``.
- ``VLLM_RS``: path to ``vllm-rs`` binary (else ``PATH`` or wheel under ``vllm``).
- Model for workers + router (HF id or local dir), first match wins:
  ``KV_AWARE_E2E_MODEL``, ``MODEL``, ``VLLM_ROUTER_MODEL``, ``MODEL_DIR``;
  default ``Qwen/Qwen3-0.6B`` (see ``kv_aware_e2e_model()``).
- ``KV_AWARE_USE_VLLM_SERVE``: ``0`` (default) use ``vllm-rs serve``.
  ``1`` is rejected: ``vllm serve`` cannot be a ``grpc://`` worker.
- Worker serve extras (shell-split, first set wins):
  ``KV_AWARE_WORKER_SERVE_EXTRA``, ``VLLM_SERVE_EXTRA``, ``VLLM_RS_SERVE_EXTRA``;
  default ``--enable-prefix-caching`` when all unset.
- ``KV_AWARE_KV_EVENTS``: ``1`` (default) append ``--kv-events-config`` per worker; ``0`` to disable.
- ``KV_AWARE_KV_EVENTS_REPLAY``: ``1`` (default) add ``replay_endpoint`` per worker (DEALER replay).
- ``KV_AWARE_KV_EVENTS_ENDPOINT_BIND``: PUB bind in JSON ``endpoint`` (default ``*`` → ``tcp://*:port``).
- ``KV_AWARE_KV_EVENTS_HOST``: TCP host for ``replay_endpoint`` (default ``127.0.0.1``).
- Router logs: run pytest with ``-s``; default ``RUST_LOG=vllm_router_rs=info`` on
  the router subprocess (override in the shell for more/less noise).
- Worker logs: subprocess PIPE → log file + stderr echo when ``pytest -s`` unless
  ``KV_AWARE_WORKER_LOG_QUIET=1`` (file-only under ``[e2e] worker logs: ...``).

Worker GPUs (two worker serve processes):

- If ``CUDA_VISIBLE_DEVICES`` is set (e.g. ``0,1,2,3`` or ``2,3,4,5``), worker A/B
  use the **first two entries** in that list (physical ids).
- If unset, query the machine via ``nvidia-smi`` and use physical ``0`` and ``1``
  (requires at least two GPUs on the host).

Routing scenario (``test_kv_aware_*`` under ``kv_aware/``):

- ``KV_AWARE_INDEX_SETTLE_SEC``: wait after prime for KV events / index (default 5).
- ``KV_AWARE_REQUEST_LOG_POLL_SEC``: max wait for ``request_id`` in worker logs (default 60).
- ``KV_AWARE_REUSE_ROUNDS``: reuse rounds per lane in the dual-lane test (default 3).
- ``KV_AWARE_DUAL_PRIME_CONCURRENT``: ``1`` (default) fire lane A/B primes together;
  ``0`` serial primes (more reliable one-worker-per-lane on cold start).

Dual-lane test **checkpoints** (terminal ``[e2e] checkpoint: ...``): prompt contract,
two-worker prime, index settle, then **first B reuse before any A reuse**.
"""

from __future__ import annotations

import os
import shlex
import subprocess

# Default when no model env is set (see ``kv_aware_e2e_model()``).
KV_AWARE_E2E_MODEL_DEFAULT = "Qwen/Qwen3-0.6B"


def kv_aware_e2e_model() -> str:
    """HF model id or local path for ``vllm-rs serve`` and ``VLLM_ROUTER_MODEL``."""
    for key in ("KV_AWARE_E2E_MODEL", "MODEL", "VLLM_ROUTER_MODEL", "MODEL_DIR"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw
    return KV_AWARE_E2E_MODEL_DEFAULT


KV_AWARE_SHARED_PREFIX = (
    "You are a concise assistant. Answer in one short sentence. Context block: "
    + ("token-" * 48)
)

# Independent long prefixes so two lanes do not share KV block hashes.
KV_AWARE_LANE_A_PREFIX = "Lane A assistant. One short sentence. Context block A: " + (
    "alpha-" * 48
)
KV_AWARE_LANE_B_PREFIX = "Lane B assistant. One short sentence. Context block B: " + (
    "beta-" * 48
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
    raw = os.environ.get("KV_AWARE_REUSE_ROUNDS", "3").strip()
    value = int(raw)
    if value < 1:
        raise ValueError("KV_AWARE_REUSE_ROUNDS must be >= 1")
    return value


def kv_aware_dual_prime_concurrent() -> bool:
    raw = os.environ.get("KV_AWARE_DUAL_PRIME_CONCURRENT", "1").strip().lower()
    return raw not in ("0", "false", "no")


def _env_flag(name: str, *, default: str = "1") -> bool:
    raw = os.environ.get(name, default).strip().lower()
    return raw not in ("0", "false", "no")


def kv_aware_use_vllm_serve() -> bool:
    return _env_flag("KV_AWARE_USE_VLLM_SERVE", default="0")


def kv_aware_kv_events_enabled() -> bool:
    return _env_flag("KV_AWARE_KV_EVENTS", default="1")


def kv_aware_kv_events_replay_enabled() -> bool:
    return _env_flag("KV_AWARE_KV_EVENTS_REPLAY", default="1")


def kv_aware_kv_events_endpoint_bind() -> str:
    raw = os.environ.get("KV_AWARE_KV_EVENTS_ENDPOINT_BIND", "*").strip()
    return raw or "*"


def kv_aware_kv_events_replay_host() -> str:
    raw = os.environ.get("KV_AWARE_KV_EVENTS_HOST", "127.0.0.1").strip()
    return raw or "127.0.0.1"


def worker_serve_extra_argv() -> list[str]:
    for key in (
        "KV_AWARE_WORKER_SERVE_EXTRA",
        "VLLM_SERVE_EXTRA",
        "VLLM_RS_SERVE_EXTRA",
    ):
        raw = os.environ.get(key, "").strip()
        if raw:
            return shlex.split(raw)
    return ["--enable-prefix-caching"]


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
