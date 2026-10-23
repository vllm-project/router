"""Spawn and tear down vllm-rs gRPC workers for kv_aware functional e2e."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from py_test.e2e.function.helpers import launch
from py_test.e2e.function.helpers.tee_stdio import TeeTextIO
from py_test.e2e.function.kv_aware.config import (
    kv_aware_worker_gpu_ids,
    kv_aware_worker_logs_quiet,
)


def _require_vllm_rs() -> str:
    path = launch.resolve_vllm_rs_bin()
    if path:
        return path
    pytest.skip(
        "vllm-rs not found (set VLLM_RS or install a vllm wheel with vllm-rs); "
        "see scripts/backend/check_vllm_rs.sh"
    )


def spawn_two_vllm_rs_grpc_workers(
    model: str,
    *,
    worker_health_timeout: float = 600.0,
    log_dir: Path | None = None,
) -> list[SimpleNamespace]:
    """Two ``vllm-rs serve`` processes; GPUs from ``kv_aware_worker_gpu_ids``."""
    _require_vllm_rs()
    gpu_a, gpu_b = kv_aware_worker_gpu_ids()
    workers: list[SimpleNamespace] = []
    host = "127.0.0.1"

    if log_dir is None:
        log_dir = Path(tempfile.mkdtemp(prefix="kv_aware_worker_logs_"))
    log_dir.mkdir(parents=True, exist_ok=True)
    quiet_logs = kv_aware_worker_logs_quiet()
    print(
        f"[e2e] worker logs: {log_dir}"
        + (
            " (file only; set KV_AWARE_WORKER_LOG_QUIET=0 for tee to terminal)"
            if quiet_logs
            else " (tee to terminal when pytest -s)"
        ),
        flush=True,
    )

    def _worker_log_sink(path: Path) -> tuple[TeeTextIO | object, str]:
        raw = open(path, "w", encoding="utf-8", buffering=1)
        if quiet_logs:
            return raw, str(path)
        return TeeTextIO(raw), str(path)

    http_a, grpc_a, http_b, grpc_b = launch.find_distinct_available_ports(4)
    print(
        f"[e2e] worker A gpu={gpu_a} http={http_a} grpc={grpc_a}",
        flush=True,
    )
    log_a, log_a_path = _worker_log_sink(log_dir / f"worker_a_grpc_{grpc_a}.log")
    proc_a = launch.popen_launch_vllm_rs_grpc_worker(
        model,
        host=host,
        http_port=http_a,
        grpc_port=grpc_a,
        base_gpu_id=gpu_a,
        log_file=log_a,  # type: ignore[arg-type]
    )
    worker_a = SimpleNamespace(
        proc=proc_a,
        url=f"grpc://{host}:{grpc_a}",
        http_url=f"http://{host}:{http_a}",
        gpu_id=gpu_a,
        log_path=log_a_path,
        log_file=log_a,
    )
    launch.wait_worker_health(
        worker_a.http_url, worker_health_timeout, proc=worker_a.proc
    )
    workers.append(worker_a)

    print(
        f"[e2e] worker B gpu={gpu_b} http={http_b} grpc={grpc_b}",
        flush=True,
    )
    log_b, log_b_path = _worker_log_sink(log_dir / f"worker_b_grpc_{grpc_b}.log")
    proc_b = launch.popen_launch_vllm_rs_grpc_worker(
        model,
        host=host,
        http_port=http_b,
        grpc_port=grpc_b,
        base_gpu_id=gpu_b,
        log_file=log_b,  # type: ignore[arg-type]
    )
    worker_b = SimpleNamespace(
        proc=proc_b,
        url=f"grpc://{host}:{grpc_b}",
        http_url=f"http://{host}:{http_b}",
        gpu_id=gpu_b,
        log_path=log_b_path,
        log_file=log_b,
    )
    launch.wait_worker_health(
        worker_b.http_url, worker_health_timeout, proc=worker_b.proc
    )
    workers.append(worker_b)

    return workers


def terminate_workers(workers) -> None:
    for w in workers:
        launch.terminate_process(w.proc)
        log_f = getattr(w, "log_file", None)
        if log_f is not None:
            try:
                log_f.close()
            except OSError:
                pass
