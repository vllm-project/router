"""Spawn and tear down vllm-rs gRPC workers for kv_aware functional e2e."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.python.e2e.function.helpers import launch
from tests.python.e2e.function.helpers.worker_log_pump import SubprocessLogPump
from tests.python.e2e.function.kv_aware.config import (
    kv_aware_kv_events_enabled,
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
    echo = not quiet_logs
    print(
        f"[e2e] worker logs: {log_dir}"
        + (
            " (file only; unset KV_AWARE_WORKER_LOG_QUIET for terminal echo)"
            if quiet_logs
            else " (PIPE tee to stderr when pytest -s)"
        ),
        flush=True,
    )

    port_count = 6 if kv_aware_kv_events_enabled() else 4
    ports = launch.find_distinct_available_ports(port_count)
    http_a, grpc_a, http_b, grpc_b = ports[:4]
    zmq_a = ports[4] if port_count == 6 else None
    zmq_b = ports[5] if port_count == 6 else None
    print(
        f"[e2e] worker A gpu={gpu_a} http={http_a} grpc={grpc_a}"
        + (f" kv_events_zmq={zmq_a}" if zmq_a is not None else ""),
        flush=True,
    )
    log_a_path = log_dir / f"worker_a_grpc_{grpc_a}.log"
    proc_a = launch.popen_launch_vllm_rs_grpc_worker(
        model,
        host=host,
        http_port=http_a,
        grpc_port=grpc_a,
        base_gpu_id=gpu_a,
        kv_events_zmq_port=zmq_a,
        capture_output=True,
    )
    pump_a = SubprocessLogPump(
        proc_a,
        log_a_path,
        echo=echo,
        prefix=f"[worker A grpc={grpc_a}] ",
    )
    worker_a = SimpleNamespace(
        proc=proc_a,
        url=f"grpc://{host}:{grpc_a}",
        http_url=f"http://{host}:{http_a}",
        gpu_id=gpu_a,
        log_path=str(log_a_path),
        log_pump=pump_a,
    )
    launch.wait_worker_health(
        worker_a.http_url, worker_health_timeout, proc=worker_a.proc
    )
    workers.append(worker_a)

    print(
        f"[e2e] worker B gpu={gpu_b} http={http_b} grpc={grpc_b}"
        + (f" kv_events_zmq={zmq_b}" if zmq_b is not None else ""),
        flush=True,
    )
    log_b_path = log_dir / f"worker_b_grpc_{grpc_b}.log"
    proc_b = launch.popen_launch_vllm_rs_grpc_worker(
        model,
        host=host,
        http_port=http_b,
        grpc_port=grpc_b,
        base_gpu_id=gpu_b,
        kv_events_zmq_port=zmq_b,
        capture_output=True,
    )
    pump_b = SubprocessLogPump(
        proc_b,
        log_b_path,
        echo=echo,
        prefix=f"[worker B grpc={grpc_b}] ",
    )
    worker_b = SimpleNamespace(
        proc=proc_b,
        url=f"grpc://{host}:{grpc_b}",
        http_url=f"http://{host}:{http_b}",
        gpu_id=gpu_b,
        log_path=str(log_b_path),
        log_pump=pump_b,
    )
    launch.wait_worker_health(
        worker_b.http_url, worker_health_timeout, proc=worker_b.proc
    )
    workers.append(worker_b)

    return workers


def terminate_workers(workers) -> None:
    for w in workers:
        launch.terminate_process(w.proc)
        pump = getattr(w, "log_pump", None)
        if pump is not None:
            pump.close()
        log_f = getattr(w, "log_file", None)
        if log_f is not None:
            try:
                log_f.close()
            except OSError:
                pass
