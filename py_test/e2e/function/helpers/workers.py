"""Spawn and tear down vllm-rs gRPC workers for kv_aware functional e2e."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from py_test.e2e.function.helpers import launch


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
) -> list[SimpleNamespace]:
    """Two ``vllm-rs serve`` processes (GPU 0/1); Router uses ``grpc://`` URLs."""
    _require_vllm_rs()
    workers: list[SimpleNamespace] = []
    host = "127.0.0.1"

    http_a = launch.find_available_port()
    grpc_a = launch.find_available_port()
    proc_a = launch.popen_launch_vllm_rs_grpc_worker(
        model, host=host, http_port=http_a, grpc_port=grpc_a, base_gpu_id=0
    )
    workers.append(
        SimpleNamespace(
            proc=proc_a,
            url=f"grpc://{host}:{grpc_a}",
            http_url=f"http://{host}:{http_a}",
        )
    )

    http_b = launch.find_available_port()
    grpc_b = launch.find_available_port()
    proc_b = launch.popen_launch_vllm_rs_grpc_worker(
        model, host=host, http_port=http_b, grpc_port=grpc_b, base_gpu_id=1
    )
    workers.append(
        SimpleNamespace(
            proc=proc_b,
            url=f"grpc://{host}:{grpc_b}",
            http_url=f"http://{host}:{http_b}",
        )
    )

    for w in workers:
        launch.wait_worker_health(w.http_url, worker_health_timeout, proc=w.proc)

    return workers


def terminate_workers(workers) -> None:
    for w in workers:
        launch.terminate_process(w.proc)
