"""Spawn and tear down vllm-rs gRPC workers for kv_aware functional e2e."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from py_test.e2e.function.helpers import launch
from py_test.e2e.function.kv_aware.config import kv_aware_worker_gpu_ids


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
    """Two ``vllm-rs serve`` processes; GPUs from ``KV_AWARE_E2E_GPU_IDS``."""
    _require_vllm_rs()
    gpu_a, gpu_b = kv_aware_worker_gpu_ids()
    workers: list[SimpleNamespace] = []
    host = "127.0.0.1"

    http_a, grpc_a, http_b, grpc_b = launch.find_distinct_available_ports(4)
    print(
        f"[e2e] worker A gpu={gpu_a} http={http_a} grpc={grpc_a}",
        flush=True,
    )
    proc_a = launch.popen_launch_vllm_rs_grpc_worker(
        model, host=host, http_port=http_a, grpc_port=grpc_a, base_gpu_id=gpu_a
    )
    worker_a = SimpleNamespace(
        proc=proc_a,
        url=f"grpc://{host}:{grpc_a}",
        http_url=f"http://{host}:{http_a}",
        gpu_id=gpu_a,
    )
    launch.wait_worker_health(
        worker_a.http_url, worker_health_timeout, proc=worker_a.proc
    )
    workers.append(worker_a)

    print(
        f"[e2e] worker B gpu={gpu_b} http={http_b} grpc={grpc_b}",
        flush=True,
    )
    proc_b = launch.popen_launch_vllm_rs_grpc_worker(
        model, host=host, http_port=http_b, grpc_port=grpc_b, base_gpu_id=gpu_b
    )
    worker_b = SimpleNamespace(
        proc=proc_b,
        url=f"grpc://{host}:{grpc_b}",
        http_url=f"http://{host}:{http_b}",
        gpu_id=gpu_b,
    )
    launch.wait_worker_health(
        worker_b.http_url, worker_health_timeout, proc=worker_b.proc
    )
    workers.append(worker_b)

    return workers


def terminate_workers(workers) -> None:
    for w in workers:
        launch.terminate_process(w.proc)
