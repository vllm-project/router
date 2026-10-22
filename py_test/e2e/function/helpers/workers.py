"""Spawn and tear down vLLM HTTP workers for functional e2e."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from types import SimpleNamespace

from py_test.e2e.function.helpers import launch


def _require_vllm() -> None:
    if importlib.util.find_spec("vllm") is not None or shutil.which("vllm"):
        return
    raise RuntimeError(
        "vLLM is not installed for the Python running pytest "
        f"({sys.executable}). Install vllm in that env or run:\n"
        "  /path/to/.venv/bin/python -m pytest ... -m e2e -v -s"
    )


def spawn_two_single_gpu_workers(
    model: str,
    *,
    worker_health_timeout: float = 600.0,
) -> list[SimpleNamespace]:
    _require_vllm()
    workers: list[SimpleNamespace] = []

    port_a = launch.find_available_port()
    url_a = f"http://127.0.0.1:{port_a}"
    proc_a = launch.popen_launch_worker(model, url_a, base_gpu_id=0)
    workers.append(SimpleNamespace(proc=proc_a, url=url_a))

    port_b = launch.find_available_port()
    url_b = f"http://127.0.0.1:{port_b}"
    proc_b = launch.popen_launch_worker(model, url_b, base_gpu_id=1)
    workers.append(SimpleNamespace(proc=proc_b, url=url_b))

    for w in workers:
        launch.wait_worker_health(w.url, worker_health_timeout, proc=w.proc)

    return workers


def terminate_workers(workers) -> None:
    for w in workers:
        launch.terminate_process(w.proc)
