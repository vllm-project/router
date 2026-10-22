"""Spawn and tear down vLLM HTTP workers for functional e2e."""

from __future__ import annotations

from types import SimpleNamespace

from py_test.e2e.function.helpers import launch


def spawn_two_single_gpu_workers(model: str) -> list[SimpleNamespace]:
    workers: list[SimpleNamespace] = []

    port_a = launch.find_available_port()
    url_a = f"http://127.0.0.1:{port_a}"
    proc_a = launch.popen_launch_worker(model, url_a, base_gpu_id=0)
    workers.append(SimpleNamespace(proc=proc_a, url=url_a))

    port_b = launch.find_available_port()
    url_b = f"http://127.0.0.1:{port_b}"
    proc_b = launch.popen_launch_worker(model, url_b, base_gpu_id=1)
    workers.append(SimpleNamespace(proc=proc_b, url=url_b))

    return workers


def terminate_workers(workers) -> None:
    for w in workers:
        launch.terminate_process(w.proc)
