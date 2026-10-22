"""Spawn and tear down vLLM HTTP workers for functional e2e."""

from __future__ import annotations

from types import SimpleNamespace

from py_test.e2e.function.helpers.launch import get_e2e_conftest


def spawn_two_single_gpu_workers(model: str) -> list[SimpleNamespace]:
    cf = get_e2e_conftest()
    workers: list[SimpleNamespace] = []

    port_a = cf._find_available_port()
    url_a = f"http://127.0.0.1:{port_a}"
    proc_a = cf._popen_launch_worker(model, url_a, base_gpu_id=0)
    workers.append(SimpleNamespace(proc=proc_a, url=url_a))

    port_b = cf._find_available_port()
    url_b = f"http://127.0.0.1:{port_b}"
    proc_b = cf._popen_launch_worker(model, url_b, base_gpu_id=1)
    workers.append(SimpleNamespace(proc=proc_b, url=url_b))

    return workers


def terminate_workers(workers) -> None:
    cf = get_e2e_conftest()
    for w in workers:
        cf._terminate(w.proc)
