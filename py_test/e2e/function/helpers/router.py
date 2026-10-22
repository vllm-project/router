"""Start and stop a router-only process for function e2e."""

from __future__ import annotations

from types import SimpleNamespace

from py_test.e2e.function.helpers import launch


def spawn_kv_aware_router(timeout: float = 180.0) -> SimpleNamespace:
    port = launch.find_available_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = launch.popen_launch_router_only(
        base_url,
        policy="kv_aware",
        timeout=timeout,
        router_temperature=0.0,
    )
    return SimpleNamespace(proc=proc, url=base_url)


def terminate_router(handle: SimpleNamespace) -> None:
    launch.terminate_process(getattr(handle, "proc", None))
