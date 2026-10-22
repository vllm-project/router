import pytest

from py_test.e2e.function.helpers.router import spawn_kv_aware_router, terminate_router
from py_test.e2e.function.helpers.workers import (
    spawn_two_single_gpu_workers,
    terminate_workers,
)
from py_test.e2e.function.kv_aware.config import KV_AWARE_E2E_MODEL


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: mark as end-to-end test")


@pytest.fixture(scope="session")
def kv_aware_two_workers():
    """Start vLLM workers and wait until each ``/health`` is OK (model load logs go to the terminal)."""
    workers = spawn_two_single_gpu_workers(KV_AWARE_E2E_MODEL)
    try:
        yield workers
    finally:
        terminate_workers(workers)


@pytest.fixture(scope="session")
def kv_aware_router(kv_aware_two_workers):
    """Router with ``--worker-urls`` (regular mode requires workers at startup)."""
    urls = [w.url for w in kv_aware_two_workers]
    handle = spawn_kv_aware_router(worker_urls=urls)
    try:
        yield handle
    finally:
        terminate_router(handle)
