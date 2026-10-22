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
def kv_aware_router():
    """Router-only with policy=kv_aware (does not use py_test/e2e/conftest)."""
    handle = spawn_kv_aware_router()
    try:
        yield handle
    finally:
        terminate_router(handle)


@pytest.fixture(scope="session")
def kv_aware_two_workers():
    workers = spawn_two_single_gpu_workers(KV_AWARE_E2E_MODEL)
    try:
        yield workers
    finally:
        terminate_workers(workers)
