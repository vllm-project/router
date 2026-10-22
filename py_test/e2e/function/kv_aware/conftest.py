import pytest

from py_test.e2e.function.helpers.workers import (
    spawn_two_single_gpu_workers,
    terminate_workers,
)
from py_test.e2e.function.kv_aware.config import KV_AWARE_E2E_MODEL


@pytest.fixture(scope="session")
def kv_aware_two_workers():
    workers = spawn_two_single_gpu_workers(KV_AWARE_E2E_MODEL)
    try:
        yield workers
    finally:
        terminate_workers(workers)
