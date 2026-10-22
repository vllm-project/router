import pytest

from py_test.e2e.function.helpers.router import spawn_kv_aware_router, terminate_router
from py_test.e2e.function.helpers.workers import (
    spawn_two_vllm_rs_grpc_workers,
    terminate_workers,
)
from py_test.e2e.function.kv_aware.config import KV_AWARE_E2E_MODEL


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: mark as end-to-end test")


@pytest.fixture(scope="session")
def kv_aware_two_workers():
    """``vllm-rs serve`` ×2; health via each worker OpenAI ``/health`` port."""
    workers = spawn_two_vllm_rs_grpc_workers(KV_AWARE_E2E_MODEL)
    try:
        yield workers
    finally:
        terminate_workers(workers)


@pytest.fixture(scope="session")
def kv_aware_router(kv_aware_two_workers):
    """Router with ``--worker-urls`` (regular mode requires workers at startup)."""
    urls = [w.url for w in kv_aware_two_workers]
    handle = spawn_kv_aware_router(
        worker_urls=urls,
        router_model=KV_AWARE_E2E_MODEL,
    )
    try:
        yield handle
    finally:
        terminate_router(handle)
