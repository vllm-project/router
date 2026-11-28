import sys
from pathlib import Path

import pytest


def _ensure_repo_root() -> None:
    """``--confcutdir=.../kv_aware`` skips ``tests/python/conftest.py``."""
    root = Path(__file__).resolve().parents[5]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def pytest_configure(config):
    _ensure_repo_root()
    config.addinivalue_line("markers", "e2e: mark as end-to-end test")


def pytest_addoption(parser):
    parser.addoption(
        "--require-gpu-platform",
        choices=("cuda", "rocm"),
        help="Fail a dedicated KV-aware lane if two GPUs are unavailable.",
    )


def pytest_sessionstart(session):
    platform = session.config.getoption("--require-gpu-platform")
    if platform:
        from tests.python.helpers.mark import device_count, is_cuda, is_rocm

        available = is_cuda() if platform == "cuda" else is_rocm()
        count = device_count()
        if not available or count < 2:
            raise pytest.UsageError(
                f"KV-aware lane requires two available {platform} GPUs; visible={count}"
            )


@pytest.fixture(scope="session")
def kv_aware_two_workers():
    """``vllm-rs serve`` ×2; health via each worker OpenAI ``/health`` port."""
    from tests.python.e2e.function.helpers.workers import (
        spawn_two_vllm_rs_grpc_workers,
        terminate_workers,
    )
    from tests.python.e2e.function.kv_aware.config import kv_aware_e2e_model

    workers: list = []
    try:
        workers = spawn_two_vllm_rs_grpc_workers(kv_aware_e2e_model())
        yield workers
    finally:
        terminate_workers(workers)


@pytest.fixture(scope="session")
def kv_aware_router(kv_aware_two_workers):
    """Router with ``--worker-urls`` (regular mode requires workers at startup)."""
    from tests.python.e2e.function.helpers.router import (
        spawn_kv_aware_router,
        terminate_router,
    )
    from tests.python.e2e.function.kv_aware.config import kv_aware_e2e_model

    urls = [w.url for w in kv_aware_two_workers]
    handle = spawn_kv_aware_router(
        worker_urls=urls,
        router_model=kv_aware_e2e_model(),
    )
    try:
        yield handle
    finally:
        terminate_router(handle)
