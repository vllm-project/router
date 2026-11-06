"""Worker fixtures for mock HTTP workers and real vLLM workers.

Loaded from ``tests/python/conftest.py``. The mock server is
``tests/python/helpers/mock_worker.py``. Real workers are launched by
``VllmWorkerManager``.
"""

from __future__ import annotations

import os
from typing import Iterable

import pytest

from tests.python.helpers.vllm_worker import VllmWorkerManager
from tests.python.helpers.worker_manager import MockWorkerManager


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: mark as end-to-end test")


@pytest.fixture
def mock_worker_manager() -> Iterable[MockWorkerManager]:
    """Mock workers for regular routing and PD disaggregation."""
    mgr = MockWorkerManager()
    try:
        yield mgr
    finally:
        mgr.stop_all()


@pytest.fixture(scope="session")
def _vllm_worker_session() -> Iterable[VllmWorkerManager]:
    """Session-owned vLLM workers reused by ``vllm_worker_manager``."""
    mgr = VllmWorkerManager()
    try:
        yield mgr
    finally:
        mgr.stop_all()


@pytest.fixture
def vllm_worker_manager(
    _vllm_worker_session: VllmWorkerManager,
) -> Iterable[VllmWorkerManager]:
    """Real vLLM workers.

    ``primary`` and ``dp2_pair`` come from the session. ``start`` stops with
    this test.
    """
    mgr = VllmWorkerManager(shared=_vllm_worker_session)
    try:
        yield mgr
    finally:
        mgr.stop_all()


@pytest.fixture(scope="session")
def e2e_model() -> str:
    from tools.e2e.vllm_test_compat import e2e_model_name

    return e2e_model_name()


@pytest.fixture(scope="session")
def e2e_embedding_model() -> str:
    """Embedding model for GPU tests. Override with ``E2E_EMBEDDING_MODEL``."""
    return os.getenv("E2E_EMBEDDING_MODEL", "intfloat/e5-mistral-7b-instruct")
