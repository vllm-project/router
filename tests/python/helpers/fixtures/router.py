"""Router fixture.

``RouterManager`` launches ``vllm_router.launch_router``. CPU tests pass mock
worker URLs or PD pools. GPU tests start a router with no workers and attach
real vLLM workers afterwards.
"""

from __future__ import annotations

from typing import Iterable

import pytest

from tests.python.helpers.router_manager import RouterManager


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark as router integration test")


@pytest.fixture
def router_manager() -> Iterable[RouterManager]:
    mgr = RouterManager()
    try:
        yield mgr
    finally:
        mgr.stop_all()
