"""Shared fixtures for manual perf e2e tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_E2E_DIR = Path(__file__).resolve().parents[1]
_HELPERS = _E2E_DIR / "helpers"
for _entry in (_HELPERS, _E2E_DIR):
    _path = str(_entry)
    if _path not in sys.path:
        sys.path.insert(0, _path)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "perf: manual performance benchmark (requires RUN_ROUTER_E2E_PERF=1)",
    )


def _perf_enabled() -> bool:
    return os.environ.get("RUN_ROUTER_E2E_PERF", "").strip() == "1"


@pytest.fixture(scope="session")
def require_router_e2e_perf():
    if not _perf_enabled():
        pytest.skip(
            "Perf e2e disabled; set RUN_ROUTER_E2E_PERF=1 and scenario env (see perf/README.md)"
        )
