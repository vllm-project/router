"""Fixtures for router_policy perf cells."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest


SCENARIO_DIR = Path(__file__).resolve().parent
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))

CASES_DIR = SCENARIO_DIR / "cases"


@pytest.fixture(autouse=True)
def _router_policy_cell_gap():
    """Let ZMQ/vLLM release ports before the next parametrized cell."""

    yield
    gap = float(os.environ.get("AGENTX_CELL_GAP_SEC", "5"))
    if gap > 0:
        time.sleep(gap)


def pytest_generate_tests(metafunc):
    if "case_file" in metafunc.fixturenames:
        files = sorted(CASES_DIR.glob("*/*.json"))
        metafunc.parametrize(
            "case_file",
            files,
            ids=[f"{p.parent.name}/{p.stem}" for p in files],
        )
