"""Fixtures for router_policy perf cells."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from tests.python.e2e.perf.router_policy.case_loader import concurrency_levels

CASES_DIR = Path(__file__).parent / "cases"


@pytest.fixture(autouse=True)
def _router_policy_cell_gap():
    """Let ZMQ/vLLM release ports before the next parametrized cell."""

    yield
    gap = float(os.environ.get("AGENTX_CELL_GAP_SEC", "5"))
    if gap > 0:
        time.sleep(gap)


def _scenario_cells() -> list[tuple[Path, int]]:
    """One JSON per policy; expand benchmark_params.max-concurrency into cells."""

    cells: list[tuple[Path, int]] = []
    for path in sorted(CASES_DIR.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        cells.extend((path, level) for level in concurrency_levels(raw))
    return cells


def pytest_generate_tests(metafunc):
    if "case_file" in metafunc.fixturenames and "concurrency" in metafunc.fixturenames:
        cells = _scenario_cells()
        metafunc.parametrize(
            ("case_file", "concurrency"),
            cells,
            ids=[f"{path.stem}/c{level}" for path, level in cells],
        )
