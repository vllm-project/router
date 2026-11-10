"""Fixtures for router_policy perf cells."""

from __future__ import annotations

import json
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


def _scenario_cells() -> list[tuple[Path, int]]:
    """One JSON per policy; expand its concurrency list into cells."""

    cells: list[tuple[Path, int]] = []
    for path in sorted(CASES_DIR.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        levels = raw.get("concurrencies")
        if not isinstance(levels, list) or not levels:
            levels = [(raw.get("benchmark_params") or {}).get("max-concurrency", 8)]
        cells.extend((path, int(level)) for level in levels)
    return cells


def pytest_generate_tests(metafunc):
    if "case_file" in metafunc.fixturenames and "concurrency" in metafunc.fixturenames:
        cells = _scenario_cells()
        metafunc.parametrize(
            ("case_file", "concurrency"),
            cells,
            ids=[f"{path.stem}/c{level}" for path, level in cells],
        )
