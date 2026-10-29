"""Fixtures for router_policy perf cells."""

from __future__ import annotations

import sys
from pathlib import Path


SCENARIO_DIR = Path(__file__).resolve().parent
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))

CASES_DIR = SCENARIO_DIR / "cases"


def pytest_generate_tests(metafunc):
    if "case_file" in metafunc.fixturenames:
        files = sorted(CASES_DIR.glob("*/*.json"))
        metafunc.parametrize(
            "case_file",
            files,
            ids=[f"{p.parent.name}/{p.stem}" for p in files],
        )
