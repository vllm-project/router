# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Pytest options for E2E benchmark runs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Import paths for agentaware perf (see `agentaware/run_benchmark.py`).
_E2E_DIR = Path(__file__).resolve().parents[1]
_AGENTAWARE_DIR = _E2E_DIR / "perf" / "agentaware"
_AGENTAWARE_HELPERS = _AGENTAWARE_DIR / "helpers"
for _path in (_AGENTAWARE_HELPERS, _AGENTAWARE_DIR, _E2E_DIR):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("e2e", "E2E performance benchmark options")
    group.addoption(
        "--test-config-file",
        action="store",
        default=None,
        metavar="PATH",
        help="Agent-aware perf case JSON (path relative to vllm-router repo root is fine).",
    )
    group.addoption(
        "--task-num",
        action="store",
        type=int,
        default=None,
        help="Override benchmark_params task-num",
    )
    group.addoption(
        "--max-concurrency",
        action="store",
        type=int,
        default=None,
        help="Override benchmark_params max-concurrency",
    )
    group.addoption(
        "--benchmark-run-as-user",
        action="store",
        default=None,
        metavar="USER",
        help="For plan-subagent only: run bench subprocess as USER via sudo when pytest runs as root",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "e2e_perf: GPU/NPU router replay performance E2E"
    )


@pytest.fixture(scope="session", autouse=True)
def _perf_tests_use_repo_root_cwd() -> None:
    """Resolve repo-relative paths in case JSON/YAML the same way as subprocess cwd."""

    root = Path(__file__).resolve().parents[3]
    previous = os.getcwd()
    os.chdir(root)
    yield
    os.chdir(previous)
