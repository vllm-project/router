# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Pytest E2E router-replay perf benchmark driven by a JSON case file."""

from __future__ import annotations

from pathlib import Path

import pytest

from _runtime import RouterReplayE2EConfig, run_router_replay
from helpers.replay_run import validate_completed_run


@pytest.mark.e2e_perf
def test_router_replay_completes(pytestconfig: pytest.Config) -> None:
    """Run one router-replay case selected by ``--test-config-file``."""

    test_config_file = pytestconfig.getoption("--test-config-file")
    if not test_config_file:
        pytest.skip("--test-config-file is required for router-replay perf benchmarks")

    config = RouterReplayE2EConfig.from_case_file(
        Path(test_config_file),
        task_num=pytestconfig.getoption("--task-num"),
        max_concurrency=pytestconfig.getoption("--max-concurrency"),
    )
    run_dir = run_router_replay(config)
    summary = validate_completed_run(run_dir)
    print(
        f"[agentaware:{config.router_arm}] completed: "
        f"tasks={summary['tasks']['completed']} requests={summary['requests']['requests']} "
        f"dir={run_dir}",
        flush=True,
    )
