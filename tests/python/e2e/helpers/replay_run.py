# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Validate completed AgentInfer replay benchmark runs for router perf E2E."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def validate_completed_run(run_dir: Path) -> dict[str, Any]:
    """Return summary JSON when a run directory contains a completed replay benchmark."""

    try:
        from agentinfer.agentbench.benchkit.compare import load_summary
    except ImportError as exc:
        raise RuntimeError(
            "The agentinfer package is required to validate replay runs "
            "(install AgentInfer / vLLM bench with --agentinfer support)."
        ) from exc

    summary = load_summary(run_dir)
    status = summary["lifecycle"]["status"]
    if status != "completed":
        raise RuntimeError(
            f"benchmark run {run_dir} did not complete (status={status!r})"
        )
    completed = summary["tasks"]["completed"]
    if completed <= 0:
        raise RuntimeError(f"benchmark run {run_dir} has no completed tasks")
    requests = summary["requests"]["requests"]
    if requests <= 0:
        raise RuntimeError(f"benchmark run {run_dir} has no requests")
    return summary
