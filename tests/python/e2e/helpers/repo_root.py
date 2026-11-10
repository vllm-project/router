# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""vllm-router repository root for E2E perf paths."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """vllm-router repository root (``tests/python/e2e/helpers`` → four levels up)."""

    return Path(__file__).resolve().parents[4]
