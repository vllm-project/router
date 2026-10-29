# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Repository-root resolution for E2E perf (run pytest from the vllm-router repo root)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Built with `cargo build --release` in this repository.
DEFAULT_ROUTER_BINARY = Path("target/release/vllm-router")


def repo_root() -> Path:
    """vllm-router repository root (``tests/python/e2e/helpers`` → four levels up)."""

    return Path(__file__).resolve().parents[4]


def ensure_repo_on_sys_path() -> Path:
    """Insert repo root on ``sys.path``."""

    root = repo_root()
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return root


def resolve_agent_infer_root(*, required: bool = False) -> Path | None:
    """AgentInfer / agentinfer checkout for replay benchmarks."""

    for key in ("AGENTX_AGENT_INFER", "AGENT_INFER_ROOT"):
        raw = os.environ.get(key, "").strip()
        if raw:
            root = Path(raw).expanduser().resolve()
            if not root.is_dir():
                raise FileNotFoundError(f"{key} is not a directory: {root}")
            return root
    if required:
        raise RuntimeError(
            "Set AGENTX_AGENT_INFER or AGENT_INFER_ROOT to an agent-infer source tree."
        )
    return None
