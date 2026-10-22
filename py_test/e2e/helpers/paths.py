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
    """vllm-router repository root (``py_test/e2e/helpers`` → four levels up)."""

    return Path(__file__).resolve().parents[3]


def ensure_repo_on_sys_path() -> Path:
    """Insert repo root on ``sys.path`` (for ``tools.calibration`` and other repo modules)."""

    root = repo_root()
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return root


def _has_calibration_tools(root: Path) -> bool:
    return (root / "tools" / "calibration" / "calibrate_prefill.py").is_file()


def resolve_agent_infer_root(*, required: bool = False) -> Path | None:
    """Optional external AgentInfer clone (legacy override when vendored tools are absent)."""

    raw = os.environ.get("AGENT_INFER_ROOT", "").strip()
    if not raw:
        if required:
            raise RuntimeError(
                "Set AGENT_INFER_ROOT to an agent-infer source tree with tools/calibration/, "
                "or use the vendored tools/calibration/ in this router repository."
            )
        return None
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"AGENT_INFER_ROOT is not a directory: {root}")
    if not _has_calibration_tools(root):
        raise FileNotFoundError(
            f"AGENT_INFER_ROOT has no tools/calibration/calibrate_prefill.py: {root}"
        )
    return root


def resolve_calibration_root(*, required: bool = True) -> Path:
    """Root used for ``python -m tools.calibration.*`` (vendored in router repo by default)."""

    router = repo_root()
    if _has_calibration_tools(router):
        return router
    external = resolve_agent_infer_root(required=False)
    if external is not None:
        return external
    if required:
        raise RuntimeError(
            "L1 calibration needs tools/calibration/ under the vllm-router repo, or AGENT_INFER_ROOT "
            "pointing at an agent-infer source checkout."
        )
    return router


def calibration_subprocess_env(calibration_root: Path) -> dict[str, str]:
    """Environment for ``python -m tools.calibration.*`` subprocesses."""

    env = os.environ.copy()
    entry = str(calibration_root)
    existing = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = entry if not existing else f"{entry}{os.pathsep}{existing}"
    return env


def ensure_agent_infer_tools_on_path() -> Path | None:
    """Legacy alias: ensure repo (or AGENT_INFER_ROOT) is importable for calibration."""

    root = resolve_calibration_root(required=False)
    if root is None:
        return None
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return root
