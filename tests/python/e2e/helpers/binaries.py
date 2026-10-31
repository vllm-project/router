# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Resolve vllm-rs and vllm-router binaries (perf/grpc + router_policy)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from .paths import DEFAULT_ROUTER_BINARY, repo_root

VLLM_RS_ENV = "VLLM_RS"
ROUTER_BIN_ENV = "ROUTER_BIN"
CHECK_VLLM_RS_SCRIPT = Path("scripts/backend/check_vllm_rs.sh")
_CHECK_VLLM_RS_LINE = re.compile(r"^vllm-rs:\s+(.+)$")


def resolve_repo_path(raw: str, repo: Path | None = None) -> Path:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty path")
    base = repo if repo is not None else repo_root()
    path = Path(text).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def resolve_binary_path(raw: str, *, repo: Path | None = None) -> str | None:
    """Absolute file, repo-relative file, or executable name on PATH."""

    text = (raw or "").strip()
    if not text:
        return None
    candidate = resolve_repo_path(text, repo)
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(text)
    return found if found else None


def _vllm_rs_wheel_path() -> Path | None:
    """Same path as ``vllm_rs_bin`` in ``scripts/backend/common.sh``."""

    try:
        import vllm  # noqa: PLC0415
    except ImportError:
        return None
    candidate = Path(vllm.__file__).resolve().parent / "vllm-rs"
    return candidate if candidate.is_file() else None


def _resolve_vllm_rs_via_check_script(repo: Path) -> str | None:
    """Run ``scripts/backend/check_vllm_rs.sh`` (VLLM_RS → PATH → wheel)."""

    script = (repo / CHECK_VLLM_RS_SCRIPT).resolve()
    if not script.is_file():
        return None
    bash = shutil.which("bash")
    if not bash:
        return None
    try:
        proc = subprocess.run(
            [bash, str(script)],
            cwd=repo,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        match = _CHECK_VLLM_RS_LINE.match(line.strip())
        if not match:
            continue
        candidate = Path(match.group(1).strip())
        if candidate.is_file():
            return str(candidate)
    return None


def discover_vllm_rs_bin(*, repo: Path | None = None, skip_env: bool = False) -> str:
    """Match ``scripts/backend/common.sh`` ``vllm_rs_bin`` (+ optional check script)."""

    if not skip_env:
        if pinned := os.environ.get(VLLM_RS_ENV, "").strip():
            return pinned
    base = repo if repo is not None else repo_root()
    if from_script := _resolve_vllm_rs_via_check_script(base):
        return from_script
    if found := shutil.which("vllm-rs"):
        return found
    if wheel := _vllm_rs_wheel_path():
        return str(wheel)
    raise FileNotFoundError(
        f"vllm-rs not found (tried {CHECK_VLLM_RS_SCRIPT}, PATH, wheel); "
        f"export {VLLM_RS_ENV} or run ./{CHECK_VLLM_RS_SCRIPT.as_posix()}"
    )


def resolve_vllm_rs(
    *,
    case_value: str = "",
    repo: Path | None = None,
    label: str = "perf",
) -> str:
    """Env → case JSON path → same discovery as ``check_vllm_rs.sh``."""

    if pinned := os.environ.get(VLLM_RS_ENV, "").strip():
        return pinned
    if resolved := resolve_binary_path(case_value, repo=repo):
        return resolved
    base = repo if repo is not None else repo_root()
    try:
        return discover_vllm_rs_bin(repo=base, skip_env=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label}: {exc}") from exc


def resolve_router_bin(
    *,
    case_value: str = "",
    repo: Path | None = None,
    label: str = "perf",
) -> str:
    base = repo if repo is not None else repo_root()
    if pinned := os.environ.get(ROUTER_BIN_ENV, "").strip():
        return pinned
    if resolved := resolve_binary_path(case_value, repo=base):
        return resolved
    default = (base / DEFAULT_ROUTER_BINARY).resolve()
    if default.is_file():
        return str(default)
    found = shutil.which("vllm-router")
    if found:
        return found
    tried = [str(default)]
    if case_value.strip():
        tried.insert(0, case_value.strip())
    raise FileNotFoundError(
        f"{label}: vllm-router not found (repo root={base}, tried {tried!r}); "
        f"set {ROUTER_BIN_ENV}, case router path, or build target/release/vllm-router"
    )
