# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Resolve vllm-rs and vllm-router binaries (perf/grpc + router_policy)."""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

from .paths import DEFAULT_ROUTER_BINARY, repo_root

VLLM_RS_ENV = "VLLM_RS"
ROUTER_BIN_ENV = "ROUTER_BIN"


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


def resolve_vllm_rs(
    *,
    case_value: str = "",
    repo: Path | None = None,
    label: str = "perf",
) -> str:
    """Same order as ``scripts/backend/common.sh`` and legacy ``stack_launcher``."""

    if pinned := os.environ.get(VLLM_RS_ENV, "").strip():
        return pinned
    if resolved := resolve_binary_path(case_value, repo=repo):
        return resolved
    found = shutil.which("vllm-rs")
    if found:
        return found
    spec = importlib.util.find_spec("vllm")
    if spec and spec.origin:
        candidate = Path(spec.origin).resolve().parent / "vllm-rs"
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"{label}: vllm-rs not found; export {VLLM_RS_ENV} or run "
        "scripts/backend/check_vllm_rs.sh"
    )


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
