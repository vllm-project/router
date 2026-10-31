"""Shared vllm-router / vllm-rs lookup for the perf suites."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

# File is tests/python/e2e/perf/binaries.py → repo root is parents[4].
REPO_ROOT = Path(__file__).resolve().parents[4]


def resolve_bin_path(raw: str) -> str | None:
    """Resolve a case JSON path (absolute, repo-relative, or PATH name)."""
    text = raw.strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(text)
    return found or None


def router_bin(case_router_bin: str | None = None) -> str:
    if raw := os.environ.get("ROUTER_BIN", "").strip():
        return raw
    if case_router_bin and (resolved := resolve_bin_path(case_router_bin)):
        return resolved
    hint = REPO_ROOT / "target" / "release" / "vllm-router"
    if hint.is_file():
        return str(hint)
    found = shutil.which("vllm-router")
    if found:
        return found
    tried = [str(hint)]
    if case_router_bin:
        tried.insert(0, case_router_bin)
    raise FileNotFoundError(
        f"vllm-router not found (repo root={REPO_ROOT}, tried {tried!r}); "
        "set ROUTER_BIN, stack.router_bin in case JSON, or build target/release/vllm-router"
    )


def preflight_router_bin(router_bin: str) -> None:
    """Fail before starting GPU workers if the router binary cannot run."""
    path = Path(router_bin)
    if path.is_file() and not os.access(path, os.X_OK):
        raise PermissionError(f"vllm-router not executable: {router_bin}")
    try:
        proc = subprocess.run(
            [router_bin, "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(
            f"vllm-router preflight failed ({router_bin!r}): {exc}"
        ) from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-2000:]
        raise RuntimeError(
            f"vllm-router preflight failed ({router_bin} --help exit "
            f"{proc.returncode}): {detail}"
        )


def vllm_rs_bin(case_vllm_rs_bin: str | None = None) -> str:
    if raw := os.environ.get("VLLM_RS", "").strip():
        return raw
    if case_vllm_rs_bin and (resolved := resolve_bin_path(case_vllm_rs_bin)):
        return resolved
    found = shutil.which("vllm-rs")
    if found:
        return found
    import importlib.util

    spec = importlib.util.find_spec("vllm")
    if spec and spec.origin:
        candidate = Path(spec.origin).parent / "vllm-rs"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("vllm-rs not found; set VLLM_RS (vLLM 0.29 rust wheel)")


def graceful_stop(proc: subprocess.Popen, timeout: float = 120.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    start = time.monotonic()
    while proc.poll() is None and time.monotonic() - start < timeout:
        time.sleep(1)
    if proc.poll() is None:
        proc.kill()
