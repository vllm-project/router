# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Poll HTTP endpoints until ready or timeout."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import httpx

_READY_POLL_INTERVAL_SECONDS = 5


def wait_for_http_ok(
    url: str,
    *,
    timeout_seconds: float,
    proc: subprocess.Popen[str] | None = None,
    log_path: Path | None = None,
    prefix: str = "",
    success_status: int = 200,
) -> None:
    """Poll ``url`` until ``success_status`` or the optional subprocess exits."""

    deadline = time.monotonic() + float(timeout_seconds)
    log_hint = f"; see log {log_path}" if log_path is not None else ""
    while time.monotonic() < deadline:
        if proc is not None:
            exit_code = proc.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"process exited with code {exit_code} before {url} became ready{log_hint}"
                )
        try:
            response = httpx.get(url, timeout=10, trust_env=False)
            if response.status_code == success_status:
                print(f"{prefix}ready: {url}", flush=True)
                return
        except httpx.HTTPError:
            pass
        time.sleep(_READY_POLL_INTERVAL_SECONDS)
    raise RuntimeError(
        f"endpoint did not become ready within {timeout_seconds}s: {url}{log_hint}"
    )
