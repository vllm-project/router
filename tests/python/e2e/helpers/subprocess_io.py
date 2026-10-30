# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Subprocess stdout streaming for perf stacks (pytest -s / AGENTX_STREAM_LOGS)."""

from __future__ import annotations

import subprocess
import sys
from typing import Any, TextIO


def sanitize_carriage_return_line(line: str) -> str:
    if "\r" not in line:
        return line
    return line.split("\r")[-1]


def pump_subprocess_stdout(
    proc: subprocess.Popen[Any],
    *,
    log_fp: TextIO | None = None,
    stderr_prefix: str = "",
    sanitize_cr: bool = True,
) -> None:
    """Read merged stdout until EOF; mirror to ``log_fp`` and/or stderr."""

    stream = proc.stdout
    if stream is None:
        return
    try:
        for line in stream:
            if log_fp is not None:
                log_fp.write(line)
                log_fp.flush()
            if stderr_prefix:
                text = sanitize_carriage_return_line(line) if sanitize_cr else line
                sys.stderr.write(f"{stderr_prefix}{text}")
                sys.stderr.flush()
    except ValueError:
        pass
    finally:
        if log_fp is not None:
            log_fp.flush()


def start_stderr_pump_thread(
    proc: subprocess.Popen[Any],
    *,
    log_fp: TextIO | None = None,
    stderr_prefix: str = "",
) -> None:
    import threading

    threading.Thread(
        target=pump_subprocess_stdout,
        args=(proc,),
        kwargs={"log_fp": log_fp, "stderr_prefix": stderr_prefix},
        daemon=True,
    ).start()
