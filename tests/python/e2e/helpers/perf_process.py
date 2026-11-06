# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Subprocess lifecycle shared by perf/grpc and router_policy stacks."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, IO, TextIO

from .http_ready import wait_for_http_ok
from .subprocess_io import pump_subprocess_stdout, start_stderr_pump_thread


def merge_process_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    merged = os.environ.copy()
    if overrides:
        merged.update(overrides)
    return merged


def popen_merged_output(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    start_new_session: bool = False,
    text: bool = True,
    bufsize: int = -1,
) -> subprocess.Popen[Any]:
    """Start a process with stdout+stderr merged to a pipe (for log pumping)."""

    return subprocess.Popen(
        cmd,
        env=merge_process_env(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=text,
        bufsize=bufsize,
        start_new_session=start_new_session,
    )


def attach_stderr_log_pump(
    proc: subprocess.Popen[Any],
    *,
    log_prefix: str,
    log_fp: TextIO | None = None,
) -> None:
    start_stderr_pump_thread(proc, log_fp=log_fp, stderr_prefix=f"[{log_prefix}] ")


def start_process_with_log(
    command: list[str],
    log_path: Path,
    env: dict[str, str],
    *,
    stream_to_stderr: bool = False,
    stderr_prefix: str = "",
    start_new_session: bool = True,
) -> tuple[subprocess.Popen[Any], threading.Thread | None, IO[Any] | None]:
    """Start a process; write merged output to ``log_path`` and optionally stderr."""

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "w", encoding="utf-8")
    if stream_to_stderr and stderr_prefix:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=start_new_session,
            text=True,
            bufsize=1,
        )
        thread = threading.Thread(
            target=pump_subprocess_stdout,
            args=(proc,),
            kwargs={"log_fp": log_fp, "stderr_prefix": stderr_prefix},
            name=f"stream-{stderr_prefix.strip() or log_path.stem}",
            daemon=True,
        )
        thread.start()
        return proc, thread, log_fp

    proc = subprocess.Popen(
        command,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=start_new_session,
    )
    return proc, None, log_fp


def join_log_pump(
    proc: subprocess.Popen[Any],
    thread: threading.Thread | None,
    *,
    join_timeout: float = 15.0,
) -> None:
    if proc.stdout is not None:
        proc.stdout.close()
    if thread is not None:
        thread.join(timeout=join_timeout)


def graceful_stop_subprocess(
    proc: subprocess.Popen[Any],
    timeout: float = 120.0,
    *,
    process_group: bool = False,
    kill_grace: float = 15.0,
) -> None:
    """Terminate then kill; optional process group (``start_new_session=True``)."""

    if proc.poll() is not None:
        return
    if process_group:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        proc.wait(timeout=kill_grace)
        return

    proc.terminate()
    start = time.monotonic()
    while proc.poll() is None and time.monotonic() - start < timeout:
        time.sleep(1)
    if proc.poll() is None:
        proc.kill()


def subprocess_exit_error(
    proc: subprocess.Popen[Any],
    label: str,
    *,
    tail_chars: int = 4000,
) -> RuntimeError:
    out = proc.stdout.read() if proc.stdout else ""
    if not isinstance(out, str):
        out = ""
    return RuntimeError(
        f"{label} exited early (code={proc.returncode}): {out[-tail_chars:]}"
    )


def router_env_for_model(model: str) -> dict[str, str]:
    """Env for ``vllm-router`` (matches ``scripts/backend/common.sh``)."""

    env: dict[str, str] = {"VLLM_ROUTER_MODEL": model}
    model_path = Path(model)
    tok = model_path / "tokenizer.json"
    if model_path.is_dir() and tok.is_file():
        env.setdefault("VLLM_ROUTER_TOKENIZER", str(tok))
    if stages := os.environ.get("VLLM_ROUTER_STAGES"):
        env["VLLM_ROUTER_STAGES"] = stages
    return env


def preflight_router_bin(router_bin: str, *, help_timeout: float = 30.0) -> None:
    """Fail before GPU workers if ``vllm-router`` cannot run ``--help``."""

    path = Path(router_bin)
    if path.is_file() and not os.access(path, os.X_OK):
        raise PermissionError(f"vllm-router not executable: {router_bin}")
    try:
        proc = subprocess.run(
            [router_bin, "--help"],
            capture_output=True,
            text=True,
            timeout=help_timeout,
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


def wait_router_http_health(
    base_url: str,
    timeout_seconds: float,
    *,
    proc: subprocess.Popen[Any] | None = None,
    log_path: Path | None = None,
    prefix: str = "",
) -> None:
    url = base_url.rstrip("/") + "/health"
    wait_for_http_ok(
        url,
        timeout_seconds=timeout_seconds,
        proc=proc,
        log_path=log_path,
        prefix=prefix,
    )
