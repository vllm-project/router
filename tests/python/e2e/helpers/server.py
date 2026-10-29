# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Manage vLLM serve subprocesses for E2E performance scenarios."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .case_loader import (
    E2EConfig,
    PerfScenario,
    build_vllm_serve_argv,
    build_vllm_serve_env,
)
from .paths import repo_root

_READY_POLL_INTERVAL_SECONDS = 5


class TerminalStream:
    """Toggle whether streamed subprocess output is copied to the pytest terminal."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._lock = threading.Lock()

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = enabled

    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled


def sanitize_terminal_line(line: str) -> str:
    """Keep the final segment after carriage-return refreshes."""

    if "\r" not in line:
        return line
    return line.split("\r")[-1]


def start_streaming_output(
    proc: subprocess.Popen[str],
    *,
    log_path: Path | None = None,
    prefix: str = "",
    terminal: TerminalStream | None = None,
) -> threading.Thread:
    """Pump one merged stdout stream to the terminal and an optional log file."""

    if proc.stdout is None:
        raise ValueError("process stdout must be piped for streaming output")

    def pump() -> None:
        log_fp = log_path.open("w", encoding="utf-8") if log_path is not None else None
        try:
            for line in proc.stdout:
                if log_fp is not None:
                    log_fp.write(line)
                    log_fp.flush()
                if terminal is None or terminal.is_enabled():
                    sys.stderr.write(f"{prefix}{sanitize_terminal_line(line)}")
                    sys.stderr.flush()
        finally:
            if log_fp is not None:
                log_fp.close()

    thread = threading.Thread(
        target=pump, name=f"stream-{prefix.strip() or 'subprocess'}", daemon=True
    )
    thread.start()
    return thread


def join_streaming_output(
    proc: subprocess.Popen[str],
    thread: threading.Thread | None,
    *,
    join_timeout_seconds: float = 10,
) -> None:
    """Close the process pipe and wait for the streaming thread to finish."""

    if proc.stdout is not None:
        proc.stdout.close()
    if thread is not None:
        thread.join(timeout=join_timeout_seconds)


def is_port_in_use(host: str, port: int) -> bool:
    """Return whether something is accepting TCP connections on ``host:port``."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def ensure_port_free(host: str, port: int) -> None:
    """Fail fast when the benchmark bind port is already occupied."""

    if is_port_in_use(host, port):
        raise RuntimeError(
            f"port {host}:{port} is already in use; stop the existing listener "
            f"or change benchmark_params.port in the case JSON"
        )


def wait_for_port_free(
    host: str,
    port: int,
    *,
    timeout_seconds: float = 30,
    poll_interval_seconds: float = 0.5,
) -> None:
    """Wait until ``host:port`` stops accepting connections."""

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not is_port_in_use(host, port):
            return
        time.sleep(poll_interval_seconds)
    raise RuntimeError(f"port {host}:{port} is still in use after {timeout_seconds}s")


def _remove_lifecycle_sockets(base_path: str) -> None:
    """Remove base and rank-scoped lifecycle socket files when present."""

    base = Path(base_path)
    candidates = [base, *base.parent.glob(f"{base.name}.dp*")]
    for path in candidates:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass


def _pre_start_cleanup(config: E2EConfig) -> None:
    """Ensure bind port and agentinfer sockets are free before launching vLLM."""

    ensure_port_free(config.host, config.port)
    if config.scenario is PerfScenario.AGENTINFER:
        _remove_lifecycle_sockets(config.lifecycle_socket)


def _wait_for_vllm_ready(
    config: E2EConfig,
    proc: subprocess.Popen[str],
    *,
    log_path: Path,
    prefix: str,
) -> None:
    """Poll ``/v1/models`` until vLLM accepts requests or the process exits."""

    models_url = f"{config.base_url.rstrip('/')}/v1/models"
    deadline = time.monotonic() + float(config.task_timeout_seconds)
    while time.monotonic() < deadline:
        exit_code = proc.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"vLLM exited with code {exit_code} before becoming ready; see log {log_path}"
            )
        try:
            response = httpx.get(models_url, timeout=10, trust_env=False)
            if response.status_code == 200:
                print(f"{prefix}ready: {models_url}", flush=True)
                return
        except httpx.HTTPError:
            pass
        time.sleep(_READY_POLL_INTERVAL_SECONDS)
    raise RuntimeError(
        f"vLLM did not become ready within {config.task_timeout_seconds}s; see log {log_path}"
    )


@dataclass
class VllmServer:
    """Launch and tear down one vLLM serve process for a benchmark case."""

    config: E2EConfig
    proc: subprocess.Popen[str] | None = None
    log_path: Path | None = None
    _log_thread: threading.Thread | None = field(default=None, repr=False)
    _terminal: TerminalStream = field(default_factory=TerminalStream, repr=False)

    def start(self) -> None:
        """Start vLLM serve and stream logs to pytest output plus a log file."""

        _pre_start_cleanup(self.config)
        self.config.vllm_log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = (
            self.config.vllm_log_dir
            / f"{self.config.scenario.value}-{self.config.run_tag}.log"
        )
        cmd = build_vllm_serve_argv(self.config)
        env = build_vllm_serve_env(self.config)
        prefix = f"[vllm:{self.config.scenario.value}] "
        print(f"{prefix}starting: {' '.join(cmd)}", flush=True)
        print(f"{prefix}log file: {self.log_path}", flush=True)
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=repo_root(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
            bufsize=1,
        )
        self._log_thread = start_streaming_output(
            self.proc,
            log_path=self.log_path,
            prefix=prefix,
            terminal=self._terminal,
        )
        if self.config.wait_for_vllm_ready:
            _wait_for_vllm_ready(
                self.config, self.proc, log_path=self.log_path, prefix=prefix
            )

    def stop(self) -> None:
        """Terminate the vLLM serve process group and remove lifecycle sockets."""

        if self.proc is None:
            return
        host, port = self.config.host, self.config.port
        proc = self.proc
        log_thread = self._log_thread
        try:
            _shutdown_process(proc)
        finally:
            join_streaming_output(proc, log_thread)
            self.proc = None
            self._log_thread = None
            try:
                wait_for_port_free(host, port, timeout_seconds=30)
            except RuntimeError as exc:
                print(f"[vllm:{self.config.scenario.value}] warning: {exc}", flush=True)
            if self.config.scenario is PerfScenario.AGENTINFER:
                _remove_lifecycle_sockets(self.config.lifecycle_socket)

    def __enter__(self) -> VllmServer:
        try:
            self.start()
        except Exception:
            self.stop()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def _shutdown_process(proc: subprocess.Popen[str]) -> None:
    """Send SIGTERM to the process group, escalating to SIGKILL when needed."""

    if os.name == "nt":
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
    except ProcessLookupError:
        return


def managed_vllm(config: E2EConfig) -> AbstractContextManager[VllmServer]:
    """Return a context manager that starts and stops vLLM for one benchmark case."""

    return VllmServer(config=config)
