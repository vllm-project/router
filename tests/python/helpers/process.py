"""Start and stop worker and router processes.

``vllm_rs_serve_cmd``, ``vllm_serve_cmd``, and ``router_cmd`` build argv.
Keyword arguments become flags, so each suite passes only the flags it needs.
``launch_worker`` and ``launch_router`` start those processes.
``launch_vllm_http_worker`` starts a real vLLM HTTP worker for GPU tests.
``reserve_port`` and the ``release_*_port_holders`` helpers keep an
OS-assigned port until the process that binds it is started.
``graceful_stop`` is the only teardown. Importing this module does not import
vLLM; the worker command builder loads only when a worker is launched.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _progress(prefix: str, msg: str) -> None:
    print(f"[{prefix}] {msg}", file=sys.stderr, flush=True)


def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _popen(
    cmd: list[str], env: dict[str, str] | None, *, prefix: str
) -> subprocess.Popen:
    _progress(prefix, "launch: " + " ".join(cmd))
    logger.info("launch: %s", " ".join(cmd))
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.Popen(
        cmd,
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def _pipe(
    proc: subprocess.Popen,
    label: str,
    sink: list[str] | None,
    lock: threading.Lock,
    log_fp=None,
) -> threading.Thread:
    def _reader() -> None:
        stream = proc.stdout
        if stream is None:
            return
        for line in stream:
            print(f"[{label}] {line}", end="", file=sys.stderr, flush=True)
            if log_fp is not None:
                log_fp.write(line)
                log_fp.flush()
            if sink is not None:
                with lock:
                    sink.append(line)

    thread = threading.Thread(target=_reader, name=f"perf-{label}", daemon=True)
    thread.start()
    return thread


def _wait_http(
    url: str,
    timeout: float,
    proc: subprocess.Popen,
    label: str,
    *,
    prefix: str,
) -> None:
    start = time.monotonic()
    last_log = start
    while time.monotonic() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(
                f"{label} exited early (code={proc.returncode}); see its log above"
            )
        if _http_ok(url):
            _progress(
                prefix,
                f"{label} ready at {url} ({time.monotonic() - start:.0f}s)",
            )
            return
        now = time.monotonic()
        if now - last_log >= 30:
            _progress(
                prefix, f"waiting for {url} ({now - start:.0f}s / {timeout:.0f}s)…"
            )
            last_log = now
        time.sleep(2)
    raise TimeoutError(f"{label} not ready at {url} within {timeout}s")


def graceful_stop(
    proc: subprocess.Popen | None,
    timeout: float = 120.0,
    *,
    process_group: bool = False,
    kill_grace: float = 15.0,
) -> None:
    """Terminate, then kill. ``process_group`` signals the whole session.

    Replay is started with ``start_new_session=True``. If its leader has
    already exited, the children can still hold ports, so that path still
    sends ``SIGKILL`` to the group. ``None`` is a no-op.
    """
    if proc is None:
        return

    if process_group:
        if proc.poll() is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return
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

    if proc.poll() is not None:
        return
    proc.terminate()
    start = time.monotonic()
    while proc.poll() is None and time.monotonic() - start < timeout:
        time.sleep(1)
    if proc.poll() is None:
        proc.kill()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _stop_pid(pid: int, timeout: float) -> None:
    if not _pid_alive(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    start = time.monotonic()
    while _pid_alive(pid) and time.monotonic() - start < timeout:
        time.sleep(1)
    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _stop_proc(
    proc: subprocess.Popen | None,
    thread: threading.Thread | None,
    timeout: float,
    label: str,
    *,
    prefix: str,
) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        _progress(prefix, f"stopping {label}")
    graceful_stop(proc, timeout)
    if thread is not None:
        thread.join(timeout=5)


@dataclass
class RunningProc:
    """One worker or one router."""

    proc: subprocess.Popen
    thread: threading.Thread
    label: str
    log_prefix: str = "perf"
    lines: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    log_fp: object | None = None

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self, timeout: float = 120.0) -> None:
        _stop_proc(
            self.proc,
            self.thread,
            timeout,
            self.label,
            prefix=self.log_prefix,
        )
        if self.log_fp is not None:
            self.log_fp.close()
            self.log_fp = None


def reserve_port(holders: list[socket.socket | None], host: str = "127.0.0.1") -> int:
    """Bind ``host:0``, keep the socket in ``holders``, and return the port.

    Holder order for a multi-worker cell is HTTP ports, then gRPC ports, then
    the router port, then the Prometheus port. ``release_worker_port_holders``
    and ``release_router_port_holders`` use that order.
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    port = int(sock.getsockname()[1])
    holders.append(sock)
    return port


def _close_port_holder(holders: list[socket.socket | None], index: int) -> None:
    if index >= len(holders) or holders[index] is None:
        return
    try:
        holders[index].close()
    except OSError:
        pass
    holders[index] = None


def release_all_port_holders(holders: list[socket.socket | None]) -> None:
    """Close every reserved socket and clear the list."""

    for index in range(len(holders)):
        _close_port_holder(holders, index)
    holders.clear()


def release_worker_port_holders(
    holders: list[socket.socket | None], worker: int, worker_count: int
) -> None:
    """Close the HTTP and gRPC holders for one worker."""

    for index in (worker, worker_count + worker):
        _close_port_holder(holders, index)


def release_router_port_holders(
    holders: list[socket.socket | None], worker_count: int
) -> None:
    """Close the router and Prometheus holders."""

    for index in (2 * worker_count, 2 * worker_count + 1):
        _close_port_holder(holders, index)


def _cli_flags(options: Mapping[str, Any]) -> list[str]:
    """Turn keyword arguments into CLI flags, in the order they were passed.

    ``None`` and ``False`` are left out. ``True`` is a bare flag. A list or
    tuple is one flag followed by each item. A dict is one JSON value.
    ``max_model_len=32768`` becomes ``--max-model-len 32768``.
    """

    args: list[str] = []
    for key, value in options.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            args.append(flag)
            continue
        args.append(flag)
        if isinstance(value, (list, tuple)):
            args.extend(str(item) for item in value)
        elif isinstance(value, dict):
            args.append(json.dumps(value))
        else:
            args.append(str(value))
    return args


def stop_any(obj: object, timeout: float = 120.0) -> None:
    """Stop a ``RunningProc``, a ``Popen``, a pid, or an object with ``.proc``."""
    try:
        if isinstance(obj, RunningProc):
            obj.stop(timeout)
            return
        if isinstance(obj, subprocess.Popen):
            graceful_stop(obj, timeout)
            return
        if isinstance(obj, int):
            _stop_pid(obj, timeout)
            return
        proc = getattr(obj, "proc", None)
        if isinstance(proc, subprocess.Popen):
            graceful_stop(proc, timeout)
    except Exception as exc:
        logger.warning("failed to stop %r: %s", obj, exc)


def vllm_rs_serve_cmd(
    binary: str,
    model: str,
    *,
    host: str,
    port: int,
    grpc_port: int,
    cli: Sequence[str] = (),
    served_model_name: str | None = None,
    extra_args: Sequence[str] = (),
    engine_flags: Mapping[str, Any] | None = None,
    **flags: Any,
) -> list[str]:
    """``vllm-rs serve`` argv.

    Keyword arguments become flags before ``--``. ``engine_flags`` are the
    engine arguments after ``--``. ``cli`` and ``extra_args`` stay raw argv
    for callers that already store a flag list.
    """

    cmd = [
        binary,
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--grpc-port",
        str(grpc_port),
        *_cli_flags(flags),
        *cli,
    ]
    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])
    cmd.extend(extra_args)
    if engine_flags:
        cmd.append("--")
        cmd.extend(_cli_flags(engine_flags))
    return cmd


def vllm_serve_cmd(
    binary: str,
    model: str,
    *,
    host: str,
    port: int,
    cli: Sequence[str] = (),
    **flags: Any,
) -> list[str]:
    """``vllm serve`` argv. Extra keyword arguments are appended as flags."""

    return [
        binary,
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        *_cli_flags(flags),
        *cli,
    ]


def router_cmd(
    binary: str,
    *,
    host: str,
    port: int,
    worker_urls: str | Sequence[str],
    worker_startup_timeout_secs: int,
    prometheus_host: str | None = None,
    prometheus_port: int | None = None,
    **flags: Any,
) -> list[str]:
    """``vllm-router`` argv.

    ``worker_urls`` is one URL or a list of URLs after a single
    ``--worker-urls``. Other keyword arguments are appended as flags.
    Prometheus flags are omitted when unset.
    """

    urls = [worker_urls] if isinstance(worker_urls, str) else list(worker_urls)
    cmd = [
        binary,
        "--host",
        host,
        "--port",
        str(port),
        "--worker-urls",
        *urls,
    ]
    if prometheus_host:
        cmd.extend(["--prometheus-host", prometheus_host])
    if prometheus_port is not None:
        cmd.extend(["--prometheus-port", str(prometheus_port)])
    cmd.extend(["--worker-startup-timeout-secs", str(int(worker_startup_timeout_secs))])
    cmd.extend(_cli_flags(flags))
    return cmd


def _open_log(log_path: Path | None):
    if log_path is None:
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return open(log_path, "w", encoding="utf-8")


def launch_worker(
    cmd: list[str],
    *,
    label: str = "worker",
    env: dict[str, str] | None = None,
    ready_url: str | None = None,
    sleep_s: float = 0,
    timeout: float = 1800,
    log_prefix: str = "perf",
    log_path: Path | None = None,
) -> RunningProc:
    """Start a worker. L1 passes ``ready_url`` (``/v1/models``). gRPC passes ``sleep_s``.

    ``log_path`` also copies stdout into that file. Callers that omit it only
    stream to stderr.
    """
    log_fp = _open_log(log_path)
    proc = _popen(cmd, env, prefix=log_prefix)
    thread = _pipe(proc, label, None, threading.Lock(), log_fp)
    try:
        if ready_url:
            _wait_http(ready_url, timeout, proc, label, prefix=log_prefix)
        elif sleep_s > 0:
            time.sleep(sleep_s)
        if proc.poll() is not None:
            raise RuntimeError(
                f"{label} exited early (code={proc.returncode}); see its log above"
            )
    except Exception:
        _stop_proc(proc, thread, 120.0, label, prefix=log_prefix)
        if log_fp is not None:
            log_fp.close()
        raise
    return RunningProc(
        proc=proc,
        thread=thread,
        label=label,
        log_prefix=log_prefix,
        log_fp=log_fp,
    )


def launch_router(
    cmd: list[str],
    *,
    router_url: str,
    env: dict[str, str] | None = None,
    timeout: float = 1800,
    keep_log: bool = False,
    log_prefix: str = "perf",
    log_path: Path | None = None,
) -> RunningProc:
    """Start a router and block until ``/health`` is OK.

    ``log_path`` also copies stdout into that file. Callers that omit it only
    stream to stderr.
    """
    lock = threading.Lock()
    lines: list[str] = []
    log_fp = _open_log(log_path)
    proc = _popen(cmd, env, prefix=log_prefix)
    thread = _pipe(proc, "vllm-router", lines if keep_log else None, lock, log_fp)
    try:
        _wait_http(
            router_url.rstrip("/") + "/health",
            timeout,
            proc,
            "router",
            prefix=log_prefix,
        )
        if proc.poll() is not None:
            raise RuntimeError(
                f"router exited early (code={proc.returncode}); "
                "see [vllm-router] log above"
            )
    except Exception:
        _stop_proc(proc, thread, 120.0, "router", prefix=log_prefix)
        if log_fp is not None:
            log_fp.close()
        raise
    return RunningProc(
        proc=proc,
        thread=thread,
        label="router",
        log_prefix=log_prefix,
        lines=lines,
        lock=lock,
        log_fp=log_fp,
    )


def _parse_http_url(base_url: str) -> tuple[str, str]:
    parsed = urlparse(base_url)
    return parsed.hostname or "127.0.0.1", (
        str(parsed.port) if parsed.port is not None else ""
    )


def launch_vllm_http_worker(
    model: str,
    base_url: str,
    *,
    dp_size: int | None = None,
    api_key: str | None = None,
    base_gpu_id: int | None = 0,
    label: str = "vllm",
) -> RunningProc:
    """Start one real vLLM HTTP worker. Callers wait for readiness themselves."""
    from tools.e2e.vllm_test_compat import (
        vllm_worker_command,
        vllm_worker_popen_env,
    )

    host, port = _parse_http_url(base_url)
    cmd = vllm_worker_command(
        model,
        host,
        port,
        dp_size=dp_size,
        api_key=api_key,
        base_gpu_id=base_gpu_id,
    )
    env = vllm_worker_popen_env(base_gpu_id=base_gpu_id, dp_size=dp_size)
    return launch_worker(cmd, env=env, label=label, log_prefix="e2e")
