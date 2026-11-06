"""Start and stop real vLLM worker processes for GPU tests.

Importing this module does not import vLLM. The worker command builder is
loaded only when a worker is launched. Router processes are started by
``RouterManager``.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import time
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _parse_url(base_url: str) -> tuple[str, str]:
    """Parse a base URL and return (host, port) as strings.

    This is more robust than simple string splitting and supports different schemes
    and URL shapes like trailing paths.
    """
    parsed = urlparse(base_url)
    return parsed.hostname or "127.0.0.1", (
        str(parsed.port) if parsed.port is not None else ""
    )


def _popen_launch_worker(
    model: str,
    base_url: str,
    *,
    dp_size: int | None = None,
    api_key: str | None = None,
    base_gpu_id: int | None = 0,
) -> subprocess.Popen:
    from tools.e2e.vllm_test_compat import (
        vllm_worker_command,
        vllm_worker_popen_env,
    )

    host, port = _parse_url(base_url)
    cmd = vllm_worker_command(
        model,
        host,
        port,
        dp_size=dp_size,
        api_key=api_key,
        base_gpu_id=base_gpu_id,
    )
    env = vllm_worker_popen_env(base_gpu_id=base_gpu_id, dp_size=dp_size)
    return subprocess.Popen(cmd, env=env)


def _terminate(proc: subprocess.Popen, timeout: float = 120) -> None:
    if proc is None:
        return
    proc.terminate()
    start = time.perf_counter()
    while proc.poll() is None:
        if time.perf_counter() - start > timeout:
            proc.kill()
            break
        time.sleep(1)


def _which(cmd: str) -> Optional[str]:
    try:
        return shutil.which(cmd)
    except Exception as e:
        logger.warning("shutil.which(%r) failed: %s", cmd, e)
        return None


def _graceful_stop_popen(proc: subprocess.Popen) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            for _ in range(5):
                if proc.poll() is not None:
                    break
                time.sleep(1)
            if proc.poll() is None:
                proc.kill()
    except Exception as e:
        logger.warning("Exception during graceful stop of popen: %s", e)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _graceful_stop_pid(pid: int) -> None:
    try:
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
            for _ in range(5):
                if not _pid_alive(pid):
                    break
                time.sleep(1)
            if _pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
    except Exception:
        pass


def _graceful_stop_any(obj) -> None:
    try:
        if isinstance(obj, subprocess.Popen):
            _graceful_stop_popen(obj)
            return
        if isinstance(obj, int):
            _graceful_stop_pid(obj)
            return
        proc_obj = getattr(obj, "proc", None)
        if isinstance(proc_obj, subprocess.Popen):
            _graceful_stop_popen(proc_obj)
    except Exception:
        pass
