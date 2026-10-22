"""Launch router/vLLM worker processes for py_test/e2e/function (no vllm.test import)."""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


def _validate_router_executable(path: str, *, env_key: str) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise RuntimeError(f"{env_key}={path!r} is not a file (resolved {candidate})")
    if not os.access(candidate, os.X_OK):
        raise RuntimeError(f"{env_key}={path!r} is not executable")
    return str(candidate.resolve())


def _router_bin_from_env() -> str | None:
    """``ROUTER_BIN`` matches scripts/backend; ``VLLM_ROUTER_BIN`` is an alias."""
    for key in ("ROUTER_BIN", "VLLM_ROUTER_BIN"):
        value = os.environ.get(key, "").strip()
        if value:
            return _validate_router_executable(value, env_key=key)
    return None


def _find_cargo_router_binary() -> str | None:
    """Walk up from this file to a directory with ``Cargo.toml``, then ``target/*/vllm-router``."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if not (parent / "Cargo.toml").is_file():
            continue
        for profile in ("release", "debug"):
            candidate = parent / "target" / profile / "vllm-router"
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def _python_router_launch_prefix() -> list[str] | None:
    """Use only when ``vllm_router.launch_router`` imports (not bare ``find_spec``)."""
    try:
        import importlib

        importlib.import_module("vllm_router.launch_router")
    except ImportError:
        return None
    return [sys.executable, "-m", "vllm_router.launch_router"]


def router_launch_prefix() -> list[str]:
    """Command prefix to start the router (binary or ``python -m vllm_router.launch_router``)."""
    env_bin = _router_bin_from_env()
    if env_bin:
        return [env_bin]

    on_path = shutil.which("vllm-router")
    if on_path:
        return [on_path]

    cargo_bin = _find_cargo_router_binary()
    if cargo_bin:
        return [cargo_bin]

    py_prefix = _python_router_launch_prefix()
    if py_prefix is not None:
        return py_prefix

    raise RuntimeError(
        "Cannot start router: export ROUTER_BIN=/path/to/vllm-router (cargo build --release), "
        "add vllm-router to PATH, build from this repo, or pip install -e ."
    )


def find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _parse_url(base_url: str) -> tuple[str, str]:
    parsed = urlparse(base_url)
    return parsed.hostname or "127.0.0.1", (
        str(parsed.port) if parsed.port is not None else ""
    )


def wait_worker_health(
    base_url: str,
    timeout: float = 600.0,
    proc: subprocess.Popen | None = None,
) -> None:
    """Wait until a vLLM worker ``GET /health`` returns 200."""
    start = time.perf_counter()
    health_url = f"{base_url.rstrip('/')}/health"
    with requests.Session() as session:
        while time.perf_counter() - start < timeout:
            if proc is not None:
                code = proc.poll()
                if code is not None:
                    raise TimeoutError(
                        f"Worker process exited with code {code} before "
                        f"health OK: {health_url}"
                    )
            try:
                r = session.get(health_url, timeout=5)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
    raise TimeoutError(f"Worker failed to become healthy in time: {health_url}")


def wait_router_health(base_url: str, timeout: float) -> None:
    start = time.perf_counter()
    with requests.Session() as session:
        while time.perf_counter() - start < timeout:
            try:
                r = session.get(f"{base_url}/health", timeout=5)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
    raise TimeoutError("Router failed to become healthy in time")


def _vllm_worker_mode() -> str:
    """``serve`` (vLLM 0.3x CLI) or legacy ``launch_server`` module."""
    override = os.environ.get("VLLM_E2E_WORKER_MODE", "").strip().lower()
    if override in ("serve", "launch_server"):
        return override
    if importlib.util.find_spec("vllm.launch_server") is None:
        return "serve"
    return "launch_server"


def popen_launch_worker(
    model: str,
    base_url: str,
    *,
    dp_size: int | None = None,
    api_key: str | None = None,
    base_gpu_id: int | None = 0,
) -> subprocess.Popen:
    host, port = _parse_url(base_url)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(base_gpu_id if base_gpu_id is not None else 0)

    mode = _vllm_worker_mode()
    if mode == "serve":
        vllm_bin = shutil.which("vllm")
        if not vllm_bin:
            raise RuntimeError(
                "vLLM 0.3x expects `vllm serve` on PATH; install vllm or set "
                "VLLM_E2E_WORKER_MODE=launch_server for older layouts"
            )
        cmd = [vllm_bin, "serve", model, "--host", host, "--port", port]
        if dp_size is not None:
            cmd += ["--data-parallel-size", str(dp_size)]
    else:
        cmd = [
            sys.executable,
            "-m",
            "vllm.launch_server",
            "--model-path",
            model,
            "--host",
            host,
            "--port",
            port,
            "--base-gpu-id",
            str(base_gpu_id or 0),
        ]
        if dp_size is not None:
            cmd += ["--dp-size", str(dp_size)]

    if api_key is not None:
        cmd += ["--api-key", api_key]
    # Inherit stdout/stderr so vLLM load logs appear when running pytest -s.
    return subprocess.Popen(cmd, env=env)


def popen_launch_router_only(
    base_url: str,
    policy: str = "round_robin",
    timeout: float = 120.0,
    *,
    intra_node_data_parallel_size: int = 1,
    enable_igw: bool = False,
    api_key: str | None = None,
    router_temperature: float | None = None,
    kv_penalty_host: float | None = None,
    kv_penalty_external: float | None = None,
    kv_penalty_disk: float | None = None,
    worker_urls: list[str] | None = None,
) -> subprocess.Popen:
    host, port = _parse_url(base_url)
    prom_port = find_available_port()
    cmd = [
        *router_launch_prefix(),
        "--host",
        host,
        "--port",
        port,
        "--policy",
        policy,
    ]
    if worker_urls:
        cmd.extend(["--worker-urls", *worker_urls])
    if intra_node_data_parallel_size > 1:
        cmd += ["--intra-node-data-parallel-size", str(intra_node_data_parallel_size)]
    if enable_igw:
        cmd += ["--enable-igw"]
    if api_key is not None:
        cmd += ["--api-key", api_key]
    if router_temperature is not None:
        cmd += ["--router-temperature", str(router_temperature)]
    if kv_penalty_host is not None:
        cmd += ["--kv-penalty-host", str(kv_penalty_host)]
    if kv_penalty_external is not None:
        cmd += ["--kv-penalty-external", str(kv_penalty_external)]
    if kv_penalty_disk is not None:
        cmd += ["--kv-penalty-disk", str(kv_penalty_disk)]
    cmd += [
        "--prometheus-port",
        str(prom_port),
        "--prometheus-host",
        "127.0.0.1",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_router_health(base_url, timeout)
    except TimeoutError as exc:
        detail = proc.stderr.read() if proc.stderr is not None else ""
        if proc.poll() is None:
            proc.kill()
        raise TimeoutError(
            f"{exc}\nrouter cmd: {cmd!r}\nrouter stderr:\n{detail}"
        ) from exc
    return proc


def terminate_process(proc: subprocess.Popen | None, timeout: float = 120) -> None:
    if proc is None:
        return
    proc.terminate()
    start = time.perf_counter()
    while proc.poll() is None:
        if time.perf_counter() - start > timeout:
            proc.kill()
            break
        time.sleep(1)
