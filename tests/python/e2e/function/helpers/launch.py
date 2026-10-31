"""Launch vllm-router and vllm-rs workers for tests/python/e2e/function (no vllm.test import)."""

from __future__ import annotations

import importlib.util
import json
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
    return find_distinct_available_ports(1)[0]


def find_distinct_available_ports(count: int) -> list[int]:
    """Return ``count`` unique ephemeral ports on 127.0.0.1 (hold until allocated)."""
    if count < 1:
        raise ValueError("count must be >= 1")
    ports: list[int] = []
    sockets: list[socket.socket] = []
    try:
        while len(ports) < count:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            if port not in ports:
                ports.append(port)
                sockets.append(s)
            else:
                s.close()
        return ports
    finally:
        for s in sockets:
            s.close()


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


def resolve_vllm_rs_bin() -> str | None:
    """``VLLM_RS``, then ``vllm-rs`` on PATH, then ``site-packages/vllm/vllm-rs``."""
    override = os.environ.get("VLLM_RS", "").strip()
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        return None
    on_path = shutil.which("vllm-rs")
    if on_path:
        return on_path
    if importlib.util.find_spec("vllm") is None:
        return None
    import vllm

    bundled = Path(vllm.__file__).resolve().parent / "vllm-rs"
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return str(bundled)
    return None


def kv_events_config_argv(
    *,
    zmq_port: int,
    replay_port: int | None = None,
    endpoint_bind: str = "*",
    replay_host: str = "127.0.0.1",
) -> list[str]:
    """``--kv-events-config`` JSON. Replay is included only when ``replay_port`` is set."""
    payload: dict[str, object] = {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "endpoint": f"tcp://{endpoint_bind}:{zmq_port}",
    }
    if replay_port is not None:
        payload["replay_endpoint"] = f"tcp://{replay_host}:{replay_port}"
    return ["--kv-events-config", json.dumps(payload, separators=(",", ":"))]


def resolve_worker_serve_launch(
    *, prefer_vllm_serve: bool = False
) -> tuple[str, list[str]]:
    """Return ``(kind, argv_prefix)`` for a ``grpc://`` worker (``vllm-rs serve``).

    ``vllm serve`` does not accept ``--grpc-port`` (``docs/backend/grpc.md``).
    ``prefer_vllm_serve`` is rejected so that path cannot be launched by mistake.
    """
    if prefer_vllm_serve:
        raise RuntimeError(
            "KV_AWARE_USE_VLLM_SERVE=1 selects `vllm serve`, which does not "
            "accept --grpc-port (docs/backend/grpc.md). Set "
            "KV_AWARE_USE_VLLM_SERVE=0 and use vllm-rs serve."
        )
    vllm_rs = resolve_vllm_rs_bin()
    if vllm_rs:
        return ("vllm-rs", [vllm_rs, "serve"])
    raise RuntimeError(
        "No vllm-rs launcher: set VLLM_RS or put vllm-rs on PATH "
        "(KV_AWARE_USE_VLLM_SERVE=0); see scripts/backend/check_vllm_rs.sh"
    )


def popen_launch_vllm_rs_grpc_worker(
    model: str,
    *,
    host: str = "127.0.0.1",
    http_port: int,
    grpc_port: int,
    base_gpu_id: int = 0,
    kv_events_zmq_port: int | None = None,
    kv_events_replay_port: int | None = None,
    extra_argv: list[str] | None = None,
    endpoint_bind: str = "*",
    replay_host: str = "127.0.0.1",
    prefer_vllm_serve: bool = False,
    capture_output: bool = False,
) -> subprocess.Popen:
    """``vllm-rs serve`` with HTTP + ``--grpc-port``.

    KV-event flags are appended only when ``kv_events_zmq_port`` is set.
    ``--grpc-port`` is passed only on this ``vllm-rs`` path.

    When ``capture_output`` is true, stdout/stderr are merged into a PIPE for
    ``SubprocessLogPump`` (do not pass a custom stdout object).
    """
    kind, prefix = resolve_worker_serve_launch(prefer_vllm_serve=prefer_vllm_serve)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(base_gpu_id)
    serve_tail = list(extra_argv or [])
    if kv_events_zmq_port is not None and "--kv-events-config" not in serve_tail:
        serve_tail = [
            *serve_tail,
            *kv_events_config_argv(
                zmq_port=kv_events_zmq_port,
                replay_port=kv_events_replay_port,
                endpoint_bind=endpoint_bind,
                replay_host=replay_host,
            ),
        ]
    cmd = [
        *prefix,
        model,
        "--host",
        host,
        "--port",
        str(http_port),
        "--grpc-port",
        str(grpc_port),
        *serve_tail,
    ]
    zmq_note = f" kv_events_zmq={kv_events_zmq_port}" if kv_events_zmq_port else ""
    replay_note = (
        f" kv_events_replay={kv_events_replay_port}"
        if kv_events_replay_port is not None
        else ""
    )
    print(
        f"[e2e] worker launch ({kind}) gpu={base_gpu_id} http={http_port} "
        f"grpc={grpc_port}{zmq_note}{replay_note}",
        flush=True,
    )
    print(f"[e2e] worker cmd ({kind}): {cmd!r}", flush=True)
    popen_kw: dict = {"env": env}
    if capture_output:
        popen_kw["stdout"] = subprocess.PIPE
        popen_kw["stderr"] = subprocess.STDOUT
        popen_kw["text"] = True
        popen_kw["encoding"] = "utf-8"
        popen_kw["errors"] = "replace"
        popen_kw["bufsize"] = 1
    return subprocess.Popen(cmd, **popen_kw)


def popen_launch_router_only(
    base_url: str,
    policy: str = "round_robin",
    timeout: float = 120.0,
    *,
    intra_node_data_parallel_size: int = 1,
    enable_igw: bool = False,
    api_key: str | None = None,
    router_temperature: float | None = None,
    worker_urls: list[str] | None = None,
    router_model: str | None = None,
) -> subprocess.Popen:
    host, port = _parse_url(base_url)
    prom_port = find_available_port()
    env = os.environ.copy()
    if router_model:
        env["VLLM_ROUTER_MODEL"] = router_model
    elif worker_urls and any(u.startswith("grpc://") for u in worker_urls):
        raise RuntimeError(
            "grpc workers require router_model= (sets VLLM_ROUTER_MODEL for tokenization)"
        )
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
    cmd += [
        "--prometheus-port",
        str(prom_port),
        "--prometheus-host",
        "127.0.0.1",
    ]
    env.setdefault("RUST_LOG", "vllm_router_rs=info")
    env.setdefault("VLLM_ROUTER_STAGES", "1")
    # Inherit stdio so ``pytest -s`` shows router tracing (same as vllm-rs workers).
    print(f"[e2e] starting router {base_url} policy={policy!r}", flush=True)
    print(f"[e2e] router cmd: {cmd!r}", flush=True)
    proc = subprocess.Popen(cmd, env=env)
    try:
        wait_router_health(base_url, timeout)
    except TimeoutError as exc:
        if proc.poll() is None:
            proc.kill()
        raise TimeoutError(
            f"{exc}\nrouter cmd: {cmd!r}\n(router logs were streamed to this terminal)"
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
