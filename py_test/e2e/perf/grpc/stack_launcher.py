"""Launch vLLM worker + vllm-router for perf cases (matches scripts/backend semantics)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from case_loader import BackendCaseFile

logger = logging.getLogger(__name__)

# ``py_test/e2e/perf/grpc`` → repo root (run pytest from repository root).
REPO_ROOT = Path(__file__).resolve().parents[3]


def _progress(msg: str) -> None:
    print(f"[perf stack] {msg}", file=sys.stderr, flush=True)


def _pipe_process_output(proc: subprocess.Popen, label: str) -> None:
    """Avoid PIPE deadlock and surface worker/router logs under pytest -s."""

    def _reader() -> None:
        stream = proc.stdout
        if stream is None:
            return
        for line in stream:
            print(f"[{label}] {line}", end="", file=sys.stderr, flush=True)

    threading.Thread(target=_reader, daemon=True).start()


@dataclass
class PerfStack:
    router_url: str
    worker_proc: subprocess.Popen
    router_proc: subprocess.Popen

    def stop(self, timeout: float = 120.0) -> None:
        for proc in (self.router_proc, self.worker_proc):
            _graceful_stop(proc, timeout)


def _graceful_stop(proc: subprocess.Popen, timeout: float) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    start = time.monotonic()
    while proc.poll() is None and time.monotonic() - start < timeout:
        time.sleep(1)
    if proc.poll() is None:
        proc.kill()


def _proc_exit_error(proc: subprocess.Popen, label: str) -> RuntimeError:
    out = proc.stdout.read() if proc.stdout else ""
    return RuntimeError(f"{label} exited early (code={proc.returncode}): {out[-4000:]}")


def _wait_router_health(
    base_url: str, timeout: float, *, proc: subprocess.Popen | None = None
) -> None:
    start = time.monotonic()
    url = base_url.rstrip("/") + "/health"
    last_log = start
    with requests.Session() as session:
        while time.monotonic() - start < timeout:
            if proc is not None:
                if proc.poll() is not None:
                    raise _proc_exit_error(proc, "router")
            try:
                r = session.get(url, timeout=5)
                if r.status_code == 200:
                    _progress(
                        f"router healthy at {url} ({time.monotonic() - start:.0f}s)"
                    )
                    return
            except requests.RequestException:
                pass
            now = time.monotonic()
            if now - last_log >= 30:
                _progress(
                    f"waiting for {url} ({now - start:.0f}s / {timeout:.0f}s timeout)…"
                )
                last_log = now
            time.sleep(2)
    raise TimeoutError(f"router not healthy at {url} within {timeout}s")


def _resolve_bin_path(raw: str) -> str | None:
    """Resolve case JSON path (absolute, repo-relative, or PATH name)."""
    text = raw.strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(text)
    if found:
        return found
    return None


def _router_bin(case: BackendCaseFile | None = None) -> str:
    if raw := os.environ.get("ROUTER_BIN", "").strip():
        return raw
    if case and case.stack.router_bin:
        if resolved := _resolve_bin_path(case.stack.router_bin):
            return resolved
    hint = REPO_ROOT / "target" / "release" / "vllm-router"
    if hint.is_file():
        return str(hint)
    found = shutil.which("vllm-router")
    if found:
        return found
    tried = [str(hint)]
    if case and case.stack.router_bin:
        tried.insert(0, case.stack.router_bin)
    raise FileNotFoundError(
        f"vllm-router not found (repo root={REPO_ROOT}, tried {tried!r}); "
        "set ROUTER_BIN, stack.router_bin in case JSON, or build target/release/vllm-router"
    )


def _vllm_rs_bin(case: BackendCaseFile | None = None) -> str:
    if raw := os.environ.get("VLLM_RS", "").strip():
        return raw
    if case and case.stack.vllm_rs_bin:
        if resolved := _resolve_bin_path(case.stack.vllm_rs_bin):
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


def _vllm_cli() -> str:
    found = shutil.which("vllm")
    if not found:
        raise FileNotFoundError("vllm not on PATH")
    return found


def _popen(cmd: list[str], *, env: dict | None = None) -> subprocess.Popen:
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
    )


def _router_env(model: str) -> dict[str, str]:
    env: dict[str, str] = {"VLLM_ROUTER_MODEL": model}
    model_path = Path(model)
    tok = model_path / "tokenizer.json"
    if model_path.is_dir() and tok.is_file():
        env.setdefault("VLLM_ROUTER_TOKENIZER", str(tok))
    if stages := os.environ.get("VLLM_ROUTER_STAGES"):
        env["VLLM_ROUTER_STAGES"] = stages
    return env


def launch_stack(case: BackendCaseFile, model: str) -> PerfStack:
    """Start worker then router; block until router /health is OK."""
    timeout = float(os.environ.get("PERF_STACK_TIMEOUT", "3600"))
    worker_startup_timeout = int(
        os.environ.get("PERF_WORKER_STARTUP_TIMEOUT", str(int(timeout)))
    )
    sleep_s = float(os.environ.get("PERF_STACK_WORKER_SLEEP", "2"))
    host = case.stack.host
    rp = case.stack.router_port
    wp = case.stack.worker_http_port
    cli = list(case.worker_cli)

    router_bin = _router_bin(case)

    backend = case.backend
    if backend == "rust_grpc":
        gp = case.stack.worker_grpc_port
        if gp is None:
            raise ValueError(f"{case.path}: rust_grpc requires stack.worker_grpc_port")
        rs = _vllm_rs_bin(case)
        worker_cmd = [
            rs,
            "serve",
            model,
            "--host",
            host,
            "--port",
            str(wp),
            "--grpc-port",
            str(gp),
            *cli,
        ]
        _progress(
            f"starting vllm-rs worker (model load may take 10–30+ min): {' '.join(worker_cmd)}"
        )
        worker_proc = _popen(worker_cmd)
        _pipe_process_output(worker_proc, "vllm-rs")
        time.sleep(sleep_s)
        worker_url = f"grpc://{host}:{gp}"
    elif backend == "rust_http":
        worker_cmd = [
            _vllm_cli(),
            "serve",
            model,
            "--host",
            host,
            "--port",
            str(wp),
            *cli,
        ]
        _progress(f"starting rust-http worker: {' '.join(worker_cmd)}")
        worker_proc = _popen(worker_cmd, env={"VLLM_USE_RUST_FRONTEND": "1"})
        _pipe_process_output(worker_proc, "vllm-rust-http")
        time.sleep(sleep_s)
        worker_url = f"http://{host}:{wp}"
    elif backend == "python_http":
        worker_cmd = [
            _vllm_cli(),
            "serve",
            model,
            "--host",
            host,
            "--port",
            str(wp),
            *cli,
        ]
        _progress(f"starting python-http worker: {' '.join(worker_cmd)}")
        worker_proc = _popen(worker_cmd, env={"VLLM_USE_RUST_FRONTEND": "0"})
        _pipe_process_output(worker_proc, "vllm-python-http")
        time.sleep(sleep_s)
        worker_url = f"http://{host}:{wp}"
    else:
        raise ValueError(f"unknown backend {backend!r} in {case.path}")

    if worker_proc.poll() is not None:
        raise _proc_exit_error(worker_proc, "worker")

    router_cmd = [
        router_bin,
        "--host",
        host,
        "--port",
        str(rp),
        "--worker-urls",
        worker_url,
        "--worker-startup-timeout-secs",
        str(worker_startup_timeout),
    ]
    if case.stack.prometheus_port is not None:
        router_cmd.extend(["--prometheus-port", str(case.stack.prometheus_port)])

    router_url = f"http://{host}:{rp}"
    _progress(f"starting vllm-router: {' '.join(router_cmd)}")
    try:
        router_proc = _popen(router_cmd, env=_router_env(model))
    except Exception:
        _graceful_stop(worker_proc, 120.0)
        raise
    _pipe_process_output(router_proc, "vllm-router")
    try:
        _wait_router_health(router_url, timeout, proc=router_proc)
    except Exception:
        PerfStack(router_url, worker_proc, router_proc).stop()
        raise

    if router_proc.poll() is not None:
        PerfStack(router_url, worker_proc, router_proc).stop()
        raise _proc_exit_error(router_proc, "router")

    return PerfStack(router_url, worker_proc, router_proc)
