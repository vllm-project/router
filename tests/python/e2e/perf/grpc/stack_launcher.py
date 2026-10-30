"""Launch vLLM worker + vllm-router for perf cases (matches scripts/backend semantics)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

_E2E_DIR = Path(__file__).resolve().parents[3]
if str(_E2E_DIR) not in sys.path:
    sys.path.insert(0, str(_E2E_DIR))

from helpers.binaries import resolve_router_bin, resolve_vllm_rs  # noqa: E402
from helpers.http_ready import wait_for_http_ok  # noqa: E402
from helpers.paths import repo_root  # noqa: E402
from helpers.subprocess_io import start_stderr_pump_thread  # noqa: E402

if TYPE_CHECKING:
    from case_loader import BackendCaseFile

logger = logging.getLogger(__name__)

REPO_ROOT = repo_root()


def _progress(msg: str) -> None:
    print(f"[perf stack] {msg}", file=sys.stderr, flush=True)


def _pipe_process_output(proc: subprocess.Popen, label: str) -> None:
    """Avoid PIPE deadlock and surface worker/router logs under pytest -s."""

    start_stderr_pump_thread(proc, stderr_prefix=f"[{label}] ")


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
    url = base_url.rstrip("/") + "/health"
    wait_for_http_ok(
        url,
        timeout_seconds=timeout,
        proc=proc,
        prefix="[perf stack] ",
    )


def _router_bin(case: BackendCaseFile | None = None) -> str:
    case_value = case.stack.router_bin if case and case.stack.router_bin else ""
    label = f"grpc case {case.path.name}" if case else "grpc perf"
    return resolve_router_bin(case_value=case_value, repo=REPO_ROOT, label=label)


def _preflight_router_bin(router_bin: str) -> None:
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


def _vllm_rs_bin(case: BackendCaseFile | None = None) -> str:
    case_value = case.stack.vllm_rs_bin if case and case.stack.vllm_rs_bin else ""
    label = f"grpc case {case.path.name}" if case else "grpc perf"
    return resolve_vllm_rs(case_value=case_value, repo=REPO_ROOT, label=label)


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
    timeout = float(os.environ.get("PERF_STACK_TIMEOUT", "1800"))
    worker_startup_timeout = int(
        os.environ.get("PERF_WORKER_STARTUP_TIMEOUT", str(int(timeout)))
    )
    sleep_s = float(os.environ.get("PERF_STACK_WORKER_SLEEP", "2"))
    host = case.stack.host
    rp = case.stack.router_port
    wp = case.stack.worker_http_port
    cli = list(case.worker_cli)

    router_bin = _router_bin(case)
    _preflight_router_bin(router_bin)

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
