"""Launch vLLM worker + vllm-router for perf cases (matches scripts/backend semantics)."""

from __future__ import annotations

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
from helpers.paths import repo_root  # noqa: E402
from helpers.perf_process import (  # noqa: E402
    attach_stderr_log_pump,
    graceful_stop_subprocess,
    popen_merged_output,
    preflight_router_bin,
    router_env_for_model,
    subprocess_exit_error,
    wait_router_http_health,
)

if TYPE_CHECKING:
    from case_loader import BackendCaseFile

REPO_ROOT = repo_root()


def _progress(msg: str) -> None:
    print(f"[perf stack] {msg}", file=sys.stderr, flush=True)


@dataclass
class PerfStack:
    router_url: str
    worker_proc: subprocess.Popen
    router_proc: subprocess.Popen

    def stop(self, timeout: float = 120.0) -> None:
        for proc in (self.router_proc, self.worker_proc):
            graceful_stop_subprocess(proc, timeout)


def _router_bin(case: BackendCaseFile | None = None) -> str:
    case_value = case.stack.router_bin if case and case.stack.router_bin else ""
    label = f"grpc case {case.path.name}" if case else "grpc perf"
    return resolve_router_bin(case_value=case_value, repo=REPO_ROOT, label=label)


def _vllm_rs_bin(case: BackendCaseFile | None = None) -> str:
    case_value = case.stack.vllm_rs_bin if case and case.stack.vllm_rs_bin else ""
    label = f"grpc case {case.path.name}" if case else "grpc perf"
    return resolve_vllm_rs(case_value=case_value, repo=REPO_ROOT, label=label)


def _vllm_cli() -> str:
    found = shutil.which("vllm")
    if not found:
        raise FileNotFoundError("vllm not on PATH")
    return found


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
    preflight_router_bin(router_bin)

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
        worker_proc = popen_merged_output(worker_cmd)
        attach_stderr_log_pump(worker_proc, log_prefix="vllm-rs")
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
        worker_proc = popen_merged_output(
            worker_cmd, env={"VLLM_USE_RUST_FRONTEND": "1"}
        )
        attach_stderr_log_pump(worker_proc, log_prefix="vllm-rust-http")
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
        worker_proc = popen_merged_output(
            worker_cmd, env={"VLLM_USE_RUST_FRONTEND": "0"}
        )
        attach_stderr_log_pump(worker_proc, log_prefix="vllm-python-http")
        time.sleep(sleep_s)
        worker_url = f"http://{host}:{wp}"
    else:
        raise ValueError(f"unknown backend {backend!r} in {case.path}")

    if worker_proc.poll() is not None:
        raise subprocess_exit_error(worker_proc, "worker")

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
        router_proc = popen_merged_output(router_cmd, env=router_env_for_model(model))
    except Exception:
        graceful_stop_subprocess(worker_proc, 120.0)
        raise
    attach_stderr_log_pump(router_proc, log_prefix="vllm-router")
    try:
        wait_router_http_health(
            router_url, timeout, proc=router_proc, prefix="[perf stack] "
        )
    except Exception:
        PerfStack(router_url, worker_proc, router_proc).stop()
        raise

    if router_proc.poll() is not None:
        PerfStack(router_url, worker_proc, router_proc).stop()
        raise subprocess_exit_error(router_proc, "router")

    return PerfStack(router_url, worker_proc, router_proc)
