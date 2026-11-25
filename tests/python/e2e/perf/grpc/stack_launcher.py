"""Launch vLLM worker + vllm-router for perf cases (matches scripts/backend semantics)."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tests.python.e2e.perf.binaries import (
    preflight_router_bin as _preflight_router_bin,
)
from tests.python.e2e.perf.binaries import (
    router_bin as _shared_router_bin,
)
from tests.python.e2e.perf.binaries import (
    vllm_rs_bin as _shared_vllm_rs_bin,
)
from tests.python.helpers.process import (
    RunningProc,
    launch_router,
    launch_worker,
    router_cmd,
    vllm_rs_serve_cmd,
    vllm_serve_cmd,
)

if TYPE_CHECKING:
    from case_loader import BackendCaseFile


def _progress(msg: str) -> None:
    print(f"[perf stack] {msg}", file=sys.stderr, flush=True)


@dataclass
class PerfStack:
    """Worker and router started for one gRPC scenario."""

    router_url: str
    worker: RunningProc
    router: RunningProc

    def stop(self, timeout: float = 120.0) -> None:
        self.router.stop(timeout)
        self.worker.stop(timeout)


def _router_bin(case: BackendCaseFile | None = None) -> str:
    case_bin = case.stack.router_bin if case is not None else None
    return _shared_router_bin(case_bin)


def _vllm_rs_bin(case: BackendCaseFile | None = None) -> str:
    case_bin = case.stack.vllm_rs_bin if case is not None else None
    return _shared_vllm_rs_bin(case_bin)


def _vllm_cli() -> str:
    found = shutil.which("vllm")
    if not found:
        raise FileNotFoundError("vllm not on PATH")
    return found


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
            raise ValueError(f"{case.path}: rust_grpc requires worker.grpc_port")
        worker_cmd = vllm_rs_serve_cmd(
            _vllm_rs_bin(case),
            model,
            host=host,
            port=wp,
            grpc_port=gp,
            cli=cli,
        )
        worker_label = "vllm-rs"
        worker_env = None
        worker_url = f"grpc://{host}:{gp}"
    elif backend == "rust_http":
        worker_cmd = vllm_serve_cmd(_vllm_cli(), model, host=host, port=wp, cli=cli)
        worker_label = "vllm-rust-http"
        worker_env = {"VLLM_USE_RUST_FRONTEND": "1"}
        worker_url = f"http://{host}:{wp}"
    elif backend == "python_http":
        worker_cmd = vllm_serve_cmd(_vllm_cli(), model, host=host, port=wp, cli=cli)
        worker_label = "vllm-python-http"
        worker_env = {"VLLM_USE_RUST_FRONTEND": "0"}
        worker_url = f"http://{host}:{wp}"
    else:
        raise ValueError(f"unknown backend {backend!r} in {case.path}")

    router_argv = router_cmd(
        router_bin,
        host=host,
        port=rp,
        worker_urls=worker_url,
        worker_startup_timeout_secs=worker_startup_timeout,
        prometheus_port=case.stack.prometheus_port,
    )

    router_url = f"http://{host}:{rp}"
    _progress("starting worker (model load may take 10–30+ min)")
    worker = launch_worker(
        worker_cmd,
        label=worker_label,
        env=worker_env,
        sleep_s=sleep_s,
        timeout=timeout,
        log_prefix="perf stack",
    )
    try:
        _progress("starting router")
        router = launch_router(
            router_argv,
            router_url=router_url,
            env=_router_env(model),
            timeout=timeout,
            log_prefix="perf stack",
        )
    except Exception:
        worker.stop()
        raise
    return PerfStack(router_url=router_url, worker=worker, router=router)
