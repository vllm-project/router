"""Spawn and tear down vllm / vllm-rs gRPC workers for kv_aware functional e2e."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.python.e2e.function.helpers import launch
from tests.python.e2e.function.helpers.worker_log_pump import SubprocessLogPump
from tests.python.e2e.function.kv_aware.config import (
    kv_aware_kv_events_enabled,
    kv_aware_kv_events_endpoint_bind,
    kv_aware_kv_events_replay_enabled,
    kv_aware_kv_events_replay_host,
    kv_aware_use_vllm_serve,
    kv_aware_worker_gpu_ids,
    kv_aware_worker_logs_quiet,
    worker_serve_extra_argv,
)


def _require_worker_launch() -> None:
    if kv_aware_use_vllm_serve():
        raise RuntimeError(
            "KV_AWARE_USE_VLLM_SERVE=1 selects `vllm serve`, which does not "
            "accept --grpc-port (docs/backend/grpc.md). Set "
            "KV_AWARE_USE_VLLM_SERVE=0 and use vllm-rs serve."
        )
    try:
        launch.resolve_worker_serve_launch(prefer_vllm_serve=False)
    except RuntimeError as exc:
        pytest.skip(str(exc))


def spawn_two_vllm_rs_grpc_workers(
    model: str,
    *,
    worker_health_timeout: float = 600.0,
    log_dir: Path | None = None,
) -> list[SimpleNamespace]:
    """Two worker serve processes; GPUs from ``kv_aware_worker_gpu_ids``."""
    _require_worker_launch()
    gpu_a, gpu_b = kv_aware_worker_gpu_ids()
    workers: list[SimpleNamespace] = []
    host = "127.0.0.1"

    if log_dir is None:
        log_dir = Path(tempfile.mkdtemp(prefix="kv_aware_worker_logs_"))
    log_dir.mkdir(parents=True, exist_ok=True)
    quiet_logs = kv_aware_worker_logs_quiet()
    echo = not quiet_logs
    print(
        f"[e2e] worker logs: {log_dir}"
        + (
            " (file only; unset KV_AWARE_WORKER_LOG_QUIET for terminal echo)"
            if quiet_logs
            else " (PIPE tee to stderr when pytest -s)"
        ),
        flush=True,
    )

    kv_on = kv_aware_kv_events_enabled()
    replay_on = kv_on and kv_aware_kv_events_replay_enabled()
    port_count = 8 if replay_on else (6 if kv_on else 4)
    ports = launch.find_distinct_available_ports(port_count)
    http_a, grpc_a, http_b, grpc_b = ports[:4]
    # Replay uses four extra ports (zmq+replay per worker). Without replay,
    # each worker still needs its own PUB port — do not share ports[4].
    if replay_on:
        zmq_a, replay_a, zmq_b, replay_b = ports[4], ports[5], ports[6], ports[7]
    elif kv_on:
        zmq_a, zmq_b = ports[4], ports[5]
        replay_a = replay_b = None
    else:
        zmq_a = zmq_b = replay_a = replay_b = None
    serve_extra = worker_serve_extra_argv()
    endpoint_bind = kv_aware_kv_events_endpoint_bind()
    replay_host = kv_aware_kv_events_replay_host()
    prefer_vllm_serve = kv_aware_use_vllm_serve()

    def _kv_log(zmq: int | None, replay: int | None) -> str:
        parts = []
        if zmq is not None:
            parts.append(f" kv_events_zmq={zmq}")
        if replay is not None:
            parts.append(f" replay={replay}")
        return "".join(parts)

    try:
        print(
            f"[e2e] worker A gpu={gpu_a} http={http_a} grpc={grpc_a}"
            f"{_kv_log(zmq_a, replay_a)}",
            flush=True,
        )
        log_a_path = log_dir / f"worker_a_grpc_{grpc_a}.log"
        proc_a = launch.popen_launch_vllm_rs_grpc_worker(
            model,
            host=host,
            http_port=http_a,
            grpc_port=grpc_a,
            base_gpu_id=gpu_a,
            kv_events_zmq_port=zmq_a,
            kv_events_replay_port=replay_a,
            extra_argv=serve_extra,
            endpoint_bind=endpoint_bind,
            replay_host=replay_host,
            prefer_vllm_serve=prefer_vllm_serve,
            capture_output=True,
        )
        worker_a = SimpleNamespace(
            proc=proc_a,
            url=f"grpc://{host}:{grpc_a}",
            http_url=f"http://{host}:{http_a}",
            gpu_id=gpu_a,
            log_path=str(log_a_path),
            log_pump=None,
        )
        workers.append(worker_a)
        worker_a.log_pump = SubprocessLogPump(
            proc_a,
            log_a_path,
            echo=echo,
            prefix=f"[worker A grpc={grpc_a}] ",
        )
        launch.wait_worker_health(
            worker_a.http_url, worker_health_timeout, proc=worker_a.proc
        )

        print(
            f"[e2e] worker B gpu={gpu_b} http={http_b} grpc={grpc_b}"
            f"{_kv_log(zmq_b, replay_b)}",
            flush=True,
        )
        log_b_path = log_dir / f"worker_b_grpc_{grpc_b}.log"
        proc_b = launch.popen_launch_vllm_rs_grpc_worker(
            model,
            host=host,
            http_port=http_b,
            grpc_port=grpc_b,
            base_gpu_id=gpu_b,
            kv_events_zmq_port=zmq_b,
            kv_events_replay_port=replay_b,
            extra_argv=serve_extra,
            endpoint_bind=endpoint_bind,
            replay_host=replay_host,
            prefer_vllm_serve=prefer_vllm_serve,
            capture_output=True,
        )
        worker_b = SimpleNamespace(
            proc=proc_b,
            url=f"grpc://{host}:{grpc_b}",
            http_url=f"http://{host}:{http_b}",
            gpu_id=gpu_b,
            log_path=str(log_b_path),
            log_pump=None,
        )
        workers.append(worker_b)
        worker_b.log_pump = SubprocessLogPump(
            proc_b,
            log_b_path,
            echo=echo,
            prefix=f"[worker B grpc={grpc_b}] ",
        )
        launch.wait_worker_health(
            worker_b.http_url, worker_health_timeout, proc=worker_b.proc
        )
        return workers
    except Exception:
        terminate_workers(workers)
        raise


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def mock_grpc_worker_exe() -> Path:
    """Built ``mock-grpc-worker`` binary. ``MOCK_GRPC_WORKER`` overrides the path."""
    override = os.environ.get("MOCK_GRPC_WORKER", "").strip()
    if override:
        path = Path(override)
        if not path.is_file():
            raise RuntimeError(f"MOCK_GRPC_WORKER is not a file: {path}")
        return path

    manifest = (
        _repo_root()
        / "tests"
        / "python"
        / "fixtures"
        / "mock_grpc_worker"
        / "Cargo.toml"
    )
    if not manifest.is_file():
        raise RuntimeError(f"missing mock grpc worker manifest: {manifest}")
    cargo = shutil.which("cargo")
    if cargo is None:
        raise RuntimeError(
            "cargo is not on PATH; build mock-grpc-worker and set MOCK_GRPC_WORKER"
        )
    print(f"[e2e] cargo build {manifest}", flush=True)
    subprocess.run(
        [cargo, "build", "--manifest-path", str(manifest)],
        check=True,
    )
    name = "mock-grpc-worker.exe" if os.name == "nt" else "mock-grpc-worker"
    exe = manifest.parent / "target" / "debug" / name
    if not exe.is_file():
        raise RuntimeError(f"mock-grpc-worker binary missing after build: {exe}")
    return exe


def spawn_two_mock_grpc_workers(
    model: str,
    *,
    worker_health_timeout: float = 30.0,
    log_dir: Path | None = None,
    text: str = "mock",
) -> list[SimpleNamespace]:
    """Two ``mock-grpc-worker`` processes. No GPU and no ``vllm-rs``."""
    exe = mock_grpc_worker_exe()
    workers: list[SimpleNamespace] = []
    host = "127.0.0.1"

    if log_dir is None:
        log_dir = Path(tempfile.mkdtemp(prefix="kv_aware_worker_logs_"))
    log_dir.mkdir(parents=True, exist_ok=True)
    quiet_logs = kv_aware_worker_logs_quiet()
    echo = not quiet_logs
    print(f"[e2e] worker logs: {log_dir}", flush=True)

    http_a, grpc_a, http_b, grpc_b = launch.find_distinct_available_ports(4)

    def _start(label: str, http_port: int, grpc_port: int) -> None:
        log_path = log_dir / f"worker_{label}_grpc_{grpc_port}.log"
        cmd = [
            str(exe),
            "--grpc-port",
            str(grpc_port),
            "--http-port",
            str(http_port),
            "--model",
            model,
            "--text",
            text,
        ]
        print(f"[e2e] mock worker {label} cmd: {cmd!r}", flush=True)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        worker = SimpleNamespace(
            proc=proc,
            url=f"grpc://{host}:{grpc_port}",
            http_url=f"http://{host}:{http_port}",
            gpu_id=None,
            log_path=str(log_path),
            log_pump=None,
        )
        workers.append(worker)
        worker.log_pump = SubprocessLogPump(
            proc,
            log_path,
            echo=echo,
            prefix=f"[worker {label} grpc={grpc_port}] ",
        )
        launch.wait_worker_health(worker.http_url, worker_health_timeout, proc=proc)

    try:
        _start("a", http_a, grpc_a)
        _start("b", http_b, grpc_b)
        return workers
    except Exception:
        terminate_workers(workers)
        raise


def terminate_workers(workers) -> None:
    for w in workers:
        launch.terminate_process(w.proc)
        pump = getattr(w, "log_pump", None)
        if pump is not None:
            pump.close()
        log_f = getattr(w, "log_file", None)
        if log_f is not None:
            try:
                log_f.close()
            except OSError:
                pass
