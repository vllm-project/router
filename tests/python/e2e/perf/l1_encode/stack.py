"""One vllm-rs worker for the L1-on matrix. Worker and router both restart before L1 off."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from tests.python.e2e.perf.binaries import graceful_stop as _graceful_stop
from tests.python.e2e.perf.binaries import preflight_router_bin as _preflight_router_bin
from tests.python.e2e.perf.binaries import router_bin as _shared_router_bin
from tests.python.e2e.perf.binaries import vllm_rs_bin as _shared_vllm_rs_bin
from tests.python.e2e.perf.l1_encode.case_loader import L1Case, check_limits


def _progress(msg: str) -> None:
    print(f"[perf] {msg}", file=sys.stderr, flush=True)


def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _router_bin(case: L1Case) -> str:
    return _shared_router_bin(case.router_bin)


def _vllm_rs_bin(case: L1Case) -> str:
    return _shared_vllm_rs_bin(case.vllm_rs_bin)


def _popen(cmd: list[str], *, env: dict[str, str]) -> subprocess.Popen:
    _progress("launch: " + " ".join(cmd))
    return subprocess.Popen(
        cmd,
        env=env,
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
) -> threading.Thread:
    def _reader() -> None:
        stream = proc.stdout
        if stream is None:
            return
        for line in stream:
            print(f"[{label}] {line}", end="", file=sys.stderr, flush=True)
            if sink is not None:
                with lock:
                    sink.append(line)

    thread = threading.Thread(target=_reader, name=f"perf-{label}", daemon=True)
    thread.start()
    return thread


class L1Stack:
    """Worker stays up while L1 is on. Crossing to L1 off restarts worker and router."""

    def __init__(self, case: L1Case, model: str) -> None:
        self.case = case
        self.model = model
        self.managed = True
        self.router_url = f"http://{case.host}:{case.router_port}"
        self.prometheus_url = f"http://{case.host}:{case.prometheus_port}/metrics"
        self.worker_proc: subprocess.Popen | None = None
        self.router_proc: subprocess.Popen | None = None
        self.worker_thread: threading.Thread | None = None
        self.router_thread: threading.Thread | None = None
        self.arm_key: tuple | None = None
        self.router_lines: list[str] = []
        self._lock = threading.Lock()
        self._router_bin = ""

    def start_worker(self) -> None:
        check_limits(self.case)
        self._router_bin = _router_bin(self.case)
        _preflight_router_bin(self._router_bin)
        case = self.case
        rs = _vllm_rs_bin(case)
        cmd = [
            rs,
            "serve",
            self.model,
            "--host",
            case.host,
            "--port",
            str(case.worker_http_port),
            "--grpc-port",
            str(case.worker_grpc_port),
            *case.worker_cli,
            "--served-model-name",
            self.model,
        ]
        if os.environ.get("L1_LANGUAGE_MODEL_ONLY", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            cmd.append("--language-model-only")
        env = os.environ.copy()
        if "CUDA_VISIBLE_DEVICES" not in os.environ and case.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = case.cuda_visible_devices
        _progress(
            "worker stays up through L1 on (router restarts per hit rate). "
            "Before L1 off, worker and router both cold-start. "
            "Logs are prefixed [vllm-rs] and [vllm-router]."
        )
        self._worker_cmd = cmd
        self._worker_env = env
        self._launch_worker()

    def ensure_router(self, l1_on: bool, hit_rate: float) -> None:
        from tests.python.e2e.perf.l1_encode.report import hit_tag

        if self.worker_proc is None or self.worker_proc.poll() is not None:
            code = None if self.worker_proc is None else self.worker_proc.returncode
            raise RuntimeError(
                f"worker is not running (code={code}); see [vllm-rs] log above"
            )
        key = ("on", hit_tag(hit_rate)) if l1_on else ("off",)
        if (
            self.arm_key == key
            and self.router_proc is not None
            and self.router_proc.poll() is None
        ):
            _progress(
                f"reuse router L1 cache {'on' if l1_on else 'off'} "
                f"hit_rate={hit_rate:g}"
            )
            return
        previous_mode = self.arm_key[0] if self.arm_key else None
        mode = "on" if l1_on else "off"
        if previous_mode is not None and previous_mode != mode:
            _progress(
                f"cold start between L1 {previous_mode} and L1 {mode}: "
                "restart worker and router"
            )
            self._stop_router()
            self._stop_worker()
            pause = float(os.environ.get("PERF_STACK_STOP_SLEEP", "2"))
            if pause > 0:
                time.sleep(pause)
            self._launch_worker()
            self._start_router(l1_on)
            self.arm_key = key
            return
        if self.router_proc is not None:
            _progress(
                "router cold start "
                f"(next: L1 cache {'on' if l1_on else 'off'}, hit_rate={hit_rate:g}); "
                "worker stays up"
            )
            self._stop_router()
            pause = float(os.environ.get("PERF_STACK_STOP_SLEEP", "2"))
            if pause > 0:
                time.sleep(pause)
        self._start_router(l1_on)
        self.arm_key = key

    def router_log_text(self) -> str:
        with self._lock:
            return "".join(self.router_lines)

    def l1_metrics_text(self) -> str:
        if not self.prometheus_url:
            return ""
        try:
            with urllib.request.urlopen(self.prometheus_url, timeout=5) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            _progress(f"metrics unavailable: {exc}")
            return ""
        return "\n".join(
            line for line in text.splitlines() if line.startswith("vllm_tokenizer_l1_")
        )

    def stop(self) -> None:
        self._stop_router()
        self._stop_worker()

    def _launch_worker(self) -> None:
        case = self.case
        self.worker_proc = _popen(self._worker_cmd, env=self._worker_env)
        self.worker_thread = _pipe(self.worker_proc, "vllm-rs", None, self._lock)
        self._wait_http(
            f"http://{case.host}:{case.worker_http_port}/v1/models",
            self._timeout(),
            self.worker_proc,
            "worker",
        )

    def _stop_worker(self) -> None:
        proc = self.worker_proc
        thread = self.worker_thread
        self.worker_proc = None
        self.worker_thread = None
        if proc is not None:
            _progress("stopping worker")
            _graceful_stop(proc)
        if thread is not None:
            thread.join(timeout=5)

    def _timeout(self) -> float:
        raw = (
            os.environ.get("PERF_WORKER_STARTUP_TIMEOUT", "").strip()
            or os.environ.get("PERF_STACK_TIMEOUT", "").strip()
            or "1800"
        )
        return float(raw)

    def _wait_http(
        self, url: str, timeout: float, proc: subprocess.Popen, label: str
    ) -> None:
        start = time.monotonic()
        last_log = start
        while time.monotonic() - start < timeout:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"{label} exited early (code={proc.returncode}); see its log above"
                )
            if _http_ok(url):
                _progress(f"{label} ready at {url} ({time.monotonic() - start:.0f}s)")
                return
            now = time.monotonic()
            if now - last_log >= 30:
                _progress(f"waiting for {url} ({now - start:.0f}s / {timeout:.0f}s)…")
                last_log = now
            time.sleep(2)
        raise TimeoutError(f"{label} not ready at {url} within {timeout}s")

    def _router_env(self, l1_on: bool) -> dict[str, str]:
        env = os.environ.copy()
        env["VLLM_ROUTER_STAGES"] = "1"
        env["VLLM_ROUTER_L0_CACHE"] = "0"
        env["VLLM_ROUTER_L1_CACHE"] = "1" if l1_on else "0"
        return env

    def _start_router(self, l1_on: bool) -> None:
        case = self.case
        worker_url = f"grpc://{case.host}:{case.worker_grpc_port}"
        cmd = [
            self._router_bin,
            "--host",
            case.host,
            "--port",
            str(case.router_port),
            "--worker-urls",
            worker_url,
            "--prometheus-host",
            case.host,
            "--prometheus-port",
            str(case.prometheus_port),
            "--worker-startup-timeout-secs",
            str(int(self._timeout())),
        ]
        with self._lock:
            self.router_lines.clear()
        self.router_proc = _popen(cmd, env=self._router_env(l1_on))
        self.router_thread = _pipe(
            self.router_proc, "vllm-router", self.router_lines, self._lock
        )
        try:
            self._wait_http(
                self.router_url.rstrip("/") + "/health",
                self._timeout(),
                self.router_proc,
                "router",
            )
        except Exception:
            self._stop_router()
            raise
        if self.router_proc.poll() is not None:
            code = self.router_proc.returncode
            self._stop_router()
            raise RuntimeError(
                f"router exited early (code={code}); see [vllm-router] log above"
            )

    def _stop_router(self) -> None:
        proc = self.router_proc
        thread = self.router_thread
        self.router_proc = None
        self.router_thread = None
        self.arm_key = None
        if proc is not None:
            _graceful_stop(proc)
        if thread is not None:
            thread.join(timeout=5)


class ExternalStack:
    """Bench against a router the caller already started. Does not switch L1."""

    def __init__(self, router_url: str) -> None:
        self.managed = False
        self.router_url = router_url
        self.prometheus_url = ""
        self.arm_key = None

    def ensure_router(self, l1_on: bool, hit_rate: float) -> None:
        _progress(
            "PERF_EXTERNAL=1: not starting or restarting the router "
            f"(this test expects L1 cache {'on' if l1_on else 'off'}, "
            f"hit_rate={hit_rate:g})"
        )

    def router_log_text(self) -> str:
        return ""

    def l1_metrics_text(self) -> str:
        return ""

    def stop(self) -> None:
        return None
