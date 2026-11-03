"""One vllm-rs worker and one router for each L1 mode. Both restart before L1 off."""

from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request

from tests.python.e2e.perf.binaries import preflight_router_bin as _preflight_router_bin
from tests.python.e2e.perf.binaries import router_bin as _shared_router_bin
from tests.python.e2e.perf.binaries import vllm_rs_bin as _shared_vllm_rs_bin
from tests.python.e2e.perf.l1_encode.case_loader import L1Case, check_limits
from tests.python.e2e.perf.stack_proc import RunningProc, launch_router, launch_worker


def _progress(msg: str) -> None:
    print(f"[perf] {msg}", file=sys.stderr, flush=True)


class L1Stack:
    """Worker and router stay up for one L1 mode. Crossing to L1 off restarts both."""

    def __init__(self, case: L1Case, model: str) -> None:
        self.case = case
        self.model = model
        self.router_url = f"http://{case.host}:{case.router_port}"
        self.prometheus_url = f"http://{case.host}:{case.prometheus_port}/metrics"
        self.arm_key: tuple | None = None
        self._worker: RunningProc | None = None
        self._router: RunningProc | None = None
        self._router_bin = ""

    def prepare(self) -> None:
        check_limits(self.case)
        self._router_bin = _shared_router_bin(self.case.router_bin)
        _preflight_router_bin(self._router_bin)
        _progress(
            "worker and router stay up through one L1 mode. "
            "Before L1 off, worker and router both cold-start. "
            "Logs are prefixed [vllm-rs] and [vllm-router]."
        )

    def ensure_router(self, l1_on: bool, hit_rate: float) -> None:
        key = ("on",) if l1_on else ("off",)
        if self.arm_key == key and self._alive():
            _progress(
                f"reuse worker and router L1 cache {'on' if l1_on else 'off'} "
                f"hit_rate={hit_rate:g}"
            )
            return
        mode = "on" if l1_on else "off"
        if self._worker is not None or self._router is not None:
            previous_mode = self.arm_key[0] if self.arm_key else None
            if previous_mode is not None and previous_mode != mode:
                _progress(
                    f"cold start between L1 {previous_mode} and L1 {mode}: "
                    "restart worker and router"
                )
            self._stop_pair()
            pause = float(os.environ.get("PERF_STACK_STOP_SLEEP", "2"))
            if pause > 0:
                time.sleep(pause)
        self._launch(l1_on)
        self.arm_key = key

    def router_log_text(self) -> str:
        router = self._router
        if router is None:
            return ""
        with router.lock:
            return "".join(router.lines)

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
        self._stop_pair()

    def _alive(self) -> bool:
        return (
            self._worker is not None
            and self._router is not None
            and self._worker.alive()
            and self._router.alive()
        )

    def _stop_pair(self) -> None:
        if self._router is not None:
            self._router.stop()
            self._router = None
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        self.arm_key = None

    def _timeout(self) -> float:
        raw = (
            os.environ.get("PERF_WORKER_STARTUP_TIMEOUT", "").strip()
            or os.environ.get("PERF_STACK_TIMEOUT", "").strip()
            or "1800"
        )
        return float(raw)

    def _worker_cmd(self) -> tuple[list[str], dict[str, str] | None]:
        case = self.case
        cmd = [
            _shared_vllm_rs_bin(case.vllm_rs_bin),
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
        env = None
        if "CUDA_VISIBLE_DEVICES" not in os.environ and case.cuda_visible_devices:
            env = {"CUDA_VISIBLE_DEVICES": case.cuda_visible_devices}
        return cmd, env

    def _router_env(self, l1_on: bool) -> dict[str, str]:
        return {
            "VLLM_ROUTER_STAGES": "1",
            "VLLM_ROUTER_L0_CACHE": "0",
            "VLLM_ROUTER_L1_CACHE": "1" if l1_on else "0",
        }

    def _router_cmd(self) -> list[str]:
        case = self.case
        return [
            self._router_bin,
            "--host",
            case.host,
            "--port",
            str(case.router_port),
            "--worker-urls",
            f"grpc://{case.host}:{case.worker_grpc_port}",
            "--prometheus-host",
            case.host,
            "--prometheus-port",
            str(case.prometheus_port),
            "--worker-startup-timeout-secs",
            str(int(self._timeout())),
        ]

    def _launch(self, l1_on: bool) -> None:
        worker_cmd, worker_env = self._worker_cmd()
        case = self.case
        timeout = self._timeout()
        worker = launch_worker(
            worker_cmd,
            label="vllm-rs",
            env=worker_env,
            ready_url=f"http://{case.host}:{case.worker_http_port}/v1/models",
            timeout=timeout,
            log_prefix="perf",
        )
        try:
            router = launch_router(
                self._router_cmd(),
                router_url=self.router_url,
                env=self._router_env(l1_on),
                timeout=timeout,
                keep_log=True,
                log_prefix="perf",
            )
        except Exception:
            worker.stop()
            raise
        self._worker = worker
        self._router = router
