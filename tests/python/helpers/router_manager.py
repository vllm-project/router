import os
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional
from urllib.parse import urlparse

import requests

from .process import release_all_port_holders, reserve_port


@dataclass
class ProcHandle:
    process: subprocess.Popen
    url: str
    prometheus_port: Optional[int] = None


class _RouterExited(Exception):
    """The router process exited before /health returned 200."""


def _tee(stream, tail: Deque[str]) -> None:
    # sys.__stderr__ is fd 2, where the router wrote before its output was piped.
    mirror = sys.__stderr__
    # Close the pipe at EOF: nothing else reads or closes proc.stdout.
    with stream:
        for line in stream:
            tail.append(line)
            if mirror is None:
                continue
            # Best effort: this is the only reader, so a closed stderr must not
            # stop draining the pipe or hide a bind failure from tail.
            try:
                mirror.write(line)
                mirror.flush()
            except (OSError, ValueError):
                mirror = None


def _owns_listener(pid: int, port: int) -> Optional[bool]:
    """Whether pid or a descendant holds a LISTEN socket on port.

    Returns None without Linux /proc, so callers can skip the check.
    """
    if not os.path.exists("/proc/net/tcp"):
        return None
    inodes = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(table) as f:
                rows = f.read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            # local_address is HEX_IP:HEX_PORT, st 0A is LISTEN, field 9 is inode.
            fields = row.split()
            if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(f"socket:[{fields[9]}]")
    if not inodes:
        # /health answered, so someone listens; tables that show nobody can't
        # be trusted (e.g. a sandboxed /proc).
        return None
    pids = [pid]
    while pids:
        p = pids.pop()
        # Any of these can vanish mid-scan if the process exits.
        try:
            fds = os.listdir(f"/proc/{p}/fd")
            tasks = os.listdir(f"/proc/{p}/task")
        except OSError:
            continue
        for fd in fds:
            try:
                if os.readlink(f"/proc/{p}/fd/{fd}") in inodes:
                    return True
            except OSError:
                pass
        for task in tasks:
            try:
                with open(f"/proc/{p}/task/{task}/children") as f:
                    pids.extend(int(child) for child in f.read().split())
            except OSError:
                pass
    return False


class RouterManager:
    """Helper to spawn a router process and interact with admin endpoints."""

    def __init__(self):
        self._children: List[subprocess.Popen] = []

    def start_router(
        self,
        worker_urls: Optional[List[str]] = None,
        policy: str = "round_robin",
        port: Optional[int] = None,
        extra: Optional[Dict] = None,
        # PD options
        vllm_pd_disaggregation: bool = False,
        prefill_urls: Optional[List[tuple]] = None,
        decode_urls: Optional[List[str]] = None,
        prefill_policy: Optional[str] = None,
        decode_policy: Optional[str] = None,
        # Prefer the Rust binary when testing features that live in the host crate
        # (e.g. WASM middleware) without requiring a rebuilt Python wheel.
        router_bin: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        *,
        timeout: float = 30.0,
    ) -> ProcHandle:
        worker_urls = worker_urls or []
        pick_port = not port
        holders: List[Optional[socket.socket]] = []
        port = port or reserve_port(holders)
        router_bin = router_bin or os.environ.get("VLLM_ROUTER_TEST_BIN")
        if router_bin:
            cmd = [
                router_bin,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--policy",
                policy,
            ]
        else:
            cmd = [
                sys.executable,
                "-m",
                "vllm_router.launch_router",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--policy",
                policy,
            ]
        prom_port = reserve_port(holders)
        # A caller's API port isn't reserved, so the allocator can return it.
        while prom_port == port:
            prom_port = reserve_port(holders)
        cmd.extend(
            ["--prometheus-port", str(prom_port), "--prometheus-host", "127.0.0.1"]
        )
        if worker_urls:
            cmd.extend(["--worker-urls", *worker_urls])

        # PD routing configuration
        if vllm_pd_disaggregation:
            cmd.append("--vllm-pd-disaggregation")
            if prefill_urls:
                for url, bport in prefill_urls:
                    if bport is None:
                        cmd.extend(["--prefill", url, "none"])
                    else:
                        cmd.extend(["--prefill", url, str(bport)])
            if decode_urls:
                for url in decode_urls:
                    cmd.extend(["--decode", url])
            if prefill_policy:
                cmd.extend(["--prefill-policy", prefill_policy])
            if decode_policy:
                cmd.extend(["--decode-policy", decode_policy])

        # Map supported extras to CLI flags (subset for integration)
        if extra:
            flag_map = {
                "max_payload_size": "--max-payload-size",
                "intra_node_data_parallel_size": "--intra-node-data-parallel-size",
                "api_key": "--api-key",
                # Health/monitoring
                "worker_startup_check_interval": "--worker-startup-check-interval",
                "worker_startup_timeout_secs": "--worker-startup-timeout-secs",
                # Cache-aware tuning
                "cache_threshold": "--cache-threshold",
                "balance_abs_threshold": "--balance-abs-threshold",
                "balance_rel_threshold": "--balance-rel-threshold",
                "router_temperature": "--router-temperature",
                "kv_penalty_host": "--kv-penalty-host",
                "kv_penalty_external": "--kv-penalty-external",
                "kv_penalty_disk": "--kv-penalty-disk",
                # kv_aware decode session affinity
                "kv_aware_session_affinity_ttl_secs": "--kv-aware-session-affinity-ttl-secs",
                # Program scheduling
                "enable_program_scheduling": "--enable-program-scheduling",
                "program_scheduling_config_json": "--program-scheduling-config-json",
                # Retry
                "retry_max_retries": "--retry-max-retries",
                "retry_initial_backoff_ms": "--retry-initial-backoff-ms",
                "retry_max_backoff_ms": "--retry-max-backoff-ms",
                "retry_backoff_multiplier": "--retry-backoff-multiplier",
                "retry_jitter_factor": "--retry-jitter-factor",
                "disable_retries": "--disable-retries",
                # Circuit breaker
                "cb_failure_threshold": "--cb-failure-threshold",
                "cb_success_threshold": "--cb-success-threshold",
                "cb_timeout_duration_secs": "--cb-timeout-duration-secs",
                "cb_window_duration_secs": "--cb-window-duration-secs",
                "disable_circuit_breaker": "--disable-circuit-breaker",
                # Rate limiting
                "max_concurrent_requests": "--max-concurrent-requests",
                "queue_size": "--queue-size",
                "queue_timeout_secs": "--queue-timeout-secs",
                "rate_limit_tokens_per_second": "--rate-limit-tokens-per-second",
                # WASM OnRequest middleware
                "wasm_middleware": "--wasm-middleware",
                "wasm_middleware_sha256": "--wasm-middleware-sha256",
            }
            for k, v in extra.items():
                if v is None:
                    continue
                if k == "wasm_middleware_routes":
                    for route in v:
                        cmd.extend(["--wasm-middleware-route", str(route)])
                    continue
                flag = flag_map.get(k)
                if not flag:
                    continue
                if isinstance(v, bool):
                    if v:
                        cmd.append(flag)
                else:
                    cmd.extend([flag, str(v)])

        # Reserved ports are freed just before Popen, so another process can
        # still take one first. Retry when the router says its bind failed.
        attempts = 3
        for attempt in range(1, attempts + 1):
            release_all_port_holders(holders)
            proc = subprocess.Popen(
                cmd,
                env={**os.environ, **(env or {})},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
            )
            tail: Deque[str] = deque(maxlen=50)
            reader = threading.Thread(
                target=_tee, args=(proc.stdout, tail), daemon=True
            )
            reader.start()
            self._children.append(proc)
            url = f"http://127.0.0.1:{port}"
            try:
                self._wait_health(url, timeout, proc=proc)
                return ProcHandle(process=proc, url=url, prometheus_port=prom_port)
            except _RouterExited:
                proc.wait()
                reader.join(timeout=5)
                self._children.remove(proc)
                output = "".join(tail)
                # API bind: "AddrInUse"; Prometheus bind: "Address already in use".
                clash = "AddrInUse" in output or "Address already in use" in output
                if not clash or attempt == attempts:
                    raise RuntimeError(
                        f"Router exited with code {proc.returncode} before becoming "
                        f"healthy (port={port}, prometheus_port={prom_port}, "
                        f"port_clash={clash}, attempt {attempt}/{attempts}). "
                        f"Last output:\n{output}"
                    ) from None
            if pick_port:
                port = reserve_port(holders)
            prom_port = reserve_port(holders)
            while prom_port == port:
                prom_port = reserve_port(holders)
            cmd[cmd.index("--port") + 1] = str(port)
            cmd[cmd.index("--prometheus-port") + 1] = str(prom_port)

    def _wait_health(
        self,
        base_url: str,
        timeout: float = 30.0,
        proc: Optional[subprocess.Popen] = None,
    ):
        start = time.time()
        with requests.Session() as s:
            while time.time() - start < timeout:
                if proc is not None and proc.poll() is not None:
                    raise _RouterExited
                try:
                    r = s.get(f"{base_url}/health", timeout=2)
                    # Another process can win the port and answer for a child
                    # that is about to exit on its failed bind.
                    if r.status_code == 200 and (
                        proc is None
                        or _owns_listener(proc.pid, urlparse(base_url).port)
                        is not False
                    ):
                        return
                except requests.RequestException:
                    pass
                time.sleep(0.2)
        raise TimeoutError(f"Router at {base_url} did not become healthy")

    def add_worker(
        self, base_url: str, worker_url: str, timeout: Optional[float] = None
    ) -> None:
        r = requests.post(
            f"{base_url}/add_worker", params={"url": worker_url}, timeout=timeout
        )
        assert r.status_code == 200, f"add_worker failed: {r.status_code} {r.text}"

    def remove_worker(self, base_url: str, worker_url: str) -> None:
        r = requests.post(f"{base_url}/remove_worker", params={"url": worker_url})
        assert r.status_code == 200, f"remove_worker failed: {r.status_code} {r.text}"

    def list_workers(self, base_url: str) -> list[str]:
        r = requests.get(f"{base_url}/list_workers")
        assert r.status_code == 200, f"list_workers failed: {r.status_code} {r.text}"
        data = r.json()
        return data.get("urls", [])

    def stop_all(self):
        for p in self._children:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        self._children.clear()
