"""Start mock workers from the test process.

The HTTP server itself stays in ``mock_worker.py`` and runs in a child process.
This manager starts regular workers, or the prefill and decode workers used by
PD disaggregation and PD response merge. The test starts the router itself
through ``RouterManager``.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import requests

from tests.python.helpers.client import get_load
from tests.python.helpers.ports import find_free_port

_MOCK_WORKER = Path(__file__).resolve().parent / "mock_worker.py"


class MockWorkerManager:
    """Spawn mock workers for regular routing and for PD prefill or decode."""

    def __init__(self):
        self._children: list[subprocess.Popen] = []
        self.prefill_urls: list[tuple[str, int | None]] = []
        self.decode_urls: list[str] = []
        self.prefill_ids: set[str] = set()
        self.decode_ids: set[str] = set()

    def start_worker(
        self, args: list[str] | None = None
    ) -> tuple[subprocess.Popen, str, str]:
        """Start one mock worker. Returns ``(process, url, worker_id)``."""
        processes, urls, worker_ids = self.start_workers(1, args)
        return processes[0], urls[0], worker_ids[0]

    def start_workers(
        self, n: int = 1, args: list[str] | None = None
    ) -> tuple[list[subprocess.Popen], list[str], list[str]]:
        """Start ``n`` mock workers that share ``args``.

        Returns ``(processes, urls, worker_ids)`` for this call only.
        """
        args = list(args or [])
        processes: list[subprocess.Popen] = []
        urls: list[str] = []
        worker_ids: list[str] = []
        for _ in range(n):
            process, url, worker_id = self._spawn(args)
            processes.append(process)
            urls.append(url)
            worker_ids.append(worker_id)
        return processes, urls, worker_ids

    def add_prefill(
        self,
        n: int = 1,
        args: list[str] | None = None,
        bootstrap_port: int | None = None,
        *,
        merge: bool = False,
        nixl_push: bool = False,
    ) -> tuple[list[str], list[str]]:
        """Start ``n`` prefill workers.

        ``merge`` makes each worker return the prefill half of a PD response.
        ``nixl_push`` marks that half as a NIXL push transfer and requires
        ``merge``. The bootstrap port stays unset unless given.
        """
        if nixl_push and not merge:
            raise ValueError("nixl_push requires merge=True")
        _, urls, worker_ids = self.start_workers(
            n, _role_args("prefill" if merge else None, nixl_push, args)
        )
        self.prefill_urls.extend((url, bootstrap_port) for url in urls)
        self.prefill_ids.update(worker_ids)
        return urls, worker_ids

    def add_decode(
        self,
        n: int = 1,
        args: list[str] | None = None,
        *,
        merge: bool = False,
    ) -> tuple[list[str], list[str]]:
        """Start ``n`` decode workers.

        ``merge`` makes each worker return the decode half of a PD response.
        Call again for a group that needs different ``args``.
        """
        _, urls, worker_ids = self.start_workers(
            n, _role_args("decode" if merge else None, False, args)
        )
        self.decode_urls.extend(urls)
        self.decode_ids.update(worker_ids)
        return urls, worker_ids

    def wait_for_decode_load(
        self, worker_id: str, minimum: int = 1, timeout: float = 5.0
    ) -> None:
        """Wait until a mock decode worker reports at least ``minimum`` in-flight load."""
        url = f"http://127.0.0.1:{str(worker_id).removeprefix('worker-')}"
        deadline = time.time() + timeout
        last = None
        with requests.Session() as session:
            while time.time() < deadline:
                last = get_load(url, session=session)
                if isinstance(last, int) and last >= minimum:
                    return
                time.sleep(0.02)
        raise AssertionError(f"{worker_id} load did not reach {minimum}: {last}")

    def stop_all(self) -> None:
        for process in self._children:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
        self._children.clear()

    def _spawn(self, args: list[str]) -> tuple[subprocess.Popen, str, str]:
        port = find_free_port()
        worker_id = f"worker-{port}"
        cmd = [
            sys.executable,
            str(_MOCK_WORKER),
            "--port",
            str(port),
            "--worker-id",
            worker_id,
            *args,
        ]
        process = subprocess.Popen(cmd)
        self._children.append(process)
        url = f"http://127.0.0.1:{port}"
        _wait_health(url)
        return process, url, worker_id


def _role_args(role: str | None, nixl_push: bool, args: list[str] | None) -> list[str]:
    extra = list(args or [])
    if role is not None:
        extra = ["--pd-role", role, *extra]
    if nixl_push:
        extra.append("--nixl-push")
    return extra


def _wait_health(url: str, timeout: float = 10.0) -> None:
    start = time.time()
    with requests.Session() as session:
        while time.time() - start < timeout:
            try:
                response = session.get(f"{url}/health", timeout=1)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.1)
    raise TimeoutError(f"Mock worker at {url} did not become healthy")
