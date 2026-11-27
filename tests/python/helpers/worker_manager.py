"""Start mock workers from the test process.

The HTTP server itself stays in ``mock_worker.py`` and runs in a child process.
This manager starts regular workers, or the prefill and decode workers used by
PD disaggregation. The test starts the router itself through ``RouterManager``.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import requests

from tests.python.helpers.client import get_load
from tests.python.helpers.process import (
    release_all_port_holders,
    reserve_port,
    stop_any,
    wait_workers_ready,
)

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
        holders = []
        try:
            ports = [reserve_port(holders) for _ in range(n)]
            for index, port in enumerate(ports):
                holders[index].close()
                holders[index] = None
                process, url, worker_id = self._spawn(args, port)
                processes.append(process)
                urls.append(url)
                worker_ids.append(worker_id)
            wait_workers_ready(
                [
                    (f"{url}/health", process, worker_id)
                    for process, url, worker_id in zip(
                        processes, urls, worker_ids, strict=True
                    )
                ],
                10,
                prefix="mock",
                interval=0.1,
            )
        except Exception:
            for process in processes:
                stop_any(process, timeout=3)
                self._children.remove(process)
            raise
        finally:
            release_all_port_holders(holders)
        return processes, urls, worker_ids

    def add_prefill(
        self,
        n: int = 1,
        args: list[str] | None = None,
        bootstrap_port: int | None = None,
    ) -> tuple[list[str], list[str]]:
        """Start ``n`` prefill workers.

        The bootstrap port stays unset unless given.
        """
        _, urls, worker_ids = self.start_workers(n, args)
        self.prefill_urls.extend((url, bootstrap_port) for url in urls)
        self.prefill_ids.update(worker_ids)
        return urls, worker_ids

    def add_decode(
        self,
        n: int = 1,
        args: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Start ``n`` decode workers.

        Call again for a group that needs different ``args``.
        """
        _, urls, worker_ids = self.start_workers(n, args)
        self.decode_urls.extend(urls)
        self.decode_ids.update(worker_ids)
        return urls, worker_ids

    def wait_for_load(
        self, worker_id: str, minimum: int = 1, timeout: float = 5.0
    ) -> None:
        """Wait until a mock worker reports at least ``minimum`` in-flight load."""
        url = f"http://127.0.0.1:{str(worker_id).removeprefix('worker-')}"
        deadline = time.monotonic() + timeout
        last = None
        with requests.Session() as session:
            while time.monotonic() < deadline:
                last = get_load(url, session=session)
                if isinstance(last, int) and last >= minimum:
                    return
                time.sleep(0.02)
        raise AssertionError(f"{worker_id} load did not reach {minimum}: {last}")

    def stop_all(self) -> None:
        for process in self._children:
            stop_any(process, timeout=3)
        self._children.clear()

    def _spawn(self, args: list[str], port: int) -> tuple[subprocess.Popen, str, str]:
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
        return process, url, worker_id
