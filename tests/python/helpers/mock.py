"""Mock prefill/decode deployment for router integration tests."""

from __future__ import annotations

import time
from typing import Any

import requests

from .client import get_load


class PdDisaggMock:
    """Prefill and decode mock workers behind one PD router.

    Integration tests that deploy PD with the mock-worker fixture share this.
    It does not start a real vLLM engine.
    """

    def __init__(self, router_manager, mock_workers):
        self._router_manager = router_manager
        self._mock_workers = mock_workers
        self.router = None
        self.prefill_urls: list[tuple[str, int | None]] = []
        self.decode_urls: list[str] = []
        self.prefill_ids: set[str] = set()
        self.decode_ids: set[str] = set()

    def add_prefill(
        self,
        n: int = 1,
        args: list[str] | None = None,
        bootstrap_port: int | None = None,
    ):
        """Start ``n`` prefill workers. Bootstrap port stays unset unless given."""
        _, urls, ids = self._mock_workers(n=n, args=args)
        self.prefill_urls.extend((url, bootstrap_port) for url in urls)
        self.prefill_ids.update(ids)
        return urls, ids

    def add_decode(self, n: int = 1, args: list[str] | None = None):
        """Start ``n`` decode workers. Call again for a group with different args."""
        _, urls, ids = self._mock_workers(n=n, args=args)
        self.decode_urls.extend(urls)
        self.decode_ids.update(ids)
        return urls, ids

    def start_router(self, policy: str, extra: dict[str, Any] | None = None):
        """Start the PD router in front of the workers added so far."""
        router_extra = {"worker_startup_check_interval": 1}
        if extra:
            router_extra.update(extra)
        self.router = self._router_manager.start_router(
            policy=policy,
            vllm_pd_disaggregation=True,
            prefill_urls=list(self.prefill_urls),
            decode_urls=list(self.decode_urls),
            extra=router_extra,
        )
        return self.router

    def start(
        self,
        policy: str,
        *,
        n_prefill: int = 2,
        n_decode: int = 3,
        prefill_args: list[str] | None = None,
        decode_args: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ):
        """Start one uniform prefill group, one uniform decode group, and the router."""
        self.add_prefill(n_prefill, prefill_args)
        self.add_decode(n_decode, decode_args)
        return self.start_router(policy, extra=extra), set(self.decode_ids)

    def wait_for_decode_load(
        self, worker_id: str, minimum: int = 1, timeout: float = 5.0
    ):
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
