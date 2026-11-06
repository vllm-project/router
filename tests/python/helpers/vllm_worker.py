"""Start real vLLM worker processes from the test process.

The command builder stays in ``tests.python.helpers.e2e.process`` and is
imported only when a worker is launched. Mock workers use ``MockWorkerManager``.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

from tests.python.helpers.e2e.process import _popen_launch_worker, _terminate
from tests.python.helpers.ports import find_free_port


class VllmWorkerManager:
    """Launch real vLLM workers.

    ``primary`` and ``dp2_pair`` are reused for the whole session. ``start``
    launches a worker that stops when the test ends.
    """

    def __init__(self, shared: VllmWorkerManager | None = None):
        self._shared = shared
        self._procs: list[subprocess.Popen] = []
        self._cache: dict = {}

    def start(
        self,
        model: str,
        *,
        dp_size: int | None = None,
        api_key: str | None = None,
        base_gpu_id: int | None = 0,
    ) -> SimpleNamespace:
        """Start one worker. Returns an object with ``proc`` and ``url``."""
        return self._launch(
            model, dp_size=dp_size, api_key=api_key, base_gpu_id=base_gpu_id
        )

    def primary(self, model: str) -> SimpleNamespace:
        """Return the session's single worker for ``model``, starting it once."""
        if self._shared is not None:
            return self._shared.primary(model)
        return self._cached(("primary", model), lambda: self._launch(model))

    def dp2_pair(self, model: str) -> list[SimpleNamespace]:
        """Return two dp=2 workers on GPUs 0-1 and 2-3, starting them once."""
        if self._shared is not None:
            return self._shared.dp2_pair(model)

        def launch() -> list[SimpleNamespace]:
            return [
                self._launch(model, dp_size=2, base_gpu_id=0),
                self._launch(model, dp_size=2, base_gpu_id=2),
            ]

        return self._cached(("dp2-pair", model), launch)

    def stop_all(self) -> None:
        for proc in self._procs:
            if proc.poll() is None:
                _terminate(proc)
        self._procs.clear()
        self._cache.clear()

    def _cached(self, key: tuple, factory):
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    def _launch(
        self,
        model: str,
        *,
        dp_size: int | None = None,
        api_key: str | None = None,
        base_gpu_id: int | None = 0,
    ) -> SimpleNamespace:
        port = find_free_port()
        url = f"http://127.0.0.1:{port}"
        proc = _popen_launch_worker(
            model,
            url,
            dp_size=dp_size,
            api_key=api_key,
            base_gpu_id=base_gpu_id,
        )
        self._procs.append(proc)
        return SimpleNamespace(proc=proc, url=url)
