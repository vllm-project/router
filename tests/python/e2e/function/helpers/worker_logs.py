"""Worker subprocess logs: capture stdout, then match a request id to one worker."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


class SubprocessLogPump:
    """``Popen(stdout=PIPE)`` lines → log file; optionally echo to ``sys.stderr``."""

    def __init__(
        self,
        proc: subprocess.Popen[str],
        log_path: Path | str,
        *,
        echo: bool = True,
        prefix: str = "",
    ) -> None:
        if proc.stdout is None:
            raise ValueError("proc.stdout must be PIPE for SubprocessLogPump")
        self._proc = proc
        self._log_path = Path(log_path)
        self._echo = echo
        self._prefix = prefix
        self._log_file = open(self._log_path, "w", encoding="utf-8", buffering=1)
        self._thread = threading.Thread(
            target=self._pump, name=f"log-pump-{self._log_path.name}", daemon=True
        )
        self._thread.start()

    def _pump(self) -> None:
        stream = self._proc.stdout
        assert stream is not None
        for line in iter(stream.readline, ""):
            self._log_file.write(line)
            self._log_file.flush()
            if self._echo:
                sys.stderr.write(f"{self._prefix}{line}")
                sys.stderr.flush()
        stream.close()

    def close(self, *, join_timeout: float = 30.0) -> None:
        self._thread.join(timeout=join_timeout)
        try:
            self._log_file.close()
        except OSError:
            pass


def worker_url_for_request_id(
    workers: list[Any],
    request_id: str,
    *,
    poll_timeout_sec: float = 60.0,
    poll_interval_sec: float = 0.25,
) -> str:
    """Return ``grpc://`` URL of the sole worker whose log mentions ``request_id``."""
    if not request_id:
        raise ValueError("empty request_id")
    pairs: list[tuple[str, Path]] = []
    for w in workers:
        path = getattr(w, "log_path", None)
        if path is None:
            raise RuntimeError(
                "workers must have log_path set (spawn with worker log capture)"
            )
        pairs.append((w.url, Path(path)))

    deadline = time.perf_counter() + poll_timeout_sec
    while time.perf_counter() < deadline:
        hits = [
            url
            for url, path in pairs
            if request_id in path.read_text(encoding="utf-8", errors="replace")
        ]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise AssertionError(
                f"request_id {request_id!r} appears on multiple workers: {hits}"
            )
        time.sleep(poll_interval_sec)

    log_paths = [str(p) for _, p in pairs]
    raise AssertionError(
        f"request_id {request_id!r} not found in worker logs within "
        f"{poll_timeout_sec}s (logs: {log_paths})"
    )
