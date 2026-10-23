"""Correlate OpenAI request ids with vllm-rs worker processes via per-worker log files."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


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
