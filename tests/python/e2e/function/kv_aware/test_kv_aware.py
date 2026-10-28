"""Functional e2e: kv_aware routing policy (PR #371).

Uses two ``mock-grpc-worker`` processes and ``grpc://`` worker URLs. The router
still tokenizes ``/v1/chat/completions`` with ``VLLM_ROUTER_MODEL``.
Run (avoids importing ``tests/python/e2e/conftest`` / ``vllm.test``)::

  pytest tests/python/e2e/function/kv_aware --confcutdir=tests/python/e2e/function/kv_aware -m e2e -v -s

Prerequisites: ``cargo`` on ``PATH`` (the fixture builds the mock), and
``ROUTER_BIN`` or ``cargo build --release``. No GPU and no ``vllm-rs``.
Prefix-stickiness tests still expect real KV-cache events, which this mock
does not publish.

  .venv/bin/python -m pytest tests/python/e2e/function/kv_aware/test_kv_aware.py -m e2e -v -s

Use ``-s`` so mock worker and vllm-router logs appear on the terminal.

kv_aware worker stickiness: ``test_kv_aware_prefix_cache.py`` (reuse → same worker as prime).
"""

from __future__ import annotations

import concurrent.futures

import pytest
import requests

from tests.python.e2e.function.helpers import http as router_http
from tests.python.e2e.function.kv_aware.config import (
    KV_AWARE_SHARED_PREFIX,
    kv_aware_e2e_model,
)


def _user_message(suffix: str) -> str:
    return KV_AWARE_SHARED_PREFIX + suffix


@pytest.mark.e2e
def test_kv_aware_live_routing_and_streaming(
    kv_aware_router,
    kv_aware_two_workers,
):
    """
    Smoke-test kv_aware with two mock gRPC workers.

    Exercises router tokenization on chat, worker registration, concurrent
    routing, and streaming. The mock returns a fixed completion string.
    """
    base = kv_aware_router.url
    workers = kv_aware_two_workers
    model = kv_aware_e2e_model()

    listed = requests.get(f"{base}/list_workers", timeout=30)
    listed.raise_for_status()
    urls = listed.json().get("urls", [])
    assert len(urls) == 2
    assert set(urls) == {w.url for w in workers}

    with requests.Session() as session:
        warm = router_http.post_chat_completion(
            session,
            base,
            model=model,
            content=_user_message(" warm"),
            stream=False,
        )
        warm.raise_for_status()
        choices = warm.json().get("choices") or []
        assert choices
        assert choices[0].get("message", {}).get("content") is not None

        def one_request(i: int) -> None:
            with requests.Session() as thread_session:
                r = router_http.post_chat_completion(
                    thread_session,
                    base,
                    model=model,
                    content=_user_message(f" req-{i}"),
                    stream=False,
                )
                r.raise_for_status()
                body = r.json()
                assert body.get("choices"), body

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(one_request, range(16)))

        streamed = router_http.post_chat_stream_timed(
            session,
            base,
            model=model,
            content=_user_message(" stream"),
            max_tokens=4,
        )
        assert streamed["ttft_ms"] >= 0
        assert streamed["request_id"]
