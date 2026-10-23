"""Functional e2e: kv_aware routing policy (PR #371).

Uses **vllm-rs serve** workers and ``grpc://`` worker URLs so the router
tokenizes ``/v1/chat/completions`` and can exercise kv_aware discovery.
Run (avoids importing ``py_test/e2e/conftest`` / ``vllm.test``)::

  pytest py_test/e2e/function/kv_aware --confcutdir=py_test/e2e/function/kv_aware -m e2e -v -s

Prerequisites: ``vllm-rs`` (``export VLLM_RS=...`` or ``scripts/backend/check_vllm_rs.sh``),
``ROUTER_BIN`` or ``cargo build --release``, two GPUs.

  .venv/bin/python -m pytest py_test/e2e/function/kv_aware/test_kv_aware.py -m e2e -v -s

Use ``-s`` so vllm-rs worker and vllm-router logs appear on the terminal.

Stronger prefix / cache checks: ``test_kv_aware_prefix_cache.py``.
"""

from __future__ import annotations

import concurrent.futures

import pytest
import requests

from py_test.e2e.function.helpers import http as router_http
from py_test.e2e.function.kv_aware.config import (
    KV_AWARE_E2E_MODEL,
    KV_AWARE_SHARED_PREFIX,
)


def _user_message(suffix: str) -> str:
    return KV_AWARE_SHARED_PREFIX + suffix


@pytest.mark.e2e
def test_kv_aware_live_routing_and_streaming(
    kv_aware_router,
    kv_aware_two_workers,
):
    """
    Smoke-test kv_aware with two gRPC (vllm-rs) workers.

    Exercises router tokenization on chat, worker registration, concurrent
    routing, and streaming. Stronger KV placement assertions can be added once
    observability (worker id / metrics) is wired in the test harness.
    Requires at least two GPUs (see ``kv_aware_two_workers``).
    """
    base = kv_aware_router.url
    workers = kv_aware_two_workers
    model = KV_AWARE_E2E_MODEL

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
            r = router_http.post_chat_completion(
                session,
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

        stream_resp = router_http.post_chat_completion(
            session,
            base,
            model=model,
            content=_user_message(" stream"),
            stream=True,
            max_tokens=4,
        )
        stream_resp.raise_for_status()
        router_http.drain_sse_stream(stream_resp)
