"""Functional e2e: kv_aware routing policy (PR #371).

Run (avoids importing ``py_test/e2e/conftest`` / ``vllm.test``)::

  pytest py_test/e2e/function/kv_aware --confcutdir=py_test/e2e/function/kv_aware -m e2e -v -s

Use ``-s`` to see vLLM worker startup logs on the terminal.

Workers need ``vllm`` on **the same interpreter as pytest** (see log line
``platform linux -- Python ...``). Prefer::

  .venv/bin/python -m pytest py_test/e2e/function/kv_aware/test_kv_aware.py -m e2e -v -s

Router binary (pick one): auto-detect ``target/release/vllm-router`` after
``cargo build --release``; or ``export ROUTER_BIN=/path/to/vllm-router``;
or ``pip install -e .`` (``python -m vllm_router.launch_router``).
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


def _prompt(suffix: str) -> str:
    return KV_AWARE_SHARED_PREFIX + suffix


@pytest.mark.e2e
def test_kv_aware_live_routing_and_streaming(
    kv_aware_router,
    kv_aware_two_workers,
):
    """
    Smoke-test kv_aware on a regular HTTP router with two vLLM workers.

    HTTP completions do not supply token IDs, so placement credit falls back to
    load-only scoring; this still validates discovery wiring, worker registration,
    selection, and request lifecycle (including streaming slot release).
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
        warm = router_http.post_completion(
            session,
            base,
            model=model,
            prompt=_prompt(" warm"),
            stream=False,
        )
        warm.raise_for_status()
        assert warm.json().get("choices")

        def one_request(i: int) -> None:
            r = router_http.post_completion(
                session,
                base,
                model=model,
                prompt=_prompt(f" req-{i}"),
                stream=False,
            )
            r.raise_for_status()
            body = r.json()
            assert body.get("choices"), body

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(one_request, range(16)))

        stream_resp = router_http.post_completion(
            session,
            base,
            model=model,
            prompt=_prompt(" stream"),
            stream=True,
            max_tokens=4,
        )
        stream_resp.raise_for_status()
        router_http.drain_completion_stream(stream_resp)
