"""kv_aware: reuse request should route to the same worker as prime."""

from __future__ import annotations

import time

import pytest
import requests

from py_test.e2e.function.helpers import http as router_http
from py_test.e2e.function.helpers.worker_logs import worker_url_for_request_id
from py_test.e2e.function.kv_aware.config import (
    KV_AWARE_E2E_MODEL,
    kv_aware_index_settle_sec,
    kv_aware_request_log_poll_sec,
)
from py_test.e2e.function.kv_aware.dataset import default_prefix_reuse_scenario


@pytest.mark.e2e
def test_kv_aware_reuse_routes_to_prime_worker(kv_aware_router, kv_aware_two_workers):
    """
    Serial shared token-prefix scenario under **kv_aware**:

    1. **prime** — long shared prefix on one worker (KV events + index).
    2. **reuse** — same prefix; Router should select the same ``grpc://`` worker.
    3. **control** — unrelated text (logged for debugging; not asserted).

    Worker choice is resolved from per-worker vllm-rs logs using the OpenAI
    ``request_id`` on each streamed completion (see ``[e2e] worker logs: ...``).
    """
    workers = kv_aware_two_workers
    base = kv_aware_router.url
    model = KV_AWARE_E2E_MODEL
    scenario = default_prefix_reuse_scenario()
    settle = kv_aware_index_settle_sec()
    log_poll = kv_aware_request_log_poll_sec()

    listed = requests.get(f"{base}/list_workers", timeout=30)
    listed.raise_for_status()
    assert len(listed.json().get("urls", [])) == 2

    with requests.Session() as session:
        prime = router_http.post_chat_stream_timed(
            session,
            base,
            model=model,
            content=scenario.prime_user,
            max_tokens=8,
        )
        prime_worker = worker_url_for_request_id(
            workers, prime["request_id"], poll_timeout_sec=log_poll
        )
        time.sleep(settle)
        reuse = router_http.post_chat_stream_timed(
            session,
            base,
            model=model,
            content=scenario.reuse_user,
            max_tokens=8,
        )
        reuse_worker = worker_url_for_request_id(
            workers, reuse["request_id"], poll_timeout_sec=log_poll
        )
        control = router_http.post_chat_stream_timed(
            session,
            base,
            model=model,
            content=scenario.control_user,
            max_tokens=8,
        )
        control_worker = worker_url_for_request_id(
            workers, control["request_id"], poll_timeout_sec=log_poll
        )

    assert reuse_worker == prime_worker, (
        f"kv_aware should send reuse to prime worker: "
        f"prime={prime_worker} reuse={reuse_worker} "
        f"(control={control_worker}, scenario={scenario.name!r}, "
        f"prime_id={prime['request_id']!r}, reuse_id={reuse['request_id']!r})"
    )
