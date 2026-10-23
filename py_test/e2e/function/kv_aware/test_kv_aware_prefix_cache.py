"""kv_aware: reuse request should route to the same worker as prime."""

from __future__ import annotations

import time
from collections import Counter

import pytest
import requests

from py_test.e2e.function.helpers import http as router_http
from py_test.e2e.function.helpers.worker_logs import worker_url_for_request_id
from py_test.e2e.function.kv_aware.config import (
    KV_AWARE_E2E_MODEL,
    kv_aware_control_rounds,
    kv_aware_index_settle_sec,
    kv_aware_request_log_poll_sec,
    kv_aware_reuse_rounds,
)
from py_test.e2e.function.kv_aware.dataset import (
    control_user_message,
    default_prefix_reuse_scenario,
    reuse_user_message,
)


def _chat_worker(
    session: requests.Session,
    *,
    base: str,
    model: str,
    workers,
    content: str,
    log_poll: float,
) -> str:
    row = router_http.post_chat_stream_timed(
        session,
        base,
        model=model,
        content=content,
        max_tokens=8,
    )
    return worker_url_for_request_id(
        workers, row["request_id"], poll_timeout_sec=log_poll
    )


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
        prime_worker = _chat_worker(
            session,
            base=base,
            model=model,
            workers=workers,
            content=scenario.prime_user,
            log_poll=log_poll,
        )
        time.sleep(settle)
        reuse_worker = _chat_worker(
            session,
            base=base,
            model=model,
            workers=workers,
            content=scenario.reuse_user,
            log_poll=log_poll,
        )
        control_worker = _chat_worker(
            session,
            base=base,
            model=model,
            workers=workers,
            content=scenario.control_user,
            log_poll=log_poll,
        )

    assert reuse_worker == prime_worker, (
        f"kv_aware should send reuse to prime worker: "
        f"prime={prime_worker} reuse={reuse_worker} "
        f"(control={control_worker}, scenario={scenario.name!r})"
    )


@pytest.mark.e2e
def test_kv_aware_multi_reuse_and_control_routing(
    kv_aware_router, kv_aware_two_workers
):
    """
    After one **prime**, send many **reuse** (same token prefix, different tails)
    and many **control** (unrelated prompts).

    - Every **reuse** must land on ``prime_worker`` (hard assert).
    - **control** distribution is printed only (may not split 50/50; kv_aware
      is cost-based, not random).
    """
    workers = kv_aware_two_workers
    base = kv_aware_router.url
    model = KV_AWARE_E2E_MODEL
    settle = kv_aware_index_settle_sec()
    log_poll = kv_aware_request_log_poll_sec()
    n_reuse = kv_aware_reuse_rounds()
    n_control = kv_aware_control_rounds()

    with requests.Session() as session:
        prime_worker = _chat_worker(
            session,
            base=base,
            model=model,
            workers=workers,
            content=default_prefix_reuse_scenario().prime_user,
            log_poll=log_poll,
        )
        print(f"[e2e] prime_worker={prime_worker}", flush=True)
        time.sleep(settle)

        reuse_workers: list[str] = []
        for i in range(n_reuse):
            w = _chat_worker(
                session,
                base=base,
                model=model,
                workers=workers,
                content=reuse_user_message(
                    f" Follow-up #{i + 1} with the same prefix block."
                ),
                log_poll=log_poll,
            )
            reuse_workers.append(w)
            print(f"[e2e] reuse[{i + 1}/{n_reuse}] worker={w}", flush=True)

        control_workers: list[str] = []
        for i in range(n_control):
            w = _chat_worker(
                session,
                base=base,
                model=model,
                workers=workers,
                content=control_user_message(i + 1),
                log_poll=log_poll,
            )
            control_workers.append(w)
            print(f"[e2e] control[{i + 1}/{n_control}] worker={w}", flush=True)

    reuse_mismatch = [
        (i + 1, w) for i, w in enumerate(reuse_workers) if w != prime_worker
    ]
    assert not reuse_mismatch, (
        f"all reuse should use prime_worker={prime_worker!r}; "
        f"mismatches (round, worker)={reuse_mismatch!r}"
    )

    reuse_counts = Counter(reuse_workers)
    control_counts = Counter(control_workers)
    print(
        f"[e2e] summary prime={prime_worker} "
        f"reuse_counts={dict(reuse_counts)} control_counts={dict(control_counts)}",
        flush=True,
    )
