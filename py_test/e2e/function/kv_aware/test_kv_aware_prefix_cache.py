"""kv_aware: reuse request should route to the same worker as prime."""

from __future__ import annotations

import concurrent.futures
import time
from collections import Counter

import pytest
import requests

from py_test.e2e.function.helpers import http as router_http
from py_test.e2e.function.helpers.worker_logs import worker_url_for_request_id
from py_test.e2e.function.kv_aware.config import (
    KV_AWARE_E2E_MODEL,
    kv_aware_dual_prime_concurrent,
    kv_aware_index_settle_sec,
    kv_aware_request_log_poll_sec,
    kv_aware_reuse_rounds,
)
from py_test.e2e.function.kv_aware.dataset import (
    default_prefix_reuse_scenario,
    lane_prime_user,
    lane_reuse_user,
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


def _prime_lane(
    lane: str,
    *,
    base: str,
    model: str,
    workers,
    log_poll: float,
) -> tuple[str, str]:
    with requests.Session() as session:
        worker = _chat_worker(
            session,
            base=base,
            model=model,
            workers=workers,
            content=lane_prime_user(lane),
            log_poll=log_poll,
        )
    return lane, worker


@pytest.mark.e2e
def test_kv_aware_reuse_routes_to_prime_worker(kv_aware_router, kv_aware_two_workers):
    """
    One **prime** + one **reuse** with a shared token prefix; reuse must use the
    same ``grpc://`` worker as prime.
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

    assert reuse_worker == prime_worker, (
        f"kv_aware should send reuse to prime worker: "
        f"prime={prime_worker} reuse={reuse_worker} "
        f"(scenario={scenario.name!r})"
    )


@pytest.mark.e2e
def test_kv_aware_dual_lane_prime_and_reuse(kv_aware_router, kv_aware_two_workers):
    """
    Two **lanes** (different long prefixes): prime A and prime B, then reuse per lane.

    - Lanes use **different** token prefixes (``alpha-`` vs ``beta-`` blocks).
    - Primes: concurrent by default (``KV_AWARE_DUAL_PRIME_CONCURRENT=1``); set ``0``
      for serial primes if cold-start routing puts both on one worker.
    - Expect **one prime per worker** when both workers are idle and costs differ
      after the first index update; if both primes land on the same worker, the
      test fails (see env above).
    - Each lane's reuse(s) must match that lane's ``prime_worker``.
    """
    workers = kv_aware_two_workers
    worker_urls = {w.url for w in workers}
    base = kv_aware_router.url
    model = KV_AWARE_E2E_MODEL
    settle = kv_aware_index_settle_sec()
    log_poll = kv_aware_request_log_poll_sec()
    n_reuse = kv_aware_reuse_rounds()
    concurrent_prime = kv_aware_dual_prime_concurrent()

    if concurrent_prime:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(
                _prime_lane,
                "A",
                base=base,
                model=model,
                workers=workers,
                log_poll=log_poll,
            )
            fut_b = pool.submit(
                _prime_lane,
                "B",
                base=base,
                model=model,
                workers=workers,
                log_poll=log_poll,
            )
            prime_pairs = [fut_a.result(), fut_b.result()]
    else:
        prime_pairs = [
            _prime_lane(
                "A",
                base=base,
                model=model,
                workers=workers,
                log_poll=log_poll,
            ),
            _prime_lane(
                "B",
                base=base,
                model=model,
                workers=workers,
                log_poll=log_poll,
            ),
        ]

    prime_by_lane = dict(prime_pairs)
    print(f"[e2e] prime_by_lane={prime_by_lane}", flush=True)
    assert set(prime_by_lane) == {"A", "B"}
    assert set(prime_by_lane.values()) <= worker_urls
    assert len(set(prime_by_lane.values())) == 2, (
        "expected lane A and lane B primes on two different workers; "
        f"got {prime_by_lane}. "
        "Cold concurrent primes often tie on one worker — retry with "
        "KV_AWARE_DUAL_PRIME_CONCURRENT=0 (serial primes)."
    )

    time.sleep(settle)

    reuse_by_lane: dict[str, list[str]] = {"A": [], "B": []}
    with requests.Session() as session:
        for lane in ("A", "B"):
            for r in range(1, n_reuse + 1):
                w = _chat_worker(
                    session,
                    base=base,
                    model=model,
                    workers=workers,
                    content=lane_reuse_user(lane, round_index=r),
                    log_poll=log_poll,
                )
                reuse_by_lane[lane].append(w)
                print(
                    f"[e2e] reuse lane {lane} [{r}/{n_reuse}] worker={w} "
                    f"(prime={prime_by_lane[lane]})",
                    flush=True,
                )

    for lane in ("A", "B"):
        bad = [
            (i + 1, w)
            for i, w in enumerate(reuse_by_lane[lane])
            if w != prime_by_lane[lane]
        ]
        assert not bad, (
            f"lane {lane}: reuse must stay on prime worker "
            f"{prime_by_lane[lane]!r}; mismatches (round, worker)={bad!r}"
        )

    print(
        f"[e2e] summary prime={prime_by_lane} "
        f"reuse_A={Counter(reuse_by_lane['A'])} "
        f"reuse_B={Counter(reuse_by_lane['B'])}",
        flush=True,
    )


@pytest.mark.e2e
def test_kv_aware_multi_reuse_after_single_prime(kv_aware_router, kv_aware_two_workers):
    """One prime, many reuse rounds on the same shared prefix (no control)."""
    workers = kv_aware_two_workers
    base = kv_aware_router.url
    model = KV_AWARE_E2E_MODEL
    settle = kv_aware_index_settle_sec()
    log_poll = kv_aware_request_log_poll_sec()
    n_reuse = kv_aware_reuse_rounds()

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

    reuse_mismatch = [
        (i + 1, w) for i, w in enumerate(reuse_workers) if w != prime_worker
    ]
    assert not reuse_mismatch, (
        f"all reuse should use prime_worker={prime_worker!r}; "
        f"mismatches (round, worker)={reuse_mismatch!r}"
    )
    print(
        f"[e2e] summary prime={prime_worker} reuse_counts={dict(Counter(reuse_workers))}",
        flush=True,
    )
