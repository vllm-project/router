"""kv_aware: reuse request should route to the same worker as prime."""

from __future__ import annotations

import time
from collections import Counter

import pytest
import requests

from tests.python.e2e.function.helpers import http as router_http
from tests.python.e2e.function.helpers.worker_logs import worker_url_for_request_id
from tests.python.e2e.function.kv_aware.config import (
    KV_AWARE_SHARED_PREFIX,
    kv_aware_e2e_model,
    kv_aware_index_settle_sec,
    kv_aware_request_log_poll_sec,
    kv_aware_reuse_rounds,
)
from tests.python.e2e.function.kv_aware.dataset import (
    default_prefix_reuse_scenario,
    lane_prefix,
    lane_prime_user,
    lane_reuse_user,
    reuse_user_message,
)


def _assert_lane_reuse_shares_prefix(lane: str, *, round_index: int = 1) -> None:
    """Reuse body must extend the same lane prefix as prime (KV credit input)."""
    prefix = lane_prefix(lane)
    prime = lane_prime_user(lane)
    reuse = lane_reuse_user(lane, round_index=round_index)
    assert prime.startswith(prefix) and reuse.startswith(prefix), (
        f"lane {lane}: prime/reuse must start with lane prefix"
    )
    assert len(reuse) > len(prefix) and len(prime) > len(prefix)


def _assert_dual_lane_prefixes_distinct() -> None:
    assert lane_prefix("A") != lane_prefix("B")
    assert not lane_prime_user("B").startswith(lane_prefix("A"))
    assert not lane_prime_user("A").startswith(lane_prefix("B"))


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


def _prime_worker_http(worker, *, model: str, content: str) -> str:
    """Seed one lane on a chosen worker, bypassing router scheduling."""
    http_url = getattr(worker, "http_url", None)
    if not http_url:
        raise RuntimeError(f"worker {getattr(worker, 'url', worker)!r} has no http_url")
    with requests.Session() as session:
        response = router_http.post_chat_completion(
            session,
            http_url,
            model=model,
            content=content,
            stream=False,
            max_tokens=8,
        )
        response.raise_for_status()
    return worker.url


@pytest.mark.e2e
def test_kv_aware_reuse_routes_to_prime_worker(kv_aware_router, kv_aware_two_workers):
    """
    One **prime** + one **reuse** with a shared token prefix; reuse must use the
    same ``grpc://`` worker as prime.
    """
    workers = kv_aware_two_workers
    base = kv_aware_router.url
    model = kv_aware_e2e_model()
    scenario = default_prefix_reuse_scenario()
    settle = kv_aware_index_settle_sec()
    log_poll = kv_aware_request_log_poll_sec()
    assert scenario.prime_user.startswith(KV_AWARE_SHARED_PREFIX)
    assert scenario.reuse_user.startswith(KV_AWARE_SHARED_PREFIX)
    print("[e2e] checkpoint: single-lane reuse shares prefix with prime", flush=True)

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
        print(f"[e2e] checkpoint: index settle {settle}s after prime", flush=True)
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
    - Primes go to each worker's own HTTP ``/v1/chat/completions``, lane A on
      the first worker and lane B on the second, so placement does not depend
      on router scheduling.
    - Each lane's reuse(s) must match that lane's ``prime_worker``.
    - Reuses are **interleaved B then A** each round so lane B is not sent only
      after many lane-A hits on one worker (load/KV index timing).

    **Checkpoints (preconditions before sticky assert):**

    1. Prompt contract — reuse shares lane prefix; A/B prefixes differ.
    2. Dual prime — lane A on workers[0], lane B on workers[1], via HTTP.
    3. Index settle — ``KV_AWARE_INDEX_SETTLE_SEC`` sleep after prime.
    4. First B reuse — one B reuse **before any A reuse**, must match ``prime_B``.
    """
    workers = kv_aware_two_workers
    worker_urls = {w.url for w in workers}
    base = kv_aware_router.url
    model = kv_aware_e2e_model()
    settle = kv_aware_index_settle_sec()
    log_poll = kv_aware_request_log_poll_sec()
    n_reuse = kv_aware_reuse_rounds()

    _assert_dual_lane_prefixes_distinct()
    for lane in ("A", "B"):
        _assert_lane_reuse_shares_prefix(lane, round_index=1)
    print(
        "[e2e] checkpoint: lane prompt contract OK (reuse shares prime prefix)",
        flush=True,
    )

    if len(workers) != 2:
        raise RuntimeError(f"dual-lane prime needs two workers, got {len(workers)}")
    prime_by_lane = {
        "A": _prime_worker_http(workers[0], model=model, content=lane_prime_user("A")),
        "B": _prime_worker_http(workers[1], model=model, content=lane_prime_user("B")),
    }
    print(f"[e2e] prime_by_lane={prime_by_lane}", flush=True)
    assert prime_by_lane == {"A": workers[0].url, "B": workers[1].url}
    assert set(prime_by_lane.values()) == worker_urls
    print("[e2e] checkpoint: dual prime on two workers OK", flush=True)

    print(f"[e2e] checkpoint: index settle {settle}s after prime", flush=True)
    time.sleep(settle)

    reuse_by_lane: dict[str, list[str]] = {"A": [], "B": []}
    with requests.Session() as session:
        w_b_first = _chat_worker(
            session,
            base=base,
            model=model,
            workers=workers,
            content=lane_reuse_user("B", round_index=1),
            log_poll=log_poll,
        )
        reuse_by_lane["B"].append(w_b_first)
        print(
            f"[e2e] checkpoint: first B reuse (no A reuse yet) worker={w_b_first} "
            f"prime_B={prime_by_lane['B']}",
            flush=True,
        )
        assert w_b_first == prime_by_lane["B"], (
            "checkpoint: first B reuse must hit B prime worker before any A reuse; "
            f"got reuse={w_b_first!r} prime_B={prime_by_lane['B']!r} "
            f"(prime_by_lane={prime_by_lane})"
        )

        for r in range(1, n_reuse + 1):
            for lane in ("B", "A"):
                if lane == "B" and r == 1:
                    continue
                _assert_lane_reuse_shares_prefix(lane, round_index=r)
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
    model = kv_aware_e2e_model()
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
