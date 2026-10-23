"""kv_aware prefix reuse: TTFT / optional cached_tokens after a priming request."""

from __future__ import annotations

import time

import pytest
import requests

from py_test.e2e.function.helpers import http as router_http
from py_test.e2e.function.kv_aware.config import (
    KV_AWARE_E2E_MODEL,
    kv_aware_index_settle_sec,
    kv_aware_min_ttft_gap_ms,
)
from py_test.e2e.function.kv_aware.dataset import default_prefix_reuse_scenario


@pytest.mark.e2e
def test_kv_aware_prefix_cache_reuse(kv_aware_router, kv_aware_two_workers):
    """
    After **prime** (shared long prefix), **reuse** should beat **control** on an
    observable signal:

    - Primary: client **TTFT** (reuse faster than control with unique prefix lead-in).
    - Bonus: ``usage.cached_tokens`` when the gRPC router path forwards it (often 0 today).

    Router enables ``VLLM_ROUTER_STAGES=1`` in e2e fixtures for ``x-router-stages``.
    Does not assert which ``grpc://`` worker was picked.
    """
    _ = kv_aware_two_workers
    base = kv_aware_router.url
    model = KV_AWARE_E2E_MODEL
    scenario = default_prefix_reuse_scenario()
    settle = kv_aware_index_settle_sec()
    min_gap = kv_aware_min_ttft_gap_ms()

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
        time.sleep(settle)
        reuse = router_http.post_chat_stream_timed(
            session,
            base,
            model=model,
            content=scenario.reuse_user,
            max_tokens=8,
        )
        control = router_http.post_chat_stream_timed(
            session,
            base,
            model=model,
            content=scenario.control_user,
            max_tokens=8,
        )

    reuse_ttft = float(reuse["ttft_ms"])
    control_ttft = float(control["ttft_ms"])
    gap = control_ttft - reuse_ttft

    if reuse["cached_tokens"] > 0 or control["cached_tokens"] > 0:
        assert (
            reuse["cached_tokens"] >= control["cached_tokens"]
        ), f"cached_tokens reuse={reuse['cached_tokens']} control={control['cached_tokens']}"
    else:
        assert gap >= min_gap, (
            f"reuse TTFT should beat control by >={min_gap}ms (gap={gap:.1f}ms): "
            f"reuse={reuse_ttft:.1f} control={control_ttft:.1f} prime={float(prime['ttft_ms']):.1f} "
            f"scenario={scenario.name!r}. "
            "Sync latest test (TTFT not cached_tokens); workers need --enable-prefix-caching."
        )
