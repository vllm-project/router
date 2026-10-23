"""kv_aware prefix reuse: observable cache signal via chat ``usage``."""

from __future__ import annotations

import time

import pytest
import requests

from py_test.e2e.function.helpers import http as router_http
from py_test.e2e.function.kv_aware.config import (
    KV_AWARE_E2E_MODEL,
    kv_aware_index_settle_sec,
    kv_aware_min_cached_on_reuse,
)
from py_test.e2e.function.kv_aware.dataset import default_prefix_reuse_scenario


@pytest.mark.e2e
def test_kv_aware_prefix_cache_reuse(kv_aware_router, kv_aware_two_workers):
    """
    Dataset-driven check that a **reuse** prompt hits prefix KV harder than a
    **control** prompt after a serial **prime**.

    Interpreting ``cached_tokens`` (worker ``usage``):
    - High reuse vs low control ⇒ likely same worker family + prefix cache
      (router kv_aware placement + engine prefix caching, end-to-end).
    - Does **not** assert gRPC worker URL directly (no response header today).
    - ``ptoken`` / internal ``cost`` components ⇒ ``cargo test`` on ``KvAwarePolicy``
      (see ``tests/kv_index_sequence_lifecycle.rs``).
    """
    _ = kv_aware_two_workers
    base = kv_aware_router.url
    model = KV_AWARE_E2E_MODEL
    scenario = default_prefix_reuse_scenario()
    settle = kv_aware_index_settle_sec()
    min_cached = kv_aware_min_cached_on_reuse()

    listed = requests.get(f"{base}/list_workers", timeout=30)
    listed.raise_for_status()
    assert len(listed.json().get("urls", [])) == 2

    with requests.Session() as session:
        prime_usage = router_http.post_chat_stream_collect_usage(
            session,
            base,
            model=model,
            content=scenario.prime_user,
            max_tokens=8,
        )
        time.sleep(settle)
        reuse_usage = router_http.post_chat_stream_collect_usage(
            session,
            base,
            model=model,
            content=scenario.reuse_user,
            max_tokens=8,
        )
        control_usage = router_http.post_chat_stream_collect_usage(
            session,
            base,
            model=model,
            content=scenario.control_user,
            max_tokens=8,
        )

    prime_cached = router_http.cached_tokens_from_usage(prime_usage)
    reuse_cached = router_http.cached_tokens_from_usage(reuse_usage)
    control_cached = router_http.cached_tokens_from_usage(control_usage)

    assert reuse_cached >= min_cached, (
        f"reuse cached_tokens={reuse_cached} expected >={min_cached} "
        f"(prime={prime_cached}, control={control_cached}, scenario={scenario.name})"
    )
    assert reuse_cached > control_cached, (
        f"reuse should beat control: reuse={reuse_cached} control={control_cached} "
        f"prime={prime_cached}"
    )
