"""Prompt scenarios for kv_aware routing (shared token prefix on prime/reuse).

These are not a static file on disk — they are structured cases built from
``KV_AWARE_SHARED_PREFIX`` so the router and workers see a long shared leading
span, then different tails (mirroring ``benches/backend/bench_prefix_kvhit.py``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from py_test.e2e.function.kv_aware.config import KV_AWARE_SHARED_PREFIX


@dataclass(frozen=True)
class PrefixReuseScenario:
    """Three chat bodies: prime cache, reuse prefix, control miss."""

    name: str
    prime_user: str
    reuse_user: str
    control_user: str


def default_prefix_reuse_scenario() -> PrefixReuseScenario:
    """Shared long prefix on prime/reuse; control is unrelated text (true miss)."""
    stamp = time.time_ns()
    # Control must not repeat KV_AWARE_SHARED_PREFIX or it also prefix-hits after prime.
    control_filler = (
        f"Unrelated control request {stamp}. "
        "Discuss weather and hiking trails in one short sentence. " + ("misc-" * 48)
    )
    return PrefixReuseScenario(
        name="shared-prefix-reuse",
        prime_user=KV_AWARE_SHARED_PREFIX + " Prime this prefix on one worker.",
        reuse_user=KV_AWARE_SHARED_PREFIX + " Follow-up with the same prefix block.",
        control_user=control_filler,
    )
