"""Prompt scenarios for kv_aware routing (shared token prefix on prime/reuse).

These are not a static file on disk — they are structured cases built from
``KV_AWARE_SHARED_PREFIX`` so the router and workers see a long shared leading
span, then different tails (mirroring ``benches/backend/bench_prefix_kvhit.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from tests.python.e2e.function.kv_aware.config import (
    KV_AWARE_LANE_A_PREFIX,
    KV_AWARE_LANE_B_PREFIX,
    KV_AWARE_SHARED_PREFIX,
)


@dataclass(frozen=True)
class PrefixReuseScenario:
    """Two chat bodies: prime cache, then reuse the same prefix."""

    name: str
    prime_user: str
    reuse_user: str


def default_prefix_reuse_scenario() -> PrefixReuseScenario:
    """Shared long prefix on prime and reuse."""
    return PrefixReuseScenario(
        name="shared-prefix-reuse",
        prime_user=KV_AWARE_SHARED_PREFIX + " Prime this prefix on one worker.",
        reuse_user=KV_AWARE_SHARED_PREFIX + " Follow-up with the same prefix block.",
    )


def reuse_user_message(suffix: str) -> str:
    """Same shared token prefix as prime; only the trailing text differs."""
    return KV_AWARE_SHARED_PREFIX + suffix


def lane_prefix(lane: str) -> str:
    if lane == "A":
        return KV_AWARE_LANE_A_PREFIX
    if lane == "B":
        return KV_AWARE_LANE_B_PREFIX
    raise ValueError(f"unknown lane {lane!r}")


def lane_prime_user(lane: str) -> str:
    return lane_prefix(lane) + f" Prime lane {lane} on one worker."


def lane_reuse_user(lane: str, *, round_index: int = 1) -> str:
    return lane_prefix(lane) + (
        f" Follow-up lane {lane} round {round_index} same prefix block."
    )


def assert_lane_reuse_shares_prefix(lane: str, *, round_index: int = 1) -> None:
    """Reuse body must extend the same lane prefix as prime (KV credit input)."""
    prefix = lane_prefix(lane)
    prime = lane_prime_user(lane)
    reuse = lane_reuse_user(lane, round_index=round_index)
    assert prime.startswith(prefix) and reuse.startswith(prefix), (
        f"lane {lane}: prime/reuse must start with lane prefix"
    )
    assert len(reuse) > len(prefix) and len(prime) > len(prefix)


def assert_dual_lane_prefixes_distinct() -> None:
    assert lane_prefix("A") != lane_prefix("B")
    assert not lane_prime_user("B").startswith(lane_prefix("A"))
    assert not lane_prime_user("A").startswith(lane_prefix("B"))
