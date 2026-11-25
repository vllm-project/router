"""Build request inputs for the KV-aware functional tests."""

from __future__ import annotations

from tests.python.e2e.function.kv_aware.config import (
    KV_AWARE_LANE_A_PREFIX,
    KV_AWARE_LANE_B_PREFIX,
)


def lane_prefix(lane: str) -> str:
    """Return one of the two independent prefixes for the dual-lane test."""
    if lane == "A":
        return KV_AWARE_LANE_A_PREFIX
    if lane == "B":
        return KV_AWARE_LANE_B_PREFIX
    raise ValueError(f"unknown lane {lane!r}")


def lane_prime_user(lane: str) -> str:
    """Build the direct-to-worker priming prompt for the dual-lane test."""
    return lane_prefix(lane) + f" Prime lane {lane} on one worker."


def lane_reuse_user(lane: str, *, round_index: int = 1) -> str:
    """Build a dual-lane probe that keeps the primed prefix and varies its tail."""
    return lane_prefix(lane) + (
        f" Follow-up lane {lane} round {round_index} same prefix block."
    )


def build_hybrid_prefix_requests(block_size: int) -> dict[str, list[int]]:
    """Build the priming prompts and token-ID probes for the hybrid reuse test."""
    prefix = list(range(1000, 1000 + 3 * block_size + block_size // 2))
    # The partial request keeps two complete cache pages.
    partial_prefix = prefix[: 2 * block_size]
    return {
        "partial_prime": [*partial_prefix, 4000],
        "warm_prime": [*prefix, 4001],
        "warm": [*prefix, 4002],
        "partial": [*partial_prefix, 4003],
        "cold": list(range(5000, 5001 + block_size)),
    }
