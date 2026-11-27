"""Prompt text for the kv_aware dual-lane stickiness test.

Lane A and lane B use different long prefixes so each reuse can be credited
only to the worker that was primed with that lane.
"""

from __future__ import annotations

from tests.python.e2e.function.kv_aware.config import (
    KV_AWARE_LANE_A_PREFIX,
    KV_AWARE_LANE_B_PREFIX,
)


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
