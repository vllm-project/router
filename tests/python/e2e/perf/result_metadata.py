"""Shared helpers for supplementing native benchmark result objects."""

from collections.abc import Mapping
from typing import Any


def add_missing_metadata(
    source: Mapping[str, Any], extra: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a JSON object with missing top-level fields filled from ``extra``.

    Existing source values win, including null/false/zero values. Nested objects
    are preserved as a whole, not recursively merged. Neither input is modified.
    Callers handle JSON loading/saving and choose the suite-specific fields.
    """
    result = dict(source)
    for key, value in extra.items():
        result.setdefault(key, value)
    return result
