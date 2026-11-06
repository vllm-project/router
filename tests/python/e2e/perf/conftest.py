"""Shared path setup for manual perf e2e tests (grpc, router_policy)."""

from __future__ import annotations

import sys
from pathlib import Path

_E2E_DIR = Path(__file__).resolve().parents[1]
_HELPERS = _E2E_DIR / "helpers"
for _entry in (_HELPERS, _E2E_DIR):
    _path = str(_entry)
    if _path not in sys.path:
        sys.path.insert(0, _path)
