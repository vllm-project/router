"""Load shared worker/router launch helpers from py_test/e2e/conftest.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_E2E_CONFTEST: ModuleType | None = None


def get_e2e_conftest() -> ModuleType:
    global _E2E_CONFTEST
    if _E2E_CONFTEST is not None:
        return _E2E_CONFTEST

    path = Path(__file__).resolve().parents[2] / "conftest.py"
    spec = importlib.util.spec_from_file_location("router_e2e_parent_conftest", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load e2e helpers from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _E2E_CONFTEST = mod
    return mod
