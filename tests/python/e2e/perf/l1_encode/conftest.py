"""Fixtures for the L1 prefix-encoding-cache perf suite."""

from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

from tests.python.e2e.perf.l1_encode.check import hit_tag
from tests.python.e2e.perf.l1_encode.stack_launcher import L1Stack

_DIR = Path(__file__).resolve().parent

BENCH_SCRIPT = _DIR.parents[4] / "benches" / "backend" / "bench_prefix_kvhit.py"


def _progress(msg: str) -> None:
    print(f"[perf] {msg}", file=sys.stderr, flush=True)


def _save_results() -> bool:
    raw = os.environ.get("PERF_SAVE_RESULTS", "").strip().lower()
    return raw not in ("0", "false", "no")


def load_bench_module():
    spec = importlib.util.spec_from_file_location("bench_prefix_kvhit_l1", BENCH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BENCH_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pytest_generate_tests(metafunc):
    if metafunc.definition.name != "test_l1_encode":
        return
    if "hit_rate" not in metafunc.fixturenames:
        return
    case = getattr(metafunc.cls, "l1_case", None)
    if case is None:
        return
    params = []
    ids = []
    for hit_rate in case.hit_rates:
        for wave in case.waves:
            params.append((hit_rate, wave.concurrency, wave.requests))
            ids.append(f"hit{hit_tag(hit_rate)}-c{wave.concurrency}-r{wave.requests}")
    metafunc.parametrize("hit_rate,concurrency,requests", params, ids=ids)


@pytest.fixture(scope="session")
def bench_mod():
    return load_bench_module()


def pytest_pycollect_makeitem(collector, name, obj):
    """Collect ``test_name`` classes from the case JSON. Pytest's default only keeps ``Test*``."""
    if not isinstance(obj, type) or getattr(obj, "l1_case", None) is None:
        return None
    return pytest.Class.from_parent(collector, name=name)


@pytest.fixture(scope="class")
def l1_case(request):
    case = request.cls.l1_case
    if not case.model:
        pytest.skip("MODEL is empty and the case has no Hugging Face model id")
    os.environ["MODEL"] = case.model
    return case


@pytest.fixture(scope="session")
def l1_output_dir():
    if not _save_results():
        yield None
        return
    raw = os.environ.get("PERF_OUTPUT_DIR", "").strip()
    path = (
        Path(raw)
        if raw
        else _DIR / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    path.mkdir(parents=True, exist_ok=True)
    _progress(f"results directory: {path.resolve()}")
    yield path


@pytest.fixture(scope="class")
def l1_stack(l1_case, l1_output_dir):  # noqa: ARG001
    stack = L1Stack(l1_case, l1_case.model)
    stack.prepare()
    try:
        stack.start()
        yield stack
    finally:
        stack.stop()
