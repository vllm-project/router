"""Fixtures for the L1 prefix-encoding-cache perf suite."""

from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

from tests.python.e2e.perf.l1_encode.case_loader import load_case
from tests.python.e2e.perf.l1_encode.report import wave_id
from tests.python.e2e.perf.l1_encode.stack import ExternalStack, L1Stack

_DIR = Path(__file__).resolve().parent

BENCH_SCRIPT = _DIR.parents[4] / "benches" / "backend" / "bench_prefix_kvhit.py"


def _progress(msg: str) -> None:
    print(f"[perf] {msg}", file=sys.stderr, flush=True)


def _save_results() -> bool:
    raw = os.environ.get("PERF_SAVE_RESULTS", "").strip().lower()
    return raw not in ("0", "false", "no")


def _external() -> bool:
    return os.environ.get("PERF_EXTERNAL", "").strip().lower() in ("1", "true", "yes")


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
    if "l1_on" not in metafunc.fixturenames:
        return
    case = load_case()
    params = []
    ids = []
    for l1_on in (True, False):
        for wave in case.waves:
            params.append((l1_on, wave.hit_rate, wave.concurrency, wave.requests))
            ids.append(wave_id(l1_on, wave.hit_rate, wave.concurrency, wave.requests))
    metafunc.parametrize("l1_on,hit_rate,concurrency,requests", params, ids=ids)


@pytest.fixture(scope="session")
def bench_mod():
    return load_bench_module()


@pytest.fixture(scope="session")
def l1_case():
    case = load_case()
    if not case.model:
        pytest.skip("MODEL is empty and the case has no Hugging Face model id")
    os.environ["MODEL"] = case.model
    return case


@pytest.fixture(scope="session")
def l1_output_dir(l1_case):  # noqa: ARG001
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


@pytest.fixture(scope="session")
def l1_stack(l1_case, l1_output_dir):  # noqa: ARG001
    if _external():
        url = os.environ.get("ROUTER_URL", "").strip()
        if not url:
            pytest.skip("PERF_EXTERNAL=1 requires ROUTER_URL")
        yield ExternalStack(url)
        return
    stack = L1Stack(l1_case, l1_case.model)
    try:
        stack.start_worker()
        yield stack
    finally:
        stack.stop()
