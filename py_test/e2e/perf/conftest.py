"""Fixtures for py_test/e2e/perf (uses parent e2e conftest / vLLM test env)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_PERF_DIR = Path(__file__).resolve().parent
if str(_PERF_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_DIR))

from case_loader import (  # noqa: E402
    WaveSpec,
    perf_case_selected,
    resolve_perf_case,
    use_external_router,
    wave_case_id,
)
from prefix_kvhit_runner import (  # noqa: E402
    default_output_dir,
    load_bench_module,
    perf_env_ready,
    read_perf_settings,
)
from stack_launcher import launch_stack  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--perf-case",
        action="store",
        default=None,
        help=(
            "Perf scenario JSON: cases/<name>.json stem (rust_grpc), "
            "or path (py_test/e2e/perf/cases/rust_grpc.json)"
        ),
    )


def pytest_configure(config):
    opt = config.getoption("--perf-case", default=None)
    if opt:
        os.environ["PERF_CASE"] = str(opt).strip()


def _perf_ready() -> bool:
    return perf_env_ready() and perf_case_selected()


@pytest.fixture(scope="session")
def bench_mod():
    return load_bench_module()


@pytest.fixture(scope="session")
def perf_case():
    if not perf_case_selected():
        pytest.skip("Set --perf-case (or PERF_CASE)")
    if not os.environ.get("MODEL", "").strip():
        pytest.skip("Set MODEL to the served model path or HF id")
    return resolve_perf_case()


@pytest.fixture(scope="session")
def perf_stack(perf_case):
    """Launch worker + router from case JSON, or use PERF_EXTERNAL + ROUTER_URL."""
    if use_external_router():
        url = os.environ.get("ROUTER_URL", "").strip()
        if not url:
            pytest.skip("PERF_EXTERNAL=1 requires ROUTER_URL")
        yield None
        return

    model = os.environ["MODEL"].strip()
    stack = launch_stack(perf_case, model)
    os.environ["ROUTER_URL"] = stack.router_url
    try:
        yield stack
    finally:
        stack.stop()


@pytest.fixture(scope="session")
def perf_settings(perf_case, perf_stack):
    settings = read_perf_settings(
        backend_label=perf_case.backend,
        bench_tokens=perf_case.bench_tokens,
        bench_max_tokens=perf_case.bench_max_tokens,
        bench_timeout=perf_case.bench_timeout,
    )
    if not settings["output_dir"] and os.environ.get(
        "PERF_SAVE_RESULTS", ""
    ).lower() in (
        "1",
        "true",
        "yes",
    ):
        settings["output_dir"] = default_output_dir(settings)
    return settings


def _wave_params(waves: tuple[WaveSpec, ...]) -> tuple[list[tuple], list[str]]:
    values = [(w.hit_rate, w.concurrency, w.requests) for w in waves]
    ids = [wave_case_id(w.hit_rate, w.concurrency, w.requests) for w in waves]
    return values, ids


def pytest_generate_tests(metafunc):
    if "run_idx" in metafunc.fixturenames:
        runs = max(1, int(os.environ.get("RUNS", "1")))
        metafunc.parametrize("run_idx", range(1, runs + 1))

    if "hit_rate" not in metafunc.fixturenames:
        return
    if metafunc.definition.name != "test_prefix_kvhit_waves":
        return
    if not _perf_ready():
        return

    try:
        case = resolve_perf_case()
    except (OSError, ValueError, RuntimeError, FileNotFoundError):
        return

    values, ids = _wave_params(case.waves)
    if not values:
        return
    metafunc.parametrize(
        "hit_rate,concurrency,requests",
        values,
        ids=ids,
    )
