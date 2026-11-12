"""Fixtures for tests/python/e2e/perf/grpc (uses parent e2e conftest; pip vLLM 0.29+ ok)."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

_PERF_DIR = Path(__file__).resolve().parent
if str(_PERF_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_DIR))

from case_loader import (  # noqa: E402
    PERF_MODEL_ENV,
    apply_case_env,
    perf_model,
    perf_case_selected,
    perf_model_ready,
    resolve_perf_case,
    use_external_router,
    wave_case_id,
)
from stack_launcher import launch_stack  # noqa: E402
from tests.python.e2e.perf.artifact_paths import perf_results_dir  # noqa: E402

BENCH_SCRIPT = _PERF_DIR.parents[4] / "benches" / "backend" / "bench_prefix_kvhit.py"


def _progress(msg: str) -> None:
    print(f"[perf] {msg}", file=sys.stderr, flush=True)


def load_bench_module():
    spec = importlib.util.spec_from_file_location("bench_prefix_kvhit", BENCH_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pytest_addoption(parser):
    parser.addoption(
        "--perf-case",
        action="store",
        default=None,
        help=(
            "Perf scenario JSON: cases/<name>.json stem (rust_grpc), "
            "or path (tests/python/e2e/perf/grpc/cases/rust_grpc.json)"
        ),
    )


def _perf_case_error(exc: BaseException) -> pytest.UsageError:
    return pytest.UsageError(f"invalid --perf-case / PERF_CASE: {exc}")


def pytest_configure(config):
    opt = config.getoption("--perf-case", default=None)
    if opt:
        os.environ["PERF_CASE"] = str(opt).strip()
    if perf_case_selected():
        try:
            apply_case_env(resolve_perf_case())
        except (OSError, ValueError, RuntimeError, FileNotFoundError) as exc:
            raise _perf_case_error(exc) from exc


def _perf_ready() -> bool:
    return perf_model_ready() and perf_case_selected()


def _perf_case_for_collection():
    """Resolve the case at collection time (does not require MODEL).

    Parametrize before runtime env is complete so ``-k hit99-c16-r32`` works;
    missing MODEL still skips at test time via ``test_prefix_kvhit_waves`` skipif.
    """
    if not perf_case_selected():
        return None
    try:
        return resolve_perf_case()
    except (OSError, ValueError, RuntimeError, FileNotFoundError) as exc:
        raise _perf_case_error(exc) from exc


@pytest.fixture(scope="session")
def bench_mod():
    return load_bench_module()


@pytest.fixture(scope="session")
def perf_case():
    if not perf_case_selected():
        pytest.skip("Set --perf-case (or PERF_CASE)")
    case = resolve_perf_case()
    apply_case_env(case)
    if not perf_model():
        pytest.skip(f"Set {PERF_MODEL_ENV} (env) or model in case JSON")
    return case


@pytest.fixture(scope="session")
def perf_output_dir(perf_case, bench_mod) -> str:
    """Always save results in one directory per pytest session."""
    path = bench_mod.default_output_dir(
        {"backend_label": perf_case.backend},
        # Dashboard copies tests/python/e2e/results only.
        results_parent=perf_results_dir("grpc"),
    )
    _progress(f"results directory: {Path(path).resolve()}")
    return path


@pytest.fixture(scope="function")
def perf_stack(perf_case, request):
    """Cold start: launch worker + router for each wave (teardown after the test)."""
    if use_external_router():
        url = os.environ.get("ROUTER_URL", "").strip()
        if not url:
            pytest.skip("PERF_EXTERNAL=1 requires ROUTER_URL")
        yield None
        return

    model = perf_model()
    wave_id = getattr(request, "node", None)
    wave_label = wave_id.nodeid if wave_id is not None else "wave"
    _progress(
        f"perf_stack cold start [{wave_label}]: {perf_case.backend} model={model!r}"
    )
    stack = launch_stack(perf_case, model)
    _progress(f"perf_stack ready: {stack.router_url}")
    os.environ["ROUTER_URL"] = stack.router_url
    try:
        yield stack
    finally:
        _progress(f"perf_stack teardown [{wave_label}]")
        stack.stop()
        stop_wait = float(os.environ.get("PERF_STACK_STOP_SLEEP", "3"))
        if stop_wait > 0:
            time.sleep(stop_wait)


@pytest.fixture(scope="function")
def perf_settings(perf_case, perf_stack, bench_mod, perf_output_dir):
    settings = bench_mod.read_perf_settings(
        backend_label=perf_case.backend,
        bench_tokens=perf_case.bench_tokens,
        bench_max_tokens=perf_case.bench_max_tokens,
        bench_timeout=perf_case.bench_timeout,
    )
    settings["output_dir"] = perf_output_dir
    return settings


def _wave_params(case) -> tuple[list[tuple], list[str]]:
    values = []
    ids = []
    for hit_rate in case.hit_rates:
        for wave in case.waves:
            values.append((hit_rate, wave.concurrency, wave.requests))
            ids.append(wave_case_id(hit_rate, wave.concurrency, wave.requests))
    return values, ids


def pytest_generate_tests(metafunc):
    if "hit_rate" not in metafunc.fixturenames:
        return
    if metafunc.definition.name != "test_prefix_kvhit_waves":
        return
    if not perf_case_selected():
        metafunc.parametrize(
            "hit_rate,concurrency,requests",
            [
                pytest.param(
                    0.0,
                    1,
                    1,
                    marks=pytest.mark.skip(reason="Set --perf-case (or PERF_CASE)"),
                )
            ],
            ids=["no-perf-case"],
        )
        return

    case = _perf_case_for_collection()
    if case is None or not case.waves:
        return

    values, ids = _wave_params(case)
    if not values:
        return
    metafunc.parametrize(
        "hit_rate,concurrency,requests",
        values,
        ids=ids,
    )
