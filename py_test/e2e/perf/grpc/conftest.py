"""Fixtures for py_test/e2e/perf/grpc (uses parent e2e conftest; pip vLLM 0.29+ ok)."""

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
    WaveSpec,
    apply_case_env,
    perf_case_selected,
    perf_model_ready,
    resolve_perf_case,
    use_external_router,
    wave_case_id,
)
from stack_launcher import launch_stack  # noqa: E402

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
            "or path (py_test/e2e/perf/grpc/cases/rust_grpc.json)"
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


def _perf_save_results_enabled() -> bool:
    """Default on; set PERF_SAVE_RESULTS=0 to skip writing results/*.json."""
    raw = os.environ.get("PERF_SAVE_RESULTS", "").strip().lower()
    if raw in ("0", "false", "no"):
        return False
    if raw in ("1", "true", "yes"):
        return True
    return True


def _perf_waves_for_collection() -> tuple[WaveSpec, ...] | None:
    """Resolve wave matrix at collection time (does not require MODEL).

    Parametrize before runtime env is complete so ``-k hit99-c16-r32`` works;
    missing MODEL still skips at test time via ``test_prefix_kvhit_waves`` skipif.
    """
    if not perf_case_selected():
        return None
    try:
        return resolve_perf_case().waves
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
    if not os.environ.get("MODEL", "").strip():
        pytest.skip("Set MODEL (env) or model in case JSON")
    return case


@pytest.fixture(scope="session")
def perf_output_dir(perf_case, bench_mod) -> str:
    """One results directory per pytest session (default: always save)."""
    existing = os.environ.get("PERF_OUTPUT_DIR", "").strip()
    if existing:
        return existing
    if not _perf_save_results_enabled():
        return ""
    path = bench_mod.default_output_dir(
        {"backend_label": perf_case.backend},
        results_parent=_PERF_DIR / "results",
    )
    os.environ["PERF_OUTPUT_DIR"] = path
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

    model = os.environ["MODEL"].strip()
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
    if not settings["output_dir"] and perf_output_dir:
        settings["output_dir"] = perf_output_dir
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
    waves = _perf_waves_for_collection()
    if not waves:
        return

    values, ids = _wave_params(waves)
    if not values:
        return
    metafunc.parametrize(
        "hit_rate,concurrency,requests",
        values,
        ids=ids,
    )
