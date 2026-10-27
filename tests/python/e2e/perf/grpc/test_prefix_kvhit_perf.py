"""Live prefix-KV perf: launch stack from case JSON, then run bench waves.

export MODEL=/path/to/model
pytest tests/python/e2e/perf/grpc --perf-case=rust_grpc
"""

from __future__ import annotations

import os
import sys

import pytest

from case_loader import perf_case_selected, perf_model_ready, use_external_router

pytestmark = pytest.mark.e2e


def _case_ready() -> bool:
    if not perf_model_ready() or not perf_case_selected():
        return False
    if use_external_router():
        return bool(os.environ.get("ROUTER_URL", "").strip())
    return True


@pytest.mark.skipif(
    not _case_ready(),
    reason="Set --perf-case and MODEL (env or case JSON); or PERF_EXTERNAL+ROUTER_URL",
)
def test_prefix_kvhit_waves(
    bench_mod,
    perf_stack,
    perf_settings,
    perf_case,
    run_idx: int,
    hit_rate: float,
    concurrency: int,
    requests: int,
) -> None:
    payload = bench_mod.run_prefix_kvhit_wave(
        perf_settings,
        hit_rate=hit_rate,
        concurrency=concurrency,
        requests=requests,
    )
    bench_mod.assert_wave_summary(payload["wave"], requests)
    bench_mod.print_wave_summary(
        payload["wave"],
        backend=perf_case.backend,
        hit_rate=hit_rate,
        concurrency=concurrency,
        requests=requests,
        run_idx=run_idx,
    )
    result_path = bench_mod.maybe_write_result(
        perf_settings,
        run_idx=run_idx,
        hit_rate=hit_rate,
        concurrency=concurrency,
        requests=requests,
        payload={**payload, "backend": perf_case.backend},
    )
    if result_path is not None:
        print(
            f"[perf] wrote result: {result_path.resolve()}",
            file=sys.stderr,
            flush=True,
        )
