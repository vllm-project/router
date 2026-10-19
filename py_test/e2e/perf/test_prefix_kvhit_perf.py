"""Live prefix-KV perf: launch stack from case JSON, then run bench waves.

  export MODEL=/path/to/model
  pytest py_test/e2e/perf --perf-case=rust_grpc -m perf
"""

from __future__ import annotations

import os

import pytest

from case_loader import perf_case_selected, use_external_router
from prefix_kvhit_runner import (
    assert_wave_summary,
    maybe_write_result,
    perf_env_ready,
    run_prefix_kvhit_wave,
)

pytestmark = pytest.mark.e2e


def _case_ready() -> bool:
    if not perf_env_ready() or not perf_case_selected():
        return False
    if use_external_router():
        return bool(os.environ.get("ROUTER_URL", "").strip())
    return True


@pytest.mark.perf
@pytest.mark.skipif(
    not _case_ready(), reason="Set MODEL, --perf-case; or PERF_EXTERNAL+ROUTER_URL"
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
    payload = run_prefix_kvhit_wave(
        bench_mod,
        perf_settings,
        hit_rate=hit_rate,
        concurrency=concurrency,
        requests=requests,
    )
    assert_wave_summary(payload["wave"], requests)
    maybe_write_result(
        perf_settings,
        run_idx=run_idx,
        hit_rate=hit_rate,
        concurrency=concurrency,
        requests=requests,
        payload={**payload, "backend": perf_case.backend},
    )
