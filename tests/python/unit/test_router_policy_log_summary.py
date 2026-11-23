"""CPU-only regression checks for per-metric performance sample counts."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.python.e2e.perf.router_policy import case_loader

pytestmark = [pytest.mark.ready, pytest.mark.cpu]


@pytest.fixture
def summary_module(monkeypatch):
    path = Path(case_loader.__file__).with_name("log_summary.py")
    spec = importlib.util.spec_from_file_location("router_policy_log_summary_ut", path)
    module = importlib.util.module_from_spec(spec)
    # The suite uses a local absolute import; avoid collisions with other suites.
    with monkeypatch.context() as context:
        context.setitem(sys.modules, "case_loader", case_loader)
        spec.loader.exec_module(module)
    return module


def test_calibration_counts_filtered_samples(summary_module, tmp_path, monkeypatch):
    rows = [
        dict(
            status="success",
            ttft_seconds=0.1,
            latency_seconds=0.5,
            output_tokens=3,
            duration_seconds=1.0,
        ),
        dict(
            status="success",
            ttft_seconds=0.2,
            latency_seconds=0.8,
            output_tokens=1,
            duration_seconds=2.0,
        ),
        dict(status="success", latency_seconds=0.9, output_tokens=3),
    ]
    result = tmp_path / "result"
    result.mkdir()
    (result / "requests.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    case = SimpleNamespace(
        policy="kv_aware",
        result_series="pd_disagg_kv_aware_affinity",
        prefill_count=2,
        concurrency=8,
        task_num=3,
        worker_count=0,
    )
    run = SimpleNamespace(case=case, out=tmp_path)
    # No workers/dashboard publication: verify the actual serialized calibration.
    monkeypatch.setattr(
        summary_module, "emit_run_summary", lambda *args, **kwargs: None
    )
    summary_module.write_calibration(
        run, result_dir=result, replay_rc=0, wall_seconds=3, stalled=False
    )
    stats = json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))
    assert stats["router_policy"] == "pd_disagg_kv_aware_affinity"
    assert stats["topology"] == "pd_disaggregated"
    assert stats["success"] == 3
    assert stats["ttft_s"]["n"] == 2
    assert stats["tpot_ms"]["n"] == 1
    assert stats["jct_s"]["n"] == 2
    assert stats["ttft_s"]["mean"] == pytest.approx(0.15)
    assert stats["tpot_ms"]["mean"] == pytest.approx(200)
    assert stats["jct_s"]["mean"] == pytest.approx(1.5)


def test_empty_samples(summary_module, tmp_path):
    expected = {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    assert summary_module._quantiles([]) == expected
    assert summary_module._jct_quantiles(tmp_path, []) == expected


@pytest.mark.parametrize(
    "metadata,expected", [({"n": 7}, 7), ({"count": 5}, 5), ({}, None)]
)
def test_precomputed_jct_count(summary_module, tmp_path, metadata, expected):
    duration = {"mean": 1.5, "p50": 1.4, "p95": 2.0, "p99": 2.1, **metadata}
    (tmp_path / "summary.json").write_text(
        json.dumps({"duration_seconds": duration}), encoding="utf-8"
    )
    assert summary_module._jct_quantiles(tmp_path, []) == {
        "n": expected,
        "mean": 1.5,
        "p50": 1.4,
        "p95": 2.0,
        "p99": 2.1,
    }
