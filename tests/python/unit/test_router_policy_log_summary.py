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
        policy="kv_aware_aff",
        scenario="pd_disagg_kv_aware_affinity",
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
    assert stats["router_policy"] == stats["arm"] == "kv_aware_aff"
    assert stats["scenario"] == "pd_disagg_kv_aware_affinity"
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


@pytest.fixture
def calibration_run(tmp_path, monkeypatch):
    monkeypatch.setattr(case_loader, "PERF_RESULTS_DIR", tmp_path / "published")
    monkeypatch.setenv(case_loader.RUN_SUFFIX_ENV, "unit_test")
    (tmp_path / "result").mkdir()
    case = SimpleNamespace(
        policy="kv_aware_aff",
        scenario="pd_disagg_kv_aware_affinity",
        prefill_count=2,
        concurrency=16,
        task_num=8,
        worker_count=0,
    )
    return SimpleNamespace(case=case, out=tmp_path)


@pytest.mark.parametrize(
    "missing,expected",
    [
        ("ttft", ["ttft_s", "tpot_ms"]),
        ("tpot", ["tpot_ms"]),
        ("jct", ["jct_s"]),
        ("empty", ["ttft_s", "tpot_ms", "jct_s"]),
        ("nonfinite", ["ttft_s", "tpot_ms"]),
    ],
)
def test_invalid_measurements_fail_before_publication(
    summary_module, calibration_run, missing, expected
):
    run = calibration_run
    row = dict(
        status="success",
        ttft_seconds=0.1,
        latency_seconds=0.5,
        output_tokens=3,
        duration_seconds=1.0,
    )
    if missing == "ttft":
        del row["ttft_seconds"]
    elif missing == "tpot":
        row["output_tokens"] = 1
    elif missing == "jct":
        del row["duration_seconds"]
    elif missing == "nonfinite":
        row["ttft_seconds"] = float("nan")
    contents = "" if missing == "empty" else json.dumps(row) + "\n"
    result = run.out / "result"
    (result / "requests.jsonl").write_text(contents, encoding="utf-8")

    with pytest.raises(SystemExit, match="1"):
        summary_module.write_calibration(
            run, result_dir=result, replay_rc=0, wall_seconds=1, stalled=False
        )

    stats = json.loads((run.out / "calibration.json").read_text(encoding="utf-8"))
    assert stats["ok"] is False
    assert stats["missing_latency_metrics"] == expected
    assert not list((run.out / "published").rglob("*.json"))


@pytest.mark.parametrize("prefill_count", [0, 2])
@pytest.mark.parametrize("jct_count", [None, 0, 8])
def test_publication_schema_and_precomputed_jct(
    summary_module, calibration_run, prefill_count, jct_count
):
    run = calibration_run
    run.case.prefill_count = prefill_count
    prefix = "pd_disagg" if prefill_count else "pd_merge"
    run.case.scenario = f"{prefix}_kv_aware_affinity"
    result = run.out / "result"
    row = dict(status="success", ttft_seconds=0.1, latency_seconds=0.5, output_tokens=3)
    (result / "requests.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    duration = dict(mean=1.5, p50=1.4, p95=2.0, p99=2.1)
    if jct_count is not None:
        duration["n"] = jct_count
    (result / "summary.json").write_text(
        json.dumps({"duration_seconds": duration}), encoding="utf-8"
    )

    if jct_count == 0:
        with pytest.raises(SystemExit, match="1"):
            summary_module.write_calibration(
                run, result_dir=result, replay_rc=0, wall_seconds=1, stalled=False
            )
        assert not list((run.out / "published").rglob("*.json"))
        return

    summary_module.write_calibration(
        run, result_dir=result, replay_rc=0, wall_seconds=1, stalled=False
    )
    published = case_loader.router_policy_result_path(
        run.case.scenario, 16, 8, "unit_test"
    )
    stats = json.loads(published.read_text(encoding="utf-8"))
    assert stats["ok"] is True
    assert stats["scenario"] == run.case.scenario
    assert stats["router_policy"] == stats["arm"] == "kv_aware_aff"
    assert stats["topology"] == ("pd_disaggregated" if prefill_count else "pd_merged")
    assert "routing_policy" not in stats
    assert stats["jct_s"]["n"] == jct_count


@pytest.mark.parametrize("scenario", ["pd-disagg-cache-aware", "pd_merge_cache_aware"])
def test_loaded_scenario_keeps_its_name_across_concurrency(
    tmp_path, monkeypatch, scenario
):
    path = tmp_path / "case.json"
    path.write_text(
        json.dumps(
            {
                "id": scenario,
                "runtime": {"model": "test-model", "trace_path": "trace.jsonl"},
                "router_params": {"policy": "cache_aware"},
                "benchmark_params": {"max-concurrency": [8, 16]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(case_loader, "_vllm_rs_bin", lambda: "vllm-rs")
    monkeypatch.setattr(
        case_loader, "packaged_replay_agentx_yaml", lambda: tmp_path / "replay.yaml"
    )
    cases = [case_loader.load_router_policy_case(path, concurrency=c) for c in (8, 16)]
    assert [case.scenario for case in cases] == [scenario, scenario]
    assert [case.policy for case in cases] == ["cache_aware", "cache_aware"]
    assert cases[0].id != cases[1].id
