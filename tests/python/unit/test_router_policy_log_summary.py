"""CPU checks for the public calibration and publication APIs."""

import json
import sys
from types import SimpleNamespace

import pytest

from tests.python.e2e.perf.router_policy import case_loader, log_summary

pytestmark = [pytest.mark.ready, pytest.mark.cpu]


def accepted_replay(result_dir, rows, *, task_num):
    """Stub an accepted validation result to isolate summary behavior."""
    return {"errors": []}


def test_calibration_counts_filtered_samples(tmp_path, monkeypatch):
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
    for i, row in enumerate(rows):
        row.update(request_id=f"request-{i}-0", session_id=f"session-{i}")
    (result / "requests.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    case = SimpleNamespace(
        policy="kv_aware",
        scenario="pd_disagg_kv_aware",
        prefill_count=2,
        concurrency=8,
        task_num=3,
        worker_count=0,
    )
    run = SimpleNamespace(case=case, out=tmp_path)
    # No workers/dashboard publication: verify the actual serialized calibration.
    monkeypatch.setattr(log_summary, "emit_run_summary", lambda *args, **kwargs: None)
    log_summary.write_calibration(
        run,
        replay_validator=accepted_replay,
        result_dir=result,
        replay_rc=0,
        wall_seconds=3,
        stalled=False,
    )
    stats = json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))
    assert stats["router_policy"] == stats["arm"] == "kv_aware"
    assert stats["scenario"] == "pd_disagg_kv_aware"
    assert stats["topology"] == "pd_disaggregated"
    assert stats["success"] == 3
    assert stats["ttft_s"]["n"] == 2
    assert stats["tpot_ms"]["n"] == 1
    assert stats["jct_s"]["n"] == 2
    assert stats["ttft_s"]["mean"] == pytest.approx(0.15)
    assert stats["tpot_ms"]["mean"] == pytest.approx(200)
    assert stats["jct_s"]["mean"] == pytest.approx(1.5)
    assert stats["latency_coverage"] == {
        "ttft_s": {"expected": 3, "actual": 2},
        "tpot_ms": {"expected": 2, "actual": 1},
    }
    assert stats["incomplete_latency_metrics"] == ["ttft_s", "tpot_ms"]


def test_empty_samples(calibration_run):
    run = calibration_run
    with pytest.raises(SystemExit, match="1"):
        log_summary.write_calibration(
            run,
            replay_validator=accepted_replay,
            result_dir=run.out / "result",
            replay_rc=0,
            wall_seconds=1,
            stalled=False,
        )
    stats = json.loads((run.out / "calibration.json").read_text(encoding="utf-8"))
    expected = {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    assert stats["ttft_s"] == {**expected, "p90": None, "max": None}
    assert stats["tpot_ms"] == expected
    assert stats["jct_s"] == expected


@pytest.mark.parametrize(
    "metadata,expected", [({"n": 7}, 7), ({"count": 5}, 5), ({}, None)]
)
def test_precomputed_jct_count(prefill_run, metadata, expected):
    run = prefill_run
    result = run.out / "result"
    row = json.loads((result / "requests.jsonl").read_text(encoding="utf-8"))
    del row["duration_seconds"]
    (result / "requests.jsonl").write_text(json.dumps(row), encoding="utf-8")
    duration = {"mean": 1.5, "p50": 1.4, "p95": 2.0, "p99": 2.1, **metadata}
    (result / "summary.json").write_text(
        json.dumps({"duration_seconds": duration}), encoding="utf-8"
    )
    assert _calibrate_prefill(run)["jct_s"] == {
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
    (tmp_path / "after").mkdir()
    (tmp_path / "after/router.prom").write_text(
        'vllm_router_kv_placement_lookups_total{result="miss"} 1\n'
        'vllm_router_prefix_routing_decisions_total{policy="kv_aware",stage="regular",outcome="sticky_hit"} 1\n',
        encoding="utf-8",
    )
    case = SimpleNamespace(
        policy="kv_aware",
        scenario="pd_disagg_kv_aware",
        prefill_count=2,
        concurrency=16,
        task_num=1,
        worker_count=0,
    )
    return SimpleNamespace(case=case, out=tmp_path)


@pytest.mark.parametrize("prefill_count", [0, 2], ids=["pd_merge", "pd_disagg"])
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
    calibration_run, prefill_count, missing, expected, capsys
):
    run = calibration_run
    run.case.prefill_count = prefill_count
    prefix = "pd_disagg" if prefill_count else "pd_merge"
    run.case.scenario = f"{prefix}_kv_aware"
    row = dict(
        request_id="request-0-0",
        session_id="session-0",
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
    if missing == "jct":
        (result / "summary.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="1"):
        log_summary.write_calibration(
            run,
            replay_validator=accepted_replay,
            result_dir=result,
            replay_rc=0,
            wall_seconds=1,
            stalled=False,
        )

    stats = json.loads((run.out / "calibration.json").read_text(encoding="utf-8"))
    assert stats["ok"] is False
    assert stats["requests"] == stats["success"] == (0 if missing == "empty" else 1)
    assert stats["scenario"] == run.case.scenario
    assert stats["topology"] == ("pd_disaggregated" if prefill_count else "pd_merged")
    assert stats["missing_latency_metrics"] == expected
    assert (
        "DIAGNOSTIC missing latency measurements: " + ", ".join(expected)
        in capsys.readouterr().out
    )
    assert not list((run.out / "published").rglob("*.json"))


@pytest.mark.parametrize("replay_rc", [0, 42])
def test_failure_summary_prints_all_applicable_diagnostics(
    calibration_run, replay_rc, capsys
):
    run = calibration_run
    stats = dict(
        ok=False,
        success=0,
        requests=28,
        missing_latency_metrics=["ttft_s", "tpot_ms", "jct_s"],
        incomplete_latency_metrics=["ttft_s"],
        latency_coverage={"ttft_s": {"actual": 0, "expected": 28}},
        replay_validation={"errors": ["request count mismatch: expected 28, got 0"]},
        router_policy_checks={"errors": ["zero KV placement lookups"]},
        snapshot_errors=["metrics snapshot failed for router: 503"],
    )
    (run.out / "calibration.json").write_text(json.dumps(stats), encoding="utf-8")
    with pytest.raises(SystemExit, match="1"):
        log_summary.emit_run_summary(run, stats, replay_rc=replay_rc, stalled=True)
    diagnostics = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("DIAGNOSTIC ")
    ]
    for reason in (
        "missing latency measurements: ttft_s, tpot_ms, jct_s",
        "incomplete latency measurements: ttft_s=0/28",
        "replay completeness: request count mismatch: expected 28, got 0",
        "zero KV placement lookups",
        "metrics snapshot failed for router: 503",
        "stalled or non-success rows in requests.jsonl",
    ):
        assert f"DIAGNOSTIC {reason}" in diagnostics
    if replay_rc:
        assert "DIAGNOSTIC replay failed (exit 42)" in diagnostics
    assert not list((run.out / "published").rglob("*.json"))


@pytest.mark.parametrize(
    "prefill_count,task_num,request_count",
    [(0, 8, 28), (2, 12, 144)],
    ids=["pd_merge", "pd_disagg"],
)
@pytest.mark.parametrize("missing_field", ["ttft_seconds", "latency_seconds"])
def test_incomplete_timing_coverage_blocks_publication(
    calibration_run,
    prefill_count,
    task_num,
    request_count,
    missing_field,
    capsys,
):
    run = calibration_run
    run.case.prefill_count = prefill_count
    run.case.task_num = run.case.concurrency = task_num
    run.case.policy = "cache_aware"
    prefix = "pd_disagg" if prefill_count else "pd_merge"
    run.case.scenario = f"{prefix}_cache_aware"
    rows = []
    for index in range(request_count):
        row = dict(
            request_id=f"request-{index}",
            session_id=f"session-{index % task_num}",
            status="success",
            ttft_seconds=0.1,
            latency_seconds=0.5,
            output_tokens=3,
            duration_seconds=1.0,
        )
        if index:
            row[missing_field] = None
        rows.append(row)
    (run.out / "result/requests.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(SystemExit, match="1"):
        log_summary.write_calibration(
            run,
            replay_validator=accepted_replay,
            result_dir=run.out / "result",
            replay_rc=0,
            wall_seconds=1,
            stalled=False,
        )

    stats = json.loads((run.out / "calibration.json").read_text(encoding="utf-8"))
    assert stats["ok"] is False
    assert stats["requests"] == stats["success"] == request_count
    expected_ttft = 1 if missing_field == "ttft_seconds" else request_count
    assert stats["latency_coverage"] == {
        "ttft_s": {"expected": request_count, "actual": expected_ttft},
        "tpot_ms": {"expected": request_count, "actual": 1},
    }
    assert stats["ttft_s"]["n"] == expected_ttft
    assert stats["tpot_ms"]["n"] == 1
    assert "tpot_ms" in stats["incomplete_latency_metrics"]
    assert "missing_latency_metrics" not in stats
    assert f"tpot_ms=1/{request_count}" in capsys.readouterr().out
    assert not list((run.out / "published").rglob("*.json"))


@pytest.mark.parametrize("prefill_count", [0, 2], ids=["pd_merge", "pd_disagg"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("ttft_seconds", float("nan")),
        ("latency_seconds", float("inf")),
        ("ttft_seconds", -0.1),
        ("latency_seconds", 0.01),
        ("ttft_seconds", True),
        ("latency_seconds", "invalid"),
    ],
    ids=["nan", "infinity", "negative", "latency_before_ttft", "boolean", "text"],
)
def test_invalid_timing_samples_cannot_satisfy_coverage(
    calibration_run, prefill_count, field, value
):
    run = calibration_run
    run.case.prefill_count = prefill_count
    run.case.policy = "cache_aware"
    prefix = "pd_disagg" if prefill_count else "pd_merge"
    run.case.scenario = f"{prefix}_cache_aware"
    rows = [
        dict(
            request_id=f"request-{index}",
            session_id="session-0",
            status="success",
            ttft_seconds=0.1,
            latency_seconds=0.5,
            output_tokens=3,
            duration_seconds=1.0,
        )
        for index in range(3)
    ]
    rows[-1][field] = value
    (run.out / "result/requests.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(SystemExit, match="1"):
        log_summary.write_calibration(
            run,
            replay_validator=accepted_replay,
            result_dir=run.out / "result",
            replay_rc=0,
            wall_seconds=1,
            stalled=False,
        )

    stats = json.loads((run.out / "calibration.json").read_text(encoding="utf-8"))
    assert stats["requests"] == stats["success"] == 3
    assert stats["latency_coverage"]["tpot_ms"] == {"expected": 3, "actual": 2}
    assert stats["tpot_ms"]["n"] == 2
    assert not list((run.out / "published").rglob("*.json"))


@pytest.mark.parametrize("prefill_count", [0, 2], ids=["pd_merge", "pd_disagg"])
@pytest.mark.parametrize("include_failed", [False, True])
def test_coverage_counts_eligible_successes_and_publishes_complete_metrics(
    calibration_run, prefill_count, include_failed
):
    run = calibration_run
    run.case.prefill_count = prefill_count
    run.case.policy = "cache_aware"
    prefix = "pd_disagg" if prefill_count else "pd_merge"
    run.case.scenario = f"{prefix}_cache_aware"
    rows = [
        dict(
            request_id=f"request-{index}",
            session_id="session-0",
            status="success",
            ttft_seconds=0.0,
            latency_seconds=None if output <= 1 else 0.0,
            output_tokens=output,
            duration_seconds=1.0,
        )
        for index, output in enumerate((0, 1, 3, 5))
    ]
    if include_failed:
        rows.append(dict(status="error", output_tokens=3))
    (run.out / "result/requests.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    def calibrate():
        log_summary.write_calibration(
            run,
            replay_validator=accepted_replay,
            result_dir=run.out / "result",
            replay_rc=0,
            wall_seconds=1,
            stalled=False,
        )

    if include_failed:
        with pytest.raises(SystemExit, match="1"):
            calibrate()
    else:
        calibrate()

    stats = json.loads((run.out / "calibration.json").read_text(encoding="utf-8"))
    assert stats["success"] == 4
    assert stats["ok"] is (not include_failed)
    assert stats["latency_coverage"] == {
        "ttft_s": {"expected": 4, "actual": 4},
        "tpot_ms": {"expected": 2, "actual": 2},
    }
    assert "incomplete_latency_metrics" not in stats
    assert stats["ttft_s"]["n"] == 4
    assert stats["tpot_ms"]["n"] == 2
    published = list((run.out / "published").rglob("*.json"))
    assert len(published) == (0 if include_failed else 1)
    if published:
        assert published[0].read_bytes() == (run.out / "calibration.json").read_bytes()


@pytest.mark.parametrize("topology", ["pd_merged", "pd_disaggregated"])
@pytest.mark.parametrize(
    "scenario_metadata",
    [{}, {"scenario": None}, {"scenario": ""}, {"scenario": " \t"}, {"scenario": 1}],
    ids=["missing", "null", "empty", "whitespace", "non_string"],
)
def test_publication_requires_scenario(calibration_run, topology, scenario_metadata):
    run = calibration_run
    stats = dict(
        ok=True,
        router_policy="cache_aware",
        topology=topology,
        concurrency=16,
        task_num=8,
        **scenario_metadata,
    )
    (run.out / "calibration.json").write_text(json.dumps(stats), encoding="utf-8")

    log_summary.emit_run_summary(run, stats, replay_rc=0, stalled=False)
    assert not list((run.out / "published").rglob("*.json"))
    assert (
        json.loads((run.out / "calibration.json").read_text(encoding="utf-8")) == stats
    )


def test_same_policy_topologies_publish_separate_results(calibration_run):
    run = calibration_run
    published = []
    for prefix, topology in [
        ("pd_merge", "pd_merged"),
        ("pd_disagg", "pd_disaggregated"),
    ]:
        stats = dict(
            ok=True,
            scenario=f"{prefix}_cache_aware",
            router_policy="cache_aware",
            topology=topology,
            concurrency=16,
            task_num=8,
        )
        (run.out / "calibration.json").write_text(json.dumps(stats), encoding="utf-8")
        log_summary.emit_run_summary(run, stats, replay_rc=0, stalled=False)
        expected = case_loader.router_policy_result_path(
            stats["scenario"], 16, 8, "unit_test"
        )
        assert expected.is_file()
        published.append((expected, stats))

    assert published[0][0] != published[1][0]
    for path, stats in published:
        assert json.loads(path.read_text(encoding="utf-8")) == stats


@pytest.mark.parametrize("prefill_count", [0, 2])
@pytest.mark.parametrize("jct_count", [None, 0, 8])
def test_publication_schema_and_precomputed_jct(
    calibration_run, prefill_count, jct_count
):
    run = calibration_run
    run.case.prefill_count = prefill_count
    prefix = "pd_disagg" if prefill_count else "pd_merge"
    run.case.scenario = f"{prefix}_kv_aware"
    result = run.out / "result"
    row = dict(
        request_id="request-0-0",
        session_id="session-0",
        status="success",
        ttft_seconds=0.1,
        latency_seconds=0.5,
        output_tokens=3,
    )
    (result / "requests.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    duration = dict(mean=1.5, p50=1.4, p95=2.0, p99=2.1)
    if jct_count is not None:
        duration["n"] = jct_count
    (result / "summary.json").write_text(
        json.dumps({"duration_seconds": duration}), encoding="utf-8"
    )

    if jct_count == 0:
        with pytest.raises(SystemExit, match="1"):
            log_summary.write_calibration(
                run,
                replay_validator=accepted_replay,
                result_dir=result,
                replay_rc=0,
                wall_seconds=1,
                stalled=False,
            )
        assert not list((run.out / "published").rglob("*.json"))
        return

    log_summary.write_calibration(
        run,
        replay_validator=accepted_replay,
        result_dir=result,
        replay_rc=0,
        wall_seconds=1,
        stalled=False,
    )
    published = case_loader.router_policy_result_path(
        run.case.scenario, 16, run.case.task_num, "unit_test"
    )
    stats = json.loads(published.read_text(encoding="utf-8"))
    assert stats["ok"] is True
    assert stats["scenario"] == run.case.scenario
    assert stats["router_policy"] == stats["arm"] == "kv_aware"
    assert stats["topology"] == ("pd_disaggregated" if prefill_count else "pd_merged")
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
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "vllm-rs").touch()
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(__file__=str(package / "__init__.py"))
    )
    monkeypatch.setattr(
        case_loader,
        "packaged_replay_agentx_yaml",
        lambda trace_type: tmp_path / "replay.yaml",
    )
    cases = [case_loader.load_router_policy_case(path, concurrency=c) for c in (8, 16)]
    assert [case.scenario for case in cases] == [scenario, scenario]
    assert [case.policy for case in cases] == ["cache_aware", "cache_aware"]
    assert cases[0].id != cases[1].id


def _write_prefill_snapshot(run, phase, worker, *, hits, queries, computed, extra=""):
    destination = run.out / phase / f"engine{worker}.prom"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(
        'vllm:request_success_total{engine="0"} 1\n'
        f'vllm:prefix_cache_hits_total{{engine="0"}} {hits}\n'
        f'vllm:prefix_cache_queries_total{{engine="0"}} {queries}\n'
        'vllm:prompt_tokens_by_source_total{engine="0",source="local_compute"} '
        f"{computed}\n" + extra,
        encoding="utf-8",
    )


@pytest.fixture
def prefill_run(calibration_run):
    run = calibration_run
    run.case.worker_count = 4
    row = dict(
        request_id="request-0-0",
        session_id="session-0",
        status="success",
        ttft_seconds=0.1,
        latency_seconds=0.5,
        output_tokens=3,
        duration_seconds=1.0,
    )
    (run.out / "result/requests.jsonl").write_text(json.dumps(row), encoding="utf-8")
    for worker in range(4):
        _write_prefill_snapshot(
            run, "before", worker, hits=20, queries=100, computed=80
        )
        _write_prefill_snapshot(
            run, "after", worker, hits=40, queries=200, computed=160
        )
    return run


def _calibrate_prefill(run):
    log_summary.write_calibration(
        run,
        replay_validator=accepted_replay,
        result_dir=run.out / "result",
        replay_rc=0,
        wall_seconds=1,
        stalled=False,
    )
    return json.loads((run.out / "calibration.json").read_text(encoding="utf-8"))


def test_supplied_replay_validator_can_prevent_publication(prefill_run):
    run = prefill_run
    calls = []

    def reject_replay(result_dir, rows, *, task_num):
        calls.append((result_dir, rows, task_num))
        return {"errors": ["injected replay validation failure"]}

    with pytest.raises(SystemExit, match="1"):
        log_summary.write_calibration(
            run,
            replay_validator=reject_replay,
            result_dir=run.out / "result",
            replay_rc=0,
            wall_seconds=1,
            stalled=False,
        )
    assert len(calls) == 1
    result_dir, rows, task_num = calls[0]
    assert result_dir == run.out / "result"
    assert len(rows) == 1
    assert task_num == run.case.task_num
    stats = json.loads((run.out / "calibration.json").read_text(encoding="utf-8"))
    assert stats["ok"] is False
    assert stats["replay_validation"]["errors"] == [
        "injected replay validation failure"
    ]
    assert not list((run.out / "published").rglob("*.json"))


@pytest.mark.parametrize(
    "prefill_count,policy",
    [(0, "kv_aware"), (0, "kv_aware_aff"), (2, "kv_aware")],
    ids=["merged_kv", "merged_affinity", "pd_kv"],
)
def test_unavailable_kv_index_fails_before_publication(
    prefill_run, prefill_count, policy, capsys
):
    run = prefill_run
    run.case.prefill_count = prefill_count
    run.case.policy = policy
    prefix = "pd_disagg" if prefill_count else "pd_merge"
    suffix = "kv_aware_affinity" if policy == "kv_aware_aff" else policy
    run.case.scenario = f"{prefix}_{suffix}"
    snapshot = run.out / "after/router.prom"
    snapshot.write_text(
        'vllm_router_kv_placement_lookups_total{result="index_unavailable"} 100\n'
        'vllm_router_kv_routing_inputs_total{transport="grpc",result="prepared"} 64\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="1"):
        _calibrate_prefill(run)

    stats = json.loads((run.out / "calibration.json").read_text(encoding="utf-8"))
    assert stats["requests"] == stats["success"] == 1
    assert stats["ok"] is False
    assert "zero KV placement lookups" in capsys.readouterr().out
    assert not list((run.out / "published").rglob("*.json"))


@pytest.mark.parametrize(
    "prefill_count,policy",
    [(0, "kv_aware"), (0, "kv_aware_aff"), (2, "kv_aware")],
    ids=["merged_kv", "merged_affinity", "pd_kv"],
)
@pytest.mark.parametrize("result", ["hit", "miss"])
def test_verified_kv_lookup_allows_publication(
    prefill_run, prefill_count, policy, result
):
    run = prefill_run
    run.case.prefill_count = prefill_count
    run.case.policy = policy
    prefix = "pd_disagg" if prefill_count else "pd_merge"
    suffix = "kv_aware_affinity" if policy == "kv_aware_aff" else policy
    run.case.scenario = f"{prefix}_{suffix}"
    snapshot = run.out / "after/router.prom"
    # Labels can be reordered and additional labels must not hide valid activity.
    with snapshot.open("a", encoding="utf-8") as stream:
        stream.write(
            f'vllm_router_kv_placement_lookups_total{{worker="0",result="{result}"}} 2\n'
            f'vllm_router_kv_placement_lookups_total{{result="{result}",worker="1"}} 3\n'
        )
    stats = _calibrate_prefill(run)

    assert stats["ok"] is True
    checks = stats["router_policy_checks"]
    assert checks["errors"] == []
    assert checks["placement_lookups"][result] == (6 if result == "miss" else 5)
    if policy == "kv_aware":
        assert "affinity" not in checks
    else:
        assert checks["affinity"]["stage"] == "regular"
    published = case_loader.router_policy_result_path(
        run.case.scenario, 16, run.case.task_num, "unit_test"
    )
    assert published.read_bytes() == (run.out / "calibration.json").read_bytes()


@pytest.mark.parametrize("prefill_count", [0, 2], ids=["pd_merge", "pd_disagg"])
@pytest.mark.parametrize("sample", [None, "", "0", "NaN", "+Inf", "-1", "invalid"])
def test_missing_or_invalid_lookup_counters_fail(prefill_run, prefill_count, sample):
    run = prefill_run
    run.case.prefill_count = prefill_count
    run.case.policy = "kv_aware"
    snapshot = run.out / "after/router.prom"
    if sample is None:
        snapshot.unlink()
    elif sample == "":
        snapshot.write_text("# no lookup samples\n", encoding="utf-8")
    else:
        snapshot.write_text(
            f'vllm_router_kv_placement_lookups_total{{result="hit"}} {sample}\n',
            encoding="utf-8",
        )
    with pytest.raises(SystemExit, match="1"):
        _calibrate_prefill(run)
    stats = json.loads((run.out / "calibration.json").read_text(encoding="utf-8"))
    assert len(stats["router_policy_checks"]["errors"]) == 1
    assert "sha256_cbor" in stats["router_policy_checks"]["errors"][0]
    assert not list((run.out / "published").rglob("*.json"))


@pytest.mark.parametrize(
    "metric,labels,passes",
    [
        (
            "prefix_routing",
            'policy="kv_aware",stage="regular",outcome="sticky_hit"',
            True,
        ),
        (
            "prefix_routing",
            'policy="cache_aware",stage="regular",outcome="sticky_hit"',
            False,
        ),
        (
            "prefix_routing",
            'policy="kv_aware",stage="prefill",outcome="sticky_hit"',
            False,
        ),
        (
            "prefix_routing",
            'policy="kv_aware",stage="regular",outcome="prefix_cost"',
            False,
        ),
        ("pd_affinity", 'stage="decode",outcome="bind"', False),
    ],
)
def test_affinity_requires_the_matching_stage_and_outcome(
    prefill_run, metric, labels, passes, capsys
):
    run = prefill_run
    run.case.prefill_count = 0
    run.case.policy = "kv_aware_aff"
    run.case.scenario = "pd_merge_kv_aware_affinity"
    (run.out / "after/router.prom").write_text(
        'vllm_router_kv_placement_lookups_total{result="miss"} 1\n'
        f"vllm_router_{metric}_decisions_total{{{labels}}} 1\n",
        encoding="utf-8",
    )
    if passes:
        assert _calibrate_prefill(run)["ok"] is True
    else:
        with pytest.raises(SystemExit, match="1"):
            _calibrate_prefill(run)
        stats = json.loads((run.out / "calibration.json").read_text(encoding="utf-8"))
        assert len(stats["router_policy_checks"]["errors"]) == 1
        assert "affinity decisions" in capsys.readouterr().out
        assert not list((run.out / "published").rglob("*.json"))


@pytest.mark.parametrize("prefill_count", [0, 2], ids=["pd_merge", "pd_disagg"])
def test_cache_aware_skips_index_and_affinity_checks(prefill_run, prefill_count):
    run = prefill_run
    run.case.policy = "cache_aware"
    run.case.prefill_count = prefill_count
    (run.out / "after/router.prom").unlink()
    stats = _calibrate_prefill(run)
    assert stats["ok"] is True
    assert stats["router_policy_checks"] == {"errors": []}


def test_prefill_deltas_exclude_decode_and_copy_to_results(prefill_run):
    run = prefill_run
    ignored = (
        'vllm:prompt_tokens_by_source_total{source="local_cache_hit"} 900000\n'
        'vllm:prompt_tokens_by_source_total{source="external_kv_transfer"} 900000\n'
        'vllm:prompt_tokens_by_source_total{source="local_compute_extra"} 900000\n'
        r'vllm:prompt_tokens_by_source_total{model_name="source=\"local_compute\"",source="local_cache_hit"} 900000'
        "\n"
    )
    _write_prefill_snapshot(
        run,
        "before",
        0,
        hits=20,
        queries=100,
        computed=20,
        extra='vllm:prompt_tokens_by_source_total{source="local_compute",engine="1"} 60\n'
        + ignored,
    )
    _write_prefill_snapshot(
        run,
        "after",
        0,
        hits=60,
        queries=200,
        computed=50,
        extra='vllm:prompt_tokens_by_source_total{engine="1",source="local_compute"} 90\n'
        + ignored,
    )
    _write_prefill_snapshot(run, "before", 1, hits=10, queries=50, computed=40)
    _write_prefill_snapshot(run, "after", 1, hits=70, queries=350, computed=280)
    for worker in (2, 3):
        _write_prefill_snapshot(run, "before", worker, hits=0, queries=0, computed=0)
        _write_prefill_snapshot(
            run, "after", worker, hits=9999, queries=10000, computed=500000
        )

    stats = _calibrate_prefill(run)
    assert stats["prefill"] == {
        "prefix_cache_hit_rate": 0.25,
        "new_prompt_tokens": 300,
        "new_prompt_tokens_max_mean": 1.6,
        "workers": [
            dict(
                worker_index=0,
                prefix_cache_hits=40,
                prefix_cache_queries=100,
                new_prompt_tokens=60,
            ),
            dict(
                worker_index=1,
                prefix_cache_hits=60,
                prefix_cache_queries=300,
                new_prompt_tokens=240,
            ),
        ],
        "unavailable_metrics": [],
    }
    published = case_loader.router_policy_result_path(
        run.case.scenario, 16, run.case.task_num, "unit_test"
    )
    assert published.read_bytes() == (run.out / "calibration.json").read_bytes()


@pytest.mark.parametrize(
    "worker_count,prefill_count,indices,tokens,skew",
    [
        (4, 0, [0, 1, 2, 3], 2160, 1000 / 540),
        (2, 1, [0], 80, 1),
        (4, 2, [0, 1], 160, 1),
    ],
    ids=["pd_merge", "1p1d", "2p2d"],
)
def test_prefill_workers_follow_topology(
    prefill_run, worker_count, prefill_count, indices, tokens, skew
):
    run = prefill_run
    run.case.worker_count = worker_count
    run.case.prefill_count = prefill_count
    if not prefill_count:
        run.case.scenario = "pd_merge_kv_aware"
    for worker in (2, 3):
        _write_prefill_snapshot(
            run, "after", worker, hits=40, queries=200, computed=1080
        )
    prefill = _calibrate_prefill(run)["prefill"]
    assert [worker["worker_index"] for worker in prefill["workers"]] == indices
    assert prefill["new_prompt_tokens"] == tokens
    assert prefill["new_prompt_tokens_max_mean"] == pytest.approx(skew)


@pytest.mark.parametrize("phase", ["before", "after"])
@pytest.mark.parametrize(
    "field,counter",
    [
        ("prefix_cache_hits", "vllm:prefix_cache_hits_total"),
        ("prefix_cache_queries", "vllm:prefix_cache_queries_total"),
        ("new_prompt_tokens", "vllm:prompt_tokens_by_source_total"),
    ],
)
def test_missing_prefill_counters_are_unknown(prefill_run, phase, field, counter):
    run = prefill_run
    path = run.out / phase / "engine0.prom"
    path.write_text(
        "\n".join(
            line
            for line in path.read_text().splitlines()
            if not line.startswith(counter)
        )
        + "\n",
        encoding="utf-8",
    )
    stats = _calibrate_prefill(run)
    assert stats["ok"] is True
    prefill = stats["prefill"]
    assert prefill["workers"][0][field] is None
    assert f"engine0.{field}" in prefill["unavailable_metrics"]
    if field == "new_prompt_tokens":
        assert prefill["new_prompt_tokens"] is None
        assert prefill["new_prompt_tokens_max_mean"] is None
        assert prefill["workers"][1][field] == 80
    else:
        assert prefill["prefix_cache_hit_rate"] is None


@pytest.mark.parametrize("value", ["NaN", "+Inf", "-1", "1e400", "invalid"])
def test_invalid_prefill_compute_counter_is_unknown(prefill_run, value):
    _write_prefill_snapshot(
        prefill_run,
        "after",
        0,
        hits=40,
        queries=200,
        computed=value,
        extra='vllm:prompt_tokens_by_source_total{engine="1",source="local_compute"} 10\n',
    )
    stats = _calibrate_prefill(prefill_run)
    assert stats["ok"] is True
    assert stats["prefill"]["new_prompt_tokens"] is None
    assert stats["prefill"]["workers"][0]["new_prompt_tokens"] is None
    assert "engine0.new_prompt_tokens" in stats["prefill"]["unavailable_metrics"]


def test_prefill_counter_reset_is_unknown(prefill_run):
    _write_prefill_snapshot(prefill_run, "after", 0, hits=40, queries=200, computed=40)
    stats = _calibrate_prefill(prefill_run)
    assert stats["ok"] is True
    assert stats["prefill"]["new_prompt_tokens"] is None
    assert "engine0.new_prompt_tokens" in stats["prefill"]["unavailable_metrics"]


def test_zero_prefill_counters_are_preserved(prefill_run):
    for phase in ("before", "after"):
        for worker in range(4):
            _write_prefill_snapshot(
                prefill_run, phase, worker, hits=0, queries=0, computed=0
            )
    prefill = _calibrate_prefill(prefill_run)["prefill"]
    assert prefill["new_prompt_tokens"] == 0
    assert prefill["new_prompt_tokens_max_mean"] is None
    assert prefill["prefix_cache_hit_rate"] is None
    assert prefill["unavailable_metrics"] == []
