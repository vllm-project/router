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
    _write_replay_artifacts(tmp_path, [1] * len(rows))
    for i, row in enumerate(rows):
        row.update(request_id=f"request-{i}-0", session_id=f"session-{i}")
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
    _write_replay_artifacts(tmp_path, [1])
    (tmp_path / "after").mkdir()
    (tmp_path / "after/router.prom").write_text(
        'vllm_router_kv_placement_lookups_total{result="miss"} 1\n'
        'vllm_router_pd_affinity_decisions_total{stage="decode",outcome="bind"} 1\n'
        'vllm_router_prefix_routing_decisions_total{policy="kv_aware",stage="regular",outcome="sticky_hit"} 1\n',
        encoding="utf-8",
    )
    case = SimpleNamespace(
        policy="kv_aware_aff",
        scenario="pd_disagg_kv_aware_affinity",
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
    summary_module, calibration_run, prefill_count, missing, expected, capsys
):
    run = calibration_run
    run.case.prefill_count = prefill_count
    prefix = "pd_disagg" if prefill_count else "pd_merge"
    run.case.scenario = f"{prefix}_kv_aware_affinity"
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
        summary_module.write_calibration(
            run, result_dir=result, replay_rc=0, wall_seconds=1, stalled=False
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


@pytest.mark.parametrize("topology", ["pd_merged", "pd_disaggregated"])
@pytest.mark.parametrize(
    "scenario_metadata",
    [{}, {"scenario": None}, {"scenario": ""}, {"scenario": " \t"}, {"scenario": 1}],
    ids=["missing", "null", "empty", "whitespace", "non_string"],
)
def test_publication_requires_scenario(
    summary_module, calibration_run, topology, scenario_metadata
):
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

    assert summary_module._write_results_summary_from_calibration(run.out) == []
    assert not list((run.out / "published").rglob("*.json"))
    assert (
        json.loads((run.out / "calibration.json").read_text(encoding="utf-8")) == stats
    )


def test_same_policy_topologies_publish_separate_results(
    summary_module, calibration_run
):
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
        artifacts = summary_module._write_results_summary_from_calibration(run.out)
        expected = case_loader.router_policy_result_path(
            stats["scenario"], 16, 8, "unit_test"
        )
        assert artifacts == [("results_json", expected)]
        published.append((expected, stats))

    assert published[0][0] != published[1][0]
    for path, stats in published:
        assert json.loads(path.read_text(encoding="utf-8")) == stats


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
            summary_module.write_calibration(
                run, result_dir=result, replay_rc=0, wall_seconds=1, stalled=False
            )
        assert not list((run.out / "published").rglob("*.json"))
        return

    summary_module.write_calibration(
        run, result_dir=result, replay_rc=0, wall_seconds=1, stalled=False
    )
    published = case_loader.router_policy_result_path(
        run.case.scenario, 16, run.case.task_num, "unit_test"
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


def _calibrate_prefill(summary_module, run):
    summary_module.write_calibration(
        run,
        result_dir=run.out / "result",
        replay_rc=0,
        wall_seconds=1,
        stalled=False,
    )
    return json.loads((run.out / "calibration.json").read_text(encoding="utf-8"))


def _write_replay_artifacts(out, requests_per_task):
    """Create independent pre-run inventory and plan, then return complete facts."""
    tasks, sources, rows = [], [], []
    for i, count in enumerate(requests_per_task):
        planned, source = [], []
        for j in range(count):
            key, request_id = f"source-{i}-{j}", f"request-{i}-{j}"
            source.append({"key": key, "replay_kind": "request"})
            planned.append(
                {
                    "source_key": key,
                    "node_type": "request",
                    "runtime_request_id": request_id,
                }
            )
            rows.append(
                dict(
                    request_id=request_id,
                    session_id=f"session-{i}",
                    status="success",
                    ttft_seconds=0.1,
                    latency_seconds=0.5,
                    output_tokens=3,
                    duration_seconds=1.0,
                )
            )
        tasks.append(
            {
                "source_session_id": f"source-{i}",
                "runtime_session_id": f"session-{i}",
                "requests": planned,
            }
        )
        sources.append(
            {"source_session_id": f"source-{i}", "replayable": True, "requests": source}
        )
    result = out / "result"
    (result / "replay-plan.json").write_text(
        json.dumps({"tasks": tasks}), encoding="utf-8"
    )
    (result / "replay-source-analysis.json").write_text(
        json.dumps({"sessions": sources}), encoding="utf-8"
    )
    return rows


@pytest.mark.parametrize("prefill_count", [0, 2], ids=["tracelab_28", "agentx_144"])
@pytest.mark.parametrize(
    "corruption",
    [
        "subset",
        "extra",
        "duplicate",
        "missing_id",
        "empty_id",
        "invalid_id",
        "wrong_id",
        "wrong_session",
    ],
)
def test_incomplete_replay_fails_before_publication(
    summary_module, prefill_run, prefill_count, corruption, capsys
):
    run = prefill_run
    run.case.prefill_count = prefill_count
    run.case.task_num = 12 if prefill_count else 8
    counts = [12] * 12 if prefill_count else [4] * 4 + [3] * 4
    rows = _write_replay_artifacts(run.out, counts)
    if corruption == "subset":
        rows = rows[:1]
    elif corruption == "extra":
        rows.append({**rows[-1], "request_id": "extra-request"})
    elif corruption == "duplicate":
        rows[-1] = rows[0].copy()
    elif corruption == "missing_id":
        del rows[-1]["request_id"]
    elif corruption == "empty_id":
        rows[-1]["request_id"] = " \t"
    elif corruption == "invalid_id":
        rows[-1]["request_id"] = 42
    elif corruption == "wrong_id":
        rows[-1]["request_id"] = "unexpected-request"
    else:
        rows[-1]["session_id"] = "wrong-session"
    (run.out / "result/requests.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="1"):
        _calibrate_prefill(summary_module, run)
    stats = json.loads((run.out / "calibration.json").read_text(encoding="utf-8"))
    assert stats["requests"] == stats["success"] == len(rows)
    assert stats["ok"] is False
    assert stats["replay_validation"]["expected_requests"] == sum(counts)
    assert "DIAGNOSTIC replay completeness:" in capsys.readouterr().out
    assert not list((run.out / "published").rglob("*.json"))


@pytest.mark.parametrize("prefill_count", [0, 2], ids=["tracelab_28", "agentx_144"])
def test_complete_replay_publishes_all_requests(
    summary_module, prefill_run, prefill_count
):
    run = prefill_run
    run.case.prefill_count = prefill_count
    run.case.task_num = 12 if prefill_count else 8
    run.case.scenario = (
        f"{'pd_disagg' if prefill_count else 'pd_merge'}_kv_aware_affinity"
    )
    counts = [12] * 12 if prefill_count else [4] * 4 + [3] * 4
    rows = _write_replay_artifacts(run.out, counts)
    (run.out / "result/requests.jsonl").write_text(
        "\n".join(json.dumps(row) for row in reversed(rows)), encoding="utf-8"
    )
    stats = _calibrate_prefill(summary_module, run)
    assert stats["ok"] is True
    assert stats["replay_validation"] == {
        "expected_requests": sum(counts),
        "observed_requests": sum(counts),
        "distinct_request_ids": sum(counts),
        "errors": [],
    }
    published = case_loader.router_policy_result_path(
        run.case.scenario, 16, run.case.task_num, "unit_test"
    )
    assert published.read_bytes() == (run.out / "calibration.json").read_bytes()


@pytest.mark.parametrize(
    "artifact", ["replay-plan.json", "replay-source-analysis.json"]
)
@pytest.mark.parametrize("corruption", ["missing", "invalid_json", "invalid_root"])
def test_invalid_replay_artifacts_prevent_publication(
    summary_module, prefill_run, artifact, corruption
):
    path = prefill_run.out / "result" / artifact
    if corruption == "missing":
        path.unlink()
    else:
        path.write_text(
            "{" if corruption == "invalid_json" else "null", encoding="utf-8"
        )
    with pytest.raises(SystemExit, match="1"):
        _calibrate_prefill(summary_module, prefill_run)
    stats = json.loads(
        (prefill_run.out / "calibration.json").read_text(encoding="utf-8")
    )
    assert stats["replay_validation"]["errors"][0].startswith(
        "invalid replay plan/source inventory:"
    )
    assert not list((prefill_run.out / "published").rglob("*.json"))


@pytest.mark.parametrize(
    "corruption",
    ["missing_task", "missing_request", "duplicate_id", "missing_source_cycle"],
)
def test_incomplete_plan_cannot_define_a_smaller_expected_load(
    summary_module, prefill_run, corruption
):
    run = prefill_run
    run.case.task_num = 2
    rows = _write_replay_artifacts(run.out, [2, 2])
    path = run.out / "result/replay-plan.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    if corruption == "missing_task":
        plan["tasks"].pop()
        rows = rows[:2]
    elif corruption == "missing_request":
        plan["tasks"][-1]["requests"].pop()
        rows.pop()
    elif corruption == "duplicate_id":
        plan["tasks"][-1]["requests"][-1]["runtime_request_id"] = rows[0]["request_id"]
        rows[-1]["request_id"] = rows[0]["request_id"]
    else:
        plan["tasks"][-1]["source_session_id"] = "source-0"
        for j, node in enumerate(plan["tasks"][-1]["requests"]):
            node["source_key"] = f"source-0-{j}"
    path.write_text(json.dumps(plan), encoding="utf-8")
    (run.out / "result/requests.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="1"):
        _calibrate_prefill(summary_module, run)
    stats = json.loads((run.out / "calibration.json").read_text(encoding="utf-8"))
    assert stats["replay_validation"]["errors"]
    assert not list((run.out / "published").rglob("*.json"))


def test_replay_allows_repeated_source_cycles_and_unmeasured_timing_nodes(
    summary_module, prefill_run
):
    run = prefill_run
    run.case.task_num = 2
    rows = _write_replay_artifacts(run.out, [1, 1])
    result = run.out / "result"
    analysis = json.loads(
        (result / "replay-source-analysis.json").read_text(encoding="utf-8")
    )
    analysis["sessions"] = analysis["sessions"][:1]
    analysis["sessions"][0]["requests"].append(
        {"key": "timer", "replay_kind": "timing_dependency"}
    )
    plan = json.loads((result / "replay-plan.json").read_text(encoding="utf-8"))
    for task in plan["tasks"]:
        task["source_session_id"] = "source-0"
        task["requests"][0]["source_key"] = "source-0-0"
        task["requests"].append(
            {"source_key": "timer", "node_type": "timing_dependency"}
        )
    (result / "replay-source-analysis.json").write_text(
        json.dumps(analysis), encoding="utf-8"
    )
    (result / "replay-plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (result / "requests.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    stats = _calibrate_prefill(summary_module, run)
    assert stats["ok"] is True
    assert stats["replay_validation"]["expected_requests"] == 2
    assert stats["replay_validation"]["errors"] == []


@pytest.mark.parametrize("prefill_count", [0, 2], ids=["pd_merge", "pd_disagg"])
@pytest.mark.parametrize("policy", ["kv_aware", "kv_aware_aff"])
def test_unavailable_kv_index_fails_before_publication(
    summary_module, prefill_run, prefill_count, policy, capsys
):
    run = prefill_run
    run.case.prefill_count = prefill_count
    run.case.policy = policy
    prefix = "pd_disagg" if prefill_count else "pd_merge"
    run.case.scenario = f"{prefix}_{policy}"
    snapshot = run.out / "after/router.prom"
    snapshot.write_text(
        'vllm_router_kv_placement_lookups_total{result="index_unavailable"} 100\n'
        'vllm_router_kv_routing_inputs_total{transport="grpc",result="prepared"} 64\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="1"):
        _calibrate_prefill(summary_module, run)

    stats = json.loads((run.out / "calibration.json").read_text(encoding="utf-8"))
    assert stats["requests"] == stats["success"] == 1
    assert stats["ok"] is False
    assert "zero KV placement lookups" in capsys.readouterr().out
    assert not list((run.out / "published").rglob("*.json"))


@pytest.mark.parametrize("prefill_count", [0, 2], ids=["pd_merge", "pd_disagg"])
@pytest.mark.parametrize("policy", ["kv_aware", "kv_aware_aff"])
@pytest.mark.parametrize("result", ["hit", "miss"])
def test_verified_kv_lookup_allows_publication(
    summary_module, prefill_run, prefill_count, policy, result
):
    run = prefill_run
    run.case.prefill_count = prefill_count
    run.case.policy = policy
    snapshot = run.out / "after/router.prom"
    # Labels can be reordered and additional labels must not hide valid activity.
    with snapshot.open("a", encoding="utf-8") as stream:
        stream.write(
            f'vllm_router_kv_placement_lookups_total{{worker="0",result="{result}"}} 2\n'
            f'vllm_router_kv_placement_lookups_total{{result="{result}",worker="1"}} 3\n'
        )
    stats = _calibrate_prefill(summary_module, run)

    assert stats["ok"] is True
    checks = stats["router_policy_checks"]
    assert checks["errors"] == []
    assert checks["placement_lookups"][result] == (6 if result == "miss" else 5)
    if policy == "kv_aware":
        assert "affinity" not in checks
    else:
        assert checks["affinity"]["stage"] == ("decode" if prefill_count else "regular")
    published = case_loader.router_policy_result_path(
        run.case.scenario, 16, run.case.task_num, "unit_test"
    )
    assert published.read_bytes() == (run.out / "calibration.json").read_bytes()


@pytest.mark.parametrize("prefill_count", [0, 2], ids=["pd_merge", "pd_disagg"])
@pytest.mark.parametrize("sample", [None, "", "0", "NaN", "+Inf", "-1", "invalid"])
def test_missing_or_invalid_lookup_counters_fail(
    summary_module, prefill_run, prefill_count, sample
):
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
        _calibrate_prefill(summary_module, run)
    stats = json.loads((run.out / "calibration.json").read_text(encoding="utf-8"))
    assert len(stats["router_policy_checks"]["errors"]) == 1
    assert "sha256_cbor" in stats["router_policy_checks"]["errors"][0]
    assert not list((run.out / "published").rglob("*.json"))


@pytest.mark.parametrize(
    "prefill_count,metric,labels,passes",
    [
        (2, "pd_affinity", 'stage="decode",outcome="bind"', True),
        (2, "pd_affinity", 'outcome="sticky_hit",stage="decode"', True),
        (2, "pd_affinity", 'stage="prefill",outcome="bind"', False),
        (2, "pd_affinity", 'stage="decode",outcome="no_session"', False),
        (2, "pd_affinity", 'stage="decode",outcome="capacity_bypass"', False),
        (
            2,
            "prefix_routing",
            'policy="kv_aware",stage="regular",outcome="sticky_hit"',
            False,
        ),
        (
            0,
            "prefix_routing",
            'policy="kv_aware",stage="regular",outcome="sticky_hit"',
            True,
        ),
        (
            0,
            "prefix_routing",
            'policy="cache_aware",stage="regular",outcome="sticky_hit"',
            False,
        ),
        (
            0,
            "prefix_routing",
            'policy="kv_aware",stage="prefill",outcome="sticky_hit"',
            False,
        ),
        (
            0,
            "prefix_routing",
            'policy="kv_aware",stage="regular",outcome="prefix_cost"',
            False,
        ),
        (0, "pd_affinity", 'stage="decode",outcome="bind"', False),
    ],
)
def test_affinity_requires_the_matching_stage_and_outcome(
    summary_module, prefill_run, prefill_count, metric, labels, passes, capsys
):
    run = prefill_run
    run.case.prefill_count = prefill_count
    (run.out / "after/router.prom").write_text(
        'vllm_router_kv_placement_lookups_total{result="miss"} 1\n'
        f"vllm_router_{metric}_decisions_total{{{labels}}} 1\n",
        encoding="utf-8",
    )
    if passes:
        assert _calibrate_prefill(summary_module, run)["ok"] is True
    else:
        with pytest.raises(SystemExit, match="1"):
            _calibrate_prefill(summary_module, run)
        stats = json.loads((run.out / "calibration.json").read_text(encoding="utf-8"))
        assert len(stats["router_policy_checks"]["errors"]) == 1
        assert "affinity decisions" in capsys.readouterr().out
        assert not list((run.out / "published").rglob("*.json"))


@pytest.mark.parametrize("prefill_count", [0, 2], ids=["pd_merge", "pd_disagg"])
def test_cache_aware_skips_index_and_affinity_checks(
    summary_module, prefill_run, prefill_count
):
    run = prefill_run
    run.case.policy = "cache_aware"
    run.case.prefill_count = prefill_count
    (run.out / "after/router.prom").unlink()
    stats = _calibrate_prefill(summary_module, run)
    assert stats["ok"] is True
    assert stats["router_policy_checks"] == {"errors": []}


def test_prefill_deltas_exclude_decode_and_copy_to_results(summary_module, prefill_run):
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

    stats = _calibrate_prefill(summary_module, run)
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
    summary_module, prefill_run, worker_count, prefill_count, indices, tokens, skew
):
    run = prefill_run
    run.case.worker_count = worker_count
    run.case.prefill_count = prefill_count
    if not prefill_count:
        run.case.scenario = "pd_merge_kv_aware_affinity"
    for worker in (2, 3):
        _write_prefill_snapshot(
            run, "after", worker, hits=40, queries=200, computed=1080
        )
    prefill = _calibrate_prefill(summary_module, run)["prefill"]
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
def test_missing_prefill_counters_are_unknown(
    summary_module, prefill_run, phase, field, counter
):
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
    stats = _calibrate_prefill(summary_module, run)
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
def test_invalid_prefill_compute_counter_is_unknown(summary_module, prefill_run, value):
    _write_prefill_snapshot(
        prefill_run,
        "after",
        0,
        hits=40,
        queries=200,
        computed=value,
        extra='vllm:prompt_tokens_by_source_total{engine="1",source="local_compute"} 10\n',
    )
    stats = _calibrate_prefill(summary_module, prefill_run)
    assert stats["ok"] is True
    assert stats["prefill"]["new_prompt_tokens"] is None
    assert stats["prefill"]["workers"][0]["new_prompt_tokens"] is None
    assert "engine0.new_prompt_tokens" in stats["prefill"]["unavailable_metrics"]


def test_prefill_counter_reset_is_unknown(summary_module, prefill_run):
    _write_prefill_snapshot(prefill_run, "after", 0, hits=40, queries=200, computed=40)
    stats = _calibrate_prefill(summary_module, prefill_run)
    assert stats["ok"] is True
    assert stats["prefill"]["new_prompt_tokens"] is None
    assert "engine0.new_prompt_tokens" in stats["prefill"]["unavailable_metrics"]


def test_zero_prefill_counters_are_preserved(summary_module, prefill_run):
    for phase in ("before", "after"):
        for worker in range(4):
            _write_prefill_snapshot(
                prefill_run, phase, worker, hits=0, queries=0, computed=0
            )
    prefill = _calibrate_prefill(summary_module, prefill_run)["prefill"]
    assert prefill["new_prompt_tokens"] == 0
    assert prefill["new_prompt_tokens_max_mean"] is None
    assert prefill["prefix_cache_hit_rate"] is None
    assert prefill["unavailable_metrics"] == []
