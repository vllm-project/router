"""CPU checks for the pinned AgentX workload and replay configuration."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.python.e2e.perf.router_policy import case_loader
from tests.python.e2e.perf.router_policy.data.extract_agentx_ci import extract
from tests.python.e2e.perf.router_policy.stack_launcher import runtime_config

pytestmark = [pytest.mark.ready, pytest.mark.cpu]
SUITE = Path(case_loader.__file__).resolve().parent


def test_agentx_slice_matches_manifest_and_pd_cases():
    manifest = json.loads(
        (SUITE / "data/agentx-ci-8x8.manifest.json").read_text(encoding="utf-8")
    )
    trace = SUITE / "data" / manifest["trace_file"]
    assert hashlib.sha256(trace.read_bytes()).hexdigest() == manifest["trace_sha256"]
    sessions = [
        json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()
    ]
    assert len(sessions) == manifest["session_count"]
    assert [s["id"] for s in sessions] == [s["id"] for s in manifest["sessions"]]
    requests = []
    for session in sessions:
        assert session["hash_id_scope"] == "local"
        assert len(session["requests"]) == manifest["turns_per_session"]
        for request in session["requests"]:
            assert request["type"] != "subagent"
            assert (
                len(request["hash_ids"])
                == (request["in"] + session["block_size"] - 1) // session["block_size"]
            )
            assert 1 < request["out"]
            assert request["in"] + request["out"] <= manifest["max_model_len"]
            requests.append(request)
    assert len(requests) == manifest["request_count"]
    assert sum(r["in"] for r in requests) == manifest["input_tokens"]
    assert sum(r["out"] for r in requests) == manifest["output_tokens"]
    assert max(r["in"] + r["out"] for r in requests) == manifest["max_request_tokens"]
    for filename in ("pd_disagg_cache_aware.json", "pd_disagg_kv_aware_affinity.json"):
        case = json.loads((SUITE / "cases" / filename).read_text(encoding="utf-8"))
        assert case["runtime"]["trace_type"] == "agentX"
        assert case["runtime"]["trace_sha256"] == manifest["trace_sha256"]
        assert (case_loader._repo_root() / case["runtime"]["trace_path"]) == trace
        assert (
            case["workers"]["per_worker"]["max_model_len"] == manifest["max_model_len"]
        )
        assert case["benchmark_params"]["task-num"] == len(sessions)


@pytest.mark.parametrize(
    "scenario,trace_type,endpoint,template",
    [
        (
            "pd_disagg_cache_aware",
            "agentX",
            "/v1/completions",
            "replay_agentX.yaml",
        ),
        (
            "pd_disagg_kv_aware_affinity",
            "agentX",
            "/v1/completions",
            "replay_agentX.yaml",
        ),
        (
            "pd_merge_cache_aware",
            "tracelab",
            "/v1/chat/completions",
            "replay_tracelab.yaml",
        ),
    ],
)
def test_replay_uses_the_matching_adapter_template(
    tmp_path, monkeypatch, scenario, trace_type, endpoint, template
):
    package = tmp_path / "agentinfer"
    configs = package / "agentbench/configs"
    configs.mkdir(parents=True)
    for filename in ("replay_agentX.yaml", "replay_tracelab.yaml"):
        (configs / filename).write_text(
            yaml.safe_dump({"experiment": {}, "backend": {}, "replay": {}}),
            encoding="utf-8",
        )
    monkeypatch.setitem(
        sys.modules,
        "agentinfer",
        SimpleNamespace(__file__=str(package / "__init__.py")),
    )
    monkeypatch.setattr(case_loader, "_vllm_rs_bin", lambda: "vllm-rs")
    case = case_loader.load_router_policy_case(
        SUITE / "cases" / f"{scenario}.json", concurrency=16
    )
    assert case.replay_config.name == template
    run = SimpleNamespace(
        case=case,
        out=tmp_path,
        ports={"router": 30000, "metrics": 30001, "http": [30002]},
    )
    config = yaml.safe_load(runtime_config(run).read_text(encoding="utf-8"))
    assert config["replay"]["trace_type"] == trace_type
    assert config["backend"]["endpoint"] == endpoint
    assert config["replay"]["sample_seed"] == 0
    assert config["replay"]["max_inflight_requests"] == 1
    assert config["replay"]["trace_same_agent_gap_scale"] == case.gap_scale
    if trace_type == "agentX":
        assert case.prefill_count == 2
        assert case.trace_sha256


@pytest.fixture
def extraction_source(tmp_path):
    session = {"id": "source-session", "requests": [{"in": 4}, {"in": 8}, {"in": 16}]}
    raw = (json.dumps(session) + "\n").encode()
    source = tmp_path / "source.jsonl"
    source.write_bytes(raw)
    expected = (
        json.dumps(
            {**session, "requests": session["requests"][:2]},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    manifest = {
        "sessions": [
            {
                "id": session["id"],
                "source_line": 1,
                "source_row_sha256": hashlib.sha256(raw).hexdigest(),
            }
        ],
        "turns_per_session": 2,
        "trace_sha256": hashlib.sha256(expected).hexdigest(),
    }
    return source, manifest, expected


def test_extraction_preserves_source_requests(extraction_source):
    source, manifest, expected = extraction_source
    assert extract(source, manifest) == expected


@pytest.mark.parametrize("tamper", ["source", "manifest"])
def test_extraction_rejects_changed_data(extraction_source, tamper):
    source, manifest, _ = extraction_source
    if tamper == "source":
        source.write_bytes(source.read_bytes().replace(b'"in": 4', b'"in": 5'))
    else:
        manifest["trace_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        extract(source, manifest)
