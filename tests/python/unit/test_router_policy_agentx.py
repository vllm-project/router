"""CPU checks for the pinned AgentX workload and replay configuration."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx
import yaml

from tests.python.e2e.perf.router_policy import case_loader
from tests.python.e2e.perf.router_policy.data.extract_agentx_ci import extract
from tests.python.e2e.perf.router_policy.stack_launcher import (
    engine_command,
    open_cell,
    runtime_config,
    snapshot,
)

pytestmark = [pytest.mark.ready, pytest.mark.cpu]
SUITE = Path(case_loader.__file__).resolve().parent


@pytest.fixture(
    params=sorted((SUITE / "cases").glob("*.json")), ids=lambda path: path.stem
)
def worker_case(request, monkeypatch, tmp_path):
    monkeypatch.setattr(case_loader, "_vllm_rs_bin", lambda: "vllm-rs")
    monkeypatch.setattr(
        case_loader,
        "packaged_replay_agentx_yaml",
        lambda trace_type: tmp_path / "replay.yaml",
    )
    return case_loader.load_router_policy_case(request.param, concurrency=8)


def test_router_policy_workers_use_portable_kv_hashes(worker_case):
    count = worker_case.worker_count
    ports = {
        name: [base + i for i in range(count)]
        for name, base in (
            ("http", 31000),
            ("grpc", 32000),
            ("kv_pub", 33000),
            ("kv_replay", 34000),
        )
    }
    run = SimpleNamespace(case=worker_case, ports=ports)
    topics = []
    for i in range(count):
        argv = engine_command(run, i)
        engine = argv[argv.index("--") + 1 :]
        assert "--prefix-caching-hash-algo" not in argv[: argv.index("--")]
        assert engine[engine.index("--prefix-caching-hash-algo") + 1] == "sha256_cbor"
        assert "--enable-prefix-caching" in engine
        events = json.loads(engine[engine.index("--kv-events-config") + 1])
        assert events["enable_kv_cache_events"] is True
        assert events["publisher"] == "zmq"
        assert events["endpoint"] == f"tcp://127.0.0.1:{ports['kv_pub'][i]}"
        assert events["replay_endpoint"] == f"tcp://127.0.0.1:{ports['kv_replay'][i]}"
        assert events["topic"]
        assert "instance_topic" not in events  # Not available in vLLM 0.30.0.
        topics.append(events["topic"])
        if worker_case.prefill_count:
            transfer = json.loads(engine[engine.index("--kv-transfer-config") + 1])
            assert transfer["kv_role"] == (
                "kv_producer" if i < worker_case.prefill_count else "kv_consumer"
            )
    assert len(set(topics)) == count
    restarted = engine_command(run, 0)
    events = json.loads(restarted[restarted.index("--kv-events-config") + 1])
    assert events["topic"] not in topics


def test_router_policy_worker_env_keeps_full_block_hashes(
    worker_case, tmp_path, monkeypatch
):
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    run = open_cell(worker_case, out_dir=tmp_path)
    try:
        assert run.env["VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES"] == "0"
    finally:
        run.clients.close()


@pytest.mark.parametrize("router_status", [200, 503])
def test_snapshot_uses_router_metrics_port_and_checks_http_status(
    tmp_path, router_status
):
    requests = []

    def handle(request):
        requests.append(str(request.url))
        if request.url.port == 30001:
            return httpx.Response(router_status, text="router metrics\n")
        return httpx.Response(200, text="worker metrics\n")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        run = SimpleNamespace(
            case=SimpleNamespace(worker_count=1),
            clients=client,
            ports={"router": 30000, "metrics": 30001, "http": [30002]},
        )
        if router_status == 200:
            snapshot(run, tmp_path)
            assert (tmp_path / "router.prom").read_text(
                encoding="utf-8"
            ) == "router metrics\n"
            assert (tmp_path / "engine0.prom").read_text(
                encoding="utf-8"
            ) == "worker metrics\n"
        else:
            with pytest.raises(httpx.HTTPStatusError):
                snapshot(run, tmp_path)
            assert not (tmp_path / "router.prom").exists()
    assert requests == [
        "http://127.0.0.1:30002/metrics",
        "http://127.0.0.1:30001/metrics",
    ]


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
    for filename in (
        "pd_disagg_cache_aware.json",
        "pd_disagg_kv_aware.json",
        "pd_disagg_round_robin.json",
    ):
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
            "pd_disagg_kv_aware",
            "agentX",
            "/v1/completions",
            "replay_agentX.yaml",
        ),
        (
            "pd_disagg_round_robin",
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
        SUITE / "cases" / f"{scenario}.json", concurrency=8
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
