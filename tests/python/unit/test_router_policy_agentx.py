"""CPU checks for the pinned AgentX workload and replay configuration."""

import hashlib
import io
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx
import yaml

from tests.python.e2e.perf.router_policy import case_loader
from tests.python.e2e.perf.router_policy.data import extract_agentx_ci
from tests.python.e2e.perf.router_policy.stack_launcher import (
    allocate_ports,
    cleanup,
    engine_command,
    open_cell,
    runtime_config,
    snapshot,
    start_workers,
)

pytestmark = [pytest.mark.ready, pytest.mark.cpu]
SUITE = Path(__file__).parents[1] / "e2e/perf/router_policy"


@pytest.fixture(
    params=(
        "pd_merge_cache_aware",
        "pd_merge_kv_aware",
        "pd_merge_kv_aware_affinity",
        "pd_disagg_cache_aware",
        "pd_disagg_kv_aware",
        "pd_disagg_round_robin",
    )
)
def worker_case(request, monkeypatch, tmp_path):
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
    return case_loader.load_router_policy_case(
        SUITE / "cases" / f"{request.param}.json"
    )


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


@pytest.mark.parametrize("configured", [False, True], ids=["unset", "configured"])
def test_worker_cache_environment_is_preserved(
    worker_startup, tmp_path, monkeypatch, configured
):
    directories = ("XDG_CACHE_HOME", "HF_HOME", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR")
    for name in directories:
        if configured:
            monkeypatch.setenv(name, str(tmp_path / "mounted" / name))
        else:
            monkeypatch.delenv(name, raising=False)
    run, processes, _ = worker_startup
    start_workers(run)
    for process in processes:
        for name in directories:
            assert name not in run.env
            if configured:
                assert process.env[name] == str(tmp_path / "mounted" / name)
            else:
                assert name not in process.env


@pytest.fixture
def worker_startup(worker_case, tmp_path, monkeypatch):
    processes, health_checks = [], []

    class Worker:
        def __init__(self, command, **kwargs):
            self.stdout = io.StringIO("")
            self.returncode = None
            self.env = kwargs["env"]
            processes.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    def health(url, **kwargs):
        assert len(processes) == worker_case.worker_count, (
            "health check blocked launch of the remaining workers"
        )
        health_checks.append(url)
        response = io.BytesIO()
        response.status = 200
        return response

    monkeypatch.setattr(subprocess, "Popen", Worker)
    monkeypatch.setattr(urllib.request, "urlopen", health)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    run = open_cell(worker_case, out_dir=tmp_path)
    allocate_ports(run)
    try:
        yield run, processes, health_checks
    finally:
        cleanup(run)
        run.clients.close()


def test_worker_startup_launches_all_gpus_before_waiting(worker_startup, monkeypatch):
    run, processes, health_checks = worker_startup
    monkeypatch.setenv("UCX_TLS", "tcp,self")
    start_workers(run)
    assert health_checks == [
        f"http://127.0.0.1:{port}/health" for port in run.ports["http"]
    ]
    assert [proc.env["CUDA_VISIBLE_DEVICES"] for proc in processes] == run.gpus
    assert all(proc.env["UCX_TLS"] == "tcp,self" for proc in processes)
    assert all(proc.poll() is None for proc in processes)


def test_worker_startup_timeout_cleans_up_every_started_process(worker_startup):
    run, processes, health_checks = worker_startup
    with pytest.raises(TimeoutError, match="not ready"):
        start_workers(run, timeout=0)
    assert len(processes) == run.case.worker_count
    assert not health_checks
    assert all(proc.poll() is not None for proc in processes)
    assert not run.stack
    assert not run.port_holders
    assert not run.kv_port_holders
    assert not run.nixl_port_holders


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


def test_agentx_slice_matches_pd_cases():
    trace = SUITE / "data" / extract_agentx_ci.TRACE_FILE
    assert (
        hashlib.sha256(trace.read_bytes()).hexdigest() == extract_agentx_ci.TRACE_SHA256
    )
    sessions = [
        json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()
    ]
    assert len(sessions) == extract_agentx_ci.SESSION_COUNT == 12
    assert tuple(session["id"] for session in sessions) == extract_agentx_ci.SESSION_IDS
    requests = []
    subagent_groups = 0
    for session in sessions:
        assert session["hash_id_scope"] == "local"
        assert len(session["requests"]) == extract_agentx_ci.TURNS_PER_SESSION == 12
        session_requests = []
        for entry in session["requests"]:
            if entry["type"] == "subagent":
                subagent_groups += 1
                session_requests.extend(entry["requests"])
            else:
                session_requests.append(entry)
        for request in session_requests:
            assert (
                len(request["hash_ids"])
                == (request["in"] + session["block_size"] - 1) // session["block_size"]
            )
            assert 0 <= request["out"]
            assert request["in"] + request["out"] <= extract_agentx_ci.MAX_MODEL_LEN
        requests.extend(session_requests)
    assert subagent_groups > 0
    assert len(requests) == 315
    for filename in (
        "pd_disagg_cache_aware.json",
        "pd_disagg_kv_aware.json",
        "pd_disagg_round_robin.json",
    ):
        case = json.loads((SUITE / "cases" / filename).read_text(encoding="utf-8"))
        assert case["runtime"]["trace_type"] == "agentX"
        assert case["runtime"]["trace_sha256"] == extract_agentx_ci.TRACE_SHA256
        assert (SUITE.parents[4] / case["runtime"]["trace_path"]) == trace
        assert (
            case["workers"]["per_worker"]["max_model_len"]
            == extract_agentx_ci.MAX_MODEL_LEN
        )
        assert case["benchmark_params"]["task-num"] == len(sessions)
        assert case["benchmark_params"]["max-concurrency"] == [12]


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
    vllm_package = tmp_path / "vllm"
    vllm_package.mkdir()
    (vllm_package / "vllm-rs").touch()
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(__file__=str(vllm_package / "__init__.py"))
    )
    case = case_loader.load_router_policy_case(SUITE / "cases" / f"{scenario}.json")
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
        assert config["experiment"]["task_num"] == 12
        assert config["experiment"]["max_concurrency"] == 12


@pytest.fixture
def extraction_source(tmp_path):
    session = {
        "id": "source-session",
        "hash_id_scope": "local",
        "block_size": 4,
        "requests": [
            {"in": 4, "out": 2, "hash_ids": [1]},
            {
                "type": "subagent",
                "agent_id": "child",
                "requests": [
                    {"in": 8, "out": 2, "hash_ids": [1, 2]},
                    {"in": 4, "out": 0, "hash_ids": [3]},
                ],
            },
            {"in": 16, "out": 2, "hash_ids": [1, 2, 3, 4]},
        ],
    }
    second = {**session, "id": "second-session"}
    raw = (json.dumps(session) + "\n" + json.dumps(second) + "\n").encode()
    source = tmp_path / "source.jsonl"
    source.write_bytes(raw)
    expected = "".join(
        json.dumps({**one, "requests": one["requests"][:2]}) + "\n"
        for one in (second, session)
    ).encode()
    options = dict(
        session_ids=("second-session", "source-session"),
        turns_per_session=2,
        expected_sha256=hashlib.sha256(expected).hexdigest(),
    )
    return source, options, expected


def test_extraction_preserves_source_requests(extraction_source):
    source, options, expected = extraction_source
    assert extract_agentx_ci.extract(source, **options) == expected


@pytest.mark.parametrize("tamper", ["source", "digest"])
def test_extraction_rejects_changed_data(extraction_source, tamper):
    source, options, _ = extraction_source
    if tamper == "source":
        source.write_bytes(source.read_bytes().replace(b'"out": 2', b'"out": 3'))
    else:
        options["expected_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        extract_agentx_ci.extract(source, **options)


def test_extraction_requires_distinct_sessions(extraction_source):
    source, options, _ = extraction_source
    options["session_ids"] = ("source-session", "source-session")
    with pytest.raises(ValueError, match="distinct session IDs"):
        extract_agentx_ci.extract(source, **options)


def test_extraction_rejects_missing_fixed_sessions(extraction_source):
    source, options, _ = extraction_source
    options["session_ids"] = ("source-session", "missing-session")
    with pytest.raises(ValueError, match="missing fixed sessions"):
        extract_agentx_ci.extract(source, **options)


def test_extraction_rejects_incomplete_sessions(extraction_source):
    source, options, _ = extraction_source
    options["turns_per_session"] = 4
    with pytest.raises(ValueError, match="fewer than 4 turns"):
        extract_agentx_ci.extract(source, **options)
