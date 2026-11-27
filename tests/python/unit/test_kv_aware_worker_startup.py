"""Check hybrid worker startup and cleanup without loading GPU models."""

import json
from types import SimpleNamespace

import pytest

from tests.python.e2e.function.helpers import launch
from tests.python.e2e.function.kv_aware import config
from tests.python.e2e.function.kv_aware import conftest as kv_fixtures

pytestmark = [pytest.mark.ready, pytest.mark.cpu]


@pytest.fixture
def worker_startup(monkeypatch, tmp_path):
    events = []
    workers = []
    clock = [0.0]
    state = SimpleNamespace(fail_worker=None)
    case = {
        "model": "startup-test",
        "local_model_env": "KV_AWARE_STARTUP_TEST_MODEL",
        "worker_count": 2,
        "worker_health_timeout_sec": 60,
        "worker_params": {"tensor_parallel_size": 2, "max_model_len": 4096},
        "router_params": {
            "--kv-hybrid-layout": {
                "ranks": [
                    {
                        "rank": 0,
                        "hash_block_size": 16,
                        "hash_seed": "0",
                        "speculative_tokens": 0,
                        "groups": [{"kind": "full_attention"}],
                    }
                ]
            }
        },
    }

    def spawn_worker(*args, **kwargs):
        index = len(workers)
        worker = SimpleNamespace(index=index, args=kwargs)
        workers.append(worker)
        events.append(("start", index))
        kwargs["log_file"].write("kv cache group sizes [16]\n")
        kwargs["log_file"].flush()
        clock[0] += 1
        return worker

    def wait_health(url, *, timeout, proc):
        events.append(("health", proc.index, timeout))
        if state.fail_worker == proc.index:
            raise TimeoutError("worker exited before health OK")
        if proc.index == 0:
            clock[0] += 30

    def spawn_router(*args, **kwargs):
        assert {event[1] for event in events if event[0] == "health"} == {0, 1}
        events.append(("router", kwargs))
        return SimpleNamespace(index="router")

    monkeypatch.setattr(config, "kv_aware_gpu_pool", lambda: [0, 1, 2, 3])
    monkeypatch.setattr(kv_fixtures.shutil, "which", lambda _: "/test/vllm")
    monkeypatch.setattr(
        kv_fixtures, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    monkeypatch.setattr(
        launch,
        "find_distinct_available_ports",
        lambda count: list(range(20000, 20000 + count)),
    )
    monkeypatch.setattr(launch, "popen_launch_vllm_http_worker", spawn_worker)
    monkeypatch.setattr(launch, "popen_launch_vllm_rs_grpc_worker", spawn_worker)
    monkeypatch.setattr(launch, "wait_worker_health", wait_health)
    monkeypatch.setattr(launch, "popen_launch_router_only", spawn_router)
    monkeypatch.setattr(
        launch, "terminate_process", lambda proc: events.append(("stop", proc.index))
    )

    def start(protocol, deployment):
        request = SimpleNamespace(param=(case, protocol, deployment))
        return kv_fixtures.hybrid_kv_aware_stack.__wrapped__(request, tmp_path)

    return SimpleNamespace(
        start=start,
        events=events,
        workers=workers,
        state=state,
        layout_path=tmp_path / "cache_layout.json",
    )


@pytest.mark.parametrize("protocol", ["http", "grpc"])
@pytest.mark.parametrize("deployment", ["merge", "1p1d_nixl"])
def test_hybrid_worker_startup_order_and_timeouts(worker_startup, protocol, deployment):
    run = worker_startup
    stack = run.start(protocol, deployment)
    try:
        result = next(stack)
        events = run.events
        order = [(event[0], event[1]) for event in events if event[0] != "router"]
        if deployment == "1p1d_nixl":
            assert order == [("start", 0), ("start", 1), ("health", 0), ("health", 1)]
            budgets = [event[2] for event in events if event[0] == "health"]
            assert budgets == pytest.approx([59, 30])
        else:
            assert order == [("start", 0), ("health", 0), ("start", 1), ("health", 1)]
            budgets = [event[2] for event in events if event[0] == "health"]
            assert budgets == pytest.approx([60, 60])
        gpu_key = "gpu_id" if protocol == "http" else "base_gpu_id"
        assert [worker.args[gpu_key] for worker in run.workers] == [[0, 1], [2, 3]]
        if protocol == "grpc":
            for worker in run.workers:
                argv = worker.args["extra_argv"]
                timeout_index = argv.index("--engine-ready-timeout-secs")
                assert argv[timeout_index + 1] == "60"
                assert timeout_index < argv.index("--")
        assert result.cache_block_size == 16
        layouts = json.loads(run.layout_path.read_text())
        assert len(layouts) == (1 if deployment == "1p1d_nixl" else 2)
        router_args = next(event[1] for event in events if event[0] == "router")
        assert (router_args["worker_urls"] is None) == (deployment == "1p1d_nixl")
    finally:
        stack.close()
    assert [event[1] for event in events if event[0] == "stop"] == ["router", 1, 0]


@pytest.mark.parametrize("protocol", ["http", "grpc"])
@pytest.mark.parametrize("failed_worker", [0, 1])
def test_pd_health_failure_cleans_up_both_workers(
    worker_startup, protocol, failed_worker
):
    run = worker_startup
    run.state.fail_worker = failed_worker
    stack = run.start(protocol, "1p1d_nixl")
    with pytest.raises(pytest.fail.Exception, match="Worker log: .*worker_"):
        next(stack)
    assert [event[1] for event in run.events if event[0] == "start"] == [0, 1]
    assert [event[1] for event in run.events if event[0] == "stop"] == [1, 0]
    assert not any(event[0] == "router" for event in run.events)
