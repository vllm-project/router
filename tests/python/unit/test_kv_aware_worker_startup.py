"""Check hybrid worker startup and cleanup without loading GPU models."""

import json
from contextlib import contextmanager
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
    state = SimpleNamespace(fail_worker=None, gpus=[0, 1, 2, 3], ports_reserved=False)
    case = {
        "model": "startup-test",
        "local_model_env": "KV_AWARE_STARTUP_TEST_MODEL",
        "worker_count": 2,
        "worker_health_timeout_sec": 60,
        "worker_env": {"OMP_NUM_THREADS": "1"},
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
        assert state.ports_reserved
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
        assert {event[1] for event in events if event[0] == "health"} == set(
            range(len(workers))
        )
        events.append(("router", kwargs))
        return SimpleNamespace(index="router")

    monkeypatch.setattr(config, "kv_aware_gpu_pool", lambda: state.gpus)
    monkeypatch.setattr(kv_fixtures.shutil, "which", lambda _: "/test/vllm")
    monkeypatch.setattr(
        kv_fixtures, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )

    @contextmanager
    def reserve_ports(count):
        state.ports_reserved = True
        try:
            yield list(range(20000, 20000 + count))
        finally:
            state.ports_reserved = False

    monkeypatch.setattr(launch, "reserve_available_ports", reserve_ports)
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
        case=case,
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
        assert all(
            worker.args["extra_env"]["OMP_NUM_THREADS"] == "1" for worker in run.workers
        )
        assert run.case["worker_env"] == {"OMP_NUM_THREADS": "1"}
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
    assert not run.state.ports_reserved


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
    assert not run.state.ports_reserved


@pytest.mark.parametrize("protocol", ["http", "grpc"])
def test_single_merge_worker_uses_all_eight_gpus(worker_startup, protocol):
    run = worker_startup
    run.case["worker_count"] = 1
    run.case["worker_health_timeout_sec"] = 3600
    run.case["worker_params"]["tensor_parallel_size"] = 8
    run.state.gpus = list(range(8))
    stack = run.start(protocol, "merge")
    try:
        result = next(stack)
        assert len(run.workers) == len(result.worker_http_urls) == 1
        gpu_key = "gpu_id" if protocol == "http" else "base_gpu_id"
        assert run.workers[0].args[gpu_key] == list(range(8))
        health = [event for event in run.events if event[0] == "health"]
        assert health == [("health", 0, 3600)]
        router = next(event[1] for event in run.events if event[0] == "router")
        assert len(router["worker_urls"]) == 1
        assert len(json.loads(run.layout_path.read_text())) == 1
        if protocol == "grpc":
            argv = run.workers[0].args["extra_argv"]
            index = argv.index("--engine-ready-timeout-secs")
            assert argv[index + 1] == "3600"
            assert index < argv.index("--")
    finally:
        stack.close()
    assert [event[1] for event in run.events if event[0] == "stop"] == ["router", 0]


@pytest.mark.parametrize("protocol", ["http", "grpc"])
def test_single_merge_health_failure_cleans_up_worker(worker_startup, protocol):
    run = worker_startup
    run.case["worker_count"] = 1
    run.state.fail_worker = 0
    stack = run.start(protocol, "merge")
    with pytest.raises(pytest.fail.Exception, match="Worker log: .*worker_0"):
        next(stack)
    assert [event[1] for event in run.events if event[0] == "stop"] == [0]
    assert not any(event[0] == "router" for event in run.events)


@pytest.mark.parametrize("protocol", ["http", "grpc"])
def test_pd_skips_before_launch_when_sixteen_gpus_are_required(
    worker_startup, protocol
):
    run = worker_startup
    run.case["worker_params"]["tensor_parallel_size"] = 8
    run.state.gpus = list(range(8))
    stack = run.start(protocol, "1p1d_nixl")
    with pytest.raises(pytest.skip.Exception, match="requires 16 GPUs"):
        next(stack)
    assert not run.events
    assert not run.workers


@pytest.mark.parametrize("protocol", ["http", "grpc"])
def test_pd_rejects_single_worker_configuration(worker_startup, protocol):
    run = worker_startup
    run.case["worker_count"] = 1
    stack = run.start(protocol, "1p1d_nixl")
    with pytest.raises(AssertionError, match="1P1D requires exactly two workers"):
        next(stack)
    assert not run.workers


def test_single_gpu_pool_preserves_dual_lane_requirement(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert config.kv_aware_gpu_pool() == [3]
    with pytest.raises(RuntimeError, match="dual-lane case requires at least two GPUs"):
        config.kv_aware_worker_gpu_ids()
