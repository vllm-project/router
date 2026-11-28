"""Port allocation for workers that bind after a long model startup."""

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.python.e2e.function.helpers import launch
from tests.python.e2e.function.helpers import workers as worker_helpers

pytestmark = [pytest.mark.ready, pytest.mark.cpu]


def test_function_ports_avoid_the_system_ephemeral_range():
    if sys.platform == "linux":
        low, high = map(
            int, Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split()
        )
    elif sys.platform == "win32":
        low, high = 49152, 65535
    else:
        pytest.skip("System ephemeral range is not known on this platform")
    ports = launch.find_distinct_available_ports(8)
    assert len(set(ports)) == 8
    assert all(not low <= port <= high for port in ports), ports


@pytest.fixture
def candidates(monkeypatch, tmp_path):
    ports = launch.find_distinct_available_ports(8)
    monkeypatch.setattr(launch, "_port_candidates", lambda: ports.copy())
    monkeypatch.setattr(launch.tempfile, "gettempdir", lambda: str(tmp_path))
    return ports


def test_delayed_ports_are_reserved_across_processes(candidates, tmp_path):
    child = """
import json, os, sys
from tests.python.e2e.function.helpers import launch
launch._port_candidates = lambda: json.loads(sys.argv[1])
launch.tempfile.gettempdir = lambda: sys.argv[2]
with launch.reserve_available_ports(2) as ports:
    print(json.dumps(ports), flush=True)
    os._exit(0)  # No Python cleanup: the OS must release the child's leases.
"""
    with launch.reserve_available_ports(2) as first:
        assert first == candidates[:2]
        # Neither process binds TCP yet, just as during slow weight loading.
        output = subprocess.check_output(
            [sys.executable, "-c", child, json.dumps(candidates), str(tmp_path)],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            timeout=20,
        )
        second = json.loads(output)
        assert second == candidates[2:4]
        with launch.reserve_available_ports(2) as after_child_exit:
            assert after_child_exit == second
    with launch.reserve_available_ports(2) as released:
        assert released == first


def test_occupied_ports_are_skipped(candidates):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("0.0.0.0", candidates[0]))
        occupied.listen()
        with launch.reserve_available_ports(1) as ports:
            assert ports == candidates[1:2]


def test_failed_allocation_releases_partial_reservations(candidates, monkeypatch):
    monkeypatch.setattr(launch, "_port_candidates", lambda: candidates[:1])
    with pytest.raises(RuntimeError, match="Could not reserve 2"):
        with launch.reserve_available_ports(2):
            pytest.fail("Only one candidate is available")
    with launch.reserve_available_ports(1) as ports:
        assert ports == candidates[:1]


@pytest.mark.parametrize("count", [0, -1])
def test_port_allocation_rejects_invalid_counts(count):
    with pytest.raises(ValueError, match="count must be >= 1"):
        with launch.reserve_available_ports(count):
            pytest.fail("Invalid port count was accepted")


def test_delayed_zmq_binds_do_not_collide(candidates):
    zmq = pytest.importorskip("zmq")
    with launch.reserve_available_ports(2) as first:
        with launch.reserve_available_ports(2) as second:
            context = zmq.Context()
            sockets = []
            try:
                for live, replay in (first, second):
                    publisher = context.socket(zmq.XPUB)
                    sockets.append(publisher)
                    publisher.bind(f"tcp://*:{live}")
                    history = context.socket(zmq.ROUTER)
                    sockets.append(history)
                    history.bind(f"tcp://127.0.0.1:{replay}")
            finally:
                for connection in sockets:
                    connection.close(linger=0)
                context.term()


@pytest.mark.parametrize("failure", [None, "first_spawn", "second_health"])
def test_dual_lane_keeps_port_leases_until_workers_stop(
    candidates, monkeypatch, tmp_path, failure
):
    spawned = []
    stopped = []
    monkeypatch.setattr(worker_helpers, "_require_worker_launch", lambda: None)
    monkeypatch.setattr(worker_helpers, "kv_aware_worker_gpu_ids", lambda: (0, 1))
    monkeypatch.setattr(worker_helpers, "kv_aware_kv_events_enabled", lambda: True)
    monkeypatch.setattr(
        worker_helpers, "kv_aware_kv_events_replay_enabled", lambda: True
    )
    monkeypatch.setattr(worker_helpers, "worker_serve_extra_argv", lambda: [])
    monkeypatch.setattr(worker_helpers, "SubprocessLogPump", lambda *a, **kw: None)

    def spawn(*args, **kwargs):
        if failure == "first_spawn":
            raise RuntimeError("first spawn failed")
        worker = object()
        spawned.append(worker)
        return worker

    def wait_health(url, timeout, proc):
        if failure == "second_health" and len(spawned) == 2 and proc is spawned[1]:
            raise RuntimeError("second health failed")

    monkeypatch.setattr(launch, "popen_launch_vllm_rs_grpc_worker", spawn)
    monkeypatch.setattr(launch, "wait_worker_health", wait_health)
    monkeypatch.setattr(launch, "terminate_process", stopped.append)

    if failure:
        with pytest.raises(RuntimeError, match="failed"):
            worker_helpers.spawn_two_vllm_rs_grpc_workers("test", log_dir=tmp_path)
    else:
        workers = worker_helpers.spawn_two_vllm_rs_grpc_workers(
            "test", log_dir=tmp_path
        )
        try:
            with pytest.raises(RuntimeError, match="Could not reserve 1"):
                with launch.reserve_available_ports(1):
                    pytest.fail("Live worker leases were released too early")
        finally:
            worker_helpers.terminate_workers(workers)
    assert stopped == spawned
    with launch.reserve_available_ports(8) as released:
        assert released == candidates
