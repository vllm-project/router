"""CPU checks for shared batch readiness and multi-worker startup ownership."""

import io
import subprocess
import sys
import time
import urllib.request
from types import SimpleNamespace

import pytest

from tests.python.e2e.function.helpers.workers import (
    spawn_two_vllm_rs_grpc_workers,
    terminate_workers,
)
from tests.python.helpers.process import (
    launch_process,
    launch_worker,
    wait_workers_ready,
)
from tests.python.helpers.worker_manager import MockWorkerManager

pytestmark = [pytest.mark.ready, pytest.mark.cpu]


def test_readiness_uses_one_deadline_for_the_whole_group(monkeypatch):
    clock = SimpleNamespace(now=0.0)

    def sleep(seconds):
        clock.now += seconds

    class Response:
        def __init__(self, url, **kwargs):
            ready = url.endswith("first/health") and clock.now >= 0.5
            self.status = 200 if ready else 503

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(urllib.request, "urlopen", Response)
    process = SimpleNamespace(returncode=None, poll=lambda: None)
    with pytest.raises(TimeoutError, match="second"):
        wait_workers_ready(
            [
                ("http://test/first/health", process, "first"),
                ("http://test/second/health", process, "second"),
            ],
            1,
            prefix="test",
            interval=0.25,
        )
    assert clock.now == 1


def test_readiness_detects_later_worker_exit_before_waiting(monkeypatch):
    def unexpected_probe(*args, **kwargs):
        pytest.fail("a worker already exited; do not wait for another worker's HTTP")

    monkeypatch.setattr(urllib.request, "urlopen", unexpected_probe)
    healthy = SimpleNamespace(returncode=None, poll=lambda: None)
    exited = SimpleNamespace(returncode=7, poll=lambda: 7)
    with pytest.raises(RuntimeError, match="second exited early.*7"):
        wait_workers_ready(
            [
                ("http://test/first/health", healthy, "first"),
                ("http://test/second/health", exited, "second"),
            ],
            600,
            prefix="test",
        )


@pytest.fixture
def child_processes(monkeypatch):
    children = []

    class Child:
        fail_at = None
        exit_at = None

        def __init__(self, command, **kwargs):
            index = len(children)
            if index == self.fail_at:
                raise OSError("spawn failed")
            self.stdout = io.StringIO("")
            self.returncode = 7 if index == self.exit_at else None
            self.env = kwargs.get("env")
            self.start_new_session = kwargs.get("start_new_session", False)
            self.terminations = 0
            children.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminations += 1
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(subprocess, "Popen", Child)
    try:
        yield children
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()


def test_mock_manager_starts_the_whole_batch_before_health(
    child_processes, monkeypatch
):
    def health(url, **kwargs):
        assert len(child_processes) == 3, "mock workers were started serially"
        response = io.BytesIO()
        response.status = 200
        return response

    monkeypatch.setattr(urllib.request, "urlopen", health)
    manager = MockWorkerManager()
    try:
        processes, urls, ids = manager.start_workers(n=3)
        assert len(processes) == len(urls) == len(ids) == 3
        assert len(set(urls)) == 3
        assert all(process.poll() is None for process in processes)
    finally:
        manager.stop_all()
    assert all(child.poll() is not None for child in child_processes)


def test_single_worker_launch_still_waits_for_health(child_processes, monkeypatch):
    checked = []

    def health(url, **kwargs):
        assert len(child_processes) == 1
        checked.append(url)
        response = io.BytesIO()
        response.status = 200
        return response

    monkeypatch.setattr(urllib.request, "urlopen", health)
    running = launch_worker([sys.executable, "worker"], ready_url="http://test/health")
    try:
        assert checked == ["http://test/health"]
        assert running.alive()
    finally:
        running.stop()


@pytest.mark.parametrize("new_session", [False, True])
def test_logged_process_inherits_environment_and_applies_overrides(
    child_processes, tmp_path, monkeypatch, new_session
):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "model-cache"))
    monkeypatch.setenv("PROCESS_ENV_TEST", "parent")
    running = launch_process(
        [sys.executable, "replay"],
        label="replay",
        log_path=tmp_path / "replay.log",
        env={"PROCESS_ENV_TEST": "case"},
        start_new_session=new_session,
    )
    try:
        assert running.proc.env["HF_HOME"] == str(tmp_path / "model-cache")
        assert running.proc.env["PROCESS_ENV_TEST"] == "case"
        assert running.proc.start_new_session is new_session
        assert (tmp_path / "replay.log").is_file()
        assert running.alive()
    finally:
        running.stop()


def test_logged_process_stops_the_child_when_log_creation_fails(
    child_processes, tmp_path
):
    with pytest.raises(OSError):
        launch_process([sys.executable, "replay"], label="replay", log_path=tmp_path)
    assert len(child_processes) == 1
    assert child_processes[0].poll() is not None


def test_failed_mock_batch_stops_new_children_and_preserves_existing_workers(
    child_processes, monkeypatch
):
    def health(url, **kwargs):
        response = io.BytesIO()
        response.status = 200
        return response

    monkeypatch.setattr(urllib.request, "urlopen", health)
    manager = MockWorkerManager()
    try:
        existing, _, _ = manager.start_worker()
        subprocess.Popen.fail_at = 2
        with pytest.raises(OSError, match="spawn failed"):
            manager.start_workers(n=2)
        assert existing.poll() is None
        assert child_processes[1].poll() is not None
    finally:
        manager.stop_all()
    assert existing.terminations == child_processes[1].terminations == 1


@pytest.mark.parametrize("exit_index", [None, 1], ids=["healthy", "early_exit"])
def test_kv_aware_workers_start_together_and_cleanup_on_failure(
    child_processes, tmp_path, monkeypatch, exit_index
):
    def health(url, **kwargs):
        assert len(child_processes) == 2, "KV-aware workers were started serially"
        response = io.BytesIO()
        response.status = 200
        return response

    monkeypatch.setenv("VLLM_RS", sys.executable)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("KV_AWARE_USE_VLLM_SERVE", "0")
    monkeypatch.setattr(urllib.request, "urlopen", health)
    subprocess.Popen.exit_at = exit_index
    if exit_index is not None:
        with pytest.raises(RuntimeError, match="worker 1 exited early"):
            spawn_two_vllm_rs_grpc_workers("test/model", log_dir=tmp_path)
        assert len(child_processes) == 2
        assert all(child.poll() is not None for child in child_processes)
    else:
        workers = spawn_two_vllm_rs_grpc_workers("test/model", log_dir=tmp_path)
        try:
            assert len(workers) == 2
            assert all(worker.proc.poll() is None for worker in workers)
        finally:
            terminate_workers(workers)
