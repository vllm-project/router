import io
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import Mock

import pytest
import requests

from ..helpers import router_manager
from ..helpers.ports import find_free_port

pytestmark = [pytest.mark.ready, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("port", "allocated_ports"),
    [
        pytest.param(None, [53000, 53000, 53000, 53001], id="automatic"),
        pytest.param(53000, [53000, 53000, 53001], id="explicit"),
    ],
)
def test_start_router_selects_distinct_prometheus_port(
    monkeypatch, port, allocated_ports
):
    manager = router_manager.RouterManager()
    popen = Mock()
    popen.return_value.stdout = io.StringIO()
    wait_health = Mock()
    monkeypatch.setattr(
        router_manager, "reserve_port", Mock(side_effect=allocated_ports)
    )
    monkeypatch.setattr(router_manager.subprocess, "Popen", popen)
    monkeypatch.setattr(manager, "_wait_health", wait_health)

    timeout = 12.0
    handle = manager.start_router(port=port, router_bin="test-router", timeout=timeout)

    popen.assert_called_once()
    cmd = popen.call_args.args[0]
    assert cmd[cmd.index("--port") + 1] == "53000"
    assert cmd[cmd.index("--prometheus-port") + 1] == "53001"
    assert handle.process is popen.return_value
    assert handle.url == "http://127.0.0.1:53000"
    assert handle.prometheus_port == 53001
    wait_health.assert_called_once_with(handle.url, timeout, proc=popen.return_value)


def _patch_start(monkeypatch, manager, allocated_ports, health, outputs):
    procs = iter(Mock(returncode=1, stdout=io.StringIO(out)) for out in outputs)
    cmds = []

    def popen(cmd, **kwargs):
        # start_router updates cmd in place on retry, so record a copy.
        cmds.append(list(cmd))
        return next(procs)

    monkeypatch.setattr(
        router_manager, "reserve_port", Mock(side_effect=allocated_ports)
    )
    monkeypatch.setattr(router_manager.subprocess, "Popen", popen)
    monkeypatch.setattr(manager, "_wait_health", Mock(side_effect=health))
    return cmds


def _ports(cmd):
    return cmd[cmd.index("--port") + 1], cmd[cmd.index("--prometheus-port") + 1]


CLASH = "Error: Os { code: 98, kind: AddrInUse, message: ... }\n"


def test_start_router_gives_up_after_three_clashes(monkeypatch):
    manager = router_manager.RouterManager()
    cmds = _patch_start(
        monkeypatch,
        manager,
        range(53000, 53006),
        [router_manager._RouterExited] * 3,
        [CLASH] * 3,
    )

    with pytest.raises(RuntimeError, match="port_clash=True, attempt 3/3"):
        manager.start_router(router_bin="test-router")

    assert len(cmds) == 3
    assert manager._children == []


def test_start_router_retry_keeps_explicit_port(monkeypatch):
    manager = router_manager.RouterManager()
    cmds = _patch_start(
        monkeypatch,
        manager,
        [53001, 53002],
        [router_manager._RouterExited, None],
        [CLASH, ""],
    )

    handle = manager.start_router(port=53000, router_bin="test-router")

    assert _ports(cmds[0]) == ("53000", "53001")
    assert _ports(cmds[1]) == ("53000", "53002")
    assert handle.url == "http://127.0.0.1:53000"
    assert handle.prometheus_port == 53002


def test_wait_health_returns_early_when_process_exits():
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"])
    manager = router_manager.RouterManager()

    start = time.monotonic()
    with pytest.raises(router_manager._RouterExited):
        manager._wait_health(f"http://127.0.0.1:{find_free_port()}", 30, proc=proc)

    assert time.monotonic() - start < 5
    assert proc.returncode == 3


FAKE_ROUTER = """\
#!{python}
import argparse
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int)
parser.add_argument("--prometheus-port", type=int)
args, _ = parser.parse_known_args()
{body}
"""

SERVE = """
class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def log_message(self, *args):
        pass

HTTPServer.allow_reuse_address = False
try:
    prom = socket.create_server(("127.0.0.1", args.prometheus_port), reuse_port=False)
    server = HTTPServer(("127.0.0.1", args.port), Health)
except OSError:
    # Exit like the real router: a moment after startup, with its bind error.
    time.sleep(0.5)
    print('Error: Os { code: 98, kind: AddrInUse, message: "Address already in use" }')
    sys.exit(1)
server.serve_forever()
"""


def _fake_router(tmp_path, body):
    path = tmp_path / "fake_router.py"
    path.write_text(FAKE_ROUTER.format(python=sys.executable, body=body))
    path.chmod(0o755)
    return str(path)


def test_start_router_retries_when_reserved_port_is_taken(monkeypatch, tmp_path):
    release = router_manager.release_all_port_holders
    taken = []

    def release_then_take_api_port(holders):
        port = holders[0].getsockname()[1] if holders else None
        release(holders)
        if port and not taken:
            # Another process grabs the API port between release and Popen.
            taken.append(socket.create_server(("127.0.0.1", port)))

    monkeypatch.setattr(
        router_manager, "release_all_port_holders", release_then_take_api_port
    )
    manager = router_manager.RouterManager()
    try:
        handle = manager.start_router(
            router_bin=_fake_router(tmp_path, SERVE), timeout=10
        )
        taken_port = taken[0].getsockname()[1]
        assert handle.url != f"http://127.0.0.1:{taken_port}"
        assert requests.get(f"{handle.url}/health", timeout=2).status_code == 200
        assert manager._children == [handle.process]
    finally:
        manager.stop_all()
        for sock in taken:
            sock.close()


def test_start_router_raises_on_crash_without_port_clash(tmp_path):
    body = 'print("boom: config is broken", flush=True)\nsys.exit(7)'
    manager = router_manager.RouterManager()
    try:
        with pytest.raises(RuntimeError) as exc:
            manager.start_router(router_bin=_fake_router(tmp_path, body), timeout=10)
        message = str(exc.value)
        assert "code 7" in message
        assert "attempt 1/3" in message
        assert "boom: config is broken" in message
        assert manager._children == []
    finally:
        manager.stop_all()


class _Healthy(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.mark.skipif(sys.platform != "linux", reason="listener ownership uses /proc")
def test_start_router_ignores_health_from_port_squatter(monkeypatch, tmp_path):
    release = router_manager.release_all_port_holders
    squatters = []

    def release_then_squat_api_port(holders):
        port = holders[0].getsockname()[1] if holders else None
        release(holders)
        if port and not squatters:
            # Another router wins the API port and answers /health with 200.
            squatter = HTTPServer(("127.0.0.1", port), _Healthy)
            threading.Thread(target=squatter.serve_forever, daemon=True).start()
            squatters.append(squatter)

    popen = subprocess.Popen
    procs = []

    def record_popen(*args, **kwargs):
        procs.append(popen(*args, **kwargs))
        return procs[-1]

    monkeypatch.setattr(
        router_manager, "release_all_port_holders", release_then_squat_api_port
    )
    monkeypatch.setattr(router_manager.subprocess, "Popen", record_popen)
    manager = router_manager.RouterManager()
    try:
        handle = manager.start_router(
            router_bin=_fake_router(tmp_path, SERVE), timeout=10
        )
        port = int(handle.url.rsplit(":", 1)[1])
        assert len(procs) == 2
        assert procs[0].returncode == 1
        assert handle.process is procs[1]
        assert port != squatters[0].server_address[1]
        assert router_manager._owns_listener(handle.process.pid, port) is True
    finally:
        manager.stop_all()
        for squatter in squatters:
            squatter.shutdown()
            squatter.server_close()
