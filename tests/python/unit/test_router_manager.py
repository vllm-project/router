from unittest.mock import Mock

import pytest

from ..helpers import router_manager

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
    wait_health = Mock()
    monkeypatch.setattr(
        router_manager, "find_free_port", Mock(side_effect=allocated_ports)
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
    wait_health.assert_called_once_with(handle.url, timeout)
