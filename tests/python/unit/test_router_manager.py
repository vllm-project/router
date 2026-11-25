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
    wait_health.assert_called_once_with(handle.url, timeout, proc=popen.return_value)


def _patch_start(monkeypatch, manager, allocated_ports, health, port_in_use):
    procs = iter([Mock(returncode=1) for _ in range(3)])
    cmds = []

    def popen(cmd, **kwargs):
        # start_router updates cmd in place on retry, so record a copy.
        cmds.append(list(cmd))
        return next(procs)

    monkeypatch.setattr(
        router_manager, "find_free_port", Mock(side_effect=allocated_ports)
    )
    monkeypatch.setattr(router_manager, "_port_in_use", Mock(side_effect=port_in_use))
    monkeypatch.setattr(router_manager.subprocess, "Popen", popen)
    monkeypatch.setattr(manager, "_wait_health", Mock(side_effect=health))
    return cmds


def _ports(cmd):
    return cmd[cmd.index("--port") + 1], cmd[cmd.index("--prometheus-port") + 1]


def test_start_router_retries_after_port_clash(monkeypatch):
    manager = router_manager.RouterManager()
    cmds = _patch_start(
        monkeypatch,
        manager,
        [53000, 53001, 53002, 53003],
        [router_manager._RouterExited, None],
        lambda port: port == 53001,
    )

    handle = manager.start_router(router_bin="test-router")

    assert len(cmds) == 2
    assert _ports(cmds[0]) == ("53000", "53001")
    assert _ports(cmds[1]) == ("53002", "53003")
    assert handle.url == "http://127.0.0.1:53002"
    assert handle.prometheus_port == 53003
    assert manager._children == [handle.process]


def test_start_router_early_exit_without_clash_raises(monkeypatch):
    manager = router_manager.RouterManager()
    cmds = _patch_start(
        monkeypatch,
        manager,
        [53000, 53001],
        [router_manager._RouterExited],
        lambda port: False,
    )

    with pytest.raises(RuntimeError, match="code 1.*attempt 1/3"):
        manager.start_router(router_bin="test-router")

    assert len(cmds) == 1
    assert manager._children == []


def test_start_router_gives_up_after_three_clashes(monkeypatch):
    manager = router_manager.RouterManager()
    cmds = _patch_start(
        monkeypatch,
        manager,
        range(53000, 53006),
        [router_manager._RouterExited] * 3,
        lambda port: True,
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
        lambda port: port == 53001,
    )

    handle = manager.start_router(port=53000, router_bin="test-router")

    assert _ports(cmds[0]) == ("53000", "53001")
    assert _ports(cmds[1]) == ("53000", "53002")
    assert handle.url == "http://127.0.0.1:53000"
    assert handle.prometheus_port == 53002
