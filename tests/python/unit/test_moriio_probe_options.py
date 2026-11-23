"""Preflight failures must not contact hosts or retain stale probe results."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = [pytest.mark.ready, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = "tests/python/e2e/pd_disagg_vllm/run_moriio_rdma_accuracy_test.sh"


@pytest.mark.parametrize(
    "options,message",
    [
        ({"RUN_PD_LIFECYCLE": "yes"}, "must be true or false"),
        ({"SMOKE_ONLY": "true"}, "requires SMOKE_ONLY=false"),
        (
            {
                "RUN_PD_LIFECYCLE": "false",
                "PD_PROBE_CONCURRENCY": "1; touch BAD",
                "PD_PROBE_ROUNDS": "not-a-number",
                "PD_PROBE_TIMEOUT": "-1",
                "ROUTER_BIN": "/nonexistent-router",
            },
            "ROUTER_BIN is not executable",
        ),
    ],
)
def test_invalid_options_do_not_touch_hosts(tmp_path, pytestconfig, options, message):
    bash = os.environ.get("TEST_BASH") or shutil.which("bash")
    if not bash:
        pytest.skip("bash is required to test the shell entrypoint")
    marker = tmp_path / "host-command-called"
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    report = log_dir / "pd-lifecycle.json"
    checkpoint = log_dir / "pd-lifecycle.json.tmp"
    report.write_text('{"status":"passed","passed":true}')
    checkpoint.write_text('{"status":"running","passed":false}')
    unrelated = log_dir / "prefill.log"
    unrelated.write_text("previous server log")
    for name in ("docker", "ssh", "curl"):
        command = tmp_path / name
        command.write_text('#!/bin/sh\nprintf called > "$PROBE_TEST_MARKER"\nexit 99\n')
        command.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "PROBE_TEST_MARKER": str(marker),
        "LOG_DIR": str(log_dir),
        "RUN_PD_LIFECYCLE": "true",
        "SMOKE_ONLY": "false",
        "PD_PROBE_CONCURRENCY": "1 2",
        "PD_PROBE_ROUNDS": "1",
        "PD_PROBE_TIMEOUT": "60",
        **options,
    }
    result = subprocess.run(
        [bash, SCRIPT],
        cwd=pytestconfig.rootpath,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2, result.stderr
    assert message in result.stderr
    assert not marker.exists()
    assert not report.exists()
    assert not checkpoint.exists()
    assert "Removing stale lifecycle report:" in result.stdout
    assert unrelated.read_text() == "previous server log"
