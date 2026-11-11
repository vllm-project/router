"""Invalid probe options must fail before contacting Docker or the worker."""

import os
import shutil
import subprocess

import pytest

pytestmark = [pytest.mark.ready, pytest.mark.cpu]
SCRIPT = "tests/python/e2e/pd_disagg_vllm/run_moriio_rdma_accuracy_test.sh"


@pytest.mark.parametrize(
    "options,message",
    [
        ({"RUN_PD_LIFECYCLE": "yes"}, "must be true or false"),
        ({"SMOKE_ONLY": "true"}, "requires SMOKE_ONLY=false"),
        ({"PD_PROBE_CONCURRENCY": "0"}, "space-separated integers"),
        ({"PD_PROBE_CONCURRENCY": "1; touch BAD"}, "space-separated integers"),
        ({"PD_PROBE_CONCURRENCY": "33"}, "must not exceed 32"),
        ({"PD_PROBE_CONCURRENCY": "1 1 1 1 1 1 1 1 1"}, "at most eight"),
        ({"PD_PROBE_ROUNDS": "0"}, "rounds must be"),
        ({"PD_PROBE_TIMEOUT": "601"}, "timeout 1..600"),
    ],
)
def test_invalid_options_do_not_touch_hosts(tmp_path, pytestconfig, options, message):
    bash = os.environ.get("TEST_BASH") or shutil.which("bash")
    if not bash:
        pytest.skip("bash is required to test the shell entrypoint")
    marker = tmp_path / "host-command-called"
    for name in ("docker", "ssh", "curl"):
        command = tmp_path / name
        command.write_text('#!/bin/sh\nprintf called > "$PROBE_TEST_MARKER"\nexit 99\n')
        command.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "PROBE_TEST_MARKER": str(marker),
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
