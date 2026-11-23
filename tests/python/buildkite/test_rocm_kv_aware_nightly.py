"""Early setup failures must not publish evidence from a previous attempt."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

pytestmark = [pytest.mark.ready, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[3]


def test_setup_failure_removes_stale_worker_logs(tmp_path):
    bash = os.environ.get("TEST_BASH") or shutil.which("bash")
    if not bash:
        pytest.skip("bash is required to execute the pipeline command")
    pipeline = yaml.safe_load(
        (ROOT / ".buildkite/test-nightly.yml").read_text(encoding="utf-8")
    )
    step = next(
        s
        for s in pipeline["steps"]
        if isinstance(s, dict) and s.get("key") == "rocm-kv-aware-e2e"
    )
    # Relocate only the artifact root. Fail at the first external setup command.
    command = (
        step["command"]
        .replace("/workdir/target/ci-logs/kv-aware-rocm", tmp_path.as_posix())
        .replace("$$", "$")
    )
    stale = tmp_path / "kv_aware_worker_logs_old" / "worker.log"
    stale.parent.mkdir()
    stale.write_text("previous success", encoding="utf-8")
    fault = "timeout() { echo setup-started; return 42; }; export -f timeout\n"
    result = subprocess.run(
        [bash, "-c", fault + command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 42, result.stderr
    assert result.stdout.strip() == "setup-started"
    assert not stale.exists()
