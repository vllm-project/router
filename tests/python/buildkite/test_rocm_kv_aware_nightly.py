"""Early setup failures must not publish evidence from a previous attempt."""

import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

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


@pytest.mark.parametrize(
    "hip,count", [(None, 2), ("test-rocm", 0), ("test-rocm", 1), ("test-rocm", 2)]
)
def test_gpu_preflight_prevents_skipped_success(tmp_path, hip, count):
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
    command = (
        step["command"]
        .replace(
            "/workdir/target/ci-logs/kv-aware-rocm", (tmp_path / "evidence").as_posix()
        )
        .replace("$$", "$")
    )
    # Simulate device discovery only; execute the real job preflight and shell.
    (tmp_path / "torch.py").write_text(
        "import os\nfrom types import SimpleNamespace\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'\n"
        "assert 'HIP_VISIBLE_DEVICES' not in os.environ\n"
        "assert 'ROCR_VISIBLE_DEVICES' not in os.environ\n"
        f"version = SimpleNamespace(hip={hip!r})\n"
        f"cuda = SimpleNamespace(device_count=lambda: {count})\n",
        encoding="utf-8",
    )
    shim = f'python3() {{ {shlex.quote(Path(sys.executable).as_posix())} "$@"; }}\n'
    shim += """timeout() {
      case "$3" in
        600) return 0 ;;
        1800) echo PYTEST_REACHED; return 0 ;;
        *) shift 3; "$@" ;;
      esac
    }
    export -f python3 timeout
    """
    result = subprocess.run(
        [bash, "-c", shim + command],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PYTHONPATH=str(tmp_path),
            HIP_VISIBLE_DEVICES="7",
            ROCR_VISIBLE_DEVICES="7",
        ),
        capture_output=True,
        text=True,
        timeout=10,
    )
    if hip and count >= 2:
        assert result.returncode == 0, result.stderr
        assert "PYTEST_REACHED" in result.stdout
    else:
        assert result.returncode != 0
        assert "requires two visible AMD GPUs" in result.stderr
        assert "PYTEST_REACHED" not in result.stdout
