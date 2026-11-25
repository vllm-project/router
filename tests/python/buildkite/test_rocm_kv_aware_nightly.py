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


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
@pytest.mark.parametrize(
    "available,count", [(True, 0), (True, 1), (False, 2), (True, 2)]
)
def test_gpu_preflight_prevents_skipped_success(tmp_path, platform, available, count):
    bash = os.environ.get("TEST_BASH") or shutil.which("bash")
    if not bash:
        pytest.skip("bash is required to execute the pipeline command")
    lane = "test-nightly.yml" if platform == "rocm" else "test-ready.yml"
    key = "rocm-kv-aware-e2e" if platform == "rocm" else "kv-aware-e2e"
    pipeline = yaml.safe_load((ROOT / ".buildkite" / lane).read_text(encoding="utf-8"))
    step = next(
        s for s in pipeline["steps"] if isinstance(s, dict) and s.get("key") == key
    )
    # Keep the real CLI, session hook and hardware marks; replace model work only.
    probe = tmp_path / "test_probe.py"
    probe.write_text(
        "import pytest\nfrom tests.python.helpers.mark import hardware_marks\n"
        "pytestmark = [pytest.mark.e2e, *hardware_marks(res={'cuda': 'L4', 'rocm': 'MI300'}, num_cards=2)]\n"
        "def test_probe(): pass\n",
        encoding="utf-8",
    )
    (tmp_path / "torch.py").write_text(
        "from types import SimpleNamespace\n"
        f"version = SimpleNamespace(hip={'test-rocm' if platform == 'rocm' else None!r})\n"
        f"cuda = SimpleNamespace(is_available=lambda: {available}, device_count=lambda: {count})\n",
        encoding="utf-8",
    )
    command = (
        step["command"]
        .replace(
            "/workdir/target/ci-logs/kv-aware-rocm", (tmp_path / "evidence").as_posix()
        )
        .replace(
            "tests/python/e2e/function/kv_aware/test_kv_aware_prefix_cache.py",
            shlex.quote(probe.as_posix()),
        )
        .replace(
            "--confcutdir=tests/python/e2e/function/kv_aware",
            f"--confcutdir={shlex.quote(tmp_path.as_posix())} -p tests.python.e2e.function.kv_aware.conftest",
        )
        .replace("$$", "$")
    )
    shim = f'python3() {{ {shlex.quote(Path(sys.executable).as_posix())} "$@"; }}\n'
    shim += """pytest() { python3 -m pytest "$@"; }
    timeout() {
      case "$3" in
        600) return 0 ;;
        *) shift 3; "$@" ;;
      esac
    }
    export -f python3 pytest timeout
    """
    result = subprocess.run(
        [bash, "-c", shim + command],
        cwd=ROOT,
        env=dict(
            os.environ,
            PYTHONPATH=os.pathsep.join((str(tmp_path), str(ROOT))),
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        ),
        capture_output=True,
        text=True,
        timeout=20,
    )
    if available and count >= 2:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1 passed" in result.stdout
    else:
        assert result.returncode != 0, result.stdout
        assert "requires two available" in result.stderr
