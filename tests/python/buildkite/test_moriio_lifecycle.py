"""The read workaround must not change the write or non-probe legs."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

pytestmark = [pytest.mark.ready, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "read_mode,probe,override,expected",
    [
        ("true", "true", None, "0"),
        ("false", "true", None, "unset"),
        ("true", "false", None, "unset"),
        ("true", "true", "1", "1"),
        ("false", "true", "1", "1"),
    ],
)
def test_effective_skinny_setting(read_mode, probe, override, expected):
    bash = os.environ.get("TEST_BASH") or shutil.which("bash")
    if not bash:
        pytest.skip("bash is required to execute the pipeline command")
    pipeline = yaml.safe_load(
        (ROOT / ".buildkite/test-nightly.yml").read_text(encoding="utf-8")
    )
    step = next(
        s
        for s in pipeline["steps"]
        if isinstance(s, dict) and s.get("key") == "pd-disagg-moriio-rdma-test"
    )
    command = (
        step["command"].replace("{{matrix.read_mode}}", read_mode).replace("$$", "$")
    )
    # Stub the two external commands; execute the actual matrix environment logic.
    capture = """bash() {
      case "$1" in
        .buildkite/scripts/install_packages.sh) return 0 ;;
        tests/python/e2e/pd_disagg_vllm/run_moriio_rdma_accuracy_test.sh)
          printf '%s' "${VLLM_ROCM_USE_SKINNY_GEMM-unset}" ;;
        *) return 99 ;;
      esac
    }
    """
    env = dict(os.environ, RUN_PD_LIFECYCLE=probe)
    env.pop("VLLM_ROCM_USE_SKINNY_GEMM", None)
    if override is not None:
        env["VLLM_ROCM_USE_SKINNY_GEMM"] = override
    result = subprocess.run(
        [bash, "-c", capture + command],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected
