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
    "read_mode,probe,override,expected,setup_exit",
    [
        ("true", "true", None, "0", 0),
        ("false", "true", None, "unset", 0),
        ("true", "false", None, "unset", 0),
        ("true", "true", "1", "1", 0),
        ("false", "true", "1", "1", 0),
        ("true", "true", None, "", 42),
    ],
)
def test_pipeline_settings_and_cleanup(
    tmp_path, read_mode, probe, override, expected, setup_exit
):
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
    reports = []
    server_logs = []
    for mode in ("true", "false"):
        log_dir = tmp_path / "target/ci-logs" / f"moriio-rdma-{mode}"
        log_dir.mkdir(parents=True)
        for name in ("pd-lifecycle.json", "pd-lifecycle.json.tmp"):
            report = log_dir / name
            report.write_text("stale report", encoding="utf-8")
            reports.append(report)
        log = log_dir / "prefill.log"
        log.write_text("server evidence", encoding="utf-8")
        server_logs.append(log)
    # Stub the two external commands; execute the actual matrix environment logic.
    capture = """bash() {
      case "$1" in
        .buildkite/scripts/install_packages.sh) return "$TEST_SETUP_EXIT" ;;
        tests/python/e2e/pd_disagg_vllm/run_moriio_rdma_accuracy_test.sh)
          printf '%s' "${VLLM_ROCM_USE_SKINNY_GEMM-unset}" ;;
        *) return 99 ;;
      esac
    }
    export -f bash
    """
    env = dict(os.environ, RUN_PD_LIFECYCLE=probe, TEST_SETUP_EXIT=str(setup_exit))
    env.pop("VLLM_ROCM_USE_SKINNY_GEMM", None)
    if override is not None:
        env["VLLM_ROCM_USE_SKINNY_GEMM"] = override
    result = subprocess.run(
        [bash, "-c", capture + command],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == setup_exit, result.stderr
    assert result.stdout == expected
    assert not any(p.exists() for p in reports)
    assert all(p.read_text(encoding="utf-8") == "server evidence" for p in server_logs)
