"""Verify coverage artifact publication without compiling Rust or downloading tools."""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.ready, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[3]
OLD_REPORT = "previous run\n"
NEW_REPORT = "TN:\nSF:src/lib.rs\nDA:1,1\nLF:1\nLH:1\nend_of_record\n"


@pytest.fixture
def coverage_runner(tmp_path):
    bash = shutil.which("bash")
    if os.name == "nt":
        git = shutil.which("git")
        candidate = Path(git).resolve().parents[1] / "bin/bash.exe" if git else None
        bash = str(candidate) if candidate and candidate.is_file() else None
    if not bash:
        pytest.skip("Bash (Git Bash on Windows) is required")

    scripts = tmp_path / ".buildkite/scripts"
    scripts.mkdir(parents=True)
    source = (ROOT / ".buildkite/scripts/rust_coverage.sh").read_bytes()
    (scripts / "rust_coverage.sh").write_bytes(source.replace(b"\r\n", b"\n"))
    tools = tmp_path / "bin"
    tools.mkdir()
    mocks = {
        "rustup": """
test ! -e "target/coverage/rust-${EXPECT_MODE}/lcov.info" || exit 99
exit "${SETUP_STATUS:-0}"
""",
        "rustc": "echo 'host: x86_64-unknown-linux-gnu'\n",
        "curl": 'exit "${DOWNLOAD_STATUS:-0}"\n',
        "tar": "cat > /dev/null\n",
        "cargo": """
case "$2" in
  --version) echo 'cargo-llvm-cov 0.6.21' ;;
  clean) exit "${CLEAN_STATUS:-0}" ;;
  report)
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --output-path ]; then output="$2"; break; fi
      shift
    done
    final="target/coverage/rust-${EXPECT_MODE}/lcov.info"
    test ! -e "$final" || exit 98
    test "$output" != "$final" || exit 97
    printf 'incomplete report' > "$output"
    if [ "${REPORT_STATUS:-0}" -ne 0 ]; then exit "$REPORT_STATUS"; fi
    printf '%s' "$NEW_REPORT" > "$output"
    ;;
  *) exit "${TEST_STATUS:-0}" ;;
esac
""",
    }
    for name, body in mocks.items():
        path = tools / name
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + body, newline="\n")
        path.chmod(0o755)

    def run(mode, **statuses):
        reports = {}
        for group in ("unit", "integration"):
            path = tmp_path / f"target/coverage/rust-{group}/lcov.info"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(OLD_REPORT)
            reports[group] = path
        env = os.environ.copy()
        env.update(
            CARGO_HOME=(tmp_path / "cargo").as_posix(),
            EXPECT_MODE=mode,
            NEW_REPORT=NEW_REPORT,
            **{name: str(status) for name, status in statuses.items()},
        )
        result = subprocess.run(
            [
                bash,
                "-c",
                'export PATH="$PWD/bin:$PATH"; bash .buildkite/scripts/rust_coverage.sh "$1"',
                "coverage-test",
                mode,
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return result, reports

    return run


@pytest.mark.parametrize("mode", ["unit", "integration"])
@pytest.mark.parametrize(
    ("statuses", "expected_status", "publishes_report"),
    [
        ({"SETUP_STATUS": 42}, 42, False),
        ({"DOWNLOAD_STATUS": 28}, 28, False),
        ({"CLEAN_STATUS": 9}, 9, False),
        ({"REPORT_STATUS": 7}, 7, False),
        ({}, 0, True),
        ({"TEST_STATUS": 101}, 101, True),
    ],
)
def test_only_complete_current_reports_are_published(
    coverage_runner, mode, statuses, expected_status, publishes_report
):
    result, reports = coverage_runner(mode, **statuses)
    assert result.returncode == expected_status, result.stdout + result.stderr
    report = reports[mode]
    if publishes_report:
        assert report.read_text() == NEW_REPORT
        if os.name == "posix":
            # The host Buildkite agent may have a different UID from Docker.
            assert stat.S_IMODE(report.stat().st_mode) == 0o644
    else:
        assert not report.exists()
    assert list(report.parent.glob(".lcov.info.*")) == []
    other = "integration" if mode == "unit" else "unit"
    assert reports[other].read_text() == OLD_REPORT


def test_invalid_mode_does_not_remove_reports(coverage_runner):
    result, reports = coverage_runner("invalid")
    assert result.returncode == 2
    assert all(path.read_text() == OLD_REPORT for path in reports.values())
