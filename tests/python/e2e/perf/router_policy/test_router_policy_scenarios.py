"""Router policy perf: one pytest per self-contained case JSON in cases/<policy>/."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCENARIO_DIR = Path(__file__).resolve().parent
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))

from helpers.case_config import (  # noqa: E402
    cell_output_dir,
    cell_results_dir,
    load_case,
    require_runtime,
)

RUN_CELL = SCENARIO_DIR / "scripts" / "run_cell.py"


def _run_cell(case_file: Path) -> Path:
    case = load_case(case_file, scenario_dir=SCENARIO_DIR)
    runtime = require_runtime(case)
    out = cell_output_dir(case)
    env = os.environ.copy()
    env["AGENTX_CASE_FILE"] = str(case_file.resolve())
    env["AGENTX_OUT"] = str(out)
    timeout = int(env.get("AGENTX_PERF_TIMEOUT_SEC", str(4 * 3600)))
    proc = subprocess.run(
        [runtime["python"], "-u", str(RUN_CELL)],
        cwd=SCENARIO_DIR,
        env=env,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        print(
            f"run_cell exit {proc.returncode}; see ARTIFACTS / LOG_TAIL above "
            f"or logs under {out.resolve()}",
            flush=True,
        )
        assert proc.returncode == 0, f"run_cell failed for {case_file.name}"
    cal_path = out / "calibration.json"
    summary: Path | None = None
    if cal_path.is_file():
        summary = cell_results_dir(case) / f"{case['id']}.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(cal_path.read_text(encoding="utf-8"), encoding="utf-8")
    print("ARTIFACTS pytest_copy:", flush=True)
    print(
        f"  results_summary.json: {summary.resolve() if summary else 'n/a'}", flush=True
    )
    return out


@pytest.mark.e2e
@pytest.mark.perf
def test_router_policy_replay_cell(
    require_router_e2e_perf,
    case_file: Path,
):
    """Workers → router → replay; single self-contained case JSON."""
    out = _run_cell(case_file)
    req_path = out / "result" / "requests.jsonl"
    assert req_path.is_file(), f"missing replay output {req_path}"
    rows = [
        json.loads(line)
        for line in req_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows, "replay produced no requests"
    failed = [r for r in rows if r.get("status") != "success"]
    assert not failed, (
        f"{len(failed)} non-success of {len(rows)} requests "
        f"(first status={failed[0].get('status')!r})"
    )
