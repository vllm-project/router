"""Router policy perf: one pytest per self-contained case JSON in cases/<policy>/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCENARIO_DIR = Path(__file__).resolve().parent
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))

from casekit.case_config import (  # noqa: E402
    cell_output_dir,
    cell_results_summary_path,
    cell_run_suffix,
    load_case,
)
from casekit.cell_runner import run_router_policy_cell  # noqa: E402


def _run_cell(case_file: Path) -> Path:
    case = load_case(case_file, scenario_dir=SCENARIO_DIR)
    run_suffix = cell_run_suffix()
    out = cell_output_dir(case, run_suffix=run_suffix)
    summary = cell_results_summary_path(case, run_suffix=run_suffix)
    cell_out = run_router_policy_cell(
        case_file,
        out_dir=out,
        run_suffix=run_suffix,
        results_summary=summary,
    )
    assert summary.is_file(), f"missing results summary copy {summary}"
    return cell_out


def test_router_policy_replay_cell(case_file: Path):
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
