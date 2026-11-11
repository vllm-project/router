"""Router policy perf: one pytest cell per concurrency in cases/<policy>.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCENARIO_DIR = Path(__file__).resolve().parent
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))

from case_loader import (  # noqa: E402
    case_at_concurrency,
    cell_output_dir,
    cell_results_summary_path,
    cell_run_suffix,
    load_case,
)
from stack_launcher import run_router_policy_cell  # noqa: E402


def _run_cell(case_file: Path, concurrency: int) -> Path:
    case = case_at_concurrency(load_case(case_file), concurrency)
    run_suffix = cell_run_suffix()
    out = cell_output_dir(case, run_suffix=run_suffix)
    summary = cell_results_summary_path(case, run_suffix=run_suffix)
    cell_out = run_router_policy_cell(
        case_file,
        concurrency=concurrency,
        out_dir=out,
        run_suffix=run_suffix,
        results_summary=summary,
    )
    assert summary.is_file(), f"missing results summary copy {summary}"
    return cell_out


def test_router_policy_replay_cell(case_file: Path, concurrency: int):
    """Workers → router → replay; one concurrency from a scenario JSON."""
    out = _run_cell(case_file, concurrency)
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
