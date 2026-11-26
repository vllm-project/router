"""Router policy perf: one pytest cell per concurrency in cases/<policy>.json."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCENARIO_DIR = Path(__file__).resolve().parent
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))

from case_loader import (  # noqa: E402
    case_at_concurrency,
    cell_output_dir,
    cell_run_suffix,
    load_case,
    router_policy_result_path,
)


def _identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing or invalid {label}")
    return value


def _records(value: Any, label: str) -> list[dict[str, Any]]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, dict) for item in value)
    ):
        raise ValueError(f"missing or invalid {label}")
    return value


def _requests(value: Any, kind_field: str) -> list[dict[str, Any]]:
    records = _records(value, "request nodes")
    if any(
        node.get(kind_field) not in {"request", "timing_dependency"} for node in records
    ):
        raise ValueError("invalid replay node type")
    return [node for node in records if node[kind_field] == "request"]


def _expected_requests(result_dir: Path, task_num: int) -> set[tuple[str, str]]:
    # AgentInfer writes both artifacts before sending any measured requests.
    # Source keys prevent a truncated plan from defining a smaller expected load.
    source = json.loads(
        (result_dir / "replay-source-analysis.json").read_text(encoding="utf-8")
    )
    plan = json.loads((result_dir / "replay-plan.json").read_text(encoding="utf-8"))
    sources: dict[str, set[str]] = {}
    for session in _records(source["sessions"], "source sessions"):
        if session.get("replayable") is not True:
            continue
        name = _identity(session.get("source_session_id"), "source session ID")
        keys = [
            _identity(node.get("key"), "source request key")
            for node in _requests(session.get("requests"), "replay_kind")
        ]
        if name in sources or not keys or len(set(keys)) != len(keys):
            raise ValueError("empty or duplicate source request identities")
        sources[name] = set(keys)

    tasks = _records(plan["tasks"], "planned tasks")
    if len(tasks) != task_num:
        raise ValueError(
            f"planned task count mismatch: expected {task_num}, got {len(tasks)}"
        )
    expected: set[tuple[str, str]] = set()
    sessions: set[str] = set()
    request_ids: set[str] = set()
    sampled_sources = []
    for task in tasks:
        session_id = _identity(task.get("runtime_session_id"), "planned session ID")
        source_id = _identity(
            task.get("source_session_id"), "planned source session ID"
        )
        if session_id in sessions or source_id not in sources:
            raise ValueError("duplicate runtime session or unknown source session")
        sessions.add(session_id)
        sampled_sources.append(source_id)
        nodes = _requests(task.get("requests"), "node_type")
        keys = [
            _identity(node.get("source_key"), "planned source key") for node in nodes
        ]
        if len(keys) != len(sources[source_id]) or set(keys) != sources[source_id]:
            raise ValueError(
                f"planned request coverage differs from source session {source_id}"
            )
        for node in nodes:
            request_id = _identity(node.get("runtime_request_id"), "planned request ID")
            if request_id in request_ids:
                raise ValueError(f"duplicate planned request ID: {request_id}")
            request_ids.add(request_id)
            expected.add((session_id, request_id))
    # The pinned cases sample one complete source cycle.
    if len(tasks) == len(sources) and Counter(sampled_sources) != Counter(
        sources.keys()
    ):
        raise ValueError("planned sessions do not cover the complete source cycle")
    return expected


def validate_replay(
    result_dir: Path, rows: list[dict[str, Any]], *, task_num: int
) -> dict[str, Any]:
    """Fail closed on incomplete, duplicate, or misidentified measured requests."""
    errors: list[str] = []
    checks: dict[str, Any] = {
        "expected_requests": None,
        "observed_requests": len(rows),
        "distinct_request_ids": 0,
        "errors": errors,
    }
    ids = []
    actual: set[tuple[str, str]] = set()
    invalid_ids = invalid_sessions = 0
    for row in rows:
        try:
            request_id = _identity(row.get("request_id"), "request ID")
        except ValueError:
            invalid_ids += 1
            continue
        ids.append(request_id)
        try:
            session_id = _identity(row.get("session_id"), "session ID")
        except ValueError:
            invalid_sessions += 1
            continue
        actual.add((session_id, request_id))
    checks["distinct_request_ids"] = len(set(ids))
    if invalid_ids:
        errors.append(f"missing or invalid request IDs in {invalid_ids} rows")
    if invalid_sessions:
        errors.append(f"missing or invalid session IDs in {invalid_sessions} rows")
    duplicates = len(ids) - len(set(ids))
    if duplicates:
        errors.append(f"duplicate request IDs in {duplicates} extra rows")
    try:
        expected = _expected_requests(result_dir, task_num)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"invalid replay plan/source inventory: {exc}")
        return checks
    checks["expected_requests"] = len(expected)
    if len(rows) != len(expected):
        errors.append(
            f"request count mismatch: expected {len(expected)}, got {len(rows)}"
        )
    if actual != expected:
        errors.append(
            f"request identity mismatch: {len(expected - actual)} missing, {len(actual - expected)} unexpected"
        )
    return checks


def _run_cell(case_file: Path, concurrency: int) -> Path:
    # log_summary imports the validators, so load the launcher only when running.
    from stack_launcher import run_router_policy_cell

    case = case_at_concurrency(load_case(case_file), concurrency)
    run_suffix = cell_run_suffix()
    out = cell_output_dir(case, run_suffix=run_suffix)
    cell_out = run_router_policy_cell(
        case_file,
        concurrency=concurrency,
        out_dir=out,
        run_suffix=run_suffix,
    )
    stats = json.loads((cell_out / "calibration.json").read_text(encoding="utf-8"))
    errors = (
        stats["replay_validation"]["errors"] + stats["router_policy_checks"]["errors"]
    )
    assert not errors, "\n".join(errors)
    summary = router_policy_result_path(
        stats["scenario"], stats["concurrency"], stats["task_num"], run_suffix
    )
    assert summary.is_file(), f"missing results JSON {summary}"
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
