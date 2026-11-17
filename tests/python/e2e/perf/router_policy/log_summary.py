"""Cell summary: TTFT/TPOT/JCT stats, calibration.json, and the printed report."""

from __future__ import annotations

import io
import json
import os
import re
import statistics
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stack_launcher import RouterPolicyStack

METRIC = re.compile(r"^([a-zA-Z0-9_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$")
RESULTS_SUMMARY_ENV = "AGENTX_RESULTS_SUMMARY"


def print_artifacts(
    out: Path,
    _workers: int,
    *,
    extras: list[tuple[str, Path]] | None = None,
) -> None:
    """Print the run directory and the results copy. Other files stay on disk."""

    entries: list[tuple[str, Path]] = [("cell_dir", out)]
    if extras:
        entries.extend(extras)
    print("ARTIFACTS", flush=True)
    for name, path in entries:
        print(f"  {name}: {path.resolve()}", flush=True)


def _write_results_summary_from_calibration(out: Path) -> list[tuple[str, Path]]:
    """Copy calibration.json only when its ``ok`` field is true."""

    raw = os.environ.get(RESULTS_SUMMARY_ENV, "").strip()
    if not raw:
        return []
    cal = out / "calibration.json"
    if not cal.is_file():
        return []
    try:
        stats = json.loads(cal.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(stats, dict) or stats.get("ok") is not True:
        return []
    dest = Path(raw)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(cal.read_text(encoding="utf-8"), encoding="utf-8")
    return [("results_summary.json", dest)]


def _flush_log_handles(run: RouterPolicyStack) -> None:
    if run.replay_log_fp is None:
        return
    try:
        run.replay_log_fp.flush()
    except (OSError, ValueError):
        pass


def _publish_beside_replay_result(run: RouterPolicyStack, text: str) -> None:
    """Append the cell summary to replay.log, right after agentinfer's Replay result."""

    out = run.out
    _flush_log_handles(run)
    replay_log = out / "replay.log"
    if replay_log.is_file():
        with replay_log.open("a", encoding="utf-8") as fp:
            if text and not text.startswith("\n"):
                fp.write("\n")
            fp.write(text)
            if text and not text.endswith("\n"):
                fp.write("\n")
            fp.flush()
    print(text, end="" if text.endswith("\n") else "\n", flush=True)


def emit_run_summary(
    run: RouterPolicyStack,
    stats: dict,
    *,
    replay_rc: int | None,
    stalled: bool,
) -> None:
    """Print STATS / ARTIFACTS next to the replay process result."""

    out = run.out
    workers = run.case.worker_count
    _flush_log_handles(run)
    summary_extra = _write_results_summary_from_calibration(out)
    failed = False
    buf = io.StringIO()
    with redirect_stdout(buf):
        print("\n--- cell summary ---", flush=True)
        print("STATS " + json.dumps(stats), flush=True)
        print_artifacts(out, workers, extras=summary_extra)
        if not stats.get("ok"):
            failed = True
            if replay_rc not in (0, None):
                print(f"DIAGNOSTIC replay failed (exit {replay_rc})", flush=True)
            elif stalled or stats["success"] != stats["requests"]:
                print(
                    "DIAGNOSTIC stalled or non-success rows in requests.jsonl",
                    flush=True,
                )
            else:
                print("DIAGNOSTIC cell was not ok", flush=True)
    _publish_beside_replay_result(run, buf.getvalue())
    if failed:
        raise SystemExit(1)


def print_failure_diagnostics(run: RouterPolicyStack, *, hint: str = "") -> None:
    """Print the failure summary. The caller stops the stack first."""

    out = run.out
    workers = run.case.worker_count
    print("\n--- cell summary (worker/router streams stopped) ---", flush=True)
    if hint:
        print(f"DIAGNOSTIC {hint}", flush=True)
    extras = _write_results_summary_from_calibration(out)
    print_artifacts(out, workers, extras=extras)


def sum_metric(path, name):
    total = 0.0
    for line in Path(path).read_text().splitlines():
        match = METRIC.match(line)
        if match and match.group(1) == name:
            total += float(match.group(3))
    return total


def pct(values, p):
    ordered = sorted(values)
    if not ordered:
        return None
    rank = (len(ordered) - 1) * p / 100
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _makespan_seconds(ok_rows: list[dict[str, Any]]) -> float | None:
    """Replay window: max(finished_at) - min(started_at) over successful requests."""

    starts: list[datetime] = []
    ends: list[datetime] = []
    for row in ok_rows:
        started = row.get("started_at")
        finished = row.get("finished_at")
        if not started or not finished:
            continue
        try:
            starts.append(datetime.fromisoformat(str(started)))
            ends.append(datetime.fromisoformat(str(finished)))
        except ValueError:
            continue
    if not starts:
        return None
    span = (max(ends) - min(starts)).total_seconds()
    if span <= 0:
        return None
    return span


def _quantiles(values: list[float]) -> dict[str, Any]:
    """Linear-interpolation mean / p50 / p95 / p99, same as the orig128 report."""

    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": pct(values, 50),
        "p95": pct(values, 95),
        "p99": pct(values, 99),
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _collect_duration_samples(node: Any, found: list[float]) -> None:
    """Collect numeric ``duration_seconds`` fields. Skip percentile objects."""

    if isinstance(node, list):
        for item in node:
            _collect_duration_samples(item, found)
        return
    if not isinstance(node, dict):
        return
    duration = node.get("duration_seconds")
    if _is_number(duration):
        status = str(node.get("status") or "success").lower()
        if status in {"success", "completed", "ok"}:
            found.append(float(duration))
    for key, child in node.items():
        if key == "duration_seconds":
            continue
        _collect_duration_samples(child, found)


def _precomputed_duration(node: Any) -> dict[str, Any] | None:
    if isinstance(node, dict):
        duration = node.get("duration_seconds")
        if isinstance(duration, dict) and any(
            key in duration for key in ("mean", "p50", "p95", "p99")
        ):
            return duration
        for child in node.values():
            found = _precomputed_duration(child)
            if found is not None:
                return found
        return None
    if isinstance(node, list):
        for child in node:
            found = _precomputed_duration(child)
            if found is not None:
                return found
    return None


def _jct_quantiles(result_dir: Path, ok_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """JCT is ``duration_seconds`` (session summary, else per-request field)."""

    summary_path = result_dir / "summary.json"
    samples: list[float] = []
    precomputed: dict[str, Any] | None = None
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = None
        if summary is not None:
            _collect_duration_samples(summary, samples)
            precomputed = _precomputed_duration(summary)
    if not samples:
        samples = [
            float(row["duration_seconds"])
            for row in ok_rows
            if _is_number(row.get("duration_seconds"))
        ]
    if samples:
        return _quantiles(samples)
    if precomputed is None:
        return _quantiles([])

    def pick(*keys: str) -> float | None:
        for key in keys:
            value = precomputed.get(key)
            if _is_number(value):
                return float(value)
        return None

    count = precomputed.get("n", precomputed.get("count"))
    return {
        "n": int(count) if _is_number(count) else None,
        "mean": pick("mean"),
        "p50": pick("p50", "median"),
        "p95": pick("p95"),
        "p99": pick("p99"),
    }


def write_calibration(
    run: RouterPolicyStack,
    *,
    result_dir: Path,
    replay_rc: int | None,
    wall_seconds: float,
    stalled: bool,
) -> None:
    """Write calibration.json from the replay rows, then print the summary."""

    case = run.case
    req_path = result_dir / "requests.jsonl"
    rows = (
        [
            json.loads(line)
            for line in req_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if req_path.is_file()
        else []
    )
    ok = [r for r in rows if r.get("status") == "success"]
    ttft = [float(r["ttft_seconds"]) for r in ok if r.get("ttft_seconds") is not None]
    lat = [r["latency_seconds"] for r in ok if r.get("latency_seconds") is not None]
    # TPOT ms/token: 1000 * (latency - ttft) / (output_tokens - 1); drop out <= 1.
    tpot_ms = [
        1000.0
        * (float(r["latency_seconds"]) - float(r["ttft_seconds"]))
        / (int(r["output_tokens"]) - 1)
        for r in ok
        if r.get("ttft_seconds") is not None
        and r.get("latency_seconds") is not None
        and (r.get("output_tokens") or 0) > 1
    ]
    output_tokens = sum(r.get("output_tokens") or 0 for r in ok)
    makespan = _makespan_seconds(ok)
    stats = {
        "router_policy": case.policy,
        "arm": case.policy,
        "returncode": replay_rc,
        "wall_seconds": wall_seconds,
        "makespan_seconds": makespan,
        "throughput_tok_s": (output_tokens / makespan) if makespan else None,
        "ok": replay_rc in (0, None) and not stalled and len(ok) == len(rows),
        "stalled": stalled,
        "requests": len(rows),
        "success": len(ok),
        "input_tokens": sum(r.get("input_tokens") or 0 for r in ok),
        "output_tokens": sum(r.get("output_tokens") or 0 for r in ok),
        "ttft_s": {
            **_quantiles(ttft),
            "p90": pct(ttft, 90),
            "max": max(ttft) if ttft else None,
        },
        "tpot_ms": _quantiles(tpot_ms),
        "jct_s": _jct_quantiles(result_dir, ok),
        "latency_s": {
            "mean": statistics.mean(lat) if lat else None,
            "p50": pct(lat, 50),
            "p90": pct(lat, 90),
            "max": max(lat) if lat else None,
        },
        "worker_requests": [
            sum_metric(
                run.out / "after" / f"engine{i}.prom", "vllm:request_success_total"
            )
            - sum_metric(
                run.out / "before" / f"engine{i}.prom",
                "vllm:request_success_total",
            )
            for i in range(case.worker_count)
        ],
        "cache_hit_rate": (lambda h, q: h / q if q else None)(
            sum(
                sum_metric(
                    run.out / p / f"engine{i}.prom",
                    "vllm:prefix_cache_hits_total",
                )
                for i in range(case.worker_count)
                for p in ["after"]
            ),
            sum(
                sum_metric(
                    run.out / "after" / f"engine{i}.prom",
                    "vllm:prefix_cache_queries_total",
                )
                for i in range(case.worker_count)
            ),
        ),
    }
    (run.out / "calibration.json").write_text(json.dumps(stats, indent=2))
    emit_run_summary(
        run,
        stats,
        replay_rc=replay_rc,
        stalled=stalled,
    )
