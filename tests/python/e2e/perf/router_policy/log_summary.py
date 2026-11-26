"""Cell summary: TTFT/TPOT/JCT stats, calibration.json, and the printed report."""

from __future__ import annotations

import io
import json
import math
import re
import statistics
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from case_loader import cell_run_suffix, router_policy_result_path
from tests.python.e2e.perf.result_metadata import add_missing_metadata

if TYPE_CHECKING:
    from stack_launcher import RouterPolicyStack

METRIC = re.compile(r"^([a-zA-Z0-9_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$")
PROM_SAMPLE = re.compile(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+(\S+)(?:\s+\S+)?")
PROM_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')
PREFILL_COUNTERS = {
    "vllm:prefix_cache_hits_total": "prefix_cache_hits",
    "vllm:prefix_cache_queries_total": "prefix_cache_queries",
    "vllm:prompt_tokens_by_source_total": "new_prompt_tokens",
}


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
    """Copy calibration.json to the results dir only when its ``ok`` field is true."""

    cal = out / "calibration.json"
    if not cal.is_file():
        return []
    try:
        stats = json.loads(cal.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(stats, dict) or stats.get("ok") is not True:
        return []
    required = ("scenario", "router_policy", "concurrency", "task_num")
    if any(key not in stats for key in required):
        return []
    scenario = stats["scenario"]
    if not isinstance(scenario, str) or not scenario.strip():
        return []
    dest = router_policy_result_path(
        scenario,
        stats["concurrency"],
        stats["task_num"],
        cell_run_suffix(),
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(cal.read_text(encoding="utf-8"), encoding="utf-8")
    return [("results_json", dest)]


def _publish_cell_summary(text: str) -> None:
    """Print the cell summary. Worker, router, and replay lines already went to stderr."""

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
            elif stats.get("missing_latency_metrics"):
                print(
                    "DIAGNOSTIC missing latency measurements: "
                    + ", ".join(stats["missing_latency_metrics"]),
                    flush=True,
                )
            elif stats.get("router_policy_checks", {}).get("errors"):
                for error in stats["router_policy_checks"]["errors"]:
                    print(f"DIAGNOSTIC {error}", flush=True)
            elif stalled or stats["success"] != stats["requests"]:
                print(
                    "DIAGNOSTIC stalled or non-success rows in requests.jsonl",
                    flush=True,
                )
            else:
                print("DIAGNOSTIC cell was not ok", flush=True)
    _publish_cell_summary(buf.getvalue())
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


def _prefill_counters(path: Path) -> dict[str, float | None]:
    """Read token counters, preserving missing or invalid samples as unknown."""

    counters: dict[str, float | None] = dict.fromkeys(PREFILL_COUNTERS.values())
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return counters
    invalid: set[str] = set()
    for line in lines:
        match = PROM_SAMPLE.fullmatch(line.strip())
        if not match or match[1] not in PREFILL_COUNTERS:
            continue
        field = PREFILL_COUNTERS[match[1]]
        if field == "new_prompt_tokens":
            labels = dict(PROM_LABEL.findall(match[2] or ""))
            if labels.get("source") != "local_compute":
                continue
        try:
            value = float(match[3])
        except ValueError:
            invalid.add(field)
            continue
        total = (counters[field] or 0.0) + value
        if not math.isfinite(value) or value < 0 or not math.isfinite(total):
            invalid.add(field)
            continue
        counters[field] = total
    for field in invalid:
        counters[field] = None
    return counters


def _prefill_metrics(run: RouterPolicyStack) -> dict[str, Any]:
    """Summarize interval counters for Prefill workers (all workers when merged)."""

    worker_count = run.case.prefill_count or run.case.worker_count
    workers: list[dict[str, Any]] = []
    unavailable: list[str] = []
    for i in range(worker_count):
        before = _prefill_counters(run.out / "before" / f"engine{i}.prom")
        after = _prefill_counters(run.out / "after" / f"engine{i}.prom")
        worker: dict[str, Any] = {"worker_index": i}
        for field in PREFILL_COUNTERS.values():
            start, end = before[field], after[field]
            if start is None or end is None or end < start:
                worker[field] = None
                unavailable.append(f"engine{i}.{field}")
            else:
                worker[field] = end - start
        workers.append(worker)

    def total(field: str) -> float | None:
        values = [worker[field] for worker in workers]
        if not values or any(value is None for value in values):
            return None
        value = sum(values)
        if not math.isfinite(value):
            unavailable.append(field)
            return None
        return value

    hits = total("prefix_cache_hits")
    queries = total("prefix_cache_queries")
    new_tokens = total("new_prompt_tokens")
    hit_rate = None
    if hits is not None and queries is not None:
        if hits > queries:
            unavailable.append("prefix_cache_hit_rate")
        elif queries > 0:
            hit_rate = hits / queries
    skew = None
    if new_tokens:
        skew = max(worker["new_prompt_tokens"] for worker in workers) / (
            new_tokens / len(workers)
        )
    return {
        "prefix_cache_hit_rate": hit_rate,
        "new_prompt_tokens": new_tokens,
        "new_prompt_tokens_max_mean": skew,
        "workers": workers,
        "unavailable_metrics": unavailable,
    }


def _router_policy_checks(run: RouterPolicyStack) -> dict[str, Any]:
    """Check index/affinity activity from the final snapshot of this cell's router."""

    checks: dict[str, Any] = {"errors": []}
    if run.case.policy not in {"kv_aware", "kv_aware_aff"}:
        return checks
    placement_metric = "vllm_router_kv_placement_lookups_total"
    selectors = {
        "hit": (placement_metric, {"result": "hit"}),
        "miss": (placement_metric, {"result": "miss"}),
    }
    if run.case.policy == "kv_aware_aff":
        if run.case.prefill_count:
            affinity_metric = "vllm_router_pd_affinity_decisions_total"
            affinity_labels = {"stage": "decode"}
            outcomes = ("bind", "sticky_hit")
        else:
            affinity_metric = "vllm_router_prefix_routing_decisions_total"
            affinity_labels = {"policy": "kv_aware", "stage": "regular"}
            outcomes = ("sticky_hit",)
        selectors.update(
            {
                outcome: (affinity_metric, {**affinity_labels, "outcome": outcome})
                for outcome in outcomes
            }
        )

    counters: dict[str, float | None] = dict.fromkeys(selectors, 0.0)
    snapshot = run.out / "after/router.prom"
    lines = (
        snapshot.read_text(encoding="utf-8").splitlines() if snapshot.is_file() else []
    )
    for line in lines:
        match = PROM_SAMPLE.fullmatch(line.strip())
        if not match:
            continue
        labels = dict(PROM_LABEL.findall(match[2] or ""))
        for field, (name, required) in selectors.items():
            if match[1] != name or any(labels.get(k) != v for k, v in required.items()):
                continue
            if counters[field] is None:
                continue
            try:
                value = float(match[3])
            except ValueError:
                value = math.nan
            total = counters[field] + value
            counters[field] = (
                total
                if value >= 0 and math.isfinite(value) and math.isfinite(total)
                else None
            )

    def active(fields):
        values = [counters[field] for field in fields]
        return all(value is not None for value in values) and any(
            value > 0 for value in values
        )

    checks["placement_lookups"] = {field: counters[field] for field in ("hit", "miss")}
    if not active(("hit", "miss")):
        checks["errors"].append(
            "kv_aware ran with zero KV placement lookups against a verified index "
            "(or invalid counters) — check that the router has active KV event "
            "subscriptions and the engine publishes the sha256_cbor contract "
            f"(see #493); snapshot={snapshot}"
        )
    if run.case.policy == "kv_aware_aff":
        checks["affinity"] = {
            "metric": affinity_metric,
            "stage": affinity_labels["stage"],
            **{outcome: counters[outcome] for outcome in outcomes},
        }
        if not active(outcomes):
            checks["errors"].append(
                f"kv_aware_aff ran with no active {affinity_labels['stage']} affinity "
                f"decisions: require nonzero {' or '.join(outcomes)} in {affinity_metric}; "
                f"check session identity and affinity configuration; snapshot={snapshot}"
            )
    return checks


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
    """Quantiles and the actual sample count, same as the orig128 report."""

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
        "arm": case.policy,
        "returncode": replay_rc,
        "wall_seconds": wall_seconds,
        "makespan_seconds": makespan,
        "throughput_tok_s": (output_tokens / makespan) if makespan else None,
        "ok": replay_rc in (0, None)
        and not stalled
        and bool(rows)
        and len(ok) == len(rows),
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
        "prefill": _prefill_metrics(run),
        "router_policy_checks": _router_policy_checks(run),
    }
    missing_metrics = [
        metric
        for metric in ("ttft_s", "tpot_ms", "jct_s")
        if (stats[metric]["n"] is not None and stats[metric]["n"] <= 0)
        or not all(
            _is_number(stats[metric][quantile])
            and math.isfinite(stats[metric][quantile])
            and stats[metric][quantile] >= 0
            for quantile in ("mean", "p50", "p95", "p99")
        )
    ]
    if missing_metrics:
        stats["ok"] = False
        stats["missing_latency_metrics"] = missing_metrics
    if stats["router_policy_checks"]["errors"]:
        stats["ok"] = False
    stats = add_missing_metadata(
        stats,
        {
            "router_policy": case.policy,
            "scenario": case.scenario,
            "topology": "pd_disaggregated" if case.prefill_count else "pd_merged",
            "concurrency": case.concurrency,
            "task_num": case.task_num,
        },
    )
    (run.out / "calibration.json").write_text(json.dumps(stats, indent=2))
    emit_run_summary(
        run,
        stats,
        replay_rc=replay_rc,
        stalled=stalled,
    )
