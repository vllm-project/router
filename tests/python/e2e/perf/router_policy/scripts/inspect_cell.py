# -*- coding: utf-8 -*-
"""Inspect one replay cell from raw artifacts.

Usage:
    python inspect_cell.py <runs_<TAG>/c<conc>-<arm>>
"""

import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def pcts(values, qs=(50, 90, 95, 99)):
    vals = sorted(values)
    n = len(vals)
    out = {}
    if not n:
        return {q: float("nan") for q in qs}, float("nan"), float("nan")
    for q in qs:
        pos = (n - 1) * q / 100.0
        lo, hi = int(pos), min(int(pos) + 1, n - 1)
        out[q] = vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)
    return out, statistics.fmean(vals), vals[-1]


def prom_value(path, metric):
    if not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith(metric):
            try:
                total += float(line.rsplit(" ", 1)[-1])
            except ValueError:
                pass
    return total


def main():
    cell = Path(sys.argv[1])
    reqs = [
        json.loads(line)
        for line in (cell / "result" / "requests.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ok = [r for r in reqs if r.get("status") == "success"]

    hit_q = sum(
        prom_value(
            cell / "after" / ("engine%d.prom" % i), "vllm:prefix_cache_queries_total"
        )
        for i in range(4)
    )
    hit_h = sum(
        prom_value(
            cell / "after" / ("engine%d.prom" % i), "vllm:prefix_cache_hits_total"
        )
        for i in range(4)
    )
    pre_after = sum(
        prom_value(cell / "after" / ("engine%d.prom" % i), "vllm:num_preemptions_total")
        for i in range(4)
    )
    pre_before = sum(
        prom_value(
            cell / "before" / ("engine%d.prom" % i), "vllm:num_preemptions_total"
        )
        for i in range(4)
    )

    calib = json.loads((cell / "calibration.json").read_text())
    ttft = [r["ttft_seconds"] for r in ok if r.get("ttft_seconds") is not None]
    lat = [r["latency_seconds"] for r in ok if r.get("latency_seconds") is not None]
    tpot = [
        1000.0 * (r["latency_seconds"] - r["ttft_seconds"]) / (r["output_tokens"] - 1)
        for r in ok
        if r.get("output_tokens", 0) > 1 and r.get("ttft_seconds") is not None
    ]
    tpot_long = [
        1000.0 * (r["latency_seconds"] - r["ttft_seconds"]) / (r["output_tokens"] - 1)
        for r in ok
        if r.get("output_tokens", 0) > 100 and r.get("ttft_seconds") is not None
    ]
    tpot_short = [
        1000.0 * (r["latency_seconds"] - r["ttft_seconds"]) / (r["output_tokens"] - 1)
        for r in ok
        if 1 < r.get("output_tokens", 0) <= 100 and r.get("ttft_seconds") is not None
    ]

    by_session = defaultdict(list)
    for r in ok:
        by_session[r["session_id"]].append(r)
    jct = []
    for rows in by_session.values():
        rows.sort(key=lambda r: r["started_at"])
        jct.append((rows[-1]["finished_at"], rows[0]["started_at"]))

    def ts(s):
        return datetime.fromisoformat(s).timestamp()

    jct = [ts(b) - ts(a) for b, a in jct]

    engine = [
        prom_value(cell / "after" / ("engine%d.prom" % i), "vllm:request_success_total")
        for i in range(4)
    ]

    starts = [ts(r["started_at"]) for r in ok]
    ends = [ts(r["finished_at"]) for r in ok]
    makespan = max(ends) - min(starts)

    print("== %s" % cell.name)
    print(
        "requests=%d success=%d sessions=%d out_tokens=%s"
        % (
            len(reqs),
            len(ok),
            len(by_session),
            f"{sum(r['output_tokens'] for r in ok):,}",
        )
    )
    print(
        "prefix hit=%.2f%%  preemptions=%d  makespan=%.0f s  wall=%.0f s"
        % (
            100.0 * hit_h / hit_q if hit_q else float("nan"),
            pre_after - pre_before,
            makespan,
            calib.get("wall_seconds", float("nan")),
        )
    )
    print("throughput=%.1f tok/s" % (sum(r["output_tokens"] for r in ok) / makespan))
    for name, vals, decimals in (
        ("TTFT s", ttft, 2),
        ("TPOT ms", tpot, 1),
        ("latency s", lat, 2),
        ("JCT s", jct, 1),
    ):
        q, mean, mx = pcts(vals)
        fmt = "%." + str(decimals) + "f"
        print(
            "%-9s mean=%s p50=%s p90=%s p95=%s p99=%s max=%s (n=%d)"
            % (
                name,
                fmt % mean,
                fmt % q[50],
                fmt % q[90],
                fmt % q[95],
                fmt % q[99],
                fmt % mx,
                len(vals),
            )
        )
    for name, vals in (("TPOT>100", tpot_long), ("TPOT<=100", tpot_short)):
        q, mean, mx = pcts(vals)
        print(
            "%-9s mean=%.1f p50=%.1f p95=%.1f p99=%.1f (n=%d)"
            % (name, mean, q[50], q[95], q[99], len(vals))
        )
    print(
        "per-engine requests (prom, before-baseline not subtracted): "
        + "/".join("%d" % x for x in engine)
    )

    dup = len(reqs) - len({r["request_id"] for r in reqs})
    short = len([r for r in reqs if r.get("output_tokens", 0) <= 1])
    print(
        "flags: duplicate_request_id=%d  output_tokens<=1=%d  non_success=%d"
        % (dup, short, len(reqs) - len(ok))
    )


if __name__ == "__main__":
    main()
