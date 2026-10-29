import json
import sys
from datetime import datetime
from pathlib import Path

root = Path(sys.argv[1])


def pct(a, p):
    a = sorted(a)
    if not a:
        return float("nan")
    r = (len(a) - 1) * p / 100.0
    lo = int(r)
    hi = min(lo + 1, len(a) - 1)
    return a[lo] + (a[hi] - a[lo]) * (r - lo)


out = {}
for cell in sorted(root.glob("c*-*")):
    f = cell / "result/requests.jsonl"
    if not f.exists():
        continue
    if not (cell / "calibration.json").exists():
        continue
    rows = [json.loads(line) for line in f.open()]
    ok = [r for r in rows if r.get("status") == "success"]
    cal = json.loads((cell / "calibration.json").read_text())
    pre = 0.0
    for p in cell.glob("after/engine*.prom"):
        for line in p.read_text().splitlines():
            if line.startswith("vllm:num_preemptions_total"):
                pre += float(line.rsplit(" ", 1)[1])
    ttft = [r["ttft_seconds"] for r in ok if r.get("ttft_seconds") is not None]
    tpot = [
        1000 * (r["latency_seconds"] - r["ttft_seconds"]) / (r["output_tokens"] - 1)
        for r in ok
        if r.get("output_tokens", 0) > 1 and r.get("ttft_seconds") is not None
    ]
    tpot_long = [
        1000 * (r["latency_seconds"] - r["ttft_seconds"]) / (r["output_tokens"] - 1)
        for r in ok
        if r.get("output_tokens", 0) > 100 and r.get("ttft_seconds") is not None
    ]
    lat = [r["latency_seconds"] for r in ok if r.get("latency_seconds") is not None]
    tpot_short = [
        1000 * (r["latency_seconds"] - r["ttft_seconds"]) / (r["output_tokens"] - 1)
        for r in ok
        if 1 < r.get("output_tokens", 0) <= 100 and r.get("ttft_seconds") is not None
    ]
    per = {}
    for r in ok:
        per.setdefault(r["session_id"], []).append(r)
    jct = [
        (
            max(datetime.fromisoformat(x["finished_at"]) for x in v)
            - min(datetime.fromisoformat(x["started_at"]) for x in v)
        ).total_seconds()
        for v in per.values()
    ]
    t0 = min(datetime.fromisoformat(r["started_at"]) for r in ok)
    t1 = max(datetime.fromisoformat(r["finished_at"]) for r in ok)
    makespan = (t1 - t0).total_seconds()
    otok = sum(r["output_tokens"] for r in ok)
    stats = {}
    for key, arr in (
        ("ttft", ttft),
        ("tpot", tpot),
        ("tpot_long", tpot_long),
        ("tpot_short", tpot_short),
        ("lat", lat),
        ("jct", jct),
    ):
        stats[key] = dict(
            n=len(arr),
            mean=sum(arr) / len(arr),
            p50=pct(arr, 50),
            p90=pct(arr, 90),
            p95=pct(arr, 95),
            p99=pct(arr, 99),
            max=max(arr),
        )
    out[cell.name] = dict(
        hit=cal["cache_hit_rate"],
        wall=cal["wall_seconds"],
        preempt=int(pre),
        worker=cal["worker_requests"],
        reqs=len(ok),
        out_tokens=otok,
        makespan=makespan,
        **stats,
    )

print(json.dumps(out, indent=1))
