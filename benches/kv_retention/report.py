#!/usr/bin/env python3
"""Tables from ``run_eval.py replay`` runs: one row per arm, the median of its seeds and their range.

    python report.py runs/replay > tables.md

Everything is read from a run's own records: AIPerf's export for what the client saw, the engine's counters for where
prompt tokens came from, the connector's receipts for what was asked, acknowledged and observed.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PAUSE_SECONDS = 1.0
"""A turn resumes when its conversation's previous turn ended at least this long before it was sent."""
POOL_TOKENS = 36_400
SAMPLE_RATE = 0.05
"""--kv-cache-metrics-sample: vLLM's KV lifecycle histograms see this share of blocks."""


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def parse_samples(text: str) -> dict[str, dict[str, float]]:
    """A Prometheus exposition: metric name to label text to value."""
    parsed: dict[str, dict[str, float]] = defaultdict(dict)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        name, _, labels = head.partition("{")
        try:
            parsed[name][labels] = float(value)
        except ValueError:
            continue
    return parsed


def delta(
    start: dict[str, dict[str, float]],
    end: dict[str, dict[str, float]],
    name: str,
    needle: str = "",
) -> float:
    return sum(
        value - start.get(name, {}).get(labels, 0.0)
        for labels, value in end.get(name, {}).items()
        if needle in labels
    )


def histogram_quantile(
    start: dict[str, dict[str, float]],
    end: dict[str, dict[str, float]],
    name: str,
    q: float,
) -> float | None:
    """A quantile of a histogram's change over a run, at the upper bound of its bucket."""
    buckets = []
    for labels, value in end.get(f"{name}_bucket", {}).items():
        marker = 'le="'
        at = labels.find(marker)
        if at < 0:
            continue
        bound = labels[at + len(marker) : labels.find('"', at + len(marker))]
        change = value - start.get(f"{name}_bucket", {}).get(labels, 0.0)
        buckets.append((float("inf") if bound == "+Inf" else float(bound), change))
    buckets.sort()
    if not buckets or buckets[-1][1] <= 0:
        return None
    target = q * buckets[-1][1]
    return next(bound for bound, count in buckets if count >= target)


def client_side(run: Path) -> dict[str, Any]:
    """TTFT and the split of computed tokens, from AIPerf's per-request export."""
    records = []
    with (run / "aiperf" / "profile_export.jsonl").open() as lines:
        for line in lines:
            record = json.loads(line)
            if record["metadata"].get("benchmark_phase") == "profiling" and not record[
                "metadata"
            ].get("was_cancelled"):
                records.append(record)
    conversations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        conversations[record["metadata"]["x_correlation_id"]].append(record)
    all_ttft: list[float] = []
    resume_ttft: list[float] = []
    cold, fresh, recomputed = 0, 0, 0
    output_tokens = 0
    for turns in conversations.values():
        turns.sort(key=lambda record: record["metadata"]["request_start_ns"])
        previous = None
        for record in turns:
            metrics = record["metrics"]
            ttft = metrics.get("time_to_first_token", {}).get("value")
            isl = metrics.get("input_sequence_length", {}).get("value", 0)
            osl = metrics.get("output_sequence_length", {}).get("value", 0)
            cached = metrics.get("usage_prompt_cache_read_tokens", {}).get("value", 0)
            computed = max(0, isl - cached)
            output_tokens += osl
            if ttft is not None:
                all_ttft.append(ttft)
            if previous is None:
                cold += computed
            else:
                gap = (
                    record["metadata"]["request_start_ns"]
                    - previous["metadata"]["request_end_ns"]
                ) / 1e9
                if gap >= PAUSE_SECONDS and ttft is not None:
                    resume_ttft.append(ttft)
                # A replay's next prompt carries the trace's text, not the output the
                # model generated: at most the previous prompt can be reused.
                reusable = (
                    previous["metrics"].get("input_sequence_length", {}).get("value", 0)
                )
                new = max(0, isl - reusable)
                fresh += min(computed, new)
                recomputed += max(0, computed - new)
            previous = record
    starts = [record["metadata"]["request_start_ns"] for record in records]
    ends = [record["metadata"]["request_end_ns"] for record in records]
    seconds = (max(ends) - min(starts)) / 1e9 if records else 0.0
    return {
        "requests": len(records),
        "ttft_p50": percentile(all_ttft, 0.5),
        "ttft_p90": percentile(all_ttft, 0.9),
        "ttft_p99": percentile(all_ttft, 0.99),
        "resumes": len(resume_ttft),
        "resume_p50": percentile(resume_ttft, 0.5),
        "resume_p90": percentile(resume_ttft, 0.9),
        "tokens_per_second": output_tokens / seconds if seconds else 0.0,
        "cold_tokens": cold,
        "new_tokens": fresh,
        "recomputed_tokens": recomputed,
        "seconds": seconds,
    }


def engine_side(run: Path, seconds: float) -> dict[str, Any]:
    """Where prompt tokens came from and what eviction looked like, from the engine's counters over the run."""
    start = parse_samples((run / "metrics_start.txt").read_text())
    end = parse_samples((run / "metrics_end.txt").read_text())
    by_source = "vllm:prompt_tokens_by_source_total"
    gpu = delta(start, end, by_source, 'source="local_cache_hit"')
    cpu = delta(start, end, by_source, 'source="external_kv_transfer"')
    computed = delta(start, end, by_source, 'source="local_compute"')
    total = gpu + cpu + computed
    idle = "vllm:kv_block_idle_before_evict_seconds"
    evicted_sampled = delta(start, end, f"{idle}_count")
    queue_count = delta(start, end, "vllm:request_queue_time_seconds_count")
    queue_sum = delta(start, end, "vllm:request_queue_time_seconds_sum")
    return {
        "gpu_tokens": gpu,
        "cpu_tokens": cpu,
        "computed_tokens": computed,
        "gpu_share": gpu / total if total else 0.0,
        "cpu_share": cpu / total if total else 0.0,
        "computed_share": computed / total if total else 0.0,
        "preemptions": delta(start, end, "vllm:num_preemptions_total"),
        "gpu_evictions_per_s": evicted_sampled / SAMPLE_RATE / seconds
        if seconds
        else 0.0,
        "idle_before_evict_p50": histogram_quantile(start, end, idle, 0.5),
        "cpu_store_failures": delta(
            start, end, "vllm:kv_offload_allocation_failure_total"
        ),
        "load_gib": delta(start, end, "vllm:kv_offload_load_bytes_total") / 2**30,
        "store_gib": delta(start, end, "vllm:kv_offload_store_bytes_total") / 2**30,
        "queue_seconds_mean": queue_sum / queue_count if queue_count else 0.0,
    }


def sampled(run: Path) -> dict[str, Any]:
    """Time averages over the run, from the 2 s samples."""
    rows = [
        json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()
    ]
    mean = lambda name: statistics.fmean(row[name] for row in rows) if rows else 0.0  # noqa: E731
    return {
        "kv_usage_mean": mean("vllm:kv_cache_usage_perc"),
        "running_mean": mean("vllm:num_requests_running"),
        "waiting_mean": mean("vllm:num_requests_waiting"),
    }


def receipts(run: Path, seconds: float) -> dict[str, Any]:
    """What was asked for, acknowledged and observed, from the connector's receipts."""
    path = run / "receipts.jsonl"
    rows = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    acks: Counter[tuple[str, str, str | None]] = Counter()
    ends: Counter[tuple[str, str]] = Counter()
    held_token_seconds = 0.0
    held_tokens = 0
    used_tokens = 0
    hit_tokens = 0
    cpu_tier: dict[str, Any] = {}
    observed_gpu = 0
    observed_cpu = 0
    for row in rows:
        event = row["event"]
        if event == "ack":
            acks[(row["tier"], row["state"], row["reason"])] += 1
        elif event == "end":
            ends[(row["tier"], row["outcome"])] += 1
            if row["tier"] == "gpu":
                held_token_seconds += row["tokens"] * row["held_seconds"]
                held_tokens += row["tokens"]
                if row["outcome"] == "used":
                    used_tokens += row["tokens"]
                    hit_tokens += row["hit_tokens"]
        elif event == "cpu_tier":
            cpu_tier = row
        elif event == "request":
            observed_gpu += row["gpu_hit_tokens"]
            observed_cpu += row["cpu_hit_tokens"]
    gpu_ended = sum(count for (tier, _), count in ends.items() if tier == "gpu")
    return {
        "gpu_accepted": sum(
            count
            for (tier, state, _), count in acks.items()
            if tier == "gpu" and state == "accepted"
        ),
        "gpu_refused": sum(
            count
            for (tier, state, _), count in acks.items()
            if tier == "gpu" and state == "rejected"
        ),
        "gpu_used": ends[("gpu", "used")],
        "gpu_expired": ends[("gpu", "expired")],
        "gpu_evicted": ends[("gpu", "evicted")],
        "gpu_released": ends[("gpu", "released")],
        "gpu_used_share": ends[("gpu", "used")] / gpu_ended if gpu_ended else None,
        "gpu_used_token_share": used_tokens / held_tokens if held_tokens else None,
        "held_pool_share": held_token_seconds / seconds / POOL_TOKENS
        if seconds
        else 0.0,
        "cpu_accepted": sum(
            count
            for (tier, state, _), count in acks.items()
            if tier == "cpu" and state == "accepted"
        ),
        "cpu_refused": sum(
            count
            for (tier, state, _), count in acks.items()
            if tier == "cpu" and state == "rejected"
        ),
        "cpu_expired": ends[("cpu", "expired")],
        "cpu_evictions_per_s": cpu_tier.get("chunks_evicted", 0) / seconds
        if seconds
        else 0.0,
        "leased_chunks_evicted": cpu_tier.get("leased_chunks_evicted", 0),
        "chunks_clipped": cpu_tier.get("chunks_clipped", 0),
        "observed_gpu_tokens": observed_gpu,
        "observed_cpu_tokens": observed_cpu,
        "reasons": {
            f"{tier}:{state}:{reason}": count
            for (tier, state, reason), count in acks.items()
        },
    }


PAUSE_EDGES = (1.0, 5.0, 10.0, 30.0, 120.0, float("inf"))
SIZE_EDGES = (0.25, 0.5, float("inf"))
"""Prompt size as a share of the pool: what a hold's budget has to fit."""
MATCH_SECONDS = 5.0
"""How far apart the connector's and the client's clocks of one turn's finish may be."""


def restore_profile(run: Path) -> dict[str, list[float]] | None:
    """Which restored tokens a hold could have protected: the tokens by the pause before the turn and by the size of
    its prompt. The receipts say what was restored, AIPerf's export says when; a turn is its prompt length and
    finish time, which works on every arm, with or without hints."""
    path = run / "receipts.jsonl"
    if not path.exists():
        return None
    finished: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["event"] == "request":
            finished[row["prompt_tokens"]].append((row["t"], row["cpu_hit_tokens"]))
    with (run / "aiperf" / "profile_export.jsonl").open() as lines:
        records = sorted(
            (json.loads(line) for line in lines),
            key=lambda record: record["metadata"]["request_start_ns"],
        )
    ended_at: dict[str, int] = {}
    by_pause = [0.0] * len(PAUSE_EDGES)
    by_size = [0.0] * len(SIZE_EDGES)
    for record in records:
        metadata = record["metadata"]
        session = metadata["x_correlation_id"]
        prompt = record["metrics"].get("input_sequence_length", {}).get("value")
        end = metadata["request_end_ns"] / 1e9
        candidates = finished.get(prompt, [])
        match = min(candidates, key=lambda row: abs(row[0] - end), default=None)
        if match is not None and abs(match[0] - end) <= MATCH_SECONDS:
            candidates.remove(match)
            if session in ended_at and match[1]:
                pause = (metadata["request_start_ns"] - ended_at[session]) / 1e9
                by_pause[next(i for i, e in enumerate(PAUSE_EDGES) if pause <= e)] += (
                    match[1]
                )
                share = prompt / POOL_TOKENS
                by_size[next(i for i, e in enumerate(SIZE_EDGES) if share <= e)] += (
                    match[1]
                )
        ended_at[session] = metadata["request_end_ns"]
    return {"by_pause": by_pause, "by_size": by_size}


def summarize(run: Path) -> dict[str, Any]:
    client = client_side(run)
    return {
        **client,
        **engine_side(run, client["seconds"]),
        **sampled(run),
        **receipts(run, client["seconds"]),
        "restore_profile": restore_profile(run),
        **json.loads((run / "run.json").read_text()),
    }


def cell(values: list[float | None], digits: int = 0, scale: float = 1.0) -> str:
    """The median of a row's seeds, and their range."""
    present = [value * scale for value in values if value is not None]
    if not present:
        return "—"
    median = statistics.median(present)
    if len(present) == 1:
        return f"{median:,.{digits}f}"
    return (
        f"{median:,.{digits}f} ({min(present):,.{digits}f}–{max(present):,.{digits}f})"
    )


def table(
    title: str,
    arms: dict[str, list[dict[str, Any]]],
    columns: list[tuple[str, str, int, float]],
) -> str:
    lines = [
        f"### {title}",
        "",
        "| Arm | Runs | " + " | ".join(column[0] for column in columns) + " |",
    ]
    lines.append("|---|---:|" + "---:|" * len(columns))
    for arm, runs in arms.items():
        cells = [
            cell([run.get(key) for run in runs], digits, scale)
            for _, key, digits, scale in columns
        ]
        lines.append(f"| {arm} | {len(runs)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def paired(
    arms: dict[str, list[dict[str, Any]]], arm: str, baseline: str, keys: list[str]
) -> str:
    """Each seed's relative change against its baseline run on the same seed: seeds are paired by construction."""
    if arm not in arms or baseline not in arms:
        return ""
    base = {run["seed"]: run for run in arms[baseline]}
    parts = []
    for key in keys:
        changes = []
        for run in sorted(arms[arm], key=lambda run: run["seed"]):
            other = base.get(run["seed"], {}).get(key)
            if other and run.get(key) is not None:
                changes.append(f"{(run[key] / other - 1) * 100:+.0f}%")
        parts.append(f"{key} {' '.join(changes)}")
    return f"- {arm} against {baseline}, seed by seed: " + "; ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", type=Path, help="also write every run's numbers here")
    args = parser.parse_args()
    arms: dict[str, list[dict[str, Any]]] = {}
    for arm_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        runs = [
            summarize(run_dir)
            for run_dir in sorted(arm_dir.iterdir())
            if (run_dir / "run.json").exists()
            and json.loads((run_dir / "run.json").read_text()).get("valid", True)
            and (run_dir / "aiperf" / "profile_export.jsonl").exists()
        ]
        if runs:
            arms[arm_dir.name] = runs
    order = [
        "stock-4x",
        "off-4x",
        "gpu10-4x",
        "gpu30-4x",
        "off-1x",
        "cpu-1x",
        "cpu30-1x",
        "both-1x",
    ]
    arms = {
        arm: arms[arm]
        for arm in [*order, *sorted(set(arms) - set(order))]
        if arm in arms
    }
    if args.json:
        args.json.write_text(json.dumps(arms, indent=1))
    print(
        table(
            "Where prompt tokens came from",
            arms,
            [
                ("GPU hit %", "gpu_share", 1, 100),
                ("CPU restore %", "cpu_share", 1, 100),
                ("Computed %", "computed_share", 1, 100),
                ("Recomputed tokens", "recomputed_tokens", 0, 1),
                ("New tokens", "new_tokens", 0, 1),
                ("Cold tokens", "cold_tokens", 0, 1),
            ],
        )
    )
    print(
        table(
            "Time to first token (ms)",
            arms,
            [
                ("Resume p50", "resume_p50", 0, 1),
                ("Resume p90", "resume_p90", 0, 1),
                ("All p50", "ttft_p50", 0, 1),
                ("All p90", "ttft_p90", 0, 1),
                ("All p99", "ttft_p99", 0, 1),
                ("Resumes", "resumes", 0, 1),
                ("Requests", "requests", 0, 1),
            ],
        )
    )
    print(
        table(
            "Retention",
            arms,
            [
                ("Holds accepted", "gpu_accepted", 0, 1),
                ("Refused", "gpu_refused", 0, 1),
                ("Used", "gpu_used", 0, 1),
                ("Expired", "gpu_expired", 0, 1),
                ("Evicted", "gpu_evicted", 0, 1),
                ("Used, by tokens %", "gpu_used_token_share", 0, 100),
                ("Pool held, mean %", "held_pool_share", 1, 100),
                ("Leases accepted", "cpu_accepted", 0, 1),
                ("Leased chunks evicted", "leased_chunks_evicted", 0, 1),
                ("Chunks clipped", "chunks_clipped", 0, 1),
            ],
        )
    )
    print(
        table(
            "Eviction pressure and cost",
            arms,
            [
                ("GPU evictions/s", "gpu_evictions_per_s", 0, 1),
                ("Idle before evict p50 (s)", "idle_before_evict_p50", 1, 1),
                ("CPU chunks evicted/s", "cpu_evictions_per_s", 0, 1),
                ("CPU store failures", "cpu_store_failures", 0, 1),
                ("Restored GiB", "load_gib", 1, 1),
                ("Stored GiB", "store_gib", 1, 1),
                ("KV usage mean %", "kv_usage_mean", 0, 100),
                ("Waiting mean", "waiting_mean", 1, 1),
                ("Queue mean (s)", "queue_seconds_mean", 2, 1),
                ("Preemptions", "preemptions", 0, 1),
                ("Tokens/s", "tokens_per_second", 0, 1),
            ],
        )
    )
    profiled = {
        arm: [
            {
                **{
                    f"pause_{i}": value
                    for i, value in enumerate(run["restore_profile"]["by_pause"])
                },
                **{
                    f"size_{i}": value
                    for i, value in enumerate(run["restore_profile"]["by_size"])
                },
            }
            for run in runs
            if run["restore_profile"]
        ]
        for arm, runs in arms.items()
    }
    print(
        table(
            "Restored tokens, by the pause before the turn and by the prompt's share of the pool",
            {arm: runs for arm, runs in profiled.items() if runs},
            [
                ("<=1 s", "pause_0", 0, 1),
                ("<=5 s", "pause_1", 0, 1),
                ("<=10 s", "pause_2", 0, 1),
                ("<=30 s", "pause_3", 0, 1),
                ("<=120 s", "pause_4", 0, 1),
                (">120 s", "pause_5", 0, 1),
                ("Prompt <=25%", "size_0", 0, 1),
                ("<=50%", "size_1", 0, 1),
                (">50%", "size_2", 0, 1),
            ],
        )
    )
    keys = ["resume_p50", "resume_p90", "ttft_p90", "tokens_per_second"]
    for arm, baseline in (
        ("off-4x", "stock-4x"),
        ("gpu10-4x", "off-4x"),
        ("gpu30-4x", "off-4x"),
        ("cpu-1x", "off-1x"),
        ("cpu30-1x", "off-1x"),
        ("both-1x", "off-1x"),
    ):
        if line := paired(arms, arm, baseline, keys):
            print(line)


if __name__ == "__main__":
    main()
