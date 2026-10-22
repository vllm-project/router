#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Measure warm-prefix decode throughput and fit its batch/context surface."""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import uuid
from pathlib import Path

import httpx
import numpy as np

from tools.calibration.common import (
    CompletionResult,
    JsonlRecorder,
    exact_prompt_tokens,
    nonnegative_least_squares,
    parse_nonnegative_float,
    parse_nonnegative_int,
    parse_positive_float,
    parse_positive_int,
    parse_positive_int_list,
    parse_unit_fraction,
    prepare_stage_directory,
    stream_completion,
    utc_now,
    write_json,
)
from tools.calibration.metadata import (
    assemble_profile,
    collect_metadata,
    collect_throughput_counters,
    counter_delta,
    infer_kv_capacity_tokens,
    persist_metadata,
    resolve_calibration_engine,
)


def steady_measurement(results: list[CompletionResult]) -> dict[str, float | int]:
    """Measure net decode throughput over the common active output interval."""
    start = max(result.token_timestamps_monotonic[0] for result in results)
    end = min(result.token_timestamps_monotonic[-1] for result in results)
    seconds = end - start
    if seconds <= 0:
        raise RuntimeError("decode outputs have no overlapping steady interval")
    tokens = sum(start <= timestamp <= end for result in results for timestamp in result.token_timestamps_monotonic)
    throughput = tokens / seconds
    return {
        "steady_seconds": seconds,
        "steady_tokens": tokens,
        "output_throughput_tokens_per_second": throughput,
        "estimated_iteration_seconds": len(results) / throughput,
    }


def fit_decode(points: list[dict[str, object]]) -> dict[str, object]:
    """Fit iteration time against batch size and total logical context."""
    if len(points) < 3:
        raise ValueError("decode calibration requires at least three points")
    design = np.asarray([[1, float(point["batch_size"]), float(point["total_context_tokens"])] for point in points])
    target = np.asarray([float(point["median_iteration_seconds"]) for point in points])
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("decode points must independently vary batch and context")
    coefficients = nonnegative_least_squares(design, target)
    unconstrained, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coefficients
    errors = target - predicted
    total = float(np.sum((target - np.mean(target)) ** 2))
    residual = float(np.sum(errors**2))
    return {
        "formula": "throughput=B/(fixed+per_request*B+per_context_token*K)",
        "coefficients": {
            "decode_step_fixed_seconds": float(coefficients[0]),
            "decode_step_seconds_per_request": float(coefficients[1]),
            "decode_step_seconds_per_context_token": float(coefficients[2]),
        },
        "unconstrained_coefficients": {
            "decode_step_fixed_seconds": float(unconstrained[0]),
            "decode_step_seconds_per_request": float(unconstrained[1]),
            "decode_step_seconds_per_context_token": float(unconstrained[2]),
        },
        "fit": {
            "r_squared": 1 - residual / total if total else 1.0,
            "rmse_seconds": float(np.sqrt(np.mean(errors**2))),
            "mae_seconds": float(np.mean(np.abs(errors))),
        },
        "points": points,
        "interpretation": (
            "The initial model is fitted on one internal-DP rank and reused by "
            "homogeneous ranks. Shared EP coupling is not modeled, so batch gain "
            "may be overestimated."
        ),
    }


async def run_case(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    context: int,
    batch: int,
    repeat: int,
    case_id: int,
    recorder: JsonlRecorder,
    request_rank: int | None,
) -> dict[str, object]:
    """Warm all private prefixes and measure one concurrent decode batch."""
    prompts = [exact_prompt_tokens(context, case_id * 1000 + index) for index in range(batch)]
    salts = [f"decode-{case_id}-{index}-{uuid.uuid4().hex}" for index in range(batch)]
    print(f"[decode case {case_id}] warming {batch} private prefixes", flush=True)
    warm_results = []
    for warm_index, (prompt, salt) in enumerate(zip(prompts, salts, strict=True), start=1):
        print(f"[decode case {case_id}] warm prefix {warm_index}/{batch}", flush=True)
        warm_results.append(
            await stream_completion(
                client,
                model=args.model,
                prompt=prompt,
                max_tokens=1,
                cache_salt=salt,
                data_parallel_rank=request_rank,
            )
        )
    for warm in warm_results:
        recorder.append(
            {
                **warm.as_json(),
                "phase": "warmup",
                "case_id": case_id,
                "batch_size": batch,
                "context_tokens_per_request": context,
            }
        )
    # The HTTP stream can finish just before EngineCore releases the request's
    # KV blocks into the reusable prefix-cache set.  Starting the measured batch
    # immediately can therefore turn a nominally warm decode case into a mixed
    # prefill/decode case, especially with internal DP.  Give the backend a short
    # quiescent interval so every warmed private prefix becomes matchable.
    await asyncio.sleep(args.warmup_settle_seconds)
    results = await asyncio.gather(
        *(
            stream_completion(
                client,
                model=args.model,
                prompt=prompt,
                max_tokens=args.output_tokens,
                cache_salt=salt,
                data_parallel_rank=request_rank,
            )
            for prompt, salt in zip(prompts, salts, strict=True)
        )
    )
    for index, result in enumerate(results):
        recorder.append(
            {
                **result.as_json(),
                "phase": "measurement",
                "case_id": case_id,
                "request_index": index,
                "batch_size": batch,
                "context_tokens_per_request": context,
            }
        )
    measurement = steady_measurement(results)
    point: dict[str, object] = {
        "case_id": case_id,
        "repeat": repeat,
        "batch_size": batch,
        "context_tokens_per_request": context,
        "total_context_tokens": batch * context,
        **measurement,
    }
    print(
        f"[decode case {case_id}] B={batch} K={batch * context} "
        f"throughput={float(point['output_throughput_tokens_per_second']):.2f} tok/s",
        flush=True,
    )
    return point


async def calibrate(args: argparse.Namespace) -> None:
    """Run decode cases on one DP rank and assemble the shared profile."""
    stage_dir = prepare_stage_directory(args.output_dir, "decode", args.overwrite)
    recorder = JsonlRecorder(stage_dir / "requests.jsonl")
    timeout = httpx.Timeout(args.timeout)
    raw_points: list[dict[str, object]] = []
    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout) as client:
        metadata, engine_ids = await collect_metadata(client, args.model)
        target_engine, request_rank = resolve_calibration_engine(args.data_parallel_rank, engine_ids)
        kv_capacity_tokens = infer_kv_capacity_tokens(metadata)
        cases = [
            (context, batch, repeat)
            for context in args.context_lengths
            for batch in args.batch_sizes
            for repeat in range(args.repeats)
            if batch * context <= kv_capacity_tokens * args.max_kv_fraction
        ]
        random.Random(args.shuffle_seed).shuffle(cases)
        if not cases:
            raise ValueError("no decode cases fit the observed KV capacity")
        write_json(
            stage_dir / "run_config.json",
            {
                "base_url": args.base_url,
                "model": args.model,
                "requested_data_parallel_rank": args.data_parallel_rank,
                "calibrated_engine": target_engine,
                "request_data_parallel_rank": request_rank,
                "batch_sizes": args.batch_sizes,
                "context_lengths": args.context_lengths,
                "output_tokens": args.output_tokens,
                "warmup_settle_seconds": args.warmup_settle_seconds,
                "repeats": args.repeats,
                "max_kv_fraction": args.max_kv_fraction,
                "shuffle_seed": args.shuffle_seed,
                "observed_kv_capacity_tokens": kv_capacity_tokens,
                "created_at_utc": utc_now(),
            },
        )
        digest = persist_metadata(args.output_dir, metadata)
        print(
            f"[decode] engines={engine_ids}; calibrating engine={target_engine}; "
            f"request rank header={request_rank}; KV capacity={kv_capacity_tokens} tokens",
            flush=True,
        )
        counters_before = await collect_throughput_counters(client, target_engine)
        for case_id, (context, batch, repeat) in enumerate(cases, start=1):
            print(
                f"[decode {case_id}/{len(cases)}] B={batch} context={context} repeat={repeat + 1}/{args.repeats}",
                flush=True,
            )
            point = await run_case(
                client,
                args,
                context,
                batch,
                repeat,
                case_id,
                recorder,
                request_rank,
            )
            raw_points.append(point)
            write_json(stage_dir / "points.partial.json", {"points": raw_points})
        # Request timestamps are authoritative for the surface. Endpoint counters
        # cross-check the complete stage, including its prefix-warmup requests.
        counters_after = await collect_throughput_counters(client, target_engine)
    aggregated = []
    for context in sorted(args.context_lengths):
        for batch in sorted(args.batch_sizes):
            selected = [
                point
                for point in raw_points
                if point["context_tokens_per_request"] == context and point["batch_size"] == batch
            ]
            if not selected:
                continue
            iterations = [float(point["estimated_iteration_seconds"]) for point in selected]
            aggregated.append(
                {
                    "batch_size": batch,
                    "context_tokens_per_request": context,
                    "total_context_tokens": batch * context,
                    "sample_count": len(selected),
                    "median_iteration_seconds": statistics.median(iterations),
                }
            )
    result = {
        "metadata_hash": digest,
        "scope": {
            "kind": "homogeneous_internal_dp",
            "calibrated_engine": target_engine,
            "reusable_engine_ids": list(engine_ids),
        },
        "created_at_utc": utc_now(),
        "calibration": {
            **fit_decode(aggregated),
            "throughput_counters_after": counters_after,
            "throughput_counter_delta": counter_delta(counters_before, counters_after),
        },
    }
    write_json(stage_dir / "result.json", result)
    profile = assemble_profile(args.output_dir)
    print(f"[decode] result: {stage_dir / 'result.json'}", flush=True)
    if profile:
        print(f"[profile] assembled: {profile}", flush=True)


def parser() -> argparse.ArgumentParser:
    """Build the decode calibration CLI."""
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--base-url", default="http://127.0.0.1:8000")
    value.add_argument("--model", required=True)
    value.add_argument("--output-dir", type=Path, default=Path("tools/calibration/results"))
    value.add_argument("--data-parallel-rank", type=parse_nonnegative_int)
    value.add_argument(
        "--batch-sizes",
        type=parse_positive_int_list,
        default=parse_positive_int_list("1,2,4,8,16,32"),
    )
    value.add_argument(
        "--context-lengths",
        type=parse_positive_int_list,
        default=parse_positive_int_list("1024,8192,16384,32768"),
    )
    value.add_argument("--output-tokens", type=parse_positive_int, default=128)
    value.add_argument("--warmup-settle-seconds", type=parse_nonnegative_float, default=1.0)
    value.add_argument("--repeats", type=parse_positive_int, default=1)
    value.add_argument("--max-kv-fraction", type=parse_unit_fraction, default=0.9)
    value.add_argument("--shuffle-seed", type=int, default=260901)
    value.add_argument("--timeout", type=parse_positive_float, default=3600)
    value.add_argument("--overwrite", action="store_true")
    return value


if __name__ == "__main__":
    asyncio.run(calibrate(parser().parse_args()))
