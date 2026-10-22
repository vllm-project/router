#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Measure cold-prefill latency and fit deployment-specific cost coefficients."""

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
    JsonlRecorder,
    exact_prompt_tokens,
    nonnegative_least_squares,
    parse_nonnegative_int,
    parse_positive_float,
    parse_positive_int,
    parse_positive_int_list,
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
    persist_metadata,
    resolve_calibration_engine,
)


def fit_prefill(points: list[dict[str, object]]) -> dict[str, object]:
    """Fit a nonnegative quadratic model against uncached tokens in thousands."""
    if len(points) < 3:
        raise ValueError("prefill calibration requires at least three prompt lengths")
    x = np.asarray([float(point["prompt_tokens"]) / 1000 for point in points])
    actual = np.asarray([float(point["median_ttft_seconds"]) for point in points])
    design = np.column_stack((np.ones(len(x)), x, x**2))
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("prefill calibration requires at least three distinct prompt lengths")
    coefficients = nonnegative_least_squares(design, actual)
    unconstrained, _, _, _ = np.linalg.lstsq(design, actual, rcond=None)
    predicted = design @ coefficients
    errors = actual - predicted
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    residual = float(np.sum(errors**2))
    return {
        "formula": "seconds = intercept + linear*x + quadratic*x^2; x=uncached_tokens/1000",
        "coefficients": {
            "ttl_prefill_model_intercept_seconds": float(coefficients[0]),
            "ttl_prefill_model_linear_seconds_per_1k_tokens": float(coefficients[1]),
            "ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared": float(coefficients[2]),
        },
        "unconstrained_coefficients": {
            "ttl_prefill_model_intercept_seconds": float(unconstrained[0]),
            "ttl_prefill_model_linear_seconds_per_1k_tokens": float(unconstrained[1]),
            "ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared": float(unconstrained[2]),
        },
        "fit": {
            "r_squared": 1 - residual / total if total else 1.0,
            "rmse_seconds": float(np.sqrt(np.mean(errors**2))),
            "mae_seconds": float(np.mean(np.abs(errors))),
        },
        "points": points,
    }


async def calibrate(args: argparse.Namespace) -> None:
    """Run cold-prefill cases on one DP rank and assemble the shared profile."""
    stage_dir = prepare_stage_directory(args.output_dir, "prefill", args.overwrite)
    recorder = JsonlRecorder(stage_dir / "requests.jsonl")
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout) as client:
        metadata, engine_ids = await collect_metadata(client, args.model)
        target_engine, request_rank = resolve_calibration_engine(args.data_parallel_rank, engine_ids)
        write_json(
            stage_dir / "run_config.json",
            {
                "base_url": args.base_url,
                "model": args.model,
                "requested_data_parallel_rank": args.data_parallel_rank,
                "calibrated_engine": target_engine,
                "request_data_parallel_rank": request_rank,
                "prompt_lengths": args.prompt_lengths,
                "repeats": args.repeats,
                "shuffle_seed": args.shuffle_seed,
                "created_at_utc": utc_now(),
            },
        )
        digest = persist_metadata(args.output_dir, metadata)
        print(
            f"[prefill] engines={engine_ids}; calibrating engine={target_engine}; request rank header={request_rank}",
            flush=True,
        )
        await stream_completion(
            client,
            model=args.model,
            prompt=exact_prompt_tokens(256, 0),
            max_tokens=1,
            cache_salt=f"prefill-warmup-{uuid.uuid4().hex}",
            data_parallel_rank=request_rank,
        )
        counters_before = await collect_throughput_counters(client, target_engine)
        cases = [(length, repeat) for length in args.prompt_lengths for repeat in range(args.repeats)]
        random.Random(args.shuffle_seed).shuffle(cases)
        rows: list[dict[str, object]] = []
        for sequence, (length, repeat) in enumerate(cases, start=1):
            print(
                f"[prefill {sequence}/{len(cases)}] tokens={length} repeat={repeat + 1}/{args.repeats}",
                flush=True,
            )
            result = await stream_completion(
                client,
                model=args.model,
                prompt=exact_prompt_tokens(length, length * 100 + repeat),
                max_tokens=1,
                cache_salt=f"prefill-{uuid.uuid4().hex}",
                data_parallel_rank=request_rank,
            )
            row = {
                **result.as_json(),
                "target_prompt_tokens": length,
                "sequence": sequence,
                "repeat": repeat,
            }
            recorder.append(row)
            rows.append(row)
            print(
                f"[prefill {sequence}/{len(cases)}] ttft={result.ttft_seconds:.4f}s "
                f"cached={result.cached_prompt_tokens}",
                flush=True,
            )
        counters_after = await collect_throughput_counters(client, target_engine)
    points = []
    for length in sorted(args.prompt_lengths):
        values = [float(row["ttft_seconds"]) for row in rows if row["target_prompt_tokens"] == length]
        points.append(
            {
                "prompt_tokens": length,
                "sample_count": len(values),
                "median_ttft_seconds": statistics.median(values),
                "mean_ttft_seconds": statistics.fmean(values),
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
            **fit_prefill(points),
            "throughput_counter_delta": counter_delta(counters_before, counters_after),
        },
    }
    write_json(stage_dir / "result.json", result)
    profile = assemble_profile(args.output_dir)
    print(f"[prefill] result: {stage_dir / 'result.json'}", flush=True)
    if profile:
        print(f"[profile] assembled: {profile}", flush=True)


def parser() -> argparse.ArgumentParser:
    """Build the cold-prefill calibration CLI."""
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--base-url", default="http://127.0.0.1:8000")
    value.add_argument("--model", required=True)
    value.add_argument("--output-dir", type=Path, default=Path("tools/calibration/results"))
    value.add_argument("--data-parallel-rank", type=parse_nonnegative_int)
    value.add_argument(
        "--prompt-lengths",
        type=parse_positive_int_list,
        default=parse_positive_int_list("1024,2048,4096,8192,16384,32768,65536"),
    )
    value.add_argument("--repeats", type=parse_positive_int, default=2)
    value.add_argument("--shuffle-seed", type=int, default=260901)
    value.add_argument("--timeout", type=parse_positive_float, default=3600)
    value.add_argument("--overwrite", action="store_true")
    return value


if __name__ == "__main__":
    asyncio.run(calibrate(parser().parse_args()))
