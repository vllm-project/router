#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Shared exact-token requests, fitting helpers, and durable artifacts."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np


@dataclass(frozen=True)
class CompletionResult:
    """Client-observed lengths and timing for one streamed completion."""

    request_id: str
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int
    ttft_seconds: float
    decode_seconds: float
    e2e_latency_seconds: float
    token_timestamps_monotonic: tuple[float, ...]

    def as_json(self) -> dict[str, object]:
        """Return compact request evidence without per-token timestamps."""
        value = asdict(self)
        del value["token_timestamps_monotonic"]
        return value


class JsonlRecorder:
    """Append and flush each completed calibration request."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def append(self, value: dict[str, object]) -> None:
        """Persist one compact JSON record immediately."""
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
            stream.flush()


def exact_prompt_tokens(length: int, variant: int) -> list[int]:
    """Generate a deterministic exact-length token-ID prompt."""
    if length <= 0:
        raise ValueError("prompt length must be positive")
    prefix = [1000 + variant % 997, 2000 + variant % 991]
    body = [3000 + variant % 127, 4000 + variant % 113]
    return (prefix + body * ((length + 1) // 2))[:length]


async def stream_completion(
    client: httpx.AsyncClient,
    *,
    model: str,
    prompt: list[int],
    max_tokens: int,
    cache_salt: str,
    data_parallel_rank: int | None,
) -> CompletionResult:
    """Send one completion, optionally pinned to an internal-DP EngineCore rank."""
    request_id = f"cal-{uuid.uuid4().hex}"
    started_at = time.monotonic()
    token_timestamps: list[float] = []
    observed_prompt_tokens = len(prompt)
    observed_completion_tokens = 0
    cached_prompt_tokens = 0
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "return_token_ids": True,
        "add_special_tokens": False,
        "cache_salt": cache_salt,
        "request_id": request_id,
    }
    headers = {"X-data-parallel-rank": str(data_parallel_rank)} if data_parallel_rank is not None else None
    async with client.stream("POST", "/v1/completions", json=payload, headers=headers) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            item = json.loads(data)
            now = time.monotonic()
            usage = item.get("usage")
            if isinstance(usage, dict):
                observed_prompt_tokens = int(usage.get("prompt_tokens") or observed_prompt_tokens)
                observed_completion_tokens = max(
                    observed_completion_tokens,
                    int(usage.get("completion_tokens") or 0),
                )
                details = usage.get("prompt_tokens_details")
                if isinstance(details, dict):
                    cached_prompt_tokens = max(cached_prompt_tokens, int(details.get("cached_tokens") or 0))
            choices = item.get("choices")
            if not isinstance(choices, list):
                continue
            for choice in choices:
                if isinstance(choice, dict) and isinstance(choice.get("token_ids"), list):
                    token_timestamps.extend([now] * len(choice["token_ids"]))
    finished_at = time.monotonic()
    if not token_timestamps:
        raise RuntimeError(f"request {request_id} returned no token IDs; enable return_token_ids")
    observed_completion_tokens = max(observed_completion_tokens, len(token_timestamps))
    return CompletionResult(
        request_id=request_id,
        prompt_tokens=observed_prompt_tokens,
        completion_tokens=observed_completion_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
        ttft_seconds=token_timestamps[0] - started_at,
        decode_seconds=finished_at - token_timestamps[0],
        e2e_latency_seconds=finished_at - started_at,
        token_timestamps_monotonic=tuple(token_timestamps),
    )


def parse_positive_int_list(value: str) -> list[int]:
    """Parse a comma-separated list of positive integers."""
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("expected comma-separated positive integers") from exc
    if not values or any(item <= 0 for item in values):
        raise ValueError("expected comma-separated positive integers")
    return values


def parse_positive_int(value: str) -> int:
    """Parse one positive integer for an argparse option."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def parse_nonnegative_int(value: str) -> int:
    """Parse one non-negative integer for an argparse option."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def parse_positive_float(value: str) -> float:
    """Parse one finite positive float for an argparse option."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return parsed


def parse_nonnegative_float(value: str) -> float:
    """Parse one finite non-negative float for an argparse option."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a non-negative number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("expected a finite non-negative number")
    return parsed


def parse_unit_fraction(value: str) -> float:
    """Parse one finite fraction in the interval (0, 1]."""
    parsed = parse_positive_float(value)
    if parsed > 1:
        raise argparse.ArgumentTypeError("expected a fraction in (0, 1]")
    return parsed


def prepare_stage_directory(root: Path, stage: str, overwrite: bool) -> Path:
    """Create one stage directory without deleting the other stage."""
    path = root / stage
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"stage directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, value: object) -> None:
    """Atomically write one human-readable JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def nonnegative_least_squares(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve a small nonnegative least-squares fit by active-set enumeration."""
    if design.ndim != 2 or target.ndim != 1 or design.shape[0] != target.shape[0]:
        raise ValueError("nonnegative fit requires paired 2-D design and 1-D target")
    best = np.zeros(design.shape[1], dtype=float)
    best_residual = math.inf
    for mask in range(1, 1 << design.shape[1]):
        active = [index for index in range(design.shape[1]) if mask & (1 << index)]
        selected, _, _, _ = np.linalg.lstsq(design[:, active], target, rcond=None)
        if np.any(selected < -1e-12):
            continue
        candidate = np.zeros(design.shape[1], dtype=float)
        candidate[active] = np.maximum(selected, 0)
        residual = float(np.sum((target - design @ candidate) ** 2))
        if residual < best_residual:
            best, best_residual = candidate, residual
    return best
