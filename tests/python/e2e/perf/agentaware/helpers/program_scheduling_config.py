# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Build Router program-scheduling JSON from a calibration profile."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_program_scheduling_config(
    profile: dict[str, Any],
    *,
    token_capacity_per_dp_rank: int,
    binding_only: bool = False,
) -> dict[str, Any]:
    """Return one program-scheduling config dict (PR 286 / Router replay shape)."""

    prefill = profile["calibration"]["prefill"]["coefficients"]
    decode = profile["calibration"]["decode"]["coefficients"]
    config: dict[str, Any] = {
        "program_scheduling_enable_key": "vllm_xargs.agentic_context",
        "binding_only": bool(binding_only),
        "global_queue": True,
        "resume_order": "mru",
        "binding_strategy": "least_program_count",
        "token_capacity_per_dp_rank": int(token_capacity_per_dp_rank),
        "metrics_interval_seconds": 1.0,
        "admission_waiting_request_threshold": 1,
        "prefill_cost_model": {
            "intercept_seconds": prefill["ttl_prefill_model_intercept_seconds"],
            "linear_seconds_per_1k_tokens": prefill[
                "ttl_prefill_model_linear_seconds_per_1k_tokens"
            ],
            "quadratic_seconds_per_1k_tokens_squared": prefill[
                "ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared"
            ],
            "decode_throughput_alpha": 0.15,
        },
        "decode_throughput_model": {
            "fixed_step_seconds": decode["decode_step_fixed_seconds"],
            "batch_step_seconds_per_request": decode["decode_step_seconds_per_request"],
            "context_step_seconds_per_token": decode[
                "decode_step_seconds_per_context_token"
            ],
        },
    }
    return config


def write_program_scheduling_configs(
    calibration_dir: Path,
    token_capacity_per_dp_rank: int,
    program_scheduling_path: Path,
    binding_only_path: Path,
) -> None:
    """Write full and binding-only program scheduling JSON from ``calibration.json``."""

    profile_path = calibration_dir / "calibration.json"
    if not profile_path.is_file():
        raise FileNotFoundError(f"calibration profile not found: {profile_path}")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    program_scheduling_path.parent.mkdir(parents=True, exist_ok=True)
    binding_only_path.parent.mkdir(parents=True, exist_ok=True)

    full = build_program_scheduling_config(
        profile,
        token_capacity_per_dp_rank=token_capacity_per_dp_rank,
        binding_only=False,
    )
    baseline = build_program_scheduling_config(
        profile,
        token_capacity_per_dp_rank=token_capacity_per_dp_rank,
        binding_only=True,
    )
    program_scheduling_path.write_text(
        json.dumps(full, indent=2) + "\n", encoding="utf-8"
    )
    binding_only_path.write_text(
        json.dumps(baseline, indent=2) + "\n", encoding="utf-8"
    )


def read_observed_kv_capacity_tokens(calibration_dir: Path) -> int | None:
    """Return decode ``observed_kv_capacity_tokens`` when ``decode/run_config.json`` exists."""

    run_config = calibration_dir / "decode" / "run_config.json"
    if not run_config.is_file():
        return None
    payload = json.loads(run_config.read_text(encoding="utf-8"))
    raw = payload.get("observed_kv_capacity_tokens")
    if isinstance(raw, int) and raw > 0:
        return raw
    if isinstance(raw, float) and raw > 0:
        return int(raw)
    return None
