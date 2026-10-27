# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Build argv for ``vllm bench serve --agentinfer replay`` in router-replay perf."""

from __future__ import annotations

import os
from pathlib import Path

from _siblings import import_sibling_module

RouterReplayE2EConfig = import_sibling_module("case_config").RouterReplayE2EConfig

_REPLAY_CLI_FLAGS: tuple[tuple[str, str], ...] = (
    ("endpoint", "--endpoint"),
    ("trace-type", "--trace-type"),
    ("interval-mode", "--interval-mode"),
    ("trace-path", "--trace-path"),
)


def build_replay_argv(config: RouterReplayE2EConfig, result_dir: Path) -> list[str]:
    """Replay traffic goes to the Router; metrics may target the worker."""

    argv = [
        "vllm",
        "bench",
        "serve",
        "--agentinfer",
        "replay",
        "--config",
        str(config.bench_config),
        "--result-dir",
        str(result_dir),
        "--model",
        config.model,
        "--task-num",
        str(config.task_num),
        "--max-concurrency",
        str(config.max_concurrency),
        "--base-url",
        config.router_base_url,
    ]
    benchmark = config.case.benchmark_params
    metrics_url = benchmark.get("worker_metrics_url") or benchmark.get("metrics-url")
    if isinstance(metrics_url, str) and metrics_url.strip():
        argv.extend(["--metrics-url", metrics_url.strip()])
    for param_key, cli_flag in _REPLAY_CLI_FLAGS:
        raw = benchmark.get(param_key)
        if isinstance(raw, str) and raw.strip():
            argv.extend([cli_flag, raw.strip()])
    return argv


def build_replay_env() -> dict[str, str]:
    env = os.environ.copy()
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["TQDM_DISABLE"] = "1"
    return env
