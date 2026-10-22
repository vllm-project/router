# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Build vLLM lifecycle config for router-replay L1 prepare from a serve JSON file."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from helpers.case_loader import (
    BenchmarkCase,
    E2EConfig,
    PerfScenario,
    _parse_deploy_case,
    _repo_root,
)
from helpers.server import managed_vllm

_LOCAL_BIND_HOSTS = frozenset({"127.0.0.1", "localhost", "0.0.0.0", "::"})


def _vllm_connect_host(bind_host: str) -> str:
    if bind_host in _LOCAL_BIND_HOSTS:
        return "127.0.0.1"
    return bind_host


def load_serve_config_payload(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"serve config not found: {path}")
    payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"serve config root must be an object: {path}")
    return payload


def calibration_model_id(
    payload: dict[str, Any], deploy_serve_args: dict[str, Any]
) -> str:
    raw = payload.get("calibration_model")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    benchmark = payload.get("benchmark_params")
    if isinstance(benchmark, dict):
        model = benchmark.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
    served = deploy_serve_args.get("served-model-name")
    if isinstance(served, str) and served.strip():
        return served.strip()
    raise ValueError(
        "calibration model id is required: set calibration_model, benchmark_params.model, "
        "or server_params.serve_args.served-model-name in the serve config"
    )


def build_l1_vllm_e2e_config(serve_config_file: Path) -> tuple[E2EConfig, str]:
    """Return vLLM E2E config and the OpenAI model id used for calibration requests."""

    payload = load_serve_config_payload(serve_config_file)
    repo = _repo_root()
    deploy = _parse_deploy_case(
        "l1-prepare",
        {
            "server_params": payload.get("server_params"),
            "serve_env": payload.get("serve_env", {}),
            "result_root": payload.get(
                "result_root", "test-results/agentaware/l1-prep"
            ),
        },
        case_label=f"L1 serve config {serve_config_file.name}",
    )
    if not deploy.result_root.is_absolute():
        deploy = replace(deploy, result_root=(repo / deploy.result_root).resolve())

    serve_args = deploy.serve_args
    bind_host = str(serve_args.get("host", "127.0.0.1"))
    connect_host = _vllm_connect_host(bind_host)
    port = int(serve_args.get("port", 8077))
    model_id = calibration_model_id(payload, serve_args)

    wait_for_vllm_ready = payload.get("wait_for_vllm_ready", True)
    if not isinstance(wait_for_vllm_ready, bool):
        raise ValueError("wait_for_vllm_ready must be boolean")

    ready_timeout = payload.get("serve_ready_timeout_seconds", 7200)
    if not isinstance(ready_timeout, int) or ready_timeout <= 0:
        raise ValueError("serve_ready_timeout_seconds must be a positive integer")

    result_root = deploy.result_root
    case = BenchmarkCase(
        test_name="router-replay-l1-prepare",
        scenario=PerfScenario.BASELINE,
        description=str(
            payload.get("description", "router-replay L1 calibration serve")
        ),
        mark=(),
        deploy=deploy,
        benchmark_params={
            "model": model_id,
            "host": connect_host,
            "port": port,
            "timeout": ready_timeout,
            "config": "py_test/e2e/perf/agentaware/configs/replay-full.yaml",
            "prepare_dataset": "noop",
            "prepare_output_dir": str(result_root / "noop-dataset"),
        },
        wait_for_vllm_ready=wait_for_vllm_ready,
    )
    e2e = E2EConfig.from_case(case, repo=repo)
    return e2e, model_id


__all__ = ["build_l1_vllm_e2e_config", "managed_vllm"]
