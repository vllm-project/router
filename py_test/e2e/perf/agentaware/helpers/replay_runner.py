# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Orchestrate vLLM serve, Router, and replay bench for one router-replay case."""

from __future__ import annotations

import subprocess
from pathlib import Path

from helpers.paths import repo_root
from helpers.router_server import managed_router
from helpers.server import managed_vllm
from _siblings import import_sibling_module

_case_config = import_sibling_module("case_config")
_replay_cli = import_sibling_module("replay_cli")
RouterReplayE2EConfig = _case_config.RouterReplayE2EConfig
build_replay_argv = _replay_cli.build_replay_argv
build_replay_env = _replay_cli.build_replay_env


def run_router_replay(
    config: RouterReplayE2EConfig, *, result_dir: Path | None = None
) -> Path:
    """Run one router-replay case and return the finalized result directory."""

    blockers = config.validate_prerequisites()
    if blockers:
        raise RuntimeError("; ".join(blockers))

    output_dir = result_dir or config.make_result_dir()
    if output_dir.exists():
        raise FileExistsError(f"result directory already exists: {output_dir}")

    vllm_e2e = config.to_vllm_e2e_config()
    prefix = f"[agentaware:{config.router_arm}] "
    print(f"{prefix}hardware slug target dir: {output_dir.name}", flush=True)

    argv = build_replay_argv(config, output_dir)
    with managed_vllm(vllm_e2e):
        with managed_router(config.router):
            print(f"{prefix}running: {' '.join(argv)}", flush=True)
            completed = subprocess.run(
                argv,
                check=False,
                text=True,
                capture_output=True,
                env=build_replay_env(),
                cwd=repo_root(),
            )
            if completed.returncode != 0:
                if completed.stdout:
                    print(completed.stdout, flush=True)
                if completed.stderr:
                    print(completed.stderr, flush=True)
                raise RuntimeError(
                    f"replay bench exited with code {completed.returncode}"
                )

    return output_dir
