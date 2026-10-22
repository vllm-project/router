# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Load router-replay case JSON and build runtime configuration."""

from __future__ import annotations

import json
import os
import shutil

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML optional at import time
    yaml = None  # type: ignore[assignment]
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from helpers.case_loader import (
    BenchmarkCase,
    E2EConfig,
    PerfScenario,
    _fetch_hardware_slug_from_platform,
    _hardware_slug_from_mark,
    _make_run_tag,
    _parse_deploy_case,
    _repo_root,
    ensure_directory,
)
from helpers.paths import DEFAULT_ROUTER_BINARY, repo_root
from helpers.router_server import RouterDeployConfig

ROUTER_REPLAY_WORKLOAD = "router-replay"
_LEGACY_ROUTER_BINARY = "/path/to/router/target/release/vllm-router"
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "0.0.0.0", "::"})


@dataclass(frozen=True)
class RouterReplayE2EConfig:
    """Runtime state for one router-replay perf case."""

    case: BenchmarkCase
    router_arm: str
    router: RouterDeployConfig
    vllm_connect_host: str
    vllm_port: int
    router_base_url: str
    bench_config: Path
    model: str
    task_num: int
    max_concurrency: int
    task_timeout_seconds: int
    result_root: Path
    vllm_log_dir: Path
    run_tag: str
    wait_for_vllm_ready: bool

    @classmethod
    def from_case_file(
        cls,
        case_file: Path,
        *,
        task_num: int | None = None,
        max_concurrency: int | None = None,
    ) -> RouterReplayE2EConfig:
        path = case_file.expanduser().resolve()
        if not path.is_file() or path.suffix != ".json":
            raise ValueError(
                f"--test-config-file must point to a .json file: {case_file}"
            )
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError(f"case JSON root must be an object: {case_file}")

        router_arm = payload.get("router_arm")
        if not isinstance(router_arm, str) or not router_arm.strip():
            raise ValueError(f"router-replay case missing router_arm: {case_file}")

        benchmark = payload.get("benchmark_params")
        if not isinstance(benchmark, dict):
            raise ValueError(
                f"router-replay case missing benchmark_params: {case_file}"
            )
        if benchmark.get("workload") != ROUTER_REPLAY_WORKLOAD:
            raise ValueError(
                f"benchmark_params.workload must be {ROUTER_REPLAY_WORKLOAD!r}: {case_file}"
            )

        router_params = payload.get("router_params")
        if not isinstance(router_params, dict):
            raise ValueError(f"router-replay case missing router_params: {case_file}")

        wait_for_vllm_ready = payload.get("wait_for_vllm_ready", True)
        if not isinstance(wait_for_vllm_ready, bool):
            raise ValueError(f"wait_for_vllm_ready must be boolean: {case_file}")

        repo = _repo_root()
        deploy = _parse_deploy_case(
            router_arm.strip(),
            {
                "server_params": payload.get("server_params"),
                "serve_env": payload.get("serve_env", {}),
                "result_root": payload.get("result_root", "test-results/agentaware"),
            },
            case_label=f"router-replay case {path.name}",
        )
        if not deploy.result_root.is_absolute():
            deploy = replace(deploy, result_root=(repo / deploy.result_root).resolve())

        serve_args = deploy.serve_args
        vllm_port = int(serve_args.get("port", 8077))
        bind_host = str(serve_args.get("host", "127.0.0.1"))
        vllm_connect_host = "127.0.0.1" if bind_host in {"0.0.0.0", "::"} else bind_host

        config_path = Path(str(benchmark.get("config", ""))).expanduser()
        if not config_path.is_absolute():
            config_path = (repo / config_path).resolve()

        run_tag = _make_run_tag()
        result_root = deploy.result_root

        router = _parse_router_deploy(
            router_params,
            repo=repo,
            log_dir=result_root / "router-logs",
            log_label=f"{router_arm.strip()}-{run_tag}",
        )
        router_host = router.host
        connect_router_host = (
            "127.0.0.1" if router_host in {"0.0.0.0", "::"} else router_host
        )
        router_base_url = f"http://{connect_router_host}:{router.port}"

        return cls(
            case=BenchmarkCase(
                test_name=str(payload.get("test_name", path.stem)),
                scenario=PerfScenario.BASELINE,
                description=str(payload.get("description", "")),
                mark=(
                    tuple(payload.get("mark", []))
                    if isinstance(payload.get("mark"), list)
                    else ()
                ),
                deploy=deploy,
                benchmark_params=dict(benchmark),
                wait_for_vllm_ready=wait_for_vllm_ready,
            ),
            router_arm=router_arm.strip(),
            router=router,
            vllm_connect_host=vllm_connect_host,
            vllm_port=vllm_port,
            router_base_url=router_base_url,
            bench_config=config_path,
            model=str(benchmark.get("model", "glm-5")),
            task_num=int(
                task_num if task_num is not None else benchmark.get("task-num", 8)
            ),
            max_concurrency=int(
                max_concurrency
                if max_concurrency is not None
                else benchmark.get("max-concurrency", 4)
            ),
            task_timeout_seconds=int(benchmark.get("timeout", 3600)),
            result_root=result_root,
            vllm_log_dir=result_root / "vllm-logs",
            run_tag=run_tag,
            wait_for_vllm_ready=wait_for_vllm_ready,
        )

    def to_vllm_e2e_config(self) -> E2EConfig:
        """Adapt to shared vLLM lifecycle configuration."""

        e2e = E2EConfig.from_case(self.case)
        return replace(
            e2e,
            host=self.vllm_connect_host,
            port=self.vllm_port,
            bench_config=self.bench_config,
            task_num=self.task_num,
            max_concurrency=self.max_concurrency,
            task_timeout_seconds=self.task_timeout_seconds,
            result_root=self.result_root,
            vllm_log_dir=self.vllm_log_dir,
            run_tag=self.run_tag,
            wait_for_vllm_ready=self.wait_for_vllm_ready,
            agent_profile="single",
        )

    def make_result_dir(self) -> Path:
        hardware_slug = (
            _fetch_hardware_slug_from_platform()
            or _hardware_slug_from_mark(self.case.mark)
            or "unknown"
        )
        host_slug = (
            "local"
            if self.vllm_connect_host in _LOCAL_HOSTS
            else self.vllm_connect_host.replace(".", "-")
        )
        name = (
            f"run-{hardware_slug}-{host_slug}-router-replay-{self.router_arm}-"
            f"{self.task_num}-{self.max_concurrency}-{self.run_tag}"
        )
        # Replay bench is not sudo-wrapped (no plan-subagent); unlike swebench_run.
        ensure_directory(self.result_root, run_as_user=None)
        return self.result_root / name

    def validate_prerequisites(self) -> list[str]:
        blockers: list[str] = []
        if not self.case.deploy.model.exists():
            blockers.append(f"model path does not exist: {self.case.deploy.model}")
        if not shutil.which("vllm"):
            blockers.append("vllm executable is unavailable on PATH")
        if not self.bench_config.is_file():
            blockers.append(f"replay config does not exist: {self.bench_config}")
        else:
            trace_path = _replay_trace_path(self.bench_config, repo_root())
            if trace_path is not None and not trace_path.is_file():
                blockers.append(
                    f"replay trace does not exist: {trace_path} "
                    f"(set replay.trace_path in {self.bench_config})"
                )
        if not _router_binary_available(self.router.binary):
            blockers.append(
                f"router binary does not exist: {self.router.binary} "
                f"(build with `cargo build --release` or set VLLM_ROUTER_BINARY)"
            )
        if not self.router.program_scheduling_config.is_file():
            blockers.append(
                "program scheduling config does not exist: "
                f"{self.router.program_scheduling_config} "
                "(run L1 once: python py_test/e2e/perf/agentaware/prepare_l1.py "
                "--serve-config-file py_test/e2e/perf/agentaware/configs/l1-serve.json)"
            )
        return blockers


def _replay_trace_path(bench_config: Path, repo: Path) -> Path | None:
    if yaml is None:
        return None
    payload = yaml.safe_load(bench_config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    replay = payload.get("replay")
    if not isinstance(replay, dict):
        return None
    raw = replay.get("trace_path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        path = (repo / path).resolve()
    return path


def _router_binary_available(binary: Path) -> bool:
    if binary.is_file():
        return True
    return shutil.which(str(binary)) is not None


def _resolve_router_binary(raw: str, *, repo: Path) -> Path:
    override = os.environ.get("VLLM_ROUTER_BINARY", "").strip()
    if override:
        raw = override
    stripped = raw.strip()
    if not stripped or stripped == _LEGACY_ROUTER_BINARY:
        candidate = (repo / DEFAULT_ROUTER_BINARY).resolve()
        if candidate.is_file():
            return candidate
        stripped = str(DEFAULT_ROUTER_BINARY)
    binary = Path(stripped).expanduser()
    if not binary.is_absolute():
        binary = (repo / binary).resolve()
    if binary.is_file():
        return binary
    on_path = shutil.which(stripped)
    if on_path:
        return Path(on_path)
    return binary


def _parse_router_deploy(
    params: dict[str, Any],
    *,
    repo: Path,
    log_dir: Path,
    log_label: str,
) -> RouterDeployConfig:
    binary_raw = params.get("binary")
    if not isinstance(binary_raw, str) or not binary_raw.strip():
        binary_raw = str(DEFAULT_ROUTER_BINARY)
    binary = _resolve_router_binary(binary_raw, repo=repo)

    program_raw = params.get("program_scheduling_config")
    if not isinstance(program_raw, str) or not program_raw.strip():
        raise ValueError("router_params.program_scheduling_config is required")
    program_path = Path(program_raw.strip()).expanduser()
    if not program_path.is_absolute():
        program_path = (repo / program_path).resolve()

    worker_urls_raw = params.get("worker_urls")
    if not isinstance(worker_urls_raw, list) or not worker_urls_raw:
        raise ValueError("router_params.worker_urls must be a non-empty list")
    worker_urls = tuple(
        str(item).strip() for item in worker_urls_raw if str(item).strip()
    )

    host = str(params.get("host", "0.0.0.0"))
    port = int(params.get("port", 3001))
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    ready_url = params.get("ready_url")
    if not isinstance(ready_url, str) or not ready_url.strip():
        ready_url = f"http://{connect_host}:{port}/health"

    return RouterDeployConfig(
        binary=binary,
        host=host,
        port=port,
        metrics_port=int(params.get("metrics_port", 29000)),
        worker_urls=worker_urls,
        intra_node_data_parallel_size=int(
            params.get("intra_node_data_parallel_size", 1)
        ),
        program_scheduling_config=program_path,
        ready_url=str(ready_url).strip(),
        ready_timeout_seconds=int(params.get("ready_timeout_seconds", 600)),
        log_dir=log_dir,
        log_label=log_label,
    )
