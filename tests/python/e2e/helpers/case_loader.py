# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Load E2E case JSON, build runtime config, and wrap plan-subagent subprocesses."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import stat
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_BASE_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost"})

PLAN_SUBAGENT_PROFILE = "plan-subagent"
_PRESERVED_ENV_KEYS = (
    "PATH",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "TQDM_DISABLE",
    "ANTHROPIC_AUTH_TOKEN",
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "TORCH_DEVICE_BACKEND_AUTOLOAD",
)
_PRESERVED_ENV_PREFIXES = (
    "ASCEND_",
    "CAN" + "N_",
    "HCCL_",
    "PYTORCH_NPU",
    "VLLM_",
    "OMP_",
    "MKL_",
)
_HARDWARE_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")


class PerfScenario(str, Enum):
    """Supported E2E deployment scenarios."""

    BASELINE = "baseline"
    AGENTINFER = "agentinfer"


@dataclass(frozen=True)
class VllmDeployConfig:
    """vLLM serve deployment parameters for one benchmark case."""

    name: str
    model: Path
    serve_args: dict[str, Any]
    middleware: tuple[str, ...]
    serve_env: dict[str, str]
    result_root: Path


@dataclass(frozen=True)
class BenchmarkCase:
    """One self-contained E2E benchmark case loaded from JSON."""

    test_name: str
    scenario: PerfScenario
    description: str
    mark: tuple[Any, ...]
    deploy: VllmDeployConfig
    benchmark_params: dict[str, Any]
    wait_for_vllm_ready: bool


@dataclass(frozen=True)
class E2EConfig:
    """Runtime state for one E2E case (perf benchmark or shared vLLM serve lifecycle)."""

    case: BenchmarkCase
    scenario: PerfScenario
    host: str
    port: int
    bench_config: Path
    prepare_dataset: str
    dataset_output_dir: Path
    agent_executable: Path
    agent_profile: str
    task_num: int
    max_concurrency: int
    task_timeout_seconds: int
    result_root: Path
    lifecycle_socket: str
    vllm_log_dir: Path
    run_tag: str
    wait_for_vllm_ready: bool
    benchmark_run_as_user: str | None

    @property
    def repo_root(self) -> Path:
        return _repo_root()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def model_path(self) -> Path:
        return self.case.deploy.model

    @property
    def served_model_name(self) -> str:
        return str(self.case.benchmark_params.get("model", "glm-5"))

    @classmethod
    def from_case(cls, case: BenchmarkCase, *, repo: Path | None = None) -> E2EConfig:
        repo = repo or _repo_root()
        benchmark = case.benchmark_params
        run_tag = _make_run_tag()
        lifecycle_socket = f"/tmp/agentinfer-e2e-{run_tag}.sock"

        config_path = benchmark.get(
            "config", "agentinfer/agentbench/configs/swebench_vllm.yaml"
        )
        bench_config = Path(str(config_path)).expanduser()
        if not bench_config.is_absolute():
            bench_config = (repo / bench_config).resolve()

        prepare_dir = benchmark.get(
            "prepare_output_dir", "agentinfer/agentbench/data/swebench"
        )
        dataset_output_dir = Path(str(prepare_dir)).expanduser()
        if not dataset_output_dir.is_absolute():
            dataset_output_dir = (repo / dataset_output_dir).resolve()

        prepare_dataset = benchmark.get("prepare_dataset", "swebench")
        if not isinstance(prepare_dataset, str) or not prepare_dataset:
            raise ValueError(
                "benchmark_params.prepare_dataset must be a non-empty string"
            )

        agent_executable = Path(
            str(benchmark.get("agent-executable", "claude"))
        ).expanduser()
        if not agent_executable.is_absolute() and len(agent_executable.parts) > 1:
            agent_executable = (repo / agent_executable).resolve()

        result_root = _resolve_repo_path(case.deploy.result_root, repo)
        benchmark_run_as_user_raw = benchmark.get("benchmark-run-as-user")
        benchmark_run_as_user = (
            str(benchmark_run_as_user_raw).strip()
            if isinstance(benchmark_run_as_user_raw, str)
            and benchmark_run_as_user_raw.strip()
            else None
        )

        return cls(
            case=case,
            scenario=case.scenario,
            host=str(benchmark.get("host", DEFAULT_HOST)),
            port=int(benchmark.get("port", DEFAULT_PORT)),
            bench_config=bench_config,
            prepare_dataset=prepare_dataset,
            dataset_output_dir=dataset_output_dir,
            agent_executable=agent_executable,
            agent_profile=str(benchmark.get("agent-profile", "plan-subagent")),
            task_num=int(benchmark.get("task-num", 8)),
            max_concurrency=int(benchmark.get("max-concurrency", 4)),
            task_timeout_seconds=int(benchmark.get("timeout", 3600)),
            result_root=result_root,
            lifecycle_socket=lifecycle_socket,
            vllm_log_dir=result_root / "vllm-logs",
            run_tag=run_tag,
            wait_for_vllm_ready=case.wait_for_vllm_ready,
            benchmark_run_as_user=benchmark_run_as_user,
        )

    def deploy(self) -> VllmDeployConfig:
        return self.case.deploy

    def effective_benchmark_run_as_user(self) -> str | None:
        return resolve_benchmark_run_as_user(
            agent_profile=self.agent_profile,
            benchmark_run_as_user=self.benchmark_run_as_user,
        )

    def make_result_dir(self, *, hardware_slug: str) -> Path:
        host_slug = (
            "local" if self.host in _LOCAL_HOSTS else self.host.replace(".", "-")
        )
        name = (
            f"run-{hardware_slug}-{host_slug}-{self.scenario.value}-"
            f"{self.task_num}-{self.max_concurrency}-{self.run_tag}"
        )
        ensure_directory(
            self.result_root, run_as_user=self.effective_benchmark_run_as_user()
        )
        return self.result_root / name

    def resolve_run_hardware_slug(self) -> str:
        """Resolve the hardware slug for ``run-<slug>-...`` directory names."""

        detected = _fetch_hardware_slug_from_platform()
        if detected:
            return detected

        mark_slug = _hardware_slug_from_mark(self.case.mark)
        if mark_slug:
            return mark_slug

        return "unknown"

    def uses_default_bind(self) -> bool:
        return self.host in _LOCAL_HOSTS and self.port == DEFAULT_PORT

    def uses_default_base_url(self) -> bool:
        return self.base_url.rstrip("/") == DEFAULT_BASE_URL

    def with_benchmark_load(
        self, *, task_num: int | None = None, max_concurrency: int | None = None
    ) -> E2EConfig:
        updates: dict[str, int] = {}
        if task_num is not None:
            updates["task_num"] = task_num
        if max_concurrency is not None:
            updates["max_concurrency"] = max_concurrency
        return replace(self, **updates) if updates else self

    def with_runtime_overrides(
        self,
        *,
        benchmark_run_as_user: str | None = None,
        benchmark_run_as_user_set: bool = False,
    ) -> E2EConfig:
        if not benchmark_run_as_user_set:
            return self
        return replace(self, benchmark_run_as_user=benchmark_run_as_user)

    def validate_prerequisites(self) -> list[str]:
        blockers: list[str] = []
        if not self.case.deploy.model.exists():
            blockers.append(
                f"model path does not exist for {self.scenario.value}: {self.case.deploy.model}"
            )
        if not shutil.which("vllm"):
            blockers.append("vllm executable is unavailable on PATH")
        if not self.bench_config.exists():
            blockers.append(f"bench config does not exist: {self.bench_config}")
        if not _agent_executable_available(self.agent_executable):
            blockers.append(f"agent executable does not exist: {self.agent_executable}")
        return blockers


def _repo_root() -> Path:
    from .paths import repo_root

    return repo_root()


def _resolve_repo_path(path: Path, repo: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (repo / expanded).resolve()


def _make_run_tag() -> str:
    """Return a unique run tag for result dirs, logs, and lifecycle sockets."""

    return f"{datetime.now().strftime('%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def _agent_executable_available(path: Path) -> bool:
    if path.is_absolute() or len(path.parts) > 1:
        return path.exists()
    return shutil.which(str(path)) is not None


def _infer_scenario(payload: dict[str, Any], case_path: Path) -> PerfScenario:
    """Resolve scenario from JSON ``scenario`` (or legacy ``arm``) or parent directory."""

    scenario_raw = payload.get("scenario", payload.get("arm"))
    if isinstance(scenario_raw, str) and scenario_raw:
        try:
            return PerfScenario(scenario_raw.lower())
        except ValueError as exc:
            raise ValueError(
                f"invalid scenario {scenario_raw!r} in {case_path}; expected 'baseline' or 'agentinfer'"
            ) from exc

    for part in case_path.resolve().parts:
        if part == PerfScenario.BASELINE.value:
            return PerfScenario.BASELINE
        if part == PerfScenario.AGENTINFER.value:
            return PerfScenario.AGENTINFER

    raise ValueError(
        f'cannot infer scenario for {case_path}; set "scenario": "baseline" or "agentinfer" in the JSON'
    )


def load_case_file(case_file: Path) -> BenchmarkCase:
    """Load one ``.json`` case file."""

    path = case_file.expanduser()
    if not path.is_file() or path.suffix != ".json":
        raise ValueError(f"--test-config-file must point to a .json file: {case_file}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"case JSON root must be an object: {case_file}")
    scenario = _infer_scenario(payload, path)
    return _parse_case(payload, scenario=scenario)


def _parse_case(payload: dict[str, Any], *, scenario: PerfScenario) -> BenchmarkCase:
    test_name = payload.get("test_name")
    if not isinstance(test_name, str) or not test_name:
        raise ValueError(f"{scenario.value} case missing non-empty test_name")

    benchmark_params = payload.get("benchmark_params")
    if not isinstance(benchmark_params, dict):
        raise ValueError(
            f"{scenario.value} case {test_name!r} missing benchmark_params object"
        )

    serve_env_raw = payload.get("serve_env", {})
    if not isinstance(serve_env_raw, dict):
        raise ValueError(
            f"{scenario.value} case {test_name!r} serve_env must be an object"
        )

    mark_raw = payload.get("mark", [])
    if not isinstance(mark_raw, list):
        raise ValueError(f"{scenario.value} case {test_name!r} mark must be a list")

    wait_for_vllm_ready = payload.get("wait_for_vllm_ready", True)
    if not isinstance(wait_for_vllm_ready, bool):
        raise ValueError(
            f"{scenario.value} case {test_name!r} wait_for_vllm_ready must be a boolean"
        )

    default_result_root = (
        "test-results/vllm"
        if scenario is PerfScenario.BASELINE
        else "test-results/agentinfer"
    )
    deploy = _parse_deploy_case(
        scenario.value,
        {
            "server_params": payload.get("server_params"),
            "serve_env": serve_env_raw,
            "result_root": payload.get("result_root", default_result_root),
        },
        case_label=f"{scenario.value} case {test_name!r}",
    )

    return BenchmarkCase(
        test_name=test_name,
        scenario=scenario,
        description=str(payload.get("description", "")),
        mark=tuple(mark_raw),
        deploy=deploy,
        benchmark_params=dict(benchmark_params),
        wait_for_vllm_ready=wait_for_vllm_ready,
    )


def _sanitize_hardware_slug(value: str) -> str:
    slug = value.strip()
    if not slug:
        raise ValueError("hardware_slug must be a non-empty string")
    slug = slug.replace(" ", "-")
    if not _HARDWARE_SLUG_RE.fullmatch(slug):
        raise ValueError(
            f"invalid hardware_slug {value!r}; use letters, digits, '.', '_', or '-'"
        )
    return slug


def _fetch_hardware_slug_from_platform(device_id: int = 0) -> str | None:
    try:
        from vllm.platforms import current_platform
    except ImportError:
        return None

    try:
        device_name = current_platform.get_device_name(device_id)
    except NotImplementedError:
        return None
    except Exception:
        return None

    if not isinstance(device_name, str) or not device_name.strip():
        return None
    try:
        return _sanitize_hardware_slug(device_name.strip())
    except ValueError:
        return None


def _hardware_slug_from_mark(mark: tuple[Any, ...]) -> str | None:
    for entry in mark:
        if not isinstance(entry, dict):
            continue
        hardware_marks = entry.get("hardware_marks")
        if not isinstance(hardware_marks, dict):
            continue
        for key in ("slug", "model", "device"):
            raw = hardware_marks.get(key)
            if isinstance(raw, str) and raw.strip():
                return _sanitize_hardware_slug(raw.strip())
        res = hardware_marks.get("res")
        if isinstance(res, dict):
            for value in res.values():
                if isinstance(value, str) and value.strip():
                    return _sanitize_hardware_slug(value.strip())
    return None


def _parse_deploy_case(
    name: str, spec: dict[str, Any], *, case_label: str
) -> VllmDeployConfig:
    server_params = spec.get("server_params")
    if not isinstance(server_params, dict):
        raise ValueError(f"{case_label} missing server_params")
    model_raw = server_params.get("model")
    if not isinstance(model_raw, str) or not model_raw:
        raise ValueError(f"{case_label} missing server_params.model")
    serve_args = server_params.get("serve_args", {})
    if not isinstance(serve_args, dict):
        raise ValueError(f"{case_label} serve_args must be an object")
    middleware_raw = server_params.get("middleware", [])
    if isinstance(middleware_raw, str):
        middleware = (middleware_raw,)
    elif isinstance(middleware_raw, list):
        middleware = tuple(str(item) for item in middleware_raw)
    else:
        raise ValueError(f"{case_label} middleware must be a list")
    serve_env_raw = spec.get("serve_env", {})
    if not isinstance(serve_env_raw, dict):
        raise ValueError(f"{case_label} serve_env must be an object")
    result_root = Path(str(spec.get("result_root", "test-results")))
    return VllmDeployConfig(
        name=name,
        model=Path(model_raw).expanduser(),
        serve_args=dict(serve_args),
        middleware=middleware,
        serve_env={str(k): str(v) for k, v in serve_env_raw.items()},
        result_root=result_root,
    )


def serve_args_to_argv(
    model: Path, serve_args: dict[str, Any], middleware: tuple[str, ...]
) -> list[str]:
    """Convert a serve_args mapping into a ``vllm serve`` argv tail."""

    argv = ["vllm", "serve", str(model)]
    for key, value in serve_args.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if isinstance(value, dict | list):
            argv.extend([flag, json.dumps(value, separators=(",", ":"))])
            continue
        argv.extend([flag, str(value)])
    for entry in middleware:
        argv.extend(["--middleware", entry])
    return argv


def benchmark_params_to_run_args(
    params: dict[str, Any], result_dir: Path, repo: Path
) -> list[str]:
    """Convert benchmark_params into BenchKit ``run`` argv flags."""

    argv = [
        "vllm",
        "bench",
        "serve",
        "--agentinfer",
        "run",
        "--result-dir",
        str(result_dir),
    ]
    flag_map = {
        "config": "--config",
        "model": "--model",
        "agent-executable": "--agent-executable",
        "agent-profile": "--agent-profile",
        "task-num": "--task-num",
        "max-concurrency": "--max-concurrency",
        "timeout": "--timeout",
    }
    for key, flag in flag_map.items():
        if key not in params:
            continue
        value = params[key]
        if key == "config":
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = (repo / path).resolve()
            argv.extend([flag, str(path)])
        else:
            argv.extend([flag, str(value)])

    host = params.get("host", "127.0.0.1")
    port = int(params.get("port", 8000))
    if host != "127.0.0.1" or port != 8000:
        argv.extend(["--base-url", f"http://{host}:{port}"])
    return argv


def _sudo_env_argv() -> list[str]:
    """Return ``env KEY=VALUE ...`` assignments copied from the current process."""

    values: dict[str, str] = {}
    for key in _PRESERVED_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            values[key] = value
    for key, value in os.environ.items():
        if any(key.startswith(prefix) for prefix in _PRESERVED_ENV_PREFIXES):
            values[key] = value
    values.setdefault("PATH", os.environ.get("PATH", ""))
    values["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    values["TQDM_DISABLE"] = "1"
    return ["env", *[f"{key}={values[key]}" for key in sorted(values)]]


def _running_as_root_for_user(
    *, run_as_user: str | None, effective_uid: int | None = None
) -> bool:
    if os.name == "nt" or not run_as_user:
        return False
    uid = os.geteuid() if effective_uid is None else effective_uid
    return uid == 0


def _chown_for_user(path: Path, run_as_user: str) -> None:
    import pwd

    pw = pwd.getpwnam(run_as_user)
    os.chown(path, pw.pw_uid, pw.pw_gid)


def _chown_tree_for_user(path: Path, run_as_user: str) -> None:
    """Recursively chown a directory tree for benchmark data handoff.

    Symlinks are chowned in place and not followed, so ownership changes cannot
    escape the intended tree via symlink indirection.
    """

    import pwd

    pw = pwd.getpwnam(run_as_user)
    uid, gid = pw.pw_uid, pw.pw_gid

    def _apply(target: Path) -> None:
        mode = os.lstat(target).st_mode
        os.lchown(target, uid, gid)
        if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
            with os.scandir(target) as entries:
                for entry in entries:
                    _apply(Path(entry.path))

    if path.exists():
        _apply(path)


def prepare_runs_as_root(
    config: E2EConfig, *, effective_uid: int | None = None
) -> bool:
    """Return whether dataset prepare/repo-cache should run as the invoking user (root)."""

    if os.name == "nt":
        return False
    run_as_user = resolve_benchmark_run_as_user(
        agent_profile=config.agent_profile,
        benchmark_run_as_user=config.benchmark_run_as_user,
    )
    if not run_as_user:
        return False
    uid = os.geteuid() if effective_uid is None else effective_uid
    return uid == 0


def grant_benchmark_user_dataset_access(
    config: E2EConfig,
    *,
    repo_cache_dir: Path | None = None,
) -> None:
    """Hand off root-prepared dataset inputs to the sudo-wrapped benchmark user."""

    run_as_user = resolve_benchmark_run_as_user(
        agent_profile=config.agent_profile,
        benchmark_run_as_user=config.benchmark_run_as_user,
    )
    if not prepare_runs_as_root(config) or not run_as_user:
        return
    paths = {config.dataset_output_dir}
    if repo_cache_dir is not None:
        paths.add(repo_cache_dir)
    for path in paths:
        _chown_tree_for_user(path, run_as_user)


def ensure_directory(
    path: Path,
    *,
    run_as_user: str | None,
    effective_uid: int | None = None,
) -> None:
    """Create a directory tree with ownership suitable for sudo-wrapped bench/prepare.

    When pytest is root and the bench runs as ``run_as_user``, create the path as
    the invoking root user. The checkout is usually owned by another account, so
    ``sudo -u <run_as_user> mkdir`` would fail with Permission denied. After the
    directory exists, chown the leaf so the wrapped process can write results.
    """

    target = path.resolve()
    if not _running_as_root_for_user(
        run_as_user=run_as_user, effective_uid=effective_uid
    ):
        target.mkdir(parents=True, exist_ok=True)
        return

    assert run_as_user is not None
    target.mkdir(parents=True, exist_ok=True)
    _chown_for_user(target, run_as_user)


def resolve_benchmark_run_as_user(
    *, agent_profile: str, benchmark_run_as_user: str | None
) -> str | None:
    """Return the OS user for sudo wrapping; single profile never wraps."""

    if agent_profile == "single":
        return None
    if agent_profile != PLAN_SUBAGENT_PROFILE:
        return None
    return benchmark_run_as_user


def wrap_argv_for_user(
    argv: Sequence[str],
    *,
    run_as_user: str,
    cwd: Path,
    effective_uid: int | None = None,
) -> list[str]:
    """Prefix argv with ``sudo -u <user> -E env ... bash -c 'cd ... && exec ...'`` when root."""

    if effective_uid is None:
        if os.name == "nt":
            return list(argv)
        uid = os.geteuid()
    else:
        uid = effective_uid
    if uid != 0:
        return list(argv)
    command = " ".join(shlex.quote(part) for part in argv)
    return [
        "sudo",
        "-u",
        run_as_user,
        "-E",
        *_sudo_env_argv(),
        "bash",
        "-c",
        f"cd {shlex.quote(str(cwd))} && exec {command}",
    ]


def maybe_wrap_argv(
    argv: Sequence[str],
    *,
    agent_profile: str,
    benchmark_run_as_user: str | None,
    cwd: Path,
    effective_uid: int | None = None,
) -> list[str]:
    """Apply sudo wrapping for plan-subagent when configured and running as root."""

    run_as_user = resolve_benchmark_run_as_user(
        agent_profile=agent_profile,
        benchmark_run_as_user=benchmark_run_as_user,
    )
    if not run_as_user:
        return list(argv)
    return wrap_argv_for_user(
        argv,
        run_as_user=run_as_user,
        cwd=cwd,
        effective_uid=effective_uid,
    )


def _scenario_serve_args(config: E2EConfig) -> dict[str, Any]:
    spec = config.deploy()
    serve_args = dict(spec.serve_args)
    if config.scenario is PerfScenario.AGENTINFER:
        additional = serve_args.get("additional-config")
        if isinstance(additional, dict):
            patched = dict(additional)
            agentcache = patched.get("agentcache")
            if isinstance(agentcache, dict):
                agentcache = dict(agentcache)
                agentcache["lifecycle_socket_path"] = config.lifecycle_socket
                patched["agentcache"] = agentcache
            serve_args["additional-config"] = patched
    if not config.uses_default_bind():
        serve_args["host"] = config.host
        serve_args["port"] = config.port
    else:
        serve_args.pop("host", None)
        serve_args.pop("port", None)
    return serve_args


def build_vllm_serve_argv(config: E2EConfig) -> list[str]:
    spec = config.deploy()
    serve_args = _scenario_serve_args(config)
    return serve_args_to_argv(spec.model, serve_args, spec.middleware)


def build_vllm_serve_env(config: E2EConfig) -> dict[str, str]:
    env = os.environ.copy()
    env.update(config.deploy().serve_env)
    if config.scenario is PerfScenario.AGENTINFER:
        env["AGENTCACHE_VLLM_LIFECYCLE_SOCKET"] = config.lifecycle_socket
    return env


def build_benchmark_argv(config: E2EConfig, result_dir: Path) -> list[str]:
    params = dict(config.case.benchmark_params)
    params["task-num"] = config.task_num
    params["max-concurrency"] = config.max_concurrency
    params["timeout"] = config.task_timeout_seconds
    params["agent-executable"] = str(config.agent_executable)
    params["host"] = config.host
    params["port"] = config.port
    argv = benchmark_params_to_run_args(params, result_dir, _repo_root())
    return maybe_wrap_argv(
        argv,
        agent_profile=config.agent_profile,
        benchmark_run_as_user=config.benchmark_run_as_user,
        cwd=_repo_root(),
    )


def build_prepare_argv(config: E2EConfig) -> list[str]:
    """Build the dataset prepare command.

    Prepare is not sudo-wrapped; it runs as the invoking pytest user (often root
    in CI) so that user's git and network credentials apply. Plan-subagent sudo
    wrapping is reserved for the benchmark run itself.
    """

    return [
        "vllm",
        "bench",
        "serve",
        "--agentinfer",
        "prepare",
        config.prepare_dataset,
        "--output-dir",
        str(config.dataset_output_dir),
    ]


def build_benchmark_env() -> dict[str, str]:
    env = os.environ.copy()
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["TQDM_DISABLE"] = "1"
    return env
