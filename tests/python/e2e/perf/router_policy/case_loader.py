"""Load router_policy perf case JSON (self-contained: runtime + workers + router)."""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

RUN_SUFFIX_ENV = "AGENTX_RUN_SUFFIX"

MODEL_ENV = "MODEL"
PYTHON_ENV_KEYS = ("AGENTX_PYTHON", "PYTHON")

_PATH_KEYS = (
    "perf_work_root",
    "trace_path",
    "replay_config",
    "openssl_lib_dir",
)

# CI image (docker/Dockerfile.ci): vllm-router is copied to /usr/local/bin.
_ROUTER_BIN = "vllm-router"


def _abs_repo_path(raw: str, repo: Path) -> Path:
    text = raw.strip()
    if not text:
        raise ValueError("empty path")
    path = Path(text).expanduser()
    return path if path.is_absolute() else (repo / path).resolve()


def _vllm_rs_bin() -> str:
    """Engine binary shipped inside the vllm package installed by Dockerfile.ci.

    That image does not put ``vllm-rs`` on ``PATH``. The file sits next to
    ``vllm/__init__.py``.
    """

    try:
        import vllm
    except ImportError as exc:
        raise RuntimeError(
            "vllm is not installed in this Python env; "
            "the CI image installs vllm==0.30.0"
        ) from exc
    candidate = Path(vllm.__file__).resolve().parent / "vllm-rs"
    if not candidate.is_file():
        raise RuntimeError(f"vllm-rs is not in the installed vllm package: {candidate}")
    return str(candidate)


def _repo_root() -> Path:
    """vllm-router checkout root (``router_policy/case_loader.py`` → five levels up)."""

    return Path(__file__).resolve().parents[5]


def _env_first(*keys: str) -> str:
    for key in keys:
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return ""


def _resolve_python(case_value: str, repo: Path) -> str:
    if pinned := _env_first(*PYTHON_ENV_KEYS):
        return pinned
    val = (case_value or "").strip()
    if val in ("", "python"):
        # Same interpreter as pytest (no absolute path in JSON).
        return sys.executable
    if "/" in val or "\\" in val:
        return str(_abs_repo_path(val, repo))
    found = shutil.which(val)
    return found if found else val


def packaged_replay_agentx_yaml() -> Path:
    """TraceLab replay template from pip-installed ``agentinfer``."""

    try:
        import agentinfer
    except ImportError as exc:
        raise RuntimeError(
            "agentinfer is not installed in this Python env "
            f"({sys.executable}); pip install the agentinfer wheel before running "
            "router_policy perf"
        ) from exc

    package_root = Path(agentinfer.__file__).resolve().parent
    for relative in (
        Path("agentbench/configs/replay_tracelab.yaml"),
        Path("agentinfer/agentbench/configs/replay_tracelab.yaml"),
    ):
        candidate = package_root / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"replay_tracelab.yaml not found under installed agentinfer ({package_root})"
    )


def load_case(case_file: Path) -> dict[str, Any]:
    case = json.loads(case_file.read_text(encoding="utf-8-sig"))
    if not isinstance(case, dict):
        raise ValueError(f"case root must be an object: {case_file}")
    return case


def resolve_runtime(case: dict[str, Any]) -> dict[str, Any]:
    """Expand repo-relative paths; keep Hugging Face ``model`` id as-is."""

    raw = case.get("runtime")
    if not isinstance(raw, dict):
        raise RuntimeError(f"case {case.get('id', '?')} missing runtime object")

    repo = _repo_root()
    rt: dict[str, Any] = dict(raw)

    case_id = str(case.get("id", "?"))
    model = (
        _env_first(MODEL_ENV) or (rt.get("model") or rt.get("model_path") or "").strip()
    )
    if not model:
        raise RuntimeError(
            f"case {case_id}: set export {MODEL_ENV}=... or runtime.model in JSON"
        )
    rt["model"] = model
    rt["served_model_name"] = (
        _env_first("VLLM_ROUTER_MODEL")
        or (rt.get("served_model_name") or model).strip()
    )

    rt["router_binary"] = _ROUTER_BIN
    rt["vllm_rs"] = _vllm_rs_bin()

    for key in _PATH_KEYS:
        val = str(rt.get(key, "") or "").strip()
        if not val:
            if key in ("perf_work_root", "trace_path"):
                raise RuntimeError(f"case {case_id}: runtime.{key} is required")
            continue
        rt[key] = str(_abs_repo_path(val, repo))

    replay_override = str(rt.get("replay_config") or "").strip()
    if replay_override:
        rt["replay_config_path"] = str(_abs_repo_path(replay_override, repo))
    else:
        # Resolved in cell bootstrap (same env as runtime.python), not at pytest import.
        rt["replay_config_path"] = ""

    rt["python"] = _resolve_python(str(rt.get("python") or ""), repo)
    return rt


@dataclass(frozen=True)
class RouterPolicyCase:
    """One router_policy case, same role as grpc ``BackendCaseFile``."""

    path: Path
    id: str
    model: str
    served_model_name: str
    python: str
    vllm_rs: str
    router_bin: str
    perf_work_root: Path
    trace_path: Path
    trace_type: str
    endpoint: str
    replay_config: Path
    agent_infer: Path | None
    openssl_lib_dir: str
    policy: str
    task_num: int
    concurrency: int
    worker_count: int
    gap_scale: float
    max_inflight: int
    stall_secs: int
    idle_secs: int
    task_stagger: float
    per_worker: dict[str, Any]
    router: dict[str, Any]


def concurrency_levels(case: dict[str, Any]) -> list[int]:
    """``benchmark_params.max-concurrency`` as a list, or one integer."""

    raw = (case.get("benchmark_params") or {}).get("max-concurrency", 8)
    if isinstance(raw, list):
        if not raw:
            raise RuntimeError("benchmark_params.max-concurrency is empty")
        return [int(level) for level in raw]
    return [int(raw)]


def case_at_concurrency(case: dict[str, Any], concurrency: int) -> dict[str, Any]:
    """Copy a scenario JSON and pin one concurrency.

    The cell id stays ``c<n>-<policy>`` so three levels from one file do not
    share an output directory.
    """

    stamped = dict(case)
    bench = dict(case.get("benchmark_params") or {})
    bench["max-concurrency"] = int(concurrency)
    stamped["benchmark_params"] = bench
    router = case.get("router_params") or {}
    policy = str(router.get("policy") or case.get("id") or "router_policy")
    stamped["id"] = f"c{int(concurrency)}-{policy}"
    return stamped


def load_router_policy_case(
    case_file: Path, *, concurrency: int | None = None
) -> RouterPolicyCase:
    """Load one case JSON into ``RouterPolicyCase``."""

    raw = load_case(case_file)
    levels = concurrency_levels(raw)
    if concurrency is None:
        if len(levels) == 1:
            concurrency = levels[0]
        else:
            raise RuntimeError(f"{case_file}: pass concurrency; file lists {levels}")
    elif int(concurrency) not in levels:
        raise RuntimeError(f"{case_file}: concurrency {concurrency} not in {levels}")
    raw = case_at_concurrency(raw, int(concurrency))
    runtime = resolve_runtime(raw)
    bench = raw.get("benchmark_params") or {}
    workers = raw.get("workers") or {}
    router = dict(raw.get("router_params") or {})
    case_id = str(raw["id"])
    policy = str(
        router.get("policy") or raw.get("router_policy") or case_id.split("-", 1)[-1]
    )
    replay_path = str(runtime.get("replay_config_path") or "").strip()
    agent_infer = str(runtime.get("agent_infer") or "").strip()
    return RouterPolicyCase(
        path=case_file,
        id=case_id,
        model=runtime["model"],
        served_model_name=runtime["served_model_name"],
        python=runtime["python"],
        vllm_rs=runtime["vllm_rs"],
        router_bin=runtime["router_binary"],
        perf_work_root=Path(runtime["perf_work_root"]),
        trace_path=Path(runtime["trace_path"]),
        trace_type=str(runtime.get("trace_type") or "tracelab"),
        endpoint=str(runtime.get("endpoint") or "/v1/chat/completions"),
        replay_config=Path(replay_path)
        if replay_path
        else packaged_replay_agentx_yaml(),
        agent_infer=Path(agent_infer) if agent_infer else None,
        openssl_lib_dir=str(runtime.get("openssl_lib_dir") or ""),
        policy=policy,
        task_num=int(bench.get("task-num", 128)),
        concurrency=int(bench.get("max-concurrency", 8)),
        worker_count=int(workers.get("count", 4)),
        gap_scale=float(bench.get("gap_scale", 0.0)),
        max_inflight=int(bench.get("max_inflight", 1)),
        stall_secs=int(bench.get("stall_secs", 900)),
        idle_secs=int(bench.get("idle_secs", 120)),
        task_stagger=float(bench.get("task_stagger", 0)),
        per_worker=dict(workers.get("per_worker") or {}),
        router=router,
    )


def cell_run_suffix(*, run_suffix: str | None = None) -> str:
    """Run id for artifact dirs (local time, seconds resolution).

    Re-use one suffix for a whole pytest matrix via ``export AGENTX_RUN_SUFFIX=…``.
    """

    if run_suffix:
        return run_suffix
    from_env = os.environ.get(RUN_SUFFIX_ENV)
    if from_env:
        return from_env
    return datetime.now().strftime("%Y%m%d%H%M%S")


def cell_output_dir(case: dict[str, Any], *, run_suffix: str | None = None) -> Path:
    runtime = resolve_runtime(case)
    cell_id = str(case["id"])
    suffix = cell_run_suffix(run_suffix=run_suffix)
    return Path(runtime["perf_work_root"]) / "runs" / f"{cell_id}-{suffix}"


def cell_results_summary_path(case: dict[str, Any], *, run_suffix: str) -> Path:
    cell_id = str(case["id"])
    return cell_results_dir(case) / f"{cell_id}-{run_suffix}.json"


def cell_results_dir(case: dict[str, Any]) -> Path:
    runtime = resolve_runtime(case)
    return Path(runtime["perf_work_root"]) / "results"
