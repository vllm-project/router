"""Load router_policy perf case JSON (self-contained: runtime + workers + router)."""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_E2E_DIR = Path(__file__).resolve().parents[3]
_HELPERS = _E2E_DIR / "helpers"
for _entry in (_HELPERS, _E2E_DIR):
    _path = str(_entry)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from helpers.binaries import (  # noqa: E402
    resolve_repo_path,
    resolve_router_bin,
    resolve_vllm_rs,
)
from helpers.paths import repo_root  # noqa: E402

RUN_SUFFIX_ENV = "AGENTX_RUN_SUFFIX"

MODEL_ENV = "MODEL"
PYTHON_ENV_KEYS = ("AGENTX_PYTHON", "PYTHON")

_PATH_KEYS = (
    "perf_work_root",
    "router_binary",
    "trace_path",
    "replay_config",
    "openssl_lib_dir",
)


def _env_first(*keys: str) -> str:
    for key in keys:
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return ""


def _resolve_python(case_value: str, repo: Path, *, case_id: str) -> str:
    if pinned := _env_first(*PYTHON_ENV_KEYS):
        return pinned
    val = (case_value or "").strip()
    if val in ("", "python"):
        # Same interpreter as pytest (no absolute path in JSON).
        return sys.executable
    if "/" in val or "\\" in val:
        return str(resolve_repo_path(val, repo))
    found = shutil.which(val)
    return found if found else val


def packaged_replay_agentx_yaml() -> Path:
    """Default replay template from pip-installed ``agentinfer`` (not a source checkout)."""

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
        Path("agentbench/configs/replay_agentX.yaml"),
        Path("agentinfer/agentbench/configs/replay_agentX.yaml"),
    ):
        candidate = package_root / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"replay_agentX.yaml not found under installed agentinfer ({package_root})"
    )


def load_case(case_file: Path, *, scenario_dir: Path | None = None) -> dict[str, Any]:
    del scenario_dir
    case = json.loads(case_file.read_text(encoding="utf-8-sig"))
    if not isinstance(case, dict):
        raise ValueError(f"case root must be an object: {case_file}")
    return case


def resolve_runtime(case: dict[str, Any]) -> dict[str, Any]:
    """Expand repo-relative paths; keep Hugging Face ``model`` id as-is."""

    raw = case.get("runtime")
    if not isinstance(raw, dict):
        raise RuntimeError(f"case {case.get('id', '?')} missing runtime object")

    repo = repo_root()
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

    try:
        rt["router_binary"] = resolve_router_bin(
            case_value=str(rt.get("router_binary") or ""),
            repo=repo,
            label=f"case {case_id}",
        )
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc

    for key in _PATH_KEYS:
        if key == "router_binary":
            continue
        val = str(rt.get(key, "") or "").strip()
        if not val:
            if key in ("perf_work_root", "trace_path"):
                raise RuntimeError(f"case {case_id}: runtime.{key} is required")
            continue
        rt[key] = str(resolve_repo_path(val, repo))

    replay_override = str(rt.get("replay_config") or "").strip()
    if replay_override:
        rt["replay_config_path"] = str(resolve_repo_path(replay_override, repo))
    else:
        # Resolved in cell bootstrap (same env as runtime.python), not at pytest import.
        rt["replay_config_path"] = ""

    rt["python"] = _resolve_python(str(rt.get("python") or ""), repo, case_id=case_id)
    try:
        rt["vllm_rs"] = resolve_vllm_rs(
            case_value=str(rt.get("vllm_rs") or ""),
            repo=repo,
            label=f"case {case_id}",
        )
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc

    return rt


def require_runtime(case: dict[str, Any]) -> dict[str, Any]:
    return resolve_runtime(case)


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
