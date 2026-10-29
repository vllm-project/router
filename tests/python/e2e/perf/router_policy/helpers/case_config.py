"""Load router_policy perf case JSON (self-contained: runtime + workers + router)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PATH_KEYS = (
    "perf_work_root",
    "router_binary",
    "trace_path",
    "replay_config",
    "openssl_lib_dir",
)
_CMD_KEYS = ("python", "vllm_rs")


def repo_root() -> Path:
    """vllm-router root (``router_policy/helpers`` → six levels up)."""

    return Path(__file__).resolve().parents[6]


def _resolve_repo_path(raw: str, repo: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (repo / path).resolve()


def packaged_replay_agentx_yaml() -> Path:
    """Default replay template from pip-installed ``agentinfer`` (not a source checkout)."""

    try:
        import agentinfer
    except ImportError as exc:
        raise RuntimeError(
            "agentinfer is not installed in this Python env; "
            "pip install the agentinfer wheel before running router_policy perf"
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

    model = (rt.get("model") or rt.get("model_path") or "").strip()
    if not model:
        raise RuntimeError(
            f"case {case.get('id')}: runtime.model (HF model id) is required"
        )
    rt["model"] = model
    rt["served_model_name"] = (rt.get("served_model_name") or model).strip()

    for key in _PATH_KEYS:
        val = str(rt.get(key, "") or "").strip()
        if not val:
            if key in ("perf_work_root", "router_binary", "trace_path"):
                raise RuntimeError(f"case {case.get('id')}: runtime.{key} is required")
            continue
        rt[key] = str(_resolve_repo_path(val, repo))

    replay_override = str(rt.get("replay_config") or "").strip()
    if replay_override:
        rt["replay_config_path"] = str(_resolve_repo_path(replay_override, repo))
    else:
        rt["replay_config_path"] = str(packaged_replay_agentx_yaml())

    for key in _CMD_KEYS:
        val = str(rt.get(key, "") or "").strip()
        if not val:
            raise RuntimeError(f"case {case.get('id')}: runtime.{key} is required")
        if "/" in val or "\\" in val:
            rt[key] = str(_resolve_repo_path(val, repo))
        else:
            rt[key] = val

    return rt


def require_runtime(case: dict[str, Any]) -> dict[str, Any]:
    return resolve_runtime(case)


def cell_output_dir(case: dict[str, Any]) -> Path:
    runtime = resolve_runtime(case)
    cell_id = str(case["id"])
    return Path(runtime["perf_work_root"]) / "runs" / cell_id


def cell_results_dir(case: dict[str, Any]) -> Path:
    runtime = resolve_runtime(case)
    return Path(runtime["perf_work_root"]) / "results"
