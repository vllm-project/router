"""E2E defaults when ``vllm.test`` is not installed (pip vLLM >= 0.29).

Upstream router tests historically imported ``vllm.test.test_utils`` from a
source/tree install. Released wheels only ship the runtime package; use these
fallbacks or env overrides (``MODEL``, ``E2E_MODEL``, ``E2E_URL``).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from typing import Callable, Optional

# --- test_utils constants (sglang/vLLM CI defaults) -------------------------

try:
    from vllm.test.test_utils import (  # type: ignore[import-not-found]
        DEFAULT_MODEL_NAME_FOR_TEST,
        DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        DEFAULT_URL_FOR_TEST,
    )
except ModuleNotFoundError:
    _cuda_slot = 0
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    try:
        _cuda_slot = int(_cvd)
    except ValueError:
        pass

    _base_port = 20000 + _cuda_slot * 1000
    DEFAULT_URL_FOR_TEST = os.environ.get(
        "E2E_URL", f"http://127.0.0.1:{_base_port + 1000}"
    )
    DEFAULT_MODEL_NAME_FOR_TEST = os.environ.get(
        "MODEL",
        os.environ.get("E2E_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
    )
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = float(
        os.environ.get("E2E_SERVER_LAUNCH_TIMEOUT", "600")
    )


def e2e_model_name() -> str:
    """Model path/id for session fixtures; ``MODEL`` wins over the default."""
    return os.environ.get("MODEL", "").strip() or DEFAULT_MODEL_NAME_FOR_TEST


def _has_launch_server_module() -> bool:
    return importlib.util.find_spec("vllm.launch_server") is not None


def _legacy_pd_cli_flags(extra: list[str]) -> list[str]:
    """Flags only meant for ``vllm.launch_server`` / legacy PD worker launches."""
    return [a for a in extra if a.startswith("--disaggregation")]


def vllm_worker_command(
    model: str,
    host: str,
    port: str | int,
    *,
    dp_size: int | None = None,
    api_key: str | None = None,
    base_gpu_id: int | None = 0,
    extra: Optional[list[str]] = None,
) -> list[str]:
    """Build argv to start a vLLM worker (legacy ``launch_server`` or v0.29 ``serve``)."""
    tail = list(extra or [])
    if _has_launch_server_module():
        cmd = [
            "python3",
            "-m",
            "vllm.launch_server",
            "--model-path",
            model,
            "--host",
            host,
            "--port",
            str(port),
            "--base-gpu-id",
            str(base_gpu_id or 0),
        ]
        if dp_size is not None:
            cmd += ["--dp-size", str(dp_size)]
        if api_key is not None:
            cmd += ["--api-key", api_key]
        cmd.extend(tail)
        return cmd

    pd_flags = _legacy_pd_cli_flags(tail)
    if pd_flags:
        raise RuntimeError(
            "vllm serve (pip vLLM >= 0.29) does not accept legacy disaggregation "
            f"worker flags {pd_flags!r}. Use a vLLM install with vllm.launch_server "
            "for test_pd_router PD workers, or run pd_disagg_vllm shell scripts."
        )

    vllm_bin = shutil.which("vllm") or "vllm"
    cmd = [vllm_bin, "serve", model, "--host", host, "--port", str(port)]
    if dp_size is not None and dp_size > 1:
        cmd += ["--data-parallel-size", str(dp_size)]
    if api_key is not None:
        cmd += ["--api-key", api_key]
    cmd.extend(tail)
    return cmd


def _cuda_visible_devices_list() -> list[str] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def vllm_worker_popen_env(
    *,
    base_gpu_id: int | None = 0,
    dp_size: int | None = None,
) -> dict[str, str]:
    """Environment for ``Popen`` when ``vllm serve`` has no ``--base-gpu-id``."""
    env = os.environ.copy()
    if _has_launch_server_module():
        return env
    start = int(base_gpu_id or 0)
    width = max(1, int(dp_size) if dp_size is not None and dp_size > 0 else 1)
    visible = _cuda_visible_devices_list()
    if visible is not None:
        end = start + width
        if end > len(visible):
            raise ValueError(
                f"requested worker GPUs base_gpu_id={start} dp_size={width} but "
                f"CUDA_VISIBLE_DEVICES only lists {len(visible)} device(s): "
                f"{','.join(visible)!r}"
            )
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible[start:end])
    else:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(start + i) for i in range(width))
    return env


def import_run_eval() -> Callable:
    """Return ``run_eval`` from vLLM/SGLang test tree, or raise with a clear message."""
    for module in ("vllm.test.run_eval", "sglang.test.run_eval"):
        try:
            mod = importlib.import_module(module)
            return mod.run_eval
        except ModuleNotFoundError:
            continue

    def _missing(*_args, **_kwargs):
        raise RuntimeError(
            "run_eval is unavailable: install vLLM/SGLang from source with test "
            "packages, or skip tests that call run_eval (mmlu, etc.)."
        )

    return _missing
