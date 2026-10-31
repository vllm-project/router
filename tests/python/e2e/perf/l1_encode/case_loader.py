"""Load the L1 encode perf case (one worker, 16k turns)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CASES_DIR = Path(__file__).resolve().parent / "cases"
CASE_FILE = CASES_DIR / "qwen3_16k.json"

# Hugging Face id. ``MODEL`` overrides this (local directory or another id).
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

# Chat-template role markers sit on top of ``--tokens``. The worker's
# ``--max-model-len`` has to cover that plus ``--max-tokens``.
_TEMPLATE_HEADROOM = 4096


@dataclass(frozen=True)
class WaveSpec:
    hit_rate: float
    concurrency: int
    requests: int


@dataclass(frozen=True)
class L1Case:
    path: Path
    model: str
    host: str
    router_port: int
    worker_http_port: int
    worker_grpc_port: int
    prometheus_port: int
    cuda_visible_devices: str
    router_bin: str | None
    vllm_rs_bin: str | None
    worker_cli: tuple[str, ...]
    bench_tokens: int
    bench_max_tokens: int
    bench_timeout: float
    turns: int
    waves: tuple[WaveSpec, ...]


def _apply_tp_override(cli: tuple[str, ...]) -> tuple[str, ...]:
    """``PERF_TENSOR_PARALLEL_SIZE`` rewrites ``--tensor-parallel-size`` for a 1-GPU run."""
    raw = os.environ.get("PERF_TENSOR_PARALLEL_SIZE", "").strip()
    if not raw:
        return cli
    items = list(cli)
    for index, arg in enumerate(items):
        if arg == "--tensor-parallel-size" and index + 1 < len(items):
            items[index + 1] = raw
            return tuple(items)
        if arg.startswith("--tensor-parallel-size="):
            items[index] = f"--tensor-parallel-size={raw}"
            return tuple(items)
    return (*items, "--tensor-parallel-size", raw)


def cli_value(cli: tuple[str, ...], flag: str) -> str | None:
    for index, arg in enumerate(cli):
        if arg == flag and index + 1 < len(cli):
            return cli[index + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def _optional_str(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def bench_tokens(case: L1Case) -> int:
    return _env_int("PERF_TOKENS", case.bench_tokens)


def bench_max_tokens(case: L1Case) -> int:
    return _env_int("PERF_MAX_TOKENS", case.bench_max_tokens)


def bench_timeout(case: L1Case) -> float:
    return _env_float("PERF_TIMEOUT", case.bench_timeout)


def check_limits(case: L1Case) -> None:
    """Fail before GPU load when the worker cannot hold the bench prompt."""
    tokens = bench_tokens(case)
    max_tokens = bench_max_tokens(case)
    headroom = tokens + max_tokens + _TEMPLATE_HEADROOM
    model_len = cli_value(case.worker_cli, "--max-model-len")
    if model_len is not None and headroom > int(model_len):
        raise RuntimeError(
            f"bench tokens {tokens} + max_tokens {max_tokens} + "
            f"{_TEMPLATE_HEADROOM} template headroom ({headroom}) exceeds "
            f"--max-model-len {model_len}. Raise --max-model-len and "
            "--max-num-batched-tokens. Do not set them equal to --tokens."
        )
    batched = cli_value(case.worker_cli, "--max-num-batched-tokens")
    if batched is not None and headroom > int(batched):
        raise RuntimeError(
            f"bench needs {headroom} tokens of room but --max-num-batched-tokens "
            f"is {batched}. Raise it with --max-model-len. Do not set either "
            "equal to --tokens."
        )
    seqs = cli_value(case.worker_cli, "--max-num-seqs")
    peak = max((wave.concurrency for wave in case.waves), default=1)
    if seqs is not None and peak > int(seqs):
        raise RuntimeError(
            f"--max-num-seqs {seqs} is below the largest wave concurrency {peak}"
        )


def _parse_wave(raw: dict[str, Any], *, where: str) -> WaveSpec:
    try:
        hit_rate = float(raw["hit_rate"])
        concurrency = int(raw["concurrency"])
        requests = int(raw["requests"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid wave in {where}: {raw!r}") from exc
    if concurrency < 1 or requests < 1:
        raise ValueError(f"invalid wave in {where}: {raw!r}")
    if not 0.0 <= hit_rate <= 1.0:
        raise ValueError(f"invalid hit_rate in {where}: {raw!r}")
    return WaveSpec(hit_rate=hit_rate, concurrency=concurrency, requests=requests)


def load_case(path: Path | None = None) -> L1Case:
    case_path = path or CASE_FILE
    data = json.loads(case_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{case_path}: root must be an object")
    stack = data.get("stack") if isinstance(data.get("stack"), dict) else {}
    worker = data.get("worker") if isinstance(data.get("worker"), dict) else {}
    bench = data.get("bench") if isinstance(data.get("bench"), dict) else {}
    cli = worker.get("cli")
    if not isinstance(cli, list) or not cli or not all(isinstance(x, str) for x in cli):
        raise ValueError(f"{case_path}: worker.cli must be a non-empty string array")
    shape = str(bench.get("shape", "turns"))
    if shape != "turns":
        raise ValueError(
            f"{case_path}: L1 encode perf requires bench.shape 'turns', got {shape!r}"
        )
    waves_raw = data.get("waves")
    if not isinstance(waves_raw, list) or not waves_raw:
        raise ValueError(f"{case_path}: missing 'waves' array")
    model = (
        os.environ.get("MODEL", "").strip()
        or _optional_str(data.get("model"))
        or DEFAULT_MODEL
    )
    return L1Case(
        path=case_path,
        model=model,
        host=str(stack.get("host", "127.0.0.1")),
        router_port=int(stack.get("router_port", 30000)),
        worker_http_port=int(stack.get("worker_http_port", 8000)),
        worker_grpc_port=int(stack.get("worker_grpc_port", 50051)),
        prometheus_port=int(stack.get("prometheus_port", 29000)),
        cuda_visible_devices=str(stack.get("cuda_visible_devices", "0")),
        router_bin=_optional_str(stack.get("router_bin")),
        vllm_rs_bin=_optional_str(stack.get("vllm_rs_bin")),
        worker_cli=_apply_tp_override(tuple(cli)),
        bench_tokens=int(bench.get("tokens", 16384)),
        bench_max_tokens=int(bench.get("max_tokens", 64)),
        bench_timeout=float(bench.get("timeout", 1800)),
        turns=int(bench.get("turns", 8)),
        waves=tuple(
            _parse_wave(item, where=f"{case_path.name}:waves") for item in waves_raw
        ),
    )
