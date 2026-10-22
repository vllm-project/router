"""Load perf wave definitions from ``cases/*.json``."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CASES_DIR = Path(__file__).resolve().parent / "cases"


@dataclass(frozen=True)
class StackConfig:
    host: str
    router_port: int
    worker_http_port: int
    worker_grpc_port: int | None
    prometheus_port: int | None
    router_bin: str | None = None
    vllm_rs_bin: str | None = None


@dataclass(frozen=True)
class WaveSpec:
    hit_rate: float
    concurrency: int
    requests: int


@dataclass(frozen=True)
class BackendCaseFile:
    backend: str
    waves: tuple[WaveSpec, ...]
    path: Path
    stack: StackConfig
    worker_cli: tuple[str, ...]
    model: str | None = None
    model_dir: str | None = None
    bench_tokens: int = 131072
    bench_max_tokens: int = 64
    bench_timeout: float = 1800.0


def wave_case_id(hit_rate: float, concurrency: int, requests: int) -> str:
    return f"hit{int(round(hit_rate * 100)):02d}-c{concurrency}-r{requests}"


def _parse_wave(raw: dict[str, Any], *, where: str) -> WaveSpec:
    try:
        hit_rate = float(raw["hit_rate"])
        concurrency = int(raw["concurrency"])
        requests = int(raw.get("requests", raw.get("tasks")))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid wave in {where}: {raw!r}") from exc
    if concurrency < 1 or requests < 1:
        raise ValueError(f"invalid wave in {where}: {raw!r}")
    return WaveSpec(hit_rate=hit_rate, concurrency=concurrency, requests=requests)


def _parse_waves(data: dict[str, Any], path: Path) -> tuple[WaveSpec, ...]:
    block = data.get("waves")
    if block is None and isinstance(data.get("matrix"), list):
        block = data["matrix"]
    if not isinstance(block, list):
        raise ValueError(f"{path}: missing or invalid 'waves' array")
    return tuple(_parse_wave(item, where=f"{path.name}:waves") for item in block)


def _parse_stack(data: dict[str, Any], path: Path, backend: str) -> StackConfig:
    stack = data.get("stack") if isinstance(data.get("stack"), dict) else {}
    host = str(stack.get("host", "127.0.0.1"))
    router_port = int(stack.get("router_port", 16002))
    worker_http_port = int(stack.get("worker_http_port", 18002))
    prom = stack.get("prometheus_port")
    prometheus_port = int(prom) if prom is not None else None
    grpc_raw = stack.get("worker_grpc_port")
    if grpc_raw is not None:
        worker_grpc_port = int(grpc_raw)
    elif backend == "rust_grpc":
        worker_grpc_port = 15002
    else:
        worker_grpc_port = None
    if backend == "rust_grpc" and prometheus_port is None:
        prometheus_port = 29002
    if backend == "rust_http" and prometheus_port is None:
        prometheus_port = 29001
    if backend == "python_http" and prometheus_port is None:
        prometheus_port = 29003
    return StackConfig(
        host=host,
        router_port=router_port,
        worker_http_port=worker_http_port,
        worker_grpc_port=worker_grpc_port,
        prometheus_port=prometheus_port,
        router_bin=_optional_str(stack.get("router_bin")),
        vllm_rs_bin=_optional_str(stack.get("vllm_rs_bin")),
    )


def _parse_worker_cli(worker: dict[str, Any], *, path: Path) -> tuple[str, ...]:
    cli = worker.get("cli")
    if isinstance(cli, list) and all(isinstance(x, str) for x in cli):
        if not cli:
            raise ValueError(f"{path}: worker.cli must be a non-empty string array")
        return tuple(cli)
    raise ValueError(
        f"{path}: missing worker.cli (vllm-rs / vllm serve flags for this case)"
    )


def _optional_str(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _parse_model_fields(
    data: dict[str, Any], worker: dict[str, Any]
) -> tuple[str | None, str | None]:
    model = _optional_str(data.get("model")) or _optional_str(worker.get("model"))
    model_dir = _optional_str(data.get("model_dir")) or _optional_str(
        worker.get("model_dir")
    )
    return model, model_dir


def apply_case_env(case: BackendCaseFile) -> None:
    """Copy case ``model`` / ``model_dir`` into env when unset (env wins)."""
    if case.model and not os.environ.get("MODEL", "").strip():
        os.environ["MODEL"] = case.model
    if case.model_dir and not os.environ.get("MODEL_DIR", "").strip():
        os.environ["MODEL_DIR"] = case.model_dir


def perf_model_ready() -> bool:
    if os.environ.get("MODEL", "").strip():
        return True
    if not perf_case_selected():
        return False
    try:
        case = resolve_perf_case()
    except (OSError, ValueError, RuntimeError, FileNotFoundError):
        return False
    return bool(case.model)


def load_case_file(path: Path) -> BackendCaseFile:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be an object")
    backend = data.get("backend") or path.stem
    if not isinstance(backend, str) or not backend:
        raise ValueError(f"{path}: backend must be a non-empty string")
    worker = data.get("worker") if isinstance(data.get("worker"), dict) else {}
    bench = data.get("bench") if isinstance(data.get("bench"), dict) else {}
    model, model_dir = _parse_model_fields(data, worker)
    return BackendCaseFile(
        backend=backend,
        waves=_parse_waves(data, path),
        path=path,
        stack=_parse_stack(data, path, backend),
        worker_cli=_parse_worker_cli(worker, path=path),
        model=model,
        model_dir=model_dir,
        bench_tokens=int(bench.get("tokens", 131072)),
        bench_max_tokens=int(bench.get("max_tokens", 64)),
        bench_timeout=float(bench.get("timeout", 1800)),
    )


def list_case_files() -> list[Path]:
    return sorted(CASES_DIR.glob("*.json"))


def _case_path_from_selector(raw: str) -> Path:
    text = raw.strip()
    if not text:
        raise RuntimeError(
            "set pytest --perf-case or env PERF_CASE (e.g. rust_grpc or cases/rust_grpc.json)"
        )
    candidate = Path(text)
    if candidate.suffix == ".json":
        if candidate.is_file():
            return candidate.resolve()
        if len(candidate.parts) > 1:
            raise FileNotFoundError(f"{candidate} not found")
        under_cases = CASES_DIR / candidate.name
        if under_cases.is_file():
            return under_cases.resolve()
        raise FileNotFoundError(f"{candidate} not found")
    path = CASES_DIR / f"{text}.json"
    if not path.is_file():
        known = ", ".join(p.stem for p in list_case_files())
        raise FileNotFoundError(f"{path} not found; available: {known}")
    return path


def resolve_perf_case(stem: str | None = None) -> BackendCaseFile:
    raw = (stem or os.environ.get("PERF_CASE") or "").strip()
    return load_case_file(_case_path_from_selector(raw))


def perf_case_selected() -> bool:
    return bool(os.environ.get("PERF_CASE", "").strip())


def use_external_router() -> bool:
    return os.environ.get("PERF_EXTERNAL", "").lower() in ("1", "true", "yes")
