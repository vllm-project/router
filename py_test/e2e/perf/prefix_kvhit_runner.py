"""Shared helpers for prefix-KV perf pytest (wraps benches/backend/bench_prefix_kvhit.py)."""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH_SCRIPT = REPO_ROOT / "benches" / "backend" / "bench_prefix_kvhit.py"


def perf_env_ready() -> bool:
    return bool(os.environ.get("MODEL", "").strip())


def load_bench_module():
    spec = importlib.util.spec_from_file_location("bench_prefix_kvhit", BENCH_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_perf_settings(
    *,
    backend_label: str | None = None,
    bench_tokens: int | None = None,
    bench_max_tokens: int | None = None,
    bench_timeout: float | None = None,
) -> dict[str, Any]:
    model = os.environ["MODEL"]
    if os.environ.get("PERF_TOKENS", "").strip():
        tokens = int(os.environ["PERF_TOKENS"].strip())
    elif bench_tokens is not None:
        tokens = bench_tokens
    else:
        tokens = 0
    label = os.environ.get("BACKEND_LABEL") or backend_label or "backend"
    default_max = bench_max_tokens if bench_max_tokens is not None else 64
    default_timeout = bench_timeout if bench_timeout is not None else 1800.0
    return {
        "router_url": os.environ["ROUTER_URL"],
        "model": model,
        "model_dir": os.environ.get("MODEL_DIR", model),
        "tokens": tokens,
        "chars": int(os.environ.get("KVHIT_CHARS", "8000")),
        "max_tokens": int(os.environ.get("PERF_MAX_TOKENS", str(default_max))),
        "timeout": float(os.environ.get("PERF_TIMEOUT", str(default_timeout))),
        "warmup": os.environ.get("PERF_WARMUP", "1").lower() in ("1", "true", "yes"),
        "backend_label": label,
        "output_dir": os.environ.get("PERF_OUTPUT_DIR", "").strip(),
        "runs": max(1, int(os.environ.get("RUNS", "1"))),
    }


def make_user_text(bench_mod, settings: dict[str, Any]) -> str:
    if settings["tokens"] > 0:
        return bench_mod.make_user_text_exact_tokens(
            settings["model_dir"], settings["tokens"]
        )
    return bench_mod.make_user_text(settings["chars"])


def run_prefix_kvhit_wave(
    bench_mod,
    settings: dict[str, Any],
    *,
    hit_rate: float,
    concurrency: int,
    requests: int,
    warmup: bool | None = None,
) -> dict[str, Any]:
    user = make_user_text(bench_mod, settings)
    do_warmup = settings["warmup"] if warmup is None else warmup
    miss_user, hit_user = bench_mod.make_bodies(hit_rate, user)

    result: dict[str, Any] = {
        "router": settings["router_url"],
        "model": settings["model"],
        "hit_rate": hit_rate,
        "concurrency": concurrency,
        "requests": requests,
    }
    if do_warmup:
        result["warmup"] = {
            "miss": bench_mod.post_chat(
                settings["router_url"],
                settings["model"],
                miss_user,
                settings["max_tokens"],
                settings["timeout"],
            ),
            "hit": bench_mod.post_chat(
                settings["router_url"],
                settings["model"],
                hit_user,
                settings["max_tokens"],
                settings["timeout"],
            ),
        }
    result["wave"] = bench_mod.post_hit_wave(
        settings["router_url"],
        settings["model"],
        bench_mod.make_wave_bodies(hit_rate, user, requests),
        settings["max_tokens"],
        settings["timeout"],
        concurrency,
    )
    return result


def assert_wave_summary(wave: dict[str, Any], expected_requests: int) -> dict[str, Any]:
    summary = wave.get("summary") or {}
    assert summary.get("count") == expected_requests
    for key in ("ttft_ms", "e2e_ms"):
        block = summary.get(key)
        assert isinstance(block, dict), f"missing {key} in {json.dumps(summary)}"
        assert block.get("avg", 0) > 0
    return summary


def hit_rate_file_tag(hit_rate: float) -> str:
    if hit_rate >= 0.99:
        return "099"
    if hit_rate <= 0.0:
        return "000"
    return str(hit_rate).replace(".", "")


def maybe_write_result(
    settings: dict[str, Any],
    *,
    run_idx: int,
    hit_rate: float,
    concurrency: int,
    requests: int,
    payload: dict[str, Any],
) -> Path | None:
    base = settings.get("output_dir") or ""
    if not base:
        return None
    out_dir = Path(base)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"run{run_idx}_hit{hit_rate_file_tag(hit_rate)}"
        f"_c{concurrency}_r{requests}.json"
    )
    path = out_dir / name
    envelope = {
        "backend_label": settings["backend_label"],
        "run_idx": run_idx,
        "hit_rate": hit_rate,
        "concurrency": concurrency,
        "requests": requests,
        **payload,
    }
    path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    return path


def default_output_dir(settings: dict[str, Any]) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = settings["backend_label"]
    return str(Path(__file__).parent / "results" / f"{label}_{stamp}")
