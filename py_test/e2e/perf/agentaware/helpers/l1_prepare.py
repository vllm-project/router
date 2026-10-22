# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""L1 prepare: start serve, calibrate, write program JSON, stop serve."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _import_sibling_module(stem: str) -> object:
    import importlib.util

    here = Path(__file__).resolve().parent
    path = here / f"{stem}.py"
    mod_name = f"_agentaware_{stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load agentaware helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


_l1_serve = _import_sibling_module("l1_serve")
build_l1_vllm_e2e_config = _l1_serve.build_l1_vllm_e2e_config
managed_vllm = _l1_serve.managed_vllm
_psc = _import_sibling_module("program_scheduling_config")
read_observed_kv_capacity_tokens = _psc.read_observed_kv_capacity_tokens
write_program_scheduling_configs = _psc.write_program_scheduling_configs

DEFAULT_OUTPUT_ROOT = Path("results/agentaware/l1")
DEFAULT_PREFILL_LENGTHS = "1024,2048,4096,8192,16384,32768,65536"
DEFAULT_DECODE_BATCH_SIZES = "1,2,4,8,12"
DEFAULT_DECODE_CONTEXT_LENGTHS = "1024,8192,16384,32768"


def _repo_root() -> Path:
    # .../py_test/e2e/perf/agentaware/helpers/l1_prepare.py → repo root
    return Path(__file__).resolve().parents[5]


def _run_module(module: str, argv: list[str]) -> None:
    from helpers.paths import calibration_subprocess_env, resolve_calibration_root

    calib_root = resolve_calibration_root(required=True)
    completed = subprocess.run(
        [sys.executable, "-m", module, *argv],
        check=False,
        cwd=calib_root,
        env=calibration_subprocess_env(calib_root),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{module} exited with code {completed.returncode}")


def _run_calibration(
    *,
    base_url: str,
    model: str,
    calibration_dir: Path,
    args: argparse.Namespace,
) -> None:
    calibration_dir.mkdir(parents=True, exist_ok=True)
    overwrite_flag = ["--overwrite"] if args.force else []
    prefill_argv = [
        "--base-url",
        base_url,
        "--model",
        model,
        "--output-dir",
        str(calibration_dir),
        "--data-parallel-rank",
        str(args.data_parallel_rank),
        "--prompt-lengths",
        args.prompt_lengths,
        "--repeats",
        str(args.prefill_repeats),
        *overwrite_flag,
    ]
    decode_argv = [
        "--base-url",
        base_url,
        "--model",
        model,
        "--output-dir",
        str(calibration_dir),
        "--data-parallel-rank",
        str(args.data_parallel_rank),
        "--batch-sizes",
        args.decode_batch_sizes,
        "--context-lengths",
        args.decode_context_lengths,
        "--output-tokens",
        str(args.decode_output_tokens),
        "--repeats",
        str(args.decode_repeats),
        *overwrite_flag,
    ]
    print(f"[prepare_l1] running prefill calibration → {calibration_dir}", flush=True)
    _run_module("tools.calibration.calibrate_prefill", prefill_argv)
    print(f"[prepare_l1] running decode calibration → {calibration_dir}", flush=True)
    _run_module("tools.calibration.calibrate_decode", decode_argv)


def _write_program_jsons(
    *,
    calibration_dir: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> None:
    profile_path = calibration_dir / "calibration.json"
    if not profile_path.is_file():
        raise FileNotFoundError(f"calibration did not produce {profile_path}")

    program_path = output_root / "program-scheduling.json"
    binding_path = output_root / "program-scheduling-binding-only.json"

    capacity = args.token_capacity_per_dp_rank
    if capacity is None:
        capacity = read_observed_kv_capacity_tokens(calibration_dir)
    if capacity is None or capacity <= 0:
        raise ValueError(
            "token capacity is required: pass --token-capacity-per-dp-rank or run decode calibration "
            "so decode/run_config.json exposes observed_kv_capacity_tokens"
        )

    print(
        f"[prepare_l1] writing program scheduling JSON (capacity={capacity})",
        flush=True,
    )
    write_program_scheduling_configs(
        calibration_dir, capacity, program_path, binding_path
    )
    print(f"  {program_path}", flush=True)
    print(f"  {binding_path}", flush=True)


def _ensure_calibration_environment(args: argparse.Namespace) -> None:
    if getattr(args, "agent_infer_root", None):
        os.environ["AGENT_INFER_ROOT"] = str(
            Path(args.agent_infer_root).expanduser().resolve()
        )
    from helpers.paths import ensure_repo_on_sys_path, resolve_calibration_root

    ensure_repo_on_sys_path()
    resolve_calibration_root(required=True)


def bootstrap(args: argparse.Namespace) -> None:
    _ensure_calibration_environment(args)
    repo = _repo_root()
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (repo / output_root).resolve()
    else:
        output_root = output_root.resolve()
    calibration_dir = output_root / "calibration"
    program_path = output_root / "program-scheduling.json"
    binding_path = output_root / "program-scheduling-binding-only.json"

    if args.skip_if_ready and program_path.is_file() and binding_path.is_file():
        print(
            f"[prepare_l1] reusing existing program configs under {output_root}",
            flush=True,
        )
        print(f"  {program_path}", flush=True)
        print(f"  {binding_path}", flush=True)
        return

    profile_path = calibration_dir / "calibration.json"
    need_calibration = args.force or not profile_path.is_file()

    def finish_calibration(base_url: str, model: str) -> None:
        if need_calibration:
            _run_calibration(
                base_url=base_url,
                model=model,
                calibration_dir=calibration_dir,
                args=args,
            )
        _write_program_jsons(
            calibration_dir=calibration_dir, output_root=output_root, args=args
        )

    if args.serve_config_file:
        serve_path = Path(args.serve_config_file).expanduser()
        vllm_config, model_id = build_l1_vllm_e2e_config(serve_path)
        base_url = vllm_config.base_url
        print(f"[prepare_l1] starting vLLM serve from {serve_path}", flush=True)
        print(
            f"[prepare_l1] calibration target {base_url} model={model_id}", flush=True
        )
        with managed_vllm(vllm_config):
            finish_calibration(base_url, model_id)
        print("[prepare_l1] vLLM serve stopped", flush=True)
        return

    if not args.base_url or not args.model:
        raise ValueError(
            "Provide --serve-config-file to start vLLM for L1 (recommended), "
            "or both --base-url and --model when a worker is already running"
        )
    finish_calibration(args.base_url.rstrip("/"), args.model)


def _parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _parse_nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--agent-infer-root",
        type=Path,
        default=None,
        help="Optional: use tools/calibration from an agent-infer source checkout instead of this repo",
    )
    value.add_argument(
        "--serve-config-file",
        type=Path,
        default=None,
        help="JSON with serve_env + server_params; starts vLLM, calibrates, then stops serve",
    )
    value.add_argument(
        "--base-url",
        default=None,
        help="Existing idle worker URL (not Router); use without --serve-config-file",
    )
    value.add_argument(
        "--model",
        default=None,
        help="Served model id for /v1/completions when using --base-url",
    )
    value.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Stable L1 directory for calibration + both program JSON files",
    )
    value.add_argument("--data-parallel-rank", type=_parse_nonnegative_int, default=0)
    value.add_argument(
        "--token-capacity-per-dp-rank",
        type=_parse_positive_int,
        default=None,
        help="KV token capacity for one DP rank; default: decode run_config observed_kv_capacity_tokens",
    )
    value.add_argument("--prompt-lengths", default=DEFAULT_PREFILL_LENGTHS)
    value.add_argument("--prefill-repeats", type=_parse_positive_int, default=2)
    value.add_argument(
        "--decode-batch-sizes",
        default=DEFAULT_DECODE_BATCH_SIZES,
        help="Default 1,2,4,8,12 for max-num-seqs=12 deployments",
    )
    value.add_argument(
        "--decode-context-lengths", default=DEFAULT_DECODE_CONTEXT_LENGTHS
    )
    value.add_argument("--decode-output-tokens", type=_parse_positive_int, default=128)
    value.add_argument("--decode-repeats", type=_parse_positive_int, default=1)
    value.add_argument(
        "--force",
        action="store_true",
        help="Re-run calibration and overwrite stage artifacts even when program JSON exists",
    )
    value.add_argument(
        "--skip-if-ready",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When both program JSON files exist, skip calibration and generation (default: true)",
    )
    return value
