"""Run genai-bench against a router and check the recorded metrics."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from tests.python.helpers.e2e.monitor import (
    _gpu_monitor_should_run,
    _launch_gpu_monitor,
    _log_and_assert_gpu_thresholds,
    _read_gpu_monitor_result,
)
from tests.python.helpers.e2e.process import _graceful_stop_any, _which

logger = logging.getLogger(__name__)


def run_genai_bench(
    *,
    router_url: str,
    model_path: str,
    experiment_folder: str,
    timeout_sec: int | None = None,
    thresholds: dict | None = None,
    extra_env: dict | None = None,
    num_concurrency: int = 32,
    traffic_scenario: str = "D(4000,100)",
    max_requests_per_run: int | None = None,
    clean_experiment: bool = True,
    kill_procs: list | None = None,
    drain_delay_sec: int = 6,
) -> None:
    cli = _which("genai-bench")
    if not cli:
        pytest.fail("genai-bench CLI not found; please install it to run benchmarks")

    if clean_experiment:
        exp_dir = Path.cwd() / experiment_folder
        if exp_dir.exists():
            shutil.rmtree(exp_dir, ignore_errors=True)

    requests_per_run = (
        max_requests_per_run
        if max_requests_per_run is not None
        else num_concurrency * 5
    )

    cmd = [
        cli,
        "benchmark",
        "--api-backend",
        "openai",
        "--api-base",
        router_url,
        "--api-key",
        "dummy-token",
        "--api-model-name",
        model_path,
        "--model-tokenizer",
        model_path,
        "--task",
        "text-to-text",
        "--num-concurrency",
        str(num_concurrency),
        "--traffic-scenario",
        traffic_scenario,
        "--max-requests-per-run",
        str(requests_per_run),
        "--max-time-per-run",
        "3",
        "--experiment-folder-name",
        experiment_folder,
        "--experiment-base-dir",
        str(Path.cwd()),
    ]

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    timeout = timeout_sec or int(os.environ.get("GENAI_BENCH_TEST_TIMEOUT", "120"))
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    monitor_path = None
    monitor_proc = None
    want_gpu_monitor = _gpu_monitor_should_run(thresholds)
    if want_gpu_monitor:
        interval = float(os.environ.get("GPU_UTIL_SAMPLE_INTERVAL", "2.0"))
        monitor_proc, monitor_path = _launch_gpu_monitor(
            bench_pid=proc.pid,
            experiment_folder=experiment_folder,
            interval=interval,
        )
    stdout = stderr = ""
    return_code = None
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            stdout, stderr = proc.communicate()
        return_code = proc.returncode

        base = Path.cwd()
        direct = base / experiment_folder
        candidates = [direct] if direct.is_dir() else []
        if not candidates:
            for path in base.rglob(experiment_folder):
                if path.is_dir() and path.name == experiment_folder:
                    candidates = [path]
                    break
        if not candidates:
            raise AssertionError(
                "Benchmark failed: experiment folder not found: "
                f"{experiment_folder}\nExit code: {return_code}\n"
                f"STDOUT (tail):\n{stdout[-1000:]}\nSTDERR (tail):\n{stderr[-1000:]}"
            )
        actual_folder = candidates[0]

        json_files = []
        for _ in range(10):
            json_files = [
                path
                for path in actual_folder.rglob("*.json")
                if "experiment_metadata" not in path.name
            ]
            if json_files:
                break
            time.sleep(1)
        if not json_files:
            raise AssertionError(
                "Benchmark failed: no JSON results found\n"
                f"Exit code: {return_code}\nSTDOUT (tail):\n{stdout[-1000:]}\n"
                f"STDERR (tail):\n{stderr[-1000:]}"
            )

        for result_path in json_files:
            with result_path.open("r") as handle:
                data = json.load(handle)
            stats = data.get("aggregated_metrics", {}).get("stats", {})
            ttft_mean = float(stats.get("ttft", {}).get("mean", float("inf")))
            e2e_latency_mean = float(
                stats.get("e2e_latency", {}).get("mean", float("inf"))
            )
            input_tp_mean = float(stats.get("input_throughput", {}).get("mean", 0.0))
            output_tp_mean = float(stats.get("output_throughput", {}).get("mean", 0.0))

            logger.info(
                "genai-bench[%s] %s ttft_mean=%.3fs e2e_latency_mean=%.3fs input_tp_mean=%.1f tok/s output_tp_mean=%.1f tok/s",
                experiment_folder,
                result_path.name,
                ttft_mean,
                e2e_latency_mean,
                input_tp_mean,
                output_tp_mean,
            )

            if thresholds is not None:
                assert ttft_mean <= thresholds["ttft_mean_max"], (
                    f"TTFT validation failed: {ttft_mean} > {thresholds['ttft_mean_max']} (file={result_path.name})"
                )
                assert e2e_latency_mean <= thresholds["e2e_latency_mean_max"], (
                    f"E2E latency validation failed: {e2e_latency_mean} > {thresholds['e2e_latency_mean_max']} (file={result_path.name})"
                )
                assert input_tp_mean >= thresholds["input_throughput_mean_min"], (
                    f"Input throughput validation failed: {input_tp_mean} < {thresholds['input_throughput_mean_min']} (file={result_path.name})"
                )
                assert output_tp_mean >= thresholds["output_throughput_mean_min"], (
                    f"Output throughput validation failed: {output_tp_mean} < {thresholds['output_throughput_mean_min']} (file={result_path.name})"
                )

        if want_gpu_monitor:
            try:
                if monitor_proc is not None:
                    monitor_proc.join(timeout=5)
            except Exception:
                pass
            _log_and_assert_gpu_thresholds(
                _read_gpu_monitor_result(monitor_path), thresholds
            )
    finally:
        if kill_procs:
            if drain_delay_sec > 0:
                try:
                    time.sleep(drain_delay_sec)
                except Exception:
                    pass
            for proc_to_stop in kill_procs:
                _graceful_stop_any(proc_to_stop)
            try:
                time.sleep(2)
            except Exception:
                pass
        if monitor_proc is not None and monitor_proc.is_alive():
            try:
                monitor_proc.terminate()
            except Exception:
                pass
