"""Sample GPU utilization while a benchmark process is running."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _gpu_monitor_should_run(thresholds: Optional[dict]) -> bool:
    """Decide whether to enable the GPU monitor.

    Runs if thresholds request GPU checks or if GPU_UTIL_LOG is truthy.
    """
    want = False
    try:
        mean_th = None if thresholds is None else thresholds.get("gpu_util_mean_min")
        p50_th = None if thresholds is None else thresholds.get("gpu_util_p50_min")
        want = bool(mean_th is not None or p50_th is not None)
    except Exception:
        want = False
    if not want:
        env_flag = os.environ.get("GPU_UTIL_LOG", "").lower() in ("1", "true", "yes")
        want = want or env_flag
    return want


def _gpu_monitor_path(experiment_folder: str) -> str:
    """Return the JSON path for storing GPU monitor results."""
    base = Path.cwd() / experiment_folder
    return str(base / "gpu_utilization.json")


def _launch_gpu_monitor(bench_pid: int, experiment_folder: str, interval: float):
    """Start the GPU monitor process. Returns (proc, path) or (None, None)."""
    try:
        from multiprocessing import Process

        out_path = _gpu_monitor_path(experiment_folder)
        proc = Process(
            target=_gpu_monitor_proc_entry,
            args=(bench_pid, out_path, interval),
            daemon=True,
        )
        proc.start()
        return proc, out_path
    except Exception as e:
        logger.warning("Failed to launch GPU monitor: %s", e)
        return None, None


def _read_gpu_monitor_result(path: Optional[str]) -> Optional[dict]:
    try:
        if path and os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("Failed to read GPU monitor result from %r: %s", path, e)
    return None


def _log_and_assert_gpu_thresholds(
    result: Optional[dict], thresholds: Optional[dict]
) -> None:
    if not result or not isinstance(result, dict) or result.get("count", 0) <= 0:
        logger.warning("GPU utilization monitor produced no samples.")
        return

    overall = result.get("overall", {}) if isinstance(result, dict) else {}
    count = int(result.get("count", 0))
    mean_th = None if thresholds is None else thresholds.get("gpu_util_mean_min")
    p50_th = None if thresholds is None else thresholds.get("gpu_util_p50_min")

    mean_v = float(overall.get("mean", 0.0))
    p50_v = overall.get("p50")

    logger.info(
        "GPU utilization overall: mean=%.2f%% p50=%s (samples=%d)",
        mean_v,
        (f"{float(p50_v):.2f}%" if p50_v is not None else "n/a"),
        count,
    )

    if mean_th is not None:
        assert mean_v >= float(mean_th), (
            f"GPU utilization mean below threshold: {mean_v:.2f}% < {mean_th}%"
        )
    if p50_th is not None and p50_v is not None:
        p50_f = float(p50_v)
        assert p50_f >= float(p50_th), (
            f"GPU utilization p50 below threshold: {p50_f:.2f}% < {p50_th}%"
        )


def _gpu_monitor_proc_entry(bench_pid: int, out_file: str, interval: float) -> None:
    """Low-impact GPU utilization monitor using NVML in a separate process.

    Writes JSON to out_file that includes overall and per-GPU raw samples and summary stats.
    """
    try:
        try:
            os.nice(10)
        except Exception:
            pass
        total = 0.0
        n = 0
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
        except Exception:
            with open(out_file, "w") as f:
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                json.dump(
                    {
                        "count": 0,
                        "overall": {"mean": 0.0},
                        "per_gpu": {},
                        "raw": {},
                    },
                    f,
                )
            return
        try:
            import pynvml  # type: ignore

            count = pynvml.nvmlDeviceGetCount()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
        except Exception:
            with open(out_file, "w") as f:
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                json.dump(
                    {
                        "count": 0,
                        "overall": {"mean": 0.0},
                        "per_gpu": {},
                        "raw": {},
                    },
                    f,
                )
            return

        per_gpu_samples: dict[str, list[float]] = {}
        overall_samples: list[float] = []

        while True:
            if not os.path.exists(f"/proc/{bench_pid}"):
                break
            try:
                vals = []
                import pynvml  # type: ignore

                for idx, handle in enumerate(handles):
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        vals.append(float(util))
                        key = str(idx)
                        per_gpu_samples.setdefault(key, []).append(float(util))
                    except Exception:
                        continue
                if vals:
                    avg = sum(vals) / len(vals)
                    overall_samples.append(avg)
                    total += avg
                    n += 1
            except Exception:
                pass
            time.sleep(interval)
    finally:
        try:
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, "w") as f:

                def pct_from(samples: list[float], p: float) -> float:
                    if not samples:
                        return 0.0
                    ordered = sorted(samples)
                    index = max(
                        0,
                        min(
                            len(ordered) - 1,
                            int(round((p / 100.0) * (len(ordered) - 1))),
                        ),
                    )
                    return float(ordered[index])

                overall_mean = (total / n) if n > 0 else 0.0

                per_gpu_summary: dict[str, dict] = {}
                for key, samples in per_gpu_samples.items():
                    per_gpu_summary[key] = {
                        "mean": float(sum(samples) / len(samples)) if samples else 0.0,
                        "p5": pct_from(samples, 5),
                        "p10": pct_from(samples, 10),
                        "p25": pct_from(samples, 25),
                        "p50": pct_from(samples, 50),
                        "p75": pct_from(samples, 75),
                        "p90": pct_from(samples, 90),
                        "p95": pct_from(samples, 95),
                        "min": float(min(samples)) if samples else 0.0,
                        "max": float(max(samples)) if samples else 0.0,
                        "count": len(samples),
                    }

                out_payload = {
                    "bench_pid": bench_pid,
                    "interval_sec": interval,
                    "count": n,
                    "overall": {
                        "mean": float(overall_mean),
                        "p5": pct_from(overall_samples, 5),
                        "p10": pct_from(overall_samples, 10),
                        "p25": pct_from(overall_samples, 25),
                        "p50": pct_from(overall_samples, 50),
                        "p75": pct_from(overall_samples, 75),
                        "p90": pct_from(overall_samples, 90),
                        "p95": pct_from(overall_samples, 95),
                        "min": float(min(overall_samples)) if overall_samples else 0.0,
                        "max": float(max(overall_samples)) if overall_samples else 0.0,
                    },
                    "per_gpu": per_gpu_summary,
                    "raw": {
                        "overall": overall_samples,
                        "per_gpu": per_gpu_samples,
                    },
                }
                json.dump(out_payload, f)
        except Exception:
            pass
        try:
            import pynvml  # type: ignore

            pynvml.nvmlShutdown()
        except Exception:
            pass
