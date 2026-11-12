"""Shared artifact roots for performance results and diagnostic run files.

Suites append their own case names, timestamps, and filenames. Keep the
Buildkite artifact globs and dashboard collector in sync when changing roots.
"""

from pathlib import Path

_E2E_DIR = Path(__file__).resolve().parents[1]


def perf_results_dir(suite: str) -> Path:
    """Dashboard result root for a performance suite."""
    return _E2E_DIR / "results" / "perf" / suite


def perf_run_dir(suite: str) -> Path:
    """Diagnostic run root for a performance suite."""
    return _E2E_DIR / "run" / "perf" / suite
