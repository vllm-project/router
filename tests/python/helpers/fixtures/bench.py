"""Benchmark fixture. The genai-bench runner lives in ``tests.python.helpers.e2e.bench``."""

from __future__ import annotations

from typing import Callable

import pytest

from tests.python.helpers.e2e.bench import run_genai_bench


@pytest.fixture(scope="session")
def genai_bench_runner() -> Callable[..., None]:
    """Run genai-bench and check the recorded metrics.

    Usage in tests:
      def test(..., genai_bench_runner):
          genai_bench_runner(router_url=..., model_path=..., experiment_folder=...)
    """
    return run_genai_bench
