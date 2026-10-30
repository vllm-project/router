"""Shared helpers for E2E tests (router perf replay, agent-aware suites)."""

from .binaries import (
    ROUTER_BIN_ENV,
    VLLM_RS_ENV,
    resolve_binary_path,
    resolve_repo_path,
    resolve_router_bin,
    resolve_vllm_rs,
)
from .http_ready import wait_for_http_ok
from .paths import DEFAULT_ROUTER_BINARY, repo_root
from .subprocess_io import (
    pump_subprocess_stdout,
    sanitize_carriage_return_line,
    start_stderr_pump_thread,
)

__all__ = [
    "DEFAULT_ROUTER_BINARY",
    "ROUTER_BIN_ENV",
    "VLLM_RS_ENV",
    "pump_subprocess_stdout",
    "repo_root",
    "resolve_binary_path",
    "resolve_repo_path",
    "resolve_router_bin",
    "resolve_vllm_rs",
    "sanitize_carriage_return_line",
    "start_stderr_pump_thread",
    "wait_for_http_ok",
]
