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
from .perf_process import (
    attach_stderr_log_pump,
    graceful_stop_subprocess,
    join_log_pump,
    merge_process_env,
    popen_merged_output,
    preflight_router_bin,
    router_env_for_model,
    start_process_with_log,
    subprocess_exit_error,
    wait_router_http_health,
)
from .subprocess_io import (
    pump_subprocess_stdout,
    sanitize_carriage_return_line,
    start_stderr_pump_thread,
)

__all__ = [
    "DEFAULT_ROUTER_BINARY",
    "ROUTER_BIN_ENV",
    "VLLM_RS_ENV",
    "attach_stderr_log_pump",
    "graceful_stop_subprocess",
    "join_log_pump",
    "merge_process_env",
    "popen_merged_output",
    "preflight_router_bin",
    "pump_subprocess_stdout",
    "repo_root",
    "resolve_binary_path",
    "resolve_repo_path",
    "resolve_router_bin",
    "resolve_vllm_rs",
    "router_env_for_model",
    "sanitize_carriage_return_line",
    "start_process_with_log",
    "start_stderr_pump_thread",
    "subprocess_exit_error",
    "wait_for_http_ok",
    "wait_router_http_health",
]
