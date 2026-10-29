"""Shared helpers for E2E tests (router perf replay, agent-aware suites)."""

from .http_ready import wait_for_http_ok
from .paths import DEFAULT_ROUTER_BINARY, repo_root

__all__ = [
    "DEFAULT_ROUTER_BINARY",
    "repo_root",
    "wait_for_http_ok",
]
