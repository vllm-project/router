"""Integration tests for WASM OnRequest middleware (real Router + mock worker)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "wasm_middleware"
WASM_COMPONENT = EXAMPLE_DIR / "wasm_middleware_example.component.wasm"


def _router_bin() -> Path:
    for candidate in (
        REPO_ROOT / "target" / "release" / "vllm-router",
        REPO_ROOT / "target" / "debug" / "vllm-router",
    ):
        if candidate.is_file():
            return candidate
    pytest.skip(
        "vllm-router binary not found; build with "
        "`cargo build --release --bin vllm-router` first"
    )


def _ensure_wasm_component() -> Path:
    if WASM_COMPONENT.is_file():
        return WASM_COMPONENT
    build_sh = EXAMPLE_DIR / "build.sh"
    if not build_sh.is_file():
        pytest.skip(f"missing example plugin build script at {build_sh}")
    subprocess.check_call(["bash", str(build_sh)], cwd=EXAMPLE_DIR)
    if not WASM_COMPONENT.is_file():
        pytest.skip(f"failed to build WASM component at {WASM_COMPONENT}")
    return WASM_COMPONENT


def _header_map(payload: dict) -> dict:
    headers = payload.get("request_headers") or {}
    return {str(k).lower(): v for k, v in headers.items()}


@pytest.mark.integration
def test_wasm_middleware_modify_reject_and_path_isolation(router_manager, mock_workers):
    """Example plugin: set header on chat, reject marker body, skip other routes."""
    wasm_path = _ensure_wasm_component()
    router_bin = _router_bin()
    _, urls, _ = mock_workers(n=1)

    rh = router_manager.start_router(
        worker_urls=urls,
        policy="round_robin",
        router_bin=str(router_bin),
        extra={
            "worker_startup_timeout_secs": 30,
            "wasm_middleware": str(wasm_path),
        },
    )

    # Modify: chat should forward with the example header.
    chat = requests.post(
        f"{rh.url}/v1/chat/completions",
        json={
            "model": "mock",
            "messages": [{"role": "user", "content": "hi"}],
        },
        timeout=30,
    )
    assert chat.status_code == 200, chat.text
    chat_headers = _header_map(chat.json())
    assert chat_headers.get("x-wasm-middleware") == "example", chat_headers

    # Reject: fail closed at the Router before the worker.
    rejected = requests.post(
        f"{rh.url}/v1/chat/completions",
        json={
            "model": "mock",
            "messages": [{"role": "user", "content": "nope"}],
            "note": "__wasm_reject__",
        },
        timeout=30,
    )
    assert rejected.status_code == 400, rejected.text

    # Path isolation: completions is not attached by default.
    completions = requests.post(
        f"{rh.url}/v1/completions",
        json={"model": "mock", "prompt": "hi"},
        timeout=30,
    )
    assert completions.status_code == 200, completions.text
    completion_headers = _header_map(completions.json())
    assert "x-wasm-middleware" not in completion_headers, completion_headers
