"""HTTP helpers for router and mock-worker endpoints. One function per endpoint."""

from __future__ import annotations

import concurrent.futures
from typing import Any

import requests


def _completion_payload(
    prompt, extra: dict[str, Any] | None, max_tokens: int
) -> dict[str, Any]:
    payload = {
        "model": "test-model",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if extra:
        payload.update(extra)
    return payload


def completion(
    base_url: str,
    prompt=None,
    *,
    prompts=None,
    extra: dict[str, Any] | None = None,
    timeout: float | None = None,
    max_tokens: int = 1,
    max_workers: int = 1,
    ignore_errors: bool = False,
):
    """POST ``/v1/completions`` and return the worker id.

    One ``prompt`` returns one worker id. ``prompts`` sends that sequence
    concurrently, in order, and returns a list. A token-id list is one prompt,
    so a batch has to be passed as ``prompts``. ``ignore_errors`` returns None
    for a failed request instead of raising.
    """
    if (prompt is None) == (prompts is None):
        raise ValueError("pass exactly one of prompt or prompts")
    url = f"{base_url.rstrip('/')}/v1/completions"

    def send(item):
        try:
            response = requests.post(
                url,
                json=_completion_payload(item, extra, max_tokens),
                timeout=timeout,
            )
            assert response.status_code == 200, response.text
            return worker_id(response)
        except Exception:
            if ignore_errors:
                return None
            raise

    if prompts is None:
        return send(prompt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(send, prompts))


def get_load(
    base_url: str,
    timeout: float = 1,
    session: requests.Session | None = None,
):
    """GET ``/get_load``. Return the reported load, or None when the probe fails."""
    request = session.get if session is not None else requests.get
    try:
        return (
            request(f"{base_url.rstrip('/')}/get_load", timeout=timeout)
            .json()
            .get("load")
        )
    except requests.RequestException:
        return None


def worker_id(response):
    return response.headers.get("X-Worker-Id") or response.json().get("worker_id")
