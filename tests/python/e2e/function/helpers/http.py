"""HTTP helpers for router functional e2e (worker admin + OpenAI chat)."""

from __future__ import annotations

import json
import time
from typing import Any

import requests


def post_chat_completion(
    session: requests.Session,
    router_url: str,
    *,
    model: str,
    content: str,
    stream: bool = False,
    max_tokens: int = 8,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> requests.Response:
    return session.post(
        f"{router_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "stream": stream,
            "temperature": temperature,
        },
        timeout=timeout,
        stream=stream,
    )


def cached_tokens_from_usage(usage: dict[str, Any]) -> int:
    if not usage:
        return 0
    for key in ("prompt_tokens_details", "input_tokens_details"):
        details = usage.get(key)
        if isinstance(details, dict):
            cached = details.get("cached_tokens")
            if isinstance(cached, int):
                return cached
    return 0


def post_chat_stream_timed(
    session: requests.Session,
    router_url: str,
    *,
    model: str,
    content: str,
    max_tokens: int = 8,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Stream chat; return client ``ttft_ms``, final ``usage``, optional ``stages``."""
    response = session.post(
        f"{router_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": temperature,
            "stream_options": {"include_usage": True},
        },
        timeout=timeout,
        stream=True,
    )
    response.raise_for_status()
    stages_raw = response.headers.get("x-router-stages")
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    usage: dict[str, Any] = {}
    request_id: str | None = None
    saw_done = False
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith(": router-stages "):
            stages_raw = line[len(": router-stages ") :].strip()
            continue
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            saw_done = True
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if obj.get("error") is not None:
            raise RuntimeError(f"SSE error: {obj['error']}")
        chunk_id = obj.get("id")
        if isinstance(chunk_id, str) and chunk_id:
            request_id = chunk_id
        if obj.get("usage"):
            usage = obj["usage"]
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = (choices[0].get("delta") or {}) if choices else {}
        chunk = (
            delta.get("content")
            or delta.get("reasoning")
            or delta.get("reasoning_content")
            or ""
        )
        if chunk and ttft_ms is None:
            ttft_ms = (time.perf_counter() - t0) * 1000.0
    if not saw_done:
        raise RuntimeError("incomplete SSE stream (EOF before [DONE])")
    if ttft_ms is None:
        raise RuntimeError("no streamed output (ttft unavailable)")
    if not request_id:
        raise RuntimeError("stream ended without OpenAI completion id")
    stages: dict[str, Any] | None = None
    if stages_raw:
        try:
            parsed = json.loads(stages_raw)
            if isinstance(parsed, dict):
                stages = parsed
        except json.JSONDecodeError:
            stages = None
    return {
        "request_id": request_id,
        "ttft_ms": ttft_ms,
        "usage": usage,
        "cached_tokens": cached_tokens_from_usage(usage),
        "stages": stages,
    }
