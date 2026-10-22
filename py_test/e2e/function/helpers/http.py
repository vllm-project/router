"""HTTP helpers for router functional e2e (worker admin + OpenAI completions)."""

from __future__ import annotations

import requests


def attach_workers(router_url: str, workers) -> None:
    for w in workers:
        r = requests.post(
            f"{router_url}/add_worker", params={"url": w.url}, timeout=180
        )
        r.raise_for_status()


def post_completion(
    session: requests.Session,
    router_url: str,
    *,
    model: str,
    prompt: str,
    stream: bool = False,
    max_tokens: int = 8,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> requests.Response:
    return session.post(
        f"{router_url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": stream,
            "temperature": temperature,
        },
        timeout=timeout,
        stream=stream,
    )


def drain_completion_stream(response: requests.Response) -> None:
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.strip() == "data: [DONE]":
            break
