"""Functional e2e: kv_aware routing policy (PR #371 / cost-based worker selection)."""

from __future__ import annotations

import concurrent.futures

import pytest
import requests

# Shared prefix long enough to exercise prefill accounting on a small model.
_KV_AWARE_SHARED_PREFIX = (
    "You are a concise assistant. Answer in one short sentence. Context block: "
    + ("token-" * 48)
)


def _attach_workers(router_url: str, workers) -> None:
    for w in workers:
        r = requests.post(
            f"{router_url}/add_worker", params={"url": w.url}, timeout=180
        )
        r.raise_for_status()


def _completion(
    session: requests.Session,
    base_url: str,
    model: str,
    prompt_suffix: str,
    *,
    stream: bool = False,
    max_tokens: int = 8,
) -> requests.Response:
    return session.post(
        f"{base_url}/v1/completions",
        json={
            "model": model,
            "prompt": _KV_AWARE_SHARED_PREFIX + prompt_suffix,
            "max_tokens": max_tokens,
            "stream": stream,
            "temperature": 0.0,
        },
        timeout=120,
        stream=stream,
    )


def _drain_completion_stream(response: requests.Response) -> None:
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.strip() == "data: [DONE]":
            break


@pytest.mark.e2e
def test_kv_aware_live_routing_and_streaming(
    e2e_router_only_kv_aware,
    e2e_two_workers_single_gpu,
    e2e_model,
):
    """
    Smoke-test kv_aware on a regular HTTP router with two vLLM workers.

    HTTP completions do not supply token IDs, so placement credit falls back to
    load-only scoring; this still validates discovery wiring, worker registration,
    selection, and request lifecycle (including streaming slot release).
    Requires at least two GPUs (see e2e_two_workers_single_gpu).
    """
    base = e2e_router_only_kv_aware.url
    workers = e2e_two_workers_single_gpu
    _attach_workers(base, workers)

    listed = requests.get(f"{base}/list_workers", timeout=30)
    listed.raise_for_status()
    urls = listed.json().get("urls", [])
    assert len(urls) == 2
    assert set(urls) == {w.url for w in workers}

    with requests.Session() as session:
        warm = _completion(session, base, e2e_model, " warm", stream=False)
        warm.raise_for_status()
        assert warm.json().get("choices")

        def one_request(i: int) -> None:
            r = _completion(session, base, e2e_model, f" req-{i}", stream=False)
            r.raise_for_status()
            body = r.json()
            assert body.get("choices"), body

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(one_request, range(16)))

        stream_resp = _completion(
            session, base, e2e_model, " stream", stream=True, max_tokens=4
        )
        stream_resp.raise_for_status()
        _drain_completion_stream(stream_resp)
