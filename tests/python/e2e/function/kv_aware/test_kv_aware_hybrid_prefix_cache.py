"""Compare Router KV credit with actual reuse on hybrid model workers.

Set ROUTER_BIN to a binary that includes PR #493 and run with vLLM 0.30.0.
Merge cases can use one worker for cache accounting or multiple workers to
also check cache-holder selection. 1P1D requires two workers. GPUs per worker
are set by tensor parallel size. Use `-k` to select a model, protocol, or variant.
An optional `case_id` preserves IDs across deployment-specific JSON files.
The 1P1D cases require NixlConnector on both workers.
Set the model's `local_model_env` from cases/hybrid_model/*.json to use a local
checkpoint.
The DeepSeek V4 Pro NVFP4 case targets Blackwell GPUs.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
import requests

from tests.python.e2e.function.kv_aware.config import (
    kv_aware_request_log_poll_sec,
    load_hybrid_kv_aware_cases,
)
from tests.python.e2e.function.kv_aware.request_builder import (
    build_hybrid_prefix_requests,
)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_PREFIX_DECISION = re.compile(
    r'KV prefix decision\b[^\n]*?\bworker="?(?P<worker>[^"\s]+)"?'
    r"\s+input_tokens=(?P<input>\d+)\s+new_prefill_tokens=\d+"
    r"\s+credited_prefix_tokens=(?P<credit>\d+)\b"
)
_PD_PREFIX_DECISION = re.compile(
    r'P/D admission\b[^\n]*?\broute="?/v1/completions"?\s+'
    r'prefill_policy="?kv_aware"?\s+[^\n]*?'
    r"\bcredited_prefix_tokens=Some\((?P<credit>\d+)\)"
)
_WORKER_COUNTER = re.compile(
    r"^(vllm:(?:prefix_cache_(?:queries|hits)(?:_total)?|"
    r"external_prefix_cache_hits_total|"
    r"nixl_bytes_transferred_(?:sum|count)|"
    r"nixl_num_(?:failed_transfers|failed_notifications|kv_expired_reqs)_total|"
    r"spec_decode_num_drafts_total))(?:\{[^\n]*\})?[ \t]+(\S+)",
    re.MULTILINE,
)
_NIXL_TRANSFERS = (
    "vllm:nixl_bytes_transferred_count",
    "vllm:nixl_bytes_transferred_sum",
)
_EXTERNAL_PREFIX_HITS = "vllm:external_prefix_cache_hits_total"
_NIXL_ERRORS = (
    "vllm:nixl_num_failed_transfers_total",
    "vllm:nixl_num_failed_notifications_total",
    "vllm:nixl_num_kv_expired_reqs_total",
)
_DRAFTS = "vllm:spec_decode_num_drafts_total"


def _hybrid_parameters(deployment: str):
    parameters = []
    case_ids = set()
    for name, case in load_hybrid_kv_aware_cases().items():
        if deployment not in case.get("deployments", ("merge",)):
            continue
        name = case.get("case_id", name)
        for protocol in case["protocols"]:
            case_id = f"{name}-{protocol}"
            assert case_id not in case_ids, f"Duplicate {deployment} case: {case_id}"
            case_ids.add(case_id)
            parameters.append(pytest.param((case, protocol, deployment), id=case_id))
    return parameters


def _wait_for_router_log(
    path: Path, pattern: re.Pattern[str], *, since: int = 0
) -> list[re.Match[str]]:
    timeout = kv_aware_request_log_poll_sec()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with path.open("rb") as log:
            log.seek(since)
            complete_lines = log.read().rpartition(b"\n")[0]
        text = _ANSI.sub("", complete_lines.decode("utf-8", errors="replace"))
        matches = list(pattern.finditer(text))
        if matches:
            return matches
        time.sleep(0.1)
    pytest.fail(f"Missing Router log {pattern.pattern!r} within {timeout}s in {path}")


def _complete(
    session: requests.Session,
    url: str,
    model: str,
    tokens: list[int],
    *,
    max_tokens: int = 1,
) -> tuple[str, dict]:
    payload = {
        "model": model,
        "prompt": tokens,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if max_tokens > 1:
        payload["ignore_eos"] = True
    response = session.post(
        f"{url}/v1/completions",
        json=payload,
        timeout=120,
    )
    assert response.ok, f"{url} returned {response.status_code}: {response.text}"
    body = response.json()
    request_id = response.headers.get("x-request-id") or body["id"]
    assert body["choices"], f"{request_id}: missing completion"
    assert all(
        choice["finish_reason"] in ("stop", "length") for choice in body["choices"]
    ), f"{request_id}: unexpected completion finish: {body['choices']}"
    usage = body["usage"]
    assert usage["prompt_tokens"] == len(tokens), f"{request_id}: {usage}"
    return request_id, usage


def _worker_counters(session: requests.Session, url: str) -> dict[str, int]:
    response = session.get(f"{url}/metrics", timeout=10)
    response.raise_for_status()
    counters: dict[str, int] = {}
    for name, sample in _WORKER_COUNTER.findall(response.text):
        value = float(sample)
        assert value.is_integer() and value >= 0, f"{url}: invalid {name}: {sample}"
        counters[name] = counters.get(name, 0) + int(value)
    return counters


def _worker_cache_totals(session: requests.Session, url: str) -> tuple[int, int]:
    counters = _worker_counters(session, url)

    totals = []
    for name in ("vllm:prefix_cache_queries", "vllm:prefix_cache_hits"):
        total = counters.get(f"{name}_total", counters.get(name))
        assert total is not None, f"{url}/metrics: missing {name} counter"
        totals.append(total)
    return totals[0], totals[1]


def _wait_worker_cache_hits(
    session: requests.Session,
    url: str,
    baseline: tuple[int, int],
    prompt_tokens: int,
    request_id: str,
) -> int:
    # The query counter includes every token queried by this worker, even on a miss.
    # Wait for it to account for this request before reading the hit delta.
    deadline = time.monotonic() + kv_aware_request_log_poll_sec()
    while time.monotonic() < deadline:
        queries, hits = _worker_cache_totals(session, url)
        query_delta = queries - baseline[0]
        hit_delta = hits - baseline[1]
        assert 0 <= query_delta <= prompt_tokens and hit_delta >= 0, (
            f"{request_id}: unexpected cache counter deltas on {url}: "
            f"queries={query_delta}, hits={hit_delta}, prompt_tokens={prompt_tokens}"
        )
        if query_delta == prompt_tokens:
            return hit_delta
        time.sleep(0.1)
    pytest.fail(f"{request_id}: {url} did not report {prompt_tokens} new cache queries")


def _wait_for_counter_increases(
    session: requests.Session, url: str, baseline: dict[str, int]
) -> None:
    deadline = time.monotonic() + kv_aware_request_log_poll_sec()
    while time.monotonic() < deadline:
        counters = _worker_counters(session, url)
        if all(counters[name] > value for name, value in baseline.items()):
            return
        time.sleep(0.1)
    pytest.fail(f"{url} did not report activity above {baseline}")


def _prefix_probes(stack):
    prompts = build_hybrid_prefix_requests(
        stack.cache_block_size, prefill_backoff=stack.prefill_backoff
    )
    max_tokens = stack.speculative_tokens + 2 if stack.speculative_tokens else 1
    longest_prompt = max(map(len, prompts.values()))
    assert longest_prompt + max_tokens <= stack.max_model_len, (
        f"{stack.model}: {longest_prompt} prompt tokens and {max_tokens} output tokens "
        f"exceed the context limit of {stack.max_model_len}"
    )
    return prompts, max_tokens


def _prime_prefixes(session, stack, prompts, scored_workers):
    warm_worker = scored_workers[0]
    warm_http_url = stack.worker_http_urls[warm_worker]
    sources = {}
    for worker in scored_workers:
        matches = _wait_for_router_log(
            stack.log_path,
            re.compile(
                rf"kv_index {re.escape(worker)}: spawned subscriber rank0 "
                r"\(instance (?P<source>[^)\n]+)\)"
            ),
        )
        sources[worker] = matches[-1]["source"]
    applied = re.compile(
        rf'kv_index (?:live|replay) sequence advanced\b[^\n]*\bsource="?{re.escape(sources[warm_worker])}"?'
        r"\s+rank=0\s+seq=\d+\b"
    )
    # Seed the shorter checkpoint first. A long Mamba prefill can skip it.
    for prime in ("partial_prime", "warm_prime"):
        tokens = prompts[prime]
        baseline = (
            _worker_cache_totals(session, warm_http_url)
            if stack.protocol == "grpc" or stack.deployment == "1p1d_nixl"
            else None
        )
        since = stack.log_path.stat().st_size
        request_id, _ = _complete(session, warm_http_url, stack.model, tokens)
        _wait_for_router_log(stack.log_path, applied, since=since)
        if baseline is not None:
            _wait_worker_cache_hits(
                session, warm_http_url, baseline, len(tokens), request_id
            )


@pytest.mark.e2e
@pytest.mark.parametrize(
    "hybrid_kv_aware_stack", _hybrid_parameters("merge"), indirect=True
)
def test_kv_aware_hybrid_prefix_reuse(hybrid_kv_aware_stack):
    stack = hybrid_kv_aware_stack
    scored_workers = tuple(stack.worker_http_urls)
    warm_worker = scored_workers[0]
    warm_http_url = stack.worker_http_urls[warm_worker]
    prompts, max_tokens = _prefix_probes(stack)

    with requests.Session() as session:
        _prime_prefixes(session, stack, prompts, scored_workers)
        drafts = (
            _worker_counters(session, warm_http_url)[_DRAFTS]
            if stack.speculative_tokens
            else None
        )
        reuse = {}
        for probe in ("warm", "partial", "cold"):
            tokens = prompts[probe]
            baselines = {}
            if stack.protocol == "grpc":
                baselines = {
                    worker: _worker_cache_totals(session, http_url)
                    for worker, http_url in stack.worker_http_urls.items()
                }
            # Send one request at a time so its decision can be matched to the log.
            since = stack.log_path.stat().st_size
            request_id, usage = _complete(
                session, stack.url, stack.model, tokens, max_tokens=max_tokens
            )
            matches = _wait_for_router_log(
                stack.log_path, _PREFIX_DECISION, since=since
            )
            assert len(matches) == 1, f"{probe} {request_id}: {matches}"
            decision = matches[0]
            worker = decision["worker"]
            router_credit = int(decision["credit"])
            assert int(decision["input"]) == len(tokens), f"{probe} {request_id}"
            assert worker in stack.worker_http_urls, f"{probe} {request_id}: {worker}"
            if stack.protocol == "grpc":
                worker_credit = _wait_worker_cache_hits(
                    session,
                    stack.worker_http_urls[worker],
                    baselines[worker],
                    len(tokens),
                    request_id,
                )
            else:
                worker_credit = (usage.get("prompt_tokens_details") or {}).get(
                    "cached_tokens"
                )
                assert type(worker_credit) is int, (
                    f"{request_id}: missing cached_tokens in {usage}"
                )
            assert 0 <= worker_credit < len(tokens), (
                f"{probe} {request_id}: {worker_credit}"
            )
            assert router_credit == worker_credit, (
                f"{probe} {request_id}: Router credited {router_credit} tokens, "
                f"{worker} reused {worker_credit}"
            )
            if probe == "warm" and len(scored_workers) > 1:
                assert worker == warm_worker, (
                    f"{request_id}: expected warm worker {warm_worker}, got {worker}"
                )
            reuse[probe] = worker_credit
            print(
                f"[e2e] {probe} {request_id} worker={worker} "
                f"router_tokens={router_credit} worker_tokens={worker_credit}",
                flush=True,
            )
        if drafts is not None:
            _wait_for_counter_increases(session, warm_http_url, {_DRAFTS: drafts})

    assert 0 < reuse["partial"] < reuse["warm"], reuse
    assert reuse["cold"] == 0, reuse


@pytest.mark.e2e
@pytest.mark.parametrize(
    "hybrid_kv_aware_stack", _hybrid_parameters("1p1d_nixl"), indirect=True
)
def test_kv_aware_hybrid_prefix_reuse_1p1d_nixl(hybrid_kv_aware_stack):
    stack = hybrid_kv_aware_stack
    prefill, decode = stack.worker_http_urls
    prefill_url = stack.worker_http_urls[prefill]
    decode_url = stack.worker_http_urls[decode]
    prompts, max_tokens = _prefix_probes(stack)

    with requests.Session() as session:
        _prime_prefixes(session, stack, prompts, (prefill,))
        prefill_before = _worker_counters(session, prefill_url)
        decode_before = _worker_counters(session, decode_url)

        reuse = {}
        for probe in ("warm", "partial", "cold"):
            tokens = prompts[probe]
            prefill_baseline = _worker_cache_totals(session, prefill_url)
            since = stack.log_path.stat().st_size
            request_id, _ = _complete(
                session, stack.url, stack.model, tokens, max_tokens=max_tokens
            )
            matches = _wait_for_router_log(
                stack.log_path, _PD_PREFIX_DECISION, since=since
            )
            assert len(matches) == 1, f"{probe} {request_id}: {matches}"
            router_credit = int(matches[0]["credit"])
            prefill_tokens = len(tokens) - stack.prefill_backoff
            prefill_credit = _wait_worker_cache_hits(
                session, prefill_url, prefill_baseline, prefill_tokens, request_id
            )
            assert 0 <= prefill_credit < len(tokens), (
                f"{probe} {request_id}: {prefill_credit}"
            )
            assert router_credit == prefill_credit, (
                f"{probe} {request_id}: Router credited {router_credit} tokens, "
                f"Prefill reused {prefill_credit}"
            )
            reuse[probe] = prefill_credit
            print(
                f"[e2e] 1p1d_nixl {probe} {request_id} "
                f"router_tokens={router_credit} prefill_tokens={prefill_credit}",
                flush=True,
            )

        # The Rust frontend can report external hits with empty NIXL byte
        # histograms. A failed KV load fails the completion.
        transfers = (
            (_EXTERNAL_PREFIX_HITS,) if stack.protocol == "grpc" else _NIXL_TRANSFERS
        )
        activity = {name: decode_before[name] for name in transfers}
        if stack.speculative_tokens:
            activity[_DRAFTS] = decode_before[_DRAFTS]
        _wait_for_counter_increases(session, decode_url, activity)
        for url, before in ((prefill_url, prefill_before), (decode_url, decode_before)):
            after = _worker_counters(session, url)
            for name in _NIXL_ERRORS:
                assert after[name] == before[name], (
                    f"{url}: {name} increased from {before[name]} to {after[name]}"
                )

    assert 0 < reuse["partial"] < reuse["warm"], reuse
    assert reuse["cold"] == 0, reuse
