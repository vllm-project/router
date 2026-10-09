import concurrent.futures

import pytest
import requests


@pytest.mark.integration
def test_rate_limit_and_queue(router_manager, mock_workers):
    # Add latency to ensure requests overlap and trigger rate limiting
    # Without latency, requests may complete too fast to trigger concurrency limits
    _, urls, _ = mock_workers(n=1, args=["--latency-ms", "100"])
    rh = router_manager.start_router(
        worker_urls=urls,
        policy="round_robin",
        extra={
            "max_concurrent_requests": 2,
            "queue_size": 0,  # no queue -> immediate 429 when limit exceeded
        },
    )

    def call_once(i):
        try:
            r = requests.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": f"p{i}",
                    "max_tokens": 1,
                    "stream": False,
                },
                timeout=3,
            )
            return r.status_code
        except Exception:
            return 599

    # Fire a burst of concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(call_once, range(16)))

    # Expect some to succeed and some to be rate limited (429)
    assert any(code == 200 for code in results)
    assert any(code == 429 for code in results)


@pytest.mark.integration
def test_rate_limit_queue_and_timeout(router_manager, mock_workers):
    # Slow backend: ~2s per request ensures queue wait > timeout
    _, urls, _ = mock_workers(n=1, args=["--latency-ms", "2000"])  # 2.0s per request

    # Allow 1 concurrent, queue up to 1, with 1s queue timeout
    rh = router_manager.start_router(
        worker_urls=urls,
        policy="round_robin",
        extra={
            "max_concurrent_requests": 1,
            "queue_size": 1,
            "queue_timeout_secs": 1,
        },
    )

    def call_once(i):
        try:
            r = requests.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": f"q{i}",
                    "max_tokens": 1,
                    "stream": False,
                },
                timeout=5,
            )
            return r.status_code
        except Exception:
            return 599

    # Fire 4 concurrent requests: 1 runs (~2s), 1 queued (times out at 1s -> 408), 2 overflow -> 429
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(call_once, range(4)))

    # We expect:
    # - Some 200s (processed)
    # - At least one 408 (queued too long and timed out)
    # - Remaining non-200s are either 429 (queue overflow) or additional 408s depending on scheduling
    assert any(code == 200 for code in results)
    assert any(code == 408 for code in results), results
    non200 = [c for c in results if c != 200]
    assert len(non200) >= 2 and all(c in (408, 429) for c in non200), results


@pytest.mark.integration
def test_rate_limit_429_carries_retry_after(router_manager, mock_workers):
    """A throttled request must come back with a Retry-After hint.

    OpenAI-compatible SDKs treat a 429 without the header as non-retryable and
    surface it to the caller instead of backing off, which turns a transient
    queue-full into a hard failure.
    """
    _, urls, _ = mock_workers(n=1, args=["--latency-ms", "100"])
    rh = router_manager.start_router(
        worker_urls=urls,
        policy="round_robin",
        extra={
            "max_concurrent_requests": 1,
            "queue_size": 0,  # no queue -> immediate 429 when the slot is taken
        },
    )

    def call_once(i):
        try:
            r = requests.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": f"r{i}",
                    "max_tokens": 1,
                    "stream": False,
                },
                timeout=3,
            )
            return r.status_code, r.headers.get("Retry-After")
        except Exception:
            return 599, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(call_once, range(8)))

    throttled = [retry_after for code, retry_after in results if code == 429]
    assert throttled, f"no request was throttled: {results}"
    assert all(v is not None and int(v) >= 1 for v in throttled), results
