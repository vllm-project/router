import collections
import concurrent.futures
import threading
import time

import pytest
import requests

pytestmark = [pytest.mark.ready, pytest.mark.cpu]

# PD program scheduling must name an explicit per-rank capacity and turn off
# batch-gain admission; those are the settings the router accepts in vLLM PD mode.
_PROGRAM_SCHEDULING_CONFIG = (
    '{"pd":{"enabled":true},"token_capacity_per_dp_rank":100000,'
    '"enable_batch_gain_admission":false,"decode_buffer_tokens":8,'
    '"metrics_interval_seconds":0.02,"queue_timeout_seconds":5,'
    '"force_resume_timeout_seconds":1,'
    '"binding_strategy":"program_round_robin"}'
)


def _worker_id(response):
    return response.headers.get("X-Worker-Id") or response.json().get("worker_id")


def _start_pd(
    router_manager,
    mock_workers,
    policy,
    *,
    n_prefill=2,
    n_decode=3,
    prefill_args=None,
    decode_args=None,
    extra=None,
):
    _, prefill_urls, _ = mock_workers(n=n_prefill, args=prefill_args)
    _, decode_urls, decode_ids = mock_workers(n=n_decode, args=decode_args)
    router_extra = {"worker_startup_check_interval": 1}
    if extra:
        router_extra.update(extra)
    router = router_manager.start_router(
        policy=policy,
        vllm_pd_disaggregation=True,
        prefill_urls=[(url, None) for url in prefill_urls],
        decode_urls=list(decode_urls),
        extra=router_extra,
    )
    return router, set(decode_ids)


def _completion(router_url, prompt, extra=None, timeout=10):
    payload = {
        "model": "test-model",
        "prompt": prompt,
        "max_tokens": 1,
        "stream": False,
    }
    if extra:
        payload.update(extra)
    response = requests.post(
        f"{router_url}/v1/completions",
        json=payload,
        timeout=timeout,
    )
    assert response.status_code == 200, response.text
    return _worker_id(response)


def _wait_for_decode_load(worker_id, minimum=1, timeout=5.0):
    port = str(worker_id).removeprefix("worker-")
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            last = (
                requests.get(f"http://127.0.0.1:{port}/get_load", timeout=1)
                .json()
                .get("load")
            )
        except requests.RequestException:
            last = None
        if isinstance(last, int) and last >= minimum:
            return
        time.sleep(0.02)
    raise AssertionError(f"{worker_id} load did not reach {minimum}: {last}")


@pytest.mark.integration
def test_pd_power_of_two_decode_attribution(router_manager, mock_workers):
    # Start two prefill and three decode mock workers via fixture
    _, prefill_urls_raw, prefill_ids = mock_workers(n=2)
    _, decode_urls_raw, decode_ids_list = mock_workers(n=3)
    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)
    decode_ids = set(decode_ids_list)

    rh = router_manager.start_router(
        policy="power_of_two",
        vllm_pd_disaggregation=True,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    counts = collections.Counter()
    with requests.Session() as s:
        for i in range(30):
            r = s.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": f"p{i}",
                    "max_tokens": 1,
                    "stream": False,
                },
            )
            assert r.status_code == 200
            wid = r.headers.get("X-Worker-Id") or r.json().get("worker_id")
            assert wid in decode_ids
            counts[wid] += 1

    assert sum(1 for v in counts.values() if v > 0) >= 2


@pytest.mark.integration
def test_pd_power_of_two_skews_to_faster_decode(router_manager, mock_workers):
    # Start two prefill workers (fast)
    _, prefill_urls_raw, _ = mock_workers(n=2)

    # Start two decode workers: one very slow, one fast
    # Use 2000ms latency to create a clear load difference
    _, [decode_slow_url], [slow_id] = mock_workers(
        n=1, args=["--latency-ms", "2000"]
    )  # 2 second latency - very slow
    _, [decode_fast_url], [fast_id] = mock_workers(n=1)
    decode_urls_raw = [decode_slow_url, decode_fast_url]

    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)

    rh = router_manager.start_router(
        policy="power_of_two",
        vllm_pd_disaggregation=True,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    # Prime the router with some initial requests
    def _prime_call(i):
        try:
            requests.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": f"warm-{i}",
                    "max_tokens": 1,
                    "stream": False,
                },
                timeout=10,
            )
        except Exception:
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_prime_call, range(50)))
    time.sleep(1)

    # Create sustained load on slow worker by sending requests in background
    # These will take 2 seconds each, keeping the slow worker loaded
    stop_background_load = False

    def _background_load():
        i = 0
        while not stop_background_load:
            try:
                requests.post(
                    f"{decode_slow_url}/v1/completions",
                    json={
                        "model": "test-model",
                        "prompt": f"bg-{i}",
                        "max_tokens": 1,
                        "stream": False,
                    },
                    timeout=10,
                )
                i += 1
            except Exception:
                pass

    # Start background load in separate threads
    import threading

    bg_threads = [
        threading.Thread(target=_background_load, daemon=True) for _ in range(4)
    ]
    for t in bg_threads:
        t.start()

    # Wait for slow worker to accumulate load
    time.sleep(3)

    # Now send test requests - power-of-two should strongly prefer the fast worker
    def call(i):
        r = requests.post(
            f"{rh.url}/v1/completions",
            json={
                "model": "test-model",
                "prompt": f"p{i}",
                "max_tokens": 1,
                "stream": False,
            },
            timeout=10,
        )
        assert r.status_code == 200
        return r.headers.get("X-Worker-Id") or r.json().get("worker_id")

    counts = collections.Counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        for wid in ex.map(call, range(200)):
            counts[wid] += 1

    # Stop background load
    stop_background_load = True

    # With sustained load on slow worker, fast worker should get significantly more requests
    # Allow for some variance, but expect at least 60/40 split favoring fast worker
    assert counts[fast_id] > counts[slow_id], (
        f"Expected fast worker to handle more requests, got {counts}"
    )
    assert counts[fast_id] >= counts[slow_id] * 1.2, (
        f"Expected fast worker to handle at least 20% more requests than slow worker, got {counts}"
    )


@pytest.mark.integration
def test_pd_round_robin_spreads_decode(router_manager, mock_workers):
    router, decode_ids = _start_pd(router_manager, mock_workers, "round_robin")

    counts = collections.Counter()
    for i in range(30):
        worker_id = _completion(router.url, f"p{i}")
        assert worker_id in decode_ids
        counts[worker_id] += 1

    # 30 requests across 3 decode workers. Prefill and decode each have their
    # own round-robin counter, so every decode request advances that cursor once.
    for worker_id in decode_ids:
        assert 7 <= counts[worker_id] <= 13, counts


@pytest.mark.integration
def test_pd_random_spreads_decode(router_manager, mock_workers):
    router, decode_ids = _start_pd(router_manager, mock_workers, "random", n_decode=4)

    counts = collections.Counter()
    total = 200
    for i in range(total):
        worker_id = _completion(router.url, f"p{i}")
        assert worker_id in decode_ids
        counts[worker_id] += 1

    mean = total / len(decode_ids)
    for worker_id in decode_ids:
        assert 0.5 * mean <= counts[worker_id] <= 1.5 * mean, counts


@pytest.mark.integration
def test_pd_consistent_hash_sticks_to_session(router_manager, mock_workers):
    router, decode_ids = _start_pd(router_manager, mock_workers, "consistent_hash")

    def session_workers(session_id):
        return [
            _completion(
                router.url,
                f"turn {i}",
                extra={"session_params": {"session_id": session_id}},
            )
            for i in range(8)
        ]

    chosen = {}
    for session_id in ("alpha", "beta", "gamma", "delta"):
        workers = session_workers(session_id)
        assert set(workers) <= decode_ids
        assert len(set(workers)) == 1, workers
        chosen[session_id] = workers[0]

    assert len(chosen) == 4


@pytest.mark.integration
def test_pd_cache_aware_prefix_overrides_load(router_manager, mock_workers):
    # Decode latency keeps one admission in flight so a later request can see
    # unequal occupancy without crossing the cache-aware imbalance threshold.
    router, decode_ids = _start_pd(
        router_manager,
        mock_workers,
        "cache_aware",
        n_decode=2,
        decode_args=["--latency-ms", "800"],
    )
    cached_prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    busy_prompt = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    other_prompt = [200, 201, 202, 203, 204, 205, 206, 207, 208, 209]

    cached = _completion(router.url, cached_prompt, timeout=10)
    assert cached in decode_ids

    holder = threading.Thread(
        target=_completion,
        args=(router.url, busy_prompt),
        kwargs={"timeout": 10},
        daemon=True,
    )
    holder.start()
    _wait_for_decode_load(cached)
    diverted = _completion(router.url, other_prompt, timeout=10)
    replay = _completion(router.url, cached_prompt, timeout=10)
    holder.join(timeout=10)

    assert diverted in decode_ids and replay in decode_ids
    assert diverted != cached, (cached, diverted, replay)
    assert replay == cached, (cached, diverted, replay)


@pytest.mark.integration
def test_pd_kv_aware_spreads_equal_decode_load(router_manager, mock_workers):
    router, decode_ids = _start_pd(router_manager, mock_workers, "kv_aware")

    counts = collections.Counter()
    for i in range(30):
        worker_id = _completion(router.url, f"p{i}")
        assert worker_id in decode_ids
        counts[worker_id] += 1

    # No session binding: equal occupancy rotates across decode workers.
    for worker_id in decode_ids:
        assert 7 <= counts[worker_id] <= 13, counts


@pytest.mark.integration
def test_pd_kv_aware_affinity_sticks_to_session(router_manager, mock_workers):
    router, decode_ids = _start_pd(
        router_manager,
        mock_workers,
        "kv_aware",
        n_decode=2,
        extra={"kv_aware_session_affinity_ttl_secs": 60},
    )

    def session_workers(session_id):
        return [
            _completion(
                router.url,
                f"turn {i}",
                extra={"session_params": {"session_id": session_id}},
            )
            for i in range(8)
        ]

    alpha = session_workers("alpha")
    beta = session_workers("beta")
    assert set(alpha) <= decode_ids and set(beta) <= decode_ids
    assert len(set(alpha)) == 1, alpha
    assert len(set(beta)) == 1, beta
    assert alpha[0] != beta[0], (alpha, beta)


@pytest.mark.integration
def test_pd_program_scheduling_sticks_to_program(router_manager, mock_workers):
    router, decode_ids = _start_pd(
        router_manager,
        mock_workers,
        "round_robin",
        n_decode=2,
        extra={
            "enable_program_scheduling": True,
            "program_scheduling_config_json": _PROGRAM_SCHEDULING_CONFIG,
        },
    )

    def program_workers(program_id):
        return [
            _completion(
                router.url,
                "hello",
                extra={
                    "max_tokens": 8,
                    "vllm_xargs": {
                        "agentic_context": {
                            "program_id": program_id,
                            "expected_resume": True,
                        }
                    },
                },
                timeout=15,
            )
            for _ in range(4)
        ]

    alpha = program_workers("program-alpha")
    beta = program_workers("program-beta")
    assert set(alpha) <= decode_ids and set(beta) <= decode_ids
    assert len(set(alpha)) == 1, alpha
    assert len(set(beta)) == 1, beta
    # program_round_robin assigns each new program to the next sorted target.
    assert alpha[0] != beta[0], (alpha, beta)
