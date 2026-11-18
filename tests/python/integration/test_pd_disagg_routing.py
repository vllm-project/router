"""PD disaggregation routing: policy selection across prefill and decode workers.

Companion to ``test_pd_merge_routing.py``, which covers the same policies when
each worker does both prefill and decode.
"""

import collections
import threading
import time

import pytest

from ..helpers.client import completion

pytestmark = [pytest.mark.ready, pytest.mark.cpu]

# PD program scheduling must name an explicit per-rank capacity and turn off
# batch-gain admission; those are the settings the router accepts in vLLM PD mode.
_PROGRAM_SCHEDULING_CONFIG = (
    '{"pd":{},"token_capacity_per_dp_rank":100000,'
    '"enable_batch_gain_admission":false,"decode_buffer_tokens":8,'
    '"metrics_interval_seconds":0.02,"queue_timeout_seconds":5,'
    '"force_resume_timeout_seconds":1,'
    '"binding_strategy":"program_round_robin"}'
)


@pytest.mark.integration
def test_pd_power_of_two_decode_attribution(router_manager, mock_worker_manager):
    mock_worker_manager.add_prefill(n=2)
    mock_worker_manager.add_decode(n=3)
    rh = router_manager.start_router(
        policy="power_of_two",
        vllm_pd_disaggregation=True,
        prefill_urls=mock_worker_manager.prefill_urls,
        decode_urls=mock_worker_manager.decode_urls,
        extra={"worker_startup_check_interval": 1},
    )
    decode_ids = mock_worker_manager.decode_ids

    counts = collections.Counter()
    for i in range(30):
        wid = completion(rh.url, f"p{i}")
        assert wid in decode_ids
        counts[wid] += 1

    assert sum(1 for v in counts.values() if v > 0) >= 2


@pytest.mark.integration
def test_pd_power_of_two_skews_to_faster_decode(router_manager, mock_worker_manager):
    mock_worker_manager.add_prefill(n=2)
    # 2 second latency keeps the slow decode worker occupied.
    [decode_slow_url], [slow_id] = mock_worker_manager.add_decode(
        n=1, args=["--latency-ms", "2000"]
    )
    _, [fast_id] = mock_worker_manager.add_decode(n=1)
    rh = router_manager.start_router(
        policy="power_of_two",
        vllm_pd_disaggregation=True,
        prefill_urls=mock_worker_manager.prefill_urls,
        decode_urls=mock_worker_manager.decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    # Prime the router with some initial requests.
    completion(
        rh.url,
        prompts=(f"warm-{i}" for i in range(50)),
        max_workers=16,
        timeout=10,
        ignore_errors=True,
    )
    time.sleep(1)

    # Create sustained load on slow worker by sending requests in background
    # These will take 2 seconds each, keeping the slow worker loaded
    stop_background_load = False

    def _background_load():
        i = 0
        while not stop_background_load:
            completion(decode_slow_url, f"bg-{i}", timeout=10, ignore_errors=True)
            i += 1

    bg_threads = [
        threading.Thread(target=_background_load, daemon=True) for _ in range(4)
    ]
    for t in bg_threads:
        t.start()

    # Wait for slow worker to accumulate load
    time.sleep(3)

    # Now send test requests - power-of-two should strongly prefer the fast worker
    counts = collections.Counter(
        completion(
            rh.url,
            prompts=(f"p{i}" for i in range(200)),
            max_workers=32,
            timeout=10,
        )
    )

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
def test_pd_round_robin_cycles_decode_in_registration_order(
    router_manager, mock_worker_manager
):
    # Prefill and decode have separate cursors. The decode cursor starts at 0
    # and advances once per request, in --decode registration order.
    mock_worker_manager.add_prefill(n=2)
    mock_worker_manager.add_decode(n=3)
    rh = router_manager.start_router(
        policy="round_robin",
        vllm_pd_disaggregation=True,
        prefill_urls=mock_worker_manager.prefill_urls,
        decode_urls=mock_worker_manager.decode_urls,
        extra={"worker_startup_check_interval": 1},
    )
    decode_ids = set(mock_worker_manager.decode_ids)
    order = [
        f"worker-{url.rstrip('/').rsplit(':', 1)[-1]}"
        for url in mock_worker_manager.decode_urls
    ]
    assert set(order) == decode_ids

    seen = [completion(rh.url, f"p{i}") for i in range(len(order) * 10)]

    # Ten exact cycles also means each decode worker receives the same count.
    assert seen == order * 10


@pytest.mark.integration
def test_pd_random_spreads_decode(router_manager, mock_worker_manager):
    mock_worker_manager.add_prefill(n=2)
    mock_worker_manager.add_decode(n=4)
    rh = router_manager.start_router(
        policy="random",
        vllm_pd_disaggregation=True,
        prefill_urls=mock_worker_manager.prefill_urls,
        decode_urls=mock_worker_manager.decode_urls,
        extra={"worker_startup_check_interval": 1},
    )
    decode_ids = set(mock_worker_manager.decode_ids)

    counts = collections.Counter()
    total = 200
    for i in range(total):
        worker_id = completion(rh.url, f"p{i}")
        assert worker_id in decode_ids
        counts[worker_id] += 1

    mean = total / len(decode_ids)
    for worker_id in decode_ids:
        assert 0.5 * mean <= counts[worker_id] <= 1.5 * mean, counts


@pytest.mark.integration
def test_pd_consistent_hash_sticks_to_session(router_manager, mock_worker_manager):
    mock_worker_manager.add_prefill(n=2)
    mock_worker_manager.add_decode(n=3)
    rh = router_manager.start_router(
        policy="consistent_hash",
        vllm_pd_disaggregation=True,
        prefill_urls=mock_worker_manager.prefill_urls,
        decode_urls=mock_worker_manager.decode_urls,
        extra={"worker_startup_check_interval": 1},
    )
    decode_ids = set(mock_worker_manager.decode_ids)

    def session_worker(session_id, turn):
        return completion(
            rh.url,
            f"turn {turn}",
            extra={"session_params": {"session_id": session_id}},
        )

    # The ring is built from worker URLs, so fixed names are not guaranteed to
    # land apart. Probe distinct session ids until every decode worker is used.
    placed = {}
    for i in range(64):
        session_id = f"session-{i}"
        worker_id = session_worker(session_id, 0)
        assert worker_id in decode_ids
        placed[session_id] = worker_id
        if set(placed.values()) == decode_ids:
            break
    else:
        raise AssertionError(placed)

    by_worker = {}
    for session_id, worker_id in placed.items():
        by_worker.setdefault(worker_id, session_id)
    assert set(by_worker) == decode_ids

    for worker_id, session_id in by_worker.items():
        workers = [session_worker(session_id, turn) for turn in range(8)]
        assert set(workers) == {worker_id}, workers


@pytest.mark.integration
def test_pd_cache_aware_prefix_overrides_load(
    router_manager, mock_worker_manager, tmp_path
):
    # The first request on each decode returns. Later requests stay in flight
    # until the release file appears, so the replay is admitted while the
    # cached decode is still busy. Least-loaded routing would miss it.
    release = tmp_path / "release"
    mock_worker_manager.add_prefill(n=2)
    mock_worker_manager.add_decode(
        n=2, args=["--hold-after", "1", "--release-file", str(release)]
    )
    rh = router_manager.start_router(
        policy="cache_aware",
        vllm_pd_disaggregation=True,
        prefill_urls=mock_worker_manager.prefill_urls,
        decode_urls=mock_worker_manager.decode_urls,
        extra={"worker_startup_check_interval": 1},
    )
    decode_ids = set(mock_worker_manager.decode_ids)
    cached_prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    busy_prompt = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    other_prompt = [200, 201, 202, 203, 204, 205, 206, 207, 208, 209]

    cached = completion(rh.url, cached_prompt, timeout=10)
    assert cached in decode_ids

    holder = threading.Thread(
        target=completion,
        args=(rh.url, busy_prompt),
        kwargs={"timeout": 10},
        daemon=True,
    )
    holder.start()
    mock_worker_manager.wait_for_decode_load(cached)
    diverted = completion(rh.url, other_prompt, timeout=10)

    replay_box = {}

    def replay():
        replay_box["worker"] = completion(rh.url, cached_prompt, timeout=10)

    replay_thread = threading.Thread(target=replay, daemon=True)
    replay_thread.start()
    try:
        mock_worker_manager.wait_for_decode_load(cached, minimum=2)
    finally:
        release.write_text("go")
    holder.join(timeout=10)
    replay_thread.join(timeout=10)

    assert diverted in decode_ids and replay_box["worker"] in decode_ids
    assert diverted != cached, (cached, diverted, replay_box)
    assert replay_box["worker"] == cached, (cached, diverted, replay_box)


@pytest.mark.integration
def test_pd_kv_aware_spreads_equal_decode_load(router_manager, mock_worker_manager):
    mock_worker_manager.add_prefill(n=2)
    mock_worker_manager.add_decode(n=3)
    rh = router_manager.start_router(
        policy="kv_aware",
        vllm_pd_disaggregation=True,
        prefill_urls=mock_worker_manager.prefill_urls,
        decode_urls=mock_worker_manager.decode_urls,
        extra={"worker_startup_check_interval": 1},
    )
    decode_ids = set(mock_worker_manager.decode_ids)

    counts = collections.Counter()
    for i in range(30):
        worker_id = completion(rh.url, f"p{i}")
        assert worker_id in decode_ids
        counts[worker_id] += 1

    # No session binding: equal occupancy rotates across decode workers.
    for worker_id in decode_ids:
        assert 7 <= counts[worker_id] <= 13, counts


@pytest.mark.integration
def test_pd_kv_aware_affinity_sticks_to_session(router_manager, mock_worker_manager):
    mock_worker_manager.add_prefill(n=2)
    mock_worker_manager.add_decode(n=2)
    rh = router_manager.start_router(
        policy="kv_aware",
        vllm_pd_disaggregation=True,
        prefill_urls=mock_worker_manager.prefill_urls,
        decode_urls=mock_worker_manager.decode_urls,
        extra={
            "worker_startup_check_interval": 1,
            "kv_aware_session_affinity_ttl_secs": 60,
        },
    )
    decode_ids = set(mock_worker_manager.decode_ids)

    def session_workers(session_id):
        return [
            completion(
                rh.url,
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
def test_pd_program_scheduling_sticks_to_program(router_manager, mock_worker_manager):
    mock_worker_manager.add_prefill(n=2)
    mock_worker_manager.add_decode(n=2)
    rh = router_manager.start_router(
        policy="round_robin",
        vllm_pd_disaggregation=True,
        prefill_urls=mock_worker_manager.prefill_urls,
        decode_urls=mock_worker_manager.decode_urls,
        extra={
            "worker_startup_check_interval": 1,
            "enable_program_scheduling": True,
            "program_scheduling_config_json": _PROGRAM_SCHEDULING_CONFIG,
        },
    )
    decode_ids = set(mock_worker_manager.decode_ids)

    def program_workers(program_id):
        return [
            completion(
                rh.url,
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
