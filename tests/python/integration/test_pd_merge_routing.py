"""PD merge routing: policy selection on colocated workers.

Each worker does both prefill and decode. The router stays in regular mode
(``--worker-urls``, no ``--vllm-pd-disaggregation``). Companion to
``test_pd_disagg_routing.py``, which covers the same policies on separate
prefill and decode pools.
"""

import collections
import threading
import time

import pytest

from ..helpers.client import completion

pytestmark = [pytest.mark.ready, pytest.mark.cpu]

# Program scheduling must name an explicit per-rank capacity and turn off
# batch-gain admission; those are the settings the router accepts.
_PROGRAM_SCHEDULING_CONFIG = (
    '{"token_capacity_per_dp_rank":100000,'
    '"enable_batch_gain_admission":false,"decode_buffer_tokens":8,'
    '"metrics_interval_seconds":0.02,"queue_timeout_seconds":5,'
    '"force_resume_timeout_seconds":1,'
    '"binding_strategy":"program_round_robin"}'
)


def _router(router_manager, urls, policy, extra=None):
    router_extra = {"worker_startup_check_interval": 1}
    if extra:
        router_extra.update(extra)
    return router_manager.start_router(
        worker_urls=urls, policy=policy, extra=router_extra
    )


@pytest.mark.integration
def test_merge_power_of_two_spreads(router_manager, mock_worker_manager):
    _, urls, ids = mock_worker_manager.start_workers(n=3)
    rh = _router(router_manager, urls, "power_of_two")

    counts = collections.Counter()
    for i in range(30):
        worker_id = completion(rh.url, f"p{i}")
        assert worker_id in ids
        counts[worker_id] += 1

    assert sum(1 for value in counts.values() if value > 0) >= 2


@pytest.mark.integration
def test_merge_power_of_two_skews_to_faster_worker(router_manager, mock_worker_manager):
    _, [slow_url], [slow_id] = mock_worker_manager.start_workers(
        n=1, args=["--latency-ms", "2000"]
    )
    _, [fast_url], [fast_id] = mock_worker_manager.start_workers(n=1)
    rh = _router(router_manager, [slow_url, fast_url], "power_of_two")

    completion(
        rh.url,
        prompts=(f"warm-{i}" for i in range(50)),
        max_workers=16,
        timeout=10,
        ignore_errors=True,
    )
    time.sleep(1)

    stop_background_load = False

    def _background_load():
        i = 0
        while not stop_background_load:
            completion(slow_url, f"bg-{i}", timeout=10, ignore_errors=True)
            i += 1

    bg_threads = [
        threading.Thread(target=_background_load, daemon=True) for _ in range(4)
    ]
    for thread in bg_threads:
        thread.start()
    time.sleep(3)

    counts = collections.Counter(
        completion(
            rh.url,
            prompts=(f"p{i}" for i in range(200)),
            max_workers=32,
            timeout=10,
        )
    )
    stop_background_load = True

    assert counts[fast_id] > counts[slow_id], counts
    assert counts[fast_id] >= counts[slow_id] * 1.2, counts


@pytest.mark.integration
def test_merge_round_robin_balances(router_manager, mock_worker_manager):
    # Regular workers are visited in registry order, which is not the
    # --worker-urls registration order. Equal counts are the contract.
    _, urls, ids = mock_worker_manager.start_workers(n=3)
    rh = _router(router_manager, urls, "round_robin")

    counts = collections.Counter(completion(rh.url, f"p{i}") for i in range(30))
    assert set(counts) == set(ids)
    for worker_id in ids:
        assert counts[worker_id] == 10, counts


@pytest.mark.integration
def test_merge_random_spreads(router_manager, mock_worker_manager):
    _, urls, ids = mock_worker_manager.start_workers(n=4)
    rh = _router(router_manager, urls, "random")

    counts = collections.Counter()
    total = 200
    for i in range(total):
        worker_id = completion(rh.url, f"p{i}")
        assert worker_id in ids
        counts[worker_id] += 1

    mean = total / len(ids)
    for worker_id in ids:
        assert 0.5 * mean <= counts[worker_id] <= 1.5 * mean, counts


@pytest.mark.integration
def test_merge_consistent_hash_sticks_to_session(router_manager, mock_worker_manager):
    _, urls, ids = mock_worker_manager.start_workers(n=3)
    rh = _router(router_manager, urls, "consistent_hash")
    worker_ids = set(ids)

    # Regular routing hashes the prompt text unless a session header is present,
    # so the session id travels in X-Session-ID, the documented primary key.
    def session_worker(session_id, turn):
        return completion(
            rh.url,
            f"turn {turn}",
            headers={"X-Session-ID": session_id},
        )

    placed = {}
    for i in range(64):
        session_id = f"session-{i}"
        worker_id = session_worker(session_id, 0)
        assert worker_id in worker_ids
        placed[session_id] = worker_id
        if set(placed.values()) == worker_ids:
            break
    else:
        raise AssertionError(placed)

    by_worker = {}
    for session_id, worker_id in placed.items():
        by_worker.setdefault(worker_id, session_id)
    assert set(by_worker) == worker_ids

    for worker_id, session_id in by_worker.items():
        workers = [session_worker(session_id, turn) for turn in range(8)]
        assert set(workers) == {worker_id}, workers


@pytest.mark.integration
def test_merge_cache_aware_prefix_overrides_load(
    router_manager, mock_worker_manager, tmp_path
):
    release = tmp_path / "release"
    _, urls, ids = mock_worker_manager.start_workers(
        n=2, args=["--hold-after", "1", "--release-file", str(release)]
    )
    rh = _router(router_manager, urls, "cache_aware")
    worker_ids = set(ids)
    cached_prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    busy_prompt = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    other_prompt = [200, 201, 202, 203, 204, 205, 206, 207, 208, 209]

    cached = completion(rh.url, cached_prompt, timeout=10)
    assert cached in worker_ids

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

    assert diverted in worker_ids and replay_box["worker"] in worker_ids
    assert diverted != cached, (cached, diverted, replay_box)
    assert replay_box["worker"] == cached, (cached, diverted, replay_box)


@pytest.mark.integration
def test_merge_kv_aware_spreads_concurrent_load(router_manager, mock_worker_manager):
    # Without KV cache data every worker costs the same, so kv_aware picks the
    # one with the fewest in-flight requests and breaks ties by position.
    # Sequential requests always see zero in flight. Overlapping ones have to
    # spread, so each worker is kept busy long enough to overlap.
    _, urls, ids = mock_worker_manager.start_workers(n=3, args=["--latency-ms", "300"])
    rh = _router(router_manager, urls, "kv_aware")

    served = completion(
        rh.url,
        prompts=(f"p{i}" for i in range(30)),
        max_workers=3,
        timeout=10,
    )
    counts = collections.Counter(served)

    assert set(counts) == set(ids), counts
    for worker_id in ids:
        assert 6 <= counts[worker_id] <= 14, counts


@pytest.mark.integration
def test_merge_kv_aware_affinity_sticks_to_session(router_manager, mock_worker_manager):
    # Idle workers tie, and a tie goes to the first worker. Two sessions only
    # land apart when the second starts while the first is still in flight.
    # After that, a session that is not on the first worker can only stay put
    # through its affinity binding.
    _, urls, ids = mock_worker_manager.start_workers(n=2, args=["--latency-ms", "500"])
    rh = _router(
        router_manager,
        urls,
        "kv_aware",
        extra={"kv_aware_session_affinity_ttl_secs": 60},
    )
    worker_ids = set(ids)

    def session_worker(session_id, turn):
        return completion(
            rh.url,
            f"turn {turn}",
            extra={"session_params": {"session_id": session_id}},
            timeout=10,
        )

    first = {}
    alpha_start = threading.Thread(
        target=lambda: first.update(alpha=session_worker("alpha", 0)), daemon=True
    )
    alpha_start.start()
    time.sleep(0.2)
    first["beta"] = session_worker("beta", 0)
    alpha_start.join(timeout=10)

    assert {first["alpha"], first["beta"]} == worker_ids, first

    alpha = [first["alpha"]] + [session_worker("alpha", turn) for turn in range(1, 5)]
    beta = [first["beta"]] + [session_worker("beta", turn) for turn in range(1, 5)]
    assert set(alpha) == {first["alpha"]}, alpha
    assert set(beta) == {first["beta"]}, beta


@pytest.mark.integration
def test_merge_program_scheduling_sticks_to_program(
    router_manager, mock_worker_manager
):
    _, urls, ids = mock_worker_manager.start_workers(n=2)
    rh = _router(
        router_manager,
        urls,
        "round_robin",
        extra={
            "enable_program_scheduling": True,
            "program_scheduling_config_json": _PROGRAM_SCHEDULING_CONFIG,
        },
    )
    worker_ids = set(ids)

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
    assert set(alpha) <= worker_ids and set(beta) <= worker_ids
    assert len(set(alpha)) == 1, alpha
    assert len(set(beta)) == 1, beta
    assert alpha[0] != beta[0], (alpha, beta)
