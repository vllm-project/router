"""kv_aware pipeline smoke with mock workers (no GPU / no vLLM).

Run without pulling in py_test/e2e/conftest (which requires vllm):

  pytest py_test/e2e/function/test_kv_aware_mock.py \\
    --confcutdir=py_test/e2e/function -m integration -v
"""

import pytest
import requests


@pytest.mark.integration
def test_kv_aware_mock_worker_request_pipeline(mock_workers, router_manager):
    """
    End-to-end through the router process: mock workers + kv_aware policy + one completion.

    KV discovery against mock workers has no /kv_event_sources or block_size in
    /get_server_info, so the policy degrades to load-only scoring; this still
    validates registration, routing, and a successful HTTP completion path.
    """
    _, urls, worker_ids = mock_workers(n=2)
    rh = router_manager.start_router(worker_urls=urls, policy="kv_aware")

    r = requests.post(
        f"{rh.url}/v1/completions",
        json={
            "model": "test-model",
            "prompt": "kv_aware mock smoke",
            "max_tokens": 1,
            "stream": False,
        },
        timeout=30,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("choices")
    wid = r.headers.get("X-Worker-Id") or body.get("worker_id")
    assert wid in worker_ids
