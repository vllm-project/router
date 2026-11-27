import pytest

from ..helpers.client import embeddings


@pytest.mark.e2e
def test_embeddings_basic(router_manager, vllm_worker_manager, e2e_embedding_model):
    base = router_manager.start_router(policy="round_robin", timeout=120).url
    worker_url = vllm_worker_manager.start(e2e_embedding_model).url

    # Attach embedding worker to router-only instance
    router_manager.add_worker(base, worker_url, timeout=180)

    data = embeddings(
        base,
        e2e_embedding_model,
        ["the quick brown fox", "jumps over the lazy dog"],
        timeout=120,
    )
    assert "data" in data and isinstance(data["data"], list)
    assert len(data["data"]) == 2

    # Validate shape of embedding objects
    for item in data["data"]:
        assert "embedding" in item and isinstance(item["embedding"], list)
        # Ensure non-empty vectors
        assert len(item["embedding"]) > 0
