"""CPU acceptance for hybrid cache-holder routing through real gRPC and ZMQ.

The JSON profiles scale cache blocks down to 16 tokens and hashes to 4 tokens.
They describe cache contracts, not model weights or inference. Expected credits
are fixed in the profiles independently of the Router's implementation.
"""

from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path

import pytest

from tests.python.helpers.mock_worker import kv_store
from tests.python.integration.load_balancing.test_kv_aware import KvScenario

pytestmark = [pytest.mark.ready, pytest.mark.cpu, pytest.mark.integration]

_SPEC_TYPES = {
    "mamba": "MambaSpec",
    "full_attention": "FullAttentionSpec",
    "mla_attention": "MLAAttentionSpec",
    "sliding_window": "SlidingWindowSpec",
    "sliding_window_mla": "SlidingWindowMLASpec",
    "scratch": "CircularBufferSpec",
}
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_DECISION = re.compile(
    r'KV prefix decision\b[^\n]*?\bworker="?(?P<worker>[^"\s]+)"?'
    r"\s+input_tokens=(?P<input>\d+)\s+new_prefill_tokens=\d+"
    r"\s+credited_prefix_tokens=(?P<credit>\d+)\b"
)


def _profiles():
    directory = Path(__file__).with_name("cases") / "kv_aware_hybrid"
    parameters = []
    for path in sorted(directory.glob("*.json")):
        profile = json.loads(path.read_text(encoding="utf-8"))
        parameters.append(pytest.param(profile, id=path.stem))
        if "mtp" in profile:
            variant = copy.deepcopy(profile)
            variant["rank"].update(variant["mtp"]["rank"])
            variant["expected"].update(variant["mtp"].get("expected", {}))
            parameters.append(pytest.param(variant, id=f"{path.stem}-mtp"))
    if not parameters:
        raise ValueError(f"No hybrid mock profiles found in {directory}")
    return parameters


def _rank(profile, *, small_blocks=False):
    rank = copy.deepcopy(profile["rank"])
    for index, group in enumerate(rank["groups"]):
        group.update(
            group_idx=index,
            spec_type=_SPEC_TYPES[group["kind"]],
            prefix_cacheable=group["kind"] != "scratch",
            prefix_replay_tokens=0,
        )
        if small_blocks:
            group["block_size"] = rank["hash_block_size"]
    return rank


def _stores(worker, tokens, cached, *, start=0, partial=False):
    """Publish group-sized hashes selected from one native fine-grained chain.

    Sliding groups retain their trailing window; scratch publishes nothing.
    Partial stores are separate hash-sized events, as in vLLM's event stream.
    """
    rank = worker.cache_layout
    unit = rank["hash_block_size"]
    prototype = kv_store(tokens[:cached], unit)
    hashes = prototype["block_hashes"]
    events = []

    def event(group, begin, end, block):
        value = {
            **prototype,
            "group_idx": group["group_idx"],
            "kv_cache_spec_kind": group["kind"],
            "block_size": block,
            "token_ids": tokens[begin:end],
            "block_hashes": [
                hashes[offset // unit - 1]
                for offset in range(begin + block, end + 1, block)
            ],
            "parent_block_hash": hashes[begin // unit - 1] if begin else None,
        }
        if "sliding_window" in group:
            value["kv_cache_spec_sliding_window"] = group["sliding_window"]
        return value

    for group in rank["groups"]:
        kind = group["kind"]
        if kind == "scratch":
            continue
        block = group["block_size"]
        end = cached // block * block
        begin = start
        if "sliding_window" in group:
            window_blocks = (group["sliding_window"] + block - 1) // block
            begin = max(begin, end - window_blocks * block)
        if begin < end:
            events.append(event(group, begin, end, block))
        if "sliding_window" not in group and (rank["partial_hash_hits"] or partial):
            for boundary in range(start + unit, cached + 1, unit):
                if boundary % block:
                    events.append(event(group, boundary - unit, boundary, unit))
    return events


class HybridScenario(KvScenario):
    def __init__(self, workers, url, profile, log_path):
        super().__init__(workers, url)
        self.profile = profile
        self.log_path = log_path

    def tokens(self, text):
        worker = self.choose(text)
        request = worker.captured[-1]
        assert request.WhichOneof("prompt") == "token_ids"
        tokens = list(request.token_ids.ids)
        assert len(tokens) == len(text.split()), tokens
        return tokens

    def decision(self, text):
        since = self.log_path.stat().st_size
        worker = self.choose(text)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with self.log_path.open("rb") as log:
                log.seek(since)
                output = _ANSI.sub("", log.read().decode("utf-8", errors="replace"))
            matches = list(_DECISION.finditer(output))
            if matches:
                assert len(matches) == 1, output
                match = matches[0]
                assert match["worker"].strip('"') == worker.url, output
                assert int(match["input"]) == len(text.split()), output
                return worker, int(match["credit"])
            time.sleep(0.01)
        pytest.fail(f"Missing KV decision in {self.log_path}: {output}")

    def expect(self, text, worker, credit):
        actual_worker, actual_credit = self.decision(text)
        assert (actual_worker.worker_id, actual_credit) == (worker.worker_id, credit)

    def publish_applied(self, worker, events):
        # A new sentinel in the same atomic batch proves both positive and
        # negative cache changes have been applied before assertions run.
        marker = " ".join(f"t{i}" for i in range(self.marker, self.marker + 33))
        self.marker += 33
        tokens = self.tokens(marker)
        worker.publish(events + _stores(worker, tokens, 32))
        deadline = time.monotonic() + 10
        expected = (worker.worker_id, self.profile["expected"]["whole"])
        seen = []
        while time.monotonic() < deadline:
            actual, credit = self.decision(marker)
            seen.append((actual.worker_id, credit))
            if seen[-1] == expected:
                return
            time.sleep(0.02)
        pytest.fail(f"KV batch not applied; expected {expected}, observed {seen[-10:]}")


@pytest.fixture(params=_profiles())
def hybrid_scenario(
    request, kv_workers, kv_st_runtime, kv_tokenizer, router_manager, tmp_path
):
    profile = request.param
    workers = [
        kv_workers(block_size=4, cache_layout=_rank(profile, small_blocks=small))
        for small in (False, True)
    ]
    log_path = tmp_path / "router.log"
    router = router_manager.start_router(
        worker_urls=[worker.url for worker in workers],
        policy="kv_aware",
        router_bin=str(kv_st_runtime[1]),
        extra={
            "router_temperature": 0,
            "disable_retries": True,
            "disable_circuit_breaker": True,
            "log_level": "debug",
        },
        env={
            "VLLM_ROUTER_TOKENIZER": str(kv_tokenizer),
            "NO_PROXY": "127.0.0.1,localhost",
        },
        log_path=log_path,
    )
    for worker in workers:
        worker.wait_subscribed()
    scenario = HybridScenario(workers, router.url, profile, log_path)
    try:
        yield scenario
    finally:
        scenario.session.close()
        scenario.stream_session.close()


def _prompt(start, tail=511):
    return " ".join([*(f"t{i}" for i in range(start, start + 32)), f"t{tail}"])


def test_hybrid_cache_holders_follow_prefixes(hybrid_scenario):
    st = hybrid_scenario
    a, b = st.workers
    for worker, start in ((a, 0), (b, 32)):
        tokens = st.tokens(_prompt(start, tail=510))
        st.publish_applied(worker, _stores(worker, tokens, 32))
    for _ in range(5):
        st.expect(_prompt(32), b, st.profile["expected"]["whole"])
        st.expect(_prompt(0), a, st.profile["expected"]["whole"])


def test_hybrid_partial_prefix_boundaries(hybrid_scenario):
    st = hybrid_scenario
    a, b = st.workers
    text = _prompt(0)
    tokens = st.tokens(_prompt(0, tail=510))
    st.publish_applied(b, _stores(b, tokens, 20))
    st.publish_applied(a, _stores(a, tokens, 16))
    st.expect(text, b, st.profile["expected"]["rival"])

    # A has one whole 16-token block plus half of its next block. B has
    # five whole 4-token blocks. Only a valid partial hit makes A preferable.
    st.publish_applied(a, _stores(a, tokens, 24, start=16, partial=True))
    expected = st.profile["expected"]
    winner = a if expected["partial_worker"] == "a" else b
    credit = expected["partial"] if winner is a else expected["rival"]
    for _ in range(5):
        st.expect(text, winner, credit)
    st.publish_applied(b, [{"type": "AllBlocksCleared"}])
    st.expect(text, a, expected["partial"])


@pytest.mark.parametrize("invalidation", ["remove", "clear"])
def test_hybrid_required_group_and_invalidation(hybrid_scenario, invalidation):
    st = hybrid_scenario
    a, b = st.workers
    text = _prompt(0)
    tokens = st.tokens(_prompt(0, tail=510))
    stores = _stores(a, tokens, 32)
    required = [event for event in stores if event["group_idx"] == 0]
    st.publish_applied(b, _stores(b, tokens, 16))
    st.publish_applied(a, [event for event in stores if event["group_idx"] != 0])
    st.expect(text, b, st.profile["expected"]["short"])

    st.publish_applied(a, required)
    st.expect(text, a, st.profile["expected"]["whole"])
    if invalidation == "clear":
        removal = {"type": "AllBlocksCleared"}
    else:
        removal = {
            "type": "BlockRemoved",
            "group_idx": 0,
            "medium": "GPU",
            "block_hashes": [
                key for event in required for key in event["block_hashes"]
            ],
        }
    st.publish_applied(a, [removal])
    for _ in range(5):
        st.expect(text, b, st.profile["expected"]["short"])
