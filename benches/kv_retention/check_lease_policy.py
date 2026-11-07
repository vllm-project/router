#!/usr/bin/env python3
"""A randomized check of ``LeasePolicy`` against vLLM's own LRU. Needs vLLM, and no GPU.

A refused eviction makes vLLM drop a store silently, so a lease must not refuse one that LRU would grant. And it must
evict a leased chunk only when nothing of a lower rank can go. Each eviction asks a fresh ``LRUCachePolicy`` holding the
same evictable chunks whether it would succeed.

    python check_lease_policy.py
"""

from __future__ import annotations

import random
from collections import Counter

from kv_retention_lifecycle import HINT_KEY, LeaseBook, hint_body, parse_hint
from kv_retention_policy import LeasePolicy
from vllm.v1.kv_offload.base import ReqContext
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

CAPACITY = 64
STEPS = 40_000
SESSIONS = 8


def lru_would_evict(evictable: set, n: int, protected: set) -> bool:
    lru = LRUCachePolicy(CAPACITY)
    for key in evictable:
        chunk = ChunkStatus(0)
        chunk.ref_cnt = 0
        lru.insert(key, chunk)
    return lru.evict(n, protected) is not None


def request(rng, policy, generations, tag):
    """A request of a random session, under its lease or none. A session keeps one priority: this prototype re-ranks
    only a lease's new chunks when a later hint changes it."""
    session = rng.randrange(SESSIONS)
    generations[session] += 1
    params = None
    if rng.random() < 0.7:
        body = hint_body(f"s{session}", generations[session], ["cpu"], 30.0, tenant="t")
        body["priority"] = session % LeaseBook.RANKS
        params = {HINT_KEY: body}
        policy.book.open(parse_hint(params)[0])
    return ReqContext(req_id=tag, kv_transfer_params=params)


def main() -> None:
    rng = random.Random(427)
    now = [0.0]
    policy = LeasePolicy(CAPACITY)
    policy.book = LeaseBook(CAPACITY, 0.5, 0.5, 30.0, "check", clock=lambda: now[0])
    chunks: dict[int, ChunkStatus] = {}
    evictable: set[int] = set()
    pinned: set[int] = set()
    generations: Counter[int] = Counter()
    next_key = 0
    seen: Counter[str] = Counter()
    for _ in range(STEPS):
        now[0] += rng.random()
        op = rng.random()
        if op < 0.3 and len(chunks) < CAPACITY:
            # A request stores new chunks and finishes.
            context = request(rng, policy, generations, f"r{next_key}")
            keys = list(range(next_key, next_key + rng.randint(1, 4)))
            keys = keys[: CAPACITY - len(chunks)]
            next_key += len(keys)
            policy.on_store_miss(keys, context)
            for key in keys:
                chunks[key] = ChunkStatus(key)
                policy.insert(key, chunks[key])
                chunks[key].ref_cnt = 0
                policy.mark_evictable(key)
                evictable.add(key)
            policy.on_request_finished([keys], set(keys), set(), context)
        elif op < 0.4 and chunks:
            # Another session's request reuses chunks that are there, so a chunk can have two leases.
            context = request(rng, policy, generations, f"reuse{next_key}")
            keys = rng.sample(sorted(chunks), min(len(chunks), 3))
            policy.on_request_finished([keys], set(), set(keys), context)
        elif op < 0.5 and evictable:
            key = rng.choice(sorted(evictable))  # a load pins it
            chunks[key].ref_cnt = 1
            policy.mark_non_evictable(key)
            evictable.discard(key)
            pinned.add(key)
        elif op < 0.6 and pinned:
            key = rng.choice(sorted(pinned))
            chunks[key].ref_cnt = 0
            policy.mark_evictable(key)
            pinned.discard(key)
            evictable.add(key)
        else:
            n = rng.randint(1, 12)
            protected = set(rng.sample(sorted(chunks), min(len(chunks), 4)))
            # As evict() will, so that the ranks read here are the ones it evicts by.
            policy._move(policy.book.expire())
            ranks = {key: policy.book.rank(key) for key in chunks}
            assert all(policy._bucket_of[key] == rank for key, rank in ranks.items())
            expected = lru_would_evict(evictable, n, protected)
            evicted = policy.evict(n, protected)
            assert (evicted is not None) == expected, (n, protected, evictable)
            if evicted is None:
                seen["refused, as LRU would"] += 1
                continue
            keys = {key for key, _ in evicted}
            assert len(keys) == n and keys <= evictable - protected
            left = evictable - protected - keys
            if left:
                highest = max(ranks[key] for key in keys)
                assert highest <= min(ranks[key] for key in left), "a higher rank went"
            seen["granted"] += 1
            seen["leased chunks evicted"] += sum(ranks[key] > 0 for key in keys)
            seen["unleased chunks evicted"] += sum(ranks[key] == 0 for key in keys)
            for key in keys:
                del chunks[key]
            evictable -= keys
    # A check that never reached a case would pass without saying anything about it.
    assert seen["refused, as LRU would"] and seen["granted"]
    assert seen["leased chunks evicted"] and seen["unleased chunks evicted"]
    print(dict(seen))


if __name__ == "__main__":
    main()
