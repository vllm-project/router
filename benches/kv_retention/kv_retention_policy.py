"""CPU-tier retention for vLLM 0.32's native offload: leases over LRU.

Load it through the connector's extra config (``kv_retention_connector`` does the rest):

    "kv_connector_extra_config": {"eviction_policy": "LeasePolicy",
                                  "cache_policy_module_path": "kv_retention_policy", ...}

Chunks without a live lease are evicted first, LRU, tail first. Leased chunks are evicted only when nothing else is
evictable, lowest rank first. So a store fails only where plain LRU would fail too: a refused eviction makes vLLM drop
the store silently. The hint is read from the request context, parsed once per request.
"""

from __future__ import annotations

from typing import NamedTuple

from kv_retention_lifecycle import CPU, RETAIN, LeaseBook, RetentionHint, parse_hint
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy, order_request_keys
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy


class _ParsedHint(NamedTuple):
    hint: RetentionHint | None


def _retain_hint(ctx) -> RetentionHint | None:
    """The request's CPU-tier retain hint, if it has one."""
    if (parsed := ctx.get_state(_ParsedHint)) is None:
        hint, _ = parse_hint(ctx.kv_transfer_params)
        parsed = _ParsedHint(
            hint
            if hint is not None and hint.op == RETAIN and CPU in hint.tiers
            else None
        )
        ctx.set_state(parsed)
    return parsed.hint


class LeasePolicy(CachePolicy):
    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        self.capacity_chunks = cache_capacity
        # No lease exists before the connector configures the book; these defaults only keep the type whole.
        self.book = LeaseBook(cache_capacity, 0.0, 0.0, 0.0, "unconfigured")
        self._buckets = [
            LRUCachePolicy(cache_capacity) for _ in range(LeaseBook.RANKS + 1)
        ]
        self._bucket_of = {}  # OffloadKey -> bucket index

    def _move(self, keys_head_to_tail):
        """Re-bucket cached keys whose rank changed, tail first so that the oldest keep the LRU order."""
        for key in reversed(list(keys_head_to_tail)):
            current = self._bucket_of.get(key)
            if current is not None and (rank := self.book.rank(key)) != current:
                chunk = self._buckets[current].get(key)
                self._buckets[current].remove(
                    key
                )  # the chunk keeps its status and ref_cnt
                self._buckets[rank].insert(key, chunk)
                self._bucket_of[key] = rank

    def _cover(self, ctx, keys):
        """Cover the request's keys with its session's lease, if it has one."""
        self._move(self.book.expire())
        if (hint := _retain_hint(ctx)) is not None:
            self._move(self.book.tag(hint.key, keys))

    def release(self, hint):
        ack, keys = self.book.release(hint)
        self._move(keys)
        return ack

    # CachePolicy interface (vllm/v1/kv_offload/cpu/policies/base.py)

    def get(self, key):
        bucket = self._bucket_of.get(key)
        return None if bucket is None else self._buckets[bucket].get(key)

    def insert(self, key, chunk):
        self._bucket_of[key] = rank = self.book.rank(key)
        self._buckets[rank].insert(key, chunk)

    def remove(self, key):
        self._buckets[self._bucket_of.pop(key)].remove(key)
        self.book.forget(key, evicted=False)

    def mark_evictable(self, key):
        self._buckets[self._bucket_of[key]].mark_evictable(key)

    def mark_non_evictable(self, key):
        self._buckets[self._bucket_of[key]].mark_non_evictable(key)

    def touch(self, keys, req_context):
        """Keys come head to tail: the tail ends up oldest."""
        by_bucket = {}
        for key in keys:
            if (bucket := self._bucket_of.get(key)) is not None:
                by_bucket.setdefault(bucket, []).append(key)
        for bucket, bucket_keys in by_bucket.items():
            self._buckets[bucket].touch(bucket_keys, req_context)

    def on_store_miss(self, keys, req_context):
        """Called before insert(): tag the request's keys so that they land in its lease's bucket."""
        self._cover(req_context, list(keys))

    def on_request_finished(
        self, key_groups, insertion_only_keys, reused_keys, req_context
    ):
        """The turn ended, so the lease's clock restarts."""
        keys = [
            key
            for key in order_request_keys(key_groups, req_context)
            if key in self._bucket_of
        ]
        self._cover(req_context, keys)
        if (hint := _retain_hint(req_context)) is not None:
            self.book.touch(hint.key)
        self.touch(keys, req_context)

    def evict(self, n, protected):
        self._move(self.book.expire())
        plan, left = [], n
        for bucket in self._buckets:  # unleased, then the leases' ranks
            available = len(bucket._evictable) - sum(
                1 for key in protected if key in bucket._evictable
            )
            if left and (take := min(left, available)) > 0:
                plan.append((bucket, take))
                left -= take
        if left:
            return None  # atomic: nothing changed, and the manager drops the store
        evicted = [
            key_chunk
            for bucket, take in plan
            for key_chunk in bucket.evict(take, protected)
        ]
        for key, _ in evicted:
            self._bucket_of.pop(key, None)
            self.book.forget(key, evicted=True)
        return evicted

    def clear(self):
        self.book.clear()
        for bucket in self._buckets:
            bucket.clear()
        self._bucket_of = {}
