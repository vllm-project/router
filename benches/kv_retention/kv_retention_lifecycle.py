"""Retain and release lifecycle of the #427 prototype, without any vLLM import.

A hint rides a request as ``kv_transfer_params.kv_retention``. Four things stay apart:

- the logical TTL is the Router's or the client's own, and never reaches this module;
- requested retention is the hint;
- acknowledged retention is an ``Ack``, which names the tier's own TTL and the worker generation;
- observed residency is what the next turn reads, reported in a request receipt.

``RetentionLedger`` is the GPU tier: a hold keeps a finished request's blocks referenced. ``LeaseBook`` is
the CPU tier: a lease orders eviction. Neither promises residency: holds yield to demand and a lease is
evicted when nothing else is evictable.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO, Tuple

HINT_KEY = "kv_retention"
SUPPORTED_VERSION = "0.1"
GPU = "gpu"
CPU = "cpu"
RETAIN = "retain"
RELEASE = "release"

ACCEPTED = "accepted"
REJECTED = "rejected"
RELEASED = "released"
USED = "used"
EXPIRED = "expired"
EVICTED = "evicted"

RENEWED = "renewed"

SessionKey = Tuple[str, str, str]
"""Tenant, cache namespace, session: no hold or lease is reachable from another tenant or namespace."""

MAX_REMEMBERED_SESSIONS = 4096
"""Sessions whose last generation is remembered after their retention ends."""


def hint_body(
    session: str,
    generation: int,
    tiers: Iterable[str],
    ttl_seconds: float,
    op: str = RETAIN,
    tenant: str = "default",
    cache_namespace: str = "default",
) -> dict[str, Any]:
    """A client's hint for one turn: its id is the session and generation, so a retry repeats it."""
    return {
        "version": SUPPORTED_VERSION,
        "hint_id": f"{session}:{generation}",
        "op": op,
        "scope": {
            "tenant": tenant,
            "cache_namespace": cache_namespace,
            "session": session,
        },
        "generation": generation,
        "tiers": list(tiers),
        "ttl_seconds": ttl_seconds,
        "priority": 0,
    }


@dataclass(frozen=True)
class RetentionHint:
    hint_id: str
    op: str
    tenant: str
    cache_namespace: str
    session: str
    generation: int
    tiers: frozenset[str]
    ttl_seconds: float
    priority: int
    worker_generation: str | None

    @property
    def key(self) -> SessionKey:
        return (self.tenant, self.cache_namespace, self.session)


def parse_hint(
    params: Mapping[str, Any] | None,
) -> tuple[RetentionHint | None, str | None]:
    """The hint in ``kv_transfer_params``, or why it is ignored. Never raises: a bad hint cannot fail a request."""
    raw = (params or {}).get(HINT_KEY)
    if raw is None:
        return None, None
    if not isinstance(raw, Mapping):
        return None, "malformed"
    if raw.get("version") != SUPPORTED_VERSION:
        return None, "unsupported_version"
    scope = raw.get("scope")
    if not isinstance(scope, Mapping):
        return None, "malformed"
    names = (
        raw.get("hint_id"),
        scope.get("tenant"),
        scope.get("cache_namespace"),
        scope.get("session"),
    )
    if not all(isinstance(name, str) and name for name in names):
        return None, "malformed"
    op = raw.get("op")
    tiers = raw.get("tiers", [])
    generation = raw.get("generation")
    ttl = raw.get("ttl_seconds", 0.0)
    priority = raw.get("priority", 0)
    worker_generation = raw.get("worker_generation")
    well_formed = (
        op in (RETAIN, RELEASE)
        and isinstance(generation, int)
        and not isinstance(generation, bool)
        and generation >= 0
        and isinstance(ttl, (int, float))
        and not isinstance(ttl, bool)
        and math.isfinite(ttl)
        and isinstance(priority, int)
        and not isinstance(priority, bool)
        and isinstance(tiers, list)
        and all(tier in (GPU, CPU) for tier in tiers)
        and (worker_generation is None or isinstance(worker_generation, str))
    )
    if not well_formed:
        return None, "malformed"
    hint_id, tenant, namespace, session = names
    return (
        RetentionHint(
            hint_id,
            op,
            tenant,
            namespace,
            session,
            generation,
            frozenset(tiers),
            float(ttl),
            priority,
            worker_generation,
        ),
        None,
    )


@dataclass(frozen=True)
class Ack:
    """The backend's answer to a hint on one tier."""

    hint_id: str
    tier: str
    state: str
    reason: str | None
    worker_generation: str
    ttl_seconds: float | None = None
    """The TTL acknowledged, at most the tier's maximum, whatever the hint asked for."""
    tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetentionEnd:
    """How one hold or lease ended: its outcome, for the receipt and for an eviction notification."""

    hint_id: str
    tier: str
    outcome: str
    reason: str | None
    session: str
    tenant: str
    tokens: int
    held_seconds: float
    worker_generation: str
    hit_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Hold:
    hint_id: str
    key: SessionKey
    request_id: str
    generation: int
    blocks: int
    tokens: int
    acquired_at: float
    deadline: float


class RetentionLedger:
    """GPU holds: at most one per session, inside an engine and a tenant budget, released oldest first under demand."""

    def __init__(
        self,
        pool_blocks: int,
        engine_share: float,
        tenant_share: float,
        max_ttl_seconds: float,
        worker_generation: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.engine_budget = int(pool_blocks * engine_share)
        self.tenant_budget = int(pool_blocks * tenant_share)
        self.max_ttl_seconds = max_ttl_seconds
        self.worker_generation = worker_generation
        self._clock = clock
        self._holds: OrderedDict[SessionKey, _Hold] = OrderedDict()  # oldest first
        self._generations: OrderedDict[SessionKey, tuple[int, str]] = OrderedDict()
        self._tenant_blocks: Counter[str] = Counter()
        # Told to free, not yet handed to the worker.
        self._to_release: set[str] = set()
        # Blocks of holds that ended and that vLLM has not freed yet.
        self._releasing: dict[str, int] = {}
        self._ended: list[RetentionEnd] = []

    @property
    def held_blocks(self) -> int:
        return sum(hold.blocks for hold in self._holds.values())

    @property
    def releasing_blocks(self) -> int:
        return sum(self._releasing.values())

    def retain(
        self, hint: RetentionHint, request_id: str, blocks: int, tokens: int
    ) -> Ack:
        """Whether to hold ``request_id``'s blocks past its finish: only when the state is accepted and the
        reason is not ``RENEWED``, whose hold is an earlier request's."""
        rejected = self._refusal(hint)
        if rejected is not None:
            return self._ack(hint, REJECTED, rejected)
        now = self._clock()
        ttl = min(hint.ttl_seconds, self.max_ttl_seconds)
        hold = self._holds.get(hint.key)
        if self._last_seen(hint.key) == (hint.generation, hint.hint_id):
            if hold is None or hold.hint_id != hint.hint_id:
                return self._ack(hint, REJECTED, "already_ended")
            hold.deadline = now + ttl
            self._holds.move_to_end(hint.key)
            return self._ack(hint, ACCEPTED, RENEWED, ttl, hold.tokens)
        tenant = hint.tenant
        superseded = hold.blocks if hold is not None else 0
        if self._tenant_blocks[tenant] - superseded + blocks > self.tenant_budget:
            return self._ack(hint, REJECTED, "over_tenant_budget")
        if self.held_blocks - superseded + blocks > self.engine_budget:
            return self._ack(hint, REJECTED, "over_engine_budget")
        if hold is not None:
            self._end(hold, RELEASED, "superseded")
        self._remember(hint)
        self._holds[hint.key] = _Hold(
            hint.hint_id,
            hint.key,
            request_id,
            hint.generation,
            blocks,
            tokens,
            now,
            now + ttl,
        )
        self._tenant_blocks[tenant] += blocks
        return self._ack(hint, ACCEPTED, None, ttl, tokens)

    def release(self, hint: RetentionHint) -> Ack:
        """Release the session's hold. Idempotent: no live hold is a success with no effect. A release from
        another tenant or namespace names another key, so it finds nothing to touch."""
        if hint.worker_generation not in (None, self.worker_generation):
            return self._ack(hint, REJECTED, "stale_worker_generation")
        hold = self._holds.get(hint.key)
        if hold is None:
            return self._ack(hint, RELEASED, "no_live_hold")
        if hint.generation < hold.generation:
            return self._ack(hint, REJECTED, "stale_generation")
        self._end(hold, RELEASED, "client")
        return self._ack(hint, RELEASED, None, tokens=hold.tokens)

    def on_next_turn(self, key: SessionKey, generation: int, hit_tokens: int) -> None:
        """A newer request of the session holds its own references now, so the hold has done its work. A retry
        of the hold's own request, which has its generation, leaves it to the renewal at that request's finish."""
        hold = self._holds.get(key)
        if hold is not None and generation > hold.generation:
            self._end(
                hold,
                USED if hit_tokens > 0 else RELEASED,
                None if hit_tokens > 0 else "no_hit",
                hit_tokens,
            )

    def expire(self) -> None:
        now = self._clock()
        for hold in [hold for hold in self._holds.values() if hold.deadline <= now]:
            self._end(hold, EXPIRED, None)

    def yield_blocks(self, needed: int, exclude: SessionKey | None = None) -> int:
        """Release holds oldest first until ``needed`` blocks are on their way back. A session's own hold is
        spared: the request that needs room already counts those blocks as hits."""
        released = 0
        for hold in list(self._holds.values()):
            if released >= needed:
                break
            if hold.key != exclude:
                self._end(hold, EVICTED, "demand")
                released += hold.blocks
        return released

    def reset(self) -> None:
        for hold in list(self._holds.values()):
            self._end(hold, RELEASED, "reset")

    def take_releases(self) -> set[str]:
        """The requests to report as finished sending in this step, once each."""
        releases, self._to_release = self._to_release, set()
        return releases

    def on_freed(self, request_ids: Iterable[str]) -> None:
        for request_id in request_ids:
            self._releasing.pop(request_id, None)

    def take_ended(self) -> list[RetentionEnd]:
        ended, self._ended = self._ended, []
        return ended

    def _refusal(self, hint: RetentionHint) -> str | None:
        if hint.worker_generation not in (None, self.worker_generation):
            return "stale_worker_generation"
        if hint.ttl_seconds <= 0:
            return "bad_ttl"
        last = self._last_seen(hint.key)
        if last is not None:
            if hint.generation < last[0]:
                return "stale_generation"
            if hint.generation == last[0] and hint.hint_id != last[1]:
                return "generation_conflict"
        return None

    def _last_seen(self, key: SessionKey) -> tuple[int, str] | None:
        """The session's newest generation and hint. Memory of ended sessions is bounded: a live hold is its own."""
        hold = self._holds.get(key)
        if hold is not None:
            return hold.generation, hold.hint_id
        return self._generations.get(key)

    def _remember(self, hint: RetentionHint) -> None:
        self._generations[hint.key] = (hint.generation, hint.hint_id)
        self._generations.move_to_end(hint.key)
        while len(self._generations) > MAX_REMEMBERED_SESSIONS:
            self._generations.popitem(last=False)

    def _end(
        self, hold: _Hold, outcome: str, reason: str | None, hit_tokens: int = 0
    ) -> None:
        del self._holds[hold.key]
        self._tenant_blocks[hold.key[0]] -= hold.blocks
        self._to_release.add(hold.request_id)
        self._releasing[hold.request_id] = hold.blocks
        self._ended.append(
            RetentionEnd(
                hold.hint_id,
                GPU,
                outcome,
                reason,
                hold.key[2],
                hold.key[0],
                hold.tokens,
                self._clock() - hold.acquired_at,
                self.worker_generation,
                hit_tokens,
            )
        )

    def _ack(
        self,
        hint: RetentionHint,
        state: str,
        reason: str | None,
        ttl: float | None = None,
        tokens: int = 0,
    ) -> Ack:
        return Ack(
            hint.hint_id, GPU, state, reason, self.worker_generation, ttl, tokens
        )


@dataclass
class _Lease:
    hint_id: str
    generation: int
    rank: int
    ttl: float
    expires_at: float
    acquired_at: float
    keys: dict[Hashable, None]


class LeaseBook:
    """CPU leases: which chunks a session's lease covers and until when, inside an engine and a tenant budget.

    A lease is one rank in the eviction order, so it is a preference: ``LeasePolicy`` evicts leased chunks when
    nothing else is evictable, so that a store fails only where plain LRU would fail too.
    """

    RANKS = 3

    def __init__(
        self,
        capacity_chunks: int,
        engine_share: float,
        tenant_share: float,
        max_ttl_seconds: float,
        worker_generation: str,
        tokens_per_chunk: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.engine_budget = int(capacity_chunks * engine_share)
        self.tenant_budget = int(capacity_chunks * tenant_share)
        self.max_ttl_seconds = max_ttl_seconds
        self.worker_generation = worker_generation
        self.tokens_per_chunk = tokens_per_chunk
        self._clock = clock
        self._leases: dict[SessionKey, _Lease] = {}
        self._sessions_of: dict[Hashable, set[SessionKey]] = {}
        self._tenant_chunks: Counter[str] = Counter()
        self._generations: OrderedDict[SessionKey, tuple[int, str]] = OrderedDict()
        self._ended: list[RetentionEnd] = []
        # Hint id to leased chunks evicted since the last take.
        self._evicted: Counter[str] = Counter()
        self.chunks_clipped = 0
        self.chunks_evicted = 0
        self.leased_chunks_evicted = 0

    @property
    def leased_chunks(self) -> int:
        return len(self._sessions_of)

    def open(self, hint: RetentionHint) -> Ack:
        """Acknowledge a retain. The lease covers keys as they reach ``tag``; its clock restarts at ``turn_ended``."""
        if hint.worker_generation not in (None, self.worker_generation):
            return self._ack(hint, REJECTED, "stale_worker_generation")
        if hint.ttl_seconds <= 0:
            return self._ack(hint, REJECTED, "bad_ttl")
        lease = self._leases.get(hint.key)
        # Memory of ended sessions is bounded: a live lease is its own.
        last = (
            (lease.generation, lease.hint_id)
            if lease is not None
            else self._generations.get(hint.key)
        )
        if last is not None and last != (hint.generation, hint.hint_id):
            if hint.generation < last[0]:
                return self._ack(hint, REJECTED, "stale_generation")
            if hint.generation == last[0]:
                return self._ack(hint, REJECTED, "generation_conflict")
        ttl = min(hint.ttl_seconds, self.max_ttl_seconds)
        now = self._clock()
        rank = 1 + min(max(hint.priority, 0), self.RANKS - 1)
        if lease is None:
            self._leases[hint.key] = _Lease(
                hint.hint_id, hint.generation, rank, ttl, now + ttl, now, {}
            )
        else:
            lease.hint_id, lease.generation = hint.hint_id, hint.generation
            lease.rank, lease.ttl, lease.expires_at = rank, ttl, now + ttl
        self._generations[hint.key] = (hint.generation, hint.hint_id)
        self._generations.move_to_end(hint.key)
        while len(self._generations) > MAX_REMEMBERED_SESSIONS:
            self._generations.popitem(last=False)
        return self._ack(hint, ACCEPTED, None, ttl)

    def tag(self, key: SessionKey, keys: Iterable[Hashable]) -> list[Hashable]:
        """Cover ``keys`` (head to tail) with the session's lease, as far as the budgets allow. A clipped lease
        still covers the prefix's head, which is what a restore can use. Returns the keys it now covers."""
        lease = self._leases.get(key)
        if lease is None:
            return []
        covered = []
        tenant = key[0]
        for chunk in keys:
            if chunk in lease.keys:
                continue
            is_new = chunk not in self._sessions_of
            over_engine = is_new and self.leased_chunks >= self.engine_budget
            if over_engine or self._tenant_chunks[tenant] >= self.tenant_budget:
                self.chunks_clipped += 1
                continue
            lease.keys[chunk] = None
            self._sessions_of.setdefault(chunk, set()).add(key)
            self._tenant_chunks[tenant] += 1
            covered.append(chunk)
        return covered

    def turn_ended(self, hint: RetentionHint) -> None:
        """The session's TTL counts from now. A turn longer than the TTL has outlived the lease its hint opened, so
        the hint opens it again; a newer generation's lease stays that generation's, and only its clock restarts."""
        self.open(hint)
        if (lease := self._leases.get(hint.key)) is not None:
            lease.expires_at = self._clock() + lease.ttl

    def release(self, hint: RetentionHint) -> tuple[Ack, list[Hashable]]:
        """Idempotent, as for a hold: a release names only its own tenant's and namespace's key."""
        if hint.worker_generation not in (None, self.worker_generation):
            return self._ack(hint, REJECTED, "stale_worker_generation"), []
        lease = self._leases.get(hint.key)
        if lease is None:
            return self._ack(hint, RELEASED, "no_live_lease"), []
        if hint.generation < lease.generation:
            return self._ack(hint, REJECTED, "stale_generation"), []
        return self._ack(hint, RELEASED, None), self._drop(hint.key, RELEASED, "client")

    def expire(self) -> list[Hashable]:
        """End the leases past their TTL. Returns the chunks whose rank may have changed."""
        now = self._clock()
        moved: list[Hashable] = []
        for key in [
            key for key, lease in self._leases.items() if lease.expires_at <= now
        ]:
            moved.extend(self._drop(key, EXPIRED, None))
        return moved

    def rank(self, chunk: Hashable) -> int:
        """0 for an unleased chunk; otherwise the highest rank among the live leases that cover it."""
        return max(
            (self._leases[key].rank for key in self._sessions_of.get(chunk, ())),
            default=0,
        )

    def forget(self, chunk: Hashable, evicted: bool) -> None:
        """A chunk left the tier. An evicted leased chunk is its leases' eviction notification."""
        self.chunks_evicted += evicted
        sessions = self._sessions_of.pop(chunk, ())
        self.leased_chunks_evicted += evicted and bool(sessions)
        for key in sessions:
            lease = self._leases[key]
            lease.keys.pop(chunk, None)
            self._tenant_chunks[key[0]] -= 1
            if evicted:
                self._evicted[lease.hint_id] += 1

    def clear(self) -> None:
        for key in list(self._leases):
            self._drop(key, RELEASED, "reset")

    def take_ended(self) -> list[RetentionEnd]:
        ended, self._ended = self._ended, []
        return ended

    def take_evictions(self) -> dict[str, int]:
        """Leased chunks evicted since the last call, by hint: the tier's eviction notification."""
        evicted, self._evicted = dict(self._evicted), Counter()
        return evicted

    def _drop(
        self, key: SessionKey, outcome: str, reason: str | None
    ) -> list[Hashable]:
        lease = self._leases.pop(key)
        for chunk in lease.keys:
            self._tenant_chunks[key[0]] -= 1
            if (sessions := self._sessions_of.get(chunk)) is not None:
                sessions.discard(key)
                if not sessions:
                    del self._sessions_of[chunk]
        self._ended.append(
            RetentionEnd(
                lease.hint_id,
                CPU,
                outcome,
                reason,
                key[2],
                key[0],
                len(lease.keys) * self.tokens_per_chunk,
                self._clock() - lease.acquired_at,
                self.worker_generation,
            )
        )
        return list(lease.keys)

    def _ack(
        self,
        hint: RetentionHint,
        state: str,
        reason: str | None,
        ttl: float | None = None,
    ) -> Ack:
        return Ack(hint.hint_id, CPU, state, reason, self.worker_generation, ttl)


class ReceiptLog:
    """One JSON line per event, flushed at once, so a killed engine loses nothing that happened before it."""

    def __init__(self, path: str | Path | None) -> None:
        self._file: TextIO | None = (
            open(path, "a", buffering=1, encoding="utf-8") if path else None
        )
        self._lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> None:
        if self._file is None:
            return
        line = json.dumps(
            {"t": time.time(), "event": event, **fields}, separators=(",", ":")
        )
        with self._lock:
            self._file.write(line + "\n")
