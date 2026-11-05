"""vLLM 0.32's OffloadingConnector with the #427 retain and release hints.

Load it with a Python path that holds this directory:

    {"kv_connector": "RetentionConnector", "kv_role": "kv_both",
     "kv_connector_module_path": "kv_retention_connector",
     "kv_connector_extra_config": {"cpu_bytes_to_use": ..., "offload_prompt_only": false,
                                   "eviction_policy": "LeasePolicy", "cache_policy_module_path": "kv_retention_policy",
                                   "retention_receipts": "receipts.jsonl", "retention_gpu_share": 0.25, ...}}

GPU tier: a retained request's blocks stay referenced past its finish: ``request_finished_all_groups`` answers
``delay=True`` and the worker reports the request as finished sending when the hold ends, on the session's next
request, a release, its TTL or demand. vLLM frees the blocks at that step. While a hold exists the engine keeps stepping
(``Scheduler.has_finished_requests`` counts it), so a hold expires on time on an idle engine too, at the cost of vLLM's
1 ms poll.

It is a hard hold: held blocks cannot be evicted, whereas vLLM's proposed eviction priority (#59714) would leave them
evictable. So it yields: a waiting request that does not fit, or a running one that cannot grow, releases holds oldest
first, at the same step, before anything is preempted.

CPU tier: ``kv_retention_policy.LeasePolicy``, when it is the configured eviction policy.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field

from kv_retention_lifecycle import (
    ACCEPTED,
    CPU,
    GPU,
    HINT_KEY,
    REJECTED,
    RELEASE,
    RENEWED,
    RETAIN,
    Ack,
    LeaseBook,
    ReceiptLog,
    RetentionHint,
    RetentionLedger,
    parse_hint,
)
from kv_retention_policy import LeasePolicy
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.request import RequestStatus

COMPLETED = (RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED)
"""Finishes whose blocks are all computed: an aborted request's may not be."""
TOTALS_PERIOD_SECONDS = 10.0


@dataclass
class RetentionMetadata(OffloadingConnectorMetadata):
    """The step's metadata, plus the holds that end at this step: scheduler to worker, whatever the process."""

    release_req_ids: set[str] = field(default_factory=set)


@dataclass
class _Turn:
    hint: RetentionHint | None
    ignored: str | None
    cpu_ack: Ack | None = None
    gpu_hit_tokens: int = 0
    cpu_hit_tokens: int = 0
    lookups: int = 0


class RetentionConnector(OffloadingConnector):
    # vLLM requires this 3-argument constructor signature.
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        if role != KVConnectorRole.SCHEDULER:
            return
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        generation = uuid.uuid4().hex
        self._block_tokens = vllm_config.cache_config.block_size
        self._turns: dict[str, _Turn] = {}
        self._receipts = ReceiptLog(extra.get("retention_receipts"))
        self._totals_at = 0.0
        self._ledger = RetentionLedger(
            kv_cache_config.num_blocks,
            float(extra.get("retention_gpu_share", 0.25)),
            float(
                extra.get(
                    "retention_gpu_tenant_share", extra.get("retention_gpu_share", 0.25)
                )
            ),
            float(extra.get("retention_gpu_max_ttl_seconds", 10.0)),
            generation,
        )
        self._policy = getattr(self.connector_scheduler.manager, "_policy", None)
        if isinstance(self._policy, LeasePolicy):
            self._policy.book = LeaseBook(
                self._policy.capacity_chunks,
                float(extra.get("retention_cpu_share", 0.5)),
                float(
                    extra.get(
                        "retention_cpu_tenant_share",
                        extra.get("retention_cpu_share", 0.5),
                    )
                ),
                float(extra.get("retention_cpu_max_ttl_seconds", 120.0)),
                generation,
                self._block_tokens * int(extra.get("blocks_per_chunk", 1)),
            )
        else:
            self._policy = None
        self._receipts.write(
            "start",
            worker_generation=generation,
            pool_blocks=kv_cache_config.num_blocks,
        )

    # Scheduler side

    def on_new_request(self, request):
        hint, ignored = parse_hint(request.kv_transfer_params)
        turn = _Turn(hint, ignored)
        self._turns[request.request_id] = turn
        if ignored is not None:
            self._receipts.write(
                "ignored", request_id=request.request_id, reason=ignored
            )
        if hint is not None and hint.op == RELEASE:
            self._release(hint, request.request_id)
        elif hint is not None and hint.op == RETAIN and CPU in hint.tiers:
            turn.cpu_ack = (
                self._policy.book.open(hint)
                if self._policy is not None
                else Ack(
                    hint.hint_id,
                    CPU,
                    REJECTED,
                    "unsupported_tier",
                    self._ledger.worker_generation,
                )
            )
            self._receipts.write(
                "ack", request_id=request.request_id, **turn.cpu_ack.to_dict()
            )
        super().on_new_request(request)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        result = super().get_num_new_matched_tokens(request, num_computed_tokens)
        turn = self._turns.get(request.request_id)
        if turn is not None:
            turn.gpu_hit_tokens = num_computed_tokens
            turn.lookups += 1
        if result[0] is not None:
            self._yield_to(request, num_computed_tokens, turn)
        return result

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        result = super().update_state_after_alloc(request, blocks, num_external_tokens)
        turn = self._turns.get(request.request_id)
        if turn is not None:
            # vLLM calls this again when a load is done, with no external tokens left to count.
            turn.cpu_hit_tokens = num_external_tokens or turn.cpu_hit_tokens
            turn.lookups = 0  # admitted: a later lookup is a preempted request's first
            if turn.hint is not None:
                # The request has taken its references, so the session's hold can go.
                self._ledger.on_next_turn(
                    turn.hint.key,
                    turn.hint.generation,
                    turn.gpu_hit_tokens + turn.cpu_hit_tokens,
                )
        return result

    def has_pending_block_frees(self):
        """A running request cannot grow: release the oldest hold rather than preempt anything."""
        return self._ledger.releasing_blocks > 0 or self._ledger.yield_blocks(1) > 0

    def update_connector_output(self, connector_output):
        super().update_connector_output(connector_output)
        self._ledger.on_freed(connector_output.finished_sending or ())

    def build_connector_meta(self, scheduler_output):
        meta = super().build_connector_meta(scheduler_output)
        self._ledger.expire()
        self._write_ended()
        return RetentionMetadata(
            meta.load_jobs,
            meta.store_jobs,
            meta.jobs_to_flush,
            self._ledger.take_releases(),
        )

    def request_finished_all_groups(self, request, block_ids):
        delay, params = super().request_finished_all_groups(request, block_ids)
        turn = self._turns.pop(request.request_id, None)
        if turn is None:
            return delay, params
        acks = [turn.cpu_ack] if turn.cpu_ack is not None else []
        hold = False
        if turn.hint is not None and turn.hint.op == RETAIN and GPU in turn.hint.tiers:
            ack = self._retain(turn.hint, request, block_ids)
            hold = ack.state == ACCEPTED and ack.reason != RENEWED
            acks.append(ack)
            self._receipts.write("ack", request_id=request.request_id, **ack.to_dict())
        observed = {
            "gpu_hit_tokens": turn.gpu_hit_tokens,
            "cpu_hit_tokens": turn.cpu_hit_tokens,
        }
        self._receipts.write(
            "request",
            request_id=request.request_id,
            hint_id=turn.hint.hint_id if turn.hint is not None else None,
            session=turn.hint.session if turn.hint is not None else None,
            prompt_tokens=request.num_prompt_tokens,
            completed=request.status in COMPLETED,
            **observed,
        )
        if turn.hint is None:
            return delay, params
        answer = {
            "worker_generation": self._ledger.worker_generation,
            "hint_id": turn.hint.hint_id,
            "acks": [ack.to_dict() for ack in acks],
            "observed": observed,
        }
        return delay or hold, {**(params or {}), HINT_KEY: answer}

    def reset_cache(self):
        self._ledger.reset()
        return super().reset_cache()

    def _retain(self, hint, request, block_ids):
        generation = self._ledger.worker_generation
        if request.status not in COMPLETED:
            return Ack(hint.hint_id, GPU, REJECTED, "request_not_completed", generation)
        # The prefix cache's namespace is the request's cache salt: a hint cannot name another one.
        if hint.cache_namespace != (request.cache_salt or "default"):
            return Ack(hint.hint_id, GPU, REJECTED, "namespace_mismatch", generation)
        blocks = sum(len(group) for group in block_ids)
        return self._ledger.retain(
            hint, request.request_id, blocks, blocks * self._block_tokens
        )

    def _release(self, hint, request_id):
        if GPU in hint.tiers:
            ack = self._ledger.release(hint)
            self._receipts.write("ack", request_id=request_id, **ack.to_dict())
        if CPU in hint.tiers and self._policy is not None:
            ack = self._policy.release(hint)
            self._receipts.write("ack", request_id=request_id, **ack.to_dict())

    def _yield_to(self, request, num_computed_tokens, turn):
        """Make room for a waiting request that does not fit. A request looked up again with nothing on its way
        back was not admitted for a reason this estimate misses (vLLM's reservations, for one): each such lookup
        takes one more hold."""
        needed = math.ceil(
            (request.num_tokens - num_computed_tokens) / self._block_tokens
        )
        releasing = self._ledger.releasing_blocks
        free = self._kv_cache_manager.block_pool.get_num_free_blocks() + releasing
        exclude = turn.hint.key if turn is not None and turn.hint is not None else None
        if needed > free:
            self._ledger.yield_blocks(needed - free, exclude)
        elif turn is not None and turn.lookups > 1 and not releasing:
            self._ledger.yield_blocks(1, exclude)

    def _write_ended(self):
        for end in self._ledger.take_ended():
            self._receipts.write("end", **end.to_dict())
        if self._policy is not None:
            for end in self._policy.book.take_ended():
                self._receipts.write("end", **end.to_dict())
            for hint_id, chunks in self._policy.book.take_evictions().items():
                self._receipts.write("cpu_evicted", hint_id=hint_id, chunks=chunks)
            if (now := time.monotonic()) - self._totals_at >= TOTALS_PERIOD_SECONDS:
                self._totals_at = now
                book = self._policy.book
                self._receipts.write(
                    "cpu_tier",
                    chunks_evicted=book.chunks_evicted,
                    leased_chunks_evicted=book.leased_chunks_evicted,
                    chunks_clipped=book.chunks_clipped,
                    leased_chunks=book.leased_chunks,
                )

    # Worker side

    def get_finished(self, finished_req_ids):
        sending, receiving = super().get_finished(finished_req_ids)
        if isinstance(self._connector_metadata, RetentionMetadata):
            sending = set(sending) | self._connector_metadata.release_req_ids
        return sending, receiving
