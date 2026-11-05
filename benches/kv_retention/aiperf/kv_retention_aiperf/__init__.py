"""The arm's only client-side difference: the retention hint a turn carries, or none."""

from __future__ import annotations

import os
from typing import Any

from aiperf.common.models import RequestInfo
from flatshare_bench.aiperf_plugin import FlatshareChatEndpoint
from kv_retention_lifecycle import HINT_KEY, hint_body


class RetentionChatEndpoint(FlatshareChatEndpoint):
    """Flatshare's endpoint, which sends the session headers, plus the hint. A conversation is one session: each
    subagent's context is a prefix of its own. Its turn index is the generation, and a last turn retains nothing
    (``expected_resume`` in Program metadata)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tiers = [
            tier for tier in os.environ.get("KV_RETENTION_TIERS", "").split(",") if tier
        ]
        self.ttl_seconds = float(os.environ.get("KV_RETENTION_TTL_SECONDS", "120"))

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        payload = super().format_payload(request_info)
        if self.tiers and not request_info.is_final_turn:
            hint = hint_body(
                request_info.x_correlation_id,
                request_info.turn_index,
                self.tiers,
                self.ttl_seconds,
                tenant="replay",
            )
            payload["kv_transfer_params"] = {HINT_KEY: hint}
        return payload
