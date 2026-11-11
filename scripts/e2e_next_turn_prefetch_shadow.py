#!/usr/bin/env python3
"""Drive a real Router through scripted tool waits and check next-turn prefetch shadow scoring.

The backend is a usage-only mock. Each scenario is one Program: a turn, a scripted wait, and the
next turn. The Router must score each wait as expected, and with the shadow off it must send the
backend exactly the same requests: the shadow reads and never acts.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MODEL = "shadow-e2e-model"
PREFIX_TOKENS = 8000
TICK_SECONDS = 0.1
# Three ticks: the Router scrapes the mock and evaluates its waits at least twice.
SETTLE_SECONDS = 3 * TICK_SECONDS
# Longer than restore_evidence_window_seconds, so restores from before a scenario that stops them
# no longer count, and several scrapes see the scenario's gauges.
CONFIGURE_SECONDS = 0.7
# Timing: a plan fires at the first tick after the idle floor (0.5-0.6 s) and its load takes 1 s, so
# it finishes at 1.5-1.6 s and stays useful until 4.5-4.6 s. Every scripted return is at least
# 0.4 s from the nearest of those boundaries.
RETURN_SHORT = 0.05
RETURN_LATE = 1.0
RETURN_IN_TIME = 2.5
RETURN_AFTER_WINDOW = 5.5
SHADOW_CONFIG = {
    "mode": "shadow",
    "min_idle_seconds": 0.5,
    "validity_seconds": 3.0,
    "min_useful_probability": 0.5,
    "min_prefix_tokens": 1024,
    "max_kv_cache_usage": 0.9,
    "max_speculative_tokens_per_target": 1_000_000,
    "restore_evidence_window_seconds": 0.5,
    "load_cost_model": {"intercept_seconds": 1.0, "seconds_per_1k_tokens": 0.0},
}


def _hint(expected: float = RETURN_IN_TIME, **extra: Any) -> dict[str, Any]:
    return {"version": "1", "expected_return_seconds": expected, **extra}


@dataclass(frozen=True)
class Scenario:
    name: str
    wait_seconds: float
    expected: str
    hint: dict[str, Any] | None = None
    next_cached_tokens: int = PREFIX_TOKENS
    backend: dict[str, Any] = field(default_factory=dict)
    # Whether the next turn streams, and whether it asks for a usage chunk.
    stream: bool = False
    include_usage: bool = False


SCENARIOS = [
    Scenario("hint_in_time", RETURN_IN_TIME, "outcome:in_time", _hint()),
    Scenario("hint_late", RETURN_LATE, "outcome:late", _hint()),
    Scenario("hint_after_window", RETURN_AFTER_WINDOW, "outcome:expired", _hint()),
    Scenario("hint_no_prefix", RETURN_IN_TIME, "outcome:no_prefix", _hint(), next_cached_tokens=10),
    Scenario("profile_in_time", RETURN_IN_TIME, "outcome:in_time"),
    Scenario("streaming_usage", RETURN_IN_TIME, "outcome:in_time", _hint(), stream=True, include_usage=True),
    Scenario("streaming_no_usage", RETURN_IN_TIME, "outcome:unverified", _hint(), stream=True),
    Scenario("short_wait", RETURN_SHORT, "skip:short_wait", _hint()),
    Scenario("low_confidence", RETURN_IN_TIME, "skip:low_probability", _hint(confidence=0.2)),
    Scenario("unsupported_hint_version", RETURN_IN_TIME, "outcome:in_time", {"version": "2"}),
    Scenario(
        "kv_pressure",
        RETURN_IN_TIME,
        "skip:capacity_pressure",
        _hint(),
        backend={"kv_usage": 0.95},
    ),
    Scenario(
        "demand_waiting",
        RETURN_IN_TIME,
        "skip:demand_waiting",
        _hint(),
        backend={"waiting": 1},
    ),
    Scenario(
        "nothing_restored",
        RETURN_IN_TIME,
        "skip:no_restore_evidence",
        _hint(),
        backend={"restoring": False},
    ),
]


class Backend:
    """What the mock serves and everything it was asked."""

    def __init__(self) -> None:
        self.kv_usage = 0.5
        self.waiting = 0
        self.restoring = True
        self.restored_tokens = 0
        self.chat_bodies: list[dict[str, Any]] = []
        self.requests: Counter[str] = Counter()
        self.lock = threading.Lock()

    def configure(self, kv_usage: float = 0.5, waiting: int = 0, restoring: bool = True) -> None:
        with self.lock:
            self.kv_usage, self.waiting, self.restoring = kv_usage, waiting, restoring

    def metrics(self) -> str:
        with self.lock:
            self.restored_tokens += 100 if self.restoring else 0
            labels = f'engine="0",model_name="{MODEL}"'
            return (
                f"vllm:kv_cache_usage_perc{{{labels}}} {self.kv_usage}\n"
                f"vllm:num_requests_running{{{labels}}} 0\n"
                f"vllm:num_requests_waiting{{{labels}}} {self.waiting}\n"
                f'vllm:prompt_tokens_by_source_total{{{labels},source="external_kv_transfer"}} '
                f"{self.restored_tokens}\n"
            )


def _handler(backend: Backend) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            backend.requests[f"GET {self.path}"] += 1
            if self.path == "/metrics":
                self._send(backend.metrics().encode(), "text/plain")
            elif self.path == "/v1/models":
                self._json({"object": "list", "data": [{"id": MODEL, "object": "model"}]})
            elif self.path == "/get_model_info":
                self._json({"model_path": MODEL, "model_id": MODEL})
            else:
                self._json({"status": "ok"})

        def do_POST(self) -> None:  # noqa: N802
            backend.requests[f"POST {self.path}"] += 1
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", "0"))) or b"{}")
            with backend.lock:
                backend.chat_bodies.append(body)
            usage = {
                "prompt_tokens": PREFIX_TOKENS,
                "completion_tokens": 8,
                "total_tokens": PREFIX_TOKENS + 8,
                "prompt_tokens_details": {
                    "cached_tokens": int(body["vllm_xargs"]["replay_cached_tokens"])
                },
            }
            if body.get("stream"):
                self._stream(usage if (body.get("stream_options") or {}).get("include_usage") else None)
                return
            self._json(
                {
                    "id": "e2e",
                    "object": "chat.completion",
                    "model": MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                }
            )

        def _stream(self, usage: dict[str, Any] | None) -> None:
            chunk = {"id": "e2e", "object": "chat.completion.chunk", "model": MODEL}
            events = [
                {**chunk, "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}]},
                {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            if usage is not None:
                events.append({**chunk, "choices": [], "usage": usage})
            lines = [f"data: {json.dumps(event)}\n\n" for event in events] + ["data: [DONE]\n\n"]
            self._send("".join(lines).encode(), "text/event-stream")

        def _json(self, value: object) -> None:
            self._send(json.dumps(value).encode(), "application/json")

        def _send(self, encoded: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode()


def _turn(port: int, scenario: Scenario, resume: bool, cached: int) -> int:
    """One turn. The next turn of a streaming scenario streams, so its usage takes that path."""
    context: dict[str, Any] = {"program_id": scenario.name, "expected_resume": resume}
    if resume and scenario.hint is not None:
        context["next_turn_hint"] = scenario.hint
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "scripted turn"}],
        "stream": False,
        "vllm_xargs": {"agentic_context": context, "replay_cached_tokens": cached},
    }
    if scenario.stream and not resume:
        payload["stream"] = True
        if scenario.include_usage:
            payload["stream_options"] = {"include_usage": True}
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()
        return response.status


def _write_profile(path: Path) -> None:
    """A global profile that returns inside the useful window.

    A lookup uses the most specific key with enough support, so after the first scenarios the
    waits' own key, fed by their live returns, decides; this profile only covers the start.
    """
    global_key = {
        key: None
        for key in (
            "workflow_profile_id",
            "profile_version",
            "model_pool",
            "agent_role",
            "spawn_reason",
            "blocks_parent",
            "expected_resume",
            "step_bucket",
        )
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "return_time_bucket_bounds_seconds": [2.0, 3.0, 8.0],
                "output_length_bucket_bounds_tokens": [64],
                "return_time_profiles": [
                    {
                        "key": global_key,
                        "bucket_counts": [0.0, 100.0, 0.0, 0.0],
                        "no_return_count": 0.0,
                        "max_observed_seconds": 3.0,
                    }
                ],
                "output_length_profiles": [],
            }
        ),
        encoding="utf-8",
    )


def _run(router_bin: Path, output_dir: Path, shadow: bool) -> dict[str, Any]:
    name = "shadow" if shadow else "off"
    backend = Backend()
    backend_port, router_port, metrics_port = (_free_port() for _ in range(3))
    server = ThreadingHTTPServer(("127.0.0.1", backend_port), _handler(backend))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    profile_path = output_dir / "profile.json"
    _write_profile(profile_path)
    config = {
        "metrics_interval_seconds": TICK_SECONDS,
        # The mock's waiting gauge must only reach the shadow, not hold back admission.
        "admission_waiting_request_threshold": 1000,
        "prediction_profile": {
            "mode": "shadow",
            "profile_path": str(profile_path.resolve()),
            "return_time_bucket_bounds_seconds": [2.0, 3.0, 8.0],
            "output_length_bucket_bounds_tokens": [64],
            "return_time_min_support": 1,
        },
        "next_turn_prefetch": SHADOW_CONFIG if shadow else {"mode": "off"},
    }
    log_path = output_dir / f"{name}-router.log"
    with log_path.open("w", encoding="utf-8") as log:
        router = subprocess.Popen(
            [
                str(router_bin.resolve()),
                "--host",
                "127.0.0.1",
                "--port",
                str(router_port),
                "--worker-urls",
                f"http://127.0.0.1:{backend_port}",
                "--policy",
                "round_robin",
                "--enable-program-scheduling",
                "--program-scheduling-config-json",
                json.dumps(config, separators=(",", ":")),
                "--prometheus-host",
                "127.0.0.1",
                "--prometheus-port",
                str(metrics_port),
                "--log-level",
                "debug",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_ready(router_port, router, log_path)
            statuses = []
            for scenario in SCENARIOS:
                backend.configure(**scenario.backend)
                time.sleep(CONFIGURE_SECONDS if scenario.backend else SETTLE_SECONDS)
                statuses.append(_turn(router_port, scenario, True, PREFIX_TOKENS))
                time.sleep(scenario.wait_seconds)
                statuses.append(_turn(router_port, scenario, False, scenario.next_cached_tokens))
            time.sleep(SETTLE_SECONDS)
            diagnostics = json.loads(_get(f"http://127.0.0.1:{router_port}/scheduling/diagnostics"))
            metrics = _get(f"http://127.0.0.1:{metrics_port}/metrics")
        finally:
            router.terminate()
            try:
                router.wait(timeout=10)
            except subprocess.TimeoutExpired:
                router.kill()
                router.wait(timeout=5)
            server.shutdown()
            server.server_close()
    return {
        "statuses": statuses,
        "diagnostics": diagnostics,
        "metrics": metrics,
        "chat_bodies": backend.chat_bodies,
        "backend_requests": backend.requests,
    }


def _wait_ready(port: int, process: subprocess.Popen[Any], log_path: Path) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Router exited early; see {log_path}")
        try:
            _get(f"http://127.0.0.1:{port}/health")
            return
        except urllib.error.URLError:
            time.sleep(0.1)
    raise TimeoutError(f"Router did not become ready; see {log_path}")


def _metric(metrics: str, name: str, **labels: str) -> float:
    total = 0.0
    pattern = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)$")
    for line in metrics.splitlines():
        match = pattern.match(line)
        if not match or match.group(1) != name:
            continue
        found = dict(re.findall(r'(\w+)="([^"]*)"', match.group(2) or ""))
        if all(found.get(key) == value for key, value in labels.items()):
            total += float(match.group(3))
    return total


def _sorted_bodies(bodies: list[dict[str, Any]]) -> list[str]:
    return sorted(json.dumps(body, sort_keys=True) for body in bodies)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-bin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shadow = _run(args.router_bin, args.output_dir, shadow=True)
    off = _run(args.router_bin, args.output_dir, shadow=False)

    expected: Counter[str] = Counter(scenario.expected for scenario in SCENARIOS)
    rank = shadow["diagnostics"]["ranks"][0]["next_turn_prefetch"]
    observed = Counter(
        {f"outcome:{key}": value for key, value in rank["outcomes"].items()}
        | {f"skip:{key}": value for key, value in rank["skips"].items()}
    )
    plans = sum(rank["plans"].values())
    metrics = shadow["metrics"]
    checks = {
        "every_scenario_scored_as_expected": observed == expected,
        "every_plan_has_one_outcome": plans == sum(rank["outcomes"].values()),
        "every_settled_plan_gave_its_budget_back": rank["speculative_tokens"] == 0,
        "both_forecast_sources_planned": set(rank["plans"]) == {"hint", "profile"},
        "every_request_served": set(shadow["statuses"] + off["statuses"]) == {200},
        "backend_saw_the_same_requests_with_the_shadow_off": _sorted_bodies(shadow["chat_bodies"])
        == _sorted_bodies(off["chat_bodies"]),
        "backend_saw_only_the_turns_and_scrapes": {
            path
            for path in shadow["backend_requests"]
            if path.startswith("POST") or "prefetch" in path or "kv" in path
        }
        == {"POST /v1/chat/completions"},
        "off_keeps_no_ledger": "next_turn_prefetch" not in off["diagnostics"]["ranks"][0],
        "off_reports_its_mode": off["diagnostics"]["next_turn_prefetch_mode"] == "off",
        "metrics_match_the_diagnostics": all(
            _metric(metrics, "vllm_router_next_turn_prefetch_shadow_outcomes_total", outcome=key)
            == value
            for key, value in rank["outcomes"].items()
        )
        and all(
            _metric(metrics, "vllm_router_next_turn_prefetch_shadow_skips_total", reason=key)
            == value
            for key, value in rank["skips"].items()
        ),
        "unsupported_hint_counted": _metric(
            metrics, "vllm_router_next_turn_hint_ignored_total", reason="unsupported_version"
        )
        == 1,
        "off_emits_no_shadow_metric": "next_turn_prefetch_shadow" not in off["metrics"],
        "margins_recorded_for_settled_plans": _metric(
            metrics, "vllm_router_next_turn_prefetch_shadow_margin_seconds_count", outcome="in_time"
        )
        == rank["outcomes"]["in_time"]
        and _metric(metrics, "vllm_router_next_turn_prefetch_shadow_margin_seconds_count", outcome="late")
        == rank["outcomes"]["late"],
        "evaluation_time_recorded": _metric(
            metrics, "vllm_router_next_turn_prefetch_shadow_evaluation_seconds_count"
        )
        > 0,
    }
    report = {
        "checks": checks,
        "expected": dict(sorted(expected.items())),
        "observed": dict(sorted(observed.items())),
        "plans": rank["plans"],
        "outcome_tokens": rank["outcome_tokens"],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
