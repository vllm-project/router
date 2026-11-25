#!/usr/bin/env python3
"""Replay held-out AgentBench request facts through Router and a usage-only backend."""

from __future__ import annotations

import argparse
import json
import random
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
WORKFLOW_PROFILE_ID = "coding_assistant"
PROFILE_VERSION = "pythia-min-adaptive-v1"
OUTPUT_BOUNDS = list(range(16, 2049, 16))


class ReplayBackendHandler(BaseHTTPRequestHandler):
    model = MODEL

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            self._json({"object": "list", "data": [{"id": self.model, "object": "model"}]})
        elif self.path == "/get_model_info":
            self._json({"model_path": self.model, "model_id": self.model})
        else:
            self._json({"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        xargs = request.get("vllm_xargs") or {}
        prompt_tokens = int(xargs.get("replay_prompt_tokens", 1))
        completion_tokens = int(xargs.get("replay_completion_tokens", 1))
        self._json(
            {
                "id": "replay",
                "object": "chat.completion",
                "model": request.get("model", self.model),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "replayed"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )

    def _json(self, value: object, status: int = 200) -> None:
        encoded = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-bin", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--offline-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("seeded", "online"), required=True)
    args = parser.parse_args()

    facts = _held_out_facts(args.requests)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = _run(args, facts)
    destination = args.output_dir / f"{args.mode}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def _run(args: argparse.Namespace, facts: list[dict[str, Any]]) -> dict[str, Any]:
    backend_port, router_port, metrics_port = (_free_port() for _ in range(3))
    backend = ThreadingHTTPServer(("127.0.0.1", backend_port), ReplayBackendHandler)
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()

    snapshot_path = (args.output_dir / f"{args.mode}-online-overlay.json").resolve()
    snapshot_path.unlink(missing_ok=True)
    config = {
        "program_scheduling_enable_key": "vllm_xargs.agentic_context",
        "metrics_interval_seconds": 0.1,
        "prediction_profile": {
            "mode": "shadow",
            "profile_path": str(args.offline_profile.resolve()) if args.mode == "seeded" else None,
            "snapshot_path": str(snapshot_path),
            "snapshot_interval_seconds": 0.2,
            "prior_weight": 1.0,
            "max_keys": 4096,
            "return_time_bucket_bounds_seconds": [1.0, 2.0, 5.0, 10.0],
            "return_time_min_support": 20,
            "return_time_horizon_seconds": 5.0,
            "step_bucket_size": 4,
            "output_length_bucket_bounds_tokens": OUTPUT_BOUNDS,
            "output_length_quantile": 0.95,
            "output_length_min_support": 50,
            "output_length_tail_samples": 3,
            "stability_window_size": 100,
            "stability_tolerance": 0.15,
        },
    }
    log_path = args.output_dir / f"{args.mode}-router.log"
    with log_path.open("w", encoding="utf-8") as log:
        router = subprocess.Popen(
            [
                str(args.router_bin.resolve()),
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
            failures = _replay(router_port, facts)
            metrics = _wait_for_metrics(metrics_port, len(facts))
            deadline = time.monotonic() + 5
            while not snapshot_path.exists() and time.monotonic() < deadline:
                time.sleep(0.1)
        finally:
            router.terminate()
            try:
                router.wait(timeout=10)
            except subprocess.TimeoutExpired:
                router.kill()
                router.wait(timeout=5)
            backend.shutdown()
            backend.server_close()
            backend_thread.join(timeout=5)

    observations = int(_metric_sum(metrics, "vllm_router_workflow_output_actual_tokens_count"))
    predictions = int(_metric_sum(metrics, "vllm_router_workflow_output_prediction_total"))
    coverage = {}
    for quantile in ("p95", "p99"):
        covered = _metric_sum(
            metrics,
            "vllm_router_workflow_output_coverage_total",
            {"quantile": quantile, "covered": "true"},
        )
        missed = _metric_sum(
            metrics,
            "vllm_router_workflow_output_coverage_total",
            {"quantile": quantile, "covered": "false"},
        )
        coverage[quantile] = covered / (covered + missed) if covered + missed else None

    source_counts = Counter()
    backoff_counts = Counter()
    for _, labels, value in _samples(metrics, "vllm_router_workflow_output_prediction_total"):
        source_counts[labels.get("source", "unknown")] += int(value)
        backoff_counts[labels.get("backoff_level", "unknown")] += int(value)

    snapshot_entries = 0
    if snapshot_path.exists():
        snapshot_entries = len(json.loads(snapshot_path.read_text())["output_length_profiles"])
    return {
        "mode": args.mode,
        "held_out_workflows": 30,
        "replayed_requests": len(facts),
        "http_failures": failures,
        "router_observations": observations,
        "router_predictions": predictions,
        "prediction_availability": predictions / observations if observations else 0.0,
        "coverage": coverage,
        "prediction_sources": dict(sorted(source_counts.items())),
        "backoff_levels": dict(sorted(backoff_counts.items())),
        "online_snapshot_entries": snapshot_entries,
        "profile_loaded": args.mode == "seeded",
        "log_path": str(log_path),
        "snapshot_path": str(snapshot_path),
    }


def _held_out_facts(path: Path) -> list[dict[str, Any]]:
    facts = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    workflow_ids = sorted({fact["session_id"] for fact in facts if fact.get("session_id")})
    random.Random(0).shuffle(workflow_ids)
    test_ids = set(workflow_ids[int(len(workflow_ids) * 0.7) :])
    selected = [
        fact
        for fact in facts
        if fact.get("session_id") in test_ids
        and isinstance(fact.get("output_tokens"), int)
        and fact["output_tokens"] >= 0
    ]
    if len(test_ids) != 30:
        raise RuntimeError(f"expected 30 held-out workflows, found {len(test_ids)}")
    return selected


def _replay(port: int, facts: list[dict[str, Any]]) -> int:
    failures = 0
    for index, fact in enumerate(facts):
        actor_id = fact["actor_id"]
        role = "explorer" if actor_id.startswith("explorer_") else actor_id
        context = {
            "program_id": f"{fact['session_id']}:{actor_id}",
            "task_id": fact["session_id"],
            "session_id": fact["session_id"],
            "agent_id": actor_id,
            "agent_role": fact["actor_role"],
            "spawn_reason": role,
            "blocks_parent": False,
            "expected_resume": True,
            "workflow_profile_id": WORKFLOW_PROFILE_ID,
            "profile_version": PROFILE_VERSION,
            "request_id": fact["request_id"],
        }
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "deterministic trace replay"}],
            "stream": False,
            "vllm_xargs": {
                "agentic_context": context,
                "replay_prompt_tokens": fact.get("input_tokens") or 1,
                "replay_completion_tokens": fact["output_tokens"],
            },
        }
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 200:
                    failures += 1
                response.read()
        except (urllib.error.URLError, TimeoutError):
            failures += 1
        if (index + 1) % 500 == 0:
            print(f"{index + 1}/{len(facts)} requests replayed", flush=True)
    return failures


def _wait_ready(port: int, process: subprocess.Popen[Any], log_path: Path) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Router exited early; see {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            time.sleep(0.1)
    raise TimeoutError(f"Router did not become ready; see {log_path}")


def _wait_for_metrics(port: int, expected: int) -> str:
    deadline = time.monotonic() + 10
    last = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as response:
                last = response.read().decode()
            if _metric_sum(last, "vllm_router_workflow_output_actual_tokens_count") >= expected:
                return last
        except urllib.error.URLError:
            pass
        time.sleep(0.1)
    return last


def _samples(metrics: str, name: str) -> list[tuple[str, dict[str, str], float]]:
    result = []
    pattern = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)$')
    for line in metrics.splitlines():
        match = pattern.match(line)
        if not match or match.group(1) != name:
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', match.group(2) or ""))
        result.append((match.group(1), labels, float(match.group(3))))
    return result


def _metric_sum(metrics: str, name: str, required_labels: dict[str, str] | None = None) -> float:
    return sum(
        value
        for _, labels, value in _samples(metrics, name)
        if not required_labels or all(labels.get(key) == expected for key, expected in required_labels.items())
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    main()
