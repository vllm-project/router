#!/usr/bin/env python3
"""Replay RFC 634 traffic against a baseline and candidate on two dedicated HTTP engines.

Both engines must allow prefix-cache reset and report streaming token usage. The KV
probe also needs a publisher on each engine. Run with --help for the arguments.
"""

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(url, method="GET"):
    with urllib.request.urlopen(
        urllib.request.Request(url, method=method), timeout=20
    ) as r:
        return r.read()


@contextlib.contextmanager
def service(cmd, log, url, env=None, timeout=120):
    with log.open("x") as stream:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                if proc.poll() is not None:
                    raise RuntimeError(f"startup exited {proc.returncode}: {log}")
                try:
                    request(url + "/health")
                    break
                except OSError:
                    time.sleep(0.25)
            else:
                raise TimeoutError(str(log))
            yield proc
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def chat(url, model, text, sid, output, hash_key, object_context=False):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        "max_tokens": output,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if sid is not None:
        ctx = {"session_id": sid, "expected_resume": False}
        body["vllm_xargs"] = {
            "agentic_context": ctx if object_context else json.dumps(ctx)
        }
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-User-ID": hash_key},
    )
    start = time.monotonic()
    first = None
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=30) as reply:
            for raw in reply:
                if not raw.startswith(b"data:"):
                    continue
                content = raw[5:].strip()
                if content == b"[DONE]":
                    break
                event = json.loads(content)
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if first is None and any(
                        delta.get(k)
                        for k in ("content", "reasoning", "reasoning_content")
                    ):
                        first = time.monotonic()
                usage = event.get("usage") or usage
    except urllib.error.HTTPError as error:
        return {
            "session": sid,
            "status": error.code,
            "error": error.read().decode()[:500],
        }
    if first is None or not usage.get("completion_tokens"):
        raise ValueError(f"missing streaming output or usage for {sid}: {usage}")
    return {
        "session": sid,
        "status": 200,
        "ttft_ms": 1000 * (first - start),
        "e2e_ms": 1000 * (time.monotonic() - start),
        "usage": usage,
    }


def metrics(url):
    result = {}
    for line in request(url).decode().splitlines():
        if line.startswith(("#", " ")):
            continue
        name, _, value = line.rpartition(" ")
        if any(
            key in name
            for key in (
                "admission_capacity_scan",
                "admission_eligible_ranks",
                "admission_selection",
                "agent_aware_queue_wait_seconds",
                "prefix_routing_decisions",
                "kv_placement_lookups",
                "kv_source_",
                "kv_routing_input",
            )
        ):
            result[name] = float(value)
    return result


def delta(before, after, name):
    return sum(
        v - before.get(k, 0) for k, v in after.items() if k.split("{")[0] == name
    )


def percentile(values, pct):
    return sorted(values)[math.ceil(len(values) * pct / 100) - 1]


def reset(workers):
    for worker in workers:
        request(worker + "/reset_prefix_cache", "POST")


@contextlib.contextmanager
def router(args, label, binary, strategy, workers):
    if args.preflight:
        with (args.output / (label + "-preflight.log")).open("x") as log:
            subprocess.run(
                [args.preflight], stdout=log, stderr=subprocess.STDOUT, check=True
            )
    listen, prom = port(), port()
    while listen == prom:
        prom = port()
    url = f"http://127.0.0.1:{listen}"
    config = {
        "binding_strategy": strategy,
        "max_active_programs_per_target": 1,
        "queue_timeout_seconds": 10,
        "force_resume_timeout_seconds": 10,
        "token_count_mode": "estimated",
        "metrics_interval_seconds": 0.1,
    }
    cmd = [
        binary,
        "--host",
        "127.0.0.1",
        "--port",
        str(listen),
        "--worker-urls",
        *workers,
        "--policy",
        "round_robin",
        "--prometheus-host",
        "127.0.0.1",
        "--prometheus-port",
        str(prom),
        "--enable-program-scheduling",
        "--program-scheduling-config-json",
        json.dumps(config),
    ]
    if strategy == "kv_aware":
        for worker, publisher in zip(workers, args.publishers):
            cmd.extend(["--kv-event-source", worker + "=" + publisher])
        if args.replay_endpoints:
            for worker, endpoint in zip(workers, args.replay_endpoints):
                cmd.extend(["--kv-event-replay-source", worker + "=" + endpoint])
        if args.event_topic:
            for worker in workers:
                cmd.extend(["--kv-event-topic", worker + "=" + args.event_topic])
        if args.kv_layout:
            cmd.extend(["--kv-hybrid-layout", str(args.kv_layout)])
        cmd.extend(
            [
                "--kv-event-model",
                args.model,
                "--kv-event-block-size",
                str(args.block_size),
            ]
        )
    with service(
        cmd,
        args.output / (label + ".log"),
        url,
        dict(os.environ, RUST_LOG="info,vllm_router_rs::program_scheduling=debug"),
    ):
        yield url, f"http://127.0.0.1:{prom}/metrics"


def measure(args, label, binary, concurrency, workers):
    reset(workers)
    prompt = (
        "Please read the following common reference. "
        + "A shared prefix for this workload. " * 80
    )
    with router(args, label, binary, "consistent_hash", workers) as (url, prom):
        warm = chat(url, args.model, prompt, "warm", 1, "pool-key")
        assert warm["status"] == 200, warm
        before = metrics(prom)
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
            rows = list(
                pool.map(
                    lambda i: chat(
                        url,
                        args.model,
                        prompt + f" Item {i}.",
                        f"s-{i}",
                        32,
                        "pool-key",
                    ),
                    range(args.requests),
                )
            )
        elapsed = time.monotonic() - start
        after = metrics(prom)
    good = [r for r in rows if r["status"] == 200]
    ttft = [r["ttft_ms"] for r in good]
    queued = delta(before, after, "vllm_router_agent_aware_queue_wait_seconds_count")
    result = {
        "label": label,
        "concurrency": concurrency,
        "requests": len(rows),
        "completed": len(good),
        "timeouts": sum(r["status"] == 429 for r in rows),
        "elapsed_s": elapsed,
        "requests_per_s": len(good) / elapsed,
        "output_tokens_per_s": sum(r["usage"]["completion_tokens"] for r in good)
        / elapsed,
        "cached_tokens": sum(
            r["usage"]["prompt_tokens_details"]["cached_tokens"] for r in good
        ),
        "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in good),
        "ttft_ms": {str(p): percentile(ttft, p) for p in (50, 95, 99)} if ttft else {},
        "mean_queue_ms": 1000
        * delta(before, after, "vllm_router_agent_aware_queue_wait_seconds_sum")
        / queued
        if queued
        else None,
        "scans": delta(before, after, "router_program_admission_capacity_scans_total"),
        "stale": sum(
            v - before.get(k, 0)
            for k, v in after.items()
            if "admission_selections_total" in k and 'result="stale"' in k
        ),
    }
    (args.output / (label + ".json")).write_text(
        json.dumps(
            dict(summary=result, rows=rows, before=before, after=after), indent=2
        )
    )
    if len(good) != len(rows):
        raise RuntimeError(f"failed benchmark requests: {result}")
    print(json.dumps(result), flush=True)
    return result


def kv_probe(args, workers):
    reset(workers)
    prompt = "Distinct KV evidence. " * 90
    with router(args, "kv-probe", args.candidate, "kv_aware", workers) as (url, prom):
        # Warm the lexicographically last worker directly. An equal-load tie would choose the first.
        # Allow the ZMQ subscription handshake before publishing the first block.
        time.sleep(1)
        seeded = chat(max(workers), args.model, prompt, None, 1, "warm")
        time.sleep(1)
        before = metrics(prom)
        measured = chat(
            url, args.model, prompt, "kv-session", 1, "test", object_context=True
        )
        after_warm = metrics(prom)
        spare = chat(url, args.model, "A different cold prompt.", "kv-cold", 1, "test")
        after = metrics(prom)
        result = dict(
            seed=seeded,
            warm=measured,
            cold=spare,
            metrics_before=before,
            metrics_after_warm=after_warm,
            metrics_after=after,
        )
        (args.output / "kv-probe.json").write_text(json.dumps(result, indent=2))
        assert seeded["status"] == measured["status"] == spare["status"] == 200, result
        assert spare["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, result
        assert any(
            v > before.get(k, 0)
            for k, v in after.items()
            if "session_binding" in k and "load_only" in k
        ), result
        assert measured["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, result
        assert any(
            v > before.get(k, 0)
            for k, v in after.items()
            if "session_binding" in k and "prefix_cost" in k
        ), result
    return "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "model"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--workers", nargs=2, required=True)
    parser.add_argument("--publishers", nargs=2)
    parser.add_argument("--replay-endpoints", nargs=2)
    parser.add_argument("--event-topic")
    parser.add_argument(
        "--kv-layout",
        type=Path,
        help="explicit worker layout for engines without discovery",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--kv-only", action="store_true", help="run only the warm/cold KV probe"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--preflight", help="optional host health snapshot executable")
    args = parser.parse_args()
    if args.kv_only and not args.publishers:
        parser.error("--kv-only needs both --publishers")
    digests = {
        arm: hashlib.sha256(Path(getattr(args, arm)).read_bytes()).hexdigest()
        for arm in ("baseline", "candidate")
    }
    if digests["baseline"] == digests["candidate"]:
        raise ValueError("A/B binaries have identical SHA-256")
    args.output.mkdir(exist_ok=False)
    (args.output / "binaries.json").write_text(json.dumps(digests, indent=2))
    if args.kv_only:
        kv_probe(args, args.workers)
        return
    summary = []
    for c in (1, 4):
        for i, arm in enumerate(("baseline", "candidate", "candidate", "baseline")):
            summary.append(
                measure(args, f"c{c}-{i}-{arm}", getattr(args, arm), c, args.workers)
            )
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.publishers:
        kv_probe(args, args.workers)


if __name__ == "__main__":
    main()
