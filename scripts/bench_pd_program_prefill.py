"""Two-turn 2P1D placement experiment against isolated, already-running engines."""

import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from bench_session_admission import port, request, reset, service


def completion(url, body):
    req = urllib.request.Request(
        url + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error
    assert result["choices"], result
    return result["usage"]


def counters(metrics, stage, urls):
    text = request(metrics).decode()
    rows = re.findall(
        rf'^vllm_router_pd_{stage}_requests_total\{{[^}}]*worker="([^"]+)"[^}}]*\}} ([\d.e+\-]+)$',
        text,
        re.MULTILINE,
    )
    return [sum(float(n) for key, n in rows if url.split("://")[-1] == key.split("://")[-1]) for url in urls]


def run_arm(args, policy, arm, reset_p=False, warm_read=False):
    if args.preflight:
        with (args.output / f"preflight-{arm}.log").open("x") as log:
            subprocess.run([args.preflight], stdout=log, stderr=subprocess.STDOUT, check=True)
    listen, metrics = port(), port()
    config = {
        "binding_strategy": "consistent_hash",
        "enable_batch_gain_admission": False,
        "token_capacity_per_dp_rank": 8192,
        "metrics_interval_seconds": 0.1,
        "queue_timeout_seconds": 15,
        "force_resume_timeout_seconds": 5,
        "token_count_mode": "estimated",
        "pd": {"max_prefill_requests_per_target": 4},
    }
    if warm_read:
        config["pd"].update(warm_remote=True, feedback=True, explicit_ttl=True)
    cmd = [args.router, "--host", "127.0.0.1", "--port", str(listen),
           "--prometheus-host", "127.0.0.1", "--prometheus-port", str(metrics),
           "--vllm-pd-disaggregation", "--kv-connector", "nixl",
           "--prefill", args.p[0], "--prefill", args.p[1], "--decode", args.d,
           "--policy", "round_robin", "--prefill-policy", policy,
           "--enable-program-scheduling", "--program-scheduling-config-json", json.dumps(config)]
    if policy == "kv_aware":
        for url, pub, replay in zip(args.p, args.publishers, args.replays):
            cmd.extend(["--pd-kv-events", f"{url}={pub}", "--kv-event-replay-source",
                        f"{url}={replay}", "--kv-event-topic", f"{url}={args.topic}"])
        cmd.extend(["--kv-event-model", args.model, "--kv-hybrid-layout", str(args.layout)])
    entry = f"http://127.0.0.1:{listen}"
    prom = f"http://127.0.0.1:{metrics}/metrics"
    rows = []
    with service(cmd, args.output / f"router-{arm}.log", entry,
                 dict(os.environ, RUST_LOG="info,vllm_router_rs::routers::http::pd_router=debug")):
        time.sleep(2)
        for case in range(args.sessions):
            seeded = case % 2
            prefix = [1280 + arm * 500 + case] * 512
            base = {"model": args.model, "max_tokens": 1, "temperature": 0}
            completion(args.p[seeded], dict(base, prompt=prefix))
            time.sleep(0.5)
            for turn in (1, 2):
                if turn == 2 and reset_p:
                    reset(args.p)
                    time.sleep(0.5)
                body = dict(base, prompt=prefix + [712] * (turn * 8),
                            vllm_xargs={"agentic_context": json.dumps({
                                "program_id": f"case-{arm}-{case}", "kv_retention_ttl_seconds": 30,
                                "expected_resume": turn == 1})})
                if reset_p:
                    body["return_token_ids"] = True
                before = counters(prom, "prefill", args.p)
                d_before = counters(prom, "decode", [args.d])[0]
                start = time.monotonic()
                usage = completion(entry, body)
                elapsed = time.monotonic() - start
                after = counters(prom, "prefill", args.p)
                delta = [b - a for a, b in zip(before, after)]
                if sorted(delta) != [0, 1]:
                    (args.output / f"metrics-error-{arm}.txt").write_bytes(request(prom))
                    raise AssertionError(f"expected one P dispatch, got {delta}")
                assert counters(prom, "decode", [args.d])[0] - d_before == 1
                rows.append(dict(policy=policy, case=case, turn=turn, seeded=seeded, chosen=delta.index(1),
                                 cached=usage.get("prompt_tokens_details", {}).get("cached_tokens"),
                                 seconds=elapsed, prompt_tokens=usage["prompt_tokens"]))
        if warm_read:
            info = json.loads(request(entry + "/get_loads"))["program_scheduling"]["warm_remote"]
            (args.output / "warm-feedback.json").write_text(json.dumps(info, indent=2))
            assert info["counters"].get("warm_external_read", 0) >= args.sessions, info
        metrics_text = request(prom).decode()
        (args.output / f"metrics-{arm}.txt").write_text(metrics_text)
        admitted = sum(float(n) for n in re.findall(
            r'^vllm_router_agent_aware_queue_wait_seconds_count(?:\{[^}]*\})? ([\d.e+\-]+)$',
            metrics_text, re.MULTILINE))
        assert admitted == len(rows), (admitted, len(rows))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("router", "model", "d", "topic"):
        parser.add_argument("--" + option, required=True)
    for option in ("p", "publishers", "replays"):
        parser.add_argument("--" + option, nargs=2, required=True)
    for option in ("layout", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--preflight")
    parser.add_argument("--verify-warm", action="store_true", help="also test retained D KV with feedback-capable engines")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    all_rows = []
    for arm, policy in enumerate(("round_robin", "kv_aware")):
        rows = run_arm(args, policy, arm)
        (args.output / f"{policy}.json").write_text(json.dumps(rows, indent=2))
        all_rows.extend(rows)
    warm = [row for row in all_rows if row["policy"] == "kv_aware" and row["turn"] == 1]
    assert all(row["chosen"] == row["seeded"] and (row["cached"] or 0) >= 496 for row in warm), warm
    if args.verify_warm:
        for arm, warm_read in [(2, False), (3, True)]:
            rows = run_arm(args, "kv_aware", arm, reset_p=True, warm_read=warm_read)
            label = "warm-on" if warm_read else "warm-off"
            (args.output / f"{label}.json").write_text(json.dumps(rows, indent=2))
