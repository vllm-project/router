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


def completion(url, body, route):
    req = urllib.request.Request(
        url + route,
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
    return [
        sum(float(n) for key, n in rows if url.split("://")[-1] == key.split("://")[-1])
        for url in urls
    ]


def turn_input(arm, case, turn, chat):
    if not chat:
        return {"prompt": [1280 + arm * 500 + case] * 512 + [712] * (turn * 8)}
    text = f"Case {arm}-{case}. " + "Review the following source code. " * 72
    messages = [{"role": "user", "content": text}]
    if turn == 2:
        messages.extend(
            [
                {"role": "assistant", "content": "OK."},
                {
                    "role": "user",
                    "content": "The test failed. Check the same code again.",
                },
            ]
        )
    return {"messages": messages}


def run_arm(args, policy, arm, reset_p=False, warm_read=False):
    if args.preflight:
        with (args.output / f"preflight-{arm}.log").open("x") as log:
            subprocess.run(
                [args.preflight], stdout=log, stderr=subprocess.STDOUT, check=True
            )
    listen, metrics = port(), port()
    config = {
        "enable_batch_gain_admission": False,
        "token_capacity_per_dp_rank": args.decode_capacity,
        "metrics_interval_seconds": 0.1,
        "queue_timeout_seconds": 15,
        "force_resume_timeout_seconds": 5,
        "token_count_mode": "estimated",
        "pd": {"max_prefill_requests_per_target": 4},
    }
    if warm_read:
        config["pd"].update(warm_remote=True, feedback=True, explicit_ttl=True)
    cmd = [
        args.router,
        "--host",
        "127.0.0.1",
        "--port",
        str(listen),
        "--prometheus-host",
        "127.0.0.1",
        "--prometheus-port",
        str(metrics),
        "--vllm-pd-disaggregation",
        "--kv-connector",
        "nixl",
        "--prefill",
        args.p[0],
        "--prefill",
        args.p[1],
        "--decode",
        args.d,
        "--policy",
        json.dumps(
            {"default": "round_robin", "program": "consistent_hash", "prefill": policy}
        ),
        "--enable-program-scheduling",
        "--program-scheduling-config-json",
        json.dumps(config),
    ]
    if policy == "kv_aware":
        for url, pub, replay in zip(args.p, args.publishers, args.replays):
            cmd.extend(
                [
                    "--pd-kv-events",
                    f"{url}={pub}",
                    "--kv-event-replay-source",
                    f"{url}={replay}",
                    "--kv-event-topic",
                    f"{url}={args.topic}",
                ]
            )
        cmd.extend(
            ["--kv-event-model", args.model, "--kv-hybrid-layout", str(args.layout)]
        )
    entry = f"http://127.0.0.1:{listen}"
    prom = f"http://127.0.0.1:{metrics}/metrics"
    rows = []
    with service(
        cmd,
        args.output / f"router-{arm}.log",
        entry,
        dict(
            os.environ, RUST_LOG="info,vllm_router_rs::routers::http::pd_router=debug"
        ),
    ):
        time.sleep(2)
        for case in range(args.sessions):
            seeded = case % 2
            route = "/v1/chat/completions" if args.chat else "/v1/completions"
            base = {"model": args.model, "max_tokens": 1, "temperature": 0}
            completion(
                args.p[seeded], {**base, **turn_input(arm, case, 0, args.chat)}, route
            )
            time.sleep(0.5)
            for turn in (1, 2):
                if turn == 2 and reset_p:
                    reset(args.p)
                    time.sleep(0.5)
                context = {
                    "program_id": f"case-{arm}-{case}",
                    "kv_retention_ttl_seconds": 30,
                    "expected_resume": turn == 1,
                }
                # Chat commonly supplies an object. The backend wire adapter also accepts
                # an already-encoded agentic_context.
                body = {
                    **base,
                    **turn_input(arm, case, turn, args.chat),
                    "vllm_xargs": {
                        "agentic_context": context
                        if args.chat and policy == "kv_aware"
                        else json.dumps(context)
                    },
                }
                if reset_p:
                    body["return_token_ids"] = True
                before = counters(prom, "prefill", args.p)
                d_before = counters(prom, "decode", [args.d])[0]
                start = time.monotonic()
                try:
                    usage = completion(entry, body, route)
                except (RuntimeError, OSError):
                    (args.output / f"error-loads-{arm}.json").write_bytes(
                        request(entry + "/get_loads")
                    )
                    raise
                elapsed = time.monotonic() - start
                after = counters(prom, "prefill", args.p)
                delta = [b - a for a, b in zip(before, after)]
                if sorted(delta) != [0, 1]:
                    (args.output / f"metrics-error-{arm}.txt").write_bytes(
                        request(prom)
                    )
                    raise AssertionError(f"expected one P dispatch, got {delta}")
                assert counters(prom, "decode", [args.d])[0] - d_before == 1
                rows.append(
                    dict(
                        policy=policy,
                        case=case,
                        turn=turn,
                        seeded=seeded,
                        chosen=delta.index(1),
                        cached=usage.get("prompt_tokens_details", {}).get(
                            "cached_tokens"
                        ),
                        seconds=elapsed,
                        prompt_tokens=usage["prompt_tokens"],
                    )
                )
        (args.output / f"turns-{arm}.json").write_text(json.dumps(rows, indent=2))
        if warm_read:
            info = json.loads(request(entry + "/get_loads"))["program_scheduling"][
                "warm_remote"
            ]
            (args.output / "warm-feedback.json").write_text(json.dumps(info, indent=2))
            assert info["counters"].get("warm_external_read", 0) >= args.sessions, info
        metrics_text = request(prom).decode()
        (args.output / f"metrics-{arm}.txt").write_text(metrics_text)
        admitted = sum(
            float(n)
            for n in re.findall(
                r"^vllm_router_agent_aware_queue_wait_seconds_count(?:\{[^}]*\})? ([\d.e+\-]+)$",
                metrics_text,
                re.MULTILINE,
            )
        )
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
    parser.add_argument(
        "--decode-capacity",
        type=int,
        required=True,
        help="actual usable KV tokens per D rank",
    )
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--preflight")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="exercise server-side chat rendering and two message turns",
    )
    parser.add_argument(
        "--verify-warm",
        action="store_true",
        help="also test retained D KV with feedback-capable engines",
    )
    args = parser.parse_args()
    if args.chat and args.verify_warm:
        parser.error("warm verification requires exact token-ID completions")
    args.output.mkdir(exist_ok=False)
    all_rows = []
    for arm, policy in enumerate(("round_robin", "kv_aware")):
        rows = run_arm(args, policy, arm)
        (args.output / f"{policy}.json").write_text(json.dumps(rows, indent=2))
        all_rows.extend(rows)
    warm = [row for row in all_rows if row["policy"] == "kv_aware" and row["turn"] == 1]
    assert all(
        row["chosen"] == row["seeded"]
        and (row["cached"] or 0) >= (128 if args.chat else 496)
        for row in warm
    ), warm
    if args.verify_warm:
        for arm, warm_read in [(2, False), (3, True)]:
            rows = run_arm(args, "kv_aware", arm, reset_p=True, warm_read=warm_read)
            label = "warm-on" if warm_read else "warm-off"
            (args.output / f"{label}.json").write_text(json.dumps(rows, indent=2))
