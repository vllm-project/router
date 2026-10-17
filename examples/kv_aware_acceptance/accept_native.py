#!/usr/bin/env python3
"""Real-engine HTTP acceptance and paired prefix-placement measurements."""

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import statistics
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

import httpx
import msgpack
import zmq
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model", default="router-qwen3-8b")
    parser.add_argument("--workers", nargs=2, required=True)
    parser.add_argument("--publishers", nargs=2, required=True)
    parser.add_argument("--kv-router", required=True)
    parser.add_argument("--baseline-router", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--contexts", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--expect-unreconciled", action="store_true")
    parser.add_argument("--verify-stale-after-seconds", type=float, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "passed": False,
        "checks": [],
        "arms": [],
        "events": [],
        "args": vars(args) | {"output": str(args.output)},
    }
    client = httpx.Client(timeout=180, trust_env=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    stop = threading.Event()

    def save():
        temporary = args.output / "result.tmp"
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(args.output / "result.json")

    def check(name, condition, **evidence):
        result["checks"].append({"name": name, "passed": bool(condition), **evidence})
        save()
        assert condition, name

    def observe(index, endpoint):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b"router-kv-real")
        socket.setsockopt(zmq.RCVHWM, 10000)
        socket.connect(endpoint)
        try:
            while not stop.is_set():
                if not socket.poll(100):
                    continue
                frames = socket.recv_multipart()
                if len(frames) != 3:
                    result["events"].append({"worker": index, "error": "frame_count"})
                    continue
                sequence = int.from_bytes(frames[1], "big", signed=True)
                batch = msgpack.unpackb(frames[2], raw=False, strict_map_key=False)
                filename = f"worker-{index}-batch-{sequence}.msgpack"
                (args.output / filename).write_bytes(frames[2])
                events = batch[1]
                result["events"].append(
                    {
                        "worker": index,
                        "sequence": sequence,
                        "timestamp": batch[0],
                        "rank": batch[2] if len(batch) > 2 else None,
                        "types": dict(Counter(event["type"] for event in events)),
                        "payload": filename,
                        "sha256": hashlib.sha256(frames[2]).hexdigest(),
                    }
                )
        except Exception:
            result["events"].append({"worker": index, "error": traceback.format_exc()})
        finally:
            socket.close(0)
            context.term()

    threads = [
        threading.Thread(target=observe, args=(i, endpoint), daemon=True)
        for i, endpoint in enumerate(args.publishers)
    ]
    for thread in threads:
        thread.start()

    def tokens(question, context=""):
        return tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "Reply with the exact answer only. Do not explain.",
                },
                {"role": "user", "content": context + "\n" + question},
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )

    def body(ids, count=1):
        return {
            "model": args.model,
            "prompt": ids,
            "temperature": 0,
            "seed": 20261015,
            "max_tokens": count,
            "return_token_ids": True,
        }

    def complete(base, payload):
        response = client.post(base + "/v1/completions", json=payload)
        response.raise_for_status()
        value = response.json()
        return {
            "mode": response.headers.get("x-router-kv-mode"),
            "reason": response.headers.get("x-router-kv-reason"),
            "body": value,
        }

    def stream(base, payload):
        start = time.perf_counter()
        first = None
        text = []
        output_ids = []
        with client.stream(
            "POST",
            base + "/v1/completions",
            json=payload | {"stream": True, "stream_options": {"include_usage": True}},
        ) as response:
            response.raise_for_status()
            mode, reason = (
                response.headers.get("x-router-kv-mode"),
                response.headers.get("x-router-kv-reason"),
            )
            usage = None
            for line in response.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                item = json.loads(line[6:])
                if item.get("error"):
                    raise RuntimeError(item["error"])
                for choice in item.get("choices", []):
                    piece = choice.get("text") or ""
                    ids = choice.get("token_ids") or []
                    if piece or ids:
                        if first is None:
                            first = time.perf_counter() - start
                        text.append(piece)
                        output_ids.extend(ids)
                usage = item.get("usage") or usage
        assert first is not None, "stream produced no generated token"
        return {
            "ttft_seconds": first,
            "elapsed_seconds": time.perf_counter() - start,
            "mode": mode,
            "reason": reason,
            "text": "".join(text),
            "token_ids": output_ids,
            "usage": usage,
        }

    def metrics():
        counters = []
        for worker in args.workers:
            response = client.get(worker + "/metrics")
            response.raise_for_status()
            values = {}
            for line in response.text.splitlines():
                if line.startswith("#"):
                    continue
                match = re.match(
                    r"(vllm:(?:prefix_cache_hits_total|prefix_cache_queries_total|request_success_total))(?:\{.*\})?\s+([0-9.eE+-]+)$",
                    line,
                )
                if match:
                    name, value = match.groups()
                    values[name] = values.get(name, 0) + float(value)
            counters.append(values)
        return counters

    small = tokens("What is 2 + 2?")

    def clear():
        for worker in args.workers:
            response = client.post(
                worker + "/reset_prefix_cache?reset_running_requests=true"
            )
            response.raise_for_status()
            complete(worker, body(small))
        time.sleep(0.4)

    try:
        time.sleep(1)
        for worker in args.workers:
            info = client.get(worker + "/get_server_info").json()
            check(
                "live_engine_discovery",
                info["kv_block_size"] > 0,
                worker=worker,
                info=info,
            )
        prefix = tokens(
            "What is 2 + 2?", "Reference: two plus two equals four.\n" * 900
        )
        check(
            "long_context_in_model_limit",
            4096 < len(prefix) < 16000,
            tokens=len(prefix),
        )
        if args.expect_unreconciled:
            initial = complete(args.kv_router, body(prefix))
            check(
                "unreconciled_long_prompt_falls_back",
                initial["mode"] == "fallback" and initial["reason"] == "unreconciled",
                response=initial,
            )
        clear()
        warm = complete(args.workers[1], body(prefix, 16))
        time.sleep(0.4)
        before = metrics()
        routed = complete(args.kv_router, body(prefix, 16))
        after = metrics()
        hits = [
            after[i].get("vllm:prefix_cache_hits_total", 0)
            - before[i].get("vllm:prefix_cache_hits_total", 0)
            for i in (0, 1)
        ]
        check(
            "exact_routing_to_second_resident_worker",
            routed["mode"] == "exact" and hits[1] >= 4096 and hits[0] == 0,
            response=routed,
            before=before,
            after=after,
            hit_delta=hits,
        )
        check(
            "greedy_token_ids_match_direct",
            warm["body"]["choices"][0].get("token_ids") is not None
            and warm["body"]["choices"][0]["token_ids"]
            == routed["body"]["choices"][0].get("token_ids"),
            direct=warm,
        )
        clear()
        before = metrics()
        cleared = complete(args.kv_router, body(prefix, 1))
        after = metrics()
        hits = sum(
            after[i].get("vllm:prefix_cache_hits_total", 0)
            - before[i].get("vllm:prefix_cache_hits_total", 0)
            for i in (0, 1)
        )
        check(
            "native_clear_removes_prefix_credit",
            cleared["mode"] == "exact" and hits < 256,
            response=cleared,
            hit_delta=hits,
        )

        hint = {
            "session_id": "real-agent-root",
            "blocks_parent": False,
            "agent_role": "researcher",
            "step_id": 1,
            "request_id": "real-agent-request",
            "priority": 3,
            "deadline_seconds": 30,
            "expected_output_tokens": 16,
            "kv_retention_ttl_seconds": 2,
        }
        hinted = complete(args.kv_router, body(small, 16) | {"agent_hint": hint})
        check(
            "agent_hint_real_generation",
            bool(hinted["body"]["choices"][0].get("text")),
            response=hinted,
        )
        bad = client.post(
            args.kv_router + "/v1/completions",
            json=body(small) | {"agent_hint": hint | {"priority": "urgent"}},
        )
        check(
            "malformed_agent_hint_rejected",
            bad.status_code == 400,
            status=bad.status_code,
            body=bad.text,
        )

        golden = []
        for question, expected in [
            ("What is 2 + 2?", "4"),
            ("What is 7 - 3?", "4"),
            ("What is 3 * 3?", "9"),
        ]:
            payload = body(tokens(question), 16)
            direct = complete(args.workers[0], payload)
            route = complete(args.kv_router, payload)
            choice = route["body"]["choices"][0]
            golden.append(
                {
                    "question": question,
                    "expected": expected,
                    "response": route,
                    "direct": direct,
                }
            )
            check(
                "known_answer_and_token_parity",
                choice.get("text", "").strip() == expected
                and choice.get("token_ids")
                == direct["body"]["choices"][0].get("token_ids"),
                case=golden[-1],
            )

        if args.verify_stale_after_seconds:
            time.sleep(args.verify_stale_after_seconds)
            stale = complete(args.kv_router, body(prefix))
            check(
                "idle_event_source_falls_back_after_expiry",
                stale["mode"] == "fallback" and stale["reason"] == "stale",
                response=stale,
            )
            clear()
            restored = complete(args.kv_router, body(prefix))
            check(
                "clear_barrier_restores_exact_mode",
                restored["mode"] == "exact",
                response=restored,
            )

        workload = [
            tokens(
                "What is 2 + 2?",
                f"Document {i:04d}.\n" + "Reference: two plus two equals four.\n" * 900,
            )
            for i in range(args.contexts)
        ]
        (args.output / "workload.json").write_text(json.dumps(workload) + "\n")
        order = list(range(args.contexts))
        random.Random(20261015).shuffle(order)
        for round_index in range(args.rounds):
            arms = [("round_robin", args.baseline_router), ("kv_aware", args.kv_router)]
            if round_index % 2:
                arms.reverse()
            for policy, base in arms:
                clear()
                warm_outputs = []
                for index, ids in enumerate(workload):
                    output = complete(args.workers[index % 2], body(ids))
                    warm_outputs.append(output["body"]["choices"][0])
                time.sleep(0.4)
                before = metrics()
                start = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=args.concurrency
                ) as executor:
                    outputs = list(
                        executor.map(
                            lambda index: stream(base, body(workload[index]))
                            | {"case": index},
                            order,
                        )
                    )
                elapsed = time.perf_counter() - start
                after = metrics()
                arm = {
                    "round": round_index,
                    "policy": policy,
                    "elapsed_seconds": elapsed,
                    "requests_per_second": len(outputs) / elapsed,
                    "before": before,
                    "after": after,
                    "outputs": outputs,
                }
                result["arms"].append(arm)
                save()
                check(
                    "benchmark_output_parity",
                    all(
                        output["text"] == warm_outputs[output["case"]]["text"]
                        and output["token_ids"]
                        == warm_outputs[output["case"]].get("token_ids", [])
                        for output in outputs
                    ),
                    round=round_index,
                    policy=policy,
                )
                if policy == "kv_aware":
                    check(
                        "all_measured_kv_requests_exact",
                        all(output["mode"] == "exact" for output in outputs),
                        round=round_index,
                    )
                print(
                    json.dumps(
                        {
                            "round": round_index,
                            "policy": policy,
                            "completed": len(outputs),
                        }
                    ),
                    flush=True,
                )
        for policy in ("round_robin", "kv_aware"):
            arms = [arm for arm in result["arms"] if arm["policy"] == policy]
            if not arms:
                continue
            outputs = [output for arm in arms for output in arm["outputs"]]
            times = sorted(output["ttft_seconds"] for output in outputs)
            queries = sum(
                arm["after"][i].get("vllm:prefix_cache_queries_total", 0)
                - arm["before"][i].get("vllm:prefix_cache_queries_total", 0)
                for arm in arms
                for i in (0, 1)
            )
            hits = sum(
                arm["after"][i].get("vllm:prefix_cache_hits_total", 0)
                - arm["before"][i].get("vllm:prefix_cache_hits_total", 0)
                for arm in arms
                for i in (0, 1)
            )
            result.setdefault("summary", {})[policy] = {
                "requests": len(outputs),
                "ttft_p50_seconds": statistics.median(times),
                "ttft_p95_seconds": times[min(len(times) - 1, int(len(times) * 0.95))],
                "requests_per_second": len(outputs)
                / sum(arm["elapsed_seconds"] for arm in arms),
                "native_cached_tokens": hits,
                "native_queried_tokens": queries,
                "native_hit_fraction": hits / queries if queries else None,
            }
        types = [name for event in result["events"] for name in event.get("types", {})]
        check(
            "native_clear_and_store_observed",
            "AllBlocksCleared" in types and "BlockStored" in types,
        )
        result["passed"] = True
    except Exception:
        result["failure"] = traceback.format_exc()
        raise
    finally:
        stop.set()
        for thread in threads:
            thread.join(2)
        save()
        client.close()


if __name__ == "__main__":
    main()
