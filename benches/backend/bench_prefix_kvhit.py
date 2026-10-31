#!/usr/bin/env python3
"""Prefix-cache miss vs hit against an already-running router.

Does not start engines or talk to Slurm/Docker. Point
``ROUTER_URL`` at a live OpenAI ``/v1/chat/completions`` endpoint (the
router, or a worker). Pair with one of the serve_*.sh scripts under
scripts/backend.

  export ROUTER_URL=http://127.0.0.1:30000
  export MODEL=my-served-model
  python benches/backend/bench_prefix_kvhit.py

By default the script writes a short synthetic chat body into a temp
file (repeated paragraph; no tokenizer required). Pass ``--tokens N``
and ``--model-dir /path/to/hf`` to size the user text to an exact
input length via transformers.

Returned JSON reports a concurrent hit-request wave by default. Pass
``--warmup`` to first send a serial miss/hit pair that primes prefix cache
state before the wave. Per-request rows include client-observed ``ttft_ms``,
``e2e_ms`` / ``total_duration_ms``, ``last_output_ms``, token counts,
throughput, and ``tpot_ms``. The optional ``stages`` field is raw router
diagnostic JSON from ``VLLM_ROUTER_STAGES=1``. Stage fields such as
``xfer_ms`` and ``engine_ms`` are boundary/residual diagnostics, not direct
network or EngineCore telemetry.

Hit rates (``--shape split``, the default; one user message):
  0.99 — same body twice (second request should prefix-hit)
  0.30 — shared prefix + unique tail
  0.00 — unique tag at the start of the second request

``--shape turns`` puts the shared part in earlier chat turns (user/assistant
pairs, ``--turns`` messages) and keeps only the last user message unique, like
a growing conversation. The router's L1 encoding cache can only reuse token
IDs up to a chat-template boundary, so ``split`` (which cuts inside one
message) measures worker KV reuse, while ``turns`` exercises both worker KV
reuse and L1. Compare a router started without and with
``VLLM_ROUTER_L1_CACHE=1`` (both with ``VLLM_ROUTER_STAGES=1``). ``turns``
never repeats a body, so an L0 exact-match hit cannot hide L1. At 0.0 each
request gets a unique system message first.

With ``VLLM_ROUTER_STAGES=1`` on a gRPC worker pool, the summary also reports
router ``encode_ms``, ``template_ms``, ``frontend_ms`` and ``encode_reuse``
(``encode_cached_tokens / n_prompt_tokens``: the share of prompt tokens that
L0/L1 served without running the tokenizer).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


def _env(name: str, default: str | None = None) -> str:
    raw = (os.environ.get(name) or "").strip()
    if raw:
        return raw
    if default is not None:
        return default
    raise SystemExit(f"set {name}")


def _default_cli_model() -> str:
    """``--model`` default from ``MODEL``, else ``default``."""
    raw = (os.environ.get("MODEL") or "").strip()
    return raw or "default"


def _tokenizer_dir_for_tokens(*, model: str, model_dir: str) -> str:
    """Local HF path for ``--tokens`` when ``--model-dir`` is omitted."""
    explicit = (model_dir or "").strip()
    if explicit:
        return explicit
    override = (os.environ.get("MODEL_DIR") or "").strip()
    if override:
        return override
    if model != "default":
        return model
    raise SystemExit(
        "--tokens needs a tokenizer path: pass --model-dir, set MODEL_DIR, "
        "or set MODEL / --model to the HF id or local snapshot"
    )


def make_bodies(hit_rate: float, text: str) -> tuple[str, str]:
    """Return (miss_user, hit_user) for the requested reuse pattern."""
    if hit_rate >= 0.99:
        return text, text
    if hit_rate <= 0.0:
        return text, f"TAG-{int(time.time() * 1000)} {text}"
    split = max(1, int(len(text) * hit_rate))
    prefix, tail = text[:split], text[split:]
    return text, prefix + f" TAIL-{int(time.time() * 1000)} " + tail


def make_wave_bodies(hit_rate: float, text: str, requests: int) -> list[str]:
    """Return request bodies for the concurrent wave.

    The 0% case must not reuse the serial warmup hit body; otherwise the wave
    measures a warmed exact-prefix body instead of misses.
    """
    if hit_rate >= 0.99:
        return [text] * requests
    stamp = time.time_ns()
    if hit_rate <= 0.0:
        return [f"TAG-WAVE-{stamp}-{idx} {text}" for idx in range(requests)]
    split = max(1, int(len(text) * hit_rate))
    prefix, tail = text[:split], text[split:]
    return [prefix + f" TAIL-WAVE-{stamp}-{idx} " + tail for idx in range(requests)]


def user_messages(user: str) -> list[dict]:
    return [{"role": "user", "content": user}]


def make_turn_messages(hit_rate: float, text: str, turns: int, tag: str) -> list[dict]:
    """Shared earlier turns, then one unique last user message.

    ``hit_rate`` of ``text`` goes into ``turns`` alternating user/assistant
    messages; the rest follows ``tag`` in the last user message. At 0.0 the
    whole text is the last message and ``tag`` also opens a system message,
    so nothing is shared.
    """
    if hit_rate <= 0.0:
        return [
            {"role": "system", "content": tag},
            {"role": "user", "content": f"{tag} {text}"},
        ]
    split = len(text) if hit_rate >= 0.99 else max(1, int(len(text) * hit_rate))
    shared, tail = text[:split], text[split:]
    turns = max(2, turns - turns % 2)
    step = max(1, len(shared) // turns)
    messages = []
    for idx in range(turns):
        chunk = shared[idx * step : len(shared) if idx == turns - 1 else (idx + 1) * step]
        messages.append({"role": "user" if idx % 2 == 0 else "assistant", "content": chunk})
    messages.append({"role": "user", "content": f"{tag} {tail}".rstrip()})
    return messages


def make_user_text(chars: int) -> str:
    unit = (
        "Batched transformer decoding reuses a prefix KV cache when the "
        "next request shares a leading token span. "
    )
    return (unit * (chars // len(unit) + 1))[:chars]


def make_user_text_exact_tokens(model_dir: str, target_tokens: int) -> str:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    unit = None
    for cand in (" the", " a", ".\n", "x", "1"):
        ids = tok.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            unit = cand
            break
    if unit is None:
        raise SystemExit("could not find a 1-token fragment")

    def n_ids(content: str) -> int:
        enc = tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        ids = enc["input_ids"]
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return len(ids)

    overhead = n_ids(unit)
    need = max(1, target_tokens - overhead + 1)
    content = unit * need
    delta = target_tokens - n_ids(content)
    if delta:
        content = unit * max(1, need + delta)
    return content


def post_chat(
    url: str, model: str, messages: list[dict], max_tokens: int, timeout: float
) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft_ms = None
    last_output_ms = None
    usage: dict = {}
    text_parts: list[str] = []
    stages = None
    saw_done = False
    saw_finish = False
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            headers = {k.lower(): v for k, v in resp.headers.items()}
            stages = headers.get("x-router-stages")
            for line in resp:
                raw = line.decode("utf-8", errors="replace").strip()
                if raw.startswith(": router-stages "):
                    stages = raw[len(": router-stages ") :]
                    continue
                if not raw.startswith("data: "):
                    continue
                data = raw[6:]
                if data == "[DONE]":
                    saw_done = True
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"malformed SSE data: {data!r}") from exc
                if obj.get("error") is not None:
                    raise RuntimeError(f"SSE error: {obj['error']}")
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    saw_finish = True
                delta_obj = choice.get("delta") or {}
                delta = (
                    delta_obj.get("content")
                    or delta_obj.get("reasoning")
                    or delta_obj.get("reasoning_content")
                    or ""
                )
                meaningful = bool(delta) or bool(delta_obj.get("tool_calls"))
                if meaningful:
                    observed_ms = (time.perf_counter() - t0) * 1000.0
                    if ttft_ms is None:
                        ttft_ms = observed_ms
                    last_output_ms = observed_ms
                    if delta:
                        text_parts.append(delta)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
    except (OSError, RuntimeError) as exc:
        raise SystemExit(f"stream failed after partial_chars={sum(map(len, text_parts))}: {exc}") from exc
    total_duration_ms = (time.perf_counter() - t0) * 1000.0
    if ttft_ms is None:
        raise SystemExit(
            f"no streamed first output from {url} after {total_duration_ms:.1f}ms "
            f"(usage={usage} chars_out={sum(len(p) for p in text_parts)})"
        )
    if not saw_finish or not saw_done:
        raise SystemExit(
            f"incomplete SSE stream finish={saw_finish} done={saw_done} "
            f"partial_chars={sum(map(len, text_parts))}"
        )
    n_prompt = usage.get("prompt_tokens")
    n_out = usage.get("completion_tokens")
    if not isinstance(n_prompt, int) or not isinstance(n_out, int):
        raise SystemExit(f"missing/invalid usage: {usage}")
    assert last_output_ms is not None
    decode_ms = max(0.0, last_output_ms - ttft_ms)
    decode_tokens = max(0, int(n_out) - 1)
    decode_tok_s = (
        float(decode_tokens) / (decode_ms / 1000.0)
        if decode_ms > 1.0 and decode_tokens
        else None
    )
    e2e_tok_s = (
        float(n_out) / (total_duration_ms / 1000.0)
        if total_duration_ms > 1.0 and n_out
        else None
    )
    tpot_ms = (decode_ms / float(decode_tokens)) if decode_tokens else None
    return {
        "ttft_ms": ttft_ms,
        "e2e_ms": total_duration_ms,
        "total_duration_ms": total_duration_ms,
        "last_output_ms": last_output_ms,
        "prompt_tokens": n_prompt,
        "completion_tokens": n_out or None,
        "chars_out": sum(len(p) for p in text_parts),
        "decode_tok_s": decode_tok_s,
        "e2e_tok_s": e2e_tok_s,
        "tpot_ms": tpot_ms,
        "stages": stages,
    }


def stage_fields(row: dict) -> dict:
    """Router stage numbers from ``VLLM_ROUTER_STAGES=1``, or {}."""
    raw = row.get("stages")
    try:
        stages = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except json.JSONDecodeError:
        return {}
    fields = {
        key: float(stages[key])
        for key in ("template_ms", "encode_ms", "frontend_ms")
        if isinstance(stages.get(key), (int, float))
    }
    cached, total = stages.get("encode_cached_tokens"), stages.get("n_prompt_tokens")
    if isinstance(cached, int) and isinstance(total, int) and total > 0:
        fields["encode_reuse"] = cached / total
    return fields


def summarize_rows(rows: list[dict]) -> dict:
    stages = [stage_fields(row) for row in rows]

    def values(key: str) -> list[float]:
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        return vals or [s[key] for s in stages if key in s]

    summary = {"count": len(rows)}
    for key in (
        "ttft_ms",
        "e2e_ms",
        "tpot_ms",
        "template_ms",
        "encode_ms",
        "frontend_ms",
        "encode_reuse",
    ):
        vals = sorted(values(key))
        if vals:
            summary[key] = {
                "min": vals[0],
                "p50": vals[len(vals) // 2],
                "avg": sum(vals) / len(vals),
                "max": vals[-1],
            }
    return summary


def post_hit_wave(
    url: str,
    model: str,
    bodies: list[list[dict]],
    max_tokens: int,
    timeout: float,
    concurrency: int,
) -> dict:
    rows: list[dict] = []
    requests = len(bodies)

    for batch_start in range(0, requests, concurrency):
        batch_size = min(concurrency, requests - batch_start)
        barrier = threading.Barrier(batch_size)

        def worker(local_idx: int) -> dict:
            request_index = batch_start + local_idx
            barrier.wait()
            row = post_chat(url, model, bodies[request_index], max_tokens, timeout)
            row["request_index"] = request_index
            row["batch_index"] = batch_start // concurrency
            return row

        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(worker, idx) for idx in range(batch_size)]
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())

    rows.sort(key=lambda row: row["request_index"])
    return {
        "concurrency": concurrency,
        "requests": requests,
        "summary": summarize_rows(rows),
        "requests_detail": rows,
    }


# --- Pytest perf (tests/python/e2e/perf/grpc) and remote bench reuse ----------


def read_perf_settings(
    *,
    backend_label: str | None = None,
    bench_tokens: int | None = None,
    bench_max_tokens: int | None = None,
    bench_timeout: float | None = None,
) -> dict[str, Any]:
    """Merge case JSON bench fields with PERF_* / ROUTER_URL env (pytest perf)."""
    model = os.environ["MODEL"]
    if os.environ.get("PERF_TOKENS", "").strip():
        tokens = int(os.environ["PERF_TOKENS"].strip())
    elif bench_tokens is not None:
        tokens = bench_tokens
    else:
        tokens = 0
    label = os.environ.get("BACKEND_LABEL") or backend_label or "backend"
    default_max = bench_max_tokens if bench_max_tokens is not None else 64
    default_timeout = bench_timeout if bench_timeout is not None else 1800.0
    return {
        "router_url": os.environ["ROUTER_URL"],
        "model": model,
        "model_dir": os.environ.get("MODEL_DIR", model),
        "tokens": tokens,
        "chars": int(os.environ.get("KVHIT_CHARS", "8000")),
        "max_tokens": int(os.environ.get("PERF_MAX_TOKENS", str(default_max))),
        "timeout": float(os.environ.get("PERF_TIMEOUT", str(default_timeout))),
        "warmup": os.environ.get("PERF_WARMUP", "1").lower() in ("1", "true", "yes"),
        "backend_label": label,
        "output_dir": os.environ.get("PERF_OUTPUT_DIR", "").strip(),
        "runs": max(1, int(os.environ.get("RUNS", "1"))),
    }


def _resolve_user_text(settings: dict[str, Any]) -> str:
    if settings["tokens"] > 0:
        return make_user_text_exact_tokens(settings["model_dir"], settings["tokens"])
    return make_user_text(settings["chars"])


def _messages_for_shape(
    *,
    shape: str,
    hit_rate: float,
    user: str,
    requests: int,
    turns: int,
) -> tuple[list[dict], list[dict], list[list[dict]]]:
    """Miss, hit, and wave chat bodies.

    ``split`` cuts inside one user message (worker KV). ``turns`` puts the
    shared span in earlier chat turns so L1 has a template boundary to reuse.
    """
    if shape == "turns":
        stamp = time.time_ns()
        miss_messages = make_turn_messages(hit_rate, user, turns, f"MISS-{stamp}")
        hit_messages = make_turn_messages(hit_rate, user, turns, f"HIT-{stamp}")
        wave_messages = [
            make_turn_messages(hit_rate, user, turns, f"TAIL-WAVE-{stamp}-{idx}")
            for idx in range(requests)
        ]
        return miss_messages, hit_messages, wave_messages
    if shape != "split":
        raise ValueError(f"shape must be split or turns, got {shape!r}")
    miss_user, hit_user = make_bodies(hit_rate, user)
    return (
        user_messages(miss_user),
        user_messages(hit_user),
        [user_messages(body) for body in make_wave_bodies(hit_rate, user, requests)],
    )


def run_prefix_kvhit_wave(
    settings: dict[str, Any],
    *,
    hit_rate: float,
    concurrency: int,
    requests: int,
    warmup: bool | None = None,
    shape: str | None = None,
    turns: int | None = None,
) -> dict[str, Any]:
    """One warmup (optional) + concurrent wave; used by pytest perf and callable from tests.

    ``shape`` defaults to ``split`` (one user message). Pass ``turns`` to exercise
    the router L1 encoding cache, which only reuses tokens up to a chat-template
    boundary.
    """
    user = _resolve_user_text(settings)
    do_warmup = settings["warmup"] if warmup is None else warmup
    shape = shape or str(settings.get("shape") or "split")
    turn_count = turns if turns is not None else int(settings.get("turns") or 8)
    miss_messages, hit_messages, wave_messages = _messages_for_shape(
        shape=shape,
        hit_rate=hit_rate,
        user=user,
        requests=requests,
        turns=turn_count,
    )
    result: dict[str, Any] = {
        "router": settings["router_url"],
        "model": settings["model"],
        "hit_rate": hit_rate,
        "shape": shape,
        "concurrency": concurrency,
        "requests": requests,
    }
    if shape == "turns":
        result["turns"] = turn_count
    if do_warmup:
        result["warmup"] = {
            "miss": post_chat(
                settings["router_url"],
                settings["model"],
                miss_messages,
                settings["max_tokens"],
                settings["timeout"],
            ),
            "hit": post_chat(
                settings["router_url"],
                settings["model"],
                hit_messages,
                settings["max_tokens"],
                settings["timeout"],
            ),
        }
    result["wave"] = post_hit_wave(
        settings["router_url"],
        settings["model"],
        wave_messages,
        settings["max_tokens"],
        settings["timeout"],
        concurrency,
    )
    return result


def assert_wave_summary(wave: dict[str, Any], expected_requests: int) -> dict[str, Any]:
    summary = wave.get("summary") or {}
    assert summary.get("count") == expected_requests
    for key in ("ttft_ms", "e2e_ms"):
        block = summary.get(key)
        assert isinstance(block, dict), f"missing {key} in {json.dumps(summary)}"
        assert block.get("avg", 0) > 0
    return summary


def _fmt_metric_block(block: dict[str, Any] | None) -> str:
    if not isinstance(block, dict):
        return "n/a"
    parts = []
    for stat in ("min", "avg", "max"):
        val = block.get(stat)
        if isinstance(val, (int, float)):
            parts.append(f"{stat}={val:.2f}")
    return " ".join(parts) if parts else "n/a"


def print_wave_summary(
    wave: dict[str, Any],
    *,
    backend: str | None = None,
    hit_rate: float | None = None,
    concurrency: int | None = None,
    requests: int | None = None,
    run_idx: int | None = None,
) -> None:
    """Human-readable wave summary for pytest -sv / bench CLI (stderr)."""
    import sys

    if os.environ.get("PERF_QUIET", "").lower() in ("1", "true", "yes"):
        return
    summary = wave.get("summary") or {}
    tags: list[str] = []
    if backend:
        tags.append(backend)
    if run_idx is not None:
        tags.append(f"run{run_idx}")
    if hit_rate is not None:
        tags.append(f"hit={hit_rate:g}")
    if concurrency is not None:
        tags.append(f"c={concurrency}")
    if requests is not None:
        tags.append(f"r={requests}")
    title = " | ".join(tags) if tags else "wave"
    lines = [
        "",
        f"======== perf summary [{title}] count={summary.get('count', '?')} ========",
    ]
    for key in ("ttft_ms", "e2e_ms", "tpot_ms"):
        lines.append(f"  {key}: {_fmt_metric_block(summary.get(key))}")
    lines.append("=" * (len(lines[-2]) if len(lines) > 1 else 40))
    print("\n".join(lines), file=sys.stderr, flush=True)


def hit_rate_file_tag(hit_rate: float) -> str:
    if hit_rate >= 0.99:
        return "099"
    if hit_rate <= 0.0:
        return "000"
    return str(hit_rate).replace(".", "")


def maybe_write_result(
    settings: dict[str, Any],
    *,
    run_idx: int,
    hit_rate: float,
    concurrency: int,
    requests: int,
    payload: dict[str, Any],
) -> Path | None:
    base = settings.get("output_dir") or ""
    if not base:
        return None
    out_dir = Path(base)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"run{run_idx}_hit{hit_rate_file_tag(hit_rate)}"
        f"_c{concurrency}_r{requests}.json"
    )
    path = out_dir / name
    envelope = {
        "backend_label": settings["backend_label"],
        "run_idx": run_idx,
        "hit_rate": hit_rate,
        "concurrency": concurrency,
        "requests": requests,
        **payload,
    }
    path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    return path


def default_output_dir(settings: dict[str, Any], *, results_parent: Path) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = settings["backend_label"]
    return str(results_parent / f"{label}_{stamp}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-url", default=_env("ROUTER_URL", "http://127.0.0.1:30000"))
    parser.add_argument("--model", default=_default_cli_model())
    parser.add_argument("--hit-rate", type=float, default=float(os.environ.get("HIT_RATE", "0.99")))
    parser.add_argument("--chars", type=int, default=int(os.environ.get("KVHIT_CHARS", "8000")))
    parser.add_argument(
        "--tokens",
        type=int,
        default=0,
        help="exact ISL via tokenizer (--model-dir, or same as --model / MODEL)",
    )
    parser.add_argument(
        "--model-dir",
        default="",
        help="HF snapshot for --tokens (optional if --model / MODEL is a local path or id)",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--keep-dataset",
        action="store_true",
        help="write the bodies to a temp json and print the path",
    )
    parser.add_argument(
        "--once",
        choices=("miss", "hit"),
        default="",
        help="post one request only (wave). default is serial miss then hit",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="concurrent hit requests in the main wave",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=0,
        help="number of hit requests in the main wave; default is max(1, 2 * --concurrency)",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="first post a serial miss/hit pair before the concurrent hit wave",
    )
    parser.add_argument(
        "--shape",
        choices=("split", "turns"),
        default=os.environ.get("KVHIT_SHAPE", "split"),
        help="split: one user message cut by hit rate; turns: shared earlier turns + unique last turn",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=8,
        help="number of shared user/assistant messages for --shape turns",
    )
    args = parser.parse_args()

    if args.tokens:
        tok_dir = _tokenizer_dir_for_tokens(model=args.model, model_dir=args.model_dir)
        user = make_user_text_exact_tokens(tok_dir, args.tokens)
    else:
        user = make_user_text(args.chars)
    wave_requests = args.requests or max(1, 2 * args.concurrency)
    stamp = time.time_ns()
    if args.shape == "turns":
        miss_body = make_turn_messages(args.hit_rate, user, args.turns, f"MISS-{stamp}")
        hit_body = make_turn_messages(args.hit_rate, user, args.turns, f"HIT-{stamp}")
        wave_bodies = [
            make_turn_messages(args.hit_rate, user, args.turns, f"TAIL-WAVE-{stamp}-{idx}")
            for idx in range(wave_requests)
        ]
    else:
        miss_user, hit_user = make_bodies(args.hit_rate, user)
        miss_body, hit_body = user_messages(miss_user), user_messages(hit_user)
        wave_bodies = [
            user_messages(body) for body in make_wave_bodies(args.hit_rate, user, wave_requests)
        ]

    def chars(messages: list[dict]) -> int:
        return sum(len(message["content"]) for message in messages)

    if args.keep_dataset:
        path = Path(tempfile.mkdtemp(prefix="prefix-kvhit-")) / "bodies.json"
        path.write_text(
            json.dumps(
                {
                    "hit_rate": args.hit_rate,
                    "shape": args.shape,
                    "miss_chars": chars(miss_body),
                    "hit_chars": chars(hit_body),
                    "miss_messages": miss_body,
                    "hit_messages": hit_body,
                }
            )
        )
        print(f"dataset_meta {path}")

    print(
        json.dumps(
            {
                "router": args.router_url,
                "model": args.model,
                "hit_rate": args.hit_rate,
                "shape": args.shape,
                "once": args.once or None,
                "miss_chars": chars(miss_body),
                "hit_chars": chars(hit_body),
            }
        )
    )
    if args.once:
        body = miss_body if args.once == "miss" else hit_body
        row = post_chat(args.router_url, args.model, body, args.max_tokens, args.timeout)
        print(json.dumps({"once": args.once, "req": row}, indent=2))
        return
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.requests < 0:
        raise SystemExit("--requests must be >= 0")
    result = {}
    if args.warmup:
        result["warmup"] = {
            "miss": post_chat(args.router_url, args.model, miss_body, args.max_tokens, args.timeout),
            "hit": post_chat(args.router_url, args.model, hit_body, args.max_tokens, args.timeout),
        }
    result["wave"] = post_hit_wave(
        args.router_url,
        args.model,
        wave_bodies,
        args.max_tokens,
        args.timeout,
        args.concurrency,
    )
    print_wave_summary(
        result["wave"],
        hit_rate=args.hit_rate,
        concurrency=args.concurrency,
        requests=wave_requests,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
