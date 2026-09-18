#!/usr/bin/env python3
"""Prefix-cache miss vs hit against an already-running router.

Does not start engines or talk to Slurm/Docker. Point
``ROUTER_URL`` at a live OpenAI ``/v1/chat/completions`` endpoint (the
router, or a worker). Pair with one of the serve_*.sh scripts in this
directory.

  export ROUTER_URL=http://127.0.0.1:30000
  export MODEL=my-served-model
  python scripts/backend/bench_prefix_kvhit.py

By default the script writes a short synthetic chat body into a temp
file (repeated paragraph; no tokenizer required). Pass ``--tokens N``
and ``--model-dir /path/to/hf`` to size the user text to an exact
input length via transformers.

Returned JSON reports client-observed ``ttft_ms``, ``e2e_ms`` /
``total_duration_ms``, ``last_output_ms``, token counts, throughput, and
``tpot_ms``. The optional ``stages`` field is raw router diagnostic JSON
from ``VLLM_ROUTER_STAGES=1``. Stage fields such as ``xfer_ms`` and
``engine_ms`` are boundary/residual diagnostics, not direct network or
EngineCore telemetry.

Hit rates:
  0.99 — same body twice (second request should prefix-hit)
  0.30 — shared prefix + unique tail
  0.00 — unique tag at the start of the second request
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def _env(name: str, default: str | None = None) -> str:
    raw = (os.environ.get(name) or "").strip()
    if raw:
        return raw
    if default is not None:
        return default
    raise SystemExit(f"set {name}")


def make_bodies(hit_rate: float, text: str) -> tuple[str, str]:
    """Return (miss_user, hit_user) for the requested reuse pattern."""
    if hit_rate >= 0.99:
        return text, text
    if hit_rate <= 0.0:
        return text, f"TAG-{int(time.time() * 1000)} {text}"
    split = max(1, int(len(text) * hit_rate))
    prefix, tail = text[:split], text[split:]
    return text, prefix + f" TAIL-{int(time.time() * 1000)} " + tail


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


def post_chat(url: str, model: str, user: str, max_tokens: int, timeout: float) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": user}],
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-url", default=_env("ROUTER_URL", "http://127.0.0.1:30000"))
    parser.add_argument("--model", default=_env("MODEL", "default"))
    parser.add_argument("--hit-rate", type=float, default=float(os.environ.get("HIT_RATE", "0.99")))
    parser.add_argument("--chars", type=int, default=int(os.environ.get("KVHIT_CHARS", "8000")))
    parser.add_argument("--tokens", type=int, default=0, help="exact ISL via --model-dir")
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", ""))
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
    args = parser.parse_args()

    if args.tokens:
        if not args.model_dir:
            raise SystemExit("--tokens requires --model-dir (or MODEL_DIR)")
        user = make_user_text_exact_tokens(args.model_dir, args.tokens)
    else:
        user = make_user_text(args.chars)
    miss_user, hit_user = make_bodies(args.hit_rate, user)

    if args.keep_dataset:
        path = Path(tempfile.mkdtemp(prefix="prefix-kvhit-")) / "bodies.json"
        path.write_text(
            json.dumps(
                {
                    "hit_rate": args.hit_rate,
                    "miss_chars": len(miss_user),
                    "hit_chars": len(hit_user),
                    "miss_user": miss_user,
                    "hit_user": hit_user,
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
                "once": args.once or None,
                "miss_chars": len(miss_user),
                "hit_chars": len(hit_user),
            }
        )
    )
    if args.once:
        body = miss_user if args.once == "miss" else hit_user
        row = post_chat(args.router_url, args.model, body, args.max_tokens, args.timeout)
        print(json.dumps({"once": args.once, "req": row}, indent=2))
        return
    miss = post_chat(args.router_url, args.model, miss_user, args.max_tokens, args.timeout)
    hit = post_chat(args.router_url, args.model, hit_user, args.max_tokens, args.timeout)
    print(json.dumps({"miss": miss, "hit": hit}, indent=2))


if __name__ == "__main__":
    main()
