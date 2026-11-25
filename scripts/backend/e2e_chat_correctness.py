#!/usr/bin/env python3
"""Small live E2E correctness observation for router chat backends.

This script does not start workers or routers. Point it at one or more already
running OpenAI-compatible router endpoints. It sends a short deterministic
streaming chat request, validates that the stream completes with usage, and
prints the generated text plus a stable hash.

The output is an observation aid, not a golden-output test. Different models,
parallelism, kernels, or frontend paths may produce slightly different text.
By default the script only validates that each stream completes, returns usage,
and produces non-empty text. Optional flags can add stricter checks for focused
debugging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request


DEFAULT_PROMPT = "Reply with exactly this text and no extra words: router-ok"


def post_chat(url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
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

    usage: dict = {}
    text_parts: list[str] = []
    saw_done = False
    saw_finish = False
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                raw = line.decode("utf-8", errors="replace").strip()
                if not raw.startswith("data: "):
                    continue
                data = raw[6:]
                if data == "[DONE]":
                    saw_done = True
                    break
                obj = json.loads(data)
                if obj.get("error") is not None:
                    raise RuntimeError(f"SSE error: {obj['error']}")
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    if choice.get("finish_reason") is not None:
                        saw_finish = True
                    delta = choice.get("delta") or {}
                    text_parts.append(
                        delta.get("content")
                        or delta.get("reasoning")
                        or delta.get("reasoning_content")
                        or ""
                    )
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{url}: HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{url}: stream failed: {exc}") from exc

    text = "".join(text_parts).strip()
    if not saw_finish or not saw_done:
        raise SystemExit(f"{url}: incomplete stream finish={saw_finish} done={saw_done}")
    if not text:
        raise SystemExit(f"{url}: empty output")
    if not isinstance(usage.get("prompt_tokens"), int) or not isinstance(
        usage.get("completion_tokens"), int
    ):
        raise SystemExit(f"{url}: missing/invalid usage: {usage}")

    return {
        "text": text,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
    }


def parse_case(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("case must be LABEL=URL")
    label, url = raw.split("=", 1)
    label = label.strip()
    url = url.strip()
    if not label or not url:
        raise argparse.ArgumentTypeError("case must be LABEL=URL")
    return label, url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--expect-substring",
        default="",
        help="case-insensitive substring expected in every generated output; empty disables",
    )
    parser.add_argument(
        "--require-exact-match",
        action="store_true",
        help="require every included case to generate exactly the same text",
    )
    args = parser.parse_args()

    results = {}
    for label, url in args.case:
        row = post_chat(url, args.model, args.prompt, args.max_tokens, args.timeout)
        expected = args.expect_substring.strip().lower()
        if expected and expected not in row["text"].lower():
            raise SystemExit(f"{label}: expected substring {args.expect_substring!r}, got {row['text']!r}")
        results[label] = row
        print(json.dumps({"label": label, **row}, ensure_ascii=False))

    if args.require_exact_match:
        labels = list(results)
        if len(labels) >= 2:
            first_label = labels[0]
            first_text = results[first_label]["text"]
            mismatches = [
                label
                for label in labels[1:]
                if results[label]["text"] != first_text
            ]
            if mismatches:
                details = {
                    label: results[label]["text"]
                    for label in [first_label, *mismatches]
                }
                raise SystemExit(
                    "generated text differs across cases: "
                    + json.dumps(details, ensure_ascii=False)
                )
            print(json.dumps({"comparison": "all_cases_exact_match", "match": True}))


if __name__ == "__main__":
    main()
