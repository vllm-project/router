"""Bounded PD request probe; validates responses and progress after disconnects.

This is not an accuracy benchmark or proof of server-side resource reclamation.
Run only against an allocated test deployment. No server is started or stopped.
"""

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

if __package__:
    from .completion_validation import validate_single_completion
else:  # The harness invokes this file directly, without installing a tests package.
    from completion_validation import validate_single_completion


async def request(session, url, model, mode, sequence):
    payload = {
        "model": model,
        "prompt": f"Request {sequence}. Explain why the sky appears blue in detail:",
        "temperature": 0,
        "max_tokens": 256 if mode == "disconnect" else 32,
        "stream": mode != "completion",
    }
    async with session.post(url, json=payload) as response:
        response.raise_for_status()
        if mode == "completion":
            validate_single_completion(await response.json())
            return "completed"
        text_seen = False
        finished = False
        async for raw in response.content:
            line = raw.decode("utf-8").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                raise ValueError("unexpected SSE line")
            data = line[5:].strip()
            if data == "[DONE]":
                if mode == "disconnect":
                    raise ValueError("stream ended before an active disconnect")
                if not text_seen or not finished:
                    raise ValueError("stream ended without text and finish_reason")
                return "completed"
            chunk = json.loads(data)
            if not isinstance(chunk, dict) or "error" in chunk:
                raise ValueError("invalid SSE completion object")
            choices = chunk.get("choices")
            if choices == [] and chunk.get("usage") is not None:
                continue
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("invalid SSE choices")
            choice = choices[0]
            if not isinstance(choice, dict) or not isinstance(choice.get("text"), str):
                raise ValueError("invalid SSE text")
            text_seen |= bool(choice["text"].strip())
            reason = choice.get("finish_reason")
            if reason is not None:
                if reason not in ("stop", "length"):
                    raise ValueError("invalid SSE finish_reason")
                finished = True
            if mode == "disconnect" and text_seen:
                if finished:
                    raise ValueError(
                        "first text already finished; cancellation untested"
                    )
                # is_eof includes unread buffered bytes, unlike at_eof().
                if response.content.is_eof():
                    raise ValueError("response already at EOF; cancellation untested")
                buffered = response.content.read_nowait()
                for pending in buffered.splitlines():
                    pending = pending.strip()
                    if not pending or pending.startswith(b":"):
                        continue
                    if not pending.startswith(b"data:"):
                        raise ValueError(
                            "buffered SSE is incomplete; cancellation untested"
                        )
                    data = pending[5:].strip()
                    if data == b"[DONE]":
                        raise ValueError(
                            "completion already buffered; cancellation untested"
                        )
                    event = json.loads(data)
                    if not isinstance(event, dict) or "error" in event:
                        raise ValueError("invalid buffered SSE completion")
                    choices = event.get("choices", [])
                    if not isinstance(choices, list) or any(
                        not isinstance(c, dict) for c in choices
                    ):
                        raise ValueError("invalid buffered SSE choices")
                    if any(c.get("finish_reason") is not None for c in choices):
                        raise ValueError(
                            "completion already buffered; cancellation untested"
                        )
                # Closing an unfinished response disconnects the HTTP client. This
                # alone does not prove that the engine propagated the abort.
                response.close()
                return "disconnected_after_text"
        raise ValueError("truncated SSE stream (missing [DONE])")


def save_report(report, path):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


async def probe(url, model, levels=(1, 2, 4, 8), rounds=2, timeout=60, output=None):
    if (
        not levels
        or len(levels) > 8
        or any(type(n) is not int or n < 1 or n > 32 for n in levels)
    ):
        raise ValueError("concurrency levels must be integers in 1..32")
    if not 1 <= rounds <= 10 or not 0 < timeout <= 600:
        raise ValueError("rounds must be 1..10 and timeout must be in (0, 600]")
    report = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "levels": list(levels),
        "rounds": rounds,
        "timeout_seconds": timeout,
        "results": [],
        "passed": False,
        "status": "running",
    }
    save_report(report, output)
    endpoint = url.rstrip("/") + "/v1/completions"
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout)
    ) as session:

        async def one(phase, mode, level, iteration, slot):
            start = time.monotonic()
            result = {
                "mode": mode,
                "phase": phase,
                "concurrency": level,
                "round": iteration,
                "slot": slot,
            }
            try:
                result["outcome"] = await request(
                    session,
                    endpoint,
                    model,
                    mode,
                    f"{level}-{iteration}-{phase}-{slot}",
                )
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                result["outcome"] = "failed"
                # Avoid URLs (which may contain credentials) and response bodies.
                result["error_type"] = type(exc).__name__
                if type(exc) is ValueError:
                    # Our response validators use fixed messages. Do not expose
                    # parser/client exception text, which can contain payloads.
                    result["detail"] = str(exc)
                if isinstance(exc, aiohttp.ClientResponseError):
                    result["http_status"] = exc.status
            result["elapsed_seconds"] = round(time.monotonic() - start, 4)
            return result

        for level in levels:
            for iteration in range(rounds):
                for phase, mode in (
                    ("baseline", "completion"),
                    ("stream", "stream"),
                    ("disconnect", "disconnect"),
                    ("recovery", "completion"),
                ):
                    batch = await asyncio.gather(
                        *(one(phase, mode, level, iteration, i) for i in range(level))
                    )
                    report["results"].extend(batch)
                    if any(r["outcome"] == "failed" for r in batch):
                        report["status"] = "failed"
                        save_report(report, output)
                        return report  # no automatic retry or higher load after failure
                    save_report(report, output)
    report["passed"] = True
    report["status"] = "passed"
    save_report(report, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = asyncio.run(
            probe(
                args.router_url,
                args.model,
                args.concurrency,
                args.rounds,
                args.timeout,
                args.output,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(
        f"PD lifecycle {report['status']}: {len(report['results'])} requests; "
        f"report={args.output}"
    )
    parser.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
