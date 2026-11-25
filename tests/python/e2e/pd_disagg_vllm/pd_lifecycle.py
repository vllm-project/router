"""Bounded PD request probe; validates responses and progress after disconnects.

This is not an accuracy benchmark or proof of server-side resource reclamation.
Run only against an allocated test deployment. No server is started or stopped.
"""

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

if not __package__:  # Support direct invocation by the eval-container harness.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.python.helpers.completion_validation import validate_single_completion


class DisconnectUntested(ValueError):
    """The response finished before the client could disconnect it."""


def _sse_event(raw):
    """Return (text, finish_reason, done), or None for a comment/blank line."""
    line = raw.strip()
    if not line or line.startswith(b":"):
        return None
    if not line.startswith(b"data:"):
        raise ValueError("unexpected SSE line")
    data = line[5:].strip()
    if data == b"[DONE]":
        return "", None, True
    event = json.loads(data.decode("utf-8"))
    if not isinstance(event, dict) or "error" in event:
        raise ValueError("invalid SSE completion object")
    choices = event.get("choices")
    if choices == [] and event.get("usage") is not None:
        return "", None, False
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("invalid SSE choices")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("text"), str):
        raise ValueError("invalid SSE text")
    reason = choice.get("finish_reason")
    if reason is not None and reason not in ("stop", "length"):
        raise ValueError("invalid SSE finish_reason")
    return choice["text"], reason, False


def _buffered_sse_completed(buffered, eof, finished=False):
    """Validate complete buffered events and identify a finished response."""
    lines = buffered.split(b"\n")
    partial = lines.pop()
    if eof and partial.strip():
        raise ValueError("truncated buffered SSE line")
    done = False
    for pending in lines:
        event = _sse_event(pending)
        if event is None:
            continue
        if done:
            raise ValueError("buffered data after [DONE]")
        _, reason, done = event
        finished |= reason is not None
    if done and not finished:
        raise ValueError("buffered stream ended without finish_reason")
    if eof and not done:
        raise ValueError("truncated SSE stream (missing [DONE])")
    return finished or done


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
            event = _sse_event(raw)
            if event is None:
                continue
            text, reason, done = event
            if done:
                if mode == "disconnect":
                    raise ValueError("stream ended before an active disconnect")
                if not text_seen or not finished:
                    raise ValueError("stream ended without text and finish_reason")
                return "completed"
            text_seen |= bool(text.strip())
            finished |= reason is not None
            if mode == "disconnect" and text_seen:
                # is_eof includes unread buffered bytes, unlike at_eof(). Drain
                # and validate them before classifying an EOF as normal completion.
                eof = response.content.is_eof()
                buffered = response.content.read_nowait()
                # Without EOF, leave a trailing partial line unclassified and
                # disconnect immediately. With EOF, it must form a complete,
                # normally terminated SSE response to count as untested.
                if _buffered_sse_completed(buffered, eof, finished):
                    raise DisconnectUntested(
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
        live_disconnect_observed = False

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
            except DisconnectUntested as exc:
                result["outcome"] = "untested"
                result["error_type"] = type(exc).__name__
                result["detail"] = str(exc)
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
            except Exception as exc:
                # CancelledError and process-level interrupts inherit BaseException
                # and still propagate; unexpected request errors become evidence.
                result["outcome"] = "failed"
                result["error_type"] = type(exc).__name__
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
                    failed = any(r["outcome"] == "failed" for r in batch)
                    if phase == "disconnect" and any(
                        r["outcome"] == "disconnected_after_text" for r in batch
                    ):
                        live_disconnect_observed = True
                    if failed:
                        report["status"] = "failed"
                        save_report(report, output)
                        return report  # no automatic retry or higher load after failure
                    save_report(report, output)
    if not live_disconnect_observed:
        report["status"] = "failed"
        report["failure_reason"] = "no live disconnect observed"
        save_report(report, output)
        return report
    report["passed"] = True
    report["status"] = "passed"
    save_report(report, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", nargs="+", default=["1", "2", "4", "8"])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = asyncio.run(
            probe(
                args.router_url,
                args.model,
                [int(n) for arg in args.concurrency for n in arg.split()],
                args.rounds,
                args.timeout,
                args.output,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))
    outcomes = Counter(result["outcome"] for result in report["results"])
    counts = ", ".join(
        f"{name}={outcomes[name]}"
        for name in ("completed", "disconnected_after_text", "untested", "failed")
    )
    print(f"PD lifecycle {report['status']}: {len(report['results'])} requests")
    print(f"outcomes: {counts}; report={args.output}")
    if reason := report.get("failure_reason"):
        print(f"reason: {reason}")
    parser.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
