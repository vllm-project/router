"""Exercise the probe against real loopback HTTP/SSE, without a GPU or router."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from aiohttp import web

from ..helpers.pd_lifecycle import probe

pytestmark = [pytest.mark.ready, pytest.mark.cpu]


async def exercise(fault=None, output=None, cli=False):
    calls = []

    async def handler(request):
        payload = await request.json()
        if output is not None:
            # The CLI must leave a parseable checkpoint even before a request
            # completes, so a killed job cannot silently lose earlier waves.
            assert json.loads(output.read_text())["status"] == "running"
        calls.append(payload)
        if fault == "http":
            return web.Response(status=503)
        if not payload["stream"]:
            if fault == "recovery" and "recovery" in payload["prompt"]:
                return web.Response(status=503)
            return web.json_response(
                {
                    "choices": [
                        {
                            "text": "" if fault == "empty" else "blue",
                            "finish_reason": "stop",
                        }
                    ]
                }
            )
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)

        async def chunk(text, reason=None):
            data = {"choices": [{"text": text, "finish_reason": reason}]}
            await response.write(("data: " + json.dumps(data) + "\n\n").encode())

        try:
            if fault in ("stall", "interrupt"):
                await asyncio.sleep(0.15)
            if fault == "sse_error":
                await response.write(b'data: {"error":"transfer failed"}\n\n')
                return response
            if fault == "malformed":
                await response.write(b"data: {not-json}\n\n")
                return response
            if fault == "bad_finish":
                await chunk("blue", "abort")
                return response
            if fault in ("buffered_eof", "buffered_done"):
                await response.write(
                    b'data: {"choices":[{"text":"blue","finish_reason":null}]}\n\n'
                    b'data: {"choices":[{"text":"","finish_reason":"stop"}]}\n\n'
                    b"data: [DONE]\n\n"
                )
                if fault == "buffered_done":
                    await asyncio.sleep(0.02)  # terminal SSE buffered before HTTP EOF
                return response
            await chunk("blue", "stop" if fault == "already_done" else None)
            if fault == "truncated":
                return response
            # Ensure disconnect can happen while the response is unfinished.
            await asyncio.sleep(0.02)
            await chunk("", "stop")
            await response.write(b"data: [DONE]\n\n")
        except (ConnectionResetError, RuntimeError):
            pass  # expected when the probe disconnects
        return response

    app = web.Application()
    app.router.add_post("/v1/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        if cli:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "helpers/pd_lifecycle.py"),
                "--router-url",
                f"http://127.0.0.1:{port}",
                "--model",
                "test",
                "--concurrency",
                "1",
                "2",
                "--rounds",
                "1",
                "--timeout",
                "2",
                "--output",
                str(output),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
            assert process.returncode == (0 if fault is None else 1), (stdout, stderr)
            assert not stderr
            assert (b"passed" if fault is None else b"failed") in stdout
            assert str(output) in stdout.decode()
            return json.loads(output.read_text()), calls
        task = asyncio.create_task(
            probe(
                f"http://127.0.0.1:{port}",
                "test",
                [1, 2],
                1,
                0.05 if fault == "stall" else 2,
                output,
            )
        )
        if fault == "interrupt":

            async def wait_for_stream():
                while len(calls) < 2:
                    await asyncio.sleep(0.001)

            await asyncio.wait_for(wait_for_stream(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return json.loads(output.read_text()), calls
        result = await task
        return result, calls
    finally:
        await runner.cleanup()


def test_success_and_progress_after_disconnect():
    result, calls = asyncio.run(exercise())
    assert result["passed"]
    assert len(calls) == 12
    assert (
        sum(r["outcome"] == "disconnected_after_text" for r in result["results"]) == 3
    )
    assert not calls[-1]["stream"]  # actual post-disconnect completion
    assert result["results"][-1]["phase"] == "recovery"


@pytest.mark.parametrize(
    "fault",
    [
        "http",
        "empty",
        "sse_error",
        "malformed",
        "bad_finish",
        "truncated",
        "stall",
        "already_done",
        "buffered_eof",
        "buffered_done",
        "recovery",
    ],
)
def test_fail_closed_without_escalating_concurrency(fault):
    result, _ = asyncio.run(exercise(fault))
    assert not result["passed"]
    assert result["results"][-1]["outcome"] == "failed"
    assert all(r["concurrency"] == 1 for r in result["results"])
    if fault == "empty":
        assert result["results"][-1]["detail"] == "completion text is empty or invalid"
    if fault == "already_done":
        assert "cancellation untested" in result["results"][-1]["detail"]
    if fault in ("http", "malformed", "stall"):
        assert "detail" not in result["results"][-1]


@pytest.mark.parametrize("levels", [[], [0], [-1], [33], [True], [1] * 9])
def test_invalid_concurrency(levels):
    with pytest.raises(ValueError):
        asyncio.run(probe("http://127.0.0.1:1", "unused", levels))


@pytest.mark.parametrize("fault", [None, "http", "recovery", "stall"])
def test_durable_report(tmp_path, fault):
    output = tmp_path / "results" / "probe.json"
    result, _ = asyncio.run(exercise(fault, output))
    assert json.loads(output.read_text()) == result
    assert result["status"] == ("passed" if fault is None else "failed")
    assert not output.with_name("probe.json.tmp").exists()
    if fault in ("http", "recovery"):
        assert result["results"][-1]["http_status"] == 503


def test_interrupted_probe_keeps_completed_wave(tmp_path):
    output = tmp_path / "probe.json"
    result, _ = asyncio.run(exercise("interrupt", output))
    assert result["status"] == "running"
    assert not result["passed"]
    assert len(result["results"]) == 1
    assert result["results"][0]["phase"] == "baseline"
    assert result["results"][0]["outcome"] == "completed"


@pytest.mark.parametrize("fault", [None, "empty", "recovery"])
def test_harness_cli_exit_and_report(tmp_path, fault):
    output = tmp_path / "evidence" / "probe.json"
    result, _ = asyncio.run(exercise(fault, output, cli=True))
    assert result["passed"] == (fault is None)
    assert result["status"] == ("passed" if fault is None else "failed")
