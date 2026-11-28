"""Exercise the probe against real loopback HTTP/SSE, without a GPU or router."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from aiohttp import StreamReader, web

from ..e2e.pd_disagg_vllm import pd_lifecycle
from ..e2e.pd_disagg_vllm.pd_lifecycle import probe

pytestmark = [pytest.mark.ready, pytest.mark.cpu]


async def exercise(fault=None, output=None, cli=False, fragment=None, levels=(1, 2)):
    calls = []
    close_timeouts = []

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
        disconnect = "-disconnect-" in payload["prompt"]

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
            if (
                fault
                in (
                    "disconnect_truncated",
                    "disconnect_buffered_error",
                    "finished_truncated",
                    "finished_buffered_error",
                )
                and "-disconnect-0" in payload["prompt"]
            ):
                reason = "stop" if fault.startswith("finished_") else None
                first = (
                    "data: "
                    + json.dumps(
                        {"choices": [{"text": "blue", "finish_reason": reason}]}
                    )
                    + "\n\n"
                ).encode()
                tail = (
                    b'data: {"error":"transfer failed"}\n\n'
                    if fault.endswith("buffered_error")
                    else b""
                )
                await response.write(first + tail)
                return response
            if fragment is not None:
                first = b'data: {"choices":[{"text":"blue","finish_reason":null}]}\n\n'
                following = 'data: {"choices":[{"text":"蓝","finish_reason":null}]}\n\n'.encode()
                cut = following.index(fragment) + len(fragment)
                await response.write(first + following[:cut])
                if not disconnect:
                    await asyncio.sleep(0.02)
                    await response.write(following[cut:])
            else:
                await chunk("blue", "stop" if fault == "already_done" else None)
            if fault == "truncated":
                return response
            if disconnect and fault != "already_done":
                transport = request.transport

                async def wait_for_close():
                    while transport is not None and not transport.is_closing():
                        await asyncio.sleep(0.01)

                try:
                    await asyncio.wait_for(wait_for_close(), 2)
                except TimeoutError:
                    close_timeouts.append(payload["prompt"])
                return response
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
                str(
                    Path(__file__).resolve().parents[1]
                    / "e2e/pd_disagg_vllm/pd_lifecycle.py"
                ),
                "--router-url",
                f"http://127.0.0.1:{port}",
                "--model",
                "test",
                "--concurrency",
                "1 2",
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
            assert b"outcomes:" in stdout
            if fault == "already_done":
                assert b"reason: no live disconnect observed" in stdout
            assert str(output) in stdout.decode()
            return json.loads(output.read_text()), calls
        task = asyncio.create_task(
            probe(
                f"http://127.0.0.1:{port}",
                "test",
                list(levels),
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
        assert not close_timeouts, f"client did not disconnect: {close_timeouts}"


@pytest.mark.parametrize("fragment", [None, b"da", b'{"choices":', b"\xe8"])
def test_success_and_progress_after_disconnect(fragment, monkeypatch):
    readline = StreamReader.readline

    async def delayed_readline(self):
        # Exercise a client that cannot consume the first event immediately.
        await asyncio.sleep(0.05)
        return await readline(self)

    monkeypatch.setattr(StreamReader, "readline", delayed_readline)
    result, calls = asyncio.run(exercise(fragment=fragment))
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
    if fault in ("http", "malformed", "stall"):
        assert "detail" not in result["results"][-1]


@pytest.mark.parametrize("fault", ["already_done", "buffered_eof", "buffered_done"])
def test_all_untested_disconnects_continue_then_fail(fault):
    result, _ = asyncio.run(exercise(fault))
    assert not result["passed"]
    assert result["failure_reason"] == "no live disconnect observed"
    assert {r["concurrency"] for r in result["results"]} == {1, 2}
    disconnects = [r for r in result["results"] if r["phase"] == "disconnect"]
    assert disconnects
    assert all(r["outcome"] == "untested" for r in disconnects)
    assert result["results"][-1]["phase"] == "recovery"


def test_disconnect_wave_allows_partial_untested(monkeypatch):
    async def mixed_request(_session, _url, _model, mode, sequence):
        if mode == "disconnect" and sequence.endswith("-0"):
            raise pd_lifecycle.DisconnectUntested(
                "completion buffered; cancellation untested"
            )
        return "disconnected_after_text" if mode == "disconnect" else "completed"

    monkeypatch.setattr(pd_lifecycle, "request", mixed_request)
    result = asyncio.run(probe("http://unused", "test", [2], 1, 2))
    assert result["passed"]
    disconnects = [r for r in result["results"] if r["phase"] == "disconnect"]
    assert [r["outcome"] for r in disconnects] == [
        "untested",
        "disconnected_after_text",
    ]


@pytest.mark.parametrize(
    "fault",
    [
        "disconnect_truncated",
        "disconnect_buffered_error",
        "finished_truncated",
        "finished_buffered_error",
    ],
)
def test_disconnect_tail_failure_is_not_masked_by_peer_success(fault):
    result, _ = asyncio.run(exercise(fault, levels=(2,)))
    assert not result["passed"]
    disconnects = [r for r in result["results"] if r["phase"] == "disconnect"]
    assert {r["outcome"] for r in disconnects} == {
        "failed",
        "disconnected_after_text",
    }


def test_unexpected_request_exception_is_reported(monkeypatch, tmp_path):
    async def fail_request(*_args):
        raise RuntimeError("transport implementation failed")

    monkeypatch.setattr(pd_lifecycle, "request", fail_request)
    output = tmp_path / "probe.json"
    result = asyncio.run(probe("http://unused", "test", [1], 1, 2, output))
    assert not result["passed"]
    assert result["status"] == "failed"
    assert result["results"][0]["outcome"] == "failed"
    assert result["results"][0]["error_type"] == "RuntimeError"
    assert "detail" not in result["results"][0]
    assert json.loads(output.read_text()) == result


@pytest.mark.parametrize("levels", [[], [0], [-1], [33], [True], [1] * 9])
def test_invalid_concurrency(levels):
    with pytest.raises(ValueError):
        asyncio.run(probe("http://127.0.0.1:1", "unused", levels))


@pytest.mark.parametrize(
    "settings", [{"rounds": 0}, {"rounds": 11}, {"timeout": 0}, {"timeout": 601}]
)
def test_invalid_request_budget(settings):
    with pytest.raises(ValueError):
        asyncio.run(probe("http://127.0.0.1:1", "unused", **settings))


def test_interrupted_probe_keeps_completed_wave(tmp_path):
    output = tmp_path / "probe.json"
    result, _ = asyncio.run(exercise("interrupt", output))
    assert result["status"] == "running"
    assert not result["passed"]
    assert len(result["results"]) == 1
    assert result["results"][0]["phase"] == "baseline"
    assert result["results"][0]["outcome"] == "completed"


@pytest.mark.parametrize("fault", [None, "already_done"])
def test_harness_cli_exit_and_report(tmp_path, fault):
    output = tmp_path / "evidence" / "probe.json"
    result, _ = asyncio.run(exercise(fault, output, cli=True))
    assert not output.with_name("probe.json.tmp").exists()
    assert result["passed"] == (fault is None)
    assert result["status"] == ("passed" if fault is None else "failed")
    if fault == "already_done":
        assert result["failure_reason"] == "no live disconnect observed"
