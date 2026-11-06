"""
Lightweight mock worker HTTP server for router integration tests.

Implements minimal endpoints used by the router:
- GET /health, /health_generate
- POST /generate, /v1/completions, /v1/chat/completions
- POST /flush_cache
- GET /get_server_info, /get_model_info, /v1/models

Behavior knobs are controlled via CLI flags to simulate failures, latency, and load.
``--pd-role prefill|decode`` returns the usage and logprobs fixtures used by
the PD response-merge tests. ``--nixl-push`` marks a prefill response as NIXL push.
"""

import argparse
import asyncio
import json
import os
import random
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

# Global state (per-process)
_inflight = 0
_failures_seen = 0
_requests_seen = 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--worker-id", default=None)
    p.add_argument("--latency-ms", type=int, default=0)
    p.add_argument("--hold-after", type=int, default=0)
    p.add_argument("--release-file", default=None)
    p.add_argument("--timeout", action="store_true")
    p.add_argument("--status-code", type=int, default=200)
    p.add_argument("--fail-first-n", type=int, default=0)
    p.add_argument("--random-fail-rate", type=float, default=0.0)
    p.add_argument("--require-api-key", action="store_true")
    p.add_argument("--api-key", default=None)
    p.add_argument("--max-payload-bytes", type=int, default=10 * 1024 * 1024)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--dp-size", type=int, default=1)
    p.add_argument("--crash-on-request", action="store_true")
    p.add_argument("--health-fail-after-ms", type=int, default=0)
    # PD response-merge fixtures. Prefill owns input usage; decode owns output usage.
    p.add_argument("--pd-role", choices=["prefill", "decode"], default=None)
    p.add_argument("--nixl-push", action="store_true")
    return p.parse_args()


def _pd_usage(path: str, role: str) -> dict:
    """Usage blocks from the in-process PD merge fixture in test_pd_routing.rs."""
    responses = path.endswith("/responses")
    if role == "prefill":
        if responses:
            return {
                "input_tokens": 70,
                "output_tokens": 1,
                "total_tokens": 71,
                "input_tokens_details": {"cached_tokens": 4},
            }
        return {
            "prompt_tokens": 70,
            "completion_tokens": 1,
            "total_tokens": 71,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
    if responses:
        return {
            "input_tokens": 999,
            "output_tokens": 8,
            "total_tokens": 1007,
            "output_tokens_details": {"reasoning_tokens": 2},
        }
    return {
        "prompt_tokens": 999,
        "completion_tokens": 8,
        "total_tokens": 1007,
        "prompt_tokens_details": {"cached_tokens": 99, "audio_tokens": 2},
    }


def _wants_logprobs(data: dict) -> bool:
    return "logprobs" in data or data.get("echo") is True


def _pd_json(args: argparse.Namespace, path: str, data: dict, worker_id: str) -> dict:
    """Prefill or decode JSON whose usage and logprobs the router must merge."""
    usage = _pd_usage(path, args.pd_role)
    body: dict = {"choices": [{"text": "answer", "index": 0}], "usage": usage}
    if args.pd_role == "prefill":
        params = {
            "remote_engine_id": "P",
            "do_remote_prefill": True,
        }
        if args.nixl_push:
            params["remote_host"] = "127.0.0.1"
            params["remote_port"] = 9100
            params["transfer_mode"] = "push"
        body["kv_transfer_params"] = params
    if _wants_logprobs(data) and path.endswith("/chat/completions"):
        if args.pd_role == "prefill":
            # Chat Completions keeps prompt_logprobs at the top level.
            body["prompt_logprobs"] = [None, -0.5, -1.2]
    elif (
        _wants_logprobs(data)
        and not path.endswith("/responses")
        and args.pd_role == "prefill"
    ):
        body["choices"] = [
            {
                "text": "prefill",
                "index": 0,
                "prompt_logprobs": [None, -0.5, -1.2],
                "logprobs": {
                    "token_logprobs": [None, -0.5, -1.2, -2.1],
                    "tokens": ["Hello", " world", " test", " extra"],
                    "text_offset": [0, 5, 11, 16],
                    "top_logprobs": [
                        None,
                        {" world": -0.5},
                        {" test": -1.2},
                        {" extra": -2.1},
                    ],
                },
            }
        ]
    elif (
        _wants_logprobs(data)
        and not path.endswith("/responses")
        and not path.endswith("/chat/completions")
    ):
        body["choices"][0]["logprobs"] = {
            "token_logprobs": [-3.5, -4.2],
            "tokens": [" output", " token"],
            "text_offset": [0, 7],
            "top_logprobs": [{" output": -3.5}, {" token": -4.2}],
        }
    body["worker_id"] = worker_id
    return body


def _pd_sse(path: str, role: str) -> str:
    """Two SSE frames then [DONE], matching the Rust PD usage fixture."""
    usage = _pd_usage(path, role)
    if path.endswith("/responses"):
        delta = {"type": "response.output_text.delta", "delta": "answer"}
        final = {"type": "response.completed", "response": {"usage": usage}}
    else:
        delta = {"choices": [{"index": 0, "text": "answer"}], "usage": None}
        final = {"choices": [], "usage": usage}
    return f"data: {json.dumps(delta)}\n\ndata: {json.dumps(final)}\n\ndata: [DONE]\n\n"


def _extract_worker_id(args: argparse.Namespace) -> str:
    if args.worker_id:
        return str(args.worker_id)
    # default to port (unique enough for tests)
    return f"worker-{args.port}"


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI()
    worker_id = _extract_worker_id(args)
    start_ts = time.time()
    crashed = {"done": False}

    async def maybe_delay():
        if args.latency_ms > 0:
            await asyncio.sleep(args.latency_ms / 1000.0)

    async def maybe_hold():
        global _requests_seen
        _requests_seen += 1
        if (
            args.hold_after > 0
            and args.release_file
            and _requests_seen > args.hold_after
        ):
            while not os.path.exists(args.release_file):
                await asyncio.sleep(0.02)

    def should_fail() -> Optional[int]:
        global _failures_seen
        # Fail first N requests (500)
        if args.fail_first_n > 0 and _failures_seen < args.fail_first_n:
            _failures_seen += 1
            return 500
        # Random failure probability (500)
        if args.random_fail_rate > 0.0 and random.random() < args.random_fail_rate:
            return 500
        # Forced status code override (non-200) for all responses
        if args.status_code != 200:
            return int(args.status_code)
        return None

    def check_api_key(request: Request):
        if not args.require_api_key:
            return
        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Unauthorized")
        key = auth.split(" ", 1)[1]
        if args.api_key and key != args.api_key:
            raise HTTPException(status_code=401, detail="Unauthorized")

    @asynccontextmanager
    async def track_inflight():
        global _inflight
        _inflight += 1
        try:
            yield
        finally:
            _inflight -= 1

    @app.get("/health")
    async def health():
        if (
            args.health_fail_after_ms
            and (time.time() - start_ts) * 1000.0 >= args.health_fail_after_ms
        ):
            return PlainTextResponse("bad", status_code=500)
        return PlainTextResponse("ok", status_code=200)

    @app.get("/health_generate")
    async def health_generate():
        return PlainTextResponse("ok", status_code=200)

    @app.post("/flush_cache")
    async def flush_cache():
        return PlainTextResponse("ok", status_code=200)

    @app.get("/get_model_info")
    async def get_model_info():
        return JSONResponse({"model": "mock", "vocab_size": 32000})

    @app.get("/v1/models")
    async def list_models():
        return JSONResponse({"data": [{"id": "mock", "object": "model"}]})

    @app.get("/get_server_info")
    async def get_server_info(request: Request):
        # Enforce API key on server info when required (used by router for worker discovery)
        check_api_key(request)
        return JSONResponse(
            {
                "worker_id": worker_id,
                "load_in_flight": _inflight,
                "cache": {"size": 0, "hit_rate": 0.0},
                "dp_size": int(args.dp_size),
            }
        )

    @app.get("/get_load")
    async def get_load():
        return JSONResponse({"load": _inflight})

    @app.get("/metrics")
    async def metrics():
        # Program scheduling scrapes vLLM's Prometheus gauges. Zeros keep the
        # scheduler able to admit while this process stays a routing mock.
        lines = []
        for rank in range(int(args.dp_size)):
            lines.append(f'vllm:kv_cache_usage_perc{{engine="{rank}"}} 0')
            lines.append(f'vllm:num_requests_running{{engine="{rank}"}} 0')
            lines.append(f'vllm:num_requests_waiting{{engine="{rank}"}} 0')
        return PlainTextResponse("\n".join(lines) + "\n")

    def make_json_response(obj: dict, status_code: int = 200) -> JSONResponse:
        resp = JSONResponse(obj, status_code=status_code)
        resp.headers["X-Worker-Id"] = worker_id
        return resp

    async def handle_text_request(request: Request):
        # Authorization
        check_api_key(request)

        # Payload limit
        body = await request.body()
        if len(body) > args.max_payload_bytes:
            return make_json_response({"error": "payload too large"}, status_code=413)

        # Simulate crash on first request
        if args.crash_on_request and not crashed["done"]:
            crashed["done"] = True
            os._exit(1)

        # Optional timeout (simulate hang)
        if args.timeout:
            await asyncio.sleep(3600)

        # Hold later requests until the test publishes the release file.
        # The in-flight counter already includes this request.
        await maybe_hold()

        # Optional latency
        await maybe_delay()

        # Optional failures
        fail_code = should_fail()
        if fail_code is not None and fail_code != 200:
            return make_json_response(
                {"error": f"mock failure {fail_code}"}, status_code=fail_code
            )

        # Build response echoing minimal shape
        try:
            data = await request.json()
        except (json.JSONDecodeError, ValueError):
            data = {}

        now = time.time()
        ret = {
            "id": f"cmpl-{int(now * 1000)}",
            "object": "text_completion",
            "created": int(now),
            "model": "mock",
            "choices": [
                {
                    "text": "ok",
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "worker_id": worker_id,
            "echo": data,
            # Helpful for middleware e2e assertions (e.g. WASM header injection).
            "request_headers": {
                k: v
                for k, v in request.headers.items()
                if k.lower().startswith("x-wasm")
                or k.lower() in {"content-type", "authorization"}
            },
        }
        return make_json_response(ret, status_code=200)

    async def handle_pd_request(request: Request):
        # Same failure knobs as the generic handler. The body is a PD fixture
        # instead of an echo, so the router can merge prefill and decode.
        check_api_key(request)
        raw = await request.body()
        if len(raw) > args.max_payload_bytes:
            return make_json_response({"error": "payload too large"}, status_code=413)
        if args.crash_on_request and not crashed["done"]:
            crashed["done"] = True
            os._exit(1)
        if args.timeout:
            await asyncio.sleep(3600)
        await maybe_hold()
        await maybe_delay()
        fail_code = should_fail()
        if fail_code is not None and fail_code != 200:
            return make_json_response(
                {"error": f"mock failure {fail_code}"}, status_code=fail_code
            )
        try:
            data = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        path = request.url.path
        # Prefill is always one buffered JSON response. Decode follows the client.
        if args.pd_role == "decode" and data.get("stream") is True:
            headers = {"X-Worker-Id": worker_id}

            async def gen():
                yield _pd_sse(path, args.pd_role)

            return StreamingResponse(
                gen(), media_type="text/event-stream", headers=headers
            )
        return make_json_response(_pd_json(args, path, data, worker_id))

    async def handle_stream_request(request: Request):
        check_api_key(request)

        async def gen():
            # minimal 2-chunk stream then [DONE]
            for i in range(2):
                await asyncio.sleep(0.01)
                chunk = {
                    "choices": [{"delta": {"content": "x"}}],
                    "worker_id": worker_id,
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        headers = {"X-Worker-Id": worker_id}
        return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

    async def dispatch(request: Request):
        if args.pd_role:
            return await handle_pd_request(request)
        if args.stream:
            return await handle_stream_request(request)
        return await handle_text_request(request)

    @app.post("/generate")
    async def generate(request: Request):
        async with track_inflight():
            return await dispatch(request)

    @app.post("/v1/completions")
    async def completions(request: Request):
        async with track_inflight():
            return await dispatch(request)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        async with track_inflight():
            return await dispatch(request)

    @app.post("/v1/responses")
    async def responses(request: Request):
        async with track_inflight():
            return await dispatch(request)

    return app


def main() -> None:
    args = _parse_args()
    app = create_app(args)
    # Handle SIGTERM gracefully for fast test teardown
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
