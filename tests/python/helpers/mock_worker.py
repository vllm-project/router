"""
Lightweight mock worker HTTP server for router integration tests.

Implements minimal endpoints used by the router:
- GET /health, /health_generate
- POST /generate, /v1/completions, /v1/chat/completions
- POST /flush_cache
- GET /get_server_info, /get_model_info, /v1/models

Behavior knobs are controlled via CLI flags to simulate failures, latency, and load.
"""

import argparse
import asyncio
import json
import os
import random
import signal
import sys
import threading
import time
from contextlib import asynccontextmanager
from hashlib import sha256
from typing import Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

# Global state (per-process)
_inflight = 0
_served = 0
_failures_seen = 0
_requests_seen = 0


KV_HASH_SEED = "vllm-none-hash"


def kv_store(
    tokens, block_size, *, medium="GPU", parent=None, adapter=None, ownership=None
):
    """Build a vLLM incremental store with full-width, group-0 hashes."""
    import cbor2

    assert tokens and len(tokens) % block_size == 0
    name, path = adapter or (None, None)
    extra_keys = (("lora", name, path),) if adapter else None
    hashes = []
    previous = (
        parent if parent is not None else sha256(cbor2.dumps(KV_HASH_SEED)).digest()
    )
    for offset in range(0, len(tokens), block_size):
        previous = sha256(
            cbor2.dumps((previous, tokens[offset : offset + block_size], extra_keys))
        ).digest()
        hashes.append(previous)
    event = dict(
        type="BlockStored",
        block_hashes=hashes,
        parent_block_hash=parent,
        token_ids=list(tokens),
        block_size=block_size,
        lora_id=None,
        lora_name=name,
        medium=medium,
        group_idx=0,
        kv_cache_spec_kind="full_attention",
    )
    if ownership is not None:
        event["ownership"] = ownership
    return event


class KvEventPublisher:
    """An XPUB live stream and a vLLM-compatible ROUTER replay endpoint."""

    def __init__(self):
        import zmq

        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.XPUB)
        self.replay = self.context.socket(zmq.ROUTER)
        for socket in (self.publisher, self.replay):
            socket.setsockopt(zmq.LINGER, 0)
            socket.bind_to_random_port("tcp://127.0.0.1")
        self.endpoint = self.publisher.last_endpoint.decode()
        self.replay_endpoint = self.replay.last_endpoint.decode()
        self.topic = f"kv/{uuid4().hex}".encode()
        self.sequence = 0
        self.subscribed = False
        self.history = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._replay, daemon=True)
        self.thread.start()

    def source(self):
        return dict(
            endpoint=self.endpoint,
            replay_endpoint=self.replay_endpoint,
            topic=self.topic.decode(),
            encoding="msgpack",
            schema_version=1,
        )

    def wait_subscribed(self, timeout=10):
        if not self.subscribed:
            assert self.publisher.poll(int(timeout * 1000)), (
                "KV subscriber did not attach"
            )
            assert self.publisher.recv() == b"\x01" + self.topic
            self.subscribed = True

    def publish(self, events, sequence=None, *, live=True, replay=True):
        import msgpack

        with self.lock:
            if sequence == 0:
                self.history.clear()
            if sequence is not None:
                self.sequence = sequence
            published = self.sequence
            seq = published.to_bytes(8, "big")
            payload = msgpack.packb([time.time(), events, 0], use_bin_type=True)
            if replay:
                self.history[published] = (seq, payload)
            self.sequence += 1
        if live:
            self.publisher.send_multipart([self.topic, seq, payload])
        return published

    def _replay(self):
        while not self.stop.is_set():
            if not self.replay.poll(50):
                continue
            sender, envelope, start = self.replay.recv_multipart()
            assert envelope == b"" and len(start) == 8
            start = int.from_bytes(start, "big")
            with self.lock:
                frames = [frame for seq, frame in self.history.items() if seq >= start]
            for seq, payload in frames:
                self.replay.send_multipart([sender, b"", self.topic, seq, payload])
            self.replay.send_multipart([sender, b"", b"", b"\xff" * 8, b""])

    def close(self):
        self.stop.set()
        self.thread.join()
        self.publisher.close()
        self.replay.close()
        self.context.term()


def mock_tokenize(text):
    """Mirror the local tokenizer the KV tests use: ``t7`` -> 8, else unknown."""
    tokens = []
    for word in text.split():
        body = word[1:]
        tokens.append(int(body) + 1 if word.startswith("t") and body.isdigit() else 0)
    return tokens


def _native_layout_field(layout: bytes) -> bytes:
    """Encode the additive field in the Router's Control RPC response."""
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    descriptor = descriptor_pb2.FileDescriptorProto(name="test_native_kv_layout.proto")
    message = descriptor.message_type.add(name="NativeKvLayout")
    message.field.add(
        name="kv_routing_capabilities",
        number=2,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
        type=descriptor_pb2.FieldDescriptorProto.TYPE_BYTES,
    )
    pool = descriptor_pool.DescriptorPool()
    field_message = message_factory.GetMessageClass(
        pool.Add(descriptor).message_types_by_name["NativeKvLayout"]
    )
    return field_message(kv_routing_capabilities=layout).SerializeToString()


class GrpcKvWorker:
    """Network mock for KV policy STs: real gRPC discovery/inference and ZMQ.

    Bindings are generated by the fixture from Cargo.lock's vllm-proto crate.
    Imports are local so the existing HTTP mock remains independently usable.
    The test thread owns the XPUB socket; gRPC callbacks run in server threads.
    """

    def __init__(
        self,
        bindings,
        worker_id,
        block_size=2,
        model="test-model",
        events_enabled=True,
        adapters=None,
    ):
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        import grpc
        from grpc_health.v1 import health, health_pb2, health_pb2_grpc

        self.control, control_grpc, self.inference, inference_grpc = bindings
        self.worker_id = worker_id
        self.block_size = block_size
        self.model = model
        self.adapters = adapters or {}
        self.failures_remaining = 0
        self.captured = []
        self.condition = threading.Condition()
        self.gates = deque()
        self.all_gates = []
        self.events = KvEventPublisher() if events_enabled else None
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.server = grpc.server(self.executor)
        control_service = control_grpc.ControlServicer()
        for method in ("GetServerInfo", "GetModelInfo", "GetKvEventSources", "Abort"):
            setattr(control_service, method, getattr(self, method))
        inference_service = inference_grpc.InferenceServicer()
        inference_service.Generate = self.Generate
        inference_service.GenerateStream = self.GenerateStream
        control_grpc.add_ControlServicer_to_server(control_service, self.server)
        inference_grpc.add_InferenceServicer_to_server(inference_service, self.server)
        reporter = health.HealthServicer()
        reporter.set("", health_pb2.HealthCheckResponse.SERVING)
        reporter.set("vllm.Inference", health_pb2.HealthCheckResponse.SERVING)
        health_pb2_grpc.add_HealthServicer_to_server(reporter, self.server)
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.url = f"grpc://127.0.0.1:{port}"
        self.server.start()

    def GetServerInfo(self, request, context):
        return self.control.ServerInfo(
            instance_id=self.worker_id, kv_block_size=self.block_size
        )

    def GetModelInfo(self, request, context):
        return self.control.ModelInfo(
            model_id=self.model, supports_token_ids_input=True
        )

    def GetKvEventSources(self, request, context):
        if self.events is None:
            return self.control.GetKvEventSourcesResponse()
        source = self.events.source()
        response = self.control.GetKvEventSourcesResponse(
            sources=[
                self.control.KvEventSource(
                    transport="zmq", data_parallel_rank=0, **source
                )
            ]
        )
        # Adapter metadata supplements the default group-0 discovery contract.
        if self.adapters:
            capabilities = json.dumps(
                {
                    "schema_version": 1,
                    "adapters": [
                        {"name": name, "path": path}
                        for name, path in self.adapters.items()
                    ],
                    "ranks": [
                        {
                            "rank": 0,
                            "declared_groups": 1,
                            "model": self.model,
                            "weight_version": "test",
                            "publisher_topic": source["topic"],
                            "instance_topic": True,
                            "hash_algorithm": "sha256_cbor",
                            "hash_seed": KV_HASH_SEED,
                            "full_hashes": True,
                            "hash_block_size": self.block_size,
                            "hash_lookahead_tokens": 0,
                            "speculative_tokens": 0,
                            "prefix_caching": True,
                            "groups": [
                                {
                                    "group_idx": 0,
                                    "kind": "full_attention",
                                    "spec_type": "FullAttentionSpec",
                                    "block_size": self.block_size,
                                    "prefix_cacheable": True,
                                    "prefix_replay_tokens": 0,
                                }
                            ],
                        }
                    ],
                }
            ).encode()
            response.MergeFromString(_native_layout_field(capabilities))
        return response

    def Abort(self, request, context):
        return self.control.AbortResponse()

    def hold_next(self):
        """Gate one request independently at prefill, first output and EOF."""
        from types import SimpleNamespace

        gate = SimpleNamespace(
            entered=threading.Event(),
            output=threading.Event(),
            finish=threading.Event(),
            done=threading.Event(),
        )
        with self.condition:
            self.gates.append(gate)
            self.all_gates.append(gate)
        return gate

    def drop_gate(self, gate):
        """Forget a gate that no request claimed, so later probes are unheld."""
        with self.condition:
            try:
                self.gates.remove(gate)
            except ValueError:
                pass
        gate.output.set()
        gate.finish.set()

    def fail_next(self):
        with self.condition:
            self.failures_remaining += 1

    def GenerateStream(self, request, context):
        with self.condition:
            self.captured.append(request)
            gate = self.gates.popleft() if self.gates else None
            fail = self.failures_remaining > 0
            if fail:
                self.failures_remaining -= 1
            self.condition.notify_all()
        if fail:
            import grpc

            context.abort(grpc.StatusCode.UNAVAILABLE, "injected worker failure")
        pb = self.inference
        try:
            if gate:
                gate.entered.set()
            yield pb.GenerateResponse(
                prompt_info=pb.PromptInfo(num_prompt_tokens=len(request.token_ids.ids))
            )
            if gate:
                while not gate.output.wait(0.05):
                    if not context.is_active():
                        return
                yield pb.GenerateResponse(
                    outputs=pb.SequenceOutput(text=self.worker_id, num_tokens=1)
                )
                while not gate.finish.wait(0.05):
                    if not context.is_active():
                        return
            yield pb.GenerateResponse(
                outputs=pb.SequenceOutput(
                    text="" if gate else self.worker_id,
                    num_tokens=1,
                    finish_info=pb.FinishInfo(
                        num_output_tokens=1, finish_reason=pb.FinishInfo.STOP
                    ),
                )
            )
        finally:
            if gate:
                gate.done.set()

    def Generate(self, request, context):
        """Unary P/D prefill leg: capture the request and finish immediately.

        The router sends the same ``GenerateRequest`` as ``GenerateStream`` and
        reads only ``PromptInfo`` plus the closing ``FinishInfo``; no
        ``kv_transfer_params`` are returned, so decode runs without them.
        """
        with self.condition:
            self.captured.append(request)
            self.condition.notify_all()
        pb = self.inference
        return pb.GenerateResponse(
            prompt_info=pb.PromptInfo(num_prompt_tokens=len(request.token_ids.ids)),
            outputs=pb.SequenceOutput(
                text=self.worker_id,
                num_tokens=1,
                finish_info=pb.FinishInfo(
                    num_output_tokens=1, finish_reason=pb.FinishInfo.STOP
                ),
            ),
        )

    def wait_subscribed(self, timeout=10):
        """XPUB handshake prevents losing the initial event (not an apply ack)."""
        self.events.wait_subscribed(timeout)

    def store(self, tokens, medium="GPU", parent=None, lora_name=None, ownership=None):
        """Build engine events from actual token IDs captured over gRPC."""
        adapter = (lora_name, self.adapters[lora_name]) if lora_name else None
        return kv_store(
            tokens,
            self.block_size,
            medium=medium,
            parent=parent,
            adapter=adapter,
            ownership=ownership,
        )

    def publish(self, events, sequence=None, *, live=True, replay=True):
        return self.events.publish(events, sequence, live=live, replay=replay)

    def close(self):
        for gate in self.all_gates:
            gate.output.set()
            gate.finish.set()
        self.server.stop(0).wait(timeout=5)
        self.executor.shutdown(wait=True)
        if self.events is not None:
            self.events.close()


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
    # P/D KV-aware routing: advertise KV events and publish blocks over ZMQ.
    p.add_argument("--kv-block-size", type=int, default=0)
    p.add_argument("--kv-model", default="test-model")
    return p.parse_args()


def _extract_worker_id(args: argparse.Namespace) -> str:
    if args.worker_id:
        return str(args.worker_id)
    # default to port (unique enough for tests)
    return f"worker-{args.port}"


def create_app(args: argparse.Namespace) -> FastAPI:
    kv_events = KvEventPublisher() if args.kv_block_size > 0 else None

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            if kv_events is not None:
                kv_events.close()

    app = FastAPI(lifespan=lifespan)
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
        global _inflight, _served
        _inflight += 1
        _served += 1
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
        model = args.kv_model if args.kv_block_size > 0 else "mock"
        return JSONResponse({"data": [{"id": model, "object": "model"}]})

    @app.get("/get_server_info")
    async def get_server_info(request: Request):
        # Enforce API key on server info when required (used by router for worker discovery)
        check_api_key(request)
        payload = {
            "worker_id": worker_id,
            "load_in_flight": _inflight,
            "cache": {"size": 0, "hit_rate": 0.0},
            "dp_size": int(args.dp_size),
        }
        if args.kv_block_size > 0:
            payload["kv_block_size"] = args.kv_block_size
            payload["model_id"] = args.kv_model
            payload["instance_id"] = worker_id
        return JSONResponse(payload)

    @app.get("/kv_event_sources")
    async def kv_event_sources():
        if kv_events is None:
            return JSONResponse({}, status_code=404)
        return JSONResponse(
            {
                "0": {
                    "publisher": "zmq",
                    "enable_kv_cache_events": True,
                    **kv_events.source(),
                }
            }
        )

    @app.post("/tokenize")
    async def tokenize(request: Request):
        body = await request.json()
        tokens = mock_tokenize(str(body.get("prompt") or body.get("text") or ""))
        return JSONResponse({"tokens": tokens, "count": len(tokens)})

    @app.post("/kv_wait_subscribed")
    async def kv_wait_subscribed():
        """Wait for the Router's XPUB handshake off the event loop."""
        if kv_events is None:
            raise HTTPException(status_code=400, detail="KV events are disabled")
        try:
            await asyncio.to_thread(kv_events.wait_subscribed)
        except AssertionError as error:
            raise HTTPException(status_code=504, detail=str(error)) from error
        return JSONResponse({"subscribed": True, "topic": kv_events.topic.decode()})

    @app.get("/kv_status")
    async def kv_status():
        """How many batches this worker published and whether a peer is attached."""
        return JSONResponse(
            {
                "subscribed": bool(kv_events and kv_events.subscribed),
                "published": kv_events.sequence if kv_events else 0,
                "topic": kv_events.topic.decode() if kv_events else "",
            }
        )

    @app.post("/kv_store")
    async def publish_kv_store(request: Request):
        """Publish a BlockStored batch for tokens this worker has cached."""
        if kv_events is None:
            raise HTTPException(status_code=400, detail="KV events are disabled")
        body = await request.json()
        tokens = [int(token) for token in body["token_ids"]]
        kv_events.publish([kv_store(tokens, args.kv_block_size)])
        return JSONResponse({"published": 1})

    @app.get("/get_load")
    async def get_load():
        return JSONResponse({"load": _inflight})

    @app.get("/request_count")
    async def request_count():
        """Completions served so far; P/D tests use deltas to see the pick."""
        return JSONResponse({"count": _served})

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
