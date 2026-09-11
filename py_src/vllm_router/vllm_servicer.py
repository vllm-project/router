"""
vLLM gRPC Servicer for vllm-router.

Exposes AsyncLLMEngine via gRPC/HTTP2 using the VllmEngine contract.
Enables high-performance binary transport, prompt_token_ids ingestion,
and native cluster admin controls.
"""

import argparse
import asyncio
import logging
import signal
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator

import grpc

from vllm_router.proto import vllm_engine_pb2, vllm_engine_pb2_grpc

try:
    from vllm import SamplingParams, TokensPrompt
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


try:
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc

    HAS_GRPC_HEALTH = True
except ImportError:
    HAS_GRPC_HEALTH = False

try:
    import setproctitle
except ImportError:
    setproctitle = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("vllm_servicer")


class VllmEngineServicer(vllm_engine_pb2_grpc.VllmEngineServicer):
    """
    gRPC Servicer implementing vllm.engine.v1.VllmEngine.
    Dispatches generation, metadata discovery, health checks, and admin controls.
    """

    def __init__(
        self,
        engine,
        model_name: str,
        max_model_len: int = 4096,
        dp_size: int = 1,
        block_size: int = 16,
    ):
        self.engine = engine
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.dp_size = dp_size
        self.block_size = block_size
        self._running_requests: int = 0
        self._waiting_requests: int = 0

    def _get_metrics(self) -> vllm_engine_pb2.WorkerMetrics:
        """Collect current queue and VRAM metrics from engine or internal counters."""
        kv_usage = 0.0
        running = self._running_requests
        waiting = self._waiting_requests

        try:
            # Query stats from engine if available
            if hasattr(self.engine, "get_stats"):
                stats = self.engine.get_stats()
                kv_usage = getattr(stats, "gpu_cache_usage", 0.0)
                waiting = getattr(stats, "num_waiting_sys", waiting)
                running = getattr(stats, "num_running_sys", running)
            elif (
                hasattr(self.engine, "stat_logger")
                and self.engine.stat_logger is not None
            ):
                stats = getattr(self.engine.stat_logger, "stats", None)
                if stats:
                    kv_usage = getattr(stats, "gpu_cache_usage", 0.0)
                    waiting = getattr(stats, "num_waiting_sys", waiting)
                    running = getattr(stats, "num_running_sys", running)
        except Exception:
            pass

        return vllm_engine_pb2.WorkerMetrics(
            running_requests=running,
            waiting_requests=waiting,
            kv_cache_usage_percent=float(kv_usage),
        )

    def _build_sampling_params(self, pb_params: vllm_engine_pb2.SamplingParams):
        """Convert Protobuf SamplingParams to vLLM SamplingParams."""
        if getattr(self.engine, "is_mock", False):
            return None
        if not HAS_VLLM:
            logger.critical(
                "vLLM is not installed. Real engine requires vLLM SamplingParams."
            )
            raise RuntimeError(
                "vLLM is not installed in the current Python environment."
            )

        kwargs = {
            "temperature": pb_params.temperature if pb_params.temperature > 0 else 0.0,
            "top_p": pb_params.top_p if pb_params.top_p > 0 else 1.0,
            "max_tokens": pb_params.max_tokens if pb_params.max_tokens > 0 else 16,
            "frequency_penalty": pb_params.frequency_penalty,
            "presence_penalty": pb_params.presence_penalty,
            "ignore_eos": pb_params.ignore_eos,
        }
        if pb_params.stop_sequences:
            kwargs["stop"] = list(pb_params.stop_sequences)
        if pb_params.stop_token_ids:
            kwargs["stop_token_ids"] = list(pb_params.stop_token_ids)
        if pb_params.top_k > 0:
            kwargs["top_k"] = pb_params.top_k
        if pb_params.HasField("seed"):
            kwargs["seed"] = pb_params.seed

        return SamplingParams(**kwargs)

    async def GenerateStream(
        self,
        request: vllm_engine_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[vllm_engine_pb2.GenerateStreamResponse, None]:
        """
        Stream generated tokens for incoming GenerateRequest over gRPC HTTP/2.
        Supports pre-tokenized prompt_token_ids, cancellation propagation,
        and multi-token delta emission for speculative/multi-step outputs.
        """
        # Validate data-parallel rank
        if request.dp_rank >= self.dp_size:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Requested dp_rank {request.dp_rank} is out of bounds for worker with dp_size {self.dp_size}.",
            )
            return

        # Validate execution mode (P/D disaggregation scheduled for later PRs)
        if request.execution_mode != vllm_engine_pb2.ExecutionMode.NORMAL:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                f"ExecutionMode {vllm_engine_pb2.ExecutionMode.Name(request.execution_mode)} "
                "is not supported in PR 1 (scheduled for disaggregated P/D in future PRs).",
            )
            return

        # Validate multimodal input (scheduled for later PRs)
        if request.multimodal_data:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "Multimodal generation is not supported in PR 1.",
            )
            return

        request_id = request.request_id or f"req-{uuid.uuid4().hex[:12]}"
        prompt_token_ids = list(request.prompt_token_ids)

        if not prompt_token_ids and request.prompt_text:
            prompt = request.prompt_text
        elif prompt_token_ids:
            if getattr(self.engine, "is_mock", False):
                prompt = {"prompt_token_ids": prompt_token_ids}
            elif HAS_VLLM:
                prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
            else:
                await context.abort(
                    grpc.StatusCode.INTERNAL,
                    "vLLM is not installed on this worker.",
                )
                return
        else:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Either prompt_token_ids or prompt_text must be provided.",
            )
            return

        sampling_params = self._build_sampling_params(request.sampling_params)

        self._waiting_requests += 1
        is_first_chunk = True
        prev_text = ""
        prev_token_count = 0

        try:
            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            async for request_output in generator:
                if is_first_chunk:
                    self._waiting_requests = max(0, self._waiting_requests - 1)
                    self._running_requests += 1
                    is_first_chunk = False

                # Check client cancellation
                if context.cancelled():
                    logger.info(f"Client cancelled stream for request {request_id}")
                    if hasattr(self.engine, "abort"):
                        await self.engine.abort(request_id)
                    break

                outputs = request_output.outputs
                if not outputs:
                    continue

                output = outputs[0]
                current_text = output.text
                current_token_ids = output.token_ids

                # Compute text and token deltas
                text_delta = current_text[len(prev_text) :]
                prev_text = current_text

                new_token_ids = current_token_ids[prev_token_count:]
                prev_token_count = len(current_token_ids)

                is_finished = request_output.finished
                finish_reason = output.finish_reason or ""

                if not new_token_ids:
                    response = vllm_engine_pb2.GenerateStreamResponse(
                        request_id=request_id,
                        text_delta=text_delta,
                        is_finished=is_finished,
                        finish_reason=finish_reason,
                        metrics=self._get_metrics(),
                    )
                    yield response
                elif len(new_token_ids) == 1:
                    response = vllm_engine_pb2.GenerateStreamResponse(
                        request_id=request_id,
                        token_id=new_token_ids[0],
                        text_delta=text_delta,
                        is_finished=is_finished,
                        finish_reason=finish_reason,
                        metrics=self._get_metrics(),
                    )
                    yield response
                else:
                    # Multi-step or speculative decoding produced multiple tokens in one step.
                    # Emit each token so no tokens are dropped from the stream.
                    for idx, tok_id in enumerate(new_token_ids):
                        is_last = idx == len(new_token_ids) - 1
                        response = vllm_engine_pb2.GenerateStreamResponse(
                            request_id=request_id,
                            token_id=tok_id,
                            text_delta=text_delta if is_last else "",
                            is_finished=is_finished if is_last else False,
                            finish_reason=finish_reason if is_last else "",
                            metrics=self._get_metrics(),
                        )
                        yield response

        except asyncio.CancelledError:
            logger.info(f"Stream cancelled by runtime for request {request_id}")
            if hasattr(self.engine, "abort"):
                await self.engine.abort(request_id)
            raise
        except Exception as e:
            logger.exception(
                f"Error during GenerateStream for request {request_id}: {e}"
            )
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
        finally:
            if is_first_chunk:
                self._waiting_requests = max(0, self._waiting_requests - 1)
            else:
                self._running_requests = max(0, self._running_requests - 1)

    async def Generate(
        self,
        request: vllm_engine_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.GenerateResponse:
        """Unary generation returning full output in a single response."""
        accumulated_text = []
        accumulated_token_ids = []
        finish_reason = ""
        if not request.request_id:
            request.request_id = f"req-{uuid.uuid4().hex[:12]}"
        request_id = request.request_id

        async for chunk in self.GenerateStream(request, context):
            if chunk.text_delta:
                accumulated_text.append(chunk.text_delta)
            if chunk.HasField("token_id"):
                accumulated_token_ids.append(chunk.token_id)
            if chunk.is_finished:
                finish_reason = chunk.finish_reason

        return vllm_engine_pb2.GenerateResponse(
            request_id=request_id,
            output_token_ids=accumulated_token_ids,
            output_text="".join(accumulated_text),
            finish_reason=finish_reason,
            metrics=self._get_metrics(),
        )

    async def GetModelInfo(
        self,
        request: vllm_engine_pb2.ModelInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.ModelInfoResponse:
        """Returns model metadata and capability info for router discovery."""
        return vllm_engine_pb2.ModelInfoResponse(
            model_name=self.model_name,
            max_model_len=self.max_model_len,
            dp_size=self.dp_size,
            block_size=self.block_size,
            stop_tokens=[],
        )

    async def HealthCheck(
        self,
        request: vllm_engine_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.HealthCheckResponse:
        """Probes engine health and serving status."""
        status = vllm_engine_pb2.HealthCheckResponse.ServingStatus.SERVING
        try:
            if hasattr(self.engine, "check_health"):
                await self.engine.check_health()
        except Exception as e:
            logger.warning(f"Engine health check failed: {e}")
            status = vllm_engine_pb2.HealthCheckResponse.ServingStatus.NOT_SERVING

        return vllm_engine_pb2.HealthCheckResponse(status=status)

    async def ResetPrefixCache(
        self,
        request: vllm_engine_pb2.EmptyRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.AdminResponse:
        """Clears prefix KV cache blocks in VRAM."""
        try:
            if hasattr(self.engine, "reset_prefix_cache"):
                await self.engine.reset_prefix_cache()
                return vllm_engine_pb2.AdminResponse(
                    success=True, message="Prefix cache reset successfully"
                )
            return vllm_engine_pb2.AdminResponse(
                success=False, message="Engine does not support reset_prefix_cache"
            )
        except Exception as e:
            logger.error(f"Failed to reset prefix cache: {e}")
            return vllm_engine_pb2.AdminResponse(success=False, message=str(e))


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM gRPC Engine Servicer")
    parser.add_argument(
        "--model", type=str, required=True, help="Model name or local filesystem path"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host interface to bind gRPC server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port to listen for incoming gRPC HTTP/2 connections",
    )
    parser.add_argument(
        "--dp-size", type=int, default=1, help="Data parallel size on this node"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size per instance",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory fraction for vLLM",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=None, help="Maximum context length"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from HuggingFace",
    )
    parser.add_argument(
        "--enforce-eager", action="store_true", help="Enforce eager execution mode"
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable automatic prefix caching (default: enabled)",
    )
    parser.add_argument(
        "--block-size", type=int, default=16, help="Token block size for PagedAttention"
    )
    parser.add_argument(
        "--mock-engine",
        action="store_true",
        help="Run with mock engine for offline unit testing without GPU",
    )
    return parser.parse_args()


@dataclass
class _MockOutput:
    index: int
    text: str
    token_ids: list[int]
    cumulative_logprob: float = 0.0
    logprobs: Any = None
    finish_reason: str | None = None


@dataclass
class _MockRequestOutput:
    request_id: str
    prompt: Any
    prompt_token_ids: list[int]
    prompt_logprobs: Any
    outputs: list[_MockOutput]
    finished: bool


class MockAsyncEngine:
    """Mock AsyncLLMEngine for GPU-free local unit testing and CI validation."""

    is_mock: bool = True

    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_stats(self):
        """Mock engine statistics for metrics testing."""

        class MockStats:
            gpu_cache_usage = 0.15
            num_waiting_sys = 0
            num_running_sys = 1

        return MockStats()

    async def generate(self, prompt, sampling_params, request_id: str):
        words = [
            "Hello",
            " world",
            "!",
            " This",
            " is",
            " a",
            " gRPC",
            " streaming",
            " test",
            ".",
        ]
        token_ids: list[int] = []
        current_text = ""
        for i, word in enumerate(words):
            await asyncio.sleep(0.01)
            token_ids.append(1000 + i)
            current_text += word
            output = _MockOutput(
                index=0,
                text=current_text,
                token_ids=list(token_ids),
                cumulative_logprob=-0.1 * (i + 1),
                logprobs=None,
                finish_reason="stop" if i == len(words) - 1 else None,
            )
            yield _MockRequestOutput(
                request_id=request_id,
                prompt=None,
                prompt_token_ids=[1, 2, 3],
                prompt_logprobs=None,
                outputs=[output],
                finished=(i == len(words) - 1),
            )

    async def check_health(self):
        return True

    async def reset_prefix_cache(self):
        logger.info("[MockEngine] reset_prefix_cache invoked")

    async def abort(self, request_id: str):
        logger.info(f"[MockEngine] abort invoked for request {request_id}")


async def serve(args):
    if setproctitle:
        setproctitle.setproctitle(f"vllm::servicer:{args.port}")

    max_model_len = args.max_model_len or 4096
    block_size = args.block_size

    if args.mock_engine:
        logger.info("Initializing MockAsyncEngine for offline testing...")
        engine = MockAsyncEngine(model_name=args.model)
    else:
        if not HAS_VLLM:
            logger.critical(
                "vLLM is not installed in the current Python environment. "
                "Cannot start the vLLM Servicer with a real engine. "
                "Please install vllm (`pip install vllm`) or pass --mock-engine for testing."
            )
            raise SystemExit("Error: vLLM is not installed.")

        logger.info(f"Initializing AsyncLLMEngine for model: {args.model}")

        engine_args_kwargs = {
            "model": args.model,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "trust_remote_code": args.trust_remote_code,
            "enforce_eager": args.enforce_eager,
            "enable_prefix_caching": args.enable_prefix_caching,
            "block_size": block_size,
        }
        if args.dp_size > 1:
            import inspect

            sig = inspect.signature(AsyncEngineArgs)
            if "data_parallel_size" in sig.parameters:
                engine_args_kwargs["data_parallel_size"] = args.dp_size
            else:
                logger.warning(
                    f"--dp-size={args.dp_size} specified, but AsyncEngineArgs does not accept "
                    "'data_parallel_size'. For multi-process DP, launch separate Servicer "
                    "processes per GPU rank or use tensor parallelism."
                )

        if args.max_model_len is not None:
            engine_args_kwargs["max_model_len"] = args.max_model_len

        engine_args = AsyncEngineArgs(**engine_args_kwargs)
        engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Retrieve resolved max_model_len and block_size from engine config if available
        try:
            if hasattr(engine, "engine") and hasattr(engine.engine, "model_config"):
                max_model_len = engine.engine.model_config.max_model_len
            if hasattr(engine, "engine") and hasattr(engine.engine, "cache_config"):
                block_size = engine.engine.cache_config.block_size
        except Exception as e:
            logger.debug(f"Could not inspect internal engine config: {e}")

    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", 128 * 1024 * 1024),
            ("grpc.max_receive_message_length", 128 * 1024 * 1024),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.keepalive_time_ms", 10000),
            ("grpc.keepalive_timeout_ms", 5000),
            ("grpc.keepalive_permit_without_calls", True),
        ]
    )

    servicer = VllmEngineServicer(
        engine=engine,
        model_name=args.model,
        max_model_len=max_model_len,
        dp_size=args.dp_size,
        block_size=block_size,
    )
    vllm_engine_pb2_grpc.add_VllmEngineServicer_to_server(servicer, server)

    if HAS_GRPC_HEALTH:
        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
        health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
        health_servicer.set(
            "vllm.engine.v1.VllmEngine", health_pb2.HealthCheckResponse.SERVING
        )

    listen_addr = f"{args.host}:{args.port}"
    server.add_insecure_port(listen_addr)

    logger.info(
        f"Starting vLLM Engine Servicer on {listen_addr} (model={args.model}, dp_size={args.dp_size})"
    )
    await server.start()

    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received. Stopping gRPC server...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Signal handling on non-Unix platforms
            pass

    await stop_event.wait()
    logger.info("Draining gRPC server streams...")
    await server.stop(grace=5.0)
    logger.info("vLLM Servicer terminated cleanly.")


def main():
    args = parse_args()
    try:
        asyncio.run(serve(args))
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
