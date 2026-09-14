"""
Live GPU integration test for the vLLM gRPC Servicer.

Skipped in CPU CI (`pytest --ignore=py_test/e2e`) and skipped whenever
vLLM is not installed. Run manually against a real GPU:

    pytest py_test/e2e/test_vllm_servicer_real_gpu.py -v
    # or:
    python py_test/e2e/test_vllm_servicer_real_gpu.py
"""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from vllm_router.vllm_servicer import HAS_VLLM

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not HAS_VLLM, reason="vLLM is required"),
]

ROUTER_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3.5-4B")
PORT = 50055
HOST = "127.0.0.1"


async def run_real_gpu_harness():
    import grpc
    from transformers import AutoTokenizer

    from vllm_router.proto import vllm_engine_pb2, vllm_engine_pb2_grpc

    print(f"=== Starting Servicer with real model {MODEL_PATH} on GPU 0 ===")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = (
        f"{ROUTER_DIR / 'py_src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    )

    cmd = [
        sys.executable,
        "-m",
        "vllm_router.vllm_servicer",
        "--model",
        MODEL_PATH,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--gpu-memory-utilization",
        "0.75",
        "--max-model-len",
        "4096",
        "--trust-remote-code",
        "--enforce-eager",
    ]

    log_file = open("/tmp/vllm_servicer_gpu_test.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)

    target = f"{HOST}:{PORT}"
    channel = grpc.aio.insecure_channel(target)
    client = vllm_engine_pb2_grpc.VllmEngineStub(channel)

    try:
        print("Waiting for Servicer to initialize and load model into VRAM...")
        healthy = False
        for attempt in range(300):
            try:
                res = await asyncio.wait_for(
                    client.HealthCheck(vllm_engine_pb2.HealthCheckRequest()),
                    timeout=2.0,
                )
                if (
                    res.status
                    == vllm_engine_pb2.HealthCheckResponse.ServingStatus.SERVING
                ):
                    healthy = True
                    print(f"Servicer healthy after {attempt * 2}s!")
                    break
            except Exception:
                pass
            if proc.poll() is not None:
                raise RuntimeError(
                    "Servicer process died unexpectedly! Check /tmp/vllm_servicer_gpu_test.log"
                )
            await asyncio.sleep(2.0)

        if not healthy:
            raise TimeoutError("Servicer failed to become healthy within timeout.")

        print("\n--- Testing RPC: GetModelInfo ---")
        info = await client.GetModelInfo(vllm_engine_pb2.ModelInfoRequest())
        print(f"Model Name: {info.model_name}")
        print(f"Max Model Len: {info.max_model_len}")
        print(f"DP Size: {info.dp_size}")
        print(f"Block Size: {info.block_size}")
        assert info.model_name == MODEL_PATH

        print(
            "\n--- Testing RPC: GenerateStream with pre-tokenized prompt_token_ids ---"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        prompt_text = "Q: What is the capital of France?\nA:"
        token_ids = tokenizer.encode(prompt_text)
        print(f"Prompt text: '{prompt_text}'")
        print(f"Tokenized IDs ({len(token_ids)} tokens): {token_ids}")

        req = vllm_engine_pb2.GenerateRequest(
            request_id="real-gpu-test-01",
            prompt_token_ids=token_ids,
            sampling_params=vllm_engine_pb2.SamplingParams(
                temperature=0.0,
                max_tokens=30,
            ),
        )

        stream = client.GenerateStream(req)
        output_tokens = []
        output_text_chunks = []
        t0 = time.perf_counter()

        async for chunk in stream:
            if chunk.HasField("token_id"):
                output_tokens.append(chunk.token_id)
            output_text_chunks.append(chunk.text_delta)
            print(chunk.text_delta, end="", flush=True)

        ttft = time.perf_counter() - t0
        full_generated_text = "".join(output_text_chunks)
        print(f"\n[Generated {len(output_tokens)} tokens in {ttft:.3f}s]")
        print(f"Full text: '{full_generated_text.strip()}'")
        assert "Paris" in full_generated_text

        print("\n--- Testing RPC: ResetPrefixCache ---")
        cache_reset = await client.ResetPrefixCache(vllm_engine_pb2.EmptyRequest())
        print(
            f"ResetPrefixCache: success={cache_reset.success}, message='{cache_reset.message}'"
        )
        assert cache_reset.success is True

        print("\n=== ALL REAL GPU gRPC TESTS PASSED PERFECTLY! ===")

    finally:
        await channel.close()
        print("Stopping Servicer process...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_file.close()


@pytest.mark.asyncio
async def test_vllm_servicer_real_gpu():
    await run_real_gpu_harness()


if __name__ == "__main__":
    if not HAS_VLLM:
        raise SystemExit("vLLM is required for this GPU harness.")
    asyncio.run(run_real_gpu_harness())
