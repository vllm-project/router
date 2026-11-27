import json
import os
import re
import shutil
import sys
import time
from contextlib import ExitStack
from math import ceil, lcm
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest


def _ensure_repo_root() -> None:
    """``--confcutdir=.../kv_aware`` skips ``tests/python/conftest.py``."""
    root = Path(__file__).resolve().parents[5]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def pytest_configure(config):
    _ensure_repo_root()
    config.addinivalue_line("markers", "e2e: mark as end-to-end test")


@pytest.fixture
def kv_aware_two_workers():
    """``vllm-rs serve`` ×2; health via each worker OpenAI ``/health`` port."""
    from tests.python.e2e.function.helpers.workers import (
        spawn_two_vllm_rs_grpc_workers,
        terminate_workers,
    )
    from tests.python.e2e.function.kv_aware.config import (
        kv_aware_e2e_model,
        load_kv_aware_case,
    )

    case = load_kv_aware_case("qwen3_0_6b")
    assert (case["protocol"], case["worker_count"]) == ("grpc", 2)
    workers: list = []
    try:
        workers = spawn_two_vllm_rs_grpc_workers(kv_aware_e2e_model())
        yield workers
    finally:
        terminate_workers(workers)


@pytest.fixture
def kv_aware_router(kv_aware_two_workers):
    """Router with ``--worker-urls`` (regular mode requires workers at startup)."""
    from tests.python.e2e.function.helpers.router import (
        spawn_kv_aware_router,
        terminate_router,
    )
    from tests.python.e2e.function.kv_aware.config import kv_aware_e2e_model

    urls = [w.url for w in kv_aware_two_workers]
    handle = spawn_kv_aware_router(
        worker_urls=urls,
        router_model=kv_aware_e2e_model(),
    )
    try:
        yield handle
    finally:
        terminate_router(handle)


@pytest.fixture
def hybrid_kv_aware_stack(request, tmp_path):
    """Run colocated or 1P1D workers with explicit KV sources and debug logs."""
    from tests.python.e2e.function.helpers import launch
    from tests.python.e2e.function.kv_aware.config import kv_aware_gpu_pool

    case, protocol, deployment = request.param
    model = case["model"]
    assert protocol in ("http", "grpc")
    assert deployment in ("colocated", "1p1d_nixl")
    pd = deployment == "1p1d_nixl"
    worker_count = case["worker_count"]
    assert worker_count == 2, "Hybrid routing probes require two workers"
    worker_health_timeout = case.get("worker_health_timeout_sec", 600.0)
    engine_params = case["worker_params"].copy()
    tensor_parallel_size = engine_params.get("tensor_parallel_size", 1)
    assert tensor_parallel_size > 0
    required_gpus = worker_count * tensor_parallel_size
    gpus = kv_aware_gpu_pool()[:required_gpus]
    if len(gpus) < required_gpus:
        pytest.skip(f"{model} requires {required_gpus} GPUs to run two workers")
    assert len(set(gpus)) == required_gpus, f"Worker GPUs must be distinct: {gpus}"
    http_executable = shutil.which("vllm") if protocol == "http" else None
    if protocol == "http" and http_executable is None:
        pytest.fail(f"{model} KV-aware E2E requires vllm serve on PATH")

    model_path = os.environ.get(case["local_model_env"]) or model
    frontend_args = ["--served-model-name", model]
    if engine_params.pop("language_model_only", False):
        frontend_args.append("--language-model-only")
    engine_args = []
    for name, value in engine_params.items():
        flag = name.replace("_", "-")
        if isinstance(value, bool):
            engine_args.append(f"--{flag}" if value else f"--no-{flag}")
        else:
            value_arg = json.dumps(value) if isinstance(value, dict) else str(value)
            engine_args.extend((f"--{flag}", value_arg))
    layout_flag = "--kv-hybrid-layout"
    (rank,) = case["router_params"][layout_flag]["ranks"]
    hash_seed = rank["hash_seed"]
    ports_per_worker = 3 + (protocol == "grpc") + pd
    router_port, *worker_ports = launch.find_distinct_available_ports(
        1 + ports_per_worker * worker_count
    )
    router_url = f"http://127.0.0.1:{router_port}"
    router_log = tmp_path / "router.log"
    layout_path = tmp_path / "cache_layout.json"
    worker_http_urls = {}
    worker_layouts = {}
    worker_block_sizes = set()
    router_args = ["--disable-retries", "--kv-event-model", model]
    if pd:
        router_args.extend(
            [
                "--vllm-pd-disaggregation",
                "--kv-connector",
                "nixl",
                "--prefill-policy",
                "kv_aware",
                "--decode-policy",
                "round_robin",
            ]
        )
    mamba_groups = any(group["kind"] == "mamba" for group in rank["groups"])
    # NIXL leaves the last prompt token on Decode for Mamba models.
    prefill_backoff = int(pd and mamba_groups)

    def wait_worker_ready(http_url, proc, log_path, deadline):
        try:
            launch.wait_worker_health(
                http_url,
                timeout=max(0.0, deadline - time.monotonic()),
                proc=proc,
            )
        except TimeoutError as error:
            pytest.fail(f"{error}. Worker log: {log_path}")

    with ExitStack() as cleanup:
        worker_processes = []
        for index in range(worker_count):
            gpu_start = index * tensor_parallel_size
            worker_gpus = gpus[gpu_start : gpu_start + tensor_parallel_size]
            start = index * ports_per_worker
            http_port, live_port, replay_port, *extra_ports = worker_ports[
                start : start + ports_per_worker
            ]
            worker_env = {}
            worker_args = engine_args.copy()
            is_decode = pd and index == 1
            if pd:
                worker_env.update(
                    VLLM_NIXL_SIDE_CHANNEL_HOST="127.0.0.1",
                    VLLM_NIXL_SIDE_CHANNEL_PORT=str(extra_ports.pop()),
                )
                if mamba_groups:
                    worker_env["VLLM_SSM_CONV_STATE_LAYOUT"] = "DS"
                transfer = {
                    "kv_connector": "NixlConnector",
                    "kv_role": "kv_consumer" if is_decode else "kv_producer",
                    "kv_load_failure_policy": "fail",
                }
                worker_args.extend(["--kv-transfer-config", json.dumps(transfer)])
            grpc_port = extra_ports[0] if extra_ports else None
            http_url = f"http://127.0.0.1:{http_port}"
            url = f"grpc://127.0.0.1:{grpc_port}" if grpc_port else http_url
            worker_http_urls[url] = http_url
            if pd:
                router_args.extend(("--decode" if is_decode else "--prefill", url))
            if not is_decode:
                topic = uuid4().hex
                router_args.extend(
                    [
                        "--pd-kv-events" if pd else "--kv-event-source",
                        f"{url}=tcp://127.0.0.1:{live_port}",
                        "--kv-event-replay-source",
                        f"{url}=tcp://127.0.0.1:{replay_port}",
                        "--kv-event-topic",
                        f"{url}={topic}",
                    ]
                )
                worker_args.extend(
                    launch.kv_events_config_argv(
                        zmq_port=live_port, replay_port=replay_port, topic=topic
                    )
                )
            log_path = tmp_path / f"worker_{index}.log"
            log_file = cleanup.enter_context(log_path.open("w", encoding="utf-8"))
            if grpc_port is not None:
                proc = launch.popen_launch_vllm_rs_grpc_worker(
                    model_path,
                    http_port=http_port,
                    grpc_port=grpc_port,
                    base_gpu_id=worker_gpus,
                    hash_seed=hash_seed,
                    extra_argv=[
                        *frontend_args,
                        "--engine-ready-timeout-secs",
                        str(ceil(worker_health_timeout)),
                        "--",
                        *worker_args,
                    ],
                    extra_env=worker_env,
                    log_file=log_file,
                )
            else:
                proc = launch.popen_launch_vllm_http_worker(
                    http_executable,
                    model_path,
                    http_url,
                    gpu_id=worker_gpus,
                    hash_seed=hash_seed,
                    extra_argv=[
                        *frontend_args,
                        "--enable-prompt-tokens-details",
                        *worker_args,
                    ],
                    extra_env=worker_env,
                    log_file=log_file,
                )
            cleanup.callback(launch.terminate_process, proc)
            deadline = time.monotonic() + worker_health_timeout
            worker_processes.append(
                (http_url, url, proc, log_path, is_decode, deadline)
            )
            if not pd:
                wait_worker_ready(http_url, proc, log_path, deadline)

        # Start both P/D engines before waiting; their health budgets begin at launch.
        for http_url, url, proc, log_path, is_decode, deadline in worker_processes:
            if pd:
                wait_worker_ready(http_url, proc, log_path, deadline)
            if is_decode:
                continue
            ready_log = log_path.read_text(encoding="utf-8", errors="replace")
            sizes = re.search(
                r"kv cache group sizes [\[(](\d+(?:,\s*\d+)*),?[)\]]", ready_log
            )
            assert sizes, f"{url}: missing KV cache group sizes in {log_path}"
            block_sizes = tuple(int(size) for size in sizes[1].split(","))
            assert len(block_sizes) == len(rank["groups"]), (
                f"{url}: cache groups {block_sizes} differ from {rank['groups']}"
            )
            worker_block_sizes.add(block_sizes)
            groups = [
                {**group, "block_size": size}
                for group, size in zip(rank["groups"], block_sizes, strict=True)
            ]
            worker_layouts[url] = {"ranks": [{**rank, "groups": groups}]}

        assert len(worker_block_sizes) == 1, (
            f"Workers must use the same cache group block sizes: {worker_layouts}"
        )
        (block_sizes,) = worker_block_sizes
        # Probe from shared block boundaries. Scratch state does not cache prefixes.
        cache_block_size = lcm(
            rank["hash_block_size"],
            *(
                size
                for group, size in zip(rank["groups"], block_sizes, strict=True)
                if group["kind"] != "scratch"
            ),
        )
        layout_path.write_text(json.dumps(worker_layouts), encoding="utf-8")
        router_args.extend([layout_flag, str(layout_path)])
        log_file = cleanup.enter_context(router_log.open("w", encoding="utf-8"))
        router = launch.popen_launch_router_only(
            router_url,
            policy="kv_aware",
            timeout=180.0,
            router_temperature=0.0,
            worker_urls=None if pd else list(worker_http_urls),
            router_model=model_path if protocol == "grpc" else None,
            extra_argv=router_args,
            log_level="debug",
            log_file=log_file,
        )
        cleanup.callback(launch.terminate_process, router)
        yield SimpleNamespace(
            protocol=protocol,
            deployment=deployment,
            model=model,
            url=router_url,
            log_path=router_log,
            worker_http_urls=worker_http_urls,
            cache_block_size=cache_block_size,
            max_model_len=engine_params["max_model_len"],
            speculative_tokens=rank["speculative_tokens"],
            prefill_backoff=prefill_backoff,
        )
