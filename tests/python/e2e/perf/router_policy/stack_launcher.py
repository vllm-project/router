"""Launch workers, the router, and replay for one router_policy cell.

Flow:
  1. Start vLLM workers (engines).
  2. Wait until every worker passes HTTP /health.
  3. Start vllm-router (worker URLs + routing policy for this cell).
  4. Wait until router passes HTTP /health.
  5. Run agentinfer replay bench against the router.

Workload is the TraceLab 8-session / 28-request subset
(``data/tracelab-ci-uniform-28.jsonl``). Replay executes that plan and writes
``result/requests.jsonl`` (per-request TTFT/latency — analysis output, not the bench input).

Configuration: ``AGENTX_CASE_FILE`` points to one self-contained case JSON
(``cases/<policy>.json`` with ``runtime``, ``workers``, ``router_params``,
and ``benchmark_params.max-concurrency``).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

SCENARIO_DIR = Path(__file__).resolve().parent
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))

from case_loader import (  # noqa: E402
    RouterPolicyCase,
    cell_run_suffix,
    load_router_policy_case,
)
from tests.python.e2e.perf.binaries import preflight_router_bin  # noqa: E402
from log_summary import (  # noqa: E402
    RESULTS_SUMMARY_ENV,
    print_failure_diagnostics,
    write_calibration,
)
from tests.python.e2e.perf.process import (  # noqa: E402
    RunningProc,
    launch_router,
    launch_worker,
    graceful_stop,
    release_all_port_holders,
    release_router_port_holders,
    release_worker_port_holders,
    reserve_port,
    router_cmd,
    vllm_rs_serve_cmd,
)


@dataclass
class RouterPolicyStack:
    """One in-flight cell. Case config stays on ``case``; processes stay here."""

    case: RouterPolicyCase
    out: Path
    env: dict[str, str]
    gpus: list[str]
    clients: httpx.Client
    ports: dict[str, Any] = field(default_factory=dict)
    stack: list[RunningProc] = field(default_factory=list)
    replay_proc: subprocess.Popen[Any] | None = None
    replay_log_fp: Any = None
    port_holders: list[socket.socket | None] = field(default_factory=list)
    kv_port_holders: list[socket.socket | None] = field(default_factory=list)
    did_cleanup: bool = False


# Set only so SIGINT/SIGTERM can stop the cell that is running.
_active: RouterPolicyStack | None = None


def open_cell(case: RouterPolicyCase, *, out_dir: Path | None) -> RouterPolicyStack:
    """Build the run directory and process env for one case."""

    if out_dir is not None:
        out = out_dir
    elif raw_out := os.environ.get("AGENTX_OUT", "").strip():
        out = Path(raw_out)
    else:
        suffix = cell_run_suffix()
        out = case.perf_work_root / "runs" / f"{case.id}-{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    print(
        "STREAM_LOGS worker+router -> stderr (pytest -sv); replay stays in replay.log",
        flush=True,
    )

    env = os.environ.copy()
    py = case.python
    py_prefix = str(Path(py).parent) if ("/" in py or "\\" in py) else ""
    path_parts = [p for p in (py_prefix, env.get("PATH", "")) if p]
    env_updates: dict[str, str] = {
        "PATH": os.pathsep.join(path_parts),
        "XDG_CACHE_HOME": str(case.perf_work_root / "cache"),
        "HF_HOME": str(case.perf_work_root / "cache/huggingface"),
        "VLLM_CACHE_ROOT": str(case.perf_work_root / "cache/vllm"),
        "TRITON_CACHE_DIR": str(case.perf_work_root / "cache/triton"),
        "VLLM_ROUTER_MODEL": case.model,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8"),
        "TOKENIZERS_PARALLELISM": "false",
        "RUST_LOG": os.environ.get("RUST_LOG", "info"),
        "AGENTX_TASK_STAGGER": str(case.task_stagger),
    }
    if case.agent_infer is not None:
        existing = env.get("PYTHONPATH", "").strip()
        env_updates["PYTHONPATH"] = (
            str(case.agent_infer)
            if not existing
            else f"{case.agent_infer}{os.pathsep}{existing}"
        )
    env.update(env_updates)
    if case.openssl_lib_dir:
        env["LD_LIBRARY_PATH"] = (
            case.openssl_lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")
        )
    gpus = env.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if gpus == [""]:
        gpus = [str(i) for i in range(case.worker_count)]
    if len(gpus) < case.worker_count:
        raise RuntimeError(f"need {case.worker_count} GPUs, got {gpus!r}")
    return RouterPolicyStack(
        case=case,
        out=out,
        env=env,
        gpus=gpus,
        clients=httpx.Client(timeout=15),
    )


def _start_replay(
    run: RouterPolicyStack, command: list[str], log: Path
) -> subprocess.Popen[Any]:
    """Agentinfer replay. Its own session so a stall can stop the whole group."""

    Path(str(log) + ".command.json").write_text(json.dumps(command, indent=2))
    log.parent.mkdir(parents=True, exist_ok=True)
    run.replay_log_fp = open(log, "w", encoding="utf-8")
    return subprocess.Popen(
        command,
        stdout=run.replay_log_fp,
        stderr=subprocess.STDOUT,
        env=run.env,
        start_new_session=True,
    )


def cleanup(run: RouterPolicyStack) -> None:
    if run.did_cleanup:
        return
    run.did_cleanup = True
    if run.replay_proc is not None:
        graceful_stop(run.replay_proc, 45.0, process_group=True, kill_grace=15)
        run.replay_proc = None
    for proc in reversed(run.stack):
        proc.stop()
    run.stack.clear()
    if run.replay_log_fp is not None:
        run.replay_log_fp.close()
        run.replay_log_fp = None
    release_all_port_holders(run.port_holders)
    release_all_port_holders(run.kv_port_holders)


def interrupted(*_):
    if _active is not None:
        cleanup(_active)
    raise SystemExit(130)


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)


# Outside the usual ephemeral range, so vLLM's own bind(0) calls during
# model load do not take the KV event ports after we close the holders.
_KV_PORT_FIRST = 25100
_KV_PORT_LAST = 25299


def _reserve_kv_port(holders: list[socket.socket | None]) -> int:
    """Bind one free port in ``25100-25299`` and keep the socket in ``holders``."""

    for port in range(_KV_PORT_FIRST, _KV_PORT_LAST + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            sock.close()
            continue
        holders.append(sock)
        return port
    raise RuntimeError(f"no free KV event port in {_KV_PORT_FIRST}-{_KV_PORT_LAST}")


def _release_kv_port_holders(run: RouterPolicyStack, worker: int) -> None:
    """Close this worker's PUB and replay holders so the engine can bind them."""

    holders = run.kv_port_holders
    for index in (worker * 2, worker * 2 + 1):
        if index >= len(holders) or holders[index] is None:
            continue
        try:
            holders[index].close()
        except OSError:
            pass
        holders[index] = None


def allocate_ports(run: RouterPolicyStack) -> None:
    """Reserve HTTP, gRPC, router, and KV event ports.

    This vLLM build echoes ``kv_events_config`` unchanged. ``tcp://*:0``
    therefore comes back from ``GetKvEventSources`` as port 0, which the
    router rejects. PUB and replay use a free port in 25100-25299 instead.
    Those sockets stay held until the matching worker process is started.
    """

    release_all_port_holders(run.port_holders)
    release_all_port_holders(run.kv_port_holders)
    holders: list[socket.socket | None] = []
    kv_holders: list[socket.socket | None] = []
    n = run.case.worker_count

    def reserve() -> int:
        return reserve_port(holders)

    alloc: dict[str, Any] = {
        name: [reserve() for _ in range(n)] for name in ["http", "grpc"]
    }
    alloc["router"] = reserve()
    alloc["metrics"] = reserve()
    kv_pub: list[int] = []
    kv_replay: list[int] = []
    for _ in range(n):
        kv_pub.append(_reserve_kv_port(kv_holders))
        kv_replay.append(_reserve_kv_port(kv_holders))
    alloc["kv_pub"] = kv_pub
    alloc["kv_replay"] = kv_replay
    run.port_holders = holders
    run.kv_port_holders = kv_holders
    run.ports = alloc


def engine_command(run: RouterPolicyStack, i: int):
    pw = run.case.per_worker
    ports = run.ports
    cpu_bytes = int(pw.get("cpu_kv_bytes", 0))
    # GiB for --kv-offloading-size. cpu_kv_bytes == 0 leaves host offload off.
    engine: dict[str, Any] = {
        "tensor_parallel_size": pw.get("tensor_parallel_size", 1),
        "block_size": pw.get("block_size"),
        "max_model_len": pw.get("max_model_len", 262144),
        "max_num_seqs": pw.get("max_num_seqs", 16),
        "max_num_batched_tokens": pw.get("max_num_batched_tokens", 32768),
        "enable_prefix_caching": True,
        "dtype": pw.get("dtype", "bfloat16"),
        "gpu_memory_utilization": pw.get("gpu_memory_utilization"),
        "kv_cache_metrics": bool(pw.get("kv_cache_metrics")),
        "kv_cache_metrics_sample": pw.get("kv_cache_metrics_sample"),
        "kv_cache_memory_bytes": pw.get("kv_cache_memory_bytes", 107374182400),
    }
    if cpu_bytes > 0:
        engine["kv_offloading_backend"] = "native"
        engine["kv_offloading_size"] = f"{cpu_bytes / (1024**3):g}"
        engine["kv_transfer_config"] = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "CPUOffloadingSpec",
                "cpu_bytes_to_use": cpu_bytes,
                "blocks_per_chunk": 16,
                "self_describing_kv_events": True,
            },
        }
    engine["safetensors_load_strategy"] = "prefetch"
    engine["kv_events_config"] = {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "endpoint": f"tcp://127.0.0.1:{ports['kv_pub'][i]}",
        "replay_endpoint": f"tcp://127.0.0.1:{ports['kv_replay'][i]}",
        "buffer_steps": 10000,
    }
    return vllm_rs_serve_cmd(
        run.case.vllm_rs,
        run.case.model,
        host="127.0.0.1",
        port=ports["http"][i],
        grpc_port=ports["grpc"][i],
        language_model_only=True,
        enable_prompt_tokens_details=True,
        reasoning_parser="none",
        tool_call_parser="none",
        served_model_name=run.case.served_model_name,
        engine_flags=engine,
    )


def router_command(run: RouterPolicyStack):
    rp = run.case.router
    ports = run.ports
    policy = str(rp.get("policy", run.case.policy))
    # kv_aware_aff is this suite's label for kv_aware plus the affinity TTL flag.
    # Every other JSON policy is passed through unchanged.
    router_policy = "kv_aware" if policy == "kv_aware_aff" else policy
    scheme = str(rp.get("protocol", "grpc"))
    url_key = "grpc" if scheme == "grpc" else "http"
    ttl = rp.get("kv_aware_session_affinity_ttl_secs")
    return router_cmd(
        run.case.router_bin,
        host="127.0.0.1",
        port=ports["router"],
        worker_urls=[
            f"{scheme}://127.0.0.1:{ports[url_key][i]}"
            for i in range(run.case.worker_count)
        ],
        worker_startup_timeout_secs=int(rp.get("worker_startup_timeout_secs", 60)),
        prometheus_host="127.0.0.1",
        prometheus_port=ports["metrics"],
        policy=router_policy,
        worker_startup_check_interval=1,
        router_temperature=rp.get("router_temperature", 0),
        kv_penalty_host=rp.get("kv_penalty_host", 0.8),
        kv_penalty_external=rp.get("kv_penalty_external", 0.6),
        kv_penalty_disk=rp.get("kv_penalty_disk", 0.3),
        cache_threshold=rp.get("cache_threshold", 0.3),
        balance_abs_threshold=rp.get("balance_abs_threshold", 64),
        balance_rel_threshold=rp.get("balance_rel_threshold", 1.5),
        max_tree_size=rp.get("max_tree_size") or None,
        kv_aware_session_affinity_ttl_secs=(
            ttl if ttl and policy == "kv_aware_aff" else None
        ),
    )


def runtime_config(run: RouterPolicyStack):
    case = run.case
    ports = run.ports
    values = yaml.safe_load(case.replay_config.read_text(encoding="utf-8"))
    served = case.served_model_name
    values["experiment"].update(
        task_num=case.task_num,
        max_concurrency=case.concurrency,
        task_timeout_seconds=7200,
        run_timeout_seconds=86400,
    )
    values["backend"].update(
        base_url=f"http://127.0.0.1:{ports['router']}",
        metrics_url=f"http://127.0.0.1:{ports['metrics']}/metrics",
        tokenizer_base_url=f"http://127.0.0.1:{ports['http'][0]}",
        model=served,
        endpoint=case.endpoint,
    )
    replay_fields = {
        "trace_type": case.trace_type,
        "trace_path": str(case.trace_path),
        "trace_same_agent_gap_scale": case.gap_scale,
        "trace_same_agent_gap_offset_seconds": 0.0,
        "interval_mode": "trace",
        "max_inflight_requests": case.max_inflight,
        "request_timeout_seconds": 3600,
    }
    if case.trace_type == "tracelab":
        replay_fields.update(
            sample_seed=0,
            context_adjustment_mode="strict",
            context_micro_trim_max_tokens=64,
            context_micro_trim_max_ratio=0.005,
            prompt_calibration_tolerance_tokens=0,
        )
    values["replay"].update(replay_fields)
    path = run.out / "runtime-replay.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False))
    return path


def _endpoint_uses_port_zero(value: str) -> bool:
    text = value.strip()
    return text.endswith(":0") or "*:0" in text


def _proto_fields(payload: bytes):
    """Yield ``(field_number, value)`` for a protobuf message.

    Varints stay ints. Length-delimited fields stay ``bytes``. Fixed32/64 are
    skipped as opaque slices. This only needs ``GetKvEventSourcesResponse``.
    """

    index = 0
    end = len(payload)
    while index < end:
        key, index = _read_varint(payload, index)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, index = _read_varint(payload, index)
            yield field, value
        elif wire == 2:
            length, index = _read_varint(payload, index)
            value = payload[index : index + length]
            index += length
            yield field, value
        elif wire == 5:
            index += 4
        elif wire == 1:
            index += 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")


def _read_varint(payload: bytes, index: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if index >= len(payload):
            raise ValueError("truncated protobuf varint")
        byte = payload[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, index
        shift += 7
        if shift > 63:
            raise ValueError("protobuf varint is too long")


def _kv_event_endpoints_from_proto(payload: bytes) -> list[str]:
    """Pull ``endpoint`` and ``replay_endpoint`` out of ``GetKvEventSourcesResponse``."""

    found: list[str] = []
    for number, value in _proto_fields(payload):
        if number != 1 or not isinstance(value, (bytes, bytearray)):
            continue
        for inner, inner_val in _proto_fields(value):
            if inner not in (2, 4) or not isinstance(inner_val, (bytes, bytearray)):
                continue
            text = bytes(inner_val).decode()
            if text:
                found.append(text)
    return found


def _grpc_kv_endpoints(port: int) -> list[str]:
    """Read bound KV event addresses from vllm-rs ``Control.GetKvEventSources``."""

    try:
        import grpc
    except ImportError as exc:
        raise RuntimeError("grpc is required to read vllm-rs KV event sources") from exc
    target = f"127.0.0.1:{port}"
    channel = grpc.insecure_channel(target)
    try:
        call = channel.unary_unary(
            "/vllm.Control/GetKvEventSources",
            request_serializer=lambda _request: b"",
            response_deserializer=_kv_event_endpoints_from_proto,
        )
        return list(call(None, timeout=5))
    except grpc.RpcError as exc:
        raise RuntimeError(
            f"GetKvEventSources grpc://{target} failed: {exc.code().name} {exc.details()}"
        ) from exc
    finally:
        channel.close()


def _http_kv_endpoints(port: int) -> list[str]:
    url = f"http://127.0.0.1:{port}/kv_event_sources"
    try:
        response = httpx.get(url, timeout=5.0)
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{url} unavailable: {exc}") from exc
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if (
                    key in {"endpoint", "replay_endpoint"}
                    and isinstance(child, str)
                    and child
                ):
                    found.append(child)
                else:
                    walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(body)
    return found


def _require_resolved_kv_events(run: RouterPolicyStack) -> None:
    """Fail if a worker still advertises a KV event port of 0.

    gRPC workers (this suite) publish the bound address on
    ``Control.GetKvEventSources``. Their OpenAI HTTP port answers 404 for
    ``/kv_event_sources``.
    """

    ports = run.ports
    scheme = str(run.case.router.get("protocol", "grpc"))
    for i in range(run.case.worker_count):
        if scheme == "grpc":
            label = f"engine{i} grpc://127.0.0.1:{ports['grpc'][i]} GetKvEventSources"
            endpoints = _grpc_kv_endpoints(ports["grpc"][i])
        else:
            label = f"engine{i} http://127.0.0.1:{ports['http'][i]}/kv_event_sources"
            endpoints = _http_kv_endpoints(ports["http"][i])
        if not endpoints:
            raise RuntimeError(f"{label} returned no KV event endpoints")
        unresolved = [item for item in endpoints if _endpoint_uses_port_zero(item)]
        if unresolved:
            raise RuntimeError(f"{label} still advertises port 0: {unresolved}")


def snapshot(run: RouterPolicyStack, dest):
    dest.mkdir(parents=True, exist_ok=True)
    ports = run.ports
    for i in range(run.case.worker_count):
        (dest / f"engine{i}.prom").write_text(
            run.clients.get(f"http://127.0.0.1:{ports['http'][i]}/metrics").text
        )
    (dest / "router.prom").write_text(
        run.clients.get(f"http://127.0.0.1:{ports['metrics']}/metrics").text
    )


def run_router_policy_cell(
    case_file: Path,
    *,
    concurrency: int | None = None,
    out_dir: Path | None = None,
    run_suffix: str | None = None,
    results_summary: Path | None = None,
) -> Path:
    """Start workers, router, and AgentX replay for one case JSON."""

    global _active
    case_path = case_file.resolve()
    os.environ["AGENTX_CASE_FILE"] = str(case_path)
    if run_suffix is not None:
        os.environ["AGENTX_RUN_SUFFIX"] = run_suffix
    if out_dir is not None:
        os.environ["AGENTX_OUT"] = str(out_dir)
    if results_summary is not None:
        os.environ[RESULTS_SUMMARY_ENV] = str(results_summary)
    case = load_router_policy_case(case_path, concurrency=concurrency)
    run = open_cell(case, out_dir=out_dir)
    _active = run
    try:
        _run_cell(run)
    finally:
        _active = None
    return run.out


def _run_cell(run: RouterPolicyStack) -> None:
    case = run.case
    settle = float(os.environ.get("AGENTX_PORT_SETTLE_SEC", "2"))
    if settle > 0:
        time.sleep(settle)
    allocate_ports(run)
    ports = run.ports
    entry = run.out / "agentx_entry.py"
    entry.write_text(
        """import asyncio
import os
import sys

from agentinfer.agentbench.replay.executor import ReplayExecutor
from agentinfer.agentbench.replay.transport import ReplayTransport
from agentinfer.agentcache.entrypoints.cli.main import main

_validate_tracelab_context = ReplayTransport.validate_tracelab_context

async def _validate_against_worker_http(self, plan, warmup_tokens):
    # Router /v1/models for grpc workers omits max_model_len. The library
    # check reads base_url, so this one call uses tokenizer_base_url.
    saved = self.upstream
    self.upstream = self.config.backend.resolved_tokenizer_base_url.rstrip("/")
    try:
        return await _validate_tracelab_context(self, plan, warmup_tokens)
    finally:
        self.upstream = saved

ReplayTransport.validate_tracelab_context = _validate_against_worker_http

_delay = float(os.environ.get("AGENTX_TASK_STAGGER", "0") or 0)
if _delay > 0:
    async def execute(self):
        semaphore = asyncio.Semaphore(self.config.experiment.max_concurrency)

        async def bounded(index, task):
            await asyncio.sleep(index * _delay)
            async with semaphore:
                return await self._execute_task(task)

        return tuple(await asyncio.gather(*(bounded(i, t) for i, t in enumerate(self.plan.tasks))))
    ReplayExecutor.execute = execute
    print(f"STAGGER {_delay}s per session", flush=True)

raise SystemExit(main(["vllm", *sys.argv[1:]]))
"""
    )
    config = runtime_config(run)
    (run.out / "ports.json").write_text(json.dumps(ports, indent=2))

    started = time.monotonic()
    try:
        try:
            preflight_router_bin(case.router_bin)
            for i in range(case.worker_count):
                release_worker_port_holders(run.port_holders, i, case.worker_count)
                _release_kv_port_holders(run, i)
                worker_cmd = engine_command(run, i)
                worker_log = run.out / f"engine{i}.log"
                Path(str(worker_log) + ".command.json").write_text(
                    json.dumps(worker_cmd, indent=2)
                )
                run.stack.append(
                    launch_worker(
                        worker_cmd,
                        label=f"engine{i}",
                        env=run.env | {"CUDA_VISIBLE_DEVICES": run.gpus[i]},
                        ready_url=f"http://127.0.0.1:{ports['http'][i]}/health",
                        timeout=1800,
                        log_prefix="router_policy",
                        log_path=worker_log,
                    )
                )
            _require_resolved_kv_events(run)
            print(f"ENGINES_READY {time.monotonic() - started:.0f}s", flush=True)
            release_router_port_holders(run.port_holders, case.worker_count)
            router_argv = router_command(run)
            router_log = run.out / "router.log"
            Path(str(router_log) + ".command.json").write_text(
                json.dumps(router_argv, indent=2)
            )
            run.stack.append(
                launch_router(
                    router_argv,
                    router_url=f"http://127.0.0.1:{ports['router']}",
                    env=run.env,
                    timeout=180,
                    log_prefix="router_policy",
                    log_path=router_log,
                )
            )
            print(f"ROUTER_READY {time.monotonic() - started:.0f}s", flush=True)
        except (OSError, RuntimeError, TimeoutError) as exc:
            cleanup(run)
            print_failure_diagnostics(run, hint=str(exc))
            raise SystemExit(1) from exc
        snapshot(run, run.out / "before")
        result = run.out / "result"
        if result.exists():
            shutil.rmtree(result)
        run.replay_proc = _start_replay(
            run,
            [
                case.python,
                "-X",
                "faulthandler",
                "-u",
                str(entry),
                "bench",
                "serve",
                "--agentinfer",
                "replay",
                "--config",
                str(config),
                "--result-dir",
                str(result),
            ],
            run.out / "replay.log",
        )
        replay = run.replay_proc

        def engines_busy():
            total = 0
            for i in range(case.worker_count):
                try:
                    txt = httpx.get(
                        f"http://127.0.0.1:{ports['http'][i]}/metrics", timeout=2.0
                    ).text
                except Exception:
                    return None
                for line in txt.splitlines():
                    if line.split("{")[0].strip() == "vllm:num_requests_running":
                        try:
                            total += int(float(line.split()[-1]))
                        except (ValueError, IndexError):
                            pass
            return total

        tick = 0
        stalled = False
        progress_file = result / "requests.jsonl"
        last_size = -1
        last_progress = time.monotonic()
        while replay.poll() is None:
            time.sleep(10)
            tick += 1
            try:
                size = progress_file.stat().st_size
            except OSError:
                size = 0
            if size != last_size:
                last_size = size
                last_progress = time.monotonic()
            else:
                idle_for = time.monotonic() - last_progress
                if case.stall_secs > 0 and idle_for > case.stall_secs:
                    stalled = True
                    print(
                        f"STALL no progress for {idle_for:.0f}s, terminating replay",
                        flush=True,
                    )
                elif idle_for > case.idle_secs and engines_busy() == 0:
                    stalled = True
                    print(
                        f"STALL no progress for {idle_for:.0f}s, all engines idle, terminating replay",
                        flush=True,
                    )
                if stalled:
                    graceful_stop(replay, 60, process_group=True, kill_grace=15)
                    break
            if tick % 6 == 0:
                try:
                    snapshot(run, run.out / "metrics" / f"{tick * 10:05}")
                except Exception as exc:
                    print(f"SNAPSHOT_FAIL {exc}", flush=True)
                replay_lines = 0
                if progress_file.is_file():
                    replay_lines = sum(
                        1
                        for line in progress_file.read_text(
                            encoding="utf-8", errors="replace"
                        ).splitlines()
                        if line.strip()
                    )
                running = engines_busy()
                tick_extra = f" replay_ok={replay_lines}"
                if running is not None:
                    tick_extra += f" engines_running={running}"
                print(f"TICK {tick * 10}s{tick_extra}", flush=True)
        try:
            replay.wait(timeout=120)
        except subprocess.TimeoutExpired:
            graceful_stop(replay, 30, process_group=True, kill_grace=15)
        snapshot(run, run.out / "after")
        replay_rc = replay.returncode
        if replay_rc not in (0, None):
            print(f"REPLAY exit code {replay_rc}", flush=True)
        cleanup(run)
        write_calibration(
            run,
            result_dir=result,
            replay_rc=replay_rc,
            wall_seconds=time.monotonic() - started,
            stalled=stalled,
        )
    finally:
        cleanup(run)
