"""Run one router_policy perf cell (4 co-located gRPC workers + router + replay).

Flow:
  1. Start vLLM workers (engines).
  2. Wait until every worker passes HTTP /health.
  3. Start vllm-router (worker URLs + routing policy for this cell).
  4. Wait until router passes HTTP /health.
  5. Run agentinfer replay bench against the router.

Workload is fixed ahead of time: AgentX session trace JSONL (``AGENTX_TRACE``), generated
once via ``make_trace.py``. Replay executes that plan and writes ``result/requests.jsonl``
(per-request TTFT/latency — analysis output, not the bench input).

Configuration: ``AGENTX_CASE_FILE`` points to one self-contained case JSON
(``cases/<policy>/c<conc>.json`` with ``runtime``, ``workers``, ``router_params``).
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

SCENARIO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCENARIO_DIR))
from helpers.case_config import cell_output_dir, load_case, require_runtime  # noqa: E402

CASE: dict[str, Any] = {}
ROOT: Path
PY: str
RS: str
MODEL: str
ROUTER: str
REPLAY_CONFIG: Path
AGENT_INFER_SOURCE: Path | None
OPENSSL_LIB: str
ROUTER_POLICY: str
TRACE: Path
TASK_NUM: int
CONCURRENCY: int
WORKERS: int
GAP_SCALE: float
MAX_INFLIGHT: int
STALL_SECS: int
IDLE_SECS: int
OUT: Path
SERVED_MODEL_NAME: str
WORKER_CFG: dict[str, Any]
ROUTER_CFG: dict[str, Any]
env: dict[str, str]
GPUS: list[str]
clients: httpx.Client

METRIC = re.compile(r"^([a-zA-Z0-9_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$")


def bootstrap(case_path: Path) -> None:
    global \
        CASE, \
        ROOT, \
        PY, \
        RS, \
        MODEL, \
        ROUTER, \
        REPLAY_CONFIG, \
        AGENT_INFER_SOURCE, \
        OPENSSL_LIB
    global ROUTER_POLICY, TRACE, TASK_NUM, CONCURRENCY, WORKERS, GAP_SCALE, MAX_INFLIGHT
    global \
        STALL_SECS, \
        IDLE_SECS, \
        OUT, \
        SERVED_MODEL_NAME, \
        WORKER_CFG, \
        ROUTER_CFG, \
        env, \
        GPUS, \
        clients

    CASE = load_case(case_path, scenario_dir=SCENARIO_DIR)
    runtime = require_runtime(CASE)
    bench = CASE.get("benchmark_params") or {}
    workers = CASE.get("workers") or {}
    ROUTER_CFG = CASE.get("router_params") or {}

    ROOT = Path(runtime["perf_work_root"])
    PY = runtime["python"]
    RS = runtime["vllm_rs"]
    MODEL = runtime["model"]
    ROUTER = runtime["router_binary"]
    REPLAY_CONFIG = Path(runtime["replay_config_path"])
    raw_source = str(runtime.get("agent_infer", "") or "").strip()
    AGENT_INFER_SOURCE = Path(raw_source) if raw_source else None
    OPENSSL_LIB = runtime.get("openssl_lib_dir") or ""
    TRACE = Path(runtime["trace_path"])
    SERVED_MODEL_NAME = runtime["served_model_name"]

    ROUTER_POLICY = str(
        ROUTER_CFG.get("policy")
        or CASE.get("router_policy")
        or CASE["id"].split("-", 1)[1]
    )
    TASK_NUM = int(bench.get("task-num", 128))
    CONCURRENCY = int(bench.get("max-concurrency", 8))
    WORKERS = int(workers.get("count", 4))
    GAP_SCALE = float(bench.get("gap_scale", 0.0))
    MAX_INFLIGHT = int(bench.get("max_inflight", 1))
    STALL_SECS = int(bench.get("stall_secs", 900))
    IDLE_SECS = int(bench.get("idle_secs", 120))
    WORKER_CFG = dict(workers.get("per_worker") or {})

    OUT = Path(os.environ.get("AGENTX_OUT") or cell_output_dir(CASE))
    OUT.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    py_prefix = str(Path(PY).parent) if ("/" in PY or "\\" in PY) else ""
    path_parts = [p for p in (py_prefix, env.get("PATH", "")) if p]
    env_updates: dict[str, str] = {
        "PATH": os.pathsep.join(path_parts),
        "XDG_CACHE_HOME": str(ROOT / "cache"),
        "HF_HOME": str(ROOT / "cache/huggingface"),
        "VLLM_CACHE_ROOT": str(ROOT / "cache/vllm"),
        "TRITON_CACHE_DIR": str(ROOT / "cache/triton"),
        "VLLM_ROUTER_MODEL": MODEL,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8"),
        "TOKENIZERS_PARALLELISM": "false",
        "RUST_LOG": os.environ.get("RUST_LOG", "info"),
        "AGENTX_TASK_STAGGER": str(bench.get("task_stagger", 0)),
    }
    if AGENT_INFER_SOURCE is not None:
        existing = env.get("PYTHONPATH", "").strip()
        env_updates["PYTHONPATH"] = (
            str(AGENT_INFER_SOURCE)
            if not existing
            else f"{AGENT_INFER_SOURCE}{os.pathsep}{existing}"
        )
    env.update(env_updates)
    if OPENSSL_LIB:
        env["LD_LIBRARY_PATH"] = OPENSSL_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    GPUS = env.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if GPUS == [""]:
        GPUS = [str(i) for i in range(WORKERS)]
    assert len(GPUS) >= WORKERS, f"need {WORKERS} GPUs, got {GPUS!r}"
    clients = httpx.Client(timeout=15)


active = []
handles = []
ports = None


def start(command, log, extra=None):
    handle = open(log, "w")
    handles.append(handle)
    proc = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=env | (extra or {}),
        start_new_session=True,
    )
    active.append(proc)
    Path(str(log) + ".command.json").write_text(json.dumps(command, indent=2))
    return proc


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=15)


def cleanup(*_):
    for proc in reversed(active):
        stop(proc)
    active.clear()
    for handle in handles:
        handle.close()
    handles.clear()


def interrupted(*_):
    cleanup()
    raise SystemExit(130)


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)


def health(url, proc, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"process {proc.pid} exited {proc.returncode}: {url}")
        try:
            if clients.get(url).is_success:
                return
        except httpx.HTTPError:
            pass
        time.sleep(3)
    raise TimeoutError(url)


def allocate_ports(n):
    socks = []

    def free():
        s = socket.socket()
        s.bind(("0.0.0.0", 0))
        socks.append(s)
        return s.getsockname()[1]

    alloc = {
        name: [free() for _ in range(n)]
        for name in ["http", "grpc", "kv_pub", "kv_replay"]
    }
    alloc.update(router=free(), metrics=free())
    for s in socks:
        s.close()
    return alloc


def engine_command(i):
    pw = WORKER_CFG
    events = {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "endpoint": f"tcp://*:{ports['kv_pub'][i]}",
        "replay_endpoint": f"tcp://127.0.0.1:{ports['kv_replay'][i]}",
        "buffer_steps": 10000,
    }
    transfer = {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "cpu_bytes_to_use": int(pw.get("cpu_kv_bytes", 17179869184)),
            "self_describing_kv_events": True,
        },
    }
    return [
        RS,
        "serve",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(ports["http"][i]),
        "--grpc-port",
        str(ports["grpc"][i]),
        "--language-model-only",
        "--enable-prompt-tokens-details",
        "--reasoning-parser",
        "none",
        "--tool-call-parser",
        "none",
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--",
        "--tensor-parallel-size",
        str(pw.get("tensor_parallel_size", 1)),
        "--max-model-len",
        str(pw.get("max_model_len", 262144)),
        "--max-num-seqs",
        str(pw.get("max_num_seqs", 16)),
        "--max-num-batched-tokens",
        str(pw.get("max_num_batched_tokens", 32768)),
        "--enable-prefix-caching",
        "--dtype",
        str(pw.get("dtype", "bfloat16")),
        "--kv-cache-memory-bytes",
        str(pw.get("kv_cache_memory_bytes", 107374182400)),
        "--safetensors-load-strategy",
        "prefetch",
        "--kv-events-config",
        json.dumps(events),
        "--kv-transfer-config",
        json.dumps(transfer),
    ]


def router_command():
    rp = ROUTER_CFG
    policy = str(rp.get("policy", ROUTER_POLICY))
    router_policy = "kv_aware" if policy.startswith("kv_aware") else "cache_aware"
    scheme = str(rp.get("protocol", "grpc"))
    url_key = "grpc" if scheme == "grpc" else "http"
    cmd = [
        ROUTER,
        "--host",
        "127.0.0.1",
        "--port",
        str(ports["router"]),
        "--worker-urls",
        *[f"{scheme}://127.0.0.1:{ports[url_key][i]}" for i in range(WORKERS)],
        "--policy",
        router_policy,
        "--worker-startup-timeout-secs",
        str(rp.get("worker_startup_timeout_secs", 60)),
        "--worker-startup-check-interval",
        "1",
        "--prometheus-host",
        "127.0.0.1",
        "--prometheus-port",
        str(ports["metrics"]),
        "--router-temperature",
        str(rp.get("router_temperature", 0)),
        "--kv-penalty-host",
        str(rp.get("kv_penalty_host", 0.8)),
        "--kv-penalty-external",
        str(rp.get("kv_penalty_external", 0.6)),
        "--kv-penalty-disk",
        str(rp.get("kv_penalty_disk", 0.3)),
        "--cache-threshold",
        str(rp.get("cache_threshold", 0.3)),
        "--balance-abs-threshold",
        str(rp.get("balance_abs_threshold", 64)),
        "--balance-rel-threshold",
        str(rp.get("balance_rel_threshold", 1.5)),
    ]
    if rp.get("max_tree_size"):
        cmd += ["--max-tree-size", str(rp["max_tree_size"])]
    ttl = rp.get("kv_aware_session_affinity_ttl_secs")
    if ttl and policy == "kv_aware_aff":
        cmd += ["--kv-aware-session-affinity-ttl-secs", str(ttl)]
    return cmd


def runtime_config():
    values = yaml.safe_load(REPLAY_CONFIG.read_text(encoding="utf-8"))
    served = SERVED_MODEL_NAME
    values["experiment"].update(
        task_num=TASK_NUM,
        max_concurrency=CONCURRENCY,
        task_timeout_seconds=7200,
        run_timeout_seconds=86400,
    )
    values["backend"].update(
        base_url=f"http://127.0.0.1:{ports['router']}",
        metrics_url=f"http://127.0.0.1:{ports['metrics']}/metrics",
        tokenizer_base_url=f"http://127.0.0.1:{ports['http'][0]}",
        model=served,
        endpoint="/v1/completions",
    )
    values["replay"].update(
        trace_type="agentX",
        trace_path=str(TRACE),
        trace_same_agent_gap_scale=GAP_SCALE,
        trace_same_agent_gap_offset_seconds=0.0,
        interval_mode="trace",
        max_inflight_requests=MAX_INFLIGHT,
        request_timeout_seconds=3600,
    )
    path = OUT / "runtime-replay.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False))
    return path


def snapshot(dest):
    dest.mkdir(parents=True, exist_ok=True)
    for i in range(WORKERS):
        (dest / f"engine{i}.prom").write_text(
            clients.get(f"http://127.0.0.1:{ports['http'][i]}/metrics").text
        )
    (dest / "router.prom").write_text(
        clients.get(f"http://127.0.0.1:{ports['metrics']}/metrics").text
    )


def sum_metric(path, name):
    total = 0.0
    for line in Path(path).read_text().splitlines():
        match = METRIC.match(line)
        if match and match.group(1) == name:
            total += float(match.group(3))
    return total


def pct(values, p):
    ordered = sorted(values)
    if not ordered:
        return None
    rank = (len(ordered) - 1) * p / 100
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def main() -> None:
    case_file = Path(os.environ["AGENTX_CASE_FILE"])
    bootstrap(case_path=case_file)
    _run_cell()


def _run_cell() -> None:
    global ports
    ports = allocate_ports(WORKERS)
    env["HHC_KV_PUB_PORTS"] = json.dumps(ports["kv_pub"])
    entry = OUT / "agentx_entry.py"
    entry.write_text(
        """import asyncio
import os
import sys

from agentinfer.agentbench.replay.executor import ReplayExecutor
from agentinfer.agentcache.entrypoints.cli.main import main

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
    config = runtime_config()
    (OUT / "ports.json").write_text(json.dumps(ports, indent=2))

    started = time.monotonic()
    try:
        engines = [
            start(
                engine_command(i),
                OUT / f"engine{i}.log",
                {"CUDA_VISIBLE_DEVICES": GPUS[i]},
            )
            for i in range(WORKERS)
        ]
        for i, proc in enumerate(engines):
            health(f"http://127.0.0.1:{ports['http'][i]}/health", proc, timeout=1800)
        print(f"ENGINES_READY {time.monotonic() - started:.0f}s", flush=True)
        router = start(router_command(), OUT / "router.log")
        health(f"http://127.0.0.1:{ports['router']}/health", router, timeout=180)
        print(f"ROUTER_READY {time.monotonic() - started:.0f}s", flush=True)
        snapshot(OUT / "before")
        result = OUT / "result"
        replay = start(
            [
                PY,
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
            OUT / "replay.log",
        )

        def engines_busy():
            total = 0
            for i in range(WORKERS):
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
                if STALL_SECS > 0 and idle_for > STALL_SECS:
                    stalled = True
                    print(
                        f"STALL no progress for {idle_for:.0f}s, terminating replay",
                        flush=True,
                    )
                elif idle_for > IDLE_SECS and engines_busy() == 0:
                    stalled = True
                    print(
                        f"STALL no progress for {idle_for:.0f}s, all engines idle, terminating replay",
                        flush=True,
                    )
                if stalled:
                    replay.terminate()
                    try:
                        replay.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        replay.kill()
                    break
            if tick % 6 == 0:
                try:
                    snapshot(OUT / "metrics" / f"{tick * 10:05}")
                except Exception as exc:
                    print(f"SNAPSHOT_FAIL {exc}", flush=True)
                print(f"TICK {tick * 10}s", flush=True)
        snapshot(OUT / "after")
        wall = time.monotonic() - started
        rows = [
            json.loads(line)
            for line in (result / "requests.jsonl").read_text().splitlines()
            if line.strip()
        ]
        ttft = [r["ttft_seconds"] for r in rows if r.get("ttft_seconds") is not None]
        lat = [
            r["latency_seconds"] for r in rows if r.get("latency_seconds") is not None
        ]
        stats = {
            "router_policy": ROUTER_POLICY,
            "arm": ROUTER_POLICY,
            "returncode": replay.returncode,
            "wall_seconds": wall,
            "stalled": stalled,
            "requests": len(rows),
            "success": sum(1 for r in rows if r.get("status") == "success"),
            "input_tokens": sum(r.get("input_tokens") or 0 for r in rows),
            "output_tokens": sum(r.get("output_tokens") or 0 for r in rows),
            "ttft_s": {
                "mean": statistics.mean(ttft) if ttft else None,
                "p50": pct(ttft, 50),
                "p90": pct(ttft, 90),
                "max": max(ttft) if ttft else None,
            },
            "latency_s": {
                "mean": statistics.mean(lat) if lat else None,
                "p50": pct(lat, 50),
                "p90": pct(lat, 90),
                "max": max(lat) if lat else None,
            },
            "worker_requests": [
                sum_metric(
                    OUT / "after" / f"engine{i}.prom", "vllm:request_success_total"
                )
                - sum_metric(
                    OUT / "before" / f"engine{i}.prom", "vllm:request_success_total"
                )
                for i in range(WORKERS)
            ],
            "cache_hit_rate": (lambda h, q: h / q if q else None)(
                sum(
                    sum_metric(
                        OUT / p / f"engine{i}.prom", "vllm:prefix_cache_hits_total"
                    )
                    for i in range(WORKERS)
                    for p in ["after"]
                ),
                sum(
                    sum_metric(
                        OUT / "after" / f"engine{i}.prom",
                        "vllm:prefix_cache_queries_total",
                    )
                    for i in range(WORKERS)
                ),
            ),
        }
        (OUT / "calibration.json").write_text(json.dumps(stats, indent=2))
        print("STATS " + json.dumps(stats), flush=True)
        if stalled or stats["success"] != stats["requests"]:
            raise SystemExit(1)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
