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
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

SCENARIO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCENARIO_DIR))
from helpers.case_config import (  # noqa: E402
    cell_output_dir,
    load_case,
    packaged_replay_agentx_yaml,
    require_runtime,
)

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
    replay_path = str(runtime.get("replay_config_path") or "").strip()
    REPLAY_CONFIG = Path(replay_path) if replay_path else packaged_replay_agentx_yaml()
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
    if _stream_logs_enabled():
        print(
            "STREAM_LOGS worker+router -> stderr (set AGENTX_STREAM_LOGS=0 to disable)",
            flush=True,
        )
    else:
        print(
            "LOGS worker/router/replay -> files under cell_dir only "
            "(export AGENTX_STREAM_LOGS=1 and use pytest -sv for live "
            "[engineN]/[router] on stderr)",
            flush=True,
        )

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
stream_pumps: list[tuple[subprocess.Popen[Any], threading.Thread]] = []
ports = None
_port_holders: list[socket.socket | None] = []
_did_cleanup = False
RESULTS_SUMMARY_ENV = "AGENTX_RESULTS_SUMMARY"


def _stream_logs_enabled() -> bool:
    return os.environ.get("AGENTX_STREAM_LOGS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _sanitize_stream_line(line: str) -> str:
    if "\r" not in line:
        return line
    return line.split("\r")[-1]


def _stream_pump(proc: subprocess.Popen[Any], log_fp, prefix: str) -> None:
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            log_fp.write(line)
            log_fp.flush()
            sys.stderr.write(f"{prefix}{_sanitize_stream_line(line)}")
            sys.stderr.flush()
    except ValueError:
        pass
    finally:
        log_fp.flush()


def start(command, log, extra=None, *, stream_prefix: str | None = None):
    extra_env = env | (extra or {})
    Path(str(log) + ".command.json").write_text(json.dumps(command, indent=2))
    component = (stream_prefix or "").strip() or Path(str(log)).stem
    if stream_prefix and _stream_logs_enabled():
        log_fp = open(log, "w")
        handles.append(log_fp)
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=extra_env,
            start_new_session=True,
            text=True,
            bufsize=1,
        )
        thread = threading.Thread(
            target=_stream_pump,
            args=(proc, log_fp, stream_prefix),
            name=f"stream-{component}",
            daemon=True,
        )
        thread.start()
        stream_pumps.append((proc, thread))
    else:
        handle = open(log, "w")
        handles.append(handle)
        proc = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=extra_env,
            start_new_session=True,
        )
    active.append(proc)
    return proc


def _join_stream_pump(proc: subprocess.Popen[Any], thread: threading.Thread) -> None:
    if proc.stdout is not None:
        proc.stdout.close()
    thread.join(timeout=15)


def stop(proc):
    stream_thread: threading.Thread | None = None
    for idx, (p, thread) in enumerate(stream_pumps):
        if p is proc:
            stream_thread = thread
            stream_pumps.pop(idx)
            break
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=15)
    if stream_thread is not None:
        _join_stream_pump(proc, stream_thread)


def _release_all_port_holders() -> None:
    global _port_holders
    for idx, sock in enumerate(_port_holders):
        if sock is None:
            continue
        try:
            sock.close()
        except OSError:
            pass
        _port_holders[idx] = None
    _port_holders.clear()


def _release_worker_port_holders(worker: int, workers: int) -> None:
    n = workers
    for idx in (worker, n + worker, 2 * n + worker, 3 * n + worker):
        if idx >= len(_port_holders) or _port_holders[idx] is None:
            continue
        try:
            _port_holders[idx].close()
        except OSError:
            pass
        _port_holders[idx] = None


def _release_router_port_holders(workers: int) -> None:
    n = workers
    for idx in (4 * n, 4 * n + 1):
        if idx >= len(_port_holders) or _port_holders[idx] is None:
            continue
        try:
            _port_holders[idx].close()
        except OSError:
            pass
        _port_holders[idx] = None


def cleanup(*_):
    global _did_cleanup
    if _did_cleanup:
        return
    _did_cleanup = True
    for proc in reversed(active):
        stop(proc)
    active.clear()
    stream_pumps.clear()
    for handle in handles:
        handle.close()
    handles.clear()
    _release_all_port_holders()


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


_LOG_ERROR_MARKERS = ("error", "exception", "traceback", "failed", "fatal", "cuda")


def _log_tail_lines() -> int:
    raw = os.environ.get("AGENTX_LOG_TAIL", "50").strip()
    try:
        return max(5, int(raw))
    except ValueError:
        return 50


def _tail_file(path: Path, *, max_lines: int = 40, label: str | None = None) -> None:
    tag = label or path.name
    if not path.is_file():
        print(f"LOG_TAIL {tag}: (missing {path})", flush=True)
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-max_lines:] if lines else []
    print(f"LOG_TAIL {tag} (last {len(tail)} of {len(lines)} lines):", flush=True)
    for line in tail:
        print(line, flush=True)


def _scan_log_errors(path: Path, label: str, *, max_hits: int = 8) -> None:
    if not path.is_file():
        return
    hits: list[tuple[int, str]] = []
    for idx, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        low = line.lower()
        if any(marker in low for marker in _LOG_ERROR_MARKERS):
            hits.append((idx + 1, line))
    if not hits:
        return
    print(
        f"LOG_ERRORS {label} ({len(hits)} hit(s), last {min(max_hits, len(hits))}):",
        flush=True,
    )
    for line_no, line in hits[-max_hits:]:
        snippet = line if len(line) <= 240 else line[:237] + "..."
        print(f"  L{line_no}: {snippet}", flush=True)


def _component_logs(out: Path, workers: int) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for i in range(workers):
        items.append((f"engine{i}.log", out / f"engine{i}.log"))
    items.extend(
        [
            ("router.log", out / "router.log"),
            ("replay.log", out / "replay.log"),
        ]
    )
    return items


def print_artifacts(
    out: Path,
    workers: int,
    *,
    extras: list[tuple[str, Path]] | None = None,
) -> None:
    entries: list[tuple[str, Path]] = [
        ("cell_dir", out),
        ("calibration.json", out / "calibration.json"),
        ("requests.jsonl", out / "result" / "requests.jsonl"),
        ("ports.json", out / "ports.json"),
    ]
    for label, log_path in _component_logs(out, workers):
        entries.append((label, log_path))
        entries.append((f"{label}.command.json", Path(str(log_path) + ".command.json")))
    if extras:
        entries.extend(extras)
    print("ARTIFACTS", flush=True)
    for name, path in entries:
        print(f"  {name}: {path.resolve()}", flush=True)
    ports = out / "ports.json"
    if ports.is_file():
        print(f"PORTS {ports.read_text(encoding='utf-8').strip()}", flush=True)


def print_log_summary(
    out: Path, workers: int, *, tail_lines: int | None = None, scan_errors: bool = True
) -> None:
    n = tail_lines if tail_lines is not None else _log_tail_lines()
    print(f"LOG_SUMMARY tail={n} lines per component:", flush=True)
    for label, log_path in _component_logs(out, workers):
        if scan_errors:
            _scan_log_errors(log_path, label)
        _tail_file(log_path, max_lines=n, label=label)


def _write_results_summary_from_calibration(out: Path) -> list[tuple[str, Path]]:
    raw = os.environ.get(RESULTS_SUMMARY_ENV, "").strip()
    if not raw:
        return []
    dest = Path(raw)
    cal = out / "calibration.json"
    if not cal.is_file():
        return []
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(cal.read_text(encoding="utf-8"), encoding="utf-8")
    return [("results_summary.json", dest)]


def _emit_run_summary(
    out: Path,
    workers: int,
    stats: dict,
    *,
    replay_rc: int | None,
    stalled: bool,
) -> None:
    """Print STATS / ARTIFACTS / LOG_SUMMARY after stopping workers (quiet streams)."""

    cleanup()
    if _stream_logs_enabled():
        print("\n--- run_cell summary (worker/router streams stopped) ---", flush=True)
    summary_extra = _write_results_summary_from_calibration(out)
    print("STATS " + json.dumps(stats), flush=True)
    print_artifacts(out, workers, extras=summary_extra)
    fail_tail = max(_log_tail_lines(), 80)
    if replay_rc not in (0, None):
        print(f"DIAGNOSTIC replay failed (exit {replay_rc})", flush=True)
        print_log_summary(out, workers, tail_lines=fail_tail)
        raise SystemExit(1)
    if stalled or stats["success"] != stats["requests"]:
        print(
            "DIAGNOSTIC stalled or non-success rows in requests.jsonl",
            flush=True,
        )
        print_log_summary(out, workers, tail_lines=fail_tail)
        raise SystemExit(1)
    print_log_summary(out, workers)


def print_failure_diagnostics(out: Path, workers: int, *, hint: str = "") -> None:
    cleanup()
    if _stream_logs_enabled():
        print("\n--- run_cell summary (worker/router streams stopped) ---", flush=True)
    if hint:
        print(f"DIAGNOSTIC {hint}", flush=True)
    extras = _write_results_summary_from_calibration(out)
    print_artifacts(out, workers, extras=extras)
    fail_tail = max(_log_tail_lines(), 80)
    print_log_summary(out, workers, tail_lines=fail_tail, scan_errors=True)


def allocate_ports(n: int) -> dict[str, Any]:
    """Reserve TCP ports until each worker/router binds (avoids ZMQ EADDRINUSE)."""

    global _port_holders
    _release_all_port_holders()
    holders: list[socket.socket] = []

    def reserve() -> int:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        port = int(s.getsockname()[1])
        holders.append(s)
        return port

    alloc: dict[str, Any] = {
        name: [reserve() for _ in range(n)]
        for name in ["http", "grpc", "kv_pub", "kv_replay"]
    }
    alloc["router"] = reserve()
    alloc["metrics"] = reserve()
    _port_holders = holders
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
        *(
            [
                "--gpu-memory-utilization",
                str(pw["gpu_memory_utilization"]),
            ]
            if pw.get("gpu_memory_utilization") is not None
            else []
        ),
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
    global _did_cleanup
    _did_cleanup = False
    settle = float(os.environ.get("AGENTX_PORT_SETTLE_SEC", "2"))
    if settle > 0:
        time.sleep(settle)
    global ports
    ports = allocate_ports(WORKERS)
    env["HHC_KV_PUB_PORTS"] = json.dumps(ports["kv_pub"])
    print(f"PORTS allocated {json.dumps(ports)}", flush=True)
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
        try:
            engines: list[subprocess.Popen[Any]] = []
            for i in range(WORKERS):
                _release_worker_port_holders(i, WORKERS)
                engines.append(
                    start(
                        engine_command(i),
                        OUT / f"engine{i}.log",
                        {"CUDA_VISIBLE_DEVICES": GPUS[i]},
                        stream_prefix=f"[engine{i}] ",
                    )
                )
                health(
                    f"http://127.0.0.1:{ports['http'][i]}/health",
                    engines[-1],
                    timeout=1800,
                )
            print(f"ENGINES_READY {time.monotonic() - started:.0f}s", flush=True)
            _release_router_port_holders(WORKERS)
            router = start(
                router_command(), OUT / "router.log", stream_prefix="[router] "
            )
            health(f"http://127.0.0.1:{ports['router']}/health", router, timeout=180)
            print(f"ROUTER_READY {time.monotonic() - started:.0f}s", flush=True)
        except (RuntimeError, TimeoutError) as exc:
            print_failure_diagnostics(OUT, WORKERS, hint=str(exc))
            raise SystemExit(1) from exc
        snapshot(OUT / "before")
        result = OUT / "result"
        if result.exists():
            shutil.rmtree(result)
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
            replay.kill()
            replay.wait(timeout=30)
        snapshot(OUT / "after")
        replay_rc = replay.returncode
        if replay_rc not in (0, None):
            print(f"REPLAY exit code {replay_rc}", flush=True)
        wall = time.monotonic() - started
        req_path = result / "requests.jsonl"
        rows = (
            [
                json.loads(line)
                for line in req_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if req_path.is_file()
            else []
        )
        ttft = [r["ttft_seconds"] for r in rows if r.get("ttft_seconds") is not None]
        lat = [
            r["latency_seconds"] for r in rows if r.get("latency_seconds") is not None
        ]
        stats = {
            "router_policy": ROUTER_POLICY,
            "arm": ROUTER_POLICY,
            "returncode": replay_rc,
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
        _emit_run_summary(
            OUT,
            WORKERS,
            stats,
            replay_rc=replay_rc,
            stalled=stalled,
        )
    finally:
        cleanup()


if __name__ == "__main__":
    main()
