#!/usr/bin/env python3
"""Hint on and off on one engine: live probes, and a trace replay per arm.

    python run_eval.py probe --out runs/probes
    python run_eval.py replay --out runs/replay --corpus data/weka-x32 --seeds 11 12 13

Each engine is started fresh here, with this directory on PYTHONPATH, and bound to 127.0.0.1. Run it inside a network
namespace of its own on a shared host. See docs/benchmarks/kv_retention_offloading.md.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kv_retention_lifecycle import RELEASE, hint_body

HERE = Path(__file__).resolve().parent
PORT = 8123
GIB = 1024**3
POOL_TOKENS = 36_400
MAX_ATTEMPTS = 3
LONG_TURN_TTL = 30.0
OWN_TURN_TTL = 30.0
BY_SOURCE = "vllm:prompt_tokens_by_source_total"
SAMPLED = (
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
)


@dataclass(frozen=True)
class Arm:
    cpu_gib: int
    tiers: tuple[str, ...] = ()
    gpu_ttl: float = 10.0
    gpu_share: float = 0.25
    cpu_ttl: float = 120.0
    cpu_share: float = 0.5
    stock: bool = False


# The hint asks for the same on every arm: what differs is the tier it names and how much the tier acknowledges.
ARMS = {
    "off-4x": Arm(20),
    # The same as the arm without the suffix, run in another round: two runs of one arm on one seed measure
    # run-to-run spread.
    "off-4x-first": Arm(20),
    "off-1x-first": Arm(5),
    "off-4x-third": Arm(20),
    "off-1x-third": Arm(5),
    "gpu10-4x": Arm(20, ("gpu",)),
    "gpu30-4x": Arm(20, ("gpu",), gpu_ttl=30.0, gpu_share=0.5),
    "off-1x": Arm(5),
    "cpu-1x": Arm(5, ("cpu",)),
    "both-1x": Arm(5, ("gpu", "cpu"), gpu_ttl=30.0, gpu_share=0.5),
    "cpu30-1x": Arm(5, ("cpu",), cpu_ttl=30.0, cpu_share=0.9),
    "cpu30-4x": Arm(20, ("cpu",), cpu_ttl=30.0, cpu_share=0.9),
    # vLLM's own connector and LRU: a control on the prototype, which an arm without hints must match.
    "stock-4x": Arm(20, stock=True),
}
REPLAY_ORDER = ("off-1x-third", "both-1x", "off-4x-third", "gpu10-4x", "gpu30-4x")


def kv_transfer_config(
    cpu_gib: int, receipts: Path | None, stock: bool = False, **retention: Any
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "cpu_bytes_to_use": cpu_gib * GIB,
        "offload_prompt_only": False,
    }
    if stock:
        return {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": extra,
        }
    extra |= {
        "eviction_policy": "LeasePolicy",
        "cache_policy_module_path": "kv_retention_policy",
        "retention_receipts": str(receipts),
        **retention,
    }
    return {
        "kv_connector": "RetentionConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "kv_retention_connector",
        "kv_connector_extra_config": extra,
    }


def gpu_apps() -> list[int]:
    """Every process on the GPU. Raises when ``nvidia-smi`` cannot tell: no run passes on an unread check."""
    done = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return [int(pid) for pid in done.stdout.split()]


def wait_for_idle_gpu() -> None:
    """Another process on the GPU would spoil a run: wait for it to end, and say so."""
    while apps := gpu_apps():
        print(time.strftime("%H:%M:%S"), "GPU busy, waiting:", apps, flush=True)
        time.sleep(15.0)


def descends_from(pid: int, ancestor: int) -> bool:
    while pid > 1:
        if pid == ancestor:
            return True
        try:
            status = Path(f"/proc/{pid}/status").read_text()
        except OSError:
            return False
        pid = int(status.split("PPid:", 1)[1].split()[0])
    return False


class GuestWatch(threading.Thread):
    """Any other process on the GPU while an engine runs, checked every ``period`` seconds and once more at the end.
    A guest, or a check that could not be read (pid -1), makes the run invalid instead of passing silently."""

    def __init__(self, engine_pid: int, period: float = 15.0) -> None:
        super().__init__(daemon=True)
        self.engine_pid, self.period = engine_pid, period
        self.stopping = threading.Event()
        self.guests: dict[int, float] = {}

    def _check(self) -> None:
        try:
            pids = [
                pid for pid in gpu_apps() if not descends_from(pid, self.engine_pid)
            ]
        except (OSError, ValueError, subprocess.SubprocessError):
            pids = [-1]
        for pid in pids:
            self.guests.setdefault(pid, time.time())

    def run(self) -> None:
        while not self.stopping.wait(self.period):
            self._check()

    def finish(self) -> dict[str, float]:
        self.stopping.set()
        self.join(timeout=60)
        self._check()
        return {str(pid): seen for pid, seen in self.guests.items()}


class Engine:
    """``vllm serve`` as a child process: fresh, on 127.0.0.1, stopped with SIGINT and waited for."""

    def __init__(
        self, vllm: str, cores: str, log: Path, config: dict[str, Any], port: int = PORT
    ) -> None:
        self.url = f"http://127.0.0.1:{port}"
        env = {
            **os.environ,
            "PYTHONPATH": f"{HERE}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        }
        command = [
            "taskset",
            "-c",
            cores,
            vllm,
            "serve",
            "--config",
            str(HERE / "engine.yaml"),
        ]
        command += [
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--kv-transfer-config",
            json.dumps(config),
        ]
        self._log = log.open("wb")
        self._process = subprocess.Popen(
            command, env=env, stdout=self._log, stderr=subprocess.STDOUT
        )
        self.pid = self._process.pid
        self._wait_healthy()
        self.model = get_json(f"{self.url}/v1/models")["data"][0]["id"]

    def _wait_healthy(self, timeout: float = 1200.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"the engine exited with status {self._process.returncode}: see {self._log.name}"
                )
            try:
                urllib.request.urlopen(f"{self.url}/health", timeout=2).close()
                return
            except (urllib.error.URLError, OSError):
                time.sleep(1.0)
        raise TimeoutError("the engine did not answer /health")

    def stop(self) -> None:
        self._process.send_signal(signal.SIGINT)
        try:
            self._process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._log.close()

    def metrics(self) -> str:
        with urllib.request.urlopen(f"{self.url}/metrics", timeout=10) as response:
            return response.read().decode()


def get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def post_json(url: str, body: dict[str, Any], timeout: float = 600.0) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def series(exposition: str, name: str) -> dict[str, float]:
    """The samples of one metric, by their label text, summed over nothing."""
    samples = {}
    for line in exposition.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            head, _, value = line.rpartition(" ")
            samples[head[len(name) :]] = float(value)
    return samples


def tokens_by_source(exposition: str) -> dict[str, float]:
    totals = {"local_cache_hit": 0.0, "external_kv_transfer": 0.0, "local_compute": 0.0}
    for labels, value in series(exposition, BY_SOURCE).items():
        for source in totals:
            if f'source="{source}"' in labels:
                totals[source] += value
    return totals


class Sampler(threading.Thread):
    """The engine's KV use and queues over time, for the time-averaged cost of a run."""

    def __init__(self, engine: Engine, path: Path, period: float = 2.0) -> None:
        super().__init__(daemon=True)
        self.engine, self.path, self.period = engine, path, period
        self.stopping = threading.Event()

    def run(self) -> None:
        with self.path.open("w") as out:
            while not self.stopping.wait(self.period):
                try:
                    exposition = self.engine.metrics()
                except (urllib.error.URLError, OSError):
                    continue
                row = {name: sum(series(exposition, name).values()) for name in SAMPLED}
                out.write(json.dumps({"t": time.time(), **row}) + "\n")
                out.flush()


class Probe:
    """Token-id completions of one token, read from ``prompt_tokens_by_source`` around each request."""

    def __init__(self, engine: Engine, seed: int = 427) -> None:
        self.engine = engine
        self.rng = random.Random(seed)

    def prompt(self, tokens: int) -> list[int]:
        return [self.rng.randrange(1000, 30000) for _ in range(tokens)]

    def send(
        self,
        prompt: list[int],
        hint: dict[str, Any] | None = None,
        max_tokens: int = 1,
        ignore_eos: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.engine.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": ignore_eos,
        }
        if hint is not None:
            body["kv_transfer_params"] = {"kv_retention": hint}
        before = tokens_by_source(self.engine.metrics())
        started = time.perf_counter()
        response = post_json(f"{self.engine.url}/v1/completions", body)
        seconds = time.perf_counter() - started
        # The engine updates its counters a moment after it answers.
        time.sleep(0.3)
        after = tokens_by_source(self.engine.metrics())
        return {
            "seconds": round(seconds, 3),
            "by_source": {
                source: round(after[source] - before[source]) for source in after
            },
            "ack": (response.get("kv_transfer_params") or {}).get("kv_retention"),
        }

    def fill(self, tokens: int, chunk: int = 12_000) -> None:
        """Push older blocks out with unique prompts, one at a time."""
        while tokens > 0:
            self.send(self.prompt(min(chunk, tokens)))
            tokens -= chunk


def receipts_of(path: Path, event: str) -> list[dict[str, Any]]:
    return [
        row
        for row in map(json.loads, path.read_text().splitlines())
        if row["event"] == event
    ]


def share(row: dict[str, Any], source: str, tokens: int) -> float:
    return row["by_source"][source] / tokens


def probe_gpu_hold(engine: Engine, receipts: Path) -> dict[str, Any]:
    """A hold keeps a prefix a GPU hit past what would have pushed it out; without one it comes back from the CPU tier."""
    probe = Probe(engine)
    tokens = 8_000
    off_prompt, on_prompt = probe.prompt(tokens), probe.prompt(tokens)
    probe.send(off_prompt)
    probe.fill(int(POOL_TOKENS * 1.5))
    off = probe.send(off_prompt)
    holds_after_off = len(receipts_of(receipts, "ack"))
    first = probe.send(on_prompt, hint_body("gpu-probe", 0, ["gpu"], 300.0))
    probe.fill(int(POOL_TOKENS * 1.5))
    on = probe.send(on_prompt, hint_body("gpu-probe", 1, ["gpu"], 300.0))
    # A conversation's last turn asks for nothing, and still ends the hold the turn before it left.
    last_turn = probe.send(on_prompt, hint_body("gpu-probe", 2, [], 300.0))
    ended = [
        row["outcome"]
        for row in receipts_of(receipts, "end")
        if row["hint_id"] == "gpu-probe:1"
    ]
    probe.send(probe.prompt(16), hint_body("gpu-probe", 3, ["gpu"], 300.0, op=RELEASE))
    return {
        "off": off,
        "ack": first["ack"],
        "on": on,
        "control_holds_with_hint_off": holds_after_off,
        "last_turn": last_turn,
        "last_turn_ended": ended,
        "passed": share(off, "external_kv_transfer", tokens) > 0.9
        and share(on, "local_cache_hit", tokens) > 0.9
        and share(on, "external_kv_transfer", tokens) < 0.1
        and holds_after_off == 0
        and ended == ["used"],
    }


def probe_refusals(engine: Engine) -> dict[str, Any]:
    """Over budget, a hint is refused and the request is served; so is a hint this prototype cannot read."""
    probe = Probe(engine)
    budget = int(POOL_TOKENS * 0.25)
    too_big = probe.send(
        probe.prompt(budget + 2_000), hint_body("refused", 0, ["gpu"], 60.0)
    )
    unknown = probe.send(probe.prompt(64), {"version": "9.9", "op": "retain"})
    return {
        "over_budget": too_big["ack"],
        "unsupported_version": unknown,
        "passed": too_big["ack"]["acks"][0]["state"] == "rejected"
        and too_big["ack"]["acks"][0]["reason"]
        in ("over_tenant_budget", "over_engine_budget")
        and unknown["ack"] is None,
    }


def probe_expiry(engine: Engine, receipts: Path) -> dict[str, Any]:
    """On an idle engine a hold ends at its TTL and is gone: the prefix comes back from the CPU tier."""
    probe = Probe(engine)
    tokens = 8_000
    prompt = probe.prompt(tokens)
    first = probe.send(prompt, hint_body("expiry", 0, ["gpu"], 3.0))
    time.sleep(5.0)
    ends = [row for row in receipts_of(receipts, "end") if row["session"] == "expiry"]
    probe.fill(int(POOL_TOKENS * 1.5))
    after = probe.send(prompt)
    return {
        "ack": first["ack"],
        "ends": ends,
        "after": after,
        "passed": len(ends) == 1
        and ends[0]["outcome"] == "expired"
        and ends[0]["held_seconds"] < 3.5
        and share(after, "external_kv_transfer", tokens) > 0.9,
    }


def probe_cpu_lease(engine: Engine) -> dict[str, Any]:
    """With a 1x CPU tier, a lease keeps a prefix a restore past what would have pushed it out of both tiers."""
    probe = Probe(engine)
    tokens = 8_000
    off_prompt, on_prompt = probe.prompt(tokens), probe.prompt(tokens)
    probe.send(off_prompt)
    probe.fill(POOL_TOKENS * 3)
    off = probe.send(off_prompt)
    first = probe.send(on_prompt, hint_body("cpu-probe", 0, ["cpu"], 300.0))
    probe.fill(POOL_TOKENS * 3)
    on = probe.send(on_prompt, hint_body("cpu-probe", 1, ["cpu"], 300.0))
    return {
        "off": off,
        "ack": first["ack"],
        "on": on,
        "passed": share(off, "local_compute", tokens) > 0.9
        and share(on, "external_kv_transfer", tokens) > 0.9,
    }


def probe_cpu_long_turn(engine: Engine) -> dict[str, Any]:
    """A turn longer than the lease's TTL still leaves its prompt leased for the pause after it. Run with a TTL shorter
    than the turn and longer than the filler."""
    probe = Probe(engine)
    tokens = 8_000
    prompt = probe.prompt(tokens)
    turn = probe.send(
        prompt,
        hint_body("long-turn", 0, ["cpu"], 300.0),
        max_tokens=3_000,
        ignore_eos=True,
    )
    started = time.perf_counter()
    probe.fill(POOL_TOKENS * 3)
    filled = time.perf_counter() - started
    after = probe.send(prompt, hint_body("long-turn", 1, ["cpu"], 300.0))
    return {
        "ack": turn["ack"],
        "turn_seconds": turn["seconds"],
        "filler_seconds": round(filled, 3),
        "after": after,
        "passed": turn["seconds"] > LONG_TURN_TTL
        and filled < LONG_TURN_TTL
        and share(after, "external_kv_transfer", tokens) > 0.9,
    }


def probe_gpu_own_turn(engine: Engine, receipts: Path) -> dict[str, Any]:
    """A session's next prompt that does not fit beside the session's own hold is not kept out until the hold's TTL:
    vLLM admits a request only if its whole prompt fits. One next prompt shares nothing with the held one, as after a
    compaction, and one shares its head, which still hits. Run with the 30 s hold's budget."""
    probe = Probe(engine)
    held, following, shared = 17_800, 21_000, 2_000
    turns = {}
    for session, head in (("own-turn-a", 0), ("own-turn-b", shared)):
        first = probe.prompt(held)
        probe.send(first, hint_body(session, 0, ["gpu"], OWN_TURN_TTL))
        turns[session] = probe.send(
            first[:head] + probe.prompt(following - head),
            hint_body(session, 1, ["gpu"], OWN_TURN_TTL),
        )
    ends = {
        row["hint_id"]: row
        for row in receipts_of(receipts, "end")
        if row["session"].startswith("own-turn-")
    }
    return {
        "turns": turns,
        "ends": ends,
        "passed": all(
            turn["seconds"] < OWN_TURN_TTL / 2
            and (ends.get(f"{session}:0") or {}).get("reason") == "own_turn"
            for session, turn in turns.items()
        )
        and share(turns["own-turn-b"], "local_cache_hit", shared) > 0.9,
    }


def probe_overload(engine: Engine, receipts: Path) -> dict[str, Any]:
    """Holds give way: a waiting request that does not fit releases the oldest, and a running one that cannot grow does
    too, with no preemption. Run with a budget that lets holds fill most of the pool."""
    probe = Probe(engine)
    for index in range(3):
        probe.send(
            probe.prompt(10_000), hint_body(f"overload-{index}", 0, ["gpu"], 300.0)
        )
    alone = probe.send(probe.prompt(20_000))  # admitted once holds give way
    ends = [
        row
        for row in receipts_of(receipts, "end")
        if row["session"].startswith("overload-")
    ]
    for index in range(3, 6):
        probe.send(
            probe.prompt(10_000), hint_body(f"overload-{index}", 0, ["gpu"], 300.0)
        )
    preemptions = sum(series(engine.metrics(), "vllm:num_preemptions_total").values())
    grown = probe.send(probe.prompt(5_000), max_tokens=2_400, ignore_eos=True)
    preempted = (
        sum(series(engine.metrics(), "vllm:num_preemptions_total").values())
        - preemptions
    )
    all_ends = [
        row
        for row in receipts_of(receipts, "end")
        if row["session"].startswith("overload-")
    ]
    return {
        "waiting_request": alone,
        "ends_for_waiting_request": ends,
        "growing_request": grown,
        "preemptions_while_growing": preempted,
        "ends": all_ends,
        "passed": any(row["outcome"] == "evicted" for row in ends)
        and alone["seconds"] < 60
        and preempted == 0
        and sum(row["outcome"] == "evicted" for row in all_ends) > len(ends),
    }


def probe(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    def on_engine(name: str, cpu_gib: int, run: Any, **retention: Any) -> None:
        receipts = out / f"{name}.receipts.jsonl"
        receipts.unlink(missing_ok=True)
        wait_for_idle_gpu()
        engine = Engine(
            args.vllm,
            args.engine_cores,
            out / f"{name}.engine.log",
            kv_transfer_config(cpu_gib, receipts, **retention),
        )
        watch = GuestWatch(engine.pid)
        watch.start()
        try:
            results[name] = run(engine, receipts)
        except Exception as error:  # noqa: BLE001 - one probe's failure must not lose the others' results
            results[name] = {"passed": False, "error": repr(error)}
        finally:
            guests = watch.finish()
            engine.stop()
        results[name]["guest_gpu_apps"] = guests
        results[name]["passed"] = results[name]["passed"] and not guests
        (out / "probes.json").write_text(json.dumps(results, indent=1))
        print(name, "PASS" if results[name]["passed"] else "FAIL", flush=True)

    selected = args.only or [
        "gpu_hold",
        "refusals",
        "expiry",
        "cpu_lease",
        "cpu_long_turn",
        "gpu_own_turn",
        "overload",
    ]
    hold_ttl = {"retention_gpu_max_ttl_seconds": 300.0}
    if "gpu_hold" in selected:
        on_engine("gpu_hold", 20, probe_gpu_hold, **hold_ttl)
    if "refusals" in selected:
        on_engine("refusals", 20, lambda engine, _: probe_refusals(engine))
    if "expiry" in selected:
        on_engine("expiry", 20, probe_expiry, **hold_ttl)
    if "cpu_lease" in selected:
        on_engine(
            "cpu_lease",
            5,
            lambda engine, _: probe_cpu_lease(engine),
            retention_cpu_max_ttl_seconds=300.0,
        )
    if "cpu_long_turn" in selected:
        on_engine(
            "cpu_long_turn",
            5,
            lambda engine, _: probe_cpu_long_turn(engine),
            retention_cpu_max_ttl_seconds=LONG_TURN_TTL,
        )
    if "gpu_own_turn" in selected:
        on_engine(
            "gpu_own_turn",
            20,
            probe_gpu_own_turn,
            retention_gpu_share=0.5,
            retention_gpu_max_ttl_seconds=OWN_TURN_TTL,
        )
    if "overload" in selected:
        on_engine("overload", 20, probe_overload, retention_gpu_share=0.9, **hold_ttl)
    if not all(result["passed"] for result in results.values()):
        sys.exit(1)


def replay_one(
    args: argparse.Namespace, name: str, seed: int, run_dir: Path
) -> dict[str, Any]:
    arm = ARMS[name]
    run_dir.mkdir(parents=True, exist_ok=True)
    receipts = run_dir / "receipts.jsonl"
    receipts.unlink(missing_ok=True)
    retention = {
        "retention_gpu_share": arm.gpu_share,
        "retention_gpu_max_ttl_seconds": arm.gpu_ttl,
        "retention_cpu_share": arm.cpu_share,
        "retention_cpu_max_ttl_seconds": arm.cpu_ttl,
    }
    config = kv_transfer_config(arm.cpu_gib, receipts, arm.stock, **retention)
    wait_for_idle_gpu()
    started = time.time()
    engine = Engine(args.vllm, args.engine_cores, run_dir / "engine.log", config)
    sampler = Sampler(engine, run_dir / "samples.jsonl")
    watch = GuestWatch(engine.pid)
    try:
        (run_dir / "metrics_start.txt").write_text(engine.metrics())
        sampler.start()
        watch.start()
        env = {
            **os.environ,
            "KV_RETENTION_URL": f"{engine.url}/v1/chat/completions",
            "KV_RETENTION_CORPUS": args.corpus,
            "KV_RETENTION_SEED": str(seed),
            "KV_RETENTION_SESSIONS": str(args.sessions),
            "KV_RETENTION_DURATION": str(args.duration),
            "KV_RETENTION_ARTIFACTS": str(run_dir / "aiperf"),
            "KV_RETENTION_TIERS": ",".join(arm.tiers),
            "KV_RETENTION_TTL_SECONDS": str(args.hint_ttl),
            "PYTHONPATH": str(HERE),
            "AIPERF_DATASET_MMAP_CACHE_ENABLED": "true",
            "AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID": "false",
            "AIPERF_DATASET_WEKA_PARALLEL_WORKERS": "4",
            "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "1800",
            "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "1800",
            "FLATSHARE_BENCH_CLASS": "background",
        }
        command = [
            "taskset",
            "-c",
            args.load_cores,
            args.aiperf,
            "profile",
            "--config",
            str(HERE / "agentic_replay.yaml"),
        ]
        with (run_dir / "aiperf.log").open("wb") as log:
            status = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT
            ).returncode
        time.sleep(3.0)
        (run_dir / "metrics_end.txt").write_text(engine.metrics())
        guests = watch.finish()
    finally:
        sampler.stopping.set()
        sampler.join(timeout=5)
        engine.stop()
    meta = {
        "arm": name,
        "seed": seed,
        "started": started,
        "ended": time.time(),
        "aiperf_status": status,
        "guest_gpu_apps": guests,
        "valid": status == 0 and not guests,
        "retention": retention,
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1))
    return meta


def replay(args: argparse.Namespace) -> None:
    for seed in args.seeds:
        for name in args.arms:
            run_dir = Path(args.out) / name / str(seed)
            if (run_dir / "run.json").exists():
                continue
            for attempt in range(MAX_ATTEMPTS):
                print(time.strftime("%H:%M:%S"), name, seed, flush=True)
                meta = replay_one(args, name, seed, run_dir)
                if meta["valid"]:
                    break
                # Kept as a record, and report.py skips it. The next attempt waits for an idle GPU.
                discarded = run_dir.with_name(f"{seed}.invalid-{int(time.time())}")
                run_dir.rename(discarded)
                print("INVALID", discarded, meta["guest_gpu_apps"], flush=True)
            else:
                sys.exit(f"{name} seed {seed}: no valid run in {MAX_ATTEMPTS} attempts")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--vllm", default="vllm")
    parser.add_argument("--engine-cores", default="8-15")
    commands = parser.add_subparsers(dest="command", required=True)
    probes = commands.add_parser("probe")
    probes.add_argument("--out", required=True)
    probes.add_argument("--only", nargs="*")
    probes.set_defaults(run=probe)
    replays = commands.add_parser("replay")
    replays.add_argument("--out", required=True)
    replays.add_argument("--corpus", required=True)
    replays.add_argument("--aiperf", default="aiperf")
    replays.add_argument("--load-cores", default="0-7")
    replays.add_argument("--seeds", nargs="+", type=int, default=[11, 12, 13])
    replays.add_argument(
        "--arms", nargs="+", default=list(REPLAY_ORDER), choices=sorted(ARMS)
    )
    replays.add_argument("--sessions", type=int, default=16)
    replays.add_argument("--duration", type=int, default=600)
    replays.add_argument(
        "--hint-ttl",
        type=float,
        default=120.0,
        help="what the hint asks for, on every tier",
    )
    replays.set_defaults(run=replay)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
