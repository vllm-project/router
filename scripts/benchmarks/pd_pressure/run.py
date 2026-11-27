#!/usr/bin/env python3
"""Record reproducible CPU stress or explicit vLLM benchmarks; never launch servers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit

STAGES = {
    1: (("admission", ["--lib"], "routers::http::pd_router::admission_tests::pd_cache_aware_pr1_stress"),
        ("http", ["--test", "test_pd_routing"], "test_pd_routing::http_pd_cache_aware_pr1_stress"),
        ("grpc", ["--test", "grpc_pd_routing"], "grpc_pd_cache_aware_pr1_stress")),
    2: None, 3: None, 4: None,
}
RESULT = re.compile(r"test result: (ok|FAILED)\. (\d+) passed; (\d+) failed;")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def machine_fingerprint():
    data = {"cpu_model": None, "visible_logical_cpus": os.cpu_count(), "memory_bytes": None}
    if sys.platform.startswith("linux"):
        try:
            fields = (line.partition(":") for line in Path("/proc/cpuinfo").read_text().splitlines())
            data["cpu_model"] = next((value.strip() for key, _, value in fields
                                      if key.strip() in ("model name", "Hardware")), None)
        except OSError:
            pass
        try:
            data["visible_logical_cpus"] = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            pass
        try:
            pages, page_size = (os.sysconf(key) for key in ("SC_PHYS_PAGES", "SC_PAGE_SIZE"))
            if pages > 0 and page_size > 0:
                data["memory_bytes"] = pages * page_size
        except (AttributeError, OSError, ValueError):
            pass
    return data


def sha256(path):
    with Path(path).open("rb") as stream:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(stream, "sha256").hexdigest()
        return _hash_stream(stream)


def _hash_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def base_url(value):
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise argparse.ArgumentTypeError("expected an explicit http(s) router URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("URL must not contain credentials, query, or fragment")
    try:
        parsed.port
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value.rstrip("/")


def cargo_command(executable, stage, suite=0, profile="release"):
    if STAGES[stage] is None:
        raise ValueError(f"PR{stage} stress test is not implemented")
    _, target, test_filter = STAGES[stage][suite]
    return [executable, "test", "--locked", *(["--release"] if profile == "release" else []), "--features", "cpu,ready",
            *target, "--message-format=json", test_filter, "--", "--ignored", "--exact",
            "--nocapture", "--test-threads=1"]


def cargo_verdict(returncode, output):
    summaries = RESULT.findall(output)
    if returncode != 0:
        return "failed"
    if not summaries or sum(int(passed) for _, passed, _ in summaries) == 0:
        return "zero_tests"
    return "passed" if all(kind == "ok" and int(failed) == 0 for kind, _, failed in summaries) else "failed"


def test_artifacts(output):
    artifacts = set()
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(event, dict) and event.get("reason") == "compiler-artifact":
            if event.get("profile", {}).get("test") and event.get("executable"):
                artifacts.add(event["executable"])
    return sorted(artifacts)


def stress_result(output, suite):
    marker = "PD_PRESSURE_STRESS_RESULT "
    records = [json.loads(line.split(marker, 1)[1]) for line in output.splitlines() if marker in line]
    if len(records) != 1:
        raise ValueError("expected exactly one PD_PRESSURE_STRESS_RESULT record")
    result = records[0]
    positive = lambda value: type(value) is int and value > 0
    if not isinstance(result, dict) or not all(positive(result.get(key)) for key in ("rounds", "concurrency")):
        raise ValueError("invalid stress rounds/concurrency")
    if result.get("attempts" if suite == "admission" else "requests") != result["rounds"] * result["concurrency"]:
        raise ValueError("stress count must equal rounds * concurrency")
    if suite == "admission":
        valid = (result.get("kind") == "pair_admission" and positive(result.get("attempts"))
                 and result.get("final_load_entries") == 0 and result.get("sentinel_preserved") is True)
    else:
        valid = (result.get("kind") == {"http": "http_pd", "grpc": "grpc_pd"}.get(suite)
                 and suite in ("http", "grpc") and positive(result.get("requests"))
                 and result.get("requests") == result.get("completed") == result.get("handoffs_verified")
                 and result.get("final_occupancy") == 0)
    seconds = result.get("wall_seconds")
    if not valid or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("stress result is incomplete or reports an invariant failure")
    return result


def serve_command(args, url, result_dir, seed):
    return [args.vllm, "bench", "serve", "--backend", "openai-chat",
            "--endpoint", "/v1/chat/completions", "--base-url", url,
            "--model", args.model, "--dataset-name", "random",
            "--random-input-len", str(args.input_len),
            "--random-output-len", str(args.output_len),
            "--num-prompts", str(args.num_prompts), "--request-rate", str(args.request_rate),
            "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95,99",
            "--seed", str(seed), "--save-result", "--save-detailed", "--result-dir", str(result_dir),
            "--result-filename", "metrics.json"]


class Recorder:
    def __init__(self, directory, repo, timeout):
        self.directory, self.repo, self.timeout = directory, repo, timeout
        directory.mkdir(parents=True, exist_ok=False)
        self.data = {"started_at": utc_now(), "status": "running", "commands": [],
                     "runner_argv": sys.argv, "python": sys.version,
                     "platform": platform.platform(), "machine": machine_fingerprint(), "repo": str(repo),
                     "environment": {key: os.environ.get(key) for key in
                                     ("CARGO_TARGET_DIR", "CARGO_BUILD_JOBS", "RUSTUP_TOOLCHAIN",
                                      "CARGO_PROFILE_DEV_DEBUG", "CARGO_PROFILE_TEST_DEBUG", "CARGO_INCREMENTAL")}}
        self.save()

    def save(self):
        (self.directory / "manifest.json").write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def run(self, name, argv, timeout=None, env_updates=None):
        record = {"name": name, "argv": argv, "cwd": str(self.repo), "started_at": utc_now()}
        record["environment_overrides"] = env_updates or {}
        log = self.directory / f"{len(self.data['commands']):02d}-{name}.log"
        record["raw_output"] = log.name
        started = time.monotonic()
        try:
            with log.open("wb") as stream:
                proc = subprocess.Popen(argv, cwd=self.repo, stdout=stream, stderr=subprocess.STDOUT,
                                        start_new_session=(os.name == "posix"),
                                        env={**os.environ, **(env_updates or {})})
                try:
                    code = proc.wait(timeout=timeout or self.timeout)
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                    record["timed_out"] = isinstance(exc, subprocess.TimeoutExpired)
                    record["interrupted"] = isinstance(exc, KeyboardInterrupt)
                    if os.name == "posix":
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                    proc.wait()
                    code = 124 if record["timed_out"] else 130
        except OSError as exc:
            log.write_text(str(exc) + "\n", encoding="utf-8")
            code = 127
        record.update(returncode=code, elapsed_seconds=time.monotonic() - started, ended_at=utc_now())
        self.data["commands"].append(record)
        self.save()
        return code, log.read_text(encoding="utf-8", errors="replace")

    def finish(self, status, code, detail=None):
        self.data.update(status=status, exit_code=code, ended_at=utc_now(), detail=detail)
        self.save()
        print(json.dumps({"status": status, "manifest": str(self.directory / "manifest.json"),
                          "detail": detail}, ensure_ascii=False))
        return code


def resolve_executable(value):
    resolved = shutil.which(value)
    if resolved is None:
        raise ValueError(f"executable not found: {value}")
    return str(Path(resolved).absolute())


def provenance(recorder, args):
    git = ["git", "-c", "core.autocrlf=true"]
    def required(name, command, timeout=30):
        code, output = recorder.run(name, command, timeout)
        if code:
            raise ValueError(f"required provenance command failed: {name} (exit {code})")
        return output

    for label, ref in (("base", args.base_ref), ("head", "HEAD")):
        output = required(f"git-{label}", [*git, "rev-parse", "--verify", f"{ref}^{{commit}}"])
        recorder.data[f"{label}_revision"] = output.strip()
    dirty = required("git-status", [*git, "status", "--porcelain", "--untracked-files=normal"])
    recorder.data["working_tree_dirty"] = bool(dirty.strip())
    required("source-patch", [*git, "diff", "--binary", "HEAD"])
    untracked = required("untracked-sources", [*git, "ls-files", "--others", "--exclude-standard", "-z", "--",
                         "src", "tests", "scripts/benchmarks/pd_pressure", "docs/benchmarks/pd_pressure.md",
                         "Cargo.toml", "Cargo.lock", "build.rs"])
    recorder.data["untracked_source_sha256"] = {}
    for relative in filter(None, untracked.split("\0")):
        source = (args.repo / relative).resolve()
        if not source.is_relative_to(args.repo):
            raise ValueError(f"source snapshot escapes repository: {relative}")
        destination = recorder.directory / "source-snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        recorder.data["untracked_source_sha256"][relative] = sha256(destination)
    recorder.data["runner_sha256"] = sha256(__file__)
    executable = args.cargo if args.action == "stress" else args.vllm
    recorder.data["tool_executable"] = executable
    recorder.data["tool_sha256"] = sha256(executable)
    required("tool-version", [executable, "--version"], 60)
    if args.action == "stress":
        required("rustc-version", ["rustc", "--version", "--verbose"])
    if args.action == "serve":
        recorder.data["endpoint_binary_mapping"] = {
            arm: {"url": getattr(args, arm + "_url"), "binary": str(getattr(args, arm + "_binary")),
                  "sha256": sha256(getattr(args, arm + "_binary"))}
            for arm in ("baseline", "candidate")}
        recorder.data["mapping_verified"] = False  # Caller supplies remote provenance; no attestation.
    recorder.save()


def probe(recorder, executable, command):
    code, output = recorder.run(f"help-{command}", [executable, "bench", command, "--help"], 90)
    available = code == 0 and "usage:" in output.lower()
    if available and "--help=all" in output:
        code, output = recorder.run(f"help-{command}-all", [executable, "bench", command, "--help=all"], 90)
        available = code == 0 and "usage:" in output.lower()
    recorder.data.setdefault("capabilities", {})[command] = {"available": available, "returncode": code}
    recorder.save()
    return output if available else None


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("stress", "serve", "forward", "probe"))
    cli.add_argument("--stage", type=int, choices=STAGES, default=1)
    cli.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    cli.add_argument("--base-ref", required=True, help="explicit comparison commit/ref")
    cli.add_argument("--out-dir", type=Path, required=True, help="new artifact directory; never overwritten")
    cli.add_argument("--cargo", default="cargo")
    cli.add_argument("--profile", choices=("test", "release"), default="release")
    cli.add_argument("--vllm", help="explicit vLLM executable; required except for stress")
    cli.add_argument("--repeat", type=positive_int, default=2)
    cli.add_argument("--stress-rounds", type=positive_int, default=os.environ.get("ROUTER_PD_STRESS_ROUNDS", "24"))
    cli.add_argument("--stress-concurrency", type=positive_int, default=os.environ.get("ROUTER_PD_STRESS_CONCURRENCY", "16"))
    cli.add_argument("--timeout", type=positive_float, default=3600)
    cli.add_argument("--baseline-url", type=base_url)
    cli.add_argument("--candidate-url", type=base_url)
    cli.add_argument("--baseline-binary", type=Path)
    cli.add_argument("--candidate-binary", type=Path)
    cli.add_argument("--model")
    cli.add_argument("--input-len", type=positive_int, default=1024)
    cli.add_argument("--output-len", type=positive_int, default=64)
    cli.add_argument("--num-prompts", type=positive_int, default=32)
    cli.add_argument("--request-rate", type=positive_float, default=4)
    cli.add_argument("--seed", type=int, default=17)
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    args.repo, args.out_dir = args.repo.resolve(), args.out_dir.resolve()
    if args.action != "stress" and not args.vllm:
        cli.error("--vllm is required for probe, serve, and forward")
    if args.action in ("serve", "forward") and not args.model:
        cli.error("--model is required")
    if args.action == "serve":
        if not all((args.baseline_url, args.candidate_url, args.baseline_binary, args.candidate_binary)):
            cli.error("serve requires both URLs and both binary paths")
        for arm in ("baseline", "candidate"):
            path = getattr(args, arm + "_binary").resolve()
            if not path.is_file():
                cli.error(f"{arm} binary does not exist: {path}")
            setattr(args, arm + "_binary", path)
    if not args.repo.is_dir() or args.out_dir.exists():
        cli.error("--repo must exist and --out-dir must be a new directory")
    recorder = Recorder(args.out_dir, args.repo, args.timeout)
    recorder.data.update(stage=args.stage, action=args.action, parameters=vars(args).copy())
    recorder.data["parameters"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    try:
        if args.action != "probe" and STAGES[args.stage] is None:
            return recorder.finish("not_implemented", 3, f"PR{args.stage} stress contract is not implemented")
        if args.action == "stress":
            args.cargo = resolve_executable(args.cargo)
        else:
            args.vllm = resolve_executable(args.vllm)
        provenance(recorder, args)
        if args.action == "stress":
            environment = {"ROUTER_PD_STRESS_ROUNDS": str(args.stress_rounds),
                           "ROUTER_PD_STRESS_CONCURRENCY": str(args.stress_concurrency)}
            for iteration in range(args.repeat):
                for index, (suite, _, _) in enumerate(STAGES[args.stage]):
                    name = f"stress-{iteration + 1}-{suite}"
                    code, output = recorder.run(name, cargo_command(args.cargo, args.stage, index, args.profile), env_updates=environment)
                    verdict = cargo_verdict(code, output)
                    hashes = {path: sha256(path) for path in test_artifacts(output) if Path(path).is_file()}
                    recorder.data.setdefault("test_binaries", {}).update(hashes)
                    for path in hashes:
                        stat = Path(path).stat()
                        recorder.data.setdefault("test_binary_metadata", {})[path] = {
                            "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                    if verdict != "passed":
                        return recorder.finish(verdict, 1, "inspect raw output; zero tests are never a pass")
                    if not hashes:
                        return recorder.finish("missing_artifact", 1, "no hashable Cargo test executable was reported")
                    result = stress_result(output, suite)
                    if result["rounds"] != args.stress_rounds or result["concurrency"] != args.stress_concurrency:
                        raise ValueError("stress result parameters differ from requested rounds/concurrency")
                    (args.out_dir / f"{name}.result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                    recorder.data.setdefault("stress_results", []).append({"iteration": iteration + 1, "suite": suite, **result})
                    recorder.save()
            return recorder.finish("passed", 0, "CPU stress only; no router TTFT or GPU benefit measured")
        recorder.run("bench-help", [args.vllm, "bench", "--help"], 90)
        if args.action == "probe":
            serve_help, forward_help = probe(recorder, args.vllm, "serve"), probe(recorder, args.vllm, "forward")
            code = 0 if serve_help is not None and forward_help is not None else 3
            return recorder.finish("available" if code == 0 else "partial_or_unavailable", code,
                                   "capability probe only; no benchmark executed")
        help_text = probe(recorder, args.vllm, args.action)
        if help_text is None:
            return recorder.finish("unsupported", 3, f"installed CLI does not support bench {args.action}")
        if args.action == "forward":
            if "--model" not in help_text:
                return recorder.finish("unsupported", 3, "forward CLI lacks the expected --model interface")
            code, _ = recorder.run("forward", [args.vllm, "bench", "forward", "--model", args.model])
            return recorder.finish("diagnostic_completed" if code == 0 else "failed", 0 if code == 0 else 1,
                                   "GPU forward diagnostic only; bypasses scheduler and cannot measure router TTFT")
        for iteration in range(args.repeat):
            arms = ("baseline", "candidate") if iteration % 2 == 0 else ("candidate", "baseline")
            for arm in arms:
                name = f"serve-{iteration + 1}-{arm}"
                result_dir = args.out_dir / name
                command = serve_command(args, getattr(args, arm + "_url"), result_dir, args.seed + iteration)
                missing = [flag for flag in command[3:] if flag.startswith("--") and flag not in help_text]
                if missing:
                    return recorder.finish("unsupported", 3, f"serve CLI is missing required options: {missing}")
                result_dir.mkdir()
                code, _ = recorder.run(name, command)
                metrics = result_dir / "metrics.json"
                if code != 0 or not metrics.is_file():
                    return recorder.finish("failed", 1, f"{name}: nonzero exit or missing metrics.json")
                payload = json.loads(metrics.read_text(encoding="utf-8"))
                if not isinstance(payload, dict) or payload.get("completed", 0) != args.num_prompts:
                    return recorder.finish("incomplete_requests", 1, f"{name}: completed count differs from requested count")
        return recorder.finish("completed_unassessed", 0, "raw paired serve runs completed; no performance acceptance decision")
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        return recorder.finish("error", 1, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
