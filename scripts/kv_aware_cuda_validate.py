#!/usr/bin/env python3
"""Finite, evidence-driven validation against two user-owned CUDA workers.

Standard library only. This program never provisions instances, opens SSH
connections, starts/stops servers, or publishes code. Run on the GPU host with
the candidate source, native executable, and processes in the same namespace.
"""

import argparse
import hashlib
import http.client
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


BASE = "bc16b190f8875a275287dd70ce5b4c9e54373dd6"
MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODELSCOPE_REVISION = "09b42cad3d112e832108974449ccb5e8e0f5b5d1"
VLLM_VERSION = "0.29.0"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
MODELSCOPE_REQUIRED_SHA256 = {
    "model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "tokenizer.json": TOKENIZER_SHA256,
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
}
MODELSCOPE_FILES = {
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "merges.txt", "vocab.json", "LICENSE", "model.safetensors",
}
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_digest(tokens):
    return hashlib.sha256(b"".join(t.to_bytes(4, "big") for t in tokens)).hexdigest()


def command(argv, cwd=None):
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)
    require(result.returncode == 0, f"{argv[0]} failed: {result.stderr[-2000:]}")
    return result.stdout.strip()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_identity(source, expected):
    source = Path(source).resolve()
    head = command(["git", "rev-parse", "HEAD"], source)
    require(re.fullmatch(r"[0-9a-f]{40}", expected), "candidate must be a full commit SHA")
    require(head == expected, f"source HEAD {head} != candidate {expected}")
    require(not command(["git", "status", "--porcelain"], source), "candidate source is dirty")
    command(["git", "merge-base", "--is-ancestor", BASE, head], source)
    return {"candidate_sha": head, "base_sha": BASE,
            "tree_sha": command(["git", "rev-parse", "HEAD^{tree}"], source)}


def verify_model_manifest(path):
    """Verify the approved local ModelScope snapshot, not an HF weight identity."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    require(manifest.get("status") == "verified" and manifest.get("source") == "ModelScope"
            and manifest.get("model") == MODEL
            and manifest.get("source_commit") == MODELSCOPE_REVISION,
            "model manifest is not the approved immutable ModelScope snapshot")
    output = Path(manifest.get("output", ""))
    require(output.is_absolute() and output.is_dir(), "model manifest needs an existing absolute output directory")
    output = output.resolve()
    required = manifest.get("required_files", [])
    require(len(required) == len(MODELSCOPE_FILES) and set(required) == MODELSCOPE_FILES,
            "model manifest required-file set is incomplete")
    entries = manifest.get("files", [])
    require(isinstance(entries, list) and len(entries) == len(MODELSCOPE_FILES),
            "model manifest must describe every required model file exactly once")
    verified = {}
    for entry in entries:
        name = entry.get("path")
        require(name in MODELSCOPE_FILES and name not in verified, "unexpected or duplicate model file")
        target = (output / name).resolve()
        require(target.parent == output and target.is_file(), "model file is missing or escapes its snapshot directory")
        actual_sha = sha256(target)
        require(entry.get("status") == "verified" and entry.get("source_commit") == MODELSCOPE_REVISION
                and actual_sha == entry.get("sha256") == entry.get("api_sha256")
                and target.stat().st_size == entry.get("bytes"),
                f"model file differs from its ModelScope manifest: {name}")
        if name in MODELSCOPE_REQUIRED_SHA256:
            require(actual_sha == MODELSCOPE_REQUIRED_SHA256[name], f"unapproved model artifact: {name}")
        verified[name] = {"sha256": actual_sha, "bytes": target.stat().st_size}
    metadata_names = {"download-manifest.json", "download-events.jsonl", "modelscope-api-files.json"}
    require({item.name for item in output.iterdir()} <= MODELSCOPE_FILES | metadata_names,
            "unmanifested files exist in the local model snapshot")
    listing = output / "modelscope-api-files.json"
    require(listing.resolve().parent == output and sha256(listing) == manifest.get("source_listing_sha256"),
            "ModelScope source listing differs from the download manifest")
    return {"source": "ModelScope", "source_commit": MODELSCOPE_REVISION, "model": MODEL,
            "local_dir": str(output), "manifest": str(path), "manifest_sha256": sha256(path),
            "source_listing_sha256": sha256(listing), "files": verified,
            "tokenizer_profile_revision": REVISION,
            "provenance_claim": "ModelScope weights at the recorded commit; tokenizer profile matches the pinned HF fixture. No HF weight identity is asserted."}


def verify_worker_model(worker, model_source):
    flags = worker["selected_arguments"]
    revision = model_source["source_commit"]
    require(worker["model_in_command"], "worker command must identify the pinned public model")
    require(flags.get("--revision") == [revision], "worker revision is not pinned to its recorded model source")
    require(flags.get("--tokenizer-revision") == [revision], "worker tokenizer revision is not pinned to its model source")
    if model_source["source"] == "ModelScope":
        require(worker["serve_model_path"] == model_source["local_dir"],
                "worker must load the verified ModelScope directory as its vllm serve positional model")


def output_directory(path, source):
    path = Path(path).resolve()
    require(not path.is_relative_to(Path(source).resolve()), "evidence must be outside source")
    path.mkdir(parents=True, exist_ok=False)
    return path


def build(args):
    """Build a clean candidate only when the caller is authorized to compile it."""
    out = output_directory(args.output, args.source)
    target = out / "target"
    argv = ["cargo", "build", "--locked", "--release", "--bin", "vllm-router",
            "--target-dir", str(target)]
    metadata = {"requested_candidate_sha": args.candidate, "command": argv,
                "architecture": platform.machine(),
                "started_at_unix": time.time(), "status": "RUNNING"}
    save(out / "build.json", metadata)
    try:
        identity = source_identity(args.source, args.candidate)
        metadata.update(identity, rustc=command(["rustc", "--version", "--verbose"]))
        save(out / "build.json", metadata)
        print("Building the clean candidate; full output is in build.log", flush=True)
        with (out / "build.log").open("w") as log:
            result = subprocess.run(argv, cwd=args.source, stdout=log, stderr=subprocess.STDOUT,
                                    check=False)
        metadata["returncode"] = result.returncode
        require(result.returncode == 0, f"native build failed with exit code {result.returncode}")
        require(source_identity(args.source, args.candidate) == identity,
                "source changed during build")
        native = target / "release" / "vllm-router"
        metadata.update(native=str(native), native_sha256=sha256(native), status="PASS")
    except (Exception, KeyboardInterrupt) as error:
        metadata.update(status="FAIL", error=f"{type(error).__name__}: {error}")
    metadata["finished_at_unix"] = time.time()
    save(out / "build.json", metadata)
    print(f"Build {metadata['status']}: {out / 'build.json'}")
    return 0 if metadata["status"] == "PASS" else 1


def engine_core_title(argv, environment):
    """vLLM 0.29 assigns this exact process title to a DP=1 EngineCore."""
    expected = environment.get("VLLM_PROCESS_NAME_PREFIX", "VLLM") + "::EngineCore"
    return expected if argv == [expected] else None


def serve_model_path(argv):
    """Capture only an explicit absolute `vllm serve` positional model path."""
    if argv.count("serve") != 1:
        return None
    position = argv.index("serve") + 1
    if position >= len(argv) or not Path(argv[position]).is_absolute():
        return None
    return str(Path(argv[position]).resolve())


def vllm_publisher_mode(endpoint):
    """Mirror vLLM 0.29 ZmqEventPublisher._socket_setup's bind heuristic."""
    require(isinstance(endpoint, str) and endpoint, "worker needs an explicit KV event endpoint")
    return ("bind" if "*" in endpoint or "::" in endpoint
            or endpoint.startswith("ipc://") or endpoint.startswith("inproc://") else "connect")


def verify_direct_publisher(endpoint):
    mode = vllm_publisher_mode(endpoint)
    require(not endpoint.startswith("tcp://") or mode == "bind",
            "vLLM 0.29 TCP publisher would connect, while this Router's subscriber also connects; "
            "the direct-worker fixture requires a bound publisher. Use tcp://*:PORT only on an "
            "explicitly authorized test host with the KV ports blocked from public access.")
    return mode


def process(pid):
    root = Path(f"/proc/{pid}")
    stat = (root / "stat").read_text().rsplit(")", 1)[1].split()
    argv_raw = (root / "cmdline").read_bytes()
    argv = [a.decode(errors="replace") for a in argv_raw.split(b"\0") if a]
    env_raw = (root / "environ").read_bytes().split(b"\0")
    allowed_env = {"PYTHONHASHSEED", "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES",
                   "CUDA_VISIBLE_DEVICES", "VLLM_SERVER_DEV_MODE", "VLLM_PROCESS_NAME_PREFIX"}
    env = {}
    for entry in env_raw:
        key, _, value = entry.partition(b"=")
        if key.decode(errors="replace") in allowed_env:
            env[key.decode()] = value.decode(errors="replace")
    # Never copy a process's complete environment or arbitrary command arguments.
    allowed_flags = {"--revision", "--tokenizer-revision", "--served-model-name", "--port",
                     "--data-parallel-size", "--tensor-parallel-size", "--pipeline-parallel-size",
                     "--prefix-caching-hash-algo", "--block-size", "--kv-events-config",
                     "--policy", "--kv-model", "--kv-block-size", "--kv-hash-algo",
                     "--kv-hash-seed", "--kv-events-port", "--kv-events-endpoint"}
    selected = {}
    for i, argument in enumerate(argv):
        flag, separator, value = argument.partition("=")
        if flag in allowed_flags:
            selected.setdefault(flag, []).append(value if separator else argv[i + 1])
    return {"pid": pid, "ppid": int(stat[1]), "start_ticks": int(stat[19]),
            "exe": os.readlink(root / "exe"), "exe_sha256": sha256(root / "exe"),
            "command_sha256": hashlib.sha256(argv_raw).hexdigest(),
            "selected_arguments": selected, "selected_environment": env,
            "engine_core_title": engine_core_title(argv, env),
            "serve_model_path": serve_model_path(argv),
            "model_in_command": MODEL in argv}


def descendant(pid, parent):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == parent:
            return True
        seen.add(pid)
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        pid = int(stat[1])
    return False


def request(base, path, payload=None, timeout=60):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read().decode()


def json_request(base, path, payload=None):
    status, _, body = request(base, path, payload)
    require(status == 200, f"{base}{path}: HTTP {status}: {body[:500]}")
    return json.loads(body)


def capture_worker_versions(workers, evidence_path):
    """Query each live API server; local package metadata is not worker proof."""
    observations = []
    for worker in workers:
        observation = {"worker": worker.rstrip("/"), "endpoint": "/version", "status": "FAIL"}
        try:
            status, _, body = request(worker, "/version")
            observation["http_status"] = status
            observation["response_body"] = body
            response = json.loads(body)
            observation["response"] = response
            require(status == 200, f"worker version endpoint returned HTTP {status}")
            require(isinstance(response, dict) and response.get("version") == VLLM_VERSION,
                    f"running worker must report exactly vLLM {VLLM_VERSION}")
            observation["status"] = "PASS"
        except Exception as error:
            observation["error"] = f"{type(error).__name__}: {error}"
        observations.append(observation)
        save(evidence_path, observations)
    require(len(observations) == 2 and all(item["status"] == "PASS" for item in observations),
            "both running workers must expose /version and report the pinned version; see worker_versions.json")
    return observations


def metrics(base):
    status, _, body = request(base, "/metrics")
    require(status == 200, f"metrics unavailable for {base}")
    result = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([^ ]+)(?:\s+.*)?", line)
        if match:
            name, labels, value = match.groups()
            try:
                result[(name, labels or "")] = float(value)
            except ValueError:
                pass
    return result


def count(values, name, label=None):
    matches = [value for (metric, labels), value in values.items()
               if metric == name and (label is None or label in labels)]
    require(matches, f"required metric {name} {label or ''} is absent")
    return sum(matches)


def parse_decisions(text):
    decisions = []
    decoder = json.JSONDecoder()
    for line in text.splitlines():
        line = ANSI.sub("", line)
        if "kv_route_decision" not in line:
            continue
        try:
            outer = json.loads(line)
            fields = outer.get("fields", outer)
            value = fields["decision"]
            decisions.append(json.loads(value) if isinstance(value, str) else value)
            continue
        except (ValueError, KeyError, TypeError):
            pass
        match = re.search(r"\bdecision=", line)
        if match:
            value, _ = decoder.raw_decode(line[match.end():].lstrip())
            decisions.append(json.loads(value) if isinstance(value, str) else value)
    return decisions


def first_nonterminal_sse(response):
    # Complete-record validation follows the downstream acceptance helper's
    # cf37596e38b5f994d83b65c0b112cd91b8614e28 fix, without its P/D framework.
    consumed, data_lines = 0, []
    while consumed <= 65536:
        line = response.readline(65536 - consumed + 1)
        require(line, "stream ended before a complete nonterminal SSE record")
        consumed += len(line)
        require(consumed <= 65536, "first SSE record exceeded the byte bound")
        require(not (line.startswith(b"event:") and line[6:].strip() == b"error"),
                "cancel stream returned an SSE error event")
        if line.startswith(b"data:"):
            payload = line[5:].strip()
            require(payload != b"[DONE]", "stream completed before cancellation")
            if payload:
                data_lines.append(payload)
        if line in (b"\n", b"\r\n") and data_lines:
            event = json.loads(b"\n".join(data_lines))
            require(isinstance(event, dict) and not event.get("error"), "invalid SSE response object")
            choices = event.get("choices")
            require(isinstance(choices, list) and choices
                    and all(isinstance(choice, dict) and "finish_reason" in choice
                            and choice["finish_reason"] is None for choice in choices),
                    "first complete SSE record was not explicitly nonterminal")
            require(isinstance(event.get("id"), str) and event["id"], "SSE response ID is missing")
            return event, consumed
    raise RuntimeError("no complete nonterminal SSE record")


def matching_abort_lines(text, response_id, worker_pid):
    # One Completion prompt/n=1 yields response.id + '-0'. Stock vLLM may add
    # exactly eight hex characters for its internal ID; never substring-match.
    request_id = re.compile(re.escape(response_id) + r"-0(?:-[0-9a-f]{8})?")
    matched = []
    for line in text.splitlines():
        clean = ANSI.sub("", line)
        process = re.search(r"\(APIServer pid=(\d+)\)", clean)
        if not process or int(process[1]) != worker_pid:
            continue
        record = re.search(r"\bAborted request\(s\) ([^\r\n]+)\.\s*$", clean)
        if record and any(request_id.fullmatch(value.strip()) for value in record[1].split(",")):
            matched.append(clean)
    return matched


def verify_cancel_active(before, active):
    running, loads = active["backend_running"], active["router_loads"]
    require(running in ([1, 0], [0, 1]) and loads == running and all(active["healthy"]),
            "one matching Router and worker active request was not observed before cancellation")
    require(active["completed"] == before["completed"], "request completed before cancellation")
    return running.index(1)


class Validation:
    def __init__(self, args, out):
        self.args, self.out = args, out
        self.workers = [args.worker0.rstrip("/"), args.worker1.rstrip("/")]
        require(len(set(self.workers)) == 2, "worker URLs must be distinct")
        self.results = []

    def idle(self):
        for _ in range(120):
            backend_idle = all(count(metrics(w), "vllm:num_requests_running") == 0
                               for w in self.workers)
            registered = json_request(self.args.router, "/workers")["workers"]
            own = [w for w in registered if w["url"].rstrip("/") in self.workers]
            require(len(own) == 2, "Router must register both independent workers")
            if backend_idle and all(w["load"] == 0 and w["is_healthy"] for w in own):
                return
            time.sleep(0.25)
        raise RuntimeError("backend running requests or Router load did not return to zero")

    def tokens(self, payload):
        if isinstance(payload.get("prompt"), list):
            return payload["prompt"]
        fields = ("model", "prompt", "messages", "add_special_tokens",
                  "chat_template_kwargs", "add_generation_prompt")
        data = {key: payload[key] for key in fields if key in payload}
        tokens = [json_request(worker, "/tokenize", data)["tokens"] for worker in self.workers]
        require(tokens[0] == tokens[1], "the two workers disagree on exact prompt tokens")
        require(tokens[0] and all(type(t) is int and 0 <= t < 2**32 for t in tokens[0]),
                "worker tokenizer returned invalid token IDs")
        return tokens[0]

    def counters(self):
        return [count(metrics(w), self.args.request_counter) for w in self.workers]

    def observed_backend(self, before):
        for _ in range(120):
            after = self.counters()
            delta = [a - b for a, b in zip(after, before)]
            if sum(delta) >= 1:
                require(delta in ([1, 0], [0, 1]),
                        f"ambiguous backend counts {delta}; dedicate both workers to this run")
                return delta.index(1), delta
            time.sleep(0.25)
        raise RuntimeError("no completed request appeared in backend metrics")

    def log_tail(self, offset):
        with Path(self.args.router_log).open("rb") as log:
            log.seek(offset)
            return log.read().decode(errors="replace")

    def send(self, name, payload, expected=None, stream=False):
        self.idle()
        path = "/v1/chat/completions" if "messages" in payload else "/v1/completions"
        tokens = self.tokens(payload)
        require(len(tokens) >= 32, "test prompt must span at least two full KV blocks")
        before = self.counters()
        offset = Path(self.args.router_log).stat().st_size
        status, headers, body = request(self.args.router, path, {**payload, "stream": stream})
        (self.out / f"{name}.router.log").write_text(self.log_tail(offset))
        save(self.out / f"{name}.response.json", {
            "http_status": status, "request": payload, "stream": stream,
            "response_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "error_body": body if status != 200 else None,
        })
        require(status == 200, f"Router returned HTTP {status}: {body[:500]}")
        if stream:
            require("text/event-stream" in headers.get("content-type", headers.get("Content-Type", "")),
                    "stream response is not SSE")
            require("data: [DONE]" in body and '"choices"' in body, "SSE completion is incomplete")
        else:
            require(json.loads(body).get("choices"), "JSON response has no choices")
        actual, delta = self.observed_backend(before)
        self.idle()
        tail = self.log_tail(offset)
        (self.out / f"{name}.router.log").write_text(tail)
        decisions = parse_decisions(tail)
        require(len(decisions) == 1, f"expected exactly one routing decision, got {len(decisions)}")
        decision = decisions[0]
        require(decision["worker"].rstrip("/") == self.workers[actual],
                "decision and independent backend counter disagree")
        require(decision["token_ids_sha256"] == token_digest(tokens),
                "Router exact token IDs differ from worker /tokenize")
        scores = {entry["worker"].rstrip("/"): entry["prefix_blocks"]
                  for entry in decision["scores"]}
        require(set(scores) == set(self.workers), "decision must report both worker scores")
        if expected is not None:
            require(actual == expected, f"expected W{expected}, observed W{actual}")
            require(scores[self.workers[expected]] > 0, "target has no real KV-event positive score")
            require(scores[self.workers[1 - expected]] == 0, "other worker has a positive prefix score")
        else:
            require(all(score == 0 for score in scores.values()), "cold/cleared prefix still has ownership")
        evidence = {"name": name, "status": "PASS", "actual_backend": actual,
                    "completed_request_deltas": delta, "decision": decision,
                    "worker_token_ids_sha256": token_digest(tokens), "prompt_tokens": len(tokens),
                    "request": payload, "response_sha256": hashlib.sha256(body.encode()).hexdigest(),
                    "stream": stream}
        save(self.out / f"{name}.json", evidence)
        return evidence

    def case(self, name, callback):
        try:
            evidence = callback()
            self.results.append(evidence or {"name": name, "status": "PASS"})
            print(f"PASS {name}", flush=True)
        except Exception as error:
            self.results.append({"name": name, "status": "FAIL", "error": str(error)})
            print(f"FAIL {name}: {error}", flush=True)
        save(self.out / "cases.json", self.results)

    def positive(self, name, kind, target, stream=False):
        nonce = uuid.uuid4().hex
        text = (f"{nonce} synthetic public cache-routing case {name}. " * 16
                + "Return a short response.")
        payload = {"model": MODEL, "max_tokens": 8, "temperature": 0}
        if kind == "chat":
            payload["messages"] = [{"role": "user", "content": text}]
        else:
            payload.update(prompt=text, add_special_tokens=True)
            if kind == "ids":
                payload["prompt"] = self.tokens(payload)
        path = "/v1/chat/completions" if kind == "chat" else "/v1/completions"
        self.idle()
        before = self.counters()
        json_request(self.workers[target], path, payload)
        actual, delta = self.observed_backend(before)
        require(actual == target, "direct warming reached the wrong worker")
        # Time for the asynchronous real ZMQ event; never warm through the Router.
        time.sleep(self.args.event_wait)
        result = self.send(name, payload, target, stream)
        result["direct_warm_completed_request_deltas"] = delta
        save(self.out / f"{name}.json", result)
        return result

    def clear(self):
        name = "real_all_blocks_cleared"
        nonce = uuid.uuid4().hex
        payload = {"model": MODEL, "prompt": (nonce + " lifecycle public fixture ") * 16,
                   "max_tokens": 4, "temperature": 0}
        json_request(self.workers[0], "/v1/completions", payload)
        time.sleep(self.args.event_wait)
        self.send(name + "_before", payload, 0)
        self.idle()
        reset = json_request(self.workers[0], "/reset_prefix_cache", {})
        require(reset.get("success") is True, "worker did not confirm prefix-cache reset")
        # vLLM 0.29 queues AllBlocksCleared at reset, but an idle engine publishes
        # it only on a subsequent scheduler step. A fresh direct-only marker
        # drives that step without restoring ownership of the cleared prefix.
        marker = {"model": MODEL, "prompt": (uuid.uuid4().hex + " clear event flush ") * 4,
                  "max_tokens": 1, "temperature": 0}
        marker_tokens = self.tokens(marker)
        require(marker_tokens[:16] != self.tokens(payload)[:16],
                "event-flush marker unexpectedly shares the cleared prefix")
        before = self.counters()
        json_request(self.workers[0], "/v1/completions", marker)
        actual, delta = self.observed_backend(before)
        require(actual == 0, "cache-clear marker did not reach only W0")
        time.sleep(self.args.event_wait)
        result = self.send(name + "_after", payload)
        result["reset_response"] = reset
        result["direct_event_flush_marker"] = {
            "request": marker, "token_ids_sha256": token_digest(marker_tokens),
            "completed_request_deltas": delta,
        }
        save(self.out / f"{name}_after.json", result)
        return result

    def error_cleanup(self):
        self.idle()
        payload = {"model": MODEL, "prompt": uuid.uuid4().hex + " error cleanup " * 40,
                   "max_tokens": 2147483647}
        offset = Path(self.args.router_log).stat().st_size
        status, _, body = request(self.args.router, "/v1/completions", payload)
        require(400 <= status < 500, f"expected backend rejection, got HTTP {status}")
        self.idle()
        tail = self.log_tail(offset)
        require(parse_decisions(tail), "error case never dispatched to a backend")
        (self.out / "backend_error_cleanup.router.log").write_text(tail)
        return {"name": "backend_error_cleanup", "status": "PASS", "http_status": status,
                "response": body, "router_and_backend_load_after": 0}

    def cancel_snapshot(self):
        values = [metrics(worker) for worker in self.workers]
        registered = json_request(self.args.router, "/workers")["workers"]
        own = {worker["url"].rstrip("/"): worker for worker in registered
               if worker["url"].rstrip("/") in self.workers}
        require(set(own) == set(self.workers), "cancel probe requires both registered workers")
        return {"observed_at_unix": time.time(),
                "backend_running": [count(value, "vllm:num_requests_running") for value in values],
                "router_loads": [own[worker]["load"] for worker in self.workers],
                "healthy": [own[worker]["is_healthy"] for worker in self.workers],
                "completed": [count(value, self.args.request_counter) for value in values],
                "aborts": [sum(value for (name, labels), value in worker.items()
                               if name == self.args.request_counter and 'finished_reason="abort"' in labels)
                           for worker in values]}

    def cancel_cleanup(self):
        name = "stream_cancel_cleanup"
        evidence = {"name": name, "status": "RUNNING", "started_at_unix": time.time()}
        try:
            self.idle()
            offset = Path(self.args.router_log).stat().st_size
            paths = [self.args.worker0_log, self.args.worker1_log]
            require(bool(paths[0]) == bool(paths[1]), "provide both worker logs or neither")
            log_offsets = []
            for path, pid in zip(paths, (self.args.worker0_pid, self.args.worker1_pid)):
                if path:
                    stat = Path(path).stat()
                    output_fds = [Path(f"/proc/{pid}/fd/{fd}").stat() for fd in (1, 2)]
                    require(any((value.st_dev, value.st_ino) == (stat.st_dev, stat.st_ino)
                                for value in output_fds), "worker log is not that HTTP process's stdout/stderr")
                    log_offsets.append({"path": path, "offset": stat.st_size,
                                        "device": stat.st_dev, "inode": stat.st_ino, "worker_pid": pid})
            evidence["worker_log_offsets"] = log_offsets
            before = self.cancel_snapshot()
            evidence["before"] = before
            parsed = urllib.parse.urlsplit(self.args.router)
            cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
            connection = cls(parsed.hostname, parsed.port, timeout=60)
            response = None
            payload = {"model": MODEL, "prompt": uuid.uuid4().hex + " cancellation fixture " * 40,
                       "max_tokens": 1024, "ignore_eos": True, "stream": True}
            evidence["request"] = payload
            try:
                evidence["dispatch_at_unix"] = time.time()
                connection.request("POST", parsed.path.rstrip("/") + "/v1/completions",
                                   json.dumps(payload), {"Content-Type": "application/json"})
                raw_socket = connection.sock
                response = connection.getresponse()
                evidence.update(headers_at_unix=time.time(), http_status=response.status,
                                response_headers=dict(response.getheaders()))
                require(response.status == 200, f"cancel stream HTTP {response.status}")
                require("text/event-stream" in response.getheader("Content-Type", ""), "cancel response is not SSE")
                event, consumed = first_nonterminal_sse(response)
                evidence.update(first_nonterminal_event=event, bytes_read_before_close=consumed,
                                first_event_at_unix=time.time())
                active = self.cancel_snapshot()
                evidence["active_before_close"] = active
                target = verify_cancel_active(before, active)
                evidence["worker"] = self.workers[target]
                require(raw_socket is not None, "cannot identify client socket for explicit shutdown")
                raw_socket.shutdown(socket.SHUT_RDWR)
                evidence["shutdown_at_unix"] = time.time()
            finally:
                if response is not None:
                    response.close()
                connection.close()
                evidence["closed_at_unix"] = time.time()
            self.idle()
            evidence["idle_at_unix"] = time.time()
            expected_delta = [int(index == target) for index in range(2)]
            proof = False
            for _ in range(120):
                after = self.cancel_snapshot()
                evidence["after"] = after
                delta = [a - b for a, b in zip(after["aborts"], before["aborts"])]
                completed_delta = [a - b for a, b in zip(after["completed"], before["completed"])]
                natural_delta = [a - b for a, b in zip(completed_delta, delta)]
                evidence.update(abort_deltas=delta, completed_request_deltas=completed_delta,
                                natural_completion_deltas=natural_delta)
                require(natural_delta == [0, 0], f"cancel request completed naturally: {natural_delta}")
                require(delta in ([0, 0], expected_delta), "unexpected abort count on the two exclusive workers")
                matches = [[], []]
                for index, log in enumerate(log_offsets):
                    stat = Path(log["path"]).stat()
                    require((stat.st_dev, stat.st_ino) == (log["device"], log["inode"])
                            and stat.st_size >= log["offset"], "worker log rotated during cancellation probe")
                    with Path(log["path"]).open("rb") as stream:
                        stream.seek(log["offset"])
                        tail = stream.read().decode(errors="replace")
                    (self.out / f"{name}.worker{index}.log").write_text(tail)
                    matches[index] = matching_abort_lines(tail, event["id"], log["worker_pid"])
                evidence["matching_abort_log_lines"] = matches
                if log_offsets:
                    proof = bool(matches[target]) and not matches[1 - target]
                    evidence["abort_evidence"] = "request-ID-correlated vLLM abort log"
                else:
                    proof = delta == expected_delta
                    evidence["abort_evidence"] = "strict abort counter delta (no worker logs provided)"
                if proof:
                    break
                time.sleep(0.25)
            tail = self.log_tail(offset)
            (self.out / f"{name}.router.log").write_text(tail)
            decisions = parse_decisions(tail)
            require(len(decisions) == 1 and decisions[0]["worker"].rstrip("/") == self.workers[target],
                    "cancel routing decision disagrees with observed active worker")
            require(proof, "backend abort was not proven by the required request log or counter")
            require(after["backend_running"] == [0, 0] and after["router_loads"] == [0, 0]
                    and all(after["healthy"]), "cancel probe did not finish healthy and idle")
            evidence.update(status="PASS", router_and_backend_load_after=0)
            return evidence
        except Exception as error:
            evidence.update(status="FAIL", error=str(error))
            raise
        finally:
            evidence["finished_at_unix"] = time.time()
            save(self.out / f"{name}.json", evidence)


def validate(args):
    out = output_directory(args.output, args.source)
    report = {"status": "RUNNING", "started_at_unix": time.time(), "command": sys.argv,
              "model": MODEL, "tokenizer_profile_revision": REVISION}
    save(out / "summary.json", report)
    try:
        require(platform.system() == "Linux", "run in the GPU processes' Linux /proc namespace")
        identity = source_identity(args.source, args.candidate)
        model_source = (verify_model_manifest(args.model_manifest) if args.model_manifest else
                        {"source": "Hugging Face", "model": MODEL, "source_commit": REVISION,
                         "tokenizer_profile_revision": REVISION})
        report.update(model_source=model_source, model_revision=model_source["source_commit"])
        save(out / "model_source.json", model_source)
        if args.model_manifest:
            require(Path(args.tokenizer).resolve() == Path(model_source["local_dir"]) / "tokenizer.json",
                    "Router tokenizer evidence must come from the verified local model snapshot")
        manifest = json.loads(Path(args.build_manifest).read_text())
        native_sha = sha256(args.native)
        require(manifest["status"] == "PASS" and manifest["candidate_sha"] == args.candidate
                and manifest["native_sha256"] == native_sha,
                "build manifest does not bind this candidate to this native executable")
        processes = [process(pid) for pid in [args.router_pid, args.worker0_pid, args.worker1_pid,
                                              args.engine0_pid, args.engine1_pid]]
        require(len({p["pid"] for p in processes}) == 5, "router, HTTP and EngineCore PIDs must be distinct")
        require(processes[0]["exe_sha256"] == native_sha, "live Router is not the candidate native executable")
        require(all(p["engine_core_title"] for p in processes[3:]),
                "engine PIDs must identify vLLM DP=1 EngineCore processes, not arbitrary children")
        require(descendant(args.engine0_pid, args.worker0_pid)
                and descendant(args.engine1_pid, args.worker1_pid), "EngineCore ancestry is not independent")
        event_endpoints = []
        for worker_index, p in enumerate(processes[1:3]):
            env, flags = p["selected_environment"], p["selected_arguments"]
            verify_worker_model(p, model_source)
            require(env.get("PYTHONHASHSEED") == "0", "worker PYTHONHASHSEED must be 0")
            require(env.get("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES") == "0", "full-byte event hashes required")
            require(flags.get("--block-size") == ["16"], "worker block size must be 16")
            require(flags.get("--prefix-caching-hash-algo") == ["sha256_cbor"], "worker hash algorithm mismatch")
            worker_url = urllib.parse.urlsplit([args.worker0, args.worker1][worker_index])
            require(flags.get("--port") == [str(worker_url.port)], "HTTP PID does not match worker URL port")
            events = json.loads(flags.get("--kv-events-config", ["{}"])[0])
            require(events.get("enable_kv_cache_events") is True and events.get("publisher") == "zmq",
                    "worker must publish real ZMQ KV events")
            verify_direct_publisher(events.get("endpoint"))
            event_endpoints.append(events.get("endpoint"))
            if args.allow_cache_reset:
                require(env.get("VLLM_SERVER_DEV_MODE") == "1", "cache-clear test needs isolated dev endpoint")
            for flag in ("--data-parallel-size", "--tensor-parallel-size", "--pipeline-parallel-size"):
                require(flags.get(flag, ["1"]) == ["1"], f"{flag} must be 1")
        require(all(event_endpoints) and len(set(event_endpoints)) == 2,
                "workers must have distinct explicitly configured event endpoints")
        gpu_selection = [p["selected_environment"].get("CUDA_VISIBLE_DEVICES") for p in processes[1:3]]
        require(gpu_selection[0] and gpu_selection[0] == gpu_selection[1]
                and "," not in gpu_selection[0], "this matrix requires the same single selected GPU")
        require(sha256(args.tokenizer) == TOKENIZER_SHA256, "tokenizer does not match pinned public revision")
        worker_versions = capture_worker_versions([args.worker0, args.worker1], out / "worker_versions.json")
        environment = {"platform": platform.platform(), "python": sys.version,
                       "harness_python_executable": sys.executable,
                       "harness_vllm_distribution_version": importlib.metadata.version("vllm"),
                       "running_worker_version_responses": worker_versions,
                       "gpu": command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                                       "--format=csv,noheader"]),
                       "processes": processes, "build": manifest,
                       "tokenizer_sha256": sha256(args.tokenizer)}
        environment["harness_torch_cuda_runtime"] = command([
            sys.executable, "-c",
            "import json, torch; print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda}))",
        ])
        require(environment["harness_vllm_distribution_version"] == VLLM_VERSION,
                f"run the harness in the pinned vLLM {VLLM_VERSION} environment")
        save(out / "environment.json", environment)
        report.update(identity, native_sha256=native_sha)
        suite = Validation(args, out)
        suite.idle()
        for kind in ("text", "ids", "chat"):
            for target in (0, 1):
                name = f"{kind}_first_route_w{target}"
                suite.case(name, lambda n=name, k=kind, t=target:
                           suite.positive(n, k, t, stream=t == 1))
        cold = {"model": MODEL, "prompt": (uuid.uuid4().hex + " cold fixture ") * 16,
                "max_tokens": 4}
        suite.case("cold_miss", lambda: suite.send("cold_miss", cold))
        if args.allow_cache_reset:
            suite.case("real_all_blocks_cleared", suite.clear)
        else:
            suite.results.append({"name": "real_all_blocks_cleared", "status": "NOT RUN",
                                  "reason": "requires --allow-cache-reset for the two owned workers"})
        suite.case("backend_error_cleanup", suite.error_cleanup)
        suite.case("stream_cancel_cleanup", suite.cancel_cleanup)
        after = [process(p["pid"]) for p in processes]
        process_keys = ("pid", "start_ticks", "exe_sha256", "command_sha256", "engine_core_title")
        require([tuple(p[key] for key in process_keys) for p in processes]
                == [tuple(p[key] for key in process_keys) for p in after],
                "a tested process identity or executable changed during validation")
        require(source_identity(args.source, args.candidate) == identity, "candidate changed during validation")
        if args.model_manifest:
            require(verify_model_manifest(args.model_manifest) == model_source,
                    "model snapshot or download manifest changed during validation")
        report["cases"] = suite.results
        report["status"] = ("FAIL" if any(r["status"] == "FAIL" for r in suite.results)
                            else "INCOMPLETE" if any(r["status"] != "PASS" for r in suite.results)
                            else "PASS")
        report["scope"] = "single GPU host, two independent DP=1 workers; no performance claim"
        report["not_covered"] = ["worker restart/new generation (CPU deterministic coverage required)",
                                 "late event and sequence gap (CPU deterministic coverage required)",
                                 "multi-GPU or multi-host behavior"]
    except Exception as error:
        report.update(status="FAIL", error=str(error))
    report["finished_at_unix"] = time.time()
    save(out / "summary.json", report)
    print(f"{report['status']}: {out / 'summary.json'}")
    return 0 if report["status"] == "PASS" else 1


def self_check():
    """Offline checks for the evidence parser; no server or GPU is contacted."""
    import contextlib
    import io
    import tempfile
    from unittest import mock

    decision = {"worker": "http://127.0.0.1:8000", "prefix_blocks": 2,
                "scores": [{"worker": "http://127.0.0.1:8000", "prefix_blocks": 2}]}
    encoded = json.dumps(decision, separators=(",", ":"))
    inputs = [
        "DEBUG kv_route_decision decision=" + encoded,
        "\x1b[32mDEBUG\x1b[0m kv_route_decision decision=" + encoded + " other=1",
        json.dumps({"message": "kv_route_decision", "decision": encoded}),
        json.dumps({"fields": {"message": "kv_route_decision", "decision": encoded}}),
        "DEBUG kv_route_decision decision=" + json.dumps(encoded),
    ]
    for text in inputs:
        require(parse_decisions(text) == [decision], "routing log parser self-check failed")
    require(parse_decisions("unrelated log") == [], "unrelated log accepted")
    publisher_cases = {
        "tcp://*:5557": "bind", "tcp://[::1]:5557": "bind",
        "tcp://127.0.0.1:*": "bind",
        "ipc:///tmp/kv-events": "bind", "inproc://kv-events": "bind",
        "tcp://127.0.0.1:5557": "connect", "tcp://localhost:5558": "connect",
        "tcp://0.0.0.0:5557": "connect", "tcp://worker:5557": "connect",
    }
    for endpoint, mode in publisher_cases.items():
        require(vllm_publisher_mode(endpoint) == mode, "vLLM publisher bind heuristic mismatch")
        try:
            verify_direct_publisher(endpoint)
            accepted = True
        except RuntimeError:
            accepted = False
        require(accepted == (not endpoint.startswith("tcp://") or mode == "bind"),
                "connect-only TCP publisher was accepted by the direct-worker fixture")
    require(token_digest([0x01020304]) == hashlib.sha256(b"\x01\x02\x03\x04").hexdigest(),
            "token digest byte order is wrong")
    try:
        count({}, "required_missing_counter")
    except RuntimeError:
        pass
    else:
        raise RuntimeError("missing metrics must fail closed")
    require(engine_core_title(["VLLM::EngineCore"], {}) == "VLLM::EngineCore",
            "DP=1 EngineCore title not recognized")
    require(engine_core_title(["fixture::EngineCore"], {"VLLM_PROCESS_NAME_PREFIX": "fixture"}),
            "configured vLLM process prefix not recognized")
    for argv in (["python", "-c", "multiprocessing.resource_tracker"],
                 ["VLLM::Worker_TP0"], ["VLLM::EngineCore_DP1"], []):
        require(engine_core_title(argv, {}) is None, "non-DP=1-EngineCore process accepted")
    # Mock every external command: exercise manifest failure handling without
    # running Cargo, rustc, git, or a GPU process.
    identity = {"candidate_sha": "a" * 40, "base_sha": BASE, "tree_sha": "b" * 40}
    scenarios = [
        ("source-change", [identity, RuntimeError("dirty source after build")], 0),
        ("missing-native", [identity, identity], 0),
        ("nonzero-build", [identity], 1),
    ]
    with tempfile.TemporaryDirectory(prefix="kv-evidence-self-check-") as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        model_dir = root / "model"
        model_dir.mkdir()
        entries = []
        for name in sorted(MODELSCOPE_FILES):
            target = model_dir / name
            target.write_text(f"offline fixture {name}\n")
            entries.append({"path": name, "status": "verified", "source_commit": MODELSCOPE_REVISION,
                            "bytes": target.stat().st_size, "sha256": sha256(target), "api_sha256": sha256(target)})
        listing = model_dir / "modelscope-api-files.json"
        listing.write_text("offline source listing\n")
        model_manifest = {"status": "verified", "source": "ModelScope", "model": MODEL,
                          "source_commit": MODELSCOPE_REVISION, "output": str(model_dir),
                          "required_files": sorted(MODELSCOPE_FILES), "files": entries,
                          "source_listing_sha256": sha256(listing)}
        model_manifest_path = model_dir / "download-manifest.json"
        save(model_manifest_path, model_manifest)
        fixture_hashes = {entry["path"]: entry["sha256"] for entry in entries
                          if entry["path"] in MODELSCOPE_REQUIRED_SHA256}

        def expect_rejection(check, message):
            try:
                check()
            except RuntimeError:
                return
            raise RuntimeError(message)

        stream_event = {"id": "cmpl-fixture", "choices": [{"finish_reason": None, "text": "x"}]}
        stream_bytes = b"data: " + json.dumps(stream_event).encode() + b"\r\n\r\n"
        require(first_nonterminal_sse(io.BytesIO(stream_bytes))[0] == stream_event,
                "complete nonterminal SSE frame was rejected")
        for invalid_stream in (
            b'data: {"id":"cmpl-fixture","choices":[{"finish_reason":"length"}]}\n\n',
            b'data: {"id":"cmpl-fixture","choices":[{}]}\n\n',
            b"data: [DONE]\n\n", stream_bytes.rstrip(),
            stream_bytes.rstrip() + b"\nevent: error\n\n",
        ):
            expect_rejection(lambda: first_nonterminal_sse(io.BytesIO(invalid_stream)),
                             "completed/malformed SSE frame was accepted as active")
        abort_line = "(APIServer pid=123) INFO [async_llm.py:836] Aborted request(s) cmpl-fixture-0-97467303."
        require(matching_abort_lines(abort_line, "cmpl-fixture", 123) == [abort_line],
                "exact vLLM internal abort ID was not recognized")
        for wrong_line in (abort_line.replace("cmpl-fixture-0", "cmpl-fixture-extra-0"),
                           abort_line.replace("-0-97467303", "-1-97467303"),
                           abort_line.replace("97467303", "974673031"),
                           abort_line.replace("pid=123", "pid=124")):
            require(not matching_abort_lines(wrong_line, "cmpl-fixture", 123),
                    "wrong request/worker abort log was accepted")
        before_cancel = {"completed": [4, 3]}
        active_cancel = {"completed": [4, 3], "backend_running": [1, 0],
                         "router_loads": [1, 0], "healthy": [True, True]}
        require(verify_cancel_active(before_cancel, active_cancel) == 0, "active worker check failed")
        for inactive in (dict(active_cancel, backend_running=[0, 0]),
                         dict(active_cancel, router_loads=[0, 0]),
                         dict(active_cancel, completed=[5, 3])):
            expect_rejection(lambda: verify_cancel_active(before_cancel, inactive),
                             "inactive/already-completed cancellation request was accepted")

        with mock.patch.dict(globals(), {"MODELSCOPE_REQUIRED_SHA256": fixture_hashes}):
            verified_model = verify_model_manifest(model_manifest_path)
            worker = {"model_in_command": True, "serve_model_path": str(model_dir),
                      "selected_arguments": {"--revision": [MODELSCOPE_REVISION],
                                             "--tokenizer-revision": [MODELSCOPE_REVISION]}}
            verify_worker_model(worker, verified_model)
            require(serve_model_path(["vllm", "serve", str(model_dir)]) == str(model_dir),
                    "explicit local model path was not recognized")
            require(serve_model_path(["vllm", "serve", MODEL]) is None,
                    "repository name was accepted as a verified local snapshot")
            for incorrect_path in (None, str(root / "other-model")):
                expect_rejection(lambda: verify_worker_model(dict(worker, serve_model_path=incorrect_path), verified_model),
                                 "wrong worker model path was accepted")
            original = (model_dir / "model.safetensors").read_bytes()
            (model_dir / "model.safetensors").write_bytes(original + b"tampered")
            expect_rejection(lambda: verify_model_manifest(model_manifest_path), "tampered weights were accepted")
            (model_dir / "model.safetensors").write_bytes(original)
            save(model_manifest_path, dict(model_manifest, source_commit="0" * 40))
            expect_rejection(lambda: verify_model_manifest(model_manifest_path), "wrong model revision was accepted")
            save(model_manifest_path, model_manifest)
            (model_dir / "unexpected-model-file").write_text("not in manifest")
            expect_rejection(lambda: verify_model_manifest(model_manifest_path), "unmanifested model file was accepted")
        hf_worker = {"model_in_command": True, "selected_arguments": {
            "--revision": [REVISION], "--tokenizer-revision": [REVISION]}}
        verify_worker_model(hf_worker, {"source": "Hugging Face", "source_commit": REVISION})
        for name, source_results, returncode in scenarios:
            args = argparse.Namespace(source=str(source), candidate="a" * 40,
                                      output=str(root / name))
            overrides = {"source_identity": mock.Mock(side_effect=source_results),
                         "command": mock.Mock(return_value="mock rustc (not executed)")}
            with mock.patch.dict(globals(), overrides), \
                    mock.patch.object(subprocess, "run", return_value=argparse.Namespace(returncode=returncode)), \
                    contextlib.redirect_stdout(io.StringIO()):
                require(build(args) == 1, "failed build was reported successful")
            manifest = json.loads((root / name / "build.json").read_text())
            require(manifest["status"] == "FAIL" and manifest.get("error")
                    and manifest.get("finished_at_unix"), "failed build left incomplete evidence")
        workers = ["http://worker0", "http://worker1"]
        good_version = (200, {}, json.dumps({"version": VLLM_VERSION}))
        version_scenarios = [
            ("both-pinned", [good_version, good_version], True),
            ("wrong-worker-version", [good_version, (200, {}, '{"version":"0.28.0"}')], False),
            ("missing-version-endpoint", [good_version, (404, {}, '{"detail":"Not Found"}')], False),
            ("unreachable-worker", [good_version, urllib.error.URLError("unreachable")], False),
        ]
        for name, responses, should_pass in version_scenarios:
            path = root / f"{name}.json"
            with mock.patch.dict(globals(), {"request": mock.Mock(side_effect=responses)}):
                try:
                    capture_worker_versions(workers, path)
                    passed = True
                except RuntimeError:
                    passed = False
            require(passed == should_pass, "worker version evidence accepted an unsupported runtime")
            captured = json.loads(path.read_text())
            require(len(captured) == 2 and captured[1]["status"] == ("PASS" if should_pass else "FAIL"),
                    "worker version responses were not retained")
    print("PASS offline log/token/metric/process/publisher/version/model-manifest/cancellation/failed-build evidence checks (no Cargo or CUDA run)")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("self-check", help="offline evidence parser checks; no GPU required")
    for action in ("build", "validate"):
        sub = commands.add_parser(action)
        sub.add_argument("--source", required=True)
        sub.add_argument("--candidate", required=True)
        sub.add_argument("--output", required=True, help="new directory outside source")
        if action == "validate":
            sub.add_argument("--native", required=True)
            sub.add_argument("--build-manifest", required=True)
            sub.add_argument("--tokenizer", required=True, help="pinned tokenizer.json")
            sub.add_argument("--model-manifest", help="verified immutable ModelScope download-manifest.json; omitted means the pinned HF model revision")
            sub.add_argument("--router", default="http://127.0.0.1:3001")
            sub.add_argument("--worker0", default="http://127.0.0.1:8000")
            sub.add_argument("--worker1", default="http://127.0.0.1:8001")
            sub.add_argument("--router-log", required=True)
            sub.add_argument("--worker0-log", help="HTTP W0 stdout/stderr log with --enable-log-requests; provide both worker logs")
            sub.add_argument("--worker1-log", help="HTTP W1 stdout/stderr log with --enable-log-requests; provide both worker logs")
            for name in ("router", "worker0", "worker1", "engine0", "engine1"):
                sub.add_argument(f"--{name}-pid", required=True, type=int)
            sub.add_argument("--event-wait", type=float, default=2.0)
            sub.add_argument("--request-counter", default="vllm:request_success_total")
            sub.add_argument("--allow-cache-reset", action="store_true")
    args = parser.parse_args()
    if args.action == "self-check":
        return self_check()
    return build(args) if args.action == "build" else validate(args)


if __name__ == "__main__":
    sys.exit(main())
