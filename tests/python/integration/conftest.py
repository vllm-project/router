import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pytest
import requests

from tests.python.helpers.ports import find_free_port
from tests.python.helpers.router_manager import RouterManager


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark as router integration test")


@pytest.fixture
def router_manager() -> Iterable[RouterManager]:
    mgr = RouterManager()
    try:
        yield mgr
    finally:
        mgr.stop_all()


def _spawn_mock_worker(args: List[str]) -> Tuple[subprocess.Popen, str, str]:
    script = Path(__file__).resolve().parents[1] / "helpers" / "mock_worker.py"
    port = find_free_port()
    worker_id = f"worker-{port}"
    base_cmd = [
        sys.executable,
        str(script),
        "--port",
        str(port),
        "--worker-id",
        worker_id,
    ]
    cmd = base_cmd + args
    proc = subprocess.Popen(cmd)
    url = f"http://127.0.0.1:{port}"
    _wait_health(url)
    return proc, url, worker_id


def _wait_health(url: str, timeout: float = 10.0):
    start = time.time()
    with requests.Session() as s:
        while time.time() - start < timeout:
            try:
                r = s.get(f"{url}/health", timeout=1)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.1)
    raise TimeoutError(f"Mock worker at {url} did not become healthy")


@pytest.fixture
def mock_worker():
    """Start a single healthy mock worker; yields (process, url, worker_id)."""
    proc, url, worker_id = _spawn_mock_worker([])
    try:
        yield proc, url, worker_id
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


@pytest.fixture
def mock_workers():
    """Factory to start N workers with custom args.

    Usage:
        procs, urls, ids = mock_workers(n=3, args=["--latency-ms", "5"])  # same args for all
        ...
    """

    procs: List[subprocess.Popen] = []

    def _start(n: int, args: Optional[List[str]] = None):
        args = args or []
        new_procs: List[subprocess.Popen] = []
        urls: List[str] = []
        ids: List[str] = []
        for _ in range(n):
            p, url, wid = _spawn_mock_worker(args)
            procs.append(p)
            new_procs.append(p)
            urls.append(url)
            ids.append(wid)
        return new_procs, urls, ids

    try:
        yield _start
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()


@pytest.fixture(scope="session")
def kv_st_runtime(tmp_path_factory):
    """Generate Python wire bindings from the exact crate used by the binary.

    Requires `cargo build --bin vllm-router` and `pip install -e '.[dev]'`.
    Missing prerequisites fail explicitly, including in CI; no silent skips or
    network model downloads. Cargo's offline metadata also honors target-dir.
    """
    from grpc_tools import protoc
    import grpc_tools

    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    repo = Path(__file__).resolve().parents[3]
    metadata = json.loads(
        subprocess.check_output(
            [
                "cargo",
                "metadata",
                "--offline",
                "--locked",
                "--format-version",
                "1",
                "--no-deps",
            ],
            cwd=repo,
            text=True,
        )
    )
    # Full metadata can try to download unused cross-target/dev dependencies,
    # even after a successful binary build. Read the locked registry version
    # directly and reuse the sources Cargo already unpacked for that build.
    with (repo / "Cargo.lock").open("rb") as lock_file:
        packages = tomllib.load(lock_file)["package"]
    versions = [p["version"] for p in packages if p["name"] == "vllm-proto"]
    assert len(versions) == 1, "Expected one locked vllm-proto version"
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    sources = list(
        (cargo_home / "registry" / "src").glob(f"*/vllm-proto-{versions[0]}")
    )
    assert sources, "Build the router first to unpack its locked vllm-proto crate"
    proto_dir = sources[0]
    target = Path(metadata["target_directory"])
    suffix = ".exe" if sys.platform == "win32" else ""
    binary = os.environ.get("VLLM_ROUTER_TEST_BIN") or next(
        (
            target / mode / f"vllm-router{suffix}"
            for mode in ("debug", "release")
            if (target / mode / f"vllm-router{suffix}").is_file()
        ),
        None,
    )
    assert binary and Path(binary).is_file(), (
        "Build the router first: cargo build --bin vllm-router"
    )
    output = tmp_path_factory.mktemp("kv-proto")
    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{proto_dir}",
            f"-I{Path(grpc_tools.__file__).parent / '_proto'}",
            f"--python_out={output}",
            f"--grpc_python_out={output}",
            str(proto_dir / "control.proto"),
            str(proto_dir / "inference.proto"),
        ]
    )
    assert result == 0, "Failed to generate locked vllm-proto bindings"
    sys.path.insert(0, str(output))
    try:
        bindings = tuple(
            importlib.import_module(name)
            for name in (
                "control_pb2",
                "control_pb2_grpc",
                "inference_pb2",
                "inference_pb2_grpc",
            )
        )
        yield bindings, binary
    finally:
        sys.path.remove(str(output))


@pytest.fixture
def kv_workers(kv_st_runtime):
    from tests.python.helpers.mock_worker import GrpcKvWorker

    workers = []

    def start(**kwargs):
        worker = GrpcKvWorker(
            kv_st_runtime[0], worker_id=f"worker-{len(workers)}", **kwargs
        )
        workers.append(worker)
        return worker

    try:
        yield start
    finally:
        for worker in workers:
            worker.close()


@pytest.fixture
def kv_tokenizer(tmp_path):
    """Small local tokenizer; real chat rendering/encoding, predictable blocks."""
    vocab = {"[UNK]": 0, **{f"t{i}": i + 1 for i in range(512)}}
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {"type": "WhitespaceSplit"},
                "post_processor": None,
                "decoder": None,
                "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
            }
        )
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "vocab_size": len(vocab),
            }
        )
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}",
            }
        )
    )
    return tmp_path
