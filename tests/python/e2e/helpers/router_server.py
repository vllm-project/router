# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Launch and tear down vLLM Router for E2E perf cases."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread

from .http_ready import wait_for_http_ok
from .paths import repo_root
from .server import (
    TerminalStream,
    _shutdown_process,
    ensure_port_free,
    join_streaming_output,
    start_streaming_output,
    wait_for_port_free,
)


@dataclass(frozen=True)
class RouterDeployConfig:
    """Router process settings from a router-replay case JSON ``router_params`` block."""

    binary: Path
    host: str
    port: int
    metrics_port: int
    worker_urls: tuple[str, ...]
    intra_node_data_parallel_size: int
    program_scheduling_config: Path
    ready_url: str
    ready_timeout_seconds: int
    log_dir: Path
    log_label: str


def compact_program_scheduling_json(config_path: Path) -> str:
    """Return a single-line JSON string for ``--program-scheduling-config-json``."""

    if not config_path.is_file():
        raise FileNotFoundError(f"program scheduling config not found: {config_path}")
    script = "import json,sys; print(json.dumps(json.load(open(sys.argv[1])), separators=(',', ':')))"
    completed = subprocess.run(
        [sys.executable, "-c", script, str(config_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_router_argv(
    deploy: RouterDeployConfig, *, program_config_json: str
) -> list[str]:
    """Build argv for ``vllm-router`` with program scheduling enabled."""

    argv = [
        str(deploy.binary),
        "--host",
        deploy.host,
        "--port",
        str(deploy.port),
        "--intra-node-data-parallel-size",
        str(deploy.intra_node_data_parallel_size),
        "--policy",
        "consistent_hash",
        "--prometheus-host",
        "0.0.0.0",
        "--prometheus-port",
        str(deploy.metrics_port),
        "--enable-program-scheduling",
        "--program-scheduling-config-json",
        program_config_json,
    ]
    for worker_url in deploy.worker_urls:
        argv.extend(["--worker-urls", worker_url])
    return argv


@dataclass
class RouterServer:
    """One Router subprocess with log streaming and readiness polling."""

    deploy: RouterDeployConfig
    proc: subprocess.Popen[str] | None = None
    log_path: Path | None = None
    _log_thread: Thread | None = field(default=None, repr=False)
    _terminal: TerminalStream = field(default_factory=TerminalStream, repr=False)

    def start(self) -> None:
        connect_host = (
            "127.0.0.1" if self.deploy.host in {"0.0.0.0", "::"} else self.deploy.host
        )
        ensure_port_free(connect_host, self.deploy.port)
        if (
            not self.deploy.binary.is_file()
            and shutil.which(str(self.deploy.binary)) is None
        ):
            raise FileNotFoundError(f"router binary not found: {self.deploy.binary}")

        self.deploy.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.deploy.log_dir / f"router-{self.deploy.log_label}.log"
        program_json = compact_program_scheduling_json(
            self.deploy.program_scheduling_config
        )
        cmd = build_router_argv(self.deploy, program_config_json=program_json)
        env = os.environ.copy()
        env["RUST_LOG"] = env.get("RUST_LOG", "info")

        prefix = f"[router:{self.deploy.log_label}] "
        print(f"{prefix}starting: {' '.join(cmd)}", flush=True)
        print(f"{prefix}log file: {self.log_path}", flush=True)
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=repo_root(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
            bufsize=1,
        )
        self._log_thread = start_streaming_output(
            self.proc,
            log_path=self.log_path,
            prefix=prefix,
            terminal=self._terminal,
        )
        wait_for_http_ok(
            self.deploy.ready_url,
            timeout_seconds=float(self.deploy.ready_timeout_seconds),
            proc=self.proc,
            log_path=self.log_path,
            prefix=prefix,
        )

    def stop(self) -> None:
        if self.proc is None:
            return
        connect_host = (
            "127.0.0.1" if self.deploy.host in {"0.0.0.0", "::"} else self.deploy.host
        )
        proc = self.proc
        log_thread = self._log_thread
        label = self.deploy.log_label
        try:
            _shutdown_process(proc)
        finally:
            join_streaming_output(proc, log_thread)
            self.proc = None
            self._log_thread = None
            try:
                wait_for_port_free(connect_host, self.deploy.port, timeout_seconds=30)
            except RuntimeError as exc:
                print(f"[router:{label}] warning: {exc}", flush=True)

    def __enter__(self) -> RouterServer:
        try:
            self.start()
        except Exception:
            self.stop()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def managed_router(deploy: RouterDeployConfig) -> AbstractContextManager[RouterServer]:
    """Context manager for one Router instance."""

    return RouterServer(deploy=deploy)
