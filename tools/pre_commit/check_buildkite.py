#!/usr/bin/env python3
"""Validate .buildkite YAML against the official Buildkite schema.

Same check as vllm-omni: check-jsonschema builtin schema vendor.buildkite.
Router pipelines are plain Buildkite YAML, so they are checked as written.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _normalize(path: str) -> str:
    return path.replace("\\", "/")


def _validate(path: Path) -> int:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "check_jsonschema",
            "--builtin-schema",
            "vendor.buildkite",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return 0

    sys.stderr.write(f"{_normalize(str(path))}: Buildkite schema validation failed\n")
    sys.stderr.write(result.stdout or "")
    sys.stderr.write(result.stderr or "")
    return 1


def main(argv: list[str]) -> int:
    rc = 0
    root = Path(__file__).resolve().parents[2]
    for raw in argv:
        path = Path(raw)
        if not path.is_file():
            path = root / path
        if not path.is_file():
            sys.stderr.write(f"{_normalize(raw)}: file not found\n")
            rc = 1
            continue
        rc |= _validate(path)
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
