#!/usr/bin/env python3
"""Validate .buildkite YAML against the official Buildkite schema.

Same check as vllm-omni: check-jsonschema builtin schema vendor.buildkite.
``mirror_hardwares`` is expanded with upload_pipeline.py first, because that
key is replaced before Buildkite sees the pipeline. YAML that is not a
pipeline (no top-level ``steps`` list) is skipped.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / ".buildkite" / "scripts"))

from upload_pipeline import render  # noqa: E402


def _normalize(path: str) -> str:
    return path.replace("\\", "/")


def _is_pipeline(path: Path) -> bool:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    return isinstance(doc, dict) and isinstance(doc.get("steps"), list)


def _validate(path: Path) -> int:
    if not _is_pipeline(path):
        return 0
    rendered = render(path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".yml",
        delete=False,
    ) as handle:
        handle.write(rendered)
        rendered_path = Path(handle.name)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "check_jsonschema",
                "--builtin-schema",
                "vendor.buildkite",
                str(rendered_path),
            ],
            capture_output=True,
            text=True,
        )
    finally:
        rendered_path.unlink(missing_ok=True)
    if result.returncode == 0:
        return 0

    sys.stderr.write(f"{_normalize(str(path))}: Buildkite schema validation failed\n")
    sys.stderr.write(result.stdout or "")
    sys.stderr.write(result.stderr or "")
    return 1


def main(argv: list[str]) -> int:
    rc = 0
    root = _REPO_ROOT
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
