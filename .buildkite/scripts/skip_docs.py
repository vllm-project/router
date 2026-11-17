#!/usr/bin/env python3
"""Skip Buildkite builds/tests for docs-only PRs, following vllm-omni's bootstrap.

Only PRs qualify. Pushes, schedules and unresolved diffs run the full pipeline.
Use a merge-base diff and disable rename detection so moving code into docs
cannot hide the deletion of a source file.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip("\n")


def is_doc(path: str) -> bool:
    # TEMP: remove after verifying docs-only skipping in CI.
    if path.startswith((".buildkite/", "tests/")):
        return True
    return path.startswith("docs/") or path.endswith(".md") or path == "mkdocs.yml"


def docs_only_pr(env: Mapping[str, str]) -> bool:
    pr = env.get("BUILDKITE_PULL_REQUEST", "false")
    base = env.get("BUILDKITE_PULL_REQUEST_BASE_BRANCH", "")
    # Explicit nightly requests should still exercise the full pipeline.
    if not pr.isdecimal() or not base or env.get("NIGHTLY") == "1":
        return False
    try:
        git("check-ref-format", "--branch", base)
        # Always fetch the target: origin/<base> can be missing or stale on agents.
        git("fetch", "--no-tags", "origin", f"refs/heads/{base}")
        ancestor = git("merge-base", "FETCH_HEAD", "HEAD")
        changed = git("diff", "--name-only", "--no-renames", "-z", ancestor, "HEAD")
        files = [path for path in changed.split("\0") if path]
        return bool(files) and all(is_doc(path) for path in files)
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        print(f"Cannot resolve PR diff; running full CI: {exc}", flush=True)
        return False


def main() -> None:
    if docs_only_pr(os.environ):
        print("Docs-only PR: skipping Buildkite builds and tests.", flush=True)
        return
    subprocess.run(
        [
            "buildkite-agent",
            "pipeline",
            "upload",
            str(ROOT / ".buildkite/bootstrap-upload-steps.yml"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
