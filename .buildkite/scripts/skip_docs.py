#!/usr/bin/env python3
"""Skip Buildkite builds/tests for documentation-only changes.

PRs, merge queues and default-branch commits qualify. Nightly and scheduled
builds always run, as do unresolved diffs.
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
    return path.startswith("docs/") or path.endswith(".md") or path == "mkdocs.yml"


def docs_only_change(env: Mapping[str, str]) -> bool:
    pr = env.get("BUILDKITE_PULL_REQUEST", "false")
    base = env.get("BUILDKITE_PULL_REQUEST_BASE_BRANCH", "")
    # Explicit nightly requests should still exercise the full pipeline.
    if env.get("NIGHTLY") == "1" or env.get("BUILDKITE_SOURCE") == "schedule":
        return False
    try:
        if env.get("BUILDKITE_MERGE_QUEUE_BASE_BRANCH"):
            ancestor = env.get("BUILDKITE_MERGE_QUEUE_BASE_COMMIT", "")
            if not ancestor:
                return False
            # Confirm the recorded queue base belongs to the proposed merge.
            ancestor = git("rev-parse", "--verify", f"{ancestor}^{{commit}}")
            git("merge-base", "--is-ancestor", ancestor, "HEAD")
        elif pr.isdecimal() and base:
            git("check-ref-format", "--branch", base)
            # Always fetch the target: agent refs can be missing or stale.
            git("fetch", "--no-tags", "origin", f"refs/heads/{base}")
            ancestor = git("merge-base", "FETCH_HEAD", "HEAD")
        elif (
            not pr.isdecimal()
            and env.get("BUILDKITE_PIPELINE_DEFAULT_BRANCH")
            and env.get("BUILDKITE_BRANCH") == env["BUILDKITE_PIPELINE_DEFAULT_BRANCH"]
        ):
            # First-parent diff captures the entire merged PR, or squash commit.
            ancestor = git("rev-parse", "--verify", "HEAD^1")
        else:
            return False
        changed = git("diff", "--name-only", "--no-renames", "-z", ancestor, "HEAD")
        files = [path for path in changed.split("\0") if path]
        return bool(files) and all(is_doc(path) for path in files)
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        print(f"Cannot resolve change diff; running full CI: {exc}", flush=True)
        return False


def main() -> None:
    if docs_only_change(os.environ):
        print("Docs-only change: skipping Buildkite builds and tests.", flush=True)
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
