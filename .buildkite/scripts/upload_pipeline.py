#!/usr/bin/env python3
"""Expand ``mirror_hardwares`` presets and optionally upload the pipeline.

An explicit name such as ``mirror_hardwares: l4_4`` is replaced with that
entry from ``ci_mirror_hardwares.yml`` (``agents`` and, when present,
``plugins``). A step-level ``retry`` is kept. This is the router-sized
slice of vllm-omni's uploader: no ``MIRROR_HW`` switch and no inference
from pytest marks.

Usage:
  python3 .buildkite/scripts/upload_pipeline.py .buildkite/test-ready.yml
  python3 .buildkite/scripts/upload_pipeline.py --upload .buildkite/test-ready.yml
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

BUILDKITE_DIR = Path(__file__).resolve().parent.parent
HARDWARE_PATH = BUILDKITE_DIR / "ci_mirror_hardwares.yml"


class _Dumper(yaml.SafeDumper):
    """Keep wait steps as ``~`` and do not emit shared anchors."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


def _represent_none(dumper: yaml.SafeDumper, _data: Any) -> Any:
    return dumper.represent_scalar("tag:yaml.org,2002:null", "~")


_Dumper.add_representer(type(None), _represent_none)


def load_presets() -> dict[str, dict[str, Any]]:
    if not HARDWARE_PATH.is_file():
        raise SystemExit(f"missing mirror_hardwares registry: {HARDWARE_PATH}")
    doc = yaml.safe_load(HARDWARE_PATH.read_text(encoding="utf-8"))
    presets = doc.get("mirror_hardwares") if isinstance(doc, dict) else None
    if not isinstance(presets, dict) or not presets:
        raise SystemExit(f"mirror_hardwares must be a non-empty mapping in {HARDWARE_PATH}")
    return presets


def _step_label(step: dict[str, Any]) -> str:
    label = step.get("label") or step.get("key") or "<unlabeled>"
    return str(label)


def expand_steps(steps: list[Any], presets: dict[str, dict[str, Any]]) -> list[Any]:
    expanded: list[Any] = []
    for step in steps:
        if not isinstance(step, dict):
            expanded.append(step)
            continue
        step = dict(step)
        nested = step.get("steps")
        if isinstance(nested, list):
            step["steps"] = expand_steps(nested, presets)
        if "mirror_hardwares" not in step:
            expanded.append(step)
            continue

        name = step["mirror_hardwares"]
        label = _step_label(step)
        if not isinstance(name, str) or not name.strip():
            raise SystemExit(f"mirror_hardwares must be a preset name in step {label!r}")
        if "agents" in step or "plugins" in step:
            raise SystemExit(
                f"step {label!r} sets mirror_hardwares together with agents/plugins; "
                "use mirror_hardwares only"
            )
        preset = presets.get(name)
        if preset is None:
            known = ", ".join(sorted(presets))
            raise SystemExit(f"unknown mirror_hardwares {name!r} in step {label!r}; known: {known}")

        merged = {key: value for key, value in step.items() if key != "mirror_hardwares"}
        merged.update(copy.deepcopy(preset))
        if "retry" in step:
            merged["retry"] = step["retry"]
        expanded.append(merged)
    return expanded


def render(pipeline_path: Path) -> str:
    doc = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not isinstance(doc.get("steps"), list):
        raise SystemExit(f"{pipeline_path} has no steps list")
    doc = dict(doc)
    doc["steps"] = expand_steps(doc["steps"], load_presets())
    return yaml.dump(
        doc,
        Dumper=_Dumper,
        sort_keys=False,
        allow_unicode=True,
        width=1000,
        default_flow_style=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline", type=Path, help="Pipeline YAML to expand")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Pass the expanded YAML to buildkite-agent pipeline upload",
    )
    args = parser.parse_args(argv)
    content = render(args.pipeline)
    if not args.upload:
        sys.stdout.write(content)
        return 0
    subprocess.run(
        ["buildkite-agent", "pipeline", "upload"],
        input=content,
        text=True,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
