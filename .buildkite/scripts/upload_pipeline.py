#!/usr/bin/env python3
"""Expand ``mirror_hardwares`` presets and optionally upload the pipeline.

An explicit name such as ``mirror_hardwares: l4_4`` is replaced with that
entry from ``ci_mirror_hardwares.yml`` (``agents`` and, when present,
``plugins``). A step-level ``retry`` is kept. A preset that injects
``VLLM_CI_HF_TOKEN`` also gets a shell export of ``HF_TOKEN`` from that
value. A build-level ``HF_TOKEN`` would override a pod env entry of the
same name, so the secret stays under the alias until the job shell runs.
This is the router-sized slice of vllm-omni's uploader: no ``MIRROR_HW``
switch and no inference from pytest marks.

Usage:
  python3 .buildkite/scripts/upload_pipeline.py .buildkite/test-ready.yml
  python3 .buildkite/scripts/upload_pipeline.py --upload .buildkite/test-ready.yml
"""

from __future__ import annotations

import argparse
import copy
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

BUILDKITE_DIR = Path(__file__).resolve().parent.parent
HARDWARE_PATH = BUILDKITE_DIR / "ci_mirror_hardwares.yml"
# ``$$`` survives Buildkite upload interpolation as a single ``$``.
CUDA_HF_TOKEN_ENV = "VLLM_CI_HF_TOKEN"
CUDA_HF_TOKEN_EXPORT = (
    f'if [ -n "$${{{CUDA_HF_TOKEN_ENV}:-}}" ]; then export HF_TOKEN="$${{{CUDA_HF_TOKEN_ENV}}}"; fi'
)


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


def _preset_env_names(preset: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    plugins = preset.get("plugins")
    if not isinstance(plugins, list):
        return names
    for plugin in plugins:
        if not isinstance(plugin, dict):
            continue
        for spec in plugin.values():
            if not isinstance(spec, dict):
                continue
            pod = spec.get("podSpec")
            if not isinstance(pod, dict):
                continue
            containers = pod.get("containers")
            if not isinstance(containers, list):
                continue
            for container in containers:
                if not isinstance(container, dict):
                    continue
                env = container.get("env")
                if not isinstance(env, list):
                    continue
                for item in env:
                    if isinstance(item, dict) and isinstance(item.get("name"), str):
                        names.add(item["name"])
    return names


def _prepend_hf_token_export(merged: dict[str, Any]) -> None:
    """Restore ``HF_TOKEN`` after Buildkite has merged build-level env."""
    key = "commands" if "commands" in merged else "command"
    existing = merged.get(key)
    if isinstance(existing, list):
        merged[key] = [CUDA_HF_TOKEN_EXPORT, *existing]
    elif existing is None:
        merged[key] = [CUDA_HF_TOKEN_EXPORT]
    else:
        merged[key] = [CUDA_HF_TOKEN_EXPORT, existing]


def shared_rocm_step(merged: dict[str, Any], count: int) -> None:
    """Opt-in client of the host's GPU lease hook; never allocate in a PR script."""
    plugin = merged["plugins"][0]["docker#v5.14.0"]
    for key in ("devices", "privileged", "ipc", "shm-size"):
        plugin.pop(key, None)
    plugin["environment"] = ["AMD_CI_GPU_LEASE_ID", "AMD_CI_GPU_COUNT"]
    merged["agents"] = {"queue": "router_rocm_mi300_shared"}
    merged["env"] = {**merged.get("env", {}), "AMD_CI_GPU_COUNT": str(count)}
    command = merged["command"]
    # Preserve the original execution budget separately from the six-hour
    # shared-resource wait. The host supervisor also bounds lease lifetime.
    minutes = merged.get("timeout_in_minutes", 60)
    merged["timeout_in_minutes"] = minutes + 390
    merged["command"] = (
        ': "$${AMD_CI_GPU_LEASE_ID:?shared GPU admission hook is missing}"\n'
        "python3 -c 'import os, torch; assert torch.cuda.device_count() == "
        'int(os.environ["AMD_CI_GPU_COUNT"])\'\n'
        f"timeout --signal=TERM --kill-after=60s {minutes * 60} "
        f"bash -euo pipefail -c {shlex.quote(command)}"
    )


def expand_steps(
    steps: list[Any], presets: dict[str, dict[str, Any]], *, rocm_shared: bool = False
) -> list[Any]:
    expanded: list[Any] = []
    for step in steps:
        if not isinstance(step, dict):
            expanded.append(step)
            continue
        step = dict(step)
        nested = step.get("steps")
        if isinstance(nested, list):
            step["steps"] = expand_steps(nested, presets, rocm_shared=rocm_shared)
        shared_count = step.pop("shared_gpu_count", None)
        if shared_count is not None and (
            type(shared_count) is not int
            or shared_count != 2
            or step.get("mirror_hardwares") != "mi300_2"
        ):
            raise SystemExit(
                "shared_gpu_count currently supports only the two-GPU mi300_2 functional preset"
            )
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
        if CUDA_HF_TOKEN_ENV in _preset_env_names(preset):
            _prepend_hf_token_export(merged)
        if "retry" in step:
            merged["retry"] = step["retry"]
        if rocm_shared and shared_count is not None:
            shared_rocm_step(merged, shared_count)
        expanded.append(merged)
    return expanded


def render(pipeline_path: Path, *, rocm_shared: bool = False) -> str:
    doc = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not isinstance(doc.get("steps"), list):
        raise SystemExit(f"{pipeline_path} has no steps list")
    doc = dict(doc)
    doc["steps"] = expand_steps(doc["steps"], load_presets(), rocm_shared=rocm_shared)
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
    if os.environ.get("ROCM_SHARED_GPUS", "0") not in ("0", "1"):
        parser.error("ROCM_SHARED_GPUS must be 0 or 1")
    content = render(
        args.pipeline, rocm_shared=os.environ.get("ROCM_SHARED_GPUS") == "1"
    )
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
