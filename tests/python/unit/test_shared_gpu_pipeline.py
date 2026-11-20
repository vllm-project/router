"""Validate opt-in expansion without building the router or using a GPU."""

import importlib.util
import copy
import os
from pathlib import Path
import subprocess
import sys
import unittest

import pytest
import yaml

pytestmark = [pytest.mark.ready, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "pipeline_uploader", ROOT / ".buildkite/scripts/upload_pipeline.py"
)
uploader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(uploader)


def leaves(steps):
    for step in steps:
        if isinstance(step, dict):
            if "steps" in step:
                yield from leaves(step["steps"])
            else:
                yield step


class SharedGpuPipelineTest(unittest.TestCase):
    def test_device_count_preflight_survives_optimized_mode(self):
        actual = len(list(Path("/dev/dri").glob("renderD*")))
        for expected, returncode in ((actual, 0), (actual + 1, 1)):
            with self.subTest(expected=expected):
                env = {**os.environ, "AMD_CI_GPU_COUNT": str(expected)}
                result = subprocess.run(
                    [
                        sys.executable,
                        "-O",
                        "-c",
                        uploader.SHARED_GPU_DEVICE_COUNT_CHECK,
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, returncode)
                if returncode:
                    self.assertIn("shared GPU device count mismatch", result.stderr)

    def test_default_preserves_original_preset_and_command(self):
        presets = uploader.load_presets()
        for name in ("test-merge.yml", "test-nightly.yml"):
            doc = yaml.safe_load((ROOT / ".buildkite" / name).read_text())
            expanded = list(leaves(uploader.expand_steps(doc["steps"], presets)))
            for original in leaves(doc["steps"]):
                if "shared_gpu_count" not in original:
                    continue
                actual = next(s for s in expanded if s.get("key") == original["key"])
                self.assertEqual(actual["command"], original["command"])
                self.assertEqual(actual["plugins"], presets["mi300_2"]["plugins"])
                self.assertEqual(
                    actual["timeout_in_minutes"], original["timeout_in_minutes"]
                )
                self.assertNotIn("shared_gpu_count", actual)
                self.assertNotIn("shared_gpu_timeout_in_minutes", actual)
                self.assertNotIn("shared_gpu_queue", actual)

    def test_opt_in_changes_only_annotated_functional_steps(self):
        presets = uploader.load_presets()
        for name in ("test-merge.yml", "test-nightly.yml"):
            path = ROOT / ".buildkite" / name
            source = yaml.safe_load(path.read_text())
            execution_minutes = {
                step["key"]: step.get(
                    "shared_gpu_timeout_in_minutes",
                    step.get("timeout_in_minutes", 60),
                )
                for step in leaves(source["steps"])
                if "shared_gpu_count" in step
            }
            normal = list(leaves(yaml.safe_load(uploader.render(path))["steps"]))
            shared = list(
                leaves(yaml.safe_load(uploader.render(path, rocm_shared=True))["steps"])
            )
            changed = []
            for before, after in zip(normal, shared, strict=True):
                if before == after:
                    continue
                changed.append(after["key"])
                self.assertEqual(after["agents"]["queue"], "router_rocm_mi300_shared")
                plugin = after["plugins"][0]["docker#v5.14.0"]
                for field in ("devices", "privileged", "ipc", "shm-size"):
                    self.assertNotIn(field, plugin)
                self.assertEqual(after["env"]["AMD_CI_GPU_COUNT"], "2")
                self.assertIn("shared GPU admission hook is missing", after["command"])
                self.assertIn(
                    f"{execution_minutes[after['key']] * 60} bash", after["command"]
                )
                self.assertEqual(after["artifact_paths"], before["artifact_paths"])
            self.assertEqual(len(changed), 1, changed)
        self.assertIn("devices", presets["mi300_2"]["plugins"][0]["docker#v5.14.0"])

    def test_execution_budget_excludes_exclusive_host_lease_wait(self):
        for name in ("test-merge.yml", "test-nightly.yml"):
            doc = yaml.safe_load((ROOT / ".buildkite" / name).read_text())
            step = next(s for s in leaves(doc["steps"]) if "shared_gpu_count" in s)
            # #508 expands the exclusive timeout to include its host lease wait.
            step["timeout_in_minutes"] = 450
            actual = uploader.expand_steps(
                [step], uploader.load_presets(), rocm_shared=True
            )[0]
            self.assertEqual(actual["timeout_in_minutes"], 450)
            self.assertIn("3600 bash", actual["command"])
            self.assertNotIn("shared_gpu_timeout_in_minutes", actual)

    def test_rejects_invalid_execution_budget(self):
        for minutes in (0, -1, True, "60"):
            with self.subTest(minutes=minutes), self.assertRaises(SystemExit):
                uploader.expand_steps(
                    [
                        {
                            "mirror_hardwares": "mi300_2",
                            "shared_gpu_count": 2,
                            "shared_gpu_timeout_in_minutes": minutes,
                            "command": "true",
                        }
                    ],
                    uploader.load_presets(),
                    rocm_shared=True,
                )
        with self.assertRaises(SystemExit):
            uploader.expand_steps(
                [{"command": "true", "shared_gpu_timeout_in_minutes": 60}], {}
            )

    def test_resource_request_is_independent_of_test_and_preset_names(self):
        preset = copy.deepcopy(uploader.load_presets()["mi300_2"])
        preset["shared_gpu_queue"] = "another-rocm-shared-pool"
        preset["plugins"][0]["docker#v5.14.0"]["environment"] = ["CUSTOM_ENV"]
        presets = {"another_rocm_host": preset}
        original = copy.deepcopy(presets)
        for count in (1, 2, 3, 4, 8):
            for key, command in (
                ("command", "echo arbitrary"),
                ("commands", ["echo one", "echo two"]),
            ):
                with self.subTest(count=count, command_key=key):
                    actual = uploader.expand_steps(
                        [
                            {
                                "mirror_hardwares": "another_rocm_host",
                                "shared_gpu_count": count,
                                key: command,
                            }
                        ],
                        presets,
                        rocm_shared=True,
                    )[0]
                    self.assertEqual(
                        actual["agents"]["queue"], "another-rocm-shared-pool"
                    )
                    self.assertEqual(actual["env"]["AMD_CI_GPU_COUNT"], str(count))
                    self.assertIn(
                        "CUSTOM_ENV",
                        actual["plugins"][0]["docker#v5.14.0"]["environment"],
                    )
                    self.assertNotIn("commands", actual)
                    self.assertNotIn("shared_gpu_queue", actual)
                    self.assertNotIn("torch", actual["command"])
        self.assertEqual(presets, original)

    def test_rejects_unsupported_hardware_and_invalid_count(self):
        for preset, count in (
            ("mi300_rdma", 2),
            ("l4_2", 2),
            ("mi300_2", 0),
            ("mi300_2", -1),
            ("mi300_2", "2"),
            ("mi300_2", True),
        ):
            with self.subTest(preset=preset, count=count), self.assertRaises(
                SystemExit
            ):
                uploader.expand_steps(
                    [{"mirror_hardwares": preset, "shared_gpu_count": count}],
                    uploader.load_presets(),
                    rocm_shared=True,
                )

    def test_declared_capability_still_requires_supported_container_client(self):
        preset = copy.deepcopy(uploader.load_presets()["l4_2"])
        preset["shared_gpu_queue"] = "not-a-docker-pool"
        with self.assertRaises(SystemExit):
            uploader.expand_steps(
                [
                    {
                        "mirror_hardwares": "incompatible",
                        "shared_gpu_count": 1,
                        "command": "true",
                    }
                ],
                {"incompatible": preset},
                rocm_shared=True,
            )
        with self.assertRaises(SystemExit):
            uploader.expand_steps(
                [
                    {
                        "mirror_hardwares": "mi300_rdma",
                        "shared_gpu_count": 1,
                        "shared_gpu_queue": "step-cannot-declare-hardware-capability",
                        "command": "true",
                    }
                ],
                uploader.load_presets(),
                rocm_shared=True,
            )


if __name__ == "__main__":
    unittest.main()
