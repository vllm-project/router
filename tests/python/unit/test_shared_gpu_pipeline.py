"""Validate opt-in expansion without building the router or using a GPU."""

import importlib.util
from pathlib import Path
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

    def test_opt_in_changes_only_annotated_functional_steps(self):
        presets = uploader.load_presets()
        for name in ("test-merge.yml", "test-nightly.yml"):
            path = ROOT / ".buildkite" / name
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
                    f"{before['timeout_in_minutes'] * 60} bash", after["command"]
                )
                self.assertEqual(after["artifact_paths"], before["artifact_paths"])
            self.assertEqual(len(changed), 1, changed)
        self.assertIn("devices", presets["mi300_2"]["plugins"][0]["docker#v5.14.0"])

    def test_rejects_shared_rdma_and_invalid_count(self):
        for preset, count in (("mi300_rdma", 2), ("mi300_2", 8), ("mi300_2", True)):
            with self.subTest(preset=preset, count=count), self.assertRaises(
                SystemExit
            ):
                uploader.expand_steps(
                    [{"mirror_hardwares": preset, "shared_gpu_count": count}],
                    uploader.load_presets(),
                    rocm_shared=True,
                )


if __name__ == "__main__":
    unittest.main()
