"""CPU checks for hardware retry expansion in the test pipeline uploader."""

import copy
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".buildkite" / "scripts"))
import upload_pipeline  # noqa: E402

pytestmark = [pytest.mark.ready, pytest.mark.cpu]


def _leaf_steps(steps):
    for step in steps:
        if not isinstance(step, dict):
            continue
        if "steps" in step:
            yield from _leaf_steps(step["steps"])
        else:
            yield step


@pytest.mark.parametrize(
    "name", ["test-ready.yml", "test-merge.yml", "test-nightly.yml"]
)
def test_render_inherits_shared_hardware_retry(name):
    path = ROOT / ".buildkite" / name
    source = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_load(upload_pipeline.render(path))
    registry = yaml.safe_load(upload_pipeline.HARDWARE_PATH.read_text(encoding="utf-8"))
    shared = registry["x-hardware-retry"]
    hardware_jobs = 0
    for original, expanded in zip(
        _leaf_steps(source["steps"]), _leaf_steps(rendered["steps"]), strict=True
    ):
        if "env" in original:
            assert expanded["env"] == original["env"]
            assert all(isinstance(value, str) for value in expanded["env"].values())
        if "mirror_hardwares" in original:
            hardware_jobs += 1
            assert "retry" not in original
            assert expanded["retry"] == shared
            assert "mirror_hardwares" not in expanded
        else:
            assert expanded == original
    assert hardware_jobs > 0
    assert shared["manual"]["allowed"] is True
    assert shared["manual"]["permit_on_passed"] is True
    assert {rule.get("exit_status") for rule in shared["automatic"]} == {-1, 128, None}
    assert {rule.get("signal_reason") for rule in shared["automatic"]} == {
        "agent_stop",
        "agent_refused",
        None,
    }
    assert all(rule["limit"] == 1 for rule in shared["automatic"])


@pytest.mark.parametrize(
    "override",
    [
        {"manual": {"allowed": False}},
        {"automatic": [{"exit_status": 1, "limit": 3}]},
    ],
)
def test_step_retry_replaces_the_entire_preset(override):
    steps = [
        {"command": "echo override", "mirror_hardwares": "l4_4", "retry": override}
    ]
    original = copy.deepcopy(steps)
    expanded = upload_pipeline.expand_steps(steps, upload_pipeline.load_presets())
    assert expanded[0]["retry"] == override
    assert steps == original


def test_shared_retry_is_copied_for_each_job():
    presets = upload_pipeline.load_presets()
    expected = copy.deepcopy(presets["l4_4"]["retry"])
    expanded = upload_pipeline.expand_steps(
        [
            {"command": "echo first", "mirror_hardwares": "l4_4"},
            {"command": "echo second", "mirror_hardwares": "l4_4"},
        ],
        presets,
    )
    expanded[0]["retry"]["automatic"][0]["limit"] = 5
    assert expanded[1]["retry"] == expected
    assert all(preset["retry"] == expected for preset in presets.values())


def test_changing_the_shared_policy_updates_all_hardware_jobs(tmp_path, monkeypatch):
    registry = yaml.safe_load(upload_pipeline.HARDWARE_PATH.read_text(encoding="utf-8"))
    registry["x-hardware-retry"]["automatic"][0]["limit"] = 3
    path = tmp_path / "hardware.yml"
    path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    monkeypatch.setattr(upload_pipeline, "HARDWARE_PATH", path)
    rendered = yaml.safe_load(
        upload_pipeline.render(ROOT / ".buildkite" / "test-nightly.yml")
    )
    hardware_jobs = [step for step in _leaf_steps(rendered["steps"]) if "retry" in step]
    assert hardware_jobs
    assert all(step["retry"]["automatic"][0]["limit"] == 3 for step in hardware_jobs)


def test_command_list_keeps_step_env_and_runtime_hf_token_export():
    commands = ["pytest rust_grpc", "pytest python_http"]
    env = {"PERF_STACK_TIMEOUT": "1800", "PERF_WORKER_STARTUP_TIMEOUT": "1800"}
    source = {"mirror_hardwares": "h100_2", "command": commands, "env": env}
    expanded = upload_pipeline.expand_steps([source], upload_pipeline.load_presets())[0]
    assert expanded["command"] == [upload_pipeline.CUDA_HF_TOKEN_EXPORT, *commands]
    assert expanded["env"] == env
    assert source["command"] == commands
