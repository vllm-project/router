"""CPU checks for the functional E2E worker process boundary."""

import os
from unittest.mock import Mock

import pytest

from tests.python.e2e.function.helpers import launch

pytestmark = [pytest.mark.ready, pytest.mark.cpu]


@pytest.mark.parametrize(
    "hip_mask,rocr_mask",
    [
        (None, None),
        ("", ""),
        ("0,1", "0,1"),
        ("0,1", "4,5"),
        (None, "4,5"),
        ("4,5", None),
    ],
)
@pytest.mark.parametrize("gpu_id", [0, 1])
def test_worker_gpu_masks(monkeypatch, hip_mask, rocr_mask, gpu_id):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    if hip_mask is None:
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", hip_mask)
    if rocr_mask is None:
        monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", rocr_mask)
    parent_env = os.environ.copy()

    monkeypatch.setattr(
        launch,
        "resolve_worker_serve_launch",
        lambda **kwargs: ("vllm-rs", ["vllm-rs", "serve"]),
    )
    popen = Mock()
    monkeypatch.setattr(launch.subprocess, "Popen", popen)
    launch.popen_launch_vllm_rs_grpc_worker(
        "test-model", http_port=8000, grpc_port=8001, base_gpu_id=gpu_id
    )

    child_env = popen.call_args.kwargs["env"]
    assert child_env["CUDA_VISIBLE_DEVICES"] == str(gpu_id)
    if hip_mask is None:
        assert "HIP_VISIBLE_DEVICES" not in child_env
    else:
        assert child_env["HIP_VISIBLE_DEVICES"] == str(gpu_id)
    if rocr_mask is None:
        assert "ROCR_VISIBLE_DEVICES" not in child_env
    else:
        assert child_env["ROCR_VISIBLE_DEVICES"] == rocr_mask
    assert dict(os.environ) == parent_env
