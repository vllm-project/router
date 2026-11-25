"""GPU accuracy test for Rust gRPC prefill/decode.

Level mark ``ready``. Machine marks come from ``hardware_test``: CUDA L4,
four cards (prefill and decode at data-parallel size 2). The CPU job selects
``ready and cpu`` and leaves this test out.
"""

import os
import subprocess
from pathlib import Path

import pytest

from ..helpers.mark import hardware_test

pytestmark = [pytest.mark.ready]


@hardware_test(res={"cuda": "L4"}, num_cards=4)
def test_grpc_pd_disagg_accuracy():
    script = (
        Path(__file__).resolve().parents[1]
        / "e2e"
        / "pd_disagg_vllm"
        / "run_grpc_pd_accuracy_test.sh"
    )
    completed = subprocess.run(
        ["bash", str(script)],
        env=os.environ.copy(),
        check=False,
    )
    assert completed.returncode == 0, completed.returncode
