import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.e2e.validate_completion import validate_single_completion

pytestmark = [pytest.mark.ready, pytest.mark.cpu]


@pytest.mark.parametrize("reason", ["stop", "length"])
def test_valid_completion(reason):
    validate_single_completion(
        {"choices": [{"text": "Paris", "finish_reason": reason}]}
    )


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        {"error": "transfer failed", "choices": [{"text": "Paris"}]},
        {},
        {"choices": []},
        {"choices": [None]},
        {"choices": [{"text": "", "finish_reason": "stop"}]},
        {"choices": [{"text": "  ", "finish_reason": "length"}]},
        {"choices": [{"text": 123, "finish_reason": "stop"}]},
        {"choices": [{"text": "Paris", "finish_reason": None}]},
        {"choices": [{"text": "Paris", "finish_reason": "error"}]},
        {"choices": [{"text": "Paris", "finish_reason": "stop"}] * 2},
    ],
)
def test_invalid_completion(data):
    with pytest.raises(ValueError):
        validate_single_completion(data)


@pytest.mark.parametrize(
    "response_kind,expected_code", [("valid", 0), ("empty", 1), ("missing", 1)]
)
def test_completion_cli(tmp_path, response_kind, expected_code):
    response = tmp_path / "response.json"
    if response_kind != "missing":
        response.write_text(
            json.dumps(
                {
                    "choices": [
                        {
                            "text": "Paris" if response_kind == "valid" else "",
                            "finish_reason": "stop",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
    result = subprocess.run(
        [sys.executable, "-m", "tools.e2e.validate_completion", str(response)],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == expected_code, result.stdout + result.stderr
    if expected_code:
        assert "Invalid completion response:" in result.stderr
