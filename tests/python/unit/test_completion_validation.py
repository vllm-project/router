import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.e2e.validate_completion import validate_single_completion

pytestmark = [pytest.mark.ready, pytest.mark.cpu]


@pytest.fixture
def smoke_response():
    # MI300X RDMA read smoke at 844a872; only the topology-bearing id is redacted.
    path = Path(__file__).with_name("fixtures") / "moriio_smoke_completion.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("reason", ["stop", "length"])
def test_valid_completion(smoke_response, reason):
    # The captured response ended with length; stop is a derived valid variant.
    smoke_response["choices"][0]["finish_reason"] = reason
    validate_single_completion(smoke_response)


def test_empty_completion_message(smoke_response):
    smoke_response["choices"][0]["text"] = " "
    with pytest.raises(ValueError, match="^completion text is empty or invalid$"):
        validate_single_completion(smoke_response)


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
    "response_kind,expected_code,expected_message",
    [
        ("valid", 0, "Completion response is structurally valid and non-empty"),
        (
            "empty",
            1,
            "Invalid completion response: completion text is empty or invalid",
        ),
        ("malformed", 1, "Invalid completion response:"),
        ("missing", 3, "Cannot read completion response"),
    ],
)
def test_completion_cli(
    tmp_path,
    pytestconfig,
    smoke_response,
    response_kind,
    expected_code,
    expected_message,
):
    response = tmp_path / "response.json"
    if response_kind == "empty":
        smoke_response["choices"][0]["text"] = ""
    if response_kind != "missing":
        response.write_text(
            "{" if response_kind == "malformed" else json.dumps(smoke_response),
            encoding="utf-8",
        )
    result = subprocess.run(
        [sys.executable, "-m", "tools.e2e.validate_completion", str(response)],
        cwd=pytestconfig.rootpath,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == expected_code, result.stdout + result.stderr
    if expected_code:
        assert expected_message in result.stderr
        assert not result.stdout
        if response_kind == "missing":
            assert str(response) in result.stderr
    else:
        assert result.stdout.strip() == expected_message
        assert not result.stderr
