import pytest

from tools.e2e.validate_completion import validate_completion

pytestmark = [pytest.mark.ready, pytest.mark.cpu]


@pytest.mark.parametrize("reason", ["stop", "length"])
def test_valid_completion(reason):
    validate_completion({"choices": [{"text": "Paris", "finish_reason": reason}]})


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
        validate_completion(data)
