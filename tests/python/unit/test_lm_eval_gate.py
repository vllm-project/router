"""GSM8K lower bound is the baseline or env value, minus the tolerance."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.ready, pytest.mark.cpu]

_PATH = (
    Path(__file__).resolve().parents[1]
    / "e2e"
    / "pd_disagg_vllm"
    / "test_lm_eval_accuracy.py"
)
_spec = importlib.util.spec_from_file_location("pd_lm_eval_accuracy", _PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_QWEN = "Qwen/Qwen3-0.6B"
_FILTER = "exact_match,flexible-extract"


def _results(value: float) -> dict:
    return {"results": {_mod.TASK: {_FILTER: value}}}


def test_default_threshold_subtracts_tolerance(monkeypatch):
    monkeypatch.delenv(_mod.BASELINE_ENV, raising=False)
    # Qwen3-0.6B baseline 0.41, lower bound 0.38.
    assert _mod.accuracy_lower_bound(0.41) == pytest.approx(0.38)
    assert not _mod.validate_accuracy(_results(0.35), _QWEN, _FILTER)
    assert _mod.validate_accuracy(_results(0.38), _QWEN, _FILTER)


def test_env_baseline_subtracts_the_same_tolerance(monkeypatch):
    monkeypatch.setenv(_mod.BASELINE_ENV, "0.35")
    assert _mod.accuracy_lower_bound(0.41) == pytest.approx(0.32)
    assert _mod.validate_accuracy(_results(0.32), _QWEN, _FILTER)
    assert not _mod.validate_accuracy(_results(0.31), _QWEN, _FILTER)


def test_env_baseline_must_be_numeric(monkeypatch):
    monkeypatch.setenv(_mod.BASELINE_ENV, "low")
    with pytest.raises(ValueError, match=_mod.BASELINE_ENV):
        _mod.accuracy_lower_bound(0.41)
