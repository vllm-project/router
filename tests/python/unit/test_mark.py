"""Unit tests for the L4 / MI300 hardware mark helpers."""

import inspect
from pathlib import Path

import pytest

from ..helpers import mark as hw
from ..helpers.mark import (
    HARDWARE_MARK_NORMALIZATION_FIXTURE,
    get_hardware_mark_list,
    get_level_markers,
    get_skus_for_platform,
    get_supported_card_counts,
    get_supported_platforms,
    hardware_marks,
    hardware_test,
)

pytestmark = [pytest.mark.ready, pytest.mark.cpu]


def _names(marks):
    names = []
    for mark in marks:
        marker = getattr(mark, "mark", mark)
        names.append(marker.name)
    return names


def _patch_off_platform(monkeypatch):
    monkeypatch.setattr(hw, "is_cuda", lambda: False)
    monkeypatch.setattr(hw, "is_rocm", lambda: False)
    monkeypatch.setattr(hw, "device_count", lambda: 0)


def test_registry_lists_ci_hardware():
    assert get_level_markers() == frozenset({"ready"})
    assert get_hardware_mark_list() == frozenset({"L4", "MI300"})
    assert get_skus_for_platform("cuda") == frozenset({"L4"})
    assert get_skus_for_platform("rocm") == frozenset({"MI300"})
    assert get_supported_platforms() == frozenset({"cpu", "cuda", "rocm", "gpu"})
    assert get_supported_card_counts() == frozenset({1, 2, 4})


def test_hardware_marks_for_l4_and_mi300():
    l4 = _names(hardware_marks(res={"cuda": "L4"}, num_cards=4))
    assert l4[:4] == ["gpu", "L4", "cuda", "cards_4"]

    mi300 = _names(hardware_marks(res={"rocm": "MI300"}, num_cards=2))
    assert mi300[:4] == ["gpu", "MI300", "rocm", "cards_2"]


def test_cuda_sku_list_and_same_card_count_union():
    names = _names(hardware_marks(res={"cuda": ["L4"]}, num_cards=1))
    assert names == ["gpu", "L4", "cuda", "cards_1"]

    @hardware_test(res={"cuda": "L4", "rocm": "MI300"}, num_cards=1)
    def sample():
        return None

    names = _names(sample.pytestmark)
    assert "L4" in names
    assert "MI300" in names
    assert "cuda" in names
    assert "rocm" in names
    assert "gpu" in names
    assert "cards_1" in names
    assert "skipif" not in names


def test_multi_card_skipif_only_on_that_platform(monkeypatch):
    monkeypatch.setattr(hw, "is_cuda", lambda: True)
    monkeypatch.setattr(hw, "device_count", lambda: 1)
    marks = hardware_marks(res={"cuda": "L4"}, num_cards=4)
    skipifs = [mark.mark for mark in marks if mark.mark.name == "skipif"]
    assert len(skipifs) == 1
    assert skipifs[0].args == (True,)
    assert "4" in skipifs[0].kwargs["reason"]

    monkeypatch.setattr(hw, "is_cuda", lambda: False)
    marks = hardware_marks(res={"cuda": "L4"}, num_cards=4)
    assert "skipif" not in _names(marks)

    monkeypatch.setattr(hw, "is_rocm", lambda: True)
    monkeypatch.setattr(hw, "device_count", lambda: 2)
    marks = hardware_marks(res={"rocm": "MI300"}, num_cards=2)
    skipifs = [mark.mark for mark in marks if mark.mark.name == "skipif"]
    assert len(skipifs) == 1
    assert skipifs[0].args == (False,)


def test_hardware_marks_rejects_bad_input():
    with pytest.raises(ValueError, match="Invalid cuda resource"):
        hardware_marks(res={"cuda": "H100"})
    with pytest.raises(ValueError, match="not a res dict key"):
        hardware_marks(res={"gpu": "L4"})
    with pytest.raises(ValueError, match="not a res dict key"):
        hardware_marks(res={"cpu": "L4"})
    with pytest.raises(ValueError, match="Unsupported platform"):
        hardware_marks(res={"xpu": "L4"})
    with pytest.raises(ValueError, match="no registered marker"):
        hardware_marks(res={"cuda": "L4"}, num_cards=3)
    with pytest.raises(ValueError, match="positive int"):
        hardware_marks(res={"cuda": "L4"}, num_cards=True)
    with pytest.raises(ValueError, match="must not be empty"):
        hardware_marks(res={"cuda": []})
    with pytest.raises(ValueError, match="rocm resource must be a string"):
        hardware_marks(res={"rocm": ["MI300"]})
    with pytest.raises(ValueError, match="not in res"):
        hardware_marks(res={"cuda": "L4"}, num_cards={"rocm": 2})
    with pytest.raises(ValueError, match="cannot attach different per-platform"):
        hardware_marks(
            res={"cuda": "L4", "rocm": "MI300"},
            num_cards={"cuda": 4, "rocm": 2},
        )


def test_mixed_card_counts_split_one_item_per_platform(monkeypatch):
    _patch_off_platform(monkeypatch)

    @hardware_test(
        res={"cuda": "L4", "rocm": "MI300"},
        num_cards={"cuda": 4},
    )
    def sample():
        return None

    marks = list(sample.pytestmark)
    assert any(mark.name == "usefixtures" for mark in marks)
    parametrize = next(mark for mark in marks if mark.name == "parametrize")
    assert parametrize.args[0] == HARDWARE_MARK_NORMALIZATION_FIXTURE
    cuda_param, rocm_param = parametrize.args[1]
    assert cuda_param.id == "cuda"
    assert rocm_param.id == "rocm"
    assert _names(cuda_param.marks)[:4] == ["gpu", "L4", "cuda", "cards_4"]
    assert "MI300" not in _names(cuda_param.marks)
    assert _names(rocm_param.marks)[:4] == ["gpu", "MI300", "rocm", "cards_1"]
    assert "L4" not in _names(rocm_param.marks)

    cuda_skip = next(mark for mark in cuda_param.marks if mark.name == "skipif")
    rocm_skip = next(mark for mark in rocm_param.marks if mark.name == "skipif")
    assert cuda_skip.args == (True,)
    assert cuda_skip.kwargs["reason"] == "Requires cuda platform"
    assert rocm_skip.args == (True,)
    assert rocm_skip.kwargs["reason"] == "Requires rocm platform"


def test_normalization_fixture_is_registered(request):
    defs = request.session._fixturemanager.getfixturedefs(
        HARDWARE_MARK_NORMALIZATION_FIXTURE, request.node
    )
    assert defs
    registered = defs[-1]
    assert registered.argname == HARDWARE_MARK_NORMALIZATION_FIXTURE
    source = Path(inspect.getfile(registered.func)).resolve()
    assert source == Path(__file__).resolve().parents[1] / "conftest.py"
